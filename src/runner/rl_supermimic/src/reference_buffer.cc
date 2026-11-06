#include "rl_supermimic/reference_buffer.h"

#include <algorithm>
#include <cmath>

namespace rl_supermimic {

namespace {

Eigen::Vector3d RotVec(Eigen::Quaterniond q) {
  q.normalize();
  if (q.w() < 0) q.coeffs() *= -1.0;
  const Eigen::Vector3d v = q.vec();
  const double s = v.norm();
  if (s < 1e-12) return 2.0 * v;
  const double angle = 2.0 * std::atan2(s, q.w());
  return v / s * angle;
}

Eigen::Quaterniond ExpRotVec(const Eigen::Vector3d& r) {
  const double a = r.norm();
  if (a < 1e-12) return Eigen::Quaterniond(1.0, 0.5 * r.x(), 0.5 * r.y(), 0.5 * r.z()).normalized();
  return Eigen::Quaterniond(Eigen::AngleAxisd(a, r / a));
}

double Smoothstep(double w) { return w * w * (3.0 - 2.0 * w); }

}  // namespace

ReferenceBuffer::ReferenceBuffer(const ReferenceConfig& cfg, const Eigen::VectorXd& q_default)
    : cfg_(cfg), q_default_(q_default), q_entry_(q_default) {
  // scipy.ndimage.gaussian_filter1d kernel: radius = int(truncate * sigma + 0.5), truncate = 4
  const int radius = static_cast<int>(4.0 * cfg_.sigma_frames + 0.5);
  double sum = 0.0;
  for (int k = -radius; k <= radius; ++k) {
    const double v = std::exp(-0.5 * (k * k) / (cfg_.sigma_frames * cfg_.sigma_frames));
    kernel_.push_back(v);
    sum += v;
  }
  for (double& v : kernel_) v /= sum;
}

void ReferenceBuffer::Reset(double t_now, const Eigen::VectorXd& q_entry) {
  samples_.clear();
  calib_z_.clear();
  last_packet_t_ = -1e9;
  z_calibrated_ = false;
  z_offset_ = 0.0;
  w_ = 0.0;
  t_last_w_ = -1.0;
  q_entry_ = q_entry;
  t_entry_ = t_now;
}

bool ReferenceBuffer::Push(double t, const Eigen::VectorXd& q, const Eigen::Quaterniond& quat,
                           const Eigen::Vector3d& pos) {
  last_packet_t_ = t;  // ANY packet (repeats included) keeps the stream alive
  if (!samples_.empty() && samples_.back().q == q && samples_.back().pos == pos) {
    // Repeat of the previous pose: a stalled PICO (publisher re-sends) or a genuinely held pose. Short runs are
    // skipped so the stall is interpolated across; a run longer than hold_anchor_s is a real hold -> anchor it, so the
    // pose is held and the next motion is not smeared over the whole hold.
    if (t - samples_.back().t > cfg_.hold_anchor_s) samples_.push_back({t, q, samples_.back().quat, pos});
    return false;
  }
  samples_.push_back({t, q, quat.normalized(), pos});
  while (samples_.size() > 200) samples_.pop_front();
  if (!z_calibrated_) {
    calib_z_.push_back(pos.z());
    if (t - samples_.front().t >= cfg_.calib_s) {
      std::vector<double> z = calib_z_;
      std::nth_element(z.begin(), z.begin() + z.size() / 2, z.end());
      double med = z[z.size() / 2];
      if (z.size() % 2 == 0) {  // numpy median of an even count = mean of the two middle values
        const double lo = *std::max_element(z.begin(), z.begin() + z.size() / 2);
        med = 0.5 * (med + lo);
      }
      z_offset_ = cfg_.stand_height - med;
      z_calibrated_ = true;
    }
  }
  return true;
}

Eigen::VectorXd ReferenceBuffer::StandJoints(double t) const {
  if (cfg_.entry_blend_s <= 0.0) return q_default_;
  const double s = Smoothstep(std::clamp((t - t_entry_) / cfg_.entry_blend_s, 0.0, 1.0));
  return q_entry_ + s * (q_default_ - q_entry_);
}

void ReferenceBuffer::Smooth(Eigen::MatrixXd& x) const {
  const int n = static_cast<int>(x.rows());
  const int r = static_cast<int>(kernel_.size() / 2);
  Eigen::MatrixXd y = Eigen::MatrixXd::Zero(n, x.cols());
  for (int i = 0; i < n; ++i)
    for (int k = -r; k <= r; ++k) y.row(i) += kernel_[k + r] * x.row(std::clamp(i + k, 0, n - 1));  // "nearest"
  x = y;
}

Eigen::VectorXf ReferenceBuffer::Goal(double t_now, ReferenceInfo* info) {
  const int nj = cfg_.num_joints, K = cfg_.future_frames;
  const int n_pre = 6, n_post = K + 3, n = n_pre + n_post + 1;
  const double tc = t_now - cfg_.delay_s;
  std::vector<double> tg(n);
  for (int i = 0; i < n; ++i) tg[i] = tc + cfg_.dt * (i - n_pre);

  const bool have_live = samples_.size() >= 2 && z_calibrated_;
  const bool fresh = have_live && (t_now - last_packet_t_) < cfg_.stale_s + cfg_.delay_s;
  if (t_last_w_ >= 0.0) {
    const double d = t_now - t_last_w_;
    w_ = fresh ? std::min(1.0, w_ + d / cfg_.blend_in_s) : std::max(0.0, w_ - d / cfg_.blend_out_s);
  }
  t_last_w_ = t_now;

  // standing reference (moves from the entry pose to the default pose after a Reset)
  Eigen::MatrixXd q(n, nj), pos(n, 3);
  for (int i = 0; i < n; ++i) {
    q.row(i) = StandJoints(tg[i]).transpose();
    pos.row(i) << 0.0, 0.0, cfg_.stand_height;
  }
  std::vector<Eigen::Quaterniond> R(n, Eigen::Quaterniond::Identity());

  if (have_live && w_ > 0.0) {
    // resample the live stream at tg (hold beyond the ends)
    Eigen::MatrixXd lq(n, nj), lp(n, 3);
    std::vector<Eigen::Quaterniond> lr(n);
    const double t0 = samples_.front().t, t1 = samples_.back().t;
    size_t j = 0;
    for (int i = 0; i < n; ++i) {
      const double t = std::clamp(tg[i], t0, t1);
      while (j + 2 < samples_.size() && samples_[j + 1].t < t) ++j;
      while (j > 0 && samples_[j].t > t) --j;
      const Sample& a = samples_[j];
      const Sample& b = samples_[std::min(j + 1, samples_.size() - 1)];
      const double span = b.t - a.t;
      const double u = span > 1e-12 ? std::clamp((t - a.t) / span, 0.0, 1.0) : 0.0;
      lq.row(i) = ((1.0 - u) * a.q + u * b.q).transpose();
      lp.row(i) = ((1.0 - u) * a.pos + u * b.pos).transpose();
      lp(i, 2) += z_offset_;
      lr[i] = a.quat.slerp(u, b.quat);
    }
    const double ws = Smoothstep(w_);
    q = (1.0 - ws) * q + ws * lq;
    pos = (1.0 - ws) * pos + ws * lp;
    // orientation: smooth the rotation relative to the window's middle sample, blend from upright
    const Eigen::Quaterniond mid = lr[n_pre];
    Eigen::MatrixXd rel(n, 3);
    for (int i = 0; i < n; ++i) rel.row(i) = RotVec(mid.conjugate() * lr[i]).transpose();
    Smooth(rel);
    const Eigen::Quaterniond base = ExpRotVec(ws * RotVec(mid));
    for (int i = 0; i < n; ++i) R[i] = (base * ExpRotVec(ws * rel.row(i).transpose())).normalized();
  }
  Smooth(q);
  Smooth(pos);

  // numpy.gradient (edge_order 1): central inside, one-sided at the ends
  auto gradient = [&](const Eigen::MatrixXd& x) {
    Eigen::MatrixXd g(x.rows(), x.cols());
    for (int i = 0; i < n; ++i) {
      if (i == 0) g.row(i) = (x.row(1) - x.row(0)) / cfg_.dt;
      else if (i == n - 1) g.row(i) = (x.row(n - 1) - x.row(n - 2)) / cfg_.dt;
      else g.row(i) = (x.row(i + 1) - x.row(i - 1)) / (2.0 * cfg_.dt);
    }
    return g;
  };
  const Eigen::MatrixXd qd = gradient(q), v = gradient(pos);
  std::vector<Eigen::Vector3d> w_world(n, Eigen::Vector3d::Zero());
  for (int i = 1; i < n - 1; ++i) w_world[i] = RotVec(R[i + 1] * R[i - 1].conjugate()) / (2.0 * cfg_.dt);
  w_world[0] = w_world[1];
  w_world[n - 1] = w_world[n - 2];

  const int F = FeatureDim();
  Eigen::VectorXf goal(K * F);
  for (int f = 0; f < K; ++f) {
    const int i = n_pre + 1 + f;  // frames current+1 .. current+K
    const Eigen::Matrix3d Rm = R[i].toRotationMatrix();
    const double yaw = std::atan2(Rm(1, 0), Rm(0, 0));
    const Eigen::Matrix3d Ryaw = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    Eigen::VectorXd feat(F);
    feat.segment(0, nj) = q.row(i).transpose() - q_default_;
    feat.segment(nj, nj) = qd.row(i).transpose();
    feat.segment(2 * nj, 3) = Rm.transpose() * Eigen::Vector3d(0.0, 0.0, -1.0);
    feat.segment(2 * nj + 3, 3) = Rm.transpose() * w_world[i];
    feat.segment(2 * nj + 6, 3) = Ryaw.transpose() * v.row(i).transpose();
    feat(2 * nj + 9) = pos(i, 2);
    goal.segment(f * F, F) = feat.cast<float>();
  }
  if (info) {
    info->q_current = q.row(n_pre).transpose();
    info->height = pos(n_pre, 2);
    info->w_live = w_;
    info->z_offset = z_offset_;
    info->z_calibrated = z_calibrated_;
    info->samples = static_cast<int>(samples_.size());
  }
  return goal;
}

}  // namespace rl_supermimic
