#include "rl_supermimic/rl_supermimic_runner.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <thread>
#include <iomanip>
#include <limits>
#include <sstream>

#include "math/rotation_matrix.h"

namespace runner {

namespace {

constexpr uint32_t kMagicV1 = 0x54545031;  // "TTP1"
constexpr uint32_t kMagicV2 = 0x54545032;  // "TTP2" (lookahead block ignored)
constexpr size_t kPacketBytes = 8 + (25 + 4 + 3) * 4;

double MonotonicNow() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}

Eigen::Matrix3d YawRotation(const Eigen::Matrix3d& R) {
  return Eigen::AngleAxisd(std::atan2(R(1, 0), R(0, 0)), Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

int PolicyInputDim(const std::string& path) {
  try {
    std::shared_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(path.c_str()), MNN::Interpreter::destroy);
    if (!net) return -1;
    MNN::ScheduleConfig cfg;
    MNN::Session* sess = net->createSession(cfg);
    if (!sess) return -1;
    MNN::Tensor* in = net->getSessionInput(sess, nullptr);
    int n = in ? static_cast<int>(in->elementSize()) : -1;
    net->releaseSession(sess);
    return n;
  } catch (...) {
    return -1;
  }
}

}  // namespace

// RL parser (classic parser OFF), exactly as the vendor walk / dance / rl_lab runners: "Disables the classic parser's
// parallel motion control, as the RL policy has exclusive control over all active joints" (rl_dance_example).
// 2026-11-05: this was `true` (copied from rl_teleop). Both runners that set it true (rl_teleop, rl_supermimic) bounced
// on hardware from entry while both vendor RL runners that set it false work; the vendor sim has no parallel ankle, so
// no sim ever showed a difference. Exit() restores true for the classic modes (as rl_lab's teardown does).
void RlSuperMimicRunner::SetupContext() { data_store_->parallel_by_classic_parser.store(false); }
void RlSuperMimicRunner::TeardownContext() {}

RlSuperMimicRunner::~RlSuperMimicRunner() {
  rx_run_ = false;
  if (rx_thread_.joinable()) rx_thread_.join();
  if (fd_ >= 0) close(fd_);
}

// Receive thread: stamps every packet at its ARRIVAL (the validated Python reference does the same). Draining the
// socket once per 20 ms tick instead quantised arrivals to tick times; with the publisher and the runner both at
// 50 Hz on unsynchronised clocks, ticks then drained 0 or 2 packets sharing one timestamp, a time-staircase that
// distorts the reference velocities the goal carries.
void RlSuperMimicRunner::RxLoop() {
  uint8_t buf[2048];
  while (rx_run_) {
    ssize_t n = recv(fd_, buf, sizeof(buf), 0);  // blocking, 100 ms timeout
    if (n <= 0) continue;
    RawPacket p;
    p.t = MonotonicNow();
    p.bytes.assign(buf, buf + n);
    std::lock_guard<std::mutex> lk(rx_mutex_);
    if (rx_queue_.size() < 1000) rx_queue_.push_back(std::move(p));
  }
}

void RlSuperMimicRunner::OpenSocket() {
  if (fd_ >= 0) return;
  fd_ = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd_ < 0) throw std::runtime_error("rl_supermimic: failed to create UDP socket");
  int reuse = 1;
  setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(static_cast<uint16_t>(param_->udp_port));
  if (bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(fd_);
    fd_ = -1;
    throw std::runtime_error("rl_supermimic: cannot bind UDP port " + std::to_string(param_->udp_port) +
                             " (another runner holding it? rl_teleop uses 47800, rl_supermimic defaults to 47801)");
  }
  timeval tv{0, 100000};
  setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
  rx_run_ = true;
  rx_thread_ = std::thread(&RlSuperMimicRunner::RxLoop, this);
  LOG(INFO) << "rl_supermimic: listening for the reference stream on UDP :" << param_->udp_port;
}

void RlSuperMimicRunner::DrainPackets(double /*now*/) {
  std::vector<RawPacket> batch;
  {
    std::lock_guard<std::mutex> lk(rx_mutex_);
    batch.swap(rx_queue_);
  }
  for (const RawPacket& rp : batch) {
    const uint8_t* buf = rp.bytes.data();
    const ssize_t n = static_cast<ssize_t>(rp.bytes.size());
    uint32_t magic;
    std::memcpy(&magic, buf, 4);
    if (static_cast<size_t>(n) < kPacketBytes || (magic != kMagicV1 && magic != kMagicV2)) {
      ++bad_packets_;
      LOG_EVERY_N(WARNING, 250) << "rl_supermimic: dropping malformed reference packet (" << n << " bytes)";
      continue;
    }
    float v[32];
    std::memcpy(v, buf + 8, sizeof(v));
    Eigen::VectorXd q(nj_);
    for (int i = 0; i < nj_; ++i) q(i) = v[packet_to_policy_(i)];
    Eigen::Quaterniond quat(v[25], v[26], v[27], v[28]);
    Eigen::Vector3d pos(v[29], v[30], v[31]);
    if (!q.allFinite() || !pos.allFinite() || quat.norm() < 0.5) {
      ++bad_packets_;
      continue;
    }
    ++packets_;
    if (!ref_->Push(rp.t, q, quat, pos)) ++repeats_;  // arrival timestamp from the receive thread
  }
}

bool RlSuperMimicRunner::Enter() {
  if (param_tag_ != last_param_tag_) {
    policy_model_.reset();
    ref_.reset();
    param_ = data::ParamManager::create<data::RlSuperMimicParam>(param_tag_);
    last_param_tag_ = param_tag_;
  }
  Init();
  OpenSocket();
  {  // packets from a previous session must never become the reference
    std::lock_guard<std::mutex> lk(rx_mutex_);
    if (!rx_queue_.empty()) LOG(INFO) << "rl_supermimic: discarded " << rx_queue_.size() << " stale packets";
    rx_queue_.clear();
  }
  GetMutableOutput().Reset();
  data_store_->joint_info.GetState(data::JointInfoType::kPosition, q_actual_);
  data_store_->joint_info.GetState(data::JointInfoType::kVelocity, qd_actual_);
  if (!EntryCheck()) return false;
  CaptureRobotRot0();
  ref_->Reset(MonotonicNow(), q_actual_(policy_idx_));
  hist_q_.clear(); hist_qd_.clear(); hist_w_.clear(); hist_g_.clear(); hist_a_.clear();
  action_.setZero(nj_);
  q_des_ = q_actual_;
  q_last_valid_ = q_actual_;
  q_des_filt_ = q_actual_;
  q_prev_p_ = q_actual_(policy_idx_);
  t_prev_ = MonotonicNow();
  ankle_p_.clear();
  for (int i = 0; i < nj_; ++i)
    if (param_->policy_joint_names[i].find("ANKLE") != std::string::npos) ankle_p_.push_back(i);
  leg_p_.clear();
  for (int i = 0; i < nj_; ++i) {
    const std::string& n = param_->policy_joint_names[i];
    if (n.find("HIP") != std::string::npos || n.find("KNEE") != std::string::npos || n.find("ANKLE") != std::string::npos)
      leg_p_.push_back(i);
  }
  qd_obs_filt_.resize(0);
  LOG(INFO) << "rl_supermimic: action_lpf_alpha " << param_->action_lpf_alpha << ", qd_from_q_ankles "
            << param_->qd_from_q_ankles << " (" << ankle_p_.size() << " ankle joints), qd_obs_lpf_alpha "
            << param_->qd_obs_lpf_alpha << " (" << leg_p_.size() << " leg joints)";
  fall_counter_ = 0;
  policy_bad_ = false;
  tick_ = packets_ = repeats_ = bad_packets_ = 0;
  log_.close();
  {
    // one flight log per entry (2026-11-05: re-entries after walk behaved differently from the first entry and the
    // single overwritten file lost them); flight_log.csv is a symlink to the newest one so existing tools still work
    const std::string dir =
        common::PathJoin(common::GlobalPathManager::GetInstance().GetConfigPath(), "rl_supermimic");
    const std::time_t now_c = std::time(nullptr);
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", std::localtime(&now_c));
    const std::string name = std::string("flight_log_") + stamp + ".csv";
    log_.open(common::PathJoin(dir, name), std::ios::trunc);
    const std::string link = common::PathJoin(dir, "flight_log.csv");
    std::remove(link.c_str());
    if (::symlink(name.c_str(), link.c_str()) != 0) LOG(WARNING) << "rl_supermimic: could not link " << link;
    LOG(INFO) << "rl_supermimic: flight log " << name;
  }
  log_ << "t,infer_ms,w_live,ref_height,z_offset,samples,packets,repeats,up_z";
  for (const char* blk : {"a", "q", "q_des", "q_ref", "qd"})
    for (int i = 0; i < nj_; ++i) log_ << "," << blk << i;
  log_ << ",gx,gy,gz,wx,wy,wz\n";
  LOG(INFO) << "rl_supermimic: entered (build " << __DATE__ << " " << __TIME__ << "), policy runs on the standing "
            << "reference; the UDP stream takes over when packets arrive";
  return true;
}

bool RlSuperMimicRunner::EntryCheck() {
  if (param_->entry_force_refuse) {
    LOG(ERROR) << "rl_supermimic: ENTRY REFUSED (entry_force_refuse test flag)";
    return false;
  }
  if (!param_->entry_check) return true;
  std::ostringstream why;
  for (int i = 0; i < nj_; ++i) {
    const std::string& n = param_->policy_joint_names[i];
    const bool leg = n.find("HIP_PITCH") != std::string::npos || n.find("KNEE") != std::string::npos ||
                     n.find("ANKLE") != std::string::npos;
    if (!leg) continue;
    const double dev = q_actual_(policy_idx_(i)) - q_default_p_(i);
    if (!std::isfinite(dev) || std::abs(dev) > param_->entry_leg_tol)
      why << " " << n << " " << dev << " rad off default (tol " << param_->entry_leg_tol << ");";
  }
  const Eigen::Vector3d w = data_store_->imu_info.Get()->angular_velocity;
  if (!w.allFinite() || w.norm() > param_->entry_gyro_max)
    why << " gyro " << w.norm() << " rad/s (max " << param_->entry_gyro_max << ");";
  const Eigen::Matrix3d Rb = RobotBaseRotation();
  const double tilt_deg = std::acos(std::clamp(Rb(2, 2), -1.0, 1.0)) * 180.0 / M_PI;
  if (!(tilt_deg <= param_->entry_tilt_max_deg))
    why << " IMU tilt " << tilt_deg << " deg (max " << param_->entry_tilt_max_deg << ");";
  if (why.str().empty()) {
    const Eigen::Vector3d gb = Rb.transpose() * Eigen::Vector3d(0.0, 0.0, -1.0);  // gravity in the IMU frame
    LOG(INFO) << "rl_supermimic: entry check passed (gyro " << w.norm() << " rad/s, tilt " << tilt_deg
              << " deg: IMU-frame roll " << std::atan2(gb.y(), -gb.z()) * 180.0 / M_PI << " deg, pitch "
              << std::asin(std::clamp(-gb.x(), -1.0, 1.0)) * 180.0 / M_PI << " deg; rot mode "
              << param_->robot_rot_mode << ")";
    return true;
  }
  LOG(ERROR) << "rl_supermimic: ENTRY REFUSED - not standing still on both feet:" << why.str()
             << " Stop walking, let the robot settle, then retry.";
  return false;
}

void RlSuperMimicRunner::Init() {
  const std::string cfg_root = common::GlobalPathManager::GetInstance().GetConfigPath();
  const std::string policy_path = common::PathJoin(cfg_root, param_->policy_path);
  nj_ = static_cast<int>(param_->policy_joint_names.size());
  const int obs_dim = param_->obs_history_length * (3 * nj_ + 6);
  const int goal_dim = param_->future_frames * (2 * nj_ + 10);

  // POLICY/CONFIG DIMENSION CHECK (see rl_teleop 2026-09-02: a mismatched .mnn returns frozen garbage that slams
  // the joints). Refuse to run on a mismatch.
  const int have = PolicyInputDim(policy_path);
  if (have != obs_dim + goal_dim) {
    std::ostringstream m;
    m << "rl_supermimic: POLICY/CONFIG DIMENSION MISMATCH - refusing to run. " << policy_path << " input " << have
      << ", config builds " << obs_dim << " obs + " << goal_dim << " goal = " << obs_dim + goal_dim;
    LOG(ERROR) << m.str();
    throw std::runtime_error(m.str());
  }
  if (!policy_model_) {
    policy_model_ = std::make_shared<math::MNNModel>(policy_path);
    // SELF-TEST through the exact inference call used for control: recorded inputs + torch outputs written by
    // export_policy.sh next to the .mnn. Catches fp16 weights ("--fp16 false" = fp16!), unsupported fused ops,
    // runtime/precision mismatches. Refuses to run above 1e-3.
    const std::string xs = policy_path + ".selftest_x.f32", ys = policy_path + ".selftest_y.f32";
    std::ifstream fx(xs, std::ios::binary), fy(ys, std::ios::binary);
    if (fx && fy) {
      const int D = obs_dim + goal_dim;
      std::vector<float> x(D), y(nj_);
      double worst = 0.0;
      int n = 0;
      while (fx.read(reinterpret_cast<char*>(x.data()), D * 4) && fy.read(reinterpret_cast<char*>(y.data()), nj_ * 4)) {
        Eigen::VectorXf a = policy_model_->Inference(Eigen::Map<Eigen::VectorXf>(x.data(), D));
        for (int i = 0; i < nj_; ++i) worst = std::max(worst, static_cast<double>(std::abs(a(i) - y[i])));
        ++n;
      }
      LOG(INFO) << "rl_supermimic: policy self-test " << n << " samples, max |mnn - torch| = " << worst;
      if (n == 0 || worst > 1e-3) {
        const std::string m = "rl_supermimic: POLICY SELF-TEST FAILED (max err " + std::to_string(worst) +
                              ") - refusing to run this .mnn";
        LOG(ERROR) << m;
        policy_model_.reset();
        throw std::runtime_error(m);
      }
    } else {
      LOG(WARNING) << "rl_supermimic: no self-test files next to the policy (" << xs << ") - inference UNVERIFIED";
    }
  }
  LOG(INFO) << "rl_supermimic: policy " << param_->policy_path << " input " << have << " = " << obs_dim << " obs + "
            << goal_dim << " goal, " << nj_ << " actions";

  const int n = static_cast<int>(model_param_->num_total_joints);
  policy_idx_.resize(nj_);
  packet_to_policy_.resize(nj_);
  q_default_p_.resize(nj_);
  action_scale_p_.resize(nj_);
  kp_ = Eigen::VectorXd::Zero(n);
  kd_ = Eigen::VectorXd::Zero(n);
  tau_max_ = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
  q_min_ = Eigen::VectorXd::Constant(n, -6.5);
  q_max_ = Eigen::VectorXd::Constant(n, 6.5);
  q_default_ = Eigen::VectorXd::Zero(n);
  for (int i = 0; i < nj_; ++i) {
    const std::string& name = param_->policy_joint_names[i];
    auto it = model_param_->joint_id_in_total_limb.find(name);
    if (it == model_param_->joint_id_in_total_limb.end()) throw std::runtime_error("rl_supermimic: joint not in model: " + name);
    const int k = it->second;
    policy_idx_(i) = k;
    packet_to_policy_(i) = static_cast<int>(
        std::find(param_->packet_joint_names.begin(), param_->packet_joint_names.end(), name) -
        param_->packet_joint_names.begin());
    const auto& jc = param_->joints.at(name);
    q_default_p_(i) = jc.q_default;
    action_scale_p_(i) = jc.action_scale;
    q_default_(k) = jc.q_default;
    kp_(k) = jc.kp;
    kd_(k) = jc.kd;
    if (jc.tau_max > 0) tau_max_(k) = jc.tau_max;
    q_min_(k) = jc.q_min;
    q_max_(k) = jc.q_max;
  }
  if (n != nj_) LOG(WARNING) << "rl_supermimic: model has " << n << " joints, policy drives " << nj_;
  q_actual_.resize(n);
  qd_actual_.resize(n);
  q_des_ = q_default_;
  qd_des_ = Eigen::VectorXd::Zero(n);
  tau_ff_ = Eigen::VectorXd::Zero(n);

  if (!ref_) {
    rl_supermimic::ReferenceConfig rc;
    rc.num_joints = nj_;
    rc.future_frames = param_->future_frames;
    rc.dt = runner_period_ > 0 ? runner_period_ : 0.02;
    rc.delay_s = param_->ref_delay_s;
    rc.sigma_frames = param_->ref_sigma_frames;
    rc.stale_s = param_->ref_stale_s;
    rc.blend_in_s = param_->ref_blend_in_s;
    rc.blend_out_s = param_->ref_blend_out_s;
    rc.calib_s = param_->ref_height_calib_s;
    rc.stand_height = param_->stand_height;
    rc.entry_blend_s = param_->entry_blend_s;
    ref_ = std::make_unique<rl_supermimic::ReferenceBuffer>(rc, q_default_p_);
    if (std::abs(rc.dt - 0.02) > 1e-6)
      LOG(ERROR) << "rl_supermimic: runner period " << rc.dt << " s != 0.02 s: the policy was trained at 50 Hz";
  }
}

Eigen::Matrix3d RlSuperMimicRunner::RobotBaseRotation() const {
  return math::RotationMatrixd(data_store_->imu_info.Get()->quaternion).matrix();
}

void RlSuperMimicRunner::CaptureRobotRot0() {
  robot_rot0_ = RobotBaseRotation();
  robot_yaw0_ = YawRotation(robot_rot0_);
  const Eigen::Matrix3d tilt0 = robot_yaw0_.transpose() * robot_rot0_;
  const double tilt_deg = std::acos(std::clamp(tilt0(2, 2), -1.0, 1.0)) * 180.0 / M_PI;
  LOG(INFO) << "rl_supermimic: entry frame captured, raw IMU tilt " << tilt_deg << " deg";
  if (tilt_deg > 10.0)
    LOG(WARNING) << "rl_supermimic: IMU tilt at entry > 10 deg - robot not upright or IMU mount not yaw-only; the "
                 << "gravity observation assumes an upright entry";
}

// Base orientation relative to the (upright) entry frame, expressed in entry-heading axes (rl_teleop heading_fixed:
// cancels a constant IMU mount offset, keeps tilt axes = body axes at any world heading).
Eigen::Matrix3d RlSuperMimicRunner::RobotRotRelative() const {
  const Eigen::Matrix3d delta_world = RobotBaseRotation() * robot_rot0_.transpose();
  if (param_->robot_rot_mode == "legacy") return delta_world;
  if (param_->robot_rot_mode == "absolute") return robot_yaw0_.transpose() * RobotBaseRotation();  // no entry zeroing
  return robot_yaw0_.transpose() * delta_world * robot_yaw0_;
}

Eigen::VectorXf RlSuperMimicRunner::BuildObservationHistory() {
  const Eigen::Matrix3d R = RobotRotRelative();
  Eigen::Vector3d w = data_store_->imu_info.Get()->angular_velocity;
  if (param_->imu_ang_vel_world) w = R.transpose() * w;
  const Eigen::Vector3d g = R.transpose() * Eigen::Vector3d(0.0, 0.0, -1.0);
  const Eigen::VectorXd qp = q_actual_(policy_idx_) - q_default_p_;
  Eigen::VectorXd qdp = qd_actual_(policy_idx_);
  {
    const Eigen::VectorXd q_now = q_actual_(policy_idx_);
    const double now = MonotonicNow();
    const double dt = now - t_prev_;
    if (param_->qd_from_q_ankles && q_prev_p_.size() == nj_ && dt > 0.005 && dt < 0.1)
      for (int i : ankle_p_) qdp(i) = (q_now(i) - q_prev_p_(i)) / dt;
    q_prev_p_ = q_now;
    t_prev_ = now;
  }
  if (param_->qd_obs_lpf_alpha < 1.0) {  // leg qd observation low-pass (once per policy tick)
    if (qd_obs_filt_.size() != nj_) qd_obs_filt_ = qdp;  // prime with the first sample
    const double al = param_->qd_obs_lpf_alpha;
    for (int i : leg_p_) {
      qd_obs_filt_(i) += al * (qdp(i) - qd_obs_filt_(i));
      qdp(i) = qd_obs_filt_(i);
    }
  }
  const int H = param_->obs_history_length;
  auto push = [&](std::deque<Eigen::VectorXd>& ring, const Eigen::VectorXd& cur) {
    if (ring.empty()) ring.assign(H, cur);  // first frame back-fills the window (as after an env reset)
    else {
      ring.push_back(cur);
      while (static_cast<int>(ring.size()) > H) ring.pop_front();
    }
  };
  push(hist_q_, qp);
  push(hist_qd_, qdp);
  push(hist_w_, w);
  push(hist_g_, g);
  push(hist_a_, action_);
  // term-major, oldest first: [q x H, qd x H, w x H, g x H, a x H]
  Eigen::VectorXf obs(H * (3 * nj_ + 6));
  int k = 0;
  for (auto* ring : {&hist_q_, &hist_qd_, &hist_w_, &hist_g_, &hist_a_})
    for (const auto& f : *ring) {
      obs.segment(k, f.size()) = f.cast<float>();
      k += static_cast<int>(f.size());
    }
  return obs;
}

void RlSuperMimicRunner::Run() {
  const double now = MonotonicNow();
  data_store_->joint_info.GetState(data::JointInfoType::kPosition, q_actual_);
  data_store_->joint_info.GetState(data::JointInfoType::kVelocity, qd_actual_);
  DrainPackets(now);
  ++tick_;

  const double up_z = RobotRotRelative()(2, 2);
  fall_counter_ = up_z < param_->fall_up_z ? fall_counter_ + 1 : 0;
  if (fall_counter_ == param_->fall_ticks)
    LOG(WARNING) << "rl_supermimic: FALL detected (tilt > 60 deg) - policy halted, damping in place. Exit the mode.";
  if (policy_bad_) {
    q_des_ = q_last_valid_;  // stiff hold of the last valid target (see the latch below)
    SendMotorCommand();
    return;
  }
  if (fall_counter_ >= param_->fall_ticks) {
    q_des_ = q_actual_;  // fallen: stop fighting the floor, damp in place
    SendMotorCommand();
    return;
  }

  rl_supermimic::ReferenceInfo info;
  const auto t0 = std::chrono::steady_clock::now();
  const Eigen::VectorXf goal = ref_->Goal(now, &info);
  const Eigen::VectorXf obs = BuildObservationHistory();
  Eigen::VectorXf x(obs.size() + goal.size());
  x << obs, goal;
  Eigen::VectorXd a = policy_model_->Inference(x).cast<double>();
  const double infer_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  // Garbage-inference latch. Judged on the joint OFFSET (action_scale * a), not raw |a|: healthy SuperMimic actions
  // reach |a| ~40 on the small-scale joints (scale 0.05 -> 2 rad); a raw |a| > 20 limit (rl_teleop's) killed the policy
  // mid-walk in the 2026-11-04 sim gate. A broken .mnn returns ~1e35 / NaN, which this still catches.
  const double max_offset =
      a.size() == nj_ ? (action_scale_p_.array() * a.array()).abs().maxCoeff() : std::numeric_limits<double>::infinity();
  if (a.size() != nj_ || !a.allFinite() || max_offset > param_->action_sanity_limit) {
    policy_bad_ = true;
    LOG(ERROR) << "rl_supermimic: INSANE POLICY OUTPUT (size " << a.size() << ", max |scale*a| " << max_offset
               << " rad > " << param_->action_sanity_limit << ") - policy LATCHED OFF, holding the last valid target";
    // hold the LAST VALID target (stiff). Holding the measured pose (q_des = q every tick) has no restoring force:
    // the robot goes limp (2026-11-04 gate: knee folded 0.4 -> 2.3 rad in 0.5 s, legs crossed, fall).
    q_des_ = q_last_valid_;
    SendMotorCommand();
    return;
  }
  action_ = a;  // raw action fed back to the observation (as training's last_action)
  a = a.cwiseMax(-param_->action_clip).cwiseMin(param_->action_clip);
  q_des_ = q_actual_;  // joints not driven by the policy (none on the T800) hold position
  for (int i = 0; i < nj_; ++i) q_des_(policy_idx_(i)) = q_default_p_(i) + action_scale_p_(i) * a(i);
  // NO joint-range clamp by default: training never clamped targets, and the policy relies on targets beyond the
  // range (ankles up to ~2 rad past the stop, 93% of ticks in a stable walk) for torque at the joint limit. Clamping
  // them (rl_teleop habit) removed ankle torque and felled the robot in the 2026-11-04 sim gate. The per-joint torque
  // limit (training effort limits) bounds the force instead.
  if (param_->action_lpf_alpha < 1.0) {
    const double al = param_->action_lpf_alpha;
    for (int i = 0; i < nj_; ++i) {
      const int j = policy_idx_(i);
      q_des_filt_(j) = al * q_des_(j) + (1.0 - al) * q_des_filt_(j);
      q_des_(j) = q_des_filt_(j);
    }
  }
  if (param_->clamp_targets_to_joint_range) q_des_ = q_des_.cwiseMax(q_min_).cwiseMin(q_max_);
  q_last_valid_ = q_des_;
  if (tick_ % 250 == 1)
    LOG(INFO) << "rl_supermimic: w_live " << info.w_live << ", ref height " << info.height << ", packets " << packets_
              << " (repeats dropped " << repeats_ << ", bad " << bad_packets_ << "), infer " << infer_ms << " ms";
  Log(a, info, infer_ms);
  SendMotorCommand();
}

void RlSuperMimicRunner::ApplyTorqueLimits() {
  // motor PD: tau = kp (q_des - q) + kd (0 - qd); clamp to the training effort limit by moving q_des
  for (int i = 0; i < q_des_.size(); ++i) {
    if (!std::isfinite(tau_max_(i)) || kp_(i) < 1e-6) continue;
    const double tau = kp_(i) * (q_des_(i) - q_actual_(i)) - kd_(i) * qd_actual_(i);
    const double c = std::clamp(tau, -tau_max_(i), tau_max_(i));
    if (c != tau) q_des_(i) = q_actual_(i) + (c + kd_(i) * qd_actual_(i)) / kp_(i);
  }
}

void RlSuperMimicRunner::SendMotorCommand() {
  qd_des_.setZero();
  tau_ff_.setZero();
  if (param_->torque_limit) ApplyTorqueLimits();
  Eigen::VectorXd q_send = q_des_;
  if (param_->range_excess_as_torque) {
    const double m = param_->range_margin;
    for (int i = 0; i < q_send.size(); ++i) {
      q_send(i) = std::clamp(q_des_(i), q_min_(i) + m, q_max_(i) - m);
      tau_ff_(i) = kp_(i) * (q_des_(i) - q_send(i));  // same PD torque as the unclamped target
    }
  }
  GetMutableOutput().SetCommand(q_send, qd_des_, kp_, kd_, tau_ff_);
}

void RlSuperMimicRunner::Log(const Eigen::VectorXd& a, const rl_supermimic::ReferenceInfo& info, double infer_ms) {
  if (!log_.is_open()) return;
  const Eigen::Matrix3d R = RobotRotRelative();
  const Eigen::Vector3d g = R.transpose() * Eigen::Vector3d(0.0, 0.0, -1.0);
  const Eigen::Vector3d w = data_store_->imu_info.Get()->angular_velocity;
  log_ << std::fixed << std::setprecision(6)
       << std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count() << ","
       << infer_ms << "," << info.w_live << "," << info.height << "," << info.z_offset << "," << info.samples << ","
       << packets_ << "," << repeats_ << "," << R(2, 2);
  const Eigen::VectorXd q = q_actual_(policy_idx_), qdes = q_des_(policy_idx_), qdot = qd_actual_(policy_idx_);
  for (const Eigen::VectorXd* v : {&a, &q, &qdes, &info.q_current, &qdot})
    for (int i = 0; i < v->size(); ++i) log_ << "," << (*v)(i);
  log_ << "," << g.x() << "," << g.y() << "," << g.z() << "," << w.x() << "," << w.y() << "," << w.z() << "\n";
}

TransitionState RlSuperMimicRunner::TryExit() { return TransitionState::kCompleted; }

bool RlSuperMimicRunner::Exit() {
  data_store_->parallel_by_classic_parser.store(true);
  if (log_.is_open()) log_.close();
  LOG(INFO) << "rl_supermimic: exit after " << tick_ << " ticks, packets " << packets_ << " (repeats " << repeats_
            << ", bad " << bad_packets_ << ")";
  return true;
}

}  // namespace runner
