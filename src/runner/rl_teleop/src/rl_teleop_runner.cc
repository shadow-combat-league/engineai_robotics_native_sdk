#include "rl_teleop/rl_teleop_runner.h"

#include <glog/logging.h>
#include <fstream>
#include <sstream>

#include "math/interpolation.h"
#include "math/roll_pitch_yaw.h"
#include "math/rotation_matrix.h"

namespace runner {

namespace {

Eigen::Matrix3d QuatWxyzToRot(const Eigen::Vector4d& q) {
  return Eigen::Quaterniond(q(0), q(1), q(2), q(3)).normalized().toRotationMatrix();
}

// Yaw-only rotation matrix extracted from R (heading about world z).
Eigen::Matrix3d YawRotation(const Eigen::Matrix3d& R) {
  double yaw = std::atan2(R(1, 0), R(0, 0));
  return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
}

}  // namespace

// classic parser (like pd_stand): our policy commands serial joint space.
// The RL examples use false because THEIR policies were trained against the
// non-classic ankle parsing; with false, our ankle targets were mangled
// (flight log: ankle tracking err 0.14 rad standing -> hip-roll windup -> fall).
void RlTeleopRunner::SetupContext() { data_store_->parallel_by_classic_parser.store(true); }
void RlTeleopRunner::TeardownContext() {}

bool RlTeleopRunner::Enter() {
  if (param_tag_ != last_param_tag_) {
    LOG(INFO) << "rl_teleop: reloading config for param_tag: " << param_tag_;
    policy_net_ = nullptr;
    policy_model_.reset();
    receiver_.reset();
    param_ = data::ParamManager::create<data::RlTeleopParam>(param_tag_);
    last_param_tag_ = param_tag_;
  }

  Init();

  transition_iter_ = 0;
  yaw_aligned_ = false;
  have_reference_ = false;
  policy_action_.setZero(param_->num_actions);
  ref_jvel_.setZero(rl_teleop::UdpReferenceReceiver::kNumJoints);

  GetMutableOutput().Reset();
  data_store_->joint_info.GetState(data::JointInfoType::kPosition, initial_joint_q_);
  q_des_ = initial_joint_q_;

  flight_log_.close();
  flight_log_.open(common::PathJoin(common::GlobalPathManager::GetInstance().GetConfigPath(),
                                    "rl_teleop/flight_log.csv"),
                   std::ios::trunc);

  if (param_->builtin_stand_reference) {
    // Run the policy immediately against a standing reference: default pose,
    // upright orientation, zero velocity. Equivalent to sim_teleop --hold.
    // The mount offset is captured NOW (robot upright in pd_stand).
    robot_rot0_ = RobotBaseRotation();
    ref_rot_ = Eigen::Matrix3d::Identity();
    ref_jpos_ = default_joint_q_(obs_joint_idx_);
    ref_jvel_.setZero();
    have_reference_ = true;
    LOG(INFO) << "rl_teleop: built-in standing reference active (policy holds stand)";
  }

  {
    // Diagnostic: robot is upright here (entered from pd_stand), so the true
    // orientation is yaw-only: quat ~ (w=cos(y/2), 0, 0, z=sin(y/2)).
    // Large x/y components => component-order scramble in the imu pipeline.
    Eigen::Quaterniond q = data_store_->imu_info.Get()->quaternion;
    LOG(INFO) << "rl_teleop: raw imu quat at entry (upright): w=" << q.w()
              << " x=" << q.x() << " y=" << q.y() << " z=" << q.z();
  }
  LOG(INFO) << "rl_teleop: entered. Waiting for reference stream on UDP :" << param_->udp_port
            << " (obs " << (param_->use_anchor_pos ? 127 : 124) << "-dim)";
  return true;
}

void RlTeleopRunner::Run() {
  UpdateState();
  UpdateReference();
  if (have_reference_) {
    CalculateMotorCommand();
  } else {
    // No reference yet: hold entry pose (PD to the pose captured at Enter).
    q_des_ = initial_joint_q_;
  }
  SendMotorCommand();
}

TransitionState RlTeleopRunner::TryExit() { return TransitionState::kCompleted; }

bool RlTeleopRunner::Exit() {
  data_store_->parallel_by_classic_parser.store(true);
  policy_net_ = nullptr;
  return true;
}

void RlTeleopRunner::Init() {
  // Policy
  if (!policy_model_) {
    std::string policy_path =
        common::PathJoin(common::GlobalPathManager::GetInstance().GetConfigPath(), param_->policy_path);
    policy_model_ = std::make_shared<math::MNNModel>(policy_path);
  }
  policy_net_ = policy_model_.get();
  if (!policy_net_) {
    throw std::runtime_error("rl_teleop: policy model is null: " + param_->policy_path);
  }

  // Receiver
  if (!receiver_) {
    receiver_ = std::make_unique<rl_teleop::UdpReferenceReceiver>(param_->udp_port);
  }

  // Joint index maps, all resolved by NAME against the model
  auto resolve = [&](const std::vector<std::string>& names) {
    Eigen::VectorXi idx(names.size());
    int i = 0;
    for (const auto& n : names) {
      auto it = model_param_->joint_id_in_total_limb.find(n);
      if (it == model_param_->joint_id_in_total_limb.end()) {
        throw std::runtime_error("rl_teleop: joint not in model: " + n);
      }
      idx(i++) = it->second;
    }
    return idx;
  };
  obs_joint_idx_ = resolve(param_->obs_joint_names);
  action_joint_idx_ = resolve(param_->action_joint_names);
  motion_joint_idx_ = resolve(param_->motion_driven_joint_names);

  // Positions of action/motion joints inside the obs-order vectors
  auto pos_in_obs = [&](const std::vector<std::string>& names) {
    std::vector<int> out;
    for (const auto& n : names) {
      auto it = std::find(param_->obs_joint_names.begin(), param_->obs_joint_names.end(), n);
      if (it == param_->obs_joint_names.end()) {
        throw std::runtime_error("rl_teleop: joint not in obs_joint_names: " + n);
      }
      out.push_back(static_cast<int>(it - param_->obs_joint_names.begin()));
    }
    return out;
  };
  action_pos_in_obs_ = pos_in_obs(param_->action_joint_names);
  motion_pos_in_obs_ = pos_in_obs(param_->motion_driven_joint_names);
  qd_zero_pos_in_obs_ = pos_in_obs(param_->qd_zero_joint_names);

  // Full-order control vectors from the name-keyed joints map
  const int n = static_cast<int>(model_param_->num_total_joints);
  default_joint_q_ = Eigen::VectorXd::Zero(n);
  joint_kp_ = Eigen::VectorXd::Zero(n);
  joint_kd_ = Eigen::VectorXd::Zero(n);
  tau_max_ = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());
  for (const auto& [name, jc] : param_->joints) {
    auto it = model_param_->joint_id_in_total_limb.find(name);
    if (it == model_param_->joint_id_in_total_limb.end()) {
      throw std::runtime_error("rl_teleop: joints map has unknown joint: " + name);
    }
    default_joint_q_(it->second) = jc.q_default;
    joint_kp_(it->second) = jc.kp;
    joint_kd_(it->second) = jc.kd;
    if (jc.tau_max > 0.0) tau_max_(it->second) = jc.tau_max;
  }

  action_scale_ = Eigen::Map<const Eigen::VectorXd>(param_->action_scale.data(),
                                                    static_cast<int>(param_->action_scale.size()));

  q_actual_.resize(n);
  qd_actual_.resize(n);
  q_des_ = default_joint_q_;
  qd_des_ = Eigen::VectorXd::Zero(n);
  tau_ff_des_ = Eigen::VectorXd::Zero(n);
  ref_jpos_ = default_joint_q_(obs_joint_idx_);
  prev_ref_jpos_ = ref_jpos_;
}

void RlTeleopRunner::UpdateState() {
  data_store_->joint_info.GetState(data::JointInfoType::kPosition, q_actual_);
  data_store_->joint_info.GetState(data::JointInfoType::kVelocity, qd_actual_);
}

Eigen::Matrix3d RlTeleopRunner::RobotBaseRotation() const {
  Eigen::Matrix3d R_install = math::RollPitchYawd(imu_install_bias_).ToRotationMatrix().matrix();
  Eigen::Matrix3d R_local = math::RotationMatrixd(data_store_->imu_info.Get()->quaternion).matrix();
  return R_local * R_install.transpose();
}

void RlTeleopRunner::UpdateReference() {
  bool got_new = receiver_->Poll();
  const auto& frame = receiver_->Latest();
  if (!frame.valid) return;

  if (got_new && frame.seq != last_ref_seq_) {
    double now = frame.recv_time;
    // Nominal publisher period, NOT measured arrival dt: UDP arrival jitter
    // would inject spikes into the finite-difference reference velocity.
    // Account for dropped packets via the seq gap instead.
    uint32_t seq_gap = (last_ref_seq_ > 0 && frame.seq > last_ref_seq_)
                           ? (frame.seq - last_ref_seq_) : 1;
    double dt = 0.02 * static_cast<double>(seq_gap);

    // Alignment on the first packet. The sim IMU is a SITE with a fixed
    // mount rotation C on the base: measured M(t) = W(t)*C (right-mult).
    // With the robot upright at entry (pd_stand) we define the common frame
    // as the robot's entry frame, so the true orientation is
    //   R_robot(t) = M(t) * M(0)^T          (mount cancels on the RIGHT).
    // The reference stream is a clean world measurement; align its heading
    // only: R_ref(t) = Yaw(ref0)^T * W_ref(t).
    if (!yaw_aligned_) {
      robot_rot0_ = RobotBaseRotation();  // M(0)
      ref_rot0_ = YawRotation(QuatWxyzToRot(frame.quat_wxyz));
      yaw_aligned_ = true;
      prev_ref_jpos_ = frame.jpos;
      ref_jvel_.setZero();
      LOG(INFO) << "rl_teleop: reference stream live (seq " << frame.seq
                << "), initial frames captured.";
    }

    // Reference joint velocity: EMA finite difference, clipped (mirrors
    // deploy/sim_teleop.py)
    const double a = param_->ref_jvel_alpha;
    ref_jvel_ = (1.0 - a) * ref_jvel_ + a * (frame.jpos - prev_ref_jpos_) / dt;
    ref_jvel_ = ref_jvel_.cwiseMax(-param_->ref_jvel_clip).cwiseMin(param_->ref_jvel_clip);
    prev_ref_jpos_ = frame.jpos;

    ref_jpos_ = frame.jpos;
    ref_rot_ = ref_rot0_.transpose() * QuatWxyzToRot(frame.quat_wxyz);  // heading-aligned
    last_ref_seq_ = frame.seq;
    last_ref_time_ = now;
    have_reference_ = true;
  }

  if (have_reference_ && receiver_->Staleness() > param_->ref_stale_timeout) {
    // Hold the last reference (tracking policy holds pose); throttled warning.
    if (++stale_log_counter_ % 250 == 1) {
      LOG(WARNING) << "rl_teleop: reference stream stale for " << receiver_->Staleness()
                   << "s - holding last reference";
    }
    ref_jvel_.setZero();
  } else {
    stale_log_counter_ = 0;
  }
}

Eigen::VectorXf RlTeleopRunner::BuildObservation() {
  const int nj = rl_teleop::UdpReferenceReceiver::kNumJoints;
  const int na = param_->num_actions;
  const int dim = 2 * nj + (param_->use_anchor_pos ? 3 : 0) + 6 + 3 + 2 * nj + na;
  Eigen::VectorXd obs(dim);
  int k = 0;

  // command: ref joint positions + velocities (Isaac obs order)
  obs.segment(k, nj) = ref_jpos_; k += nj;
  obs.segment(k, nj) = ref_jvel_; k += nj;

  // anchor position error in robot base frame — v1: zeros (see header note)
  if (param_->use_anchor_pos) {
    obs.segment(k, 3).setZero(); k += 3;
  }

  // anchor orientation error: first two COLUMNS of R_robot^T * R_ref in the
  // common (robot-entry) frame, flattened ROW-major to match numpy's
  // mat[:, :2].reshape(-1). Mount offset cancels on the RIGHT (see above).
  Eigen::Matrix3d robot_rot = RobotBaseRotation() * robot_rot0_.transpose();
  Eigen::Matrix3d rel = robot_rot.transpose() * ref_rot_;
  obs(k + 0) = rel(0, 0); obs(k + 1) = rel(0, 1);
  obs(k + 2) = rel(1, 0); obs(k + 3) = rel(1, 1);
  obs(k + 4) = rel(2, 0); obs(k + 5) = rel(2, 1);
  k += 6;

  // base angular velocity in BASE frame. The sim publishes frameangvel =
  // WORLD-frame angular velocity; rotate into the (mount-corrected) base.
  {
    Eigen::Vector3d w_world = data_store_->imu_info.Get()->angular_velocity;
    obs.segment(k, 3) = robot_rot.transpose() * w_world;
    k += 3;
  }

  // proprioception (Isaac obs order); masked joints get zero velocity
  obs.segment(k, nj) = q_actual_(obs_joint_idx_) - default_joint_q_(obs_joint_idx_); k += nj;
  {
    Eigen::VectorXd qd_obs = qd_actual_(obs_joint_idx_);
    for (int p : qd_zero_pos_in_obs_) qd_obs(p) = 0.0;
    obs.segment(k, nj) = qd_obs; k += nj;
  }
  obs.segment(k, na) = policy_action_; k += na;

  return obs.cast<float>();
}

void RlTeleopRunner::CalculateMotorCommand() {
  Eigen::VectorXf obs = BuildObservation();

  // Debug: dump the first few observation/action pairs for offline diffing
  // against the Python deploy sim (deploy/sim_teleop.py) on the same clip.
  static int dump_count = 0;
  if (dump_count < 3) {
    std::ostringstream ss;
    ss.precision(5);
    for (int i = 0; i < obs.size(); ++i) ss << obs(i) << (i + 1 < obs.size() ? "," : "");
    LOG(INFO) << "rl_teleop OBS[" << dump_count << "] dim=" << obs.size() << " : " << ss.str();
  }

  policy_action_ = policy_net_->Inference(obs).cast<double>();

  if (dump_count < 3) {
    std::ostringstream ss;
    ss.precision(5);
    for (int i = 0; i < policy_action_.size(); ++i)
      ss << policy_action_(i) << (i + 1 < policy_action_.size() ? "," : "");
    LOG(INFO) << "rl_teleop ACT[" << dump_count << "] : " << ss.str();
    ++dump_count;
  }

  // Flight recorder: truncated at each mode entry (one clean take per
  // attempt). Columns: obs(127|124), act(15), q_actual(25), q_des(25).
  if (flight_log_.is_open()) {
    flight_log_.precision(6);
    for (int i = 0; i < obs.size(); ++i) flight_log_ << obs(i) << ",";
    for (int i = 0; i < policy_action_.size(); ++i) flight_log_ << policy_action_(i) << ",";
    Eigen::VectorXd qo = q_actual_(obs_joint_idx_);
    for (int i = 0; i < qo.size(); ++i) flight_log_ << qo(i) << ",";
    Eigen::VectorXd qd_cmd = q_des_(obs_joint_idx_);
    for (int i = 0; i < qd_cmd.size(); ++i) flight_log_ << qd_cmd(i) << (i + 1 < qd_cmd.size() ? "," : "\n");
  }
  policy_action_ = policy_action_.cwiseMax(-param_->action_clip).cwiseMin(param_->action_clip);

  // Targets: default + scaled action for action joints; reference passthrough
  // for motion-driven (arm) joints — mirrors DeployedPolicy.act()
  q_des_ = default_joint_q_;
  for (int i = 0; i < policy_action_.size(); ++i) {
    q_des_(action_joint_idx_(i)) =
        default_joint_q_(action_joint_idx_(i)) + action_scale_(i) * policy_action_(i);
  }
  for (size_t i = 0; i < motion_pos_in_obs_.size(); ++i) {
    q_des_(motion_joint_idx_(static_cast<int>(i))) = ref_jpos_(motion_pos_in_obs_[i]);
  }

  // Blend in from the entry pose over transition_time
  double ratio = ComputeTransitionRatio();
  if (ratio < 1.0) {
    q_des_ = math::LinearInterpolate(initial_joint_q_, q_des_, ratio);
    ++transition_iter_;
  }
}

double RlTeleopRunner::ComputeTransitionRatio() const {
  double t = param_->transition_time;
  return t > 0.0 ? std::min(1.0, static_cast<double>(transition_iter_) * runner_period_ / t) : 1.0;
}

void RlTeleopRunner::SendMotorCommand() {
  qd_des_.setZero();
  tau_ff_des_.setZero();

  if (param_->torque_limit) {
    ApplyTorqueLimits(q_des_);
  }
  for (int i = 0; i < q_des_.size(); ++i) {
    if (q_des_(i) > 6.5) {
      LOG_EVERY_N(WARNING, 100) << "rl_teleop: joint " << i << " q_des clamps at 6.5";
      q_des_(i) = 6.5;
    }
  }
  GetMutableOutput().SetCommand(q_des_, qd_des_, joint_kp_, joint_kd_, tau_ff_des_);
}

void RlTeleopRunner::ApplyTorqueLimits(Eigen::VectorXd& q_des) {
  const int n = static_cast<int>(q_des.size());
  Eigen::VectorXd tau =
      (joint_kp_.array() * (q_des - q_actual_).array() - joint_kd_.array() * qd_actual_.array()).matrix();

  for (int i = 0; i < n; ++i) {
    const double m = tau_max_(i);
    if (std::isfinite(m)) tau(i) = std::clamp(tau(i), -m, m);
  }
  const int lb = param_->lower_body_joint_count;
  if (n >= lb && param_->max_lower_body_torque > 0.0) {
    const double sum_abs = tau.head(lb).cwiseAbs().sum();
    if (sum_abs > param_->max_lower_body_torque) {
      tau.head(lb) *= (param_->max_lower_body_torque / sum_abs);
    }
  }
  const double eps = 1e-6;
  for (int i = 0; i < n; ++i) {
    if (std::abs(joint_kp_(i)) > eps && std::isfinite(joint_kp_(i))) {
      q_des(i) = q_actual_(i) + (tau(i) + joint_kd_(i) * qd_actual_(i)) / joint_kp_(i);
    }
  }
}

}  // namespace runner
