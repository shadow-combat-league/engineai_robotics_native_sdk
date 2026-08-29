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

// A/B COMPLETE (2026-08-25, fresh binaries both arms): false = under-driven
// even standing (vendor RL modes use it because THEIR policies trained
// against that transform); true (classic, like pd_stand) = correct for our
// serial joint-space policy. Do not flip again.
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
  fall_counter_ = 0;
  lpf_primed_ = false;
  resume_blend_ = 1.0;
  // Packets buffered from a previous session must never become the reference
  receiver_->Reset();
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
    // Run the policy immediately against a standing reference, STARTING AT
    // THE MEASURED ENTRY POSE. pd_stand parks the robot away from the
    // policy default (measured: hip yaws +-0.27 rad, elbows 0.15) — a
    // reference that jumps straight to the default forces the policy to
    // twist loaded legs at t=0 (the entry stumble). Starting the reference
    // at the current pose makes entry a zero-error event; the stale-decay
    // path (no packets -> decay to stand_ref_jpos_, ~1 s tau) then migrates
    // the pose to the default stand as a slow commanded motion.
    // The mount offset is captured NOW (robot upright in pd_stand).
    robot_rot0_ = RobotBaseRotation();
    ref_rot_ = Eigen::Matrix3d::Identity();
    ref_jpos_ = initial_joint_q_(obs_joint_idx_);
    ref_jvel_.setZero();
    have_reference_ = true;
    LOG(INFO) << "rl_teleop: built-in standing reference active (entry pose -> "
              << "default stand via decay; max entry offset "
              << (initial_joint_q_(obs_joint_idx_) - default_joint_q_(obs_joint_idx_))
                     .cwiseAbs().maxCoeff() << " rad)";
  }

  {
    // Diagnostic: robot is upright here (entered from pd_stand), so the true
    // orientation is yaw-only: quat ~ (w=cos(y/2), 0, 0, z=sin(y/2)).
    // Large x/y components => component-order scramble in the imu pipeline.
    Eigen::Quaterniond q = data_store_->imu_info.Get()->quaternion;
    LOG(INFO) << "rl_teleop: raw imu quat at entry (upright): w=" << q.w()
              << " x=" << q.x() << " y=" << q.y() << " z=" << q.z();
  }
  LOG(INFO) << "rl_teleop: build " << __DATE__ << " " << __TIME__
            << " — entered. Waiting for reference stream on UDP :" << param_->udp_port
            << " (obs " << (param_->use_anchor_pos ? 127 : 124) << "-dim)";
  return true;
}

void RlTeleopRunner::Run() {
  UpdateState();
  UpdateReference();

  // Fall detection: base tilted > ~60 deg from upright for 0.5s -> stop
  // executing the policy (on hardware: stop fighting the floor). Damps to
  // zero targets at low gains until the operator exits the mode.
  {
    Eigen::Matrix3d robot_rot = RobotBaseRotation() * robot_rot0_.transpose();
    double up_z = robot_rot(2, 2);  // world-z component of base up-axis
    fall_counter_ = (up_z < 0.5) ? fall_counter_ + 1 : 0;
    if (fall_counter_ == 25) {
      LOG(WARNING) << "rl_teleop: FALL detected (base tilt > 60 deg) - policy halted, "
                   << "damping. LB+RB or mode exit to recover.";
    }
  }
  if (fall_counter_ >= 25) {
    q_des_ = q_actual_;  // damp in place: hold measured pose at PD, no policy
    SendMotorCommand();
    return;
  }

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
  q_min_ = Eigen::VectorXd::Constant(n, -6.5);
  q_max_ = Eigen::VectorXd::Constant(n, 6.5);
  for (const auto& [name, jc] : param_->joints) {
    auto it = model_param_->joint_id_in_total_limb.find(name);
    if (it == model_param_->joint_id_in_total_limb.end()) {
      throw std::runtime_error("rl_teleop: joints map has unknown joint: " + name);
    }
    default_joint_q_(it->second) = jc.q_default;
    joint_kp_(it->second) = jc.kp;
    joint_kd_(it->second) = jc.kd;
    if (jc.tau_max > 0.0) tau_max_(it->second) = jc.tau_max;
    q_min_(it->second) = jc.q_min;
    q_max_(it->second) = jc.q_max;
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
  stand_ref_jpos_ = ref_jpos_;
}

void RlTeleopRunner::UpdateState() {
  data_store_->joint_info.GetState(data::JointInfoType::kPosition, q_actual_);
  data_store_->joint_info.GetState(data::JointInfoType::kVelocity, qd_actual_);

  // PROBE (anchor_pos v2 feasibility): is the base state estimator populated?
  // If this stays zero, the estimator does not run in this deployment and
  // anchor_pos_b must stay zeros; if it tracks walking, wire it up.
  {
    const auto base = data_store_->base_state_in_world.Get();
    LOG_EVERY_N(INFO, 250) << "rl_teleop probe base_state_in_world pos=["
                           << base->frame.pose.position.transpose() << "] vel=["
                           << base->frame.twist.linear.transpose() << "]";
  }
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

  // Publisher restart (seq reset) OR resume after a stale gap => treat as a
  // fresh stream: re-align its heading to the robot's CURRENT orientation and
  // BLEND from the held reference into the new stream instead of snapping.
  // (Measured 2026-08-25: held mid-stride frame -> clip-start snap put the
  // robot 6.5 deg forward with wound-up ankles; it fell 4 s later.)
  if (got_new && have_reference_ &&
      (frame.seq < last_ref_seq_ ||
       frame.recv_time - last_ref_time_ > param_->ref_stale_timeout)) {
    LOG(INFO) << "rl_teleop: reference stream "
              << (frame.seq < last_ref_seq_ ? "restarted" : "resumed after stale gap")
              << " (seq " << last_ref_seq_ << " -> " << frame.seq
              << ") - re-aligning and blending in";
    yaw_aligned_ = false;
    resume_from_jpos_ = ref_jpos_;
    resume_from_rot_ = ref_rot_;
    resume_blend_ = 0.0;
  }

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
      slew_primed_ = false;
      prev_ref_jpos_ = frame.jpos;
      ref_jvel_.setZero();
      // anchor_pos v2: capture position anchors so displacements start at 0
      ref_pos0_ = frame.pos;
      prev_ref_pos_ = frame.pos;
      anchor_err_c_.setZero();
      {
        const auto base = data_store_->base_state_in_world.Get();
        est_pos0_ = base->frame.pose.position;
        est_yaw0_ = YawRotation(base->frame.pose.quaternion.toRotationMatrix());
      }
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
    ref_pos_curr_ = frame.pos;

    // "velocity" anchor mode: integrate the velocity mismatch with a leak.
    // v_ref from packet deltas (exact); v_robot from the estimator's twist
    // (its instantaneous velocity is usable even though its integrated
    // position under-reports). Entry-yaw common frame on both sides.
    if (param_->anchor_pos_source == "velocity") {
      Eigen::Vector3d v_ref_c = ref_rot0_.transpose() * (frame.pos - prev_ref_pos_) / dt;
      const auto base = data_store_->base_state_in_world.Get();
      Eigen::Vector3d v_rob_c = est_yaw0_.transpose() * base->frame.twist.linear;
      anchor_err_c_ = std::exp(-dt / param_->anchor_vel_tau) * anchor_err_c_ +
                      dt * (v_ref_c - v_rob_c);
      LOG_EVERY_N(INFO, 50) << "rl_teleop anchor-vel: |v_ref|=" << v_ref_c.norm()
                            << " |v_rob|=" << v_rob_c.norm()
                            << " err=[" << anchor_err_c_.transpose() << "]";
    }
    prev_ref_pos_ = frame.pos;

    ref_rot_ = ref_rot0_.transpose() * QuatWxyzToRot(frame.quat_wxyz);  // heading-aligned

    // Slew-limit the reference orientation rate: consume large rotations
    // (spins beyond the plant's yaw ceiling) at ref_ang_slew instead of
    // letting the anchor error grow uncloseably. Inert for normal walking.
    if (!slew_primed_) {
      prev_ref_rot_ = ref_rot_;
      slew_primed_ = true;
    } else if (param_->ref_ang_slew < 10.0) {
      Eigen::AngleAxisd aa(prev_ref_rot_.transpose() * ref_rot_);
      const double max_step = param_->ref_ang_slew * dt;
      if (std::abs(aa.angle()) > max_step) {
        ref_rot_ = prev_ref_rot_ *
                   Eigen::AngleAxisd(max_step * (aa.angle() > 0 ? 1.0 : -1.0),
                                     aa.axis()).toRotationMatrix();
      }
      prev_ref_rot_ = ref_rot_;
    } else {
      prev_ref_rot_ = ref_rot_;
    }

    last_ref_seq_ = frame.seq;
    last_ref_time_ = now;
    have_reference_ = true;

    // Blend from the held reference into the (re)started stream over 0.5 s
    if (resume_blend_ < 1.0) {
      resume_blend_ = std::min(1.0, resume_blend_ + runner_period_ / 0.5);
      const double r = resume_blend_;
      ref_jpos_ = (1.0 - r) * resume_from_jpos_ + r * ref_jpos_;
      Eigen::Quaterniond qa(resume_from_rot_), qb(ref_rot_);
      ref_rot_ = qa.slerp(r, qb).toRotationMatrix();
      // (ref_jvel restart is handled by the yaw re-align block above, which
      // resets prev_ref_jpos_ to the raw stream on the first packet)
    }
  }

  if (have_reference_ && receiver_->Staleness() > param_->ref_stale_timeout) {
    // Decay the held reference to the built-in stand (default pose, upright
    // at the current heading) with a ~1 s time constant. Holding an arbitrary
    // frozen frame is NOT safe: a mid-stride frame is single-support and
    // base-pitched (measured: held walking02 frame 824, 5.9 deg forward,
    // robot leaned 6.5 deg and fell when the stream resumed).
    if (++stale_log_counter_ % 250 == 1) {
      LOG(WARNING) << "rl_teleop: reference stream stale for " << receiver_->Staleness()
                   << "s - decaying to stand reference";
    }
    ref_jvel_.setZero();
    const double k = runner_period_ / 1.0;  // ~1 s time constant
    ref_jpos_ += k * (stand_ref_jpos_ - ref_jpos_);
    Eigen::Quaterniond qr(ref_rot_), qup(YawRotation(ref_rot_));
    ref_rot_ = qr.slerp(k, qup).toRotationMatrix();
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

  Eigen::Matrix3d robot_rot = RobotBaseRotation() * robot_rot0_.transpose();

  // anchor position error in the robot base frame (training's
  // subtract_frame_transforms, clipped +-1 m like deploy/sim_teleop.py).
  // Both positions enter as displacements from their first-packet anchors,
  // expressed in the entry-yaw common frame; "none" keeps v1 zeros.
  if (param_->use_anchor_pos) {
    if (param_->anchor_pos_source == "estimator" && yaw_aligned_) {
      const auto base = data_store_->base_state_in_world.Get();
      Eigen::Vector3d p_robot = est_yaw0_.transpose() * (base->frame.pose.position - est_pos0_);
      Eigen::Vector3d p_ref = ref_rot0_.transpose() * (ref_pos_curr_ - ref_pos0_);
      Eigen::Vector3d err = robot_rot.transpose() * (p_ref - p_robot);
      obs.segment(k, 3) = err.cwiseMax(-1.0).cwiseMin(1.0);
    } else if (param_->anchor_pos_source == "velocity" && yaw_aligned_) {
      Eigen::Vector3d err = robot_rot.transpose() * anchor_err_c_;
      obs.segment(k, 3) = err.cwiseMax(-1.0).cwiseMin(1.0);
    } else {
      obs.segment(k, 3).setZero();
    }
    k += 3;
  }

  // anchor orientation error: first two COLUMNS of R_robot^T * R_ref in the
  // common (robot-entry) frame, flattened ROW-major to match numpy's
  // mat[:, :2].reshape(-1). Mount offset cancels on the RIGHT (see above).
  Eigen::Matrix3d rel = robot_rot.transpose() * ref_rot_;
  obs(k + 0) = rel(0, 0); obs(k + 1) = rel(0, 1);
  obs(k + 2) = rel(1, 0); obs(k + 3) = rel(1, 1);
  obs(k + 4) = rel(2, 0); obs(k + 5) = rel(2, 1);
  k += 6;

  // base angular velocity in BASE frame. Sim: frameangvel is WORLD-frame ->
  // rotate into the (mount-corrected) base. Hardware: the IMU gyro already
  // reports body-frame -> pass through.
  {
    Eigen::Vector3d w = data_store_->imu_info.Get()->angular_velocity;
    obs.segment(k, 3) = param_->imu_ang_vel_world ? (robot_rot.transpose() * w).eval() : w;
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
    for (int i = 0; i < qd_cmd.size(); ++i) flight_log_ << qd_cmd(i) << ",";
    // raw imu quat (wxyz) + latest received reference quat (wxyz): lets the
    // host-side analyzer check component conventions directly
    Eigen::Quaterniond iq = data_store_->imu_info.Get()->quaternion;
    flight_log_ << iq.w() << "," << iq.x() << "," << iq.y() << "," << iq.z() << ",";
    const auto& fr = receiver_->Latest();
    flight_log_ << fr.quat_wxyz(0) << "," << fr.quat_wxyz(1) << ","
                << fr.quat_wxyz(2) << "," << fr.quat_wxyz(3) << "\n";
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

  // Clamp targets to physical joint ranges: beyond-range targets excite the
  // downstream command pipeline in sim (measured 1.1 rad ankle demands vs a
  // 0.68 rad range at the turn trip) and slam hard stops on hardware.
  q_des_ = q_des_.cwiseMax(q_min_).cwiseMin(q_max_);

  // One-pole low-pass on the commanded targets (see action_lpf_alpha in the
  // param header). last_action_ fed back to the obs stays raw.
  if (param_->action_lpf_alpha < 1.0) {
    if (!lpf_primed_) {
      q_des_filt_ = q_des_;
      lpf_primed_ = true;
    }
    q_des_filt_ = param_->action_lpf_alpha * q_des_ +
                  (1.0 - param_->action_lpf_alpha) * q_des_filt_;
    q_des_ = q_des_filt_;
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
