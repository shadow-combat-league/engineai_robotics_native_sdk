#pragma once

// Live PICO teleoperation runner.
//
// Consumes the reference stream published by
// terminator-teleop-system/deploy/pico_udp_publisher.py (which runs the full
// PICO -> GMR -> calibration/slew pipeline on the operator's machine) and
// executes the exported tracking policy (.mnn) with the exact observation
// layout of deploy/sim_teleop.py:
//   [ref_jpos(25), ref_jvel(25), (anchor_pos_b(3)), anchor_ori_6d(6),
//    base_ang_vel(3), q - q_default(25), qd(25), last_action(15)]
//
// v1 scope notes:
//  - anchor_pos_b is fed as ZEROS (config: use_anchor_pos selects 127- vs
//    124-dim obs). Wiring the base state estimator's position is a follow-up;
//    zero-anchor operation is verified safe in sim (policy degrades to
//    legacy-style following, no instability).
//  - Reference yaw is aligned to the robot's IMU yaw on the first packet
//    after Enter(), so the operator's world frame and the robot's frame
//    agree on heading at mode entry.

#include <fstream>

#include "basic/motion_runner.h"
#include "basic/runner_registry.h"
#include "math/mnn_model.h"
#include "parameter/global_config_initializer.h"
#include "rl_teleop/rl_teleop_param.h"
#include "rl_teleop/udp_reference_receiver.h"

namespace runner {

class RlTeleopRunner : public MotionRunner {
 public:
  RlTeleopRunner(std::string_view name, const std::shared_ptr<data::DataStore>& data_store)
      : MotionRunner(name, data_store) {
    param_ = data::ParamManager::create<data::RlTeleopParam>();
  }
  ~RlTeleopRunner() = default;

  bool Enter() override;
  void Run() override;
  TransitionState TryExit() override;
  bool Exit() override;
  void SetupContext() override;
  void TeardownContext() override;

 private:
  void Init();
  void UpdateState();
  void UpdateReference();
  Eigen::VectorXf BuildObservation();
  void CalculateMotorCommand();
  void SendMotorCommand();
  void ApplyTorqueLimits(Eigen::VectorXd& q_des);
  double ComputeTransitionRatio() const;
  Eigen::Matrix3d RobotBaseRotation() const;

  std::shared_ptr<data::RlTeleopParam> param_;
  std::string last_param_tag_ = "";

  std::shared_ptr<math::MNNModel> policy_model_;
  math::MNNModel* policy_net_ = nullptr;
  std::unique_ptr<rl_teleop::UdpReferenceReceiver> receiver_;

  // Joint index maps (into the model's total-joint order)
  Eigen::VectorXi obs_joint_idx_;      // 25, Isaac obs order
  Eigen::VectorXi action_joint_idx_;   // 15
  Eigen::VectorXi motion_joint_idx_;   // 10
  std::vector<int> action_pos_in_obs_; // where each action joint sits in obs order
  std::vector<int> motion_pos_in_obs_; // where each motion-driven joint sits in obs order
  std::vector<int> qd_zero_pos_in_obs_; // obs positions whose velocity is masked to zero

  // Full-order control vectors (distributed from the name-keyed config)
  Eigen::VectorXd default_joint_q_;
  Eigen::VectorXd joint_kp_, joint_kd_, tau_max_;
  Eigen::VectorXd q_min_, q_max_;
  Eigen::VectorXd action_scale_;

  // State
  Eigen::VectorXd q_actual_, qd_actual_;
  Eigen::VectorXd q_des_, qd_des_, tau_ff_des_;
  Eigen::VectorXd q_des_filt_;      // action LPF state (primed on first tick)
  bool lpf_primed_ = false;
  Eigen::VectorXd initial_joint_q_;
  Eigen::VectorXd policy_action_;

  // Reference state
  Eigen::VectorXd ref_jpos_;   // 25, Isaac order, yaw-aligned stream
  Eigen::VectorXd ref_jvel_;   // 25, EMA finite difference
  Eigen::VectorXd prev_ref_jpos_;
  Eigen::VectorXd stand_ref_jpos_;    // default pose in obs order (safe hold)
  // Resume blending: after a stream restart or stale gap, blend from the
  // held reference into the incoming stream instead of snapping (a snap from
  // a held mid-stride frame to clip start measured as the forward-fall cause)
  Eigen::VectorXd resume_from_jpos_;
  Eigen::Matrix3d resume_from_rot_ = Eigen::Matrix3d::Identity();
  double resume_blend_ = 1.0;         // 1 = no blend in progress
  // Reference-orientation slew limiter state (see ref_ang_slew)
  Eigen::Matrix3d prev_ref_rot_ = Eigen::Matrix3d::Identity();
  bool slew_primed_ = false;
  // anchor_pos v2 (anchor_pos_source == "estimator"): entry anchors so both
  // positions are used as displacements in the entry-yaw common frame
  Eigen::Vector3d ref_pos_curr_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d ref_pos0_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d est_pos0_ = Eigen::Vector3d::Zero();
  Eigen::Matrix3d est_yaw0_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d ref_rot_ = Eigen::Matrix3d::Identity();
  uint32_t last_ref_seq_ = 0;
  double last_ref_time_ = 0.0;
  bool have_reference_ = false;
  bool yaw_aligned_ = false;
  // Initial frames captured at first packet; all orientations are used
  // relative to these so constant IMU mount/convention offsets cancel.
  Eigen::Matrix3d robot_rot0_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d ref_rot0_ = Eigen::Matrix3d::Identity();

  std::ofstream flight_log_;
  Eigen::Vector3d imu_install_bias_ = Eigen::Vector3d::Zero();
  int transition_iter_ = 0;
  int stale_log_counter_ = 0;
  int fall_counter_ = 0;
};

}  // namespace runner

REGISTER_RUNNER(RlTeleopRunner, "rl_teleop_runner", kMotion)
