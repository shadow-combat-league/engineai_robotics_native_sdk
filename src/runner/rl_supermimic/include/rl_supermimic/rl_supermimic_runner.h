#pragma once

// SuperMimic live-teleop runner (T800, all 25 joints).
//
// Reference: the terminator-teleop UDP stream from deploy/pico_udp_publisher.py (TTP1, or TTP2 with the lookahead
// block ignored) -> ReferenceBuffer (200 ms delay buffer, centred smoothing, standing fallback, see
// reference_buffer.h) -> goal of 10 future frames.
// Policy (MNN, one input): [observation history | goal] -> 25 actions; q_des = q_default + action_scale * a for
// every joint, sent with the training kp/kd. No output blending: on entry the policy runs immediately and the
// REFERENCE moves from the entry pose to the stand pose (entry_blend_s), then into the live stream when packets
// arrive. Observation (per frame, term-major history of 10, oldest first):
//   [q - q_default (25), qd (25), base angular velocity body (3), projected gravity body (3), last action (25)]
//
// Safety: policy/config dimension check at load (refuses to run), action sanity latch (non-finite or huge action ->
// policy off, hold the last valid target), per-joint torque limit (no target clamp: training never clamped), fall detection (base tilt >
// 60 deg for 0.5 s -> policy off, damp in place), stale stream -> standing reference.
// Everything that touches the IMU convention mirrors rl_teleop (body-frame gyro, entry-heading tilt).

#include <atomic>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
#include <fstream>
#include <memory>

#include "basic/motion_runner.h"
#include "basic/runner_registry.h"
#include "math/mnn_model.h"
#include "parameter/global_config_initializer.h"
#include "rl_supermimic/reference_buffer.h"
#include "rl_supermimic/rl_supermimic_param.h"

namespace runner {

class RlSuperMimicRunner : public MotionRunner {
 public:
  RlSuperMimicRunner(std::string_view name, const std::shared_ptr<data::DataStore>& data_store)
      : MotionRunner(name, data_store) {
    param_ = data::ParamManager::create<data::RlSuperMimicParam>();
  }
  ~RlSuperMimicRunner();

  bool Enter() override;
  bool EntryCheck();  // false = refuse takeover (reason logged)
  void Run() override;
  TransitionState TryExit() override;
  bool Exit() override;
  void SetupContext() override;
  void TeardownContext() override;

 private:
  void Init();
  void OpenSocket();
  void DrainPackets(double now);
  void RxLoop();
  Eigen::Matrix3d RobotBaseRotation() const;
  Eigen::Matrix3d RobotRotRelative() const;
  void CaptureRobotRot0();
  Eigen::VectorXf BuildObservationHistory();
  void ApplyTorqueLimits();
  void SendMotorCommand();
  void Log(const Eigen::VectorXd& action, const rl_supermimic::ReferenceInfo& info, double infer_ms);

  std::shared_ptr<data::RlSuperMimicParam> param_;
  std::string last_param_tag_;
  std::shared_ptr<math::MNNModel> policy_model_;
  std::unique_ptr<rl_supermimic::ReferenceBuffer> ref_;
  int fd_ = -1;
  struct RawPacket {
    double t;
    std::vector<uint8_t> bytes;
  };
  std::thread rx_thread_;
  std::atomic<bool> rx_run_{false};
  std::mutex rx_mutex_;
  std::vector<RawPacket> rx_queue_;

  int nj_ = 0;
  Eigen::VectorXi policy_idx_;      // policy joint i -> model total-joint index
  Eigen::VectorXi packet_to_policy_;  // policy joint i -> position in the UDP packet
  Eigen::VectorXd q_default_p_, action_scale_p_;  // policy order
  Eigen::VectorXd kp_, kd_, tau_max_, q_min_, q_max_, q_default_;  // model total order

  Eigen::VectorXd q_actual_, qd_actual_, q_des_, qd_des_, tau_ff_;
  Eigen::VectorXd q_last_valid_;  // last sane target (latch fallback)
  Eigen::VectorXd action_;  // last raw action (policy order)
  Eigen::VectorXd q_des_filt_;  // low-passed policy targets (model total order), action_lpf_alpha
  Eigen::VectorXd q_prev_p_;    // previous tick's measured q (policy order), qd_from_q_ankles
  double t_prev_ = 0.0;
  std::vector<int> ankle_p_;    // policy indices of ANKLE joints
  std::vector<int> leg_p_;      // policy indices of HIP/KNEE/ANKLE joints (qd_obs_lpf_alpha)
  Eigen::VectorXd qd_obs_filt_;  // low-passed leg qd observation state (empty = not primed)
  std::deque<Eigen::VectorXd> hist_q_, hist_qd_, hist_w_, hist_g_, hist_a_;

  Eigen::Matrix3d robot_rot0_ = Eigen::Matrix3d::Identity();
  Eigen::Matrix3d robot_yaw0_ = Eigen::Matrix3d::Identity();
  int fall_counter_ = 0;
  bool policy_bad_ = false;
  long tick_ = 0;
  long packets_ = 0, repeats_ = 0, bad_packets_ = 0;
  std::ofstream log_;
};

}  // namespace runner

REGISTER_RUNNER(RlSuperMimicRunner, "rl_supermimic_runner", kMotion)
