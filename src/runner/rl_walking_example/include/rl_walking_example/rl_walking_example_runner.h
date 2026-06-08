#pragma once

#include "basic/motion_runner.h"
#include "basic/runner_registry.h"
#include "math/first_order_low_pass_filter.h"
#include "math/mnn_model.h"
#include "model/pinocchio_interface.h"
#include "parameter/global_config_initializer.h"
#include "tool/string_join.h"
#include "rl_walking_example_param/rl_walking_example_param.h"

#include <random>
#include <type_traits>

template <typename T>
class RandomNumber {
  static_assert(std::is_integral<T>::value, "T必须是整数类型");

 private:
  std::mt19937 gen;
  std::uniform_int_distribution<T> dis;

 public:
  RandomNumber(T min, T max) : gen(std::random_device{}()), dis(min, max) {}

  T sample() { return dis(gen); }
};
namespace runner {

class RlWalkingExampleRunner : public MotionRunner {
 public:
  RlWalkingExampleRunner(std::string_view name, const std::shared_ptr<data::DataStore>& data_store);
  ~RlWalkingExampleRunner() = default;

  bool Enter() override;
  void Run() override;
  TransitionState TryExit() override;
  bool Exit() override;
  void End() override;
  void SetupContext() override;
  void TeardownContext() override;

  void Log();

 private:
  void InitializePinocchioModel();
  void UpdateIMUInstalBias();
  void UpdateRemoteCommandBias();
  void UpdateRemoteCommand();
  void ManageGaitTransition();
  void ManageRunnerTransition();
  void UpdateGaitParameters();
  void CalculateObservation();
  void CalculateMotorCommand();
  void SendMotorCommand();
  void ReplaceCommand(Eigen::VectorXd& q, Eigen::VectorXd& qd, Eigen::VectorXd& tau_ff, Eigen::VectorXd& kp,
                      Eigen::VectorXd& kd);
  void ShakeHeadOnce();
  void ShakeHeadRandom();

  enum GaitType {
    kWalk = 0,
    kStand,
  };

  std::shared_ptr<data::RlWalkingExampleParam> param_;
  std::string last_param_tag_ = "";
  float time_ = 0.0;
  float global_phase_ = 0.0;
  bool is_first_time_ = true;

  std::unique_ptr<math::MNNModel> mlp_net_;
  Eigen::MatrixXd mlp_net_observation_;
  Eigen::VectorXd mlp_net_action_;

  Eigen::VectorXd q_real_;
  Eigen::VectorXd qd_real_;
  Eigen::VectorXd q_des_;
  Eigen::VectorXd qd_des_;
  Eigen::VectorXd tau_ff_des_;
  Eigen::VectorXi active_joint_idx_;
  Eigen::VectorXd initial_joint_q_;

  Eigen::VectorXd default_joint_q_;
  Eigen::VectorXd joint_kp_;
  Eigen::VectorXd joint_kd_;
  Eigen::VectorXd action_scale_;
  Eigen::VectorXd tau_limit_;
  Eigen::VectorXd q_des_lb_;
  Eigen::VectorXd q_des_ub_;

  Eigen::Vector3d imu_install_bias_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d command_bias_ = Eigen::Vector3d::Zero();
  data::GamepadInfo last_remote_command_;
  Eigen::Vector3d command_ = Eigen::Vector3d::Zero();
  std::unique_ptr<math::FirstOrderLowPassFilter<Eigen::Vector3d>> lpf_command_;

  // gait info
  double gait_frequency_ = 0.0;
  double gait_stand_duty_cycle_ = 1.0;
  double gait_global_phase_ = 0.0;
  Eigen::Vector2d gait_phase_offset_ = Eigen::Vector2d::Zero();
  Eigen::Vector4d gait_foot_signal_;
  GaitType command_gait_ = GaitType::kStand;
  GaitType current_gait_ = GaitType::kStand;
  float stop_elapsed_time_ = 0.0;

  std::unique_ptr<model::PinocchioInterface> fixed_base_model_interface_;

  float desired_yaw_ = 0.0;
  float current_yaw_ = 0.0;
  float last_yaw_ = 0.0;
  float current_dyaw_ = 0.0;

  enum class HeadShakeMode {
    kNone = 0,
    kOnce,    // ShakeHeadOnce - sinusoidal motion
    kRandom,  // ShakeHeadRandom - random target positions
  };

  Eigen::VectorXi head_joint_idx_;
  HeadShakeMode head_shake_mode_ = HeadShakeMode::kNone;
  bool run_head_shake_ = false;
  double run_start_time_ = 0.0;
  bool flip_direction_ = false;
  bool first_head_shake_ = true;
  double segment_start_time_ = 0.0;
  bool returning_to_zero_ = false;  // Flag for returning to zero position
  std::unique_ptr<RandomNumber<int>> rng_;
  Eigen::VectorXd head_random_start_;   // Random target positions for head joints
  Eigen::VectorXd head_random_target_;  // Random target positions for head joints

  bool speed_mode_ = false;  // false: low speed, true: high speed
};
}  // namespace runner

REGISTER_RUNNER(RlWalkingExampleRunner, "rl_walking_example_runner", kMotion)
