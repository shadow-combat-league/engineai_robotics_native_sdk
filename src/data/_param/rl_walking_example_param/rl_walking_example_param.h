#pragma once

#include <iostream>

#include "basic_param/basic_param.h"
#include "parameter/parameter_loader.h"
namespace data {

class RlWalkingExampleParam : public BasicParam {
 public:
  RlWalkingExampleParam(std::string_view tag = "rl_walking_example") : BasicParam(tag) {
    num_actions = active_joint_names.size();
    observation_scale = Eigen::VectorXd::Zero(num_observations);
    observation_scale << Eigen::VectorXd::Constant(
        num_actions, observation_scale_dof_pos),                            // joint position - joint default position
        Eigen::VectorXd::Constant(num_actions, observation_scale_dof_vel),  // joint velocity
        Eigen::VectorXd::Ones(num_actions),                                 // last joint action
        Eigen::Vector3d::Constant(observation_scale_angular_vel),           // base angular velocity
        Eigen::Vector3d::Constant(observation_scale_quat);                  // base euler angle xyz
    command_obs_scale = Eigen::VectorXd::Zero(3);
    command_obs_scale << Eigen::Vector2d::Constant(observation_scale_linear_vel), observation_scale_angular_vel;
  };

  DEFINE_PARAM_SCOPE(scope_);

  // Sets mlp net parameters
  std::string LOAD_PARAM(policy_file);
  int LOAD_PARAM(num_observations);
  std::vector<std::string> LOAD_PARAM(active_joint_names);
  int num_actions;
  int LOAD_PARAM(num_include_obs_steps);

  // Sets observation parameters
  float LOAD_PARAM(observation_scale_linear_vel);
  float LOAD_PARAM(observation_scale_angular_vel);
  float LOAD_PARAM(observation_scale_dof_pos);
  float LOAD_PARAM(observation_scale_dof_vel);
  float LOAD_PARAM(observation_scale_quat);
  float LOAD_PARAM(observation_clip);
  Eigen::VectorXd observation_scale;
  Eigen::VectorXd command_obs_scale;

  // Sets gait parameters
  float LOAD_PARAM(cycle_time);
  float LOAD_PARAM(still_ratio);

  // Sets joint control parameters
  float LOAD_PARAM(action_clip);
  std::vector<Eigen::VectorXd> LOAD_PARAM(default_joint_q);
  std::vector<Eigen::VectorXd> LOAD_PARAM(joint_kp);
  std::vector<Eigen::VectorXd> LOAD_PARAM(joint_kd);
  std::vector<Eigen::VectorXd> LOAD_PARAM(action_scale);
  std::vector<Eigen::VectorXd> LOAD_PARAM(soft_torque_limit);
  float LOAD_PARAM(control_dt);

  // Sets sim to real fine tune parameters
  bool LOAD_PARAM(enable_remote_command_lpf);
  float LOAD_PARAM(remote_command_sampling_frequency);
  float LOAD_PARAM(remote_command_cut_off_frequency);
  Eigen::Vector3d LOAD_PARAM(command_scale_pos);
  Eigen::Vector3d LOAD_PARAM(command_scale_neg);
  Eigen::Vector3d LOAD_PARAM(command_bias);
  std::optional<float> LOAD_PARAM(command_deadband);
  float LOAD_PARAM(linear_vel_delta_bias);
  float LOAD_PARAM(angular_vel_delta_bias);
  float LOAD_PARAM(imu_install_delta_bias);
  std::optional<Eigen::Vector3d> LOAD_PARAM(imu_install_bias);
  std::optional<float> LOAD_PARAM(transition_time);

  // Sets replaced limbs parameters
  std::optional<std::vector<std::string>> LOAD_PARAM(replaced_limbs);
  std::optional<bool> LOAD_PARAM(replaced_gamepad);

  // # try to stop stablely
  bool LOAD_PARAM(use_gait);
  bool LOAD_PARAM(tune_stop);
  float LOAD_PARAM(gait_base_frequency);
  float LOAD_PARAM(gait_base_stance_duty);
  std::optional<Eigen::Vector4d> LOAD_PARAM(customized_gait_velocity);
  std::optional<Eigen::Vector4d> LOAD_PARAM(customized_gait_frequency);
  std::optional<Eigen::Vector4d> LOAD_PARAM(customized_gait_stance_duty);
  int LOAD_PARAM(num_gait_command_obs);
  float LOAD_PARAM(stop_vel_threshold);
  Eigen::Vector2f LOAD_PARAM(stop_left_phase_condition);
  Eigen::Vector2f LOAD_PARAM(stop_right_phase_condition);
  float LOAD_PARAM(stop_least_duration);
  Eigen::Vector2f LOAD_PARAM(lost_balance_in_x_threshold);
  Eigen::Vector2f LOAD_PARAM(lost_balance_in_y_threshold);
  std::vector<std::string> LOAD_PARAM(ankle_roll_link_names);

  std::optional<bool> LOAD_PARAM(enable_auto_fall_transition);
  std::optional<float> LOAD_PARAM(fall_threshold);

  std::optional<float> LOAD_PARAM(stop_foot_pos_x_threshold);

  std::optional<bool> LOAD_PARAM(use_pd_control_for_yaw);
  std::optional<float> LOAD_PARAM(yaw_control_kp);
  std::optional<float> LOAD_PARAM(yaw_control_kd);
  std::optional<Eigen::Vector2d> LOAD_PARAM(pd_yaw_cmd_limit);

  std::optional<bool> LOAD_PARAM(head_shake_enable);
  std::optional<std::vector<std::string>> LOAD_PARAM(head_joint_names);
  std::optional<Eigen::VectorXd> LOAD_PARAM(head_shake_amplitude);
  std::optional<float> LOAD_PARAM(head_shake_frequency);
  std::optional<Eigen::VectorXi> LOAD_PARAM(head_shake_flip);
  std::optional<float> LOAD_PARAM(head_random_duration);
  std::optional<std::vector<Eigen::VectorXd>> LOAD_PARAM(head_random_target_positions);

  std::optional<bool> LOAD_PARAM(speed_mode);
  std::optional<float> LOAD_PARAM(speed_ratio);

  void Update() {
    LOAD_PARAM(still_ratio);
    LOAD_PARAM(default_joint_q);
    LOAD_PARAM(joint_kp);
    LOAD_PARAM(joint_kd);
    LOAD_PARAM(enable_remote_command_lpf);
    LOAD_PARAM(remote_command_cut_off_frequency);
    LOAD_PARAM(command_scale_pos);
    LOAD_PARAM(command_scale_neg);
    LOAD_PARAM(command_bias);
    LOAD_PARAM(replaced_limbs);
    LOAD_PARAM(replaced_gamepad);
  }
};
}  // namespace data
