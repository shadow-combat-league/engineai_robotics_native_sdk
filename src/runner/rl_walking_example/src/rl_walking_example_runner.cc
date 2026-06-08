#include "rl_walking_example/rl_walking_example_runner.h"

#include <glog/logging.h>
#include <iostream>

#include "math/interpolation.h"
#include "math/ranges.h"
#include "math/rotation_matrix.h"
#include "tool/concatenate_vector.h"

namespace runner {

RlWalkingExampleRunner::RlWalkingExampleRunner(std::string_view name, const std::shared_ptr<data::DataStore>& data_store)
    : MotionRunner(name, data_store) {
  // Creates param ptr
  param_ = data::ParamManager::create<data::RlWalkingExampleParam>();
  InitializePinocchioModel();
}

void RlWalkingExampleRunner::SetupContext() { data_store_->parallel_by_classic_parser.store(false); }

void RlWalkingExampleRunner::TeardownContext() {}

void RlWalkingExampleRunner::InitializePinocchioModel() {
  std::string urdf_file_path =
      common::PathJoin(common::GlobalPathManager::GetInstance().GetUrdfPath(), model_param_->urdf);
  fixed_base_model_interface_ = std::make_unique<model::PinocchioInterface>(
      urdf_file_path, model::FloatingBaseType::kFixed, model_param_->verbose);
}

bool RlWalkingExampleRunner::Enter() {
  if (!param_tag_.empty() && param_tag_ != last_param_tag_) {
    param_ = data::ParamManager::create<data::RlWalkingExampleParam>(param_tag_);
    last_param_tag_ = param_tag_;
  }

  // Creates mlp net model
  mlp_net_ = std::make_unique<math::MNNModel>(
      common::PathJoin(common::GlobalPathManager::GetInstance().GetConfigPath(), param_->policy_file));
  mlp_net_observation_.setZero(param_->num_observations, param_->num_include_obs_steps);
  mlp_net_action_.setZero(param_->num_actions);

  // Initializes some low pass filters
  lpf_command_ = std::make_unique<math::FirstOrderLowPassFilter<Eigen::Vector3d>>(
      param_->remote_command_sampling_frequency, param_->remote_command_cut_off_frequency);

  // Initializes active joint num
  active_joint_idx_.setZero(param_->num_actions);
  int i = 0;
  for (const std::string& name : param_->active_joint_names) {
    active_joint_idx_(i++) = model_param_->joint_id_in_total_limb.at(name);
  }

  // Initializes head joint num
  if (param_->head_shake_enable.value_or(false) && param_->head_joint_names.has_value()) {
    head_joint_idx_.setZero(param_->head_joint_names.value().size());
    int j = 0;
    for (const std::string& name : param_->head_joint_names.value()) {
      head_joint_idx_(j++) = model_param_->joint_id_in_total_limb.at(name);
    }
    head_random_start_.setZero(head_joint_idx_.size());
    head_random_target_.setZero(head_joint_idx_.size());
    run_head_shake_ = false;
  }
  if (param_->head_random_target_positions.has_value() && !param_->head_random_target_positions.value().empty()) {
    // Randomly sample from predefined target positions
    const auto& target_positions = param_->head_random_target_positions.value();
    rng_ = std::make_unique<RandomNumber<int>>(0, target_positions.size() - 1);
  }
  head_shake_mode_ = HeadShakeMode::kNone;
  first_head_shake_ = true;

  // Initializes gait command
  command_gait_ = GaitType::kStand;
  current_gait_ = GaitType::kStand;

  default_joint_q_ = common::ConcatenateVectors(param_->default_joint_q);
  joint_kp_ = common::ConcatenateVectors(param_->joint_kp);
  joint_kd_ = common::ConcatenateVectors(param_->joint_kd);
  action_scale_ = common::ConcatenateVectors(param_->action_scale);
  data_store_->joint_info.GetTorqueLimit(tau_limit_);
  const Eigen::VectorXd soft_torque_limit = common::ConcatenateVectors(param_->soft_torque_limit);
  // add soft torque limit
  tau_limit_ = tau_limit_.cwiseProduct(soft_torque_limit);

  time_ = 0.0;
  global_phase_ = 0.0;
  is_first_time_ = true;
  command_.setZero();
  last_remote_command_ = *data_store_->gamepad_info.Get();
  imu_install_bias_ = param_->imu_install_bias.value_or(Eigen::Vector3d::Zero());

  // Initializes some low pass filters
  lpf_command_->Reset();

  // Sets all motor command to zero
  GetMutableOutput().Reset();

  data_store_->joint_info.GetState(data::JointInfoType::kPosition, q_real_);
  initial_joint_q_ = q_real_;

  desired_yaw_ = 0.0;
  current_yaw_ = 0.0;
  last_yaw_ = 0.0;

  speed_mode_ = false;

  return true;
}

void RlWalkingExampleRunner::Run() {
  // Gets current joint pos and vel
  data_store_->joint_info.GetState(data::JointInfoType::kPosition, q_real_);
  data_store_->joint_info.GetState(data::JointInfoType::kVelocity, qd_real_);
  UpdateIMUInstalBias();
  UpdateRemoteCommandBias();
  UpdateRemoteCommand();
  if (param_->use_gait) {
    ManageGaitTransition();
    UpdateGaitParameters();
  }
  last_remote_command_ = *data_store_->gamepad_info.Get();
  CalculateObservation();
  CalculateMotorCommand();

  // Call appropriate head shake function based on mode
  if (head_shake_mode_ == HeadShakeMode::kOnce) {
    ShakeHeadOnce();
  } else if (head_shake_mode_ == HeadShakeMode::kRandom || returning_to_zero_) {
    ShakeHeadRandom();
  }

  SendMotorCommand();
  time_ += param_->control_dt;

  ManageRunnerTransition();
}

TransitionState RlWalkingExampleRunner::TryExit() {
  auto joint_override_command = data_store_->joint_override_command.Get();
  if (joint_override_command->IsEnable() == true) {
    Run();
    return TransitionState::kTrying;
  } else {
    return TransitionState::kCompleted;
  }
}

bool RlWalkingExampleRunner::Exit() { return true; }

void RlWalkingExampleRunner::End() {}

void RlWalkingExampleRunner::UpdateIMUInstalBias() {
  bool imu_flag = false;

  if (data_store_->gamepad_info.Get()->START && data_store_->gamepad_info.Get()->CROSS_Y == -1 &&
      data_store_->gamepad_info.Get()->CROSS_Y < last_remote_command_.CROSS_Y) {
    imu_install_bias_.x() += param_->imu_install_delta_bias;
    imu_flag = true;
  }
  if (data_store_->gamepad_info.Get()->START && data_store_->gamepad_info.Get()->CROSS_Y == 1 &&
      data_store_->gamepad_info.Get()->CROSS_Y > last_remote_command_.CROSS_Y) {
    imu_install_bias_.x() -= param_->imu_install_delta_bias;
    imu_flag = true;
  }
  if (data_store_->gamepad_info.Get()->START && data_store_->gamepad_info.Get()->CROSS_X == 1 &&
      data_store_->gamepad_info.Get()->CROSS_X > last_remote_command_.CROSS_X) {
    imu_install_bias_.y() += param_->imu_install_delta_bias;
    imu_flag = true;
  }
  if (data_store_->gamepad_info.Get()->START && data_store_->gamepad_info.Get()->CROSS_X == -1 &&
      data_store_->gamepad_info.Get()->CROSS_X < last_remote_command_.CROSS_X) {
    imu_install_bias_.y() -= param_->imu_install_delta_bias;
    imu_flag = true;
  }
  if (imu_flag) {
    LOG(INFO) << "imu_install_bias: " << imu_install_bias_;
  }
}

void RlWalkingExampleRunner::UpdateRemoteCommandBias() {
  bool command_flag = false;
  if (data_store_->gamepad_info.Get()->BACK && data_store_->gamepad_info.Get()->CROSS_Y == -1 &&
      data_store_->gamepad_info.Get()->CROSS_Y < last_remote_command_.CROSS_Y) {
    command_bias_.y() -= param_->linear_vel_delta_bias;
    command_flag = true;
  }
  if (data_store_->gamepad_info.Get()->BACK && data_store_->gamepad_info.Get()->CROSS_Y == 1 &&
      data_store_->gamepad_info.Get()->CROSS_Y > last_remote_command_.CROSS_Y) {
    command_bias_.y() += param_->linear_vel_delta_bias;
    command_flag = true;
  }
  if (data_store_->gamepad_info.Get()->BACK && data_store_->gamepad_info.Get()->CROSS_X == 1 &&
      data_store_->gamepad_info.Get()->CROSS_X > last_remote_command_.CROSS_X) {
    command_bias_.x() += param_->linear_vel_delta_bias;
    command_flag = true;
  }
  if (data_store_->gamepad_info.Get()->BACK && data_store_->gamepad_info.Get()->CROSS_X == -1 &&
      data_store_->gamepad_info.Get()->CROSS_X < last_remote_command_.CROSS_X) {
    command_bias_.x() -= param_->linear_vel_delta_bias;
    command_flag = true;
  }
  if (command_flag) {
    LOG(INFO) << "command_bias: " << command_bias_;
  }
}

void RlWalkingExampleRunner::UpdateRemoteCommand() {
  const auto& gamepad = data_store_->gamepad_info.Get();

  // Handle speed mode switching with CROSS_X up/down buttons
  double speed_factor = 1.0;
  if (param_->speed_mode.value_or(false)) {
    if (gamepad->CROSS_X == 1 && gamepad->CROSS_X != last_remote_command_.CROSS_X) {
      speed_mode_ = true;  // High speed
      LOG(INFO) << "Speed mode: HIGH";
    } else if (gamepad->CROSS_X == -1 && gamepad->CROSS_X != last_remote_command_.CROSS_X) {
      speed_mode_ = false;  // Low speed
      LOG(INFO) << "Speed mode: LOW";
    }

    // Apply speed scaling factor
    speed_factor = speed_mode_ ? 1.0 : param_->speed_ratio.value_or(0.5);  // High speed: 1.0, Low speed: 0.5
  }

  command_.x() = (gamepad->LeftStick_X >= 0) ? gamepad->LeftStick_X * param_->command_scale_pos.x() * speed_factor
                                             : gamepad->LeftStick_X * param_->command_scale_neg.x();
  command_.y() = (gamepad->LeftStick_Y >= 0) ? gamepad->LeftStick_Y * param_->command_scale_pos.y()
                                             : gamepad->LeftStick_Y * param_->command_scale_neg.y();
  command_.z() = (gamepad->RightStick_Y >= 0) ? gamepad->RightStick_Y * param_->command_scale_pos.z()
                                              : gamepad->RightStick_Y * param_->command_scale_neg.z();
  if (param_->enable_remote_command_lpf) {
    command_ = lpf_command_->Update(command_);
  }

  command_.z() = (gamepad->RightStick_Y >= 0) ? gamepad->RightStick_Y * param_->command_scale_pos.z()
                                              : gamepad->RightStick_Y * param_->command_scale_neg.z();

  // clear low forward speed
  Eigen::Vector3d left_ankle_pos, right_ankle_pos;
  Eigen::VectorXd q_joint_tmp = Eigen::VectorXd::Zero(fixed_base_model_interface_->GetJointDimensions());
  q_joint_tmp = q_real_;
  fixed_base_model_interface_->SetQ(q_joint_tmp);
  fixed_base_model_interface_->UpdateKinematics();
  fixed_base_model_interface_->GetFramePosition(param_->ankle_roll_link_names[0], left_ankle_pos);
  fixed_base_model_interface_->GetFramePosition(param_->ankle_roll_link_names[1], right_ankle_pos);
  if (std::abs(command_.x()) < param_->command_deadband.value_or(0.3)) {
    if (std::abs(left_ankle_pos.x() - right_ankle_pos.x()) < param_->stop_foot_pos_x_threshold.value_or(10.0)) {
      command_.x() = 0.;
    } else {
      command_.x() = math::Sign(command_.x()) * param_->command_deadband.value_or(0.3);
    }
  }

  command_ += command_bias_ + param_->command_bias;

  if (command_.norm() < param_->command_deadband.value_or(0.3)) {
    desired_yaw_ = current_yaw_;
  } else {
    desired_yaw_ = desired_yaw_ + command_.z() * param_->control_dt;
  }


  // A + CROSS_Y: Toggle ShakeHeadOnce mode (sinusoidal motion)
  // CROSS_Y = 1 (left): Left turn
  // CROSS_Y = -1 (right): Right turn
  if (data_store_->gamepad_info.Get()->A && data_store_->gamepad_info.Get()->CROSS_Y != 0 &&
      data_store_->gamepad_info.Get()->CROSS_Y != last_remote_command_.CROSS_Y) {
    if (param_->head_shake_enable.value_or(false)) {
      if (head_shake_mode_ == HeadShakeMode::kOnce) {
        // Exit ShakeHeadOnce mode
        head_shake_mode_ = HeadShakeMode::kNone;
        run_head_shake_ = false;
        LOG(INFO) << "Head shake once mode stopped.";
      } else if (head_shake_mode_ == HeadShakeMode::kNone) {
        // Enter ShakeHeadOnce mode
        head_shake_mode_ = HeadShakeMode::kOnce;
        run_head_shake_ = true;
        run_start_time_ = time_;
        if (data_store_->gamepad_info.Get()->CROSS_Y == 1) {
          flip_direction_ = true;  // Turn left
          LOG(INFO) << "Head shake once started - turning left.";
        } else if (data_store_->gamepad_info.Get()->CROSS_Y == -1) {
          flip_direction_ = false;  // Turn right
          LOG(INFO) << "Head shake once started - turning right.";
        }
      }
    }
  }

  // B + CROSS_Y: Control ShakeHeadRandom mode (random target positions)
  // CROSS_Y = 1 (left): Enter random mode
  // CROSS_Y = -1 (right): Exit random mode and return to zero
  if (data_store_->gamepad_info.Get()->B && data_store_->gamepad_info.Get()->CROSS_Y != 0 &&
      data_store_->gamepad_info.Get()->CROSS_Y != last_remote_command_.CROSS_Y) {
    if (param_->head_shake_enable.value_or(false)) {
      if (data_store_->gamepad_info.Get()->CROSS_Y == 1) {
        // CROSS_Y = 1: Enter ShakeHeadRandom mode
        if (head_shake_mode_ == HeadShakeMode::kNone) {
          head_shake_mode_ = HeadShakeMode::kRandom;
          run_head_shake_ = true;
          first_head_shake_ = true;
          run_start_time_ = time_;
          segment_start_time_ = time_;
          returning_to_zero_ = false;
          LOG(INFO) << "Head random mode started.";
        }
      } else if (data_store_->gamepad_info.Get()->CROSS_Y == -1) {
        // CROSS_Y = -1: Exit ShakeHeadRandom mode - start returning to zero
        if (head_shake_mode_ == HeadShakeMode::kRandom) {
          returning_to_zero_ = true;
          segment_start_time_ = time_;
          head_random_start_ = q_des_(head_joint_idx_);
          head_random_target_ = default_joint_q_(head_joint_idx_);
          LOG(INFO) << "Head random mode stopping - returning to zero position.";
        }
      }
    }
  }
}

void RlWalkingExampleRunner::ManageGaitTransition() {
  // Gets current gait command.
  command_gait_ = command_.norm() > param_->stop_vel_threshold ? GaitType::kWalk : GaitType::kStand;

  // Checks if the robot is near lost balance
  Eigen::Vector3d left_ankle_pos, right_ankle_pos;
  Eigen::VectorXd q_joint_tmp = Eigen::VectorXd::Zero(fixed_base_model_interface_->GetJointDimensions());
  q_joint_tmp = q_real_;
  fixed_base_model_interface_->SetQ(q_joint_tmp);
  fixed_base_model_interface_->UpdateKinematics();
  fixed_base_model_interface_->GetFramePosition(param_->ankle_roll_link_names[0], left_ankle_pos);
  fixed_base_model_interface_->GetFramePosition(param_->ankle_roll_link_names[1], right_ankle_pos);
  bool near_lost_balance_in_y = std::abs(left_ankle_pos.y()) < param_->lost_balance_in_y_threshold.x() ||
                                std::abs(left_ankle_pos.y()) > param_->lost_balance_in_y_threshold.y() ||
                                std::abs(right_ankle_pos.y()) < param_->lost_balance_in_y_threshold.x() ||
                                std::abs(right_ankle_pos.y()) > param_->lost_balance_in_y_threshold.y();
  bool near_lost_balance_in_x = mlp_net_action_(4) < param_->lost_balance_in_x_threshold.x() ||
                                mlp_net_action_(4) > param_->lost_balance_in_x_threshold.y() ||
                                mlp_net_action_(10) < param_->lost_balance_in_x_threshold.x() ||
                                mlp_net_action_(10) > param_->lost_balance_in_x_threshold.y();
  bool near_lost_balance = near_lost_balance_in_x || near_lost_balance_in_y;

  // Checks if the robot is just lifting
  bool left_just_lifting = gait_global_phase_ > param_->stop_left_phase_condition.x() &&
                           gait_global_phase_ < param_->stop_left_phase_condition.y();
  bool right_just_lifting = gait_global_phase_ > param_->stop_right_phase_condition.x() &&
                            gait_global_phase_ < param_->stop_right_phase_condition.y();
  bool just_lifting = left_just_lifting || right_just_lifting;
  // Manages gait transition
  if (command_gait_ == GaitType::kWalk) {
    current_gait_ = GaitType::kWalk;
    stop_elapsed_time_ = 0.0;
    return;
  }

  if (current_gait_ == GaitType::kStand && command_gait_ == GaitType::kStand) {
    if (near_lost_balance) {
      current_gait_ = GaitType::kWalk;
      stop_elapsed_time_ = 0.0;
      if (near_lost_balance_in_y) {
        gait_global_phase_ = data_store_->imu_info.Get()->rpy.x() < 0 ? 0.5 : 0.0;
      }
    }
    return;
  }

  if (current_gait_ == GaitType::kWalk && command_gait_ == GaitType::kStand) {
    stop_elapsed_time_ += param_->control_dt;
    if (!param_->tune_stop && just_lifting && !near_lost_balance && stop_elapsed_time_ > param_->stop_least_duration) {
      current_gait_ = GaitType::kStand;
      stop_elapsed_time_ = 0.0;
      return;
    }
  }
}

void RlWalkingExampleRunner::UpdateGaitParameters() {
  // Sets gait parameters
  if (current_gait_ == GaitType::kWalk) {
    if (param_->customized_gait_velocity) {
      Eigen::Vector4d& cgv = param_->customized_gait_velocity.value();      // [-1.0, 1.0, 2.0, 4.0]
      Eigen::Vector4d& cgf = param_->customized_gait_frequency.value();     // [1.0, 1.0, 2.0, 2.0]
      Eigen::Vector4d& cgsd = param_->customized_gait_stance_duty.value();  // [0.5, 0.5, 0.3, 0.3]
      // Eigen::Vector4d& cgsth = param_->customized_gait_stance_height.value();
      // Eigen::Vector4d& cgswh = param_->customized_gait_swing_height.value();

      double ratio = (command_.x() - cgv(1)) * (cgv(2) - cgv(1));
      ratio = std::clamp(ratio, 0.0, 1.0);
      gait_frequency_ = ratio * (cgf(2) - cgf(1)) + cgf(1);
      gait_stand_duty_cycle_ = ratio * (cgsd(2) - cgsd(1)) + cgsd(1);

      ratio = (command_.x() - cgv(1)) * (cgv(3) - cgv(1));
      ratio = std::clamp(ratio, 0.0, 1.0);
      // gait_stance_height_ = ratio * (cgsth(3) - cgsth(1)) + cgsth(1);
      // gait_swing_height_ = ratio * (cgswh(3) - cgswh(1)) + cgswh(1);
    } else {
      gait_frequency_ = param_->gait_base_frequency;
      gait_stand_duty_cycle_ = param_->gait_base_stance_duty;
      // gait_swing_height_ = param_->gait_base_swing_height;
      // gait_stance_height_ = param_->gait_base_stance_height;
    }
    gait_phase_offset_ << 0.0, 0.5;
    gait_global_phase_ += param_->control_dt * gait_frequency_;
    gait_global_phase_ -= int(gait_global_phase_);
  } else if (current_gait_ == GaitType::kStand) {
    gait_frequency_ = 0.0;
    gait_stand_duty_cycle_ = 1.0;
    gait_phase_offset_ << 0.0, 0.0;
    // gait_swing_height_ = 0.0;
    // if (param_->customized_gait_stance_height) {
    //   gait_stance_height_ = param_->customized_gait_stance_height.value()(0);
    // } else {
    //   gait_stance_height_ = param_->gait_base_stance_height;
    // }
    command_.setZero();
    gait_global_phase_ = 0.0;
  }
  Eigen::Vector2d foot_phase;
  foot_phase.array() = gait_global_phase_ + gait_phase_offset_.array();
  gait_foot_signal_ << std::sin(math::k2Pi * foot_phase[0]), std::cos(math::k2Pi * foot_phase[0]),
      std::sin(math::k2Pi * foot_phase[1]), std::cos(math::k2Pi * foot_phase[1]);
}

void RlWalkingExampleRunner::CalculateObservation() {
  // Calculates phase info
  global_phase_ += param_->control_dt / param_->cycle_time;
  global_phase_ -= int(global_phase_);

  Eigen::Vector2d clock_signal(std::sin(math::k2Pi * global_phase_), std::cos(math::k2Pi * global_phase_));

  // Calculates base state
  Eigen::Matrix3d R_install = math::RotationMatrixd(math::RollPitchYawd(imu_install_bias_)).matrix();
  Eigen::Matrix3d R_local = math::RotationMatrixd(data_store_->imu_info.Get()->quaternion).matrix();
  Eigen::Matrix3d R_real = R_local * R_install.transpose();
  Eigen::Vector3d w_real = R_real.transpose() * R_local * data_store_->imu_info.Get()->angular_velocity;
  Eigen::Vector3d euler_xyz = math::RollPitchYawd(math::RotationMatrixd(R_real)).vector();
  Eigen::Vector3d projected_gravity_real = -R_real.transpose() * Eigen::Vector3d::UnitZ();

  float delta_yaw = euler_xyz.z() - last_yaw_;
  if (delta_yaw > math::kPi) {
    delta_yaw -= math::k2Pi;
  } else if (delta_yaw < -math::kPi) {
    delta_yaw += math::k2Pi;
  }
  current_yaw_ += delta_yaw;
  last_yaw_ = euler_xyz.z();
  current_dyaw_ = w_real.z();

  // Stacks the observation
  Eigen::VectorXd mlp_net_observation_single = Eigen::VectorXd::Zero(param_->num_observations);
  mlp_net_observation_single <<
      (q_real_ - default_joint_q_)(active_joint_idx_),  //  joint position - joint default position: kDoFs
      qd_real_(active_joint_idx_),                      //  joint velocity: kDoFs
      mlp_net_action_,                                  //  last joint action: kDoFs
      w_real,                                           //  base angular velocity w.r.t base frame: 3
      projected_gravity_real;                           //  base euler angle rpy w.r.t base frame: 3

  // Scales and clips the observation
  mlp_net_observation_single.array() *= param_->observation_scale.array();
  mlp_net_observation_single =
      mlp_net_observation_single.cwiseMax(-param_->observation_clip).cwiseMin(param_->observation_clip);

  // Updates the observation buffer
  if (is_first_time_) {
    is_first_time_ = false;
    mlp_net_observation_.setZero(param_->num_observations, param_->num_include_obs_steps);
    mlp_net_action_.setZero(param_->num_actions);
    mlp_net_observation_.colwise() = mlp_net_observation_single;
  } else {
    mlp_net_observation_.leftCols(param_->num_include_obs_steps - 1) =
        mlp_net_observation_.rightCols(param_->num_include_obs_steps - 1);
    mlp_net_observation_.rightCols(1) = mlp_net_observation_single;
  }

}

void RlWalkingExampleRunner::CalculateMotorCommand() {
  Eigen::Vector3d command_obs = command_.cwiseProduct(param_->command_obs_scale);
  if (param_->use_pd_control_for_yaw.value_or(false)) {
    double dyaw_cmd = param_->yaw_control_kp.value_or(0.5) * (desired_yaw_ - current_yaw_);
    dyaw_cmd = std::clamp(dyaw_cmd, param_->pd_yaw_cmd_limit.value_or(Eigen::Vector2d(-1.0, 1.0)).x(),
                          param_->pd_yaw_cmd_limit.value_or(Eigen::Vector2d(-1.0, 1.0)).y());
    command_obs.z() = dyaw_cmd * param_->command_obs_scale.z();
  }
  Eigen::VectorXd obs;
  if (param_->use_gait) {
    obs = Eigen::VectorXd::Zero(param_->num_observations * param_->num_include_obs_steps + 3 +
                                param_->num_gait_command_obs);
    obs.head(param_->num_observations * param_->num_include_obs_steps) =
        Eigen::Map<Eigen::VectorXd>(mlp_net_observation_.transpose().data(), mlp_net_observation_.size());
    obs.tail(3 + param_->num_gait_command_obs) << command_obs, gait_foot_signal_, gait_frequency_,
        gait_stand_duty_cycle_, gait_phase_offset_;
  } else {
    obs = Eigen::VectorXd::Zero(param_->num_observations * param_->num_include_obs_steps + 3);
    obs.head(param_->num_observations * param_->num_include_obs_steps) =
        Eigen::Map<Eigen::VectorXd>(mlp_net_observation_.transpose().data(), mlp_net_observation_.size());
    obs.tail(3) = command_obs;
  }
  mlp_net_action_ = (mlp_net_->Inference(obs.cast<float>())).cast<double>();
  mlp_net_action_ = mlp_net_action_.cwiseMax(-param_->action_clip).cwiseMin(param_->action_clip);
  q_des_ = default_joint_q_;
  q_des_(active_joint_idx_) += mlp_net_action_.cwiseProduct(action_scale_);
  if (param_->transition_time.has_value() && param_->transition_time.value() > 0 &&
      time_ < param_->transition_time.value()) {
    float ratio = time_ / param_->transition_time.value();
    q_des_ = ratio * q_des_ + (1.0f - ratio) * initial_joint_q_;
  }

  // add PD torque limit (only for active joints)
  const double kp_min_threshold = 1e-3;
  const Eigen::VectorXd joint_kp_safe = joint_kp_.cwiseMax(kp_min_threshold);
  q_des_lb_ = (-tau_limit_ + joint_kd_.cwiseProduct(qd_real_)).array() / joint_kp_safe.array();
  q_des_ub_ = (tau_limit_ + joint_kd_.cwiseProduct(qd_real_)).array() / joint_kp_safe.array();
  q_des_lb_ += q_real_;
  q_des_ub_ += q_real_;
  q_des_(active_joint_idx_) =
      q_des_(active_joint_idx_).cwiseMax(q_des_lb_(active_joint_idx_)).cwiseMin(q_des_ub_(active_joint_idx_));
}

void RlWalkingExampleRunner::SendMotorCommand() {
  qd_des_ = Eigen::VectorXd::Zero(model_param_->num_total_joints);
  tau_ff_des_ = Eigen::VectorXd::Zero(model_param_->num_total_joints);
  // 缓存输出
  GetMutableOutput().SetCommand(q_des_, qd_des_, joint_kp_, joint_kd_, tau_ff_des_);
}

void RlWalkingExampleRunner::ManageRunnerTransition() {
  if (param_->enable_auto_fall_transition && param_->enable_auto_fall_transition.value()) {
    const Eigen::Vector3d& imu_rpy = data_store_->imu_info.Get()->rpy;
    if (std::abs(imu_rpy.y()) > param_->fall_threshold.value()) {
      LOG_EVERY_T(WARNING, 1) << "Fall detected, trying to exit...";
      LOG_EVERY_T(WARNING, 1) << "imu_rpy: " << imu_rpy.transpose();
      SetRunnerState(RunnerState::kTryExit);
    }
  }
}

void RlWalkingExampleRunner::ShakeHeadOnce() {
  if (!param_->head_shake_enable.value_or(false)) {
    return;
  }

  if (run_head_shake_) {
    const Eigen::VectorXd head_shake_amplitude =
        param_->head_shake_amplitude.value_or(Eigen::VectorXd::Constant(head_joint_idx_.size(), 0.1));
    const Eigen::VectorXi head_shake_flip =
        param_->head_shake_flip.value_or(Eigen::VectorXi::Constant(head_joint_idx_.size(), 1));
    double shake_time = time_ - run_start_time_;
    const float head_shake_duration = 1.0 / param_->head_shake_frequency.value_or(1.0);
    if (shake_time > head_shake_duration / 2.0) {
      run_head_shake_ = false;
      q_des_(head_joint_idx_) = default_joint_q_(head_joint_idx_);
      return;
    }
    for (int i = 0; i < head_joint_idx_.size(); ++i) {
      int joint_id = head_joint_idx_(i);
      double neutral_q = default_joint_q_(joint_id);
      double direction = 1.0;
      if (head_shake_flip(i) > 0) {
        direction = flip_direction_ ? 1.0 : -1.0;
      } else {
        direction = 1.0;
      }
      q_des_(joint_id) = neutral_q + direction * head_shake_amplitude(i) *
                                         std::sin(2.0 * math::kPi / head_shake_duration * shake_time);
    }
  }
}

void RlWalkingExampleRunner::ShakeHeadRandom() {
  if (!param_->head_shake_enable.value_or(false)) {
    return;
  }

  const float head_shake_duration = param_->head_random_duration.value_or(1.0);
  double shake_time_elpsed = time_ - segment_start_time_;

  // Handle returning to zero position
  if (returning_to_zero_) {
    if (shake_time_elpsed >= head_shake_duration) {
      // Finished returning to zero
      returning_to_zero_ = false;
      head_shake_mode_ = HeadShakeMode::kNone;
      run_head_shake_ = false;
      q_des_(head_joint_idx_) = default_joint_q_(head_joint_idx_);
      LOG(INFO) << "Head returned to zero position. Random mode stopped.";
      return;
    }

    // Interpolate to zero position
    double segment_elapsed = shake_time_elpsed;
    Eigen::VectorXd q_head_des, v_head_des;
    math::CubicInterpolate(head_random_start_, head_random_target_, head_shake_duration, segment_elapsed, q_head_des,
                           v_head_des);
    q_des_(head_joint_idx_) = q_head_des;
    return;
  }

  // Normal random mode operation
  if (!run_head_shake_) {
    return;
  }

  // Initialize random target on first call or when reaching target
  if (first_head_shake_ || shake_time_elpsed >= head_shake_duration) {
    first_head_shake_ = false;

    // reset time
    segment_start_time_ = time_;
    shake_time_elpsed = 0.0;

    // Update start position for next interpolation
    head_random_start_ = head_random_target_;

    // Check if predefined target positions are available
    if (param_->head_random_target_positions.has_value() && !param_->head_random_target_positions.value().empty()) {
      // Randomly sample from predefined target positions
      const auto& target_positions = param_->head_random_target_positions.value();
      int random_index = rng_->sample();
      const Eigen::VectorXd& selected_target = target_positions[random_index];

      // Copy selected target to head_random_target_
      if (selected_target.size() == head_joint_idx_.size()) {
        head_random_target_ = selected_target;
        // LOG(INFO) << "Last target: " << head_random_start_.transpose();
        // LOG(INFO) << "Random index: " << random_index << " Set new target: " << head_random_target_.transpose();
      } else {
        LOG(WARNING) << "Head shake target position size mismatch. Expected " << head_joint_idx_.size() << ", got "
                     << selected_target.size();
        // Fallback to default position
        for (int i = 0; i < head_joint_idx_.size(); ++i) {
          head_random_target_(i) = default_joint_q_(head_joint_idx_(i));
        }
      }
    } else {
      // Fallback to default position
      for (int i = 0; i < head_joint_idx_.size(); ++i) {
        head_random_target_(i) = default_joint_q_(head_joint_idx_(i));
      }
    }
  }

  // Calculate progress within current segment
  double segment_elapsed = shake_time_elpsed;

  // Clamp phase to [0, 1]
  double phase = std::clamp(segment_elapsed / head_shake_duration, 0.0, 1.0);

  // Use CubicInterpolate for smooth interpolation between targets
  Eigen::VectorXd q_head_des, v_head_des;
  math::CubicInterpolate(head_random_start_, head_random_target_, head_shake_duration, segment_elapsed, q_head_des,
                         v_head_des);
  q_des_(head_joint_idx_) = q_head_des;
}
}  // namespace runner
