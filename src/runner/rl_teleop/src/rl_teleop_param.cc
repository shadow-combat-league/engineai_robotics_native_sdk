#include "rl_teleop/rl_teleop_param.h"

namespace data {

RlTeleopParam::RlTeleopParam(std::string_view tag) : BasicParam(tag) {
  LOAD_PARAM(policy_path);
  LOAD_PARAM(num_actions);
  LOAD_PARAM(action_clip);
  LOAD_PARAM(transition_time);

  LOAD_PARAM(obs_joint_names);
  LOAD_PARAM(action_joint_names);
  LOAD_PARAM(motion_driven_joint_names);
  LOAD_PARAM(action_scale);

  joints = common::ScopedParameterGetter<std::map<std::string, TeleopJointConfig>>::Get(scope_, "joints");

  LOAD_PARAM(use_anchor_pos);
  LOAD_PARAM_DEFAULT(anchor_pos_source, std::string("none"));
  LOAD_PARAM_DEFAULT(anchor_vel_tau, 2.0);
  if (anchor_pos_source != "none" && anchor_pos_source != "estimator" &&
      anchor_pos_source != "velocity") {
    throw std::runtime_error(
        "rl_teleop: anchor_pos_source must be 'none', 'estimator', or 'velocity'");
  }
  LOAD_PARAM_DEFAULT(imu_ang_vel_world, true);
  LOAD_PARAM_DEFAULT(qd_zero_joint_names, std::vector<std::string>{});
  LOAD_PARAM_DEFAULT(residual_joint_names, std::vector<std::string>{});
  LOAD_PARAM_DEFAULT(lookahead_offsets_s, std::vector<double>{});
  LOAD_PARAM_DEFAULT(ref_jvel_clip, 12.0);
  LOAD_PARAM_DEFAULT(ref_jvel_alpha, 0.3);
  LOAD_PARAM_DEFAULT(action_lpf_alpha, 1.0);
  LOAD_PARAM_DEFAULT(ref_ang_slew, 12.0);
  LOAD_PARAM_DEFAULT(action_sanity_limit, 20.0);

  LOAD_PARAM_DEFAULT(builtin_stand_reference, true);
  LOAD_PARAM(udp_port);
  LOAD_PARAM_DEFAULT(ref_stale_timeout, 0.5);

  LOAD_PARAM(torque_limit);
  LOAD_PARAM_DEFAULT(max_lower_body_torque, 500.0);
  LOAD_PARAM_DEFAULT(lower_body_joint_count, 12);

  // Sanity checks: fail loudly at load time, not mid-motion
  if (action_lpf_alpha <= 0.0 || action_lpf_alpha > 1.0) {
    throw std::runtime_error("rl_teleop: action_lpf_alpha must be in (0, 1]");
  }
  if (static_cast<int>(action_joint_names.size() + residual_joint_names.size()) != num_actions) {
    throw std::runtime_error(
        "rl_teleop: action_joint_names + residual_joint_names size != num_actions");
  }
  if (static_cast<int>(action_scale.size()) != num_actions) {
    throw std::runtime_error("rl_teleop: action_scale size != num_actions");
  }
  for (const auto& name : obs_joint_names) {
    if (joints.find(name) == joints.end()) {
      throw std::runtime_error("rl_teleop: joint missing from joints map: " + name);
    }
  }
}

}  // namespace data
