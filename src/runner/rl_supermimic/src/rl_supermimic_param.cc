#include "rl_supermimic/rl_supermimic_param.h"

#include <algorithm>
#include <stdexcept>

namespace data {

RlSuperMimicParam::RlSuperMimicParam(std::string_view tag) : BasicParam(tag) {
  LOAD_PARAM(policy_path);
  LOAD_PARAM(policy_joint_names);
  LOAD_PARAM(packet_joint_names);
  LOAD_PARAM_DEFAULT(obs_history_length, 10);
  LOAD_PARAM_DEFAULT(future_frames, 10);
  joints = common::ScopedParameterGetter<std::map<std::string, SuperMimicJointConfig>>::Get(scope_, "joints");

  LOAD_PARAM_DEFAULT(udp_port, 47801);
  LOAD_PARAM_DEFAULT(ref_delay_s, 0.2);
  LOAD_PARAM_DEFAULT(ref_sigma_frames, 1.5);
  LOAD_PARAM_DEFAULT(ref_stale_s, 0.5);
  LOAD_PARAM_DEFAULT(ref_blend_in_s, 1.5);
  LOAD_PARAM_DEFAULT(ref_blend_out_s, 1.0);
  LOAD_PARAM_DEFAULT(ref_height_calib_s, 1.0);
  LOAD_PARAM_DEFAULT(stand_height, 1.0165);
  LOAD_PARAM_DEFAULT(entry_blend_s, 2.0);

  LOAD_PARAM_DEFAULT(robot_rot_mode, std::string("heading_fixed"));
  LOAD_PARAM_DEFAULT(imu_ang_vel_world, false);

  LOAD_PARAM_DEFAULT(action_sanity_limit, 20.0);
  LOAD_PARAM_DEFAULT(action_clip, 100.0);
  LOAD_PARAM_DEFAULT(torque_limit, true);
  LOAD_PARAM_DEFAULT(clamp_targets_to_joint_range, false);
  LOAD_PARAM_DEFAULT(range_excess_as_torque, true);
  LOAD_PARAM_DEFAULT(range_margin, 0.02);
  LOAD_PARAM_DEFAULT(action_lpf_alpha, 1.0);
  LOAD_PARAM_DEFAULT(qd_from_q_ankles, false);
  LOAD_PARAM_DEFAULT(qd_obs_lpf_alpha, 1.0);
  if (!(qd_obs_lpf_alpha > 0.0 && qd_obs_lpf_alpha <= 1.0))
    throw std::runtime_error("rl_supermimic: qd_obs_lpf_alpha must be in (0, 1]");
  if (!(action_lpf_alpha > 0.0 && action_lpf_alpha <= 1.0))
    throw std::runtime_error("rl_supermimic: action_lpf_alpha must be in (0, 1]");
  LOAD_PARAM_DEFAULT(fall_up_z, 0.5);
  LOAD_PARAM_DEFAULT(fall_ticks, 25);
  LOAD_PARAM_DEFAULT(entry_check, true);
  LOAD_PARAM_DEFAULT(entry_leg_tol, 0.15);
  LOAD_PARAM_DEFAULT(entry_gyro_max, 0.5);
  LOAD_PARAM_DEFAULT(entry_tilt_max_deg, 10.0);
  LOAD_PARAM_DEFAULT(entry_force_refuse, false);

  // Fail loudly at load time, not mid-motion
  if (robot_rot_mode != "heading_fixed" && robot_rot_mode != "legacy" && robot_rot_mode != "absolute")
    throw std::runtime_error("rl_supermimic: robot_rot_mode must be 'heading_fixed', 'legacy' or 'absolute'");
  if (policy_joint_names.size() != packet_joint_names.size())
    throw std::runtime_error("rl_supermimic: policy_joint_names / packet_joint_names size mismatch");
  for (const auto& n : policy_joint_names) {
    if (joints.find(n) == joints.end()) throw std::runtime_error("rl_supermimic: joint missing from joints map: " + n);
    if (std::find(packet_joint_names.begin(), packet_joint_names.end(), n) == packet_joint_names.end())
      throw std::runtime_error("rl_supermimic: joint missing from packet_joint_names: " + n);
  }
  if (obs_history_length < 1 || future_frames < 1)
    throw std::runtime_error("rl_supermimic: obs_history_length / future_frames must be >= 1");
}

}  // namespace data
