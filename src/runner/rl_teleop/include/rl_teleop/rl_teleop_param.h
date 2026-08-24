#pragma once

// Parameters for the rl_teleop runner: live PICO teleoperation with the
// terminator-teleop-system tracking policy (obs layout mirrors
// deploy/sim_teleop.py DeployedPolicy.act()).
//
// Per-joint control values are stored as a NAME-KEYED map and distributed to
// the model's total-joint order at runtime via joint_id_in_total_limb — no
// ordering assumptions anywhere in the config.

#include <glog/logging.h>
#include <Eigen/Dense>
#include <map>
#include <string>
#include <vector>

#include "basic_param/basic_param.h"
#include "parameter/parameter_loader.h"

namespace data {

class TeleopJointConfig {
 public:
  double q_default = 0.0;   // default joint position (rad)
  double kp = 50.0;         // PD stiffness
  double kd = 1.0;          // PD damping
  double tau_max = 100.0;   // per-joint torque cap
};

class RlTeleopParam : public BasicParam {
 public:
  RlTeleopParam(std::string_view tag = "rl_teleop");

  DEFINE_PARAM_SCOPE(scope_);

  // Policy
  std::string policy_path;         // .mnn, relative to config root
  int num_actions = 15;
  float action_clip = 100.0f;
  float transition_time = 2.0f;    // blend-in seconds on mode entry

  // Joint lists (define observation/action ordering; must match the
  // exported deploy_config.yaml of the policy)
  std::vector<std::string> obs_joint_names;           // 25, Isaac obs order
  std::vector<std::string> action_joint_names;        // 15, Isaac action order
  std::vector<std::string> motion_driven_joint_names; // 10, arms from reference
  std::vector<double> action_scale;                   // 15, per action joint

  // Per-joint control values keyed by joint name
  std::map<std::string, TeleopJointConfig> joints;

  // Observation options
  bool use_anchor_pos = true;      // gen-two (127-dim) vs legacy (124-dim)
  // Joints whose measured velocity is zeroed in the obs (vendor configs mask
  // ankle velocities on this stack — the estimates are noisy and destabilize
  // velocity-consuming policies)
  std::vector<std::string> qd_zero_joint_names;
  double ref_jvel_clip = 12.0;     // rad/s clip on reference joint velocity
  double ref_jvel_alpha = 0.3;     // EMA new-sample weight for ref jvel

  // Reference stream (UDP)
  int udp_port = 47800;
  double ref_stale_timeout = 0.5;  // s without packets -> hold last reference

  // Torque limiting
  bool torque_limit = true;
  double max_lower_body_torque = 500.0;
  int lower_body_joint_count = 12;
};

}  // namespace data

namespace YAML {
template <>
struct convert<data::TeleopJointConfig> {
  static bool decode(const Node& node, data::TeleopJointConfig& v) {
    if (node["q_default"]) v.q_default = node["q_default"].as<double>();
    if (node["kp"]) v.kp = node["kp"].as<double>();
    if (node["kd"]) v.kd = node["kd"].as<double>();
    if (node["tau_max"]) v.tau_max = node["tau_max"].as<double>();
    return true;
  }
};
}  // namespace YAML
