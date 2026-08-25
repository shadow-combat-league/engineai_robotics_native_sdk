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
  double q_min = -6.5;      // commanded-target clamp (joint range)
  double q_max = 6.5;
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
  // Source for the anchor_pos_b observation (gen-two policies):
  //   "none"      - feed zeros (v1 behavior; policy degrades to legacy-style
  //                 following with no distance feedback)
  //   "estimator" - base_state_in_world displacement vs the streamed ref_pos.
  //                 MEASURED HARMFUL (2026-08-25): the estimator's integrated
  //                 position under-reports translation -> phantom lag error.
  //   "velocity"  - leaky integral of (ref velocity - estimator velocity) in
  //                 the entry-yaw common frame, ~2 s time constant, clipped
  //                 +-1 m. Needs only short-horizon velocity quality, not
  //                 consistent odometry: drift leaks away, a sustained lag
  //                 saturates near the clip like training values.
  std::string anchor_pos_source = "none";
  double anchor_vel_tau = 2.0;  // leak time constant (s) for "velocity" mode
  // Angular-velocity frame of imu_info: the SIM publishes world-frame
  // (frameangvel of the imu site); real IMU gyros report body-frame.
  // true = rotate world->base (sim); false = pass through (hardware).
  bool imu_ang_vel_world = true;
  // Joints whose measured velocity is zeroed in the obs (vendor configs mask
  // ankle velocities on this stack — the estimates are noisy and destabilize
  // velocity-consuming policies)
  std::vector<std::string> qd_zero_joint_names;
  double ref_jvel_clip = 12.0;     // rad/s clip on reference joint velocity
  double ref_jvel_alpha = 0.3;     // EMA new-sample weight for ref jvel

  // Slew limit on the reference BASE ORIENTATION rate (rad/s). The clip can
  // command spins beyond the plant's yaw-rate ceiling (walking02 pirouette:
  // 4.05 rad/s commanded vs ~2.4-2.6 achievable) — the anchor error then
  // grows uncloseably and the robot falls at the spin exit. Limiting the
  // reference to just under the ceiling lets the robot turn at its own pace
  // with a small, closable error (same principle as the publisher's 3 m/s
  // root-position slew). 12.0 = effectively off; ~2.2 recommended.
  double ref_ang_slew = 12.0;

  // One-pole low-pass on the commanded joint targets (1.0 = off). Training
  // uses implicit PhysX drives that filter 25 Hz action chatter for free;
  // explicit PD (sim + hardware) tracks it faithfully -> visible vibration
  // and audible buzz. 0.5 measured in the MuJoCo rig: ~30% less target
  // jitter, kills the 25 Hz band, 0 falls on walking02 at 40 ms staleness
  // (0.35 over-lags and destabilizes — do not go below ~0.4).
  // last_action in the obs stays RAW, matching the training convention.
  double action_lpf_alpha = 1.0;

  // Start executing the policy immediately with a built-in standing
  // reference (default pose, upright, zero velocity) instead of holding the
  // entry pose until the first UDP packet. The stream takes over on arrival.
  bool builtin_stand_reference = true;

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
    if (node["q_min"]) v.q_min = node["q_min"].as<double>();
    if (node["q_max"]) v.q_max = node["q_max"].as<double>();
    return true;
  }
};
}  // namespace YAML
