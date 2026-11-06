#pragma once

// SuperMimic live reference: turns the irregular terminator-teleop UDP pose stream into the policy's goal
// (reference frames k+1 .. k+10 with velocities). C++ port of
// SCL/Experiments/t800_port/sdk_sim/udp_reference.py (validated there against the training MotionLibrary).
//
// The reference is played `delay_s` (0.2 s) behind the newest packet, so the 10 future frames (20..200 ms) are
// real. Packets are timestamped on arrival, exact repeats of the previous pose (the publisher re-sending a stalled
// PICO pose) are dropped, the window is resampled on a uniform 50 Hz grid (linear joints/position, slerp
// orientation), smoothed with a CENTRED Gaussian (scipy gaussian_filter1d, mode "nearest", truncate 4) and
// differentiated with central differences (numpy.gradient). Features per frame (SuperMimic goal_features):
//   [q - q_default (nj), qd (nj), gravity in body (3), angular vel in body (3), linear vel in heading frame (3),
//    root height (1)]
// No data / stale stream -> blend (smoothstep) to a standing reference; data back -> blend in. On session start
// the standing reference itself moves from the entry joint pose to the default pose over entry_blend_s.
// Height auto-zero: the first calib_s of live data maps the stream's root height onto stand_height.
//
// Eigen only (no SDK dependency) so it can be unit-tested on the host.

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <deque>
#include <vector>

namespace rl_supermimic {

struct ReferenceConfig {
  int num_joints = 25;
  int future_frames = 10;
  double dt = 0.02;
  double delay_s = 0.2;
  double sigma_frames = 1.5;
  double stale_s = 0.5;
  double blend_in_s = 1.5;
  double blend_out_s = 1.0;
  double calib_s = 1.0;
  double stand_height = 1.0165;
  double entry_blend_s = 2.0;
  double hold_anchor_s = 0.3;  // repeats longer than this = a real hold (anchored), shorter = stall (skipped)
};

struct ReferenceInfo {
  Eigen::VectorXd q_current;  // smoothed reference joint pose at the current (delayed) time
  double height = 0.0;
  double w_live = 0.0;        // 0 = standing reference, 1 = live stream
  double z_offset = 0.0;
  bool z_calibrated = false;
  int samples = 0;
};

class ReferenceBuffer {
 public:
  ReferenceBuffer(const ReferenceConfig& cfg, const Eigen::VectorXd& q_default);

  // New session (mode entry): forget the stream, restart the standing reference from q_entry.
  void Reset(double t_now, const Eigen::VectorXd& q_entry);

  // One received packet (joint order = policy order). Returns false if it was an exact repeat (dropped).
  bool Push(double t_arrival, const Eigen::VectorXd& q, const Eigen::Quaterniond& quat_wxyz,
            const Eigen::Vector3d& pos);

  // Goal (future_frames * (2*nj + 10)) for the current wall time.
  Eigen::VectorXf Goal(double t_now, ReferenceInfo* info = nullptr);

  int FeatureDim() const { return 2 * cfg_.num_joints + 10; }
  double LastSampleTime() const { return samples_.empty() ? -1e9 : samples_.back().t; }

 private:
  struct Sample {
    double t;
    Eigen::VectorXd q;
    Eigen::Quaterniond quat;
    Eigen::Vector3d pos;
  };
  Eigen::VectorXd StandJoints(double t) const;
  void Smooth(Eigen::MatrixXd& x) const;  // rows = time

  ReferenceConfig cfg_;
  Eigen::VectorXd q_default_;
  Eigen::VectorXd q_entry_;
  double t_entry_ = 0.0;
  std::deque<Sample> samples_;
  std::vector<double> calib_z_;
  bool z_calibrated_ = false;
  double z_offset_ = 0.0;
  double w_ = 0.0;
  double t_last_w_ = -1.0;
  double last_packet_t_ = -1e9;
  std::vector<double> kernel_;
};

}  // namespace rl_supermimic
