#pragma once

// Non-blocking UDP receiver for the live teleop reference stream.
//
// Packet layout (little-endian float32 unless noted):
//   uint32 magic   = 0x54545031 ("TTP1") or 0x54545032 ("TTP2", gen7)
//   uint32 seq
//   float  ref_jpos[25]   Isaac obs order (publisher converts)
//   float  ref_quat[4]    wxyz, reference base orientation (world)
//   float  ref_pos[3]     xyz,  reference anchor position (world, anchored)
//   -- TTP2 only, gen7 reference lookahead --
//   float  lookahead[28*H]  H horizons, each: 25 joint-position deltas from
//                           the CURRENT frame (Isaac order), then dx, dy in
//                           the current reference heading frame, then dyaw.
// Total: v1 = 8 + 32*4 = 136 bytes; v2 = 136 + 28*H*4 (H=2 -> 360 bytes).
//
// The magic is VERSIONED on purpose: a gen<=6 publisher physically cannot
// feed a gen7 runner (or vice versa) without being rejected loudly. Silent
// train/deploy mismatches on observation channels have cost this project
// days — see the ankle-qd mask and the anchor-source drift.

#include <Eigen/Dense>
#include <cstdint>
#include <mutex>

namespace rl_teleop {

struct ReferenceFrame {
  uint32_t seq = 0;
  Eigen::VectorXd jpos;      // 25
  Eigen::Vector4d quat_wxyz; // reference base orientation
  Eigen::Vector3d pos;       // reference anchor position
  Eigen::VectorXd lookahead; // 28*H, empty for v1 publishers
  double recv_time = 0.0;    // monotonic seconds
  bool valid = false;
};

class UdpReferenceReceiver {
 public:
  static constexpr uint32_t kMagic = 0x54545031;    // v1
  static constexpr uint32_t kMagicV2 = 0x54545032;  // v2: + lookahead
  static constexpr int kNumJoints = 25;
  static constexpr size_t kPacketBytes = 8 + (kNumJoints + 4 + 3) * 4;
  static constexpr int kLookaheadPerHorizon = 28;   // 25 joint deltas + dx,dy,dyaw
  static constexpr size_t kMaxPacketBytes = kPacketBytes + kLookaheadPerHorizon * 8 * 4;

  // expected_lookahead: 28*H, or 0 to require v1 packets
  explicit UdpReferenceReceiver(int port, int expected_lookahead = 0);
  ~UdpReferenceReceiver();

  UdpReferenceReceiver(const UdpReferenceReceiver&) = delete;
  UdpReferenceReceiver& operator=(const UdpReferenceReceiver&) = delete;

  // Drain the socket, keep the newest valid packet. Returns true if a new
  // frame arrived since the last call.
  bool Poll();

  // Number of lookahead values this receiver expects (28*H).
  int ExpectedLookahead() const { return expected_lookahead_; }

  // Latest frame (valid=false until the first packet arrives).
  const ReferenceFrame& Latest() const { return latest_; }

  // Seconds since the last packet (large if none yet).
  double Staleness() const;

  // Drain and DISCARD everything buffered in the socket and invalidate the
  // latest frame. Call on mode entry: packets from a previous session sit in
  // the kernel buffer and would otherwise be adopted as a "fresh" reference
  // (measured: a mid-stride walking02 frame held as the stand target).
  void Reset();

 private:
  int fd_ = -1;
  int expected_lookahead_ = 0;
  ReferenceFrame latest_;
};

}  // namespace rl_teleop
