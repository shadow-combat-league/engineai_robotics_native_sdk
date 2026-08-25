#pragma once

// Non-blocking UDP receiver for the live teleop reference stream.
//
// Packet layout (little-endian float32 unless noted):
//   uint32 magic   = 0x54545031 ("TTP1")
//   uint32 seq
//   float  ref_jpos[25]   Isaac obs order (publisher converts)
//   float  ref_quat[4]    wxyz, reference base orientation (world)
//   float  ref_pos[3]     xyz,  reference anchor position (world, anchored)
// Total: 8 + 32*4 = 136 bytes.

#include <Eigen/Dense>
#include <cstdint>
#include <mutex>

namespace rl_teleop {

struct ReferenceFrame {
  uint32_t seq = 0;
  Eigen::VectorXd jpos;      // 25
  Eigen::Vector4d quat_wxyz; // reference base orientation
  Eigen::Vector3d pos;       // reference anchor position
  double recv_time = 0.0;    // monotonic seconds
  bool valid = false;
};

class UdpReferenceReceiver {
 public:
  static constexpr uint32_t kMagic = 0x54545031;
  static constexpr int kNumJoints = 25;
  static constexpr size_t kPacketBytes = 8 + (kNumJoints + 4 + 3) * 4;

  explicit UdpReferenceReceiver(int port);
  ~UdpReferenceReceiver();

  UdpReferenceReceiver(const UdpReferenceReceiver&) = delete;
  UdpReferenceReceiver& operator=(const UdpReferenceReceiver&) = delete;

  // Drain the socket, keep the newest valid packet. Returns true if a new
  // frame arrived since the last call.
  bool Poll();

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
  ReferenceFrame latest_;
};

}  // namespace rl_teleop
