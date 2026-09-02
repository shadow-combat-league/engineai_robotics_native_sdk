#include "rl_teleop/udp_reference_receiver.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <cstring>
#include <vector>
#include <stdexcept>

namespace rl_teleop {

namespace {
double MonotonicNow() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}
}  // namespace

UdpReferenceReceiver::UdpReferenceReceiver(int port, int expected_lookahead)
    : expected_lookahead_(expected_lookahead) {
  fd_ = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd_ < 0) {
    throw std::runtime_error("rl_teleop: failed to create UDP socket");
  }
  int reuse = 1;
  setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (bind(fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(fd_);
    fd_ = -1;
    throw std::runtime_error("rl_teleop: failed to bind UDP port " + std::to_string(port));
  }
  int flags = fcntl(fd_, F_GETFL, 0);
  fcntl(fd_, F_SETFL, flags | O_NONBLOCK);

  latest_.jpos = Eigen::VectorXd::Zero(kNumJoints);
  latest_.quat_wxyz << 1.0, 0.0, 0.0, 0.0;
  latest_.pos.setZero();
  latest_.lookahead = Eigen::VectorXd::Zero(expected_lookahead_);
  LOG(INFO) << "rl_teleop: listening for reference stream on UDP :" << port
            << " (protocol " << (expected_lookahead_ > 0 ? "TTP2" : "TTP1")
            << ", lookahead values " << expected_lookahead_ << ")";
}

UdpReferenceReceiver::~UdpReferenceReceiver() {
  if (fd_ >= 0) close(fd_);
}

bool UdpReferenceReceiver::Poll() {
  if (fd_ < 0) return false;
  uint8_t buf[kMaxPacketBytes];
  const size_t want_bytes = kPacketBytes + expected_lookahead_ * 4;
  const uint32_t want_magic = expected_lookahead_ > 0 ? kMagicV2 : kMagic;
  bool got_new = false;

  // Drain everything queued; keep only the newest well-formed packet.
  while (true) {
    ssize_t n = recv(fd_, buf, sizeof(buf), 0);
    if (n < 0) break;  // EWOULDBLOCK -> drained
    if (static_cast<size_t>(n) != want_bytes) {
      LOG_EVERY_N(WARNING, 250)
          << "rl_teleop: dropping reference packet of " << n << " bytes (want "
          << want_bytes << ") — publisher/policy generation mismatch?";
      continue;
    }

    uint32_t magic, seq;
    std::memcpy(&magic, buf, 4);
    std::memcpy(&seq, buf + 4, 4);
    if (magic != want_magic) {
      LOG_EVERY_N(WARNING, 250)
          << "rl_teleop: dropping reference packet with magic 0x" << std::hex
          << magic << " (want 0x" << want_magic << std::dec
          << ") — publisher/policy generation mismatch?";
      continue;
    }

    float vals[kNumJoints + 4 + 3];
    std::memcpy(vals, buf + 8, sizeof(vals));

    for (int i = 0; i < kNumJoints; ++i) latest_.jpos(i) = vals[i];
    for (int i = 0; i < 4; ++i) latest_.quat_wxyz(i) = vals[kNumJoints + i];
    for (int i = 0; i < 3; ++i) latest_.pos(i) = vals[kNumJoints + 4 + i];
    if (expected_lookahead_ > 0) {
      std::vector<float> la(expected_lookahead_);
      std::memcpy(la.data(), buf + kPacketBytes, expected_lookahead_ * 4);
      for (int i = 0; i < expected_lookahead_; ++i) latest_.lookahead(i) = la[i];
    }
    latest_.seq = seq;
    latest_.recv_time = MonotonicNow();
    latest_.valid = true;
    got_new = true;
  }
  return got_new;
}

double UdpReferenceReceiver::Staleness() const {
  if (!latest_.valid) return 1e9;
  return MonotonicNow() - latest_.recv_time;
}

void UdpReferenceReceiver::Reset() {
  if (fd_ >= 0) {
    uint8_t buf[kPacketBytes];
    int discarded = 0;
    while (recv(fd_, buf, sizeof(buf), 0) >= 0) ++discarded;
    if (discarded > 0) {
      LOG(INFO) << "rl_teleop: discarded " << discarded
                << " buffered reference packets from a previous session";
    }
  }
  latest_.valid = false;
  latest_.seq = 0;
  latest_.recv_time = 0.0;
}

}  // namespace rl_teleop
