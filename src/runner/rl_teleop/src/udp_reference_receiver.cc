#include "rl_teleop/udp_reference_receiver.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

namespace rl_teleop {

namespace {
double MonotonicNow() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<double>(ts.tv_sec) + 1e-9 * static_cast<double>(ts.tv_nsec);
}
}  // namespace

UdpReferenceReceiver::UdpReferenceReceiver(int port) {
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
  LOG(INFO) << "rl_teleop: listening for reference stream on UDP :" << port;
}

UdpReferenceReceiver::~UdpReferenceReceiver() {
  if (fd_ >= 0) close(fd_);
}

bool UdpReferenceReceiver::Poll() {
  if (fd_ < 0) return false;
  uint8_t buf[kPacketBytes];
  bool got_new = false;

  // Drain everything queued; keep only the newest well-formed packet.
  while (true) {
    ssize_t n = recv(fd_, buf, sizeof(buf), 0);
    if (n < 0) break;  // EWOULDBLOCK -> drained
    if (static_cast<size_t>(n) != kPacketBytes) continue;

    uint32_t magic, seq;
    std::memcpy(&magic, buf, 4);
    std::memcpy(&seq, buf + 4, 4);
    if (magic != kMagic) continue;

    float vals[kNumJoints + 4 + 3];
    std::memcpy(vals, buf + 8, sizeof(vals));

    for (int i = 0; i < kNumJoints; ++i) latest_.jpos(i) = vals[i];
    for (int i = 0; i < 4; ++i) latest_.quat_wxyz(i) = vals[kNumJoints + i];
    for (int i = 0; i < 3; ++i) latest_.pos(i) = vals[kNumJoints + 4 + i];
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

}  // namespace rl_teleop
