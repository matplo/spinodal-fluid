#pragma once
// Host-side random number generator of the particle sampler: xoshiro256** seeded through splitmix64.
// The sequence is identical on every platform; nothing here depends on the thread that runs it.
#include <cmath>
#include <cstdint>

namespace spf::particlisation {

inline std::uint64_t splitmix64(std::uint64_t& s) {
  std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Seed of one sample from (base seed, event id, sample id): three chained splitmix64 mixes.
inline std::uint64_t derive_seed(std::uint64_t base, std::uint64_t event_id, std::uint64_t sample_id) {
  std::uint64_t s = base;
  std::uint64_t a = splitmix64(s);
  s = a ^ (event_id + 0x632BE59BD9B4E019ULL);
  std::uint64_t b = splitmix64(s);
  s = b ^ (sample_id + 0x9E3779B185EBCA87ULL);
  return splitmix64(s);
}

class Rng {
 public:
  explicit Rng(std::uint64_t seed = 1) {
    std::uint64_t s = seed;
    for (auto& v : s_) v = splitmix64(s);
  }
  std::uint64_t next() {
    const std::uint64_t result = rotl(s_[1] * 5, 7) * 9;
    const std::uint64_t t = s_[1] << 17;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45);
    return result;
  }
  // uniform in [0, 1)
  double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
  // uniform in (0, 1]
  double uniform_pos() { return (static_cast<double>(next() >> 11) + 1.0) * (1.0 / 9007199254740992.0); }
  // integer in [0, n)
  std::uint64_t below(std::uint64_t n) { return static_cast<std::uint64_t>(uniform() * static_cast<double>(n)) % n; }

  // Poisson variate: product method for lambda < 10, transformed rejection (Hoermann 1993, PTRS) otherwise.
  std::uint64_t poisson(double lambda) {
    if (!(lambda > 0.0)) return 0;
    if (lambda < 10.0) {
      const double L = std::exp(-lambda);
      double p = 1.0;
      std::uint64_t k = 0;
      do {
        ++k;
        p *= uniform();
      } while (p > L);
      return k - 1;
    }
    const double slam = std::sqrt(lambda), loglam = std::log(lambda);
    const double b = 0.931 + 2.53 * slam, a = -0.059 + 0.02483 * b;
    const double invalpha = 1.1239 + 1.1328 / (b - 3.4), vr = 0.9277 - 3.6224 / (b - 2.0);
    for (;;) {
      const double U = uniform() - 0.5, V = uniform_pos();
      const double us = 0.5 - std::fabs(U);
      const double kd = std::floor((2.0 * a / us + b) * U + lambda + 0.43);
      if (us >= 0.07 && V <= vr) return static_cast<std::uint64_t>(kd);
      if (kd < 0.0 || (us < 0.013 && V > us)) continue;
      if (std::log(V) + std::log(invalpha) - std::log(a / (us * us) + b) <= -lambda + kd * loglam - std::lgamma(kd + 1.0))
        return static_cast<std::uint64_t>(kd);
    }
  }

 private:
  static std::uint64_t rotl(std::uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  std::uint64_t s_[4];
};

}  // namespace spf::particlisation
