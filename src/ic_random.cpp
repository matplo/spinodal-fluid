// Smooth random initial state for tests and benchmarks (host only).
#include "spf/hydro3d.hpp"

#include <cmath>

namespace spf {
namespace {

// splitmix64: platform independent (std distributions are not)
struct SplitMix {
  std::uint64_t s;
  std::uint64_t next() {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
};

}  // namespace

HostState smooth_random_state(int nx, int ny, int nz, std::uint64_t seed, int n_modes) {
  SplitMix rng{seed};
  const double two_pi = 6.283185307179586476925287;
  struct Mode { int k[3]; double amp[4], phase[4]; };
  std::vector<Mode> modes;
  while (static_cast<int>(modes.size()) < n_modes) {
    Mode m;
    for (int d = 0; d < 3; ++d) m.k[d] = static_cast<int>(rng.next() % 5) - 2;  // -2 .. 2
    if (m.k[0] == 0 && m.k[1] == 0 && m.k[2] == 0) continue;
    for (int f = 0; f < 4; ++f) { m.amp[f] = rng.uniform() - 0.5; m.phase[f] = two_pi * rng.uniform(); }
    modes.push_back(m);
  }
  const size_t n = static_cast<size_t>(nx) * ny * nz;
  std::vector<double> s[4];
  for (auto& v : s) v.assign(n, 0.0);
  for (int ix = 0; ix < nx; ++ix)
    for (int iy = 0; iy < ny; ++iy)
      for (int iz = 0; iz < nz; ++iz) {
        const size_t id = (static_cast<size_t>(ix) * ny + iy) * nz + iz;
        for (const Mode& m : modes) {
          const double arg = two_pi * (m.k[0] * (ix + 0.5) / nx + m.k[1] * (iy + 0.5) / ny + m.k[2] * (iz + 0.5) / nz);
          for (int f = 0; f < 4; ++f) s[f][id] += m.amp[f] * std::cos(arg + m.phase[f]);
        }
      }
  // fields 0..2 -> (eps, rho, v-direction scalars) normalised to max |s| = 1; field 3 -> third velocity component
  HostState out;
  out.eps.resize(n); out.rho.resize(n); out.vx.resize(n); out.vy.resize(n); out.vz.resize(n);
  double mx[4] = {0, 0, 0, 0};
  for (int f = 0; f < 4; ++f)
    for (size_t i = 0; i < n; ++i) mx[f] = std::fmax(mx[f], std::fabs(s[f][i]));
  for (size_t i = 0; i < n; ++i) {
    out.eps[i] = 1.5 + 0.5 * s[0][i] / mx[0];
    out.rho[i] = 0.3 + 0.1 * s[1][i] / mx[1];
    // each component bounded by 0.3 / sqrt(3), so |v| <= 0.3
    const double c = 0.3 / std::sqrt(3.0);
    out.vx[i] = c * s[2][i] / mx[2];
    out.vy[i] = c * s[3][i] / mx[3];
    out.vz[i] = c * (0.5 * (s[0][i] / mx[0]) - 0.5 * (s[1][i] / mx[1]));
  }
  return out;
}

}  // namespace spf
