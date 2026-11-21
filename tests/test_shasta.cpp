// T1.1: SHASTA transport with a prescribed constant velocity (no source term), periodic box.
#include <doctest/doctest.h>

#include "spf/shasta.hpp"

#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

using namespace spf;

namespace {

struct AdvResult {
  Real sum0, sum1, max0, max1, min0, min1, l1;
};

// Advect a profile U0(x), x in [0,1), with v = 0.5 for one box crossing (5 nx steps at lambda = 0.4).
AdvResult advect(int nx, const std::function<Real(Real)>& profile, const ShastaParams& prm) {
  const Real L = 1.0, dx = L / nx, v0 = 0.5;
  Grid gr = Grid::make_1d(nx, dx);
  Field3 U = gr.make_field("U"), Un = gr.make_field("Un"), v = gr.make_field("v"), f = gr.make_field("f");
  ShastaWork w(gr);
  {
    auto h = Kokkos::create_mirror_view(U);
    for (int i = 0; i < nx; ++i) h(gr.g[0] + i, 0, 0) = profile((i + 0.5) * dx);
    Kokkos::deep_copy(U, h);
    Kokkos::deep_copy(v, v0);
    Kokkos::deep_copy(f, 0.0);
  }
  const Field3 U_init = gr.make_field("U_init");
  Kokkos::deep_copy(U_init, U);
  AdvResult r;
  r.sum0 = field_sum(gr, U); r.max0 = field_max(gr, U); r.min0 = field_min(gr, U);
  const long nsteps = std::lround(L / (v0 * prm.lambda * dx));
  Real max_seen = r.max0, min_seen = r.min0;
  for (long s = 0; s < nsteps; ++s) {
    fill_ghost(gr, 0, prm.bc, U);
    shasta_step(gr, 0, prm.lambda, prm, U, v, f, Un, w);
    std::swap(U, Un);
    max_seen = std::fmax(max_seen, field_max(gr, U));
    min_seen = std::fmin(min_seen, field_min(gr, U));
  }
  r.sum1 = field_sum(gr, U); r.max1 = max_seen; r.min1 = min_seen;
  auto hU = Kokkos::create_mirror_view(U);
  auto h0 = Kokkos::create_mirror_view(U_init);
  Kokkos::deep_copy(hU, U); Kokkos::deep_copy(h0, U_init);
  Real l1 = 0.0;
  for (int i = 0; i < nx; ++i) l1 += std::fabs(hU(gr.g[0] + i, 0, 0) - h0(gr.g[0] + i, 0, 0)) * dx;
  r.l1 = l1;
  return r;
}

Real square(Real x) { return (x >= 0.4 && x < 0.6) ? 1.0 : 0.0; }
Real gauss(Real x) { return std::exp(-(x - 0.5) * (x - 0.5) / (2.0 * 0.05 * 0.05)); }

}  // namespace

TEST_CASE("T1.1 advection, v = 0.5, periodic: conservation and no new extrema") {
  ShastaParams prm;  // phoenical, eta_ad = 1/8, lambda = 0.4, periodic
  for (int which = 0; which < 2; ++which) {
    const bool sq = which == 0;
    for (int nx : {100, 200, 400}) {
      const AdvResult r = advect(nx, sq ? square : gauss, prm);
      const Real cons = std::fabs(r.sum1 - r.sum0) / std::fabs(r.sum0);
      std::printf("[WP1] T1.1 %s nx=%d: |dSum|/Sum = %.3e  max %.15g -> %.15g  min %.15g -> %.15g  L1 = %.6e\n",
                  sq ? "square" : "gauss", nx, cons, r.max0, r.max1, r.min0, r.min1, r.l1);
      CHECK(cons < 1e-12);
      CHECK(r.max1 <= r.max0 + 1e-12);
      CHECK(r.min1 >= r.min0 - 1e-12);
    }
  }
}

TEST_CASE("T1.1 Gaussian: L1 error at 100, 200, 400 cells and convergence order") {
  ShastaParams prm;
  Real e[3];
  const int nxs[3] = {100, 200, 400};
  for (int k = 0; k < 3; ++k) e[k] = advect(nxs[k], gauss, prm).l1;
  const Real o1 = std::log2(e[0] / e[1]), o2 = std::log2(e[1] / e[2]);
  std::printf("[WP1] T1.1 gauss L1: %.6e (100) %.6e (200) %.6e (400); order 100->200 = %.3f, 200->400 = %.3f\n", e[0], e[1],
              e[2], o1, o2);
  CHECK(e[1] < e[0]);
  CHECK(e[2] < e[1]);
}

TEST_CASE("T1.1 explicit antidiffusion: conservation and no new extrema") {
  ShastaParams prm;
  prm.phoenical = false;
  for (int which = 0; which < 2; ++which) {
    const AdvResult r = advect(200, which == 0 ? square : gauss, prm);
    CHECK(std::fabs(r.sum1 - r.sum0) / std::fabs(r.sum0) < 1e-12);
    CHECK(r.max1 <= r.max0 + 1e-12);
    CHECK(r.min1 >= r.min0 - 1e-12);
  }
}

TEST_CASE("ghost fill: periodic and outflow along direction 0, 1, 2 of a 3D field") {
  Grid gr = Grid::make_3d(5, 4, 3, 1.0);
  for (int dir = 0; dir < 3; ++dir) {
    for (int per = 0; per < 2; ++per) {
      Field3 F = gr.make_field("F");
      auto h = Kokkos::create_mirror_view(F);
      auto value = [](int i, int j, int k) { return 1000.0 * i + 100.0 * j + k; };
      for (int i = 0; i < gr.size(0); ++i)
        for (int j = 0; j < gr.size(1); ++j)
          for (int k = 0; k < gr.size(2); ++k) h(i, j, k) = value(i, j, k);
      Kokkos::deep_copy(F, h);
      fill_ghost(gr, dir, per ? Boundary::Periodic : Boundary::Outflow, F);
      Kokkos::deep_copy(h, F);
      const int g = gr.g[dir], n = gr.n[dir];
      int bad = 0;
      for (int a = 0; a < g; ++a) {
        const int srcL = per ? a + n : g, srcR = per ? g + a : g + n - 1;
        for (int p = 0; p < gr.size((dir + 1) % 3); ++p)
          for (int q = 0; q < gr.size((dir + 2) % 3); ++q) {
            int c[3], s[3];
            c[dir] = a; c[(dir + 1) % 3] = p; c[(dir + 2) % 3] = q;
            s[dir] = srcL; s[(dir + 1) % 3] = p; s[(dir + 2) % 3] = q;
            if (h(c[0], c[1], c[2]) != value(s[0], s[1], s[2])) ++bad;
            c[dir] = g + n + a; s[dir] = srcR;
            if (h(c[0], c[1], c[2]) != value(s[0], s[1], s[2])) ++bad;
          }
      }
      CHECK(bad == 0);
    }
  }
}
