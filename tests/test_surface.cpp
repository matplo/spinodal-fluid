// WP10a: switching (particlisation) surface.  Geometry of the 4D surface finder (T10a.0), static and suddenly shrinking sphere (T10a.2),
// uniformly moving slab (T10a.3), Hubble-like expansion (T10a.4), closure of the surface integrals (T10a.1), isochronous versus isoenergy
// (T10a.5) and the backend reference (T10a.6).  Every test prints its numbers with the prefix [WP10a].
#include <doctest/doctest.h>

#include "spf/config.hpp"
#include "spf/eos/eos.hpp"
#include "spf/eos/table.hpp"
#include "spf/hydro3d.hpp"
#include "spf/initial_state.hpp"
#include "spf/surface.hpp"
#include "test_eos_ideal.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace spf;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kEpsS = 0.94 * 0.153;

Config surf_config() {
  Config cfg = Config::from_file(SPF_DEFAULT_CONFIG);
  cfg.set_from_string("particlisation.enabled=true");
  return cfg;
}

// Element accessors of a sorted host record vector.
struct Rec {
  const std::vector<Real>& v;
  size_t n() const { return v.size() / kSurfRecord; }
  Real at(size_t r, int q) const { return v[r * kSurfRecord + q]; }
};

// Host-filled level fields (eps, rho, u^x, u^y, u^z) on a grid; f(x, y, z, out[5]) at the physical cell centre.
template <class F>
void fill_level(const Grid& g, const std::array<Real, 3>& org, Field3 lev[5], F&& f) {
  for (int c = 0; c < 5; ++c) {
    lev[c] = g.make_field("test_level");
    auto h = Kokkos::create_mirror_view(lev[c]);
    Kokkos::deep_copy(h, 0.0);
    for (int i = 0; i < g.n[0]; ++i)
      for (int j = 0; j < g.n[1]; ++j)
        for (int k = 0; k < g.n[2]; ++k) {
          Real out[5];
          f(org[0] + i * g.dx, org[1] + j * g.dx, org[2] + k * g.dx, out);
          h(g.g[0] + i, g.g[1] + j, g.g[2] + k) = out[c];
        }
    Kokkos::deep_copy(lev[c], h);
  }
}

template <class EoS>
SurfaceFinder<EoS> finder_no_files(const EoS& eos, Real eps_sw_eps_s, const Grid& g, const std::array<Real, 3>& org, Real t_init = 0.0) {
  Config cfg = surf_config();
  cfg.set_from_string("particlisation.eps_sw_eps_s=" + std::to_string(eps_sw_eps_s));
  ParticlisationParams pp = ParticlisationParams::from_config(cfg);
  typename SurfaceFinder<EoS>::Meta meta;
  meta.t_init = t_init;
  meta.origin = org;
  return SurfaceFinder<EoS>(eos, pp, g, "", meta);
}

// ------------------------------------------------------------------ T10a.0 geometry of one hypercube and of closed lattice surfaces
const double kL1[4] = {1.0, 1.0, 1.0, 1.0};

int pieces_of(const Real f[16], surf::Piece* out) { return surf::find_pieces(f, kL1, out, surf::kMaxPieces); }

}  // namespace

TEST_CASE("T10a.0 hypercube geometry: planes, orientation, closed random surfaces with several pieces per hypercube") {
  surf::Piece pc[surf::kMaxPieces];
  // isochronous plane t = 0.5 (above before, below after): dSigma = (1, 0, 0, 0)
  Real f[16];
  for (int c = 0; c < 16; ++c) f[c] = 0.5 - (c & 1);
  REQUIRE(pieces_of(f, pc) == 1);
  CHECK(pc[0].n[0] == doctest::Approx(1.0).epsilon(1e-14));
  CHECK(std::fabs(pc[0].n[1]) + std::fabs(pc[0].n[2]) + std::fabs(pc[0].n[3]) < 1e-14);
  // random planes: the element normal is the analytic normal (towards decreasing f) times the 3-volume of the plane inside the hypercube
  std::mt19937_64 rng(7);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::uniform_real_distribution<double> ud(0.2, 0.8);
  double worst_angle = 0.0;
  for (int t = 0; t < 500; ++t) {
    double gr[4], x0[4];
    for (int a = 0; a < 4; ++a) { gr[a] = nd(rng); x0[a] = ud(rng); }
    for (int c = 0; c < 16; ++c) {
      double s = 0.0;
      for (int a = 0; a < 4; ++a) s += (((c >> a) & 1) - x0[a]) * gr[a];
      f[c] = -s;
    }
    const int np = pieces_of(f, pc);
    REQUIRE(np == 1);
    double dot = 0, nn = 0, gg = 0;
    for (int a = 0; a < 4; ++a) { dot += pc[0].n[a] * gr[a]; nn += pc[0].n[a] * pc[0].n[a]; gg += gr[a] * gr[a]; }
    worst_angle = std::max(worst_angle, std::acos(std::min(1.0, dot / std::sqrt(nn * gg))));
  }
  std::printf("[WP10a] T10a.0 random planes: largest angle between element normal and analytic normal %.3e rad (500 planes)\n", worst_angle);
  CHECK(worst_angle < 1e-7);
  // closed surfaces of random lattice fields (values N(0,1) on 8^4 nodes, boundary nodes below): the element normals add up to zero
  for (int trial = 0; trial < 3; ++trial) {
    const int N = 8;
    std::vector<double> fld(N * N * N * N);
    auto id = [&](int a, int b, int c, int d) { return ((a * N + b) * N + c) * N + d; };
    for (int a = 0; a < N; ++a)
      for (int b = 0; b < N; ++b)
        for (int c = 0; c < N; ++c)
          for (int d = 0; d < N; ++d) {
            const bool edge = a == 0 || b == 0 || c == 0 || d == 0 || a == N - 1 || b == N - 1 || c == N - 1 || d == N - 1;
            fld[id(a, b, c, d)] = edge ? -1.0 : nd(rng);
          }
    double sum[4] = {0, 0, 0, 0}, abs_sum = 0;
    long nel = 0, multi = 0, errs = 0;
    int maxp = 0;
    for (int a = 0; a < N - 1; ++a)
      for (int b = 0; b < N - 1; ++b)
        for (int c = 0; c < N - 1; ++c)
          for (int d = 0; d < N - 1; ++d) {
            for (int q = 0; q < 16; ++q) f[q] = fld[id(a + (q & 1), b + ((q >> 1) & 1), c + ((q >> 2) & 1), d + ((q >> 3) & 1))];
            const int np = pieces_of(f, pc);
            if (np < 0) { ++errs; continue; }
            nel += np;
            if (np > 1) ++multi;
            maxp = std::max(maxp, np);
            for (int p = 0; p < np; ++p)
              for (int x = 0; x < 4; ++x) { sum[x] += pc[p].n[x]; abs_sum += std::fabs(pc[p].n[x]); }
          }
    const double m = std::max({std::fabs(sum[0]), std::fabs(sum[1]), std::fabs(sum[2]), std::fabs(sum[3])}) / abs_sum;
    std::printf("[WP10a] T10a.0 random closed surface %d: %ld elements, %ld hypercubes with more than one piece (max %d), topology errors %ld, "
                "max |sum dSigma_mu| / sum |dSigma_mu| = %.3e\n", trial, nel, multi, maxp, errs, m);
    CHECK(errs == 0);
    CHECK(multi > 0);
    CHECK(m < 1e-13);
  }
}

// ------------------------------------------------------------------ T10a.2 static sphere and suddenly shrinking sphere (prescribed levels)
TEST_CASE("T10a.2 sphere: static surface area and swept volume of a suddenly shrinking sphere") {
  const int n = 40;
  const Real dx = 0.25;
  const Grid g = Grid::make_3d(n, n, n, dx);
  const std::array<Real, 3> org = centred_origin(g);
  const Real sw_eps_s = 4.0, sw = sw_eps_s * kEpsS, w = 2.0;
  auto sphere = [&](Real R) {
    return [=](Real x, Real y, Real z, Real* o) {
      const Real r = std::sqrt(x * x + y * y + z * z);
      o[0] = sw * (1.0 + (R - r) / w);
      o[1] = 0.3;
      o[2] = o[3] = o[4] = 0.0;
    };
  };
  auto sf = finder_no_files(TestEosIdeal{}, sw_eps_s, g, org);
  const Real dt = 0.1;
  // (a) static sphere R = 3.5 fm: the surface is the tube S^2 x [0, dt]
  {
    const Real R = 3.5;
    Field3 a[5], b[5];
    fill_level(g, org, a, sphere(R));
    fill_level(g, org, b, sphere(R));
    SurfFlux fl;
    const auto v = sf.elements_between(a, b, 0.0, dt, &fl);
    Rec E{v};
    double n0 = 0, nr = 0, nx = 0, ny = 0, nz = 0, sabs = 0;
    for (size_t r = 0; r < E.n(); ++r) {
      const double x = E.at(r, 1), y = E.at(r, 2), z = E.at(r, 3), rr = std::sqrt(x * x + y * y + z * z);
      n0 += E.at(r, 4); nx += E.at(r, 5); ny += E.at(r, 6); nz += E.at(r, 7);
      nr += (x * E.at(r, 5) + y * E.at(r, 6) + z * E.at(r, 7)) / rr;
      sabs += std::fabs(E.at(r, 5)) + std::fabs(E.at(r, 6)) + std::fabs(E.at(r, 7));
    }
    const double area = nr / dt, area_ex = 4.0 * kPi * R * R;
    std::printf("[WP10a] T10a.2a static sphere R = %.2f fm, dx = %.2f fm: %zu elements, area %.6f fm^2 vs 4 pi R^2 = %.6f (rel. %.3e); "
                "sum dSigma_0 = %.2e, |sum dSigma_i| / sum |dSigma_i| = %.2e\n",
                R, dx, E.n(), area, area_ex, area / area_ex - 1.0, n0, std::max({std::fabs(nx), std::fabs(ny), std::fabs(nz)}) / sabs);
    CHECK(std::fabs(area / area_ex - 1.0) < 1e-2);
    CHECK(std::fabs(n0) < 1e-12 * sabs);
    CHECK(std::max({std::fabs(nx), std::fabs(ny), std::fabs(nz)}) < 1e-12 * sabs);
  }
  // (b) sudden change: R0 = 4.5 fm at t = 0, R1 = 2.5 fm at t = dt; the region between the spheres leaves the fluid within dt
  {
    const Real R0 = 4.5, R1 = 2.5;
    Field3 a[5], b[5];
    fill_level(g, org, a, sphere(R0));
    fill_level(g, org, b, sphere(R1));
    const auto v = sf.elements_between(a, b, 0.0, dt);
    Rec E{v};
    double n0 = 0, nr = 0;
    for (size_t r = 0; r < E.n(); ++r) {
      const double x = E.at(r, 1), y = E.at(r, 2), z = E.at(r, 3), rr = std::sqrt(x * x + y * y + z * z);
      n0 += E.at(r, 4);
      nr += (x * E.at(r, 5) + y * E.at(r, 6) + z * E.at(r, 7)) / rr;
    }
    const double vol_ex = 4.0 * kPi / 3.0 * (R0 * R0 * R0 - R1 * R1 * R1);
    const double nr_ex = 4.0 * kPi * dt * (R0 * R0 * R0 - R1 * R1 * R1) / (3.0 * (R0 - R1));
    std::printf("[WP10a] T10a.2b sphere R = %.1f -> %.1f fm within dt: sum dSigma_0 = %.6f fm^3 vs volume %.6f (rel. %.3e); "
                "sum radial dSigma = %.6f vs %.6f (rel. %.3e)\n", R0, R1, n0, vol_ex, n0 / vol_ex - 1.0, nr, nr_ex, nr / nr_ex - 1.0);
    CHECK(std::fabs(n0 / vol_ex - 1.0) < 1e-2);
    CHECK(std::fabs(nr / nr_ex - 1.0) < 1e-2);
  }
}

// ------------------------------------------------------------------ T10a.3 uniformly moving slab (prescribed levels, exact)
TEST_CASE("T10a.3 uniformly moving slab: normals and surface integrals equal the analytic values") {
  const int n = 32;
  const Real dx = 0.25, dt = 0.1, v = 0.5, h = 2.0, wr = 1.0;
  const Grid g = Grid::make_3d(n, n, n, dx);
  const std::array<Real, 3> org = centred_origin(g);
  const Real sw_eps_s = 4.0, sw = sw_eps_s * kEpsS, rho = 0.4;
  const Real e3[3] = {1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0};  // slab normal
  const Real gam = 1.0 / std::sqrt(1.0 - v * v);
  // eps = sw (1 + (h - |e.x - v t - s0|) / wr): planes e.x = s0 + v t +- h; s0 chosen so that no plane passes through a lattice node
  const Real s0 = 0.0123;
  auto slab = [&](Real t) {
    return [=](Real x, Real y, Real z, Real* o) {
      const Real s = e3[0] * x + e3[1] * y + e3[2] * z - v * t - s0;
      o[0] = sw * (1.0 + (h - std::fabs(s)) / wr);
      o[1] = rho;
      o[2] = gam * v * e3[0]; o[3] = gam * v * e3[1]; o[4] = gam * v * e3[2];
    };
  };
  Field3 a[5], b[5];
  fill_level(g, org, a, slab(0.0));
  fill_level(g, org, b, slab(dt));
  auto sf = finder_no_files(TestEosIdeal{}, sw_eps_s, g, org);
  const auto vv = sf.elements_between(a, b, 0.0, dt);
  Rec E{vv};
  const Real p = sw / 3.0;
  double worst_dir = 0, worst_ud = 0, worst_E = 0, worst_P = 0, worst_eps = 0, worst_pos = 0;
  long n_plus = 0, n_minus = 0;
  double A_plus = 0, A_minus = 0;
  for (size_t r = 0; r < E.n(); ++r) {
    const double x = E.at(r, 1), y = E.at(r, 2), z = E.at(r, 3), t = E.at(r, 0);
    const double s = e3[0] * x + e3[1] * y + e3[2] * z - v * t - s0;
    const double side = s > 0 ? 1.0 : -1.0;
    (side > 0 ? n_plus : n_minus) += 1;
    worst_pos = std::max(worst_pos, std::fabs(std::fabs(s) - h));
    const double* nn = &vv[r * kSurfRecord + 4];
    const double sp = std::sqrt(nn[1] * nn[1] + nn[2] * nn[2] + nn[3] * nn[3]);  // = A dt (spatial part)
    (side > 0 ? A_plus : A_minus) += sp;
    // analytic dSigma_mu = A dt (-side v, side e)
    const double an[4] = {-side * v, side * e3[0], side * e3[1], side * e3[2]};
    for (int q = 0; q < 4; ++q) worst_dir = std::max(worst_dir, std::fabs(nn[q] / sp - an[q]));
    const double* u = &vv[r * kSurfRecord + 8];
    const double ud = u[0] * nn[0] + u[1] * nn[1] + u[2] * nn[2] + u[3] * nn[3];
    worst_ud = std::max(worst_ud, std::fabs(ud) / (u[0] * sp));
    const double eps = vv[r * kSurfRecord + 12];
    worst_eps = std::max(worst_eps, std::fabs(eps / sw - 1.0));
    // T^{0 mu} dSigma_mu = -p dSigma_0 = side p v A dt; T^{i mu} dSigma_mu = p dSigma_i
    const double Eflux = (eps + p) * u[0] * ud - p * nn[0];
    worst_E = std::max(worst_E, std::fabs(Eflux - side * p * v * sp) / (p * sp));
    for (int i = 1; i <= 3; ++i) worst_P = std::max(worst_P, std::fabs((eps + p) * u[i] * ud + p * nn[i] - side * p * e3[i - 1] * sp) / (p * sp));
  }
  std::printf("[WP10a] T10a.3 moving slab v = %.2f along (1,2,2)/3: %zu elements (%ld on the front, %ld on the back face), "
              "max deviation: plane position %.2e fm, normal direction %.2e, u.dSigma / (u0 |dSigma|) %.2e, eps / eps_sw - 1 %.2e, "
              "energy flux %.2e, momentum flux %.2e (relative to p |dSigma|); summed |dSigma_spatial| front %.6f back %.6f fm^3\n",
              v, E.n(), n_plus, n_minus, worst_pos, worst_dir, worst_ud, worst_eps, worst_E, worst_P, A_plus, A_minus);
  CHECK(n_plus > 0); CHECK(n_minus > 0);
  CHECK(worst_pos < 1e-12);
  CHECK(worst_dir < 1e-12);
  CHECK(worst_ud < 1e-12);
  CHECK(worst_eps < 1e-12);
  CHECK(worst_E < 1e-12);
  CHECK(worst_P < 1e-12);
}
