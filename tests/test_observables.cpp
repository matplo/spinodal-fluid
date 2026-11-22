// WP6: observables. T6.1 uniform density, T6.2 Gaussian moments, T6.3 clump finder, T6.4 transverse flow, T6.5 phase plane and unstable
// fraction, T6.6 recorder, post-processing, end of the fluid stage and ensemble script.
#include <doctest/doctest.h>

#include "spf/config.hpp"
#include "spf/eos/eos.hpp"
#include "spf/hydro3d.hpp"
#include "spf/observables.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

using namespace spf;

namespace {

constexpr Real kRhoS = 0.153, kEpsS = 0.940 * 0.153, kPi = 3.14159265358979323846;
const std::string kSrcDir = SPF_SOURCE_DIR;

Config default_config() { return Config::from_file(SPF_DEFAULT_CONFIG); }

// Fields on a Grid; fn(ix, iy, iz, x, y, z, v) with cell-centre coordinates measured from the box centre.
struct Cell { Real R = 0, rho = 0, eps = 0, vx = 0, vy = 0; };
ObsFields make_fields(const Grid& g, const std::function<void(int, int, int, Real, Real, Real, Cell&)>& fn) {
  ObsFields f;
  f.grid = g;
  Field3* t[5] = {&f.R, &f.rho, &f.eps, &f.vx, &f.vy};
  const char* nm[5] = {"R", "rho", "eps", "vx", "vy"};
  Field3Host h[5];
  for (int q = 0; q < 5; ++q) { *t[q] = g.make_field(nm[q]); h[q] = Kokkos::create_mirror_view(*t[q]); Kokkos::deep_copy(h[q], 0.0); }
  for (int i = 0; i < g.n[0]; ++i)
    for (int j = 0; j < g.n[1]; ++j)
      for (int k = 0; k < g.n[2]; ++k) {
        Cell c;
        fn(i, j, k, (i + 0.5 - 0.5 * g.n[0]) * g.dx, (j + 0.5 - 0.5 * g.n[1]) * g.dx, (k + 0.5 - 0.5 * g.n[2]) * g.dx, c);
        h[0](g.g[0] + i, g.g[1] + j, g.g[2] + k) = c.R;
        h[1](g.g[0] + i, g.g[1] + j, g.g[2] + k) = c.rho;
        h[2](g.g[0] + i, g.g[1] + j, g.g[2] + k) = c.eps;
        h[3](g.g[0] + i, g.g[1] + j, g.g[2] + k) = c.vx;
        h[4](g.g[0] + i, g.g[1] + j, g.g[2] + k) = c.vy;
      }
  for (int q = 0; q < 5; ++q) Kokkos::deep_copy(*t[q], h[q]);
  return f;
}

using NeverUnstable = Vs2Mask<EosIdeal>;

ObsResult observe(const ObsFields& f, const ObsParams& p) { return compute_observables(f, p, NeverUnstable{}, 0.0, 0); }

ObsParams default_params() { return ObsParams::from_config(default_config()); }

Real gauss3(Real r2, Real B, Real sigma) { return B / std::pow(2.0 * kPi, 1.5) / (sigma * sigma * sigma) * std::exp(-0.5 * r2 / (sigma * sigma)); }

// Closed form (docs/equations.md, Sect. 11): Gaussians of equal width, baryon numbers B_i, well separated:
//   <rho^N>/<rho>^N = 2^(3N/2) (N+1)^(-3/2) * [sum B_i^(N+1) / A] / [sum B_i^2 / A]^N,  A = sum B_i
Real gauss_moment(const std::vector<Real>& B, int N) {
  Real A = 0, s2 = 0, sN = 0;
  for (Real b : B) { A += b; s2 += b * b; sN += std::pow(b, N + 1); }
  return std::pow(2.0, 1.5 * N) * std::pow(N + 1.0, -1.5) * (sN / A) / std::pow(s2 / A, N);
}

}  // namespace

TEST_CASE("T6.1 uniform density: moments 1, no flow, no clump below threshold") {
  const Grid g = Grid::make_3d(24, 20, 16, 0.5);
  const ObsFields f = make_fields(g, [](int, int, int, Real, Real, Real, Cell& c) { c.R = 0.5; c.rho = 0.45; c.eps = 1.0; });
  for (int rest = 0; rest < 2; ++rest) {
    ObsParams p = default_params();
    p.rest_frame = rest == 1;
    const ObsResult r = observe(f, p);
    Real w = 0;
    for (int n = 1; n <= 7; ++n) w = std::max(w, std::fabs(r.moment[n] - 1.0));
    const Real dens = rest ? 0.45 : 0.5;
    printf("[T6.1] rest=%d max|moment-1| = %.3e, U_perp = %g, A = %.12g (expected %.12g)\n", rest, w, r.u_perp, r.A, dens * 24 * 20 * 16 * 0.125);
    CHECK(w <= 1e-12);
    CHECK(r.u_perp == 0.0);
    CHECK(std::fabs(r.A - dens * 24 * 20 * 16 * 0.125) <= 1e-12 * r.A);
    CHECK(find_clumps(f, p).baryon.empty());  // 0.5 and 0.45 are below 7 rho_s = 1.071
  }
  // above the threshold the whole periodic box is one clump, in 6 and 26 connectivity; open boundaries: also one
  const ObsFields hi = make_fields(g, [](int, int, int, Real, Real, Real, Cell& c) { c.R = 2.0; c.rho = 2.0; });
  for (int conn : {6, 26}) {
    ObsParams p = default_params();
    p.connectivity = conn;
    const ClumpResult c = find_clumps(hi, p);
    REQUIRE(c.baryon.size() == 1);
    CHECK(std::fabs(c.baryon[0] - 2.0 * 24 * 20 * 16 * 0.125) <= 1e-12 * c.baryon[0]);
  }
}

TEST_CASE("T6.2 well-separated Gaussians: closed-form moments") {
  const Real dx = 0.25, sigma = 3.0 * dx;  // sigma = 3 dx, the coarsest allowed resolution
  const Grid g = Grid::make_3d(160, 48, 48, dx);  // 40 x 12 x 12 fm
  struct Case { std::vector<Real> B; };
  const std::vector<Case> cases = {{{2.5, 2.5}}, {{1.0, 1.0, 1.0, 1.0}}, {{1.0, 3.0}}, {{0.5, 2.0, 4.0}}};
  for (const auto& cs : cases) {
    const int M = static_cast<int>(cs.B.size());
    // centres on the x axis, 10 fm = 13.3 sigma apart (>= 10 sigma), at least 5 fm from the walls in x
    std::vector<Real> cx(M);
    for (int m = 0; m < M; ++m) cx[m] = (m - 0.5 * (M - 1)) * 10.0;
    const ObsFields f = make_fields(g, [&](int, int, int, Real x, Real y, Real z, Cell& c) {
      Real s = 0;
      for (int m = 0; m < M; ++m) s += gauss3((x - cx[m]) * (x - cx[m]) + y * y + z * z, cs.B[m], sigma);
      c.R = s; c.rho = s;
    });
    const ObsResult r = observe(f, default_params());
    Real Btot = 0;
    for (Real b : cs.B) Btot += b;
    Real worst = 0;
    std::string line;
    for (int N = 2; N <= 7; ++N) {
      const Real ref = gauss_moment(cs.B, N);
      const Real dev = std::fabs(r.moment[N] / ref - 1.0);
      worst = std::max(worst, dev);
      char b[64]; std::snprintf(b, sizeof b, " N%d:%.1e", N, dev); line += b;
    }
    printf("[T6.2] M=%d B0=%g: A=%.10g (B total %.10g), max rel dev %.3e |%s\n", M, cs.B[0], r.A, Btot, worst, line.c_str());
    CHECK(worst <= 1e-3);
    CHECK(std::fabs(r.A / Btot - 1.0) <= 1e-6);
    if (cs.B.size() == 2 && cs.B[0] == cs.B[1]) CHECK(std::fabs(gauss_moment(cs.B, 5) - std::pow(2.0, 7.5) / std::pow(6.0, 1.5)) <= 1e-12);
  }
}

TEST_CASE("T6.3 clump finder on planted blobs") {
  const Real dx = 0.2;
  const Grid g = Grid::make_3d(48, 48, 40, dx);
  const Real dxf = 0.1;  // finer grid for the curved blobs
  const Grid gf = Grid::make_3d(96, 96, 80, dxf);
  const Real rho0 = 2.0;  // 13 rho_s, above 7 rho_s
  const Real dV = dx * dx * dx;
  auto box = [&](const ObsFields&, int, int, int) {};
  (void)box;
  SUBCASE("cuboids, edge and corner contact, 6 versus 26 connectivity") {
    // A: cells [4,8) x [4,8) x [4,8); B shares an edge with A (cells (8..11, 8..11, 4..7), diagonal in xy);
    // C touches A's corner only (cells 8..11 in all of x y z, but z = 8 vs A's z = 7); D isolated sphere
    auto inbox = [](int i, int j, int k, int a0, int a1, int b0, int b1, int c0, int c1) { return i >= a0 && i < a1 && j >= b0 && j < b1 && k >= c0 && k < c1; };
    const ObsFields f = make_fields(g, [&](int i, int j, int k, Real, Real, Real, Cell& c) {
      const bool A = inbox(i, j, k, 4, 8, 4, 8, 4, 8);
      const bool B = inbox(i, j, k, 8, 12, 8, 12, 4, 8);
      const bool C = inbox(i, j, k, 24, 28, 24, 28, 24, 28);
      const bool Cc = inbox(i, j, k, 28, 32, 28, 32, 28, 32);  // corner contact with C
      if (A || B || C || Cc) { c.R = rho0; c.rho = rho0; }
    });
    ObsParams p = default_params();
    p.periodic = false;
    p.connectivity = 6;
    const ClumpResult c6 = find_clumps(f, p);
    p.connectivity = 26;
    const ClumpResult c26 = find_clumps(f, p);
    const double blk = 64.0 * rho0 * dV;  // 4^3 cells
    printf("[T6.3] cuboids: 6-conn %zu clumps, 26-conn %zu clumps (cell block = %.6g)\n", c6.baryon.size(), c26.baryon.size(), blk);
    REQUIRE(c6.baryon.size() == 4);
    for (double b : c6.baryon) CHECK(std::fabs(b / blk - 1.0) <= 1e-3);
    REQUIRE(c26.baryon.size() == 2);
    for (double b : c26.baryon) CHECK(std::fabs(b / (2 * blk) - 1.0) <= 1e-3);
  }
  SUBCASE("sphere, Gaussian: baryon number above threshold") {
    const Real sg = 0.6;  // fm (6 dx_f)
    const Real Bg = 40.0;  // peak 40 / (15.75 * 0.343) = 7.4 fm^-3
    const Real rs = 1.5;   // sphere radius, fm
    const ObsFields f = make_fields(gf, [&](int, int, int, Real x, Real y, Real z, Cell& c) {
      const Real r2s = (x + 3.0) * (x + 3.0) + y * y + z * z;  // sphere at x = -3
      const Real r2g = (x - 2.0) * (x - 2.0) + y * y + z * z;  // Gaussian at x = +2
      Real v = 0;
      if (r2s < rs * rs) v = rho0;
      v += gauss3(r2g, Bg, sg);
      c.R = v; c.rho = v;
    });
    ObsParams p = default_params();
    p.connectivity = 6;
    // A low threshold (0.02 rho_s) puts the cut at about 4 sigma, where the lattice-point noise of the cut surface is negligible; the
    // analytic reference is the truncated Gaussian integral. (A cut at 1.4 fm radius gave 3e-3 to 5e-3 from that noise: not a bug, the reference
    // is a continuum integral.)
    p.clump_rho_min_rho_s = 0.02;
    const ClumpResult c = find_clumps(f, p);
    REQUIRE(c.baryon.size() == 2);
    // analytic baryon number of the Gaussian above the threshold thr: B [erf(a/sqrt2) - sqrt(2/pi) a exp(-a^2/2)], a = r_c / sigma
    const Real thr = p.clump_rho_min_rho_s * kRhoS;
    const Real rc = sg * std::sqrt(2.0 * std::log(Bg / std::pow(2 * kPi, 1.5) / (sg * sg * sg) / thr));
    const Real a = rc / sg;
    const Real B_gauss = Bg * (std::erf(a / std::sqrt(2.0)) - std::sqrt(2.0 / kPi) * a * std::exp(-0.5 * a * a));
    // the sphere: planted cell count (independent loop over the same predicate), compared with 4/3 pi r^3 rho0 to the cell-discretisation level
    long ncell = 0;
    for (int i = 0; i < 96; ++i) for (int j = 0; j < 96; ++j) for (int k = 0; k < 80; ++k) {
      const Real x = (i + 0.5 - 48) * dxf, y = (j + 0.5 - 48) * dxf, z = (k + 0.5 - 40) * dxf;
      if ((x + 3) * (x + 3) + y * y + z * z < rs * rs) ++ncell;
    }
    const double B_sph_cells = ncell * rho0 * dxf * dxf * dxf;
    const double B_sph_analytic = 4.0 / 3.0 * kPi * rs * rs * rs * rho0;
    const bool first_is_sphere = c.baryon[0] > c.baryon[1] ? B_sph_cells > B_gauss : B_sph_cells < B_gauss;
    (void)first_is_sphere;
    // identify by cell count: the sphere is the clump with ncell cells
    const int is = c.cells[0] == ncell ? 0 : 1;
    REQUIRE(c.cells[is] == ncell);
    printf("[T6.3] sphere: finder %.8g, planted cells %.8g, analytic 4/3 pi r^3 rho0 %.8g (rel %.2e); Gaussian: finder %.8g, analytic %.8g (rel %.2e)\n",
           c.baryon[is], B_sph_cells, B_sph_analytic, c.baryon[is] / B_sph_analytic - 1.0, c.baryon[1 - is], B_gauss,
           c.baryon[1 - is] / B_gauss - 1.0);
    CHECK(std::fabs(c.baryon[is] / B_sph_cells - 1.0) <= 1e-3);
    // The continuum volume 4/3 pi r^3 rho0 differs by 1.35e-2 (printed above): lattice-point count of the sphere. Information only, not an acceptance criterion.
    CHECK(std::fabs(c.baryon[1 - is] / B_gauss - 1.0) <= 1e-3);
  }
  SUBCASE("periodic wrap") {
    // cuboid covering x cells {46, 47, 0, 1}: one clump when the box wraps, two when it does not
    const ObsFields f = make_fields(g, [&](int i, int j, int k, Real, Real, Real, Cell& c) {
      if ((i >= 46 || i < 2) && j >= 10 && j < 14 && k >= 10 && k < 14) { c.R = rho0; c.rho = rho0; }
    });
    const double full = 4 * 4 * 4 * rho0 * dV;
    ObsParams p = default_params();
    p.periodic = true;
    const ClumpResult cw = find_clumps(f, p);
    p.periodic = false;
    const ClumpResult co = find_clumps(f, p);
    printf("[T6.3] periodic wrap: wrapping %zu clump(s) B = %.6g (planted %.6g); open %zu clumps\n", cw.baryon.size(), cw.baryon.empty() ? 0.0 : cw.baryon[0], full, co.baryon.size());
    REQUIRE(cw.baryon.size() == 1);
    CHECK(std::fabs(cw.baryon[0] / full - 1.0) <= 1e-3);
    CHECK(co.baryon.size() == 2);
    // diagonal contact across the periodic face: a 26-connected merge only
    const ObsFields f2 = make_fields(g, [&](int i, int j, int k, Real, Real, Real, Cell& c) {
      if ((i == 47 && j == 10 && k == 10) || (i == 0 && j == 11 && k == 10)) { c.R = rho0; c.rho = rho0; }
    });
    p.periodic = true; p.connectivity = 6;
    CHECK(find_clumps(f2, p).baryon.size() == 2);
    p.connectivity = 26;
    CHECK(find_clumps(f2, p).baryon.size() == 1);
  }
  SUBCASE("clump histogram bins") {
    long over = 0;
    const std::vector<long> h = clump_histogram({0.2, 0.6, 1.4, 3.49, 3.5, 25.4, 25.6, 40.0}, 25, over);
    CHECK(h[0] == 1); CHECK(h[1] == 2); CHECK(h[3] == 1); CHECK(h[4] == 1); CHECK(h[25] == 1);
    CHECK(over == 2);
  }
}

TEST_CASE("T6.4 Hubble-like transverse flow on a Gaussian") {
  const Real dx = 0.25, sigma = 0.75, H = 0.04;  // |v| <= H * 10 sqrt2 < 0.6 would be too much: box limited below
  const Grid g = Grid::make_3d(64, 64, 32, dx);  // 16 x 16 x 8 fm; |x|,|y| <= 8 -> |v| <= 0.32 per component
  const ObsFields f = make_fields(g, [&](int, int, int, Real x, Real y, Real z, Cell& c) {
    const Real d = gauss3(x * x + y * y + z * z, 5.0, sigma);
    c.R = d; c.rho = d * 0.9; c.vx = H * x; c.vy = H * y;
  });
  Real vmax = 0;
  for (Real x : {-8.0, 8.0}) vmax = std::max(vmax, std::sqrt(2.0) * H * std::fabs(x));
  REQUIRE(vmax < 0.5);
  const Real expected = H * std::sqrt(2.0) * sigma;  // U = H Q, Q^2 = <x^2 + y^2> = 2 sigma^2
  for (int rest = 0; rest < 2; ++rest) {
    ObsParams p = default_params();
    p.rest_frame = rest == 1;
    const ObsResult r = observe(f, p);
    printf("[T6.4] rest=%d U_perp = %.12g, analytic H sqrt2 sigma = %.12g, rel dev %.2e\n", rest, r.u_perp, expected, r.u_perp / expected - 1.0);
    CHECK(std::fabs(r.u_perp / expected - 1.0) <= 1e-6);
  }
}

namespace {
// device evaluation of the cold curve (rule 9: the host does not call equation-of-state members)
Real eps_cold_device(const EosColdThermal& eos, Real rho) {
  Kokkos::View<Real*> out("eps_c", 1);
  Kokkos::parallel_for("eps_cold", 1, KOKKOS_LAMBDA(const int) { out(0) = eos.eps_cold(rho); });
  auto h = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(h, out);
  return h(0);
}
}  // namespace

TEST_CASE("T6.5 phase-plane histogram and unstable fraction") {
  const Grid g = Grid::make_3d(20, 20, 20, 0.5);
  SUBCASE("conservation of baryon number and bin placement") {
    ObsParams p = default_params();
    const ObsFields f = make_fields(g, [&](int i, int j, int k, Real x, Real y, Real z, Cell& c) {
      c.rho = kRhoS * (0.5 + 5.0 * std::fabs(std::sin(0.3 * i + 0.2 * j))) ;
      c.R = c.rho * 1.1;
      c.eps = kEpsS * (1.0 + 20.0 * std::fabs(std::cos(0.1 * k + 0.4 * j)));
      (void)x; (void)y; (void)z;
    });
    const ObsResult r = observe(f, p);
    const PhaseHist h = compute_phase_plane(f, p, NeverUnstable{});
    const double rel = std::fabs(h.total() / r.baryon_number - 1.0);
    printf("[T6.5] histogram total %.15g vs baryon number %.15g (rel %.2e), outside %.3g\n", h.total(), r.baryon_number, rel, h.outside);
    CHECK(rel <= 1e-12);
    CHECK(h.outside == 0.0);
    // range overflow is accounted for in 'outside'
    ObsParams q = p;
    q.n_rho = 20;  // 0 .. 2 rho_s only
    const PhaseHist hq = compute_phase_plane(f, q, NeverUnstable{});
    CHECK(hq.outside > 0.0);
    CHECK(std::fabs(hq.total() / r.baryon_number - 1.0) <= 1e-12);
    // one planted cell
    const ObsFields one = make_fields(g, [&](int i, int j, int k, Real, Real, Real, Cell& c) {
      if (i == 3 && j == 4 && k == 5) { c.rho = 3.05 * kRhoS; c.R = 3.05 * kRhoS; c.eps = 7.1 * kEpsS; }
    });
    const PhaseHist h1 = compute_phase_plane(one, p, NeverUnstable{});
    const int ir = 30, ie = 28;  // 3.05 / 0.1 = 30.5, 7.1 / 0.25 = 28.4
    CHECK(std::fabs(h1.baryon[static_cast<size_t>(ir) * h1.n_eps + ie] - 3.05 * kRhoS * 0.125) <= 1e-14);
    CHECK(h1.total() == doctest::Approx(3.05 * kRhoS * 0.125).epsilon(1e-14));
  }
  SUBCASE("unstable fraction of the cold-thermal two-phase EoS") {
    const ColdCurveTable table = ColdCurveTable::load(kSrcDir + "/data/eos/cold_curve.dat");
    const EosColdThermal eos(table, ColdVariant::two_phase);
    ObsParams p = default_params();
    Real fr[2];
    int q = 0;
    for (Real c : {5.8, 2.0}) {
      const Real ec = eps_cold_device(eos, c * kRhoS);
      const ObsFields f = make_fields(g, [&](int, int, int, Real, Real, Real, Cell& cell) { cell.R = c * kRhoS; cell.rho = c * kRhoS; cell.eps = ec + 2.0 * kEpsS; });
      const ObsResult r = compute_observables(f, p, Vs2Mask<EosColdThermal>{eos}, 0.0, 0);
      fr[q++] = r.unstable_fraction;
      printf("[T6.5] rho = %.1f rho_s, eps = eps_c + 2 eps_s = %.6g GeV/fm^3: unstable fraction %.17g\n", c, ec + 2 * kEpsS, r.unstable_fraction);
    }
    CHECK(fr[0] == 1.0);
    CHECK(fr[1] == 0.0);
  }
}

// ------------------------------------------------------------------------------------------------ T6.6
namespace {

std::vector<std::vector<double>> read_csv(const std::string& path, std::vector<std::string>& header) {
  std::ifstream in(path);
  REQUIRE(bool(in));
  std::string line;
  std::getline(in, line);
  header.clear();
  { std::stringstream ss(line); std::string tok; while (std::getline(ss, tok, ',')) header.push_back(tok); }
  std::vector<std::vector<double>> rows;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    std::string tok;
    std::vector<double> r;
    while (std::getline(ss, tok, ',')) r.push_back(std::stod(tok));
    rows.push_back(r);
  }
  return rows;
}

nlohmann::json read_json(const std::string& path) {
  std::ifstream in(path);
  REQUIRE(bool(in));
  return nlohmann::json::parse(in);
}

Real max_rel_csv_dev(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b) {
  REQUIRE(a.size() == b.size());
  Real w = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    REQUIRE(a[i].size() == b[i].size());
    for (size_t j = 0; j < a[i].size(); ++j) {
      const Real m = std::max(std::fabs(a[i][j]), std::fabs(b[i][j]));
      if (m > 0) w = std::max(w, std::fabs(a[i][j] - b[i][j]) / m);
    }
  }
  return w;
}

}  // namespace

TEST_CASE("T6.6 recorder, post-processing, end of the fluid stage, ensemble script") {
  const int n = 48;
  const long steps_total = 60, every = 5;
  const std::string root = "obs_t66";
  std::filesystem::remove_all(root);
  Config cfg = default_config();
  cfg.set_from_string("eos.model=ideal");
  cfg.set_from_string("obs.every_steps=" + std::to_string(every));
  cfg.set_from_string("obs.hist_times_fm=[0.0, 0.4, 100.0]");  // the last time is never reached
  Hydro3DParams hp = Hydro3DParams::from_config(cfg);
  hp.nx = hp.ny = hp.nz = n;
  const Real dx = hp.dx;
  Hydro3D<EosIdeal> h(EosIdeal{}, hp);
  h.set_state_fn([&](int ix, int iy, int iz, Real& e, Real& r, Real& ux, Real& uy, Real& uz) {
    const Real x = (ix + 0.5 - 0.5 * n) * dx, y = (iy + 0.5 - 0.5 * n) * dx, z = (iz + 0.5 - 0.5 * n) * dx;
    const Real g2 = std::exp(-0.5 * (x * x + y * y + z * z));  // sigma = 1 fm
    e = (0.5 + 8.0 * g2) * kEpsS; r = 8.0 * kRhoS * g2;  // background 0.5 eps_s: the solver fails to recover cells next to near-vacuum ux = uy = uz = 0.0;
  });
  ObservableRecorder<EosIdeal> rec(cfg, EosIdeal{}, root + "/inrun");
  const std::vector<std::string> fields = {"R", "rho", "eps", "vx", "vy"};
  std::vector<std::string> snaps;
  for (long s = 0; s <= steps_total; ++s) {
    if (s > 0) h.step();
    rec.on_step(h);
    if (h.steps % every == 0) {
      const std::string b = root + "/snap_" + std::to_string(h.steps);
      h.write_snapshot(b, fields, cfg.json());
      snaps.push_back(b);
    }
  }
  rec.finalize();
  CHECK(h.n_recover_fail_total == 0);

  std::vector<std::string> hd;
  const auto rows = read_csv(root + "/inrun/observables.csv", hd);
  const std::vector<std::string> expect_hd = {"t_fm", "A", "mom2", "mom3", "mom4", "mom5", "mom6", "mom7", "U_perp", "unstable_fraction", "eps_max_eps_s", "step"};
  CHECK(hd == expect_hd);
  REQUIRE(rows.size() == static_cast<size_t>(steps_total / every + 1));
  Real tdev = 0;
  for (size_t i = 0; i < rows.size(); ++i) tdev = std::max(tdev, std::fabs(rows[i][0] - static_cast<Real>(i * every) * hp.shasta.lambda * dx));
  printf("[T6.6] CSV: %zu rows, columns OK, max |t - step dt| = %.2e fm; A(0) = %.10g, A(end) = %.10g; moments N=2..7 at t0: %.4f %.4f %.4f %.4f %.4f %.4f\n",
         rows.size(), tdev, rows.front()[1], rows.back()[1], rows[0][2], rows[0][3], rows[0][4], rows[0][5], rows[0][6], rows[0][7]);
  CHECK(tdev <= 1e-12);
  CHECK(rows.front()[0] == 0.0);
  CHECK(rows.front()[11] == 0.0);
  CHECK(std::fabs(rows.front()[8]) <= 1e-12);  // at rest
  CHECK(rows[3][8] > 0.0);                      // the blob expands: positive transverse flow (before the periodic images meet)
  CHECK(rows[3][2] < rows.front()[2]);          // moments decrease as the blob expands
  for (const auto& r : rows) CHECK(r[9] == 0.0);  // ideal EoS: nothing is unstable

  // end of the fluid stage
  size_t first = rows.size();
  for (size_t i = 0; i < rows.size(); ++i) if (rows[i][10] < 5.0) { first = i; break; }
  REQUIRE(first < rows.size());
  REQUIRE(first > 0);
  CHECK(rows[first - 1][10] >= 5.0);
  CHECK(rec.fluid_end_time() == rows[first][0]);
  printf("[T6.6] fluid end: first row with eps_max < 5 eps_s is step %ld, t = %.6g fm (eps_max/eps_s %.6g; previous %.6g)\n", static_cast<long>(rows[first][11]),
         rows[first][0], rows[first][10], rows[first - 1][10]);
  const nlohmann::json sum = read_json(root + "/inrun/observables_summary.json");
  CHECK(sum.at("fluid_end_time_fm").get<Real>() == rows[first][0]);
  CHECK(sum.at("max_enhancement").at("N7").at("value").get<Real>() >= 1.0);

  // post-processing with the library and with the application
  observe_snapshots(cfg, EosIdeal{}, snaps, root + "/post");
  std::vector<std::string> hd2;
  const auto rows2 = read_csv(root + "/post/observables.csv", hd2);
  const Real dev_lib = max_rel_csv_dev(rows, rows2);
  printf("[T6.6] post-processing (library) vs in-run: max relative deviation over all CSV entries = %.3e\n", dev_lib);
  CHECK(dev_lib <= 1e-12);
  std::string cmd = std::string("\"") + SPF_OBSERVE_EXE + "\" --out " + root + "/app";
  for (const auto& s : snaps) cmd += " " + s;
  cmd += " > " + root + "/app.log 2>&1";
  const int rc = std::system(cmd.c_str());
  REQUIRE(rc == 0);
  std::vector<std::string> hd3;
  const auto rows3 = read_csv(root + "/app/observables.csv", hd3);
  const Real dev_app = max_rel_csv_dev(rows, rows3);
  printf("[T6.6] spf_observe (executable) vs in-run: max relative deviation = %.3e\n", dev_app);
  CHECK(dev_app <= 1e-12);
  CHECK(read_json(root + "/app/observables_summary.json").at("fluid_end_time_fm") == sum.at("fluid_end_time_fm"));

  // histograms written at the hist times: step of the first record >= 2.4 fm is the same in-run and post-processed
  int n_hist = 0;
  for (const auto& e : std::filesystem::directory_iterator(root + "/inrun"))
    if (e.path().filename().string().rfind("clumps_step", 0) == 0) {
      ++n_hist;
      const std::string nm = e.path().filename().string();
      const nlohmann::json a = read_json(root + "/inrun/" + nm), b = read_json(root + "/app/" + nm);
      CHECK(a.at("histogram") == b.at("histogram"));
      CHECK(a.at("baryon_per_clump") == b.at("baryon_per_clump"));
      const std::string pn = "phase_" + nm.substr(7);
      const nlohmann::json pa = read_json(root + "/inrun/" + pn), pb = read_json(root + "/app/" + pn);
      Real w = 0;
      for (size_t i = 0; i < pa.at("baryon").size(); ++i) w = std::max(w, std::fabs(pa.at("baryon")[i].get<Real>() - pb.at("baryon")[i].get<Real>()));
      CHECK(w <= 1e-14 * pa.at("total").get<Real>());
      size_t row = 0;
      for (size_t q = 0; q < rows.size(); ++q) if (static_cast<long>(rows[q][11]) == a.at("step").get<long>()) row = q;
      CHECK(std::fabs(pa.at("total").get<Real>() / rows[row][1] - 1.0) <= 1e-12);  // total baryon number = A of that record
      printf("[T6.6] %s: n_clumps %d, histogram equal in-run/post; phase-plane max |diff| %.2e\n", nm.c_str(), a.at("n_clumps").get<int>(), w);
    }
  CHECK(n_hist == 2);  // t = 0.0 and 0.4 fm; the third time is never reached

  // ensemble script on three synthetic events with known maximum enhancements 2, 4, 9
#ifdef SPF_PYTHON
  {
    const std::vector<Real> peak = {2.0, 4.0, 9.0};
    std::string dirs;
    for (size_t e = 0; e < peak.size(); ++e) {
      const std::string d = root + "/ens" + std::to_string(e);
      std::filesystem::create_directories(d);
      std::ofstream out(d + "/observables.csv");
      out << obs_csv_header() << "\n";
      for (int s = 0; s < 4; ++s) {
        ObsResult r;
        r.time = 1.0 + s;  // t_init = 1
        r.step = s;
        r.A = 100.0;
        const Real f = s == 0 ? 1.0 : (s == 2 ? peak[e] : 0.5 * (1.0 + peak[e]));
        for (int N = 2; N <= 7; ++N) r.moment[N] = N * f;  // reference moment at t_init is N
        out << obs_csv_line(r) << "\n";
      }
      dirs += " " + d;
    }
    const std::string c = std::string("\"") + SPF_PYTHON + "\" \"" + SPF_ENSEMBLE_SCRIPT + "\" --out " + root + "/ens_out" + dirs + " > " + root + "/ens.log 2>&1";
    REQUIRE(std::system(c.c_str()) == 0);
    const nlohmann::json j = read_json(root + "/ens_out/ensemble_summary.json");
    const Real mean = 5.0, stderr_ = std::sqrt(13.0) / std::sqrt(3.0), spread = 9.0 / 5.0;
    Real w = 0;
    for (int N = 2; N <= 7; ++N) {
      const auto& m = j.at("max_enhancement").at("N" + std::to_string(N));
      w = std::max({w, std::fabs(m.at("mean").get<Real>() - mean), std::fabs(m.at("stderr").get<Real>() - stderr_), std::fabs(m.at("spread_max_over_mean").get<Real>() - spread)});
    }
    printf("[T6.6] ensemble script: mean 5, stderr %.10g, spread 1.8 -> max abs deviation %.2e\n", stderr_, w);
    CHECK(w <= 1e-12);
    CHECK(j.at("n_events").get<int>() == 3);
  }
#else
  MESSAGE("no Python interpreter found at configure time: ensemble script not tested");
#endif
}
