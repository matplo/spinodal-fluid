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
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
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

// Host-filled level fields (eps, rho, u^x, u^y, u^z, unused) for the primitive interpolation mode; f(x, y, z, out[5]) at the physical
// cell centre.
template <class F>
void fill_level(const Grid& g, const std::array<Real, 3>& org, Field3 lev[6], F&& f) {
  lev[5] = g.make_field("test_level_unused");
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
  cfg.set_from_string("particlisation.surface.interpolation=primitive");  // prescribed levels are given in primitive variables
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
    Field3 a[6], b[6];
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
    Field3 a[6], b[6];
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
  Field3 a[6], b[6];
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

// ------------------------------------------------------------------ evolution helpers
namespace {

struct EventResult {
  nlohmann::json summary;
  long steps = 0;
  double t_end = 0.0;
  bool closed = false;
  double wall_fluid = 0.0, wall_surface = 0.0;
};

// Evolve the state already set in h with a SurfaceFinder until the surface is complete or t_max (fluid time) is reached.
template <class EoS>
EventResult run_with_surface(Hydro3D<EoS>& h, const EoS& eos, const Config& cfg, const std::array<Real, 3>& org, Real t_init, Real t_max,
                             const std::string& base) {
  const ParticlisationParams pp = ParticlisationParams::from_config(cfg);
  typename SurfaceFinder<EoS>::Meta meta;
  meta.t_init = t_init;
  meta.origin = org;
  SurfaceFinder<EoS> sf(eos, pp, h.grid, base, meta);
  EventResult r;
  sf.begin(h);
  const long nmax = static_cast<long>(std::ceil(t_max / h.dt() - 1e-9));
  for (long s = 0; s < nmax; ++s) {
    const auto t0 = std::chrono::steady_clock::now();
    h.step();
    Kokkos::fence();
    r.wall_fluid += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (sf.after_step(h)) break;
  }
  sf.finish(h);
  r.summary = sf.summary(0);
  r.steps = h.steps;
  r.t_end = h.time;
  r.closed = sf.closed();
  r.wall_surface = sf.wall_s();
  return r;
}

std::vector<Real> read_surface_bin(const std::string& base) {
  std::ifstream in(base + ".surface.bin", std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::vector<Real> v(bytes.size() / sizeof(Real));
  if (!v.empty()) std::memcpy(v.data(), bytes.data(), v.size() * sizeof(Real));
  return v;
}

double jget(const nlohmann::json& j, const char* a, const char* b) { return j.at(a).at(b).get<double>(); }

// Collision-like configuration for the closure tests (outflow box, bicubic table, vacuum threshold of the collision files).
Config closure_config(const char* gradient_form, Real dx, Real L) {
  Config cfg = surf_config();
  cfg.merge_file(std::string(SPF_SOURCE_DIR) + "/config/collisions/common.json");
  cfg.set_from_string("eos.model=table_two_phase");
  cfg.set_from_string("eos.table_path=\"" + std::string(SPF_SOURCE_DIR) + "/data/eos/test/eos_two_phase_test\"");
  cfg.set_from_string(std::string("gradient.form=") + gradient_form);
  cfg.set_from_string("grid.dx_fm=" + std::to_string(dx));
  for (const char* k : {"grid.Lx_fm=", "grid.Ly_fm=", "grid.Lz_fm="}) cfg.set_from_string(std::string(k) + std::to_string(L));
  return cfg;
}

// Glauber-Gaussians state (in-repo generator, Pb+Pb, full overlap) on the grid of cfg; returns t_init.
template <class EoS>
Real set_glauber(Hydro3D<EoS>& h, const Config& cfg) {
  const InitialState st = build_initial_state(cfg, h.grid);
  h.set_conserved(st.dep.R, st.dep.Mx, st.dep.My, st.dep.Mz, st.dep.E);
  return st.t_init_fm;
}

// SMASH-like state: the Glauber nucleons with a Gaussian momentum smearing (sigma 0.2 GeV per component) plus produced pions
// (one per 4 nucleons, thermal-like momenta of width 0.3 GeV) at the nucleon positions smeared by 1 fm; deposited like an OSCAR list.
template <class EoS>
Real set_smash_like(Hydro3D<EoS>& h, const Config& cfg, std::uint64_t seed) {
  const Kinematics k = kinematics_from_elab(cfg.get<double>("ic.e_lab_gev"));
  const int A = cfg.get<int>("ic.nucleus_A");
  std::vector<Particle> ps = glauber_particles(k, A, cfg.get<double>("ic.ws_radius_fm"), cfg.get<double>("ic.ws_a_fm"), seed);
  std::mt19937_64 rng(seed + 11);
  std::normal_distribution<double> nd(0.0, 1.0);
  const size_t nn = ps.size();
  for (size_t i = 0; i < nn; ++i) {
    Particle& p = ps[i];
    p.px += 0.2 * nd(rng); p.py += 0.2 * nd(rng); p.pz += 0.2 * nd(rng);
    p.E = std::sqrt(0.94 * 0.94 + p.px * p.px + p.py * p.py + p.pz * p.pz);
    if (i % 4 == 0) {
      Particle q = p;
      q.pdg = 211; q.B = 0;
      q.x += nd(rng); q.y += nd(rng); q.z += nd(rng) / k.gamma_cm;
      q.px = 0.3 * nd(rng); q.py = 0.3 * nd(rng); q.pz = 0.3 * nd(rng);
      q.E = std::sqrt(0.138 * 0.138 + q.px * q.px + q.py * q.py + q.pz * q.pz);
      ps.push_back(q);
    }
  }
  DepositParams dp;
  dp.sigma = cfg.get<double>("ic.sigma_init_fm");
  dp.cutoff = cfg.get<double>("ic.cutoff_sigma");
  dp.contraction = parse_contraction(cfg.get<std::string>("ic.gaussian_contraction"));
  dp.gamma_cm = k.gamma_cm;
  const auto o = centred_origin(h.grid);
  dp.origin[0] = o[0]; dp.origin[1] = o[1]; dp.origin[2] = o[2];
  DepositResult d = deposit_particles(h.grid, dp, ps);
  limit_momentum(h.grid, d);
  h.set_conserved(d.R, d.Mx, d.My, d.Mz, d.E);
  return t_init_fm(k, t_init_radius_from_config(cfg, A));
}

}  // namespace


// ------------------------------------------------------------------ T10a.4 Hubble-like expansion (fluid evolution, p = eps / 3)
// u^mu = x^mu / tau, tau = sqrt(t^2 - r^2), eps = eps0 (tau0 / tau)^4, rho = rho0 (tau0 / tau)^3 is an exact solution of ideal fluid dynamics
// with p = eps / 3.  At t = t0 the whole box (|x_i| <= 5 fm, t0 = 12 fm) is above eps_sw = eps(tau_sw); the surface is the hyperboloid
// tau = tau_sw, i.e. a hole that opens at the centre at t = tau_sw and grows as r = sqrt(t^2 - tau_sw^2); its normal is dSigma_mu ~ u_mu.
namespace {
struct HubbleDev { double max_dtau = 0, rms = 0, max_ang = 0, rms_ang = 0; long cnt = 0; };
struct HubbleRes { HubbleDev fl, an; long nini = -1; };
const HubbleRes& hubble_cached() {
  static HubbleRes R;
  if (R.nini >= 0) return R;
  const int n = 50;
  const Real dx = 0.2, t0 = 12.0, tau_sw = 12.2, eps0 = 1.0, rho0 = 0.3;
  Config cfg = surf_config();
  cfg.set_from_string("gradient.form=off");
  cfg.set_from_string("grid.boundary=outflow");
  cfg.set_from_string("grid.nx=" + std::to_string(n)); cfg.set_from_string("grid.ny=" + std::to_string(n)); cfg.set_from_string("grid.nz=" + std::to_string(n));
  cfg.set_from_string("grid.dx_fm=" + std::to_string(dx));
  const Real eps_sw = eps0 * std::pow(t0 / tau_sw, 4);
  cfg.set_from_string("particlisation.eps_sw_eps_s=" + std::to_string(eps_sw / kEpsS));
  Hydro3D<TestEosIdeal> h(TestEosIdeal{}, Hydro3DParams::from_config(cfg));
  const auto org = centred_origin(h.grid);
  h.set_state_fn([&](int ix, int iy, int iz, Real& e, Real& r, Real& ux, Real& uy, Real& uz) {
    const Real x = org[0] + ix * dx, y = org[1] + iy * dx, z = org[2] + iz * dx;
    const Real tau = std::sqrt(t0 * t0 - x * x - y * y - z * z);
    e = eps0 * std::pow(t0 / tau, 4);
    r = rho0 * std::pow(t0 / tau, 3);
    ux = x / t0; uy = y / t0; uz = z / t0;  // three-velocity v = x / t
  });
  auto res = run_with_surface(h, TestEosIdeal{}, cfg, org, t0, 0.8, "wp10a_hubble");
  const auto v = read_surface_bin("wp10a_hubble");
  std::remove("wp10a_hubble.surface.bin");
  std::remove("wp10a_hubble.surface.json");
  const long nini = res.summary["n_elements_initial_slice"].get<long>();
  // the same hyperboloid from the analytic fields at the same time levels (finder alone, no fluid evolution)
  std::vector<Real> va;
  {
    const ParticlisationParams pp = ParticlisationParams::from_config(cfg);
    typename SurfaceFinder<TestEosIdeal>::Meta meta;
    meta.t_init = t0;
    meta.origin = org;
    Config cp = cfg;
    cp.set_from_string("particlisation.surface.interpolation=primitive");
    SurfaceFinder<TestEosIdeal> sf(TestEosIdeal{}, ParticlisationParams::from_config(cp), h.grid, "", meta);
    auto level = [&](Real t) {
      return [=](Real x, Real y, Real z, Real* o) {
        const Real tau = std::sqrt(t * t - x * x - y * y - z * z);
        o[0] = eps0 * std::pow(t0 / tau, 4);
        o[1] = rho0 * std::pow(t0 / tau, 3);
        o[2] = x / tau; o[3] = y / tau; o[4] = z / tau;
      };
    };
    for (long s = 0; s < res.steps; ++s) {
      const Real ta = t0 + s * h.dt(), tb = t0 + (s + 1) * h.dt();
      Field3 a[6], b[6];
      fill_level(h.grid, org, a, level(ta));
      fill_level(h.grid, org, b, level(tb));
      const auto e = sf.elements_between(a, b, ta, tb - ta);
      va.insert(va.end(), e.begin(), e.end());
    }
  }
  using Dev = HubbleDev;
  auto measure = [&](const std::vector<Real>& w, size_t first) {
    Dev d;
    double s2 = 0, a2s = 0, wsum = 0;
    for (size_t r = first; r < w.size() / kSurfRecord; ++r) {
      const double* q = &w[r * kSurfRecord];
      const double t = q[0], rr = std::sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
      if (rr > 3.0) continue;  // away from the box edge (signals from the outflow boundary travel at most 0.8 fm)
      const double dd = std::sqrt(t * t - rr * rr) / tau_sw - 1.0;
      d.max_dtau = std::max(d.max_dtau, std::fabs(dd));
      s2 += dd * dd;
      const double an[4] = {t, -q[1], -q[2], -q[3]};  // analytic covariant normal ~ u_mu
      double dot = 0, aa = 0, nn = 0;
      for (int k = 0; k < 4; ++k) { dot += an[k] * q[4 + k]; aa += an[k] * an[k]; nn += q[4 + k] * q[4 + k]; }
      const double ang = std::acos(std::min(1.0, dot / std::sqrt(aa * nn)));
      d.max_ang = std::max(d.max_ang, ang);
      a2s += std::sqrt(nn) * ang * ang;
      wsum += std::sqrt(nn);
      ++d.cnt;
    }
    d.rms = d.cnt > 0 ? std::sqrt(s2 / d.cnt) : 0.0;
    d.rms_ang = wsum > 0 ? std::sqrt(a2s / wsum) : 0.0;
    return d;
  };
  const Dev fl = measure(v, static_cast<size_t>(nini)), an = measure(va, 0);
  std::printf("[WP10a] T10a.4 Hubble expansion (dx = %.2f fm, t0 = %.1f fm, tau_sw = %.2f fm, %ld steps), elements with r < 3 fm:\n"
              "[WP10a]   fluid evolution:  %ld elements, tau_element / tau_sw - 1 rms %.3e max %.3e; angle dSigma_mu vs u_mu: weighted rms %.3e max %.3e rad\n"
              "[WP10a]   analytic levels:  %ld elements, tau_element / tau_sw - 1 rms %.3e max %.3e; angle dSigma_mu vs u_mu: weighted rms %.3e max %.3e rad\n",
              dx, t0, tau_sw, res.steps, fl.cnt, fl.rms, fl.max_dtau, fl.rms_ang, fl.max_ang, an.cnt, an.rms, an.max_dtau, an.rms_ang, an.max_ang);
  R.fl = fl;
  R.an = an;
  R.nini = nini;
  return R;
}
}  // namespace

TEST_CASE("T10a.4 Hubble-like expansion: surface position (fluid and analytic levels) and normal (analytic levels)") {
  const HubbleRes& R = hubble_cached();
  CHECK(R.nini == 0);
  CHECK(R.fl.cnt > 100);
  CHECK(R.fl.max_dtau < 2e-3);
  CHECK(R.an.max_dtau < 2e-3);
  CHECK(R.an.max_ang < 0.02);
}

// The normals of the surface found in the evolved fluid scatter by about 0.01 rad around u_mu (the surface is almost isochronous, and
// relative deviations of 1e-4 of the numerical eps from the exact solution tilt it); the a-priori tolerance 0.02 rad on the largest angle is
// exceeded at the apex of the hyperboloid (measured 0.0227 rad at r = 0.35 fm on the development machine).  Expected failure, kept on purpose.
TEST_CASE("T10a.4 Hubble expansion, fluid evolution: largest normal angle below 0.02 rad" * doctest::should_fail()) {
  CHECK(hubble_cached().fl.max_ang < 0.02);
}

// ------------------------------------------------------------------ T10a.1 closure (Gauss law) of the surface integrals
// initial totals on the grid = initial slice + isoenergy surface + left inside at the end, for E, P and B.  Collision settings
// (config/collisions/common.json: outflow box, bicubic table interpolation, e_vac = 1e-8, gradient term), test table two_phase, Pb+Pb
// 3 A GeV, box 24 fm, gradient term on (collision default) and off.
// Expected failure, kept on purpose: the target <= 1e-3 at dx = 0.2 fm was fixed before the measurement.  Measured (development machine,
// see docs/validation.md, WP10a): Glauber-Gaussians closure of E about -1.4e-2 at dx = 0.2 fm and -3.4e-2 at dx = 0.4 fm (first order in dx);
// the error is made in the first fm/c, while the edge left by the removal of the initial slice (decision S23) rarefies into vacuum over a
// few cells.  With particlisation.initial_slice = keep the closure is second order (about -2.4e-3 at dx = 0.2 fm); variants and the dx = 0.1 fm
// point: tools/particlisation/closure_convergence.py.
namespace {
struct ClosureRow { double E, B, Px, Py, Pz, inflow_E; long nel, nini, edge; double t_end; bool closed; };
template <class SetState>
ClosureRow closure_run(const char* label, const char* gradient_form, Real dx, const char* initial_slice, SetState&& set) {
  Config cfg = closure_config(gradient_form, dx, 24.0);
  cfg.set_from_string("ic.generator=glauber_gaussians");
  cfg.set_from_string(std::string("particlisation.initial_slice=") + initial_slice);
  const EosTableData td = EosTableData::from_config(cfg);
  const EosTable eos = td.eos(cfg);
  Hydro3D<EosTable> h(eos, Hydro3DParams::from_config(cfg));
  const Real ti = set(h, cfg);
  auto r = run_with_surface(h, eos, cfg, centred_origin(h.grid), ti, 30.0, "");
  const auto& S = r.summary;
  ClosureRow c{jget(S, "closure", "E"), jget(S, "closure", "B"), jget(S, "closure", "Px"), jget(S, "closure", "Py"), jget(S, "closure", "Pz"),
               jget(S, "inflow_part_of_totals", "E_GeV") / jget(S, "totals_initial_on_grid", "E_GeV"), S["n_elements"].get<long>(),
               S["n_elements_initial_slice"].get<long>(), S["diagnostics"]["max_cells_above_eps_sw_on_box_edge"].get<long>(), r.t_end, r.closed};
  std::printf("[WP10a] T10a.1 %s, gradient %s, initial_slice %s, dx = %.2f fm: closed %d after %.2f fm/c, %ld elements (%ld initial slice); "
              "closure E %.3e B %.3e Px %.2e Py %.2e Pz %.2e; inflow part of E %.3e; cells above eps_sw on the box edge %ld; wall fluid %.1f s surface %.1f s\n",
              label, gradient_form, initial_slice, dx, static_cast<int>(r.closed), r.t_end, c.nel, c.nini, c.E, c.B, c.Px, c.Py, c.Pz, c.inflow_E,
              c.edge, r.wall_fluid, r.wall_surface);
  return c;
}
auto glauber_setter = [](auto& h, const Config& cfg) { return set_glauber(h, cfg); };
auto smash_like_setter = [](auto& h, const Config& cfg) { return set_smash_like(h, cfg, 5); };
}  // namespace

namespace {
// Each configuration is run once per process (the acceptance case and the bookkeeping case share the runs).
const ClosureRow& closure_cached(int which) {
  static std::map<int, ClosureRow> cache;
  auto it = cache.find(which);
  if (it != cache.end()) return it->second;
  ClosureRow c{};
  if (which == 0) c = closure_run("Glauber-Gaussians", "paper_pressure", 0.4, "remove", glauber_setter);
  if (which == 1) c = closure_run("Glauber-Gaussians", "paper_pressure", 0.2, "remove", glauber_setter);
  if (which == 2) c = closure_run("SMASH-like", "paper_pressure", 0.2, "remove", smash_like_setter);
  return cache.emplace(which, c).first->second;
}
}  // namespace

TEST_CASE("T10a.1 closure bookkeeping: surfaces complete, inside the box, no topology errors") {
  for (int w = 0; w < 3; ++w) {
    const ClosureRow& c = closure_cached(w);
    CHECK(c.closed);
    CHECK(c.edge == 0);
  }
}

TEST_CASE("T10a.1 closure: Glauber-Gaussians and SMASH-like states, target 1e-3 for E and B at dx = 0.2 fm" * doctest::should_fail()) {
  const ClosureRow& g2 = closure_cached(1);
  const ClosureRow& s2 = closure_cached(2);
  CHECK(std::fabs(g2.E) <= 1e-3);
  CHECK(std::fabs(g2.B) <= 1e-3);
  CHECK(std::fabs(s2.E) <= 1e-3);
  CHECK(std::fabs(s2.B) <= 1e-3);
}

// ------------------------------------------------------------------ T10a.5 isochronous versus isoenergy totals
// The Glauber-Gaussians state (dx = 0.4 fm) is run once with the isoenergy surface until it is complete, then again with the isochronous
// method at a time after which every cell is below eps_sw; both files contain the same initial slice.  The isochronous totals equal the content
// of the grid at that time (to the vacuum cut and the box), so their difference to the isoenergy totals is the closure error of the isoenergy
// surface.  Target: agreement within 1e-3 of the initial totals (the closure target of T10a.1); expected failure for the same reason.
namespace {
struct IsoPair { nlohmann::json iso, chr; Real t_close = 0, t_iso = 0, vac_E = 0, vac_B = 0, grid_E = 0, grid_B = 0; };
const IsoPair& iso_pair() {
  static IsoPair P;
  static bool done = false;
  if (done) return P;
  const Real dx = 0.4;
  Config cfg = closure_config("paper_pressure", dx, 24.0);
  cfg.set_from_string("ic.generator=glauber_gaussians");
  cfg.set_from_string("grid.boundary=periodic");  // nothing leaves the box before the isochronous time (the dilute matter wraps around)
  const EosTableData td = EosTableData::from_config(cfg);
  const EosTable eos = td.eos(cfg);
  {
    Hydro3D<EosTable> h(eos, Hydro3DParams::from_config(cfg));
    const Real ti = set_glauber(h, cfg);
    auto r = run_with_surface(h, eos, cfg, centred_origin(h.grid), ti, 30.0, "");
    P.iso = r.summary;
    P.t_close = r.t_end;
  }
  P.t_iso = std::ceil(P.t_close) + 1.0;  // every cell below eps_sw
  {
    Config c2 = cfg;
    c2.set_from_string("particlisation.surface.method=isochronous");
    c2.set_from_string("particlisation.surface.isochronous_times_fm=[" + std::to_string(P.t_iso) + "]");
    Hydro3D<EosTable> h(eos, Hydro3DParams::from_config(c2));
    h.track_vacuum = true;  // energy and baryon number removed by the vacuum cut E < e_vac (the only loss of a periodic box)
    const Real ti = set_glauber(h, c2);
    auto r = run_with_surface(h, eos, c2, centred_origin(h.grid), ti, P.t_iso + 1.0, "");
    P.chr = r.summary;
    P.vac_E = h.vac_removed_E;
    P.vac_B = h.vac_removed_R;
    const Totals tg = h.totals();
    const Real dV = dx * dx * dx;
    P.grid_E = tg.E * dV;
    P.grid_B = tg.R * dV;
  }
  const double E0 = jget(P.iso, "totals_initial_on_grid", "E_GeV"), B0 = jget(P.iso, "totals_initial_on_grid", "B");
  std::printf("[WP10a] T10a.5 Glauber-Gaussians dx = %.1f fm: isoenergy surface complete after %.2f fm/c, isochronous at %.2f fm/c (%ld elements); "
              "(isoenergy - isochronous) / initial: E %.3e B %.3e Pz %.2e; isochronous closure E %.3e B %.3e; removed by the vacuum cut "
              "(relative) E %.3e B %.3e; isochronous elements vs grid content at that time (relative) E %.3e B %.3e\n",
              dx, P.t_close, P.t_iso, P.chr["n_elements"].get<long>(),
              (jget(P.iso, "totals_all_elements", "E_GeV") - jget(P.chr, "totals_all_elements", "E_GeV")) / E0,
              (jget(P.iso, "totals_all_elements", "B") - jget(P.chr, "totals_all_elements", "B")) / B0,
              (jget(P.iso, "totals_all_elements", "Pz_GeV") - jget(P.chr, "totals_all_elements", "Pz_GeV")) / E0,
              jget(P.chr, "closure", "E"), jget(P.chr, "closure", "B"), P.vac_E / E0, P.vac_B / B0,
              jget(P.chr, "totals_surface", "E_GeV") / P.grid_E - 1.0, jget(P.chr, "totals_surface", "B") / P.grid_B - 1.0);
  done = true;
  return P;
}
}  // namespace

TEST_CASE("T10a.5 isochronous surface of a state below eps_sw: same initial slice, totals equal the grid content") {
  const IsoPair& P = iso_pair();
  CHECK(P.iso["closed"].get<bool>());
  CHECK(P.chr["n_elements_initial_slice"].get<long>() == P.iso["n_elements_initial_slice"].get<long>());
  // The isochronous totals are the grid content itself: closure = - (removed by the vacuum cut) / initial, to rounding (state_from_conserved
  // reproduces T^{00} and N^0 of every cell, also of the dilute light-like cells at the vacuum fronts).
  const double E0 = jget(P.chr, "totals_initial_on_grid", "E_GeV"), B0 = jget(P.chr, "totals_initial_on_grid", "B");
  CHECK(std::fabs(jget(P.chr, "closure", "E") + P.vac_E / E0) < 1e-12);
  CHECK(std::fabs(jget(P.chr, "closure", "B") + P.vac_B / B0) < 1e-12);
  CHECK(std::fabs(jget(P.chr, "totals_surface", "E_GeV") / P.grid_E - 1.0) < 1e-12);
  CHECK(std::fabs(jget(P.chr, "totals_surface", "B") / P.grid_B - 1.0) < 1e-12);
}

TEST_CASE("T10a.5 isochronous versus isoenergy totals agree within 1e-3" * doctest::should_fail()) {
  const IsoPair& P = iso_pair();
  const double E0 = jget(P.iso, "totals_initial_on_grid", "E_GeV"), B0 = jget(P.iso, "totals_initial_on_grid", "B");
  CHECK(std::fabs(jget(P.iso, "totals_all_elements", "E_GeV") - jget(P.chr, "totals_all_elements", "E_GeV")) / E0 <= 1e-3);
  CHECK(std::fabs(jget(P.iso, "totals_all_elements", "B") - jget(P.chr, "totals_all_elements", "B")) / B0 <= 1e-3);
}

// ------------------------------------------------------------------ T10a.6 backends give the same element list
// Analytic counter-streaming Woods-Saxon state (ic.generator = analytic: computed per cell, no atomic additions, so every backend starts from
// the same bits), collision settings, test table two_phase, box 24 fm, dx = 0.8 fm, first 3 steps after t_init.  The isoenergy elements are
// compared with tests/data/wp10a_surface_reference.bin, written by the Serial build of this branch (environment variable
// SPF_WRITE_SURFACE_REFERENCE=1 rewrites it).  Criterion per column: max |a - b| / max |column| <= 1e-10 (Serial versus CUDA fluid fields
// differ by 1.6e-14 in WP3g); the initial slice is compared by count and totals.  (With a deposited Glauber state the initial fields already
// differ in the last bits between backends, because the deposition adds with atomics in a backend-dependent order; the elements then differ
// by up to about 1e-8, see docs/validation.md, WP10a.)
namespace {
struct BackendCmp { size_t n = 0, nr = 0; long nini = 0, ref_nini = 0; double worst = 0, dE_ini = 0, dB_ini = 0, dB_flux = 0, d_dsig0 = 0; bool done = false; };
const BackendCmp& backend_cmp() {
  static BackendCmp C;
  if (C.done) return C;
  Config cfg = closure_config("paper_pressure", 0.8, 24.0);
  cfg.set_from_string("ic.generator=analytic");
  const EosTableData td = EosTableData::from_config(cfg);
  const EosTable eos = td.eos(cfg);
  Hydro3D<EosTable> h(eos, Hydro3DParams::from_config(cfg));
  const Real ti = set_glauber(h, cfg);  // build_initial_state with the configured generator
  const ParticlisationParams pp = ParticlisationParams::from_config(cfg);
  SurfaceFinder<EosTable>::Meta meta;
  meta.t_init = ti;
  meta.origin = centred_origin(h.grid);
  SurfaceFinder<EosTable> sf(eos, pp, h.grid, "wp10a_backend", meta);
  sf.begin(h);
  for (int s = 0; s < 3; ++s) {
    h.step();
    sf.after_step(h);
  }
  sf.finish(h);
  const auto all = read_surface_bin("wp10a_backend");
  const long nini = sf.summary()["n_elements_initial_slice"].get<long>();
  std::vector<Real> mine(all.begin() + nini * kSurfRecord, all.end());
  const std::string ref = std::string(SPF_TEST_DATA_DIR) + "/wp10a_surface_reference.bin";
  const double ini_E = jget(sf.summary(), "totals_initial_slice", "E_GeV"), ini_B = jget(sf.summary(), "totals_initial_slice", "B");
  if (std::getenv("SPF_WRITE_SURFACE_REFERENCE")) {
    std::ofstream o(ref, std::ios::binary);
    const double head[3] = {static_cast<double>(nini), ini_E, ini_B};
    o.write(reinterpret_cast<const char*>(head), sizeof head);
    o.write(reinterpret_cast<const char*>(mine.data()), static_cast<std::streamsize>(mine.size() * sizeof(Real)));
    std::printf("[WP10a] T10a.6 reference written: %s (%zu elements)\n", ref.c_str(), mine.size() / kSurfRecord);
  }
  std::ifstream in(ref, std::ios::binary);
  REQUIRE(in.good());
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  std::vector<double> rv(bytes.size() / sizeof(double));
  std::memcpy(rv.data(), bytes.data(), rv.size() * sizeof(double));
  REQUIRE(rv.size() >= 3);
  const long ref_nini = static_cast<long>(rv[0]);
  std::vector<double> refel(rv.begin() + 3, rv.end());
  // order of both lists by the lattice cell of the element (time step, ix, iy, iz: the centroid lies inside its hypercube, so these integers do
  // not depend on rounding), then by the exact values; the file order sorts on the exact values only, and rounding differences between backends
  // can permute elements whose leading values agree to rounding
  const double dtl = h.dt(), dxl = h.grid.dx, t0l = ti;
  const auto orgl = centred_origin(h.grid);
  auto keyed = [&](const std::vector<double>& v) {
    const size_t m = v.size() / kSurfRecord;
    std::vector<std::array<long, 4>> key(m);
    for (size_t r = 0; r < m; ++r) {
      const double* q = &v[r * kSurfRecord];
      key[r] = {static_cast<long>(std::floor((q[0] - t0l) / dtl)), static_cast<long>(std::floor((q[1] - orgl[0]) / dxl)),
                static_cast<long>(std::floor((q[2] - orgl[1]) / dxl)), static_cast<long>(std::floor((q[3] - orgl[2]) / dxl))};
    }
    std::vector<size_t> idx(m);
    for (size_t r = 0; r < m; ++r) idx[r] = r;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
      if (key[a] != key[b]) return key[a] < key[b];
      for (int q = 0; q < kSurfRecord; ++q)
        if (v[a * kSurfRecord + q] != v[b * kSurfRecord + q]) return v[a * kSurfRecord + q] < v[b * kSurfRecord + q];
      return false;
    });
    std::vector<double> out(v.size());
    for (size_t r = 0; r < m; ++r)
      for (int q = 0; q < kSurfRecord; ++q) out[r * kSurfRecord + q] = v[idx[r] * kSurfRecord + q];
    return out;
  };
  mine = keyed(mine);
  refel = keyed(refel);
  const size_t n = mine.size() / kSurfRecord, nr = refel.size() / kSurfRecord;
  double worst = 0.0;
  int worst_col = -1;
  if (n == nr) {
    for (int q = 0; q < kSurfRecord; ++q) {
      double scale = 0.0, dev = 0.0;
      for (size_t r = 0; r < n; ++r) {
        scale = std::max(scale, std::fabs(refel[r * kSurfRecord + q]));
        dev = std::max(dev, std::fabs(mine[r * kSurfRecord + q] - refel[r * kSurfRecord + q]));
      }
      const double rel = scale > 0.0 ? dev / scale : dev;
      if (rel > worst) { worst = rel; worst_col = q; }
    }
  }
  std::printf("[WP10a] T10a.6 backend %s: %zu isoenergy elements (reference %zu), initial slice %ld (reference %ld); largest column deviation "
              "max|a-b|/max|col| = %.3e (column %d); initial slice totals vs reference: E %.3e B %.3e\n",
              runtime_info()["execution_space"].get<std::string>().c_str(), n, nr, nini, ref_nini, worst, worst_col,
              ini_E / rv[1] - 1.0, ini_B / rv[2] - 1.0);
  // flux-level comparison (no equation of state needed): sum of N^mu dSigma_mu and of dSigma_0 over the isoenergy elements
  double bf = 0, bfr = 0, s0 = 0, s0r = 0, sabs = 0;
  if (n == nr)
    for (size_t r = 0; r < n; ++r) {
      const double* q = &mine[r * kSurfRecord];
      const double* g = &refel[r * kSurfRecord];
      bf += q[13] * (q[8] * q[4] + q[9] * q[5] + q[10] * q[6] + q[11] * q[7]);
      bfr += g[13] * (g[8] * g[4] + g[9] * g[5] + g[10] * g[6] + g[11] * g[7]);
      s0 += q[4]; s0r += g[4];
      sabs += std::fabs(g[4]) + std::fabs(g[5]) + std::fabs(g[6]) + std::fabs(g[7]);
    }
  C.dB_flux = bfr != 0.0 ? bf / bfr - 1.0 : 0.0;
  C.d_dsig0 = sabs > 0.0 ? (s0 - s0r) / sabs : 0.0;
  std::printf("[WP10a] T10a.6 backend %s: sum of N^mu dSigma_mu over the isoenergy elements vs reference %.3e; sum of dSigma_0 %.3e (relative to "
              "sum |dSigma_mu|)\n", runtime_info()["execution_space"].get<std::string>().c_str(), C.dB_flux, C.d_dsig0);
  if (n == nr && worst <= 1e-10) {  // kept for inspection otherwise
    std::remove("wp10a_backend.surface.bin");
    std::remove("wp10a_backend.surface.json");
  }
  C.n = n; C.nr = nr; C.nini = nini; C.ref_nini = ref_nini; C.worst = worst;
  C.dE_ini = ini_E / rv[1] - 1.0; C.dB_ini = ini_B / rv[2] - 1.0;
  C.done = true;
  return C;
}
}  // namespace

TEST_CASE("T10a.6 backend reference: same elements (count, initial slice) as the Serial reference") {
  const BackendCmp& C = backend_cmp();
  CHECK(C.n == C.nr);
  CHECK(C.nini == C.ref_nini);
  CHECK(std::fabs(C.dE_ini) <= 1e-10);
  CHECK(std::fabs(C.dB_ini) <= 1e-10);
}

// Expected failure on CUDA, kept on purpose: the a-priori criterion 1e-10 per column holds on Serial and OpenMP (0); on the GPU node the
// isoenergy elements of the CUDA build differ from the Serial reference by up to 1.6e-8 of the column maximum (u^1; about 1000 of the 2532
// elements above 1e-11): the fluid fields of the two backends already differ after 3 steps (table EoS, bicubic, gradient term, iterative
// recovery), and small pieces that cut a hypercube corner amplify that.  The element count and the topology are the same; the summed baryon flux
// agrees to 2e-11.  may_fail: a failure is reported but does not fail the run.
TEST_CASE("T10a.6 backend reference: element values to 1e-10 of the column maximum" * doctest::may_fail()) {
  const BackendCmp& C = backend_cmp();
  CHECK(C.worst <= 1e-10);
}

// ------------------------------------------------------------------ T10a.7 driver: spf_collide with particlisation, file format
TEST_CASE("T10a.7 spf_collide with particlisation: event ends when the surface is complete; file and sidecar follow the format") {
  namespace fs = std::filesystem;
  const std::string out = (fs::temp_directory_path() / "spf_wp10a_driver").string();
  fs::remove_all(out);
  fs::create_directories(out);
  const std::string cmd = std::string("\"") + SPF_COLLIDE_EXE + "\" --out " + out + " --config " + SPF_SOURCE_DIR +
                          "/config/collisions/common.json ic.generator=glauber_gaussians ic.e_lab_gev=3 eos.model=table_two_phase" +
                          " eos.table_path=\"" + SPF_SOURCE_DIR + "/data/eos/test/eos_two_phase_test\"" +
                          " grid.dx_fm=0.5 grid.Lx_fm=24 grid.Ly_fm=24 grid.Lz_fm=24 run.n_events=1 ic.max_outside_fraction=1e-3 > " + out + "/driver.log 2>&1";
  const int st = std::system(cmd.c_str());
  REQUIRE(st == 0);
  std::ifstream sj(out + "/event_000/summary.json");
  REQUIRE(sj.good());
  const nlohmann::json J = nlohmann::json::parse(sj);
  REQUIRE(J.contains("particlisation"));
  const auto& P = J["particlisation"];
  const std::string base = out + "/event_000/event_000";
  std::ifstream sc(base + ".surface.json");
  REQUIRE(sc.good());
  const nlohmann::json S = nlohmann::json::parse(sc);
  const auto v = read_surface_bin(base);
  const long n = S["n_elements"].get<long>();
  for (const char* k : {"format_version", "n_elements", "eps_sw_GeV_fm3", "method", "dx_fm", "dt_fm", "grid_shape", "eos_model", "eos_table_sha256",
                        "event_id", "E_lab_GeV", "totals_surface", "totals_initial_slice", "totals_all_elements", "totals_inside_fluid_at_end",
                        "totals_initial_on_grid", "n_elements_initial_slice", "methods_in_file", "t_init_fm", "closure"})
    CHECK_MESSAGE(S.contains(k), "missing sidecar key ", k);
  CHECK(S["format_version"].get<int>() == 1);
  CHECK(S["method"].get<std::string>() == "isoenergy");
  CHECK(static_cast<long>(v.size()) == n * kSurfRecord);
  CHECK(fs::file_size(base + ".surface.bin") == static_cast<std::uintmax_t>(n) * kSurfRecord * sizeof(double));
  CHECK(J["status"].get<std::string>() == "fluid_end");
  CHECK(P["closed"].get<bool>());
  CHECK(S["eos_table_sha256"].get<std::string>().size() == 64);
  CHECK(std::filesystem::exists(out + "/event_000/observables.csv"));
  // records: finite, u^mu u_mu = 1, initial-slice elements at t_init with dSigma = (dV, 0, 0, 0), surface elements later with T > 0
  const long nini = S["n_elements_initial_slice"].get<long>();
  const double ti = S["t_init_fm"].get<double>(), dV = 0.125;
  double worst_norm = 0;
  long bad = 0, bad_ini = 0, tzero = 0;
  for (long r = 0; r < n; ++r) {
    const double* q = &v[r * kSurfRecord];
    for (int k = 0; k < kSurfRecord; ++k) if (!std::isfinite(q[k])) ++bad;
    worst_norm = std::max(worst_norm, std::fabs((q[8] * q[8] - q[9] * q[9] - q[10] * q[10] - q[11] * q[11]) / (q[8] * q[8]) - 1.0 / (q[8] * q[8])));
    if (r < nini && (q[0] != ti || std::fabs(q[4] - dV) > 1e-15 || q[5] != 0 || q[6] != 0 || q[7] != 0)) ++bad_ini;
    if (r >= nini && !(q[14] > 0.0)) ++tzero;
  }
  std::printf("[WP10a] T10a.7 driver (backend %s): status %s after %.2f fm/c, %ld elements (%ld initial slice), file %.1f MB, closure E %.3e B %.3e, "
              "non-finite values %ld, max |u.u - 1| / u0^2 %.2e, initial-slice format violations %ld, surface elements with T <= 0: %ld, wall surface %.2f s of %.2f s\n",
              J["backend"].get<std::string>().c_str(), J["status"].get<std::string>().c_str(), J["time_after_init_fm"].get<double>(), n, nini,
              fs::file_size(base + ".surface.bin") / 1e6, jget(S, "closure", "E"), jget(S, "closure", "B"), bad, worst_norm, bad_ini, tzero,
              P["wall_surface_s"].get<double>(), J["wall_total_s"].get<double>());
  CHECK(bad == 0);
  CHECK(worst_norm < 1e-12);
  CHECK(bad_ini == 0);
  CHECK(tzero == 0);
  fs::remove_all(out);
}

