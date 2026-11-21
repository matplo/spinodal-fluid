// T1.2 - T1.5: relativistic shock tube, expansion into vacuum, configuration variants, backend reproducibility.
#include <doctest/doctest.h>

#include "exact_solutions.hpp"
#include "spf/config.hpp"
#include "spf/hydro1d.hpp"
#include "test_eos_ideal.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace spf;

namespace {

// ---------------------------------------------------------------- shock tube (T1.2, T1.4, T1.5)
constexpr Real kBox = 100.0, kX0 = 50.0, kTime = 40.0;  // fm, fm, fm (c = 1)

struct ShockRun {
  std::vector<Real> eps, rho, v;
  Real l1 = 0, rel_E = 0, rel_R = 0, t = 0;
  long fails = 0, bad_cells = 0, clamps = 0;
};

ShockRun run_shock(int nx, const ShastaParams& prm_in) {
  ShastaParams prm = prm_in;
  prm.bc = Boundary::Outflow;
  const Real dx = kBox / nx;
  Hydro1D<TestEosIdeal> h(TestEosIdeal{}, nx, dx, prm);
  std::vector<Real> e(nx), r(nx), u(nx, 0.0);
  for (int i = 0; i < nx; ++i) {
    const bool left = (i + 0.5) * dx < kX0;
    e[i] = left ? 10.0 : 1.0;
    r[i] = left ? 1.0 : 0.5;
  }
  h.set_state(e, r, u);
  const Real E0 = h.total_E(), R0 = h.total_R();
  const long nsteps = std::lround(kTime / (prm.lambda * dx));
  for (long s = 0; s < nsteps; ++s) h.step();
  ShockRun out;
  out.t = nsteps * prm.lambda * dx;
  out.fails = h.update_primitives() + h.n_recover_fail_total;
  out.bad_cells = h.count_E_lt_M();
  out.clamps = h.n_clamped_total;
  out.rel_E = std::fabs(h.total_E() - E0) / E0;
  out.rel_R = std::fabs(h.total_R() - R0) / R0;
  out.eps = h.line(h.eps); out.rho = h.line(h.rho); out.v = h.line(h.v);
  const exact::ShockTube ex(10.0, 1.0);
  Real l1 = 0.0;
  for (int i = 0; i < nx; ++i) l1 += std::fabs(out.eps[i] - ex.eps(((i + 0.5) * dx - kX0) / out.t)) * dx;
  out.l1 = l1;
  return out;
}

Config default_config() { return Config::from_file(SPF_DEFAULT_CONFIG); }

// ---------------------------------------------------------------- expansion into vacuum (T1.3)
struct VacuumRun {
  Real l1_T00 = 0, l1_eps = 0, rel_T00 = 0, rel_eps = 0, leak = 0, leak_frac = 0, emax_acausal = 0;
  long fails = 0, bad_cells = 0, clamps = 0;
};

VacuumRun run_vacuum(const ShastaParams& prm_in) {
  ShastaParams prm = prm_in;
  prm.bc = Boundary::Outflow;
  const int nx = 300, n0 = 150;  // interface at the face between cell 149 and 150
  const Real dx = 1.0, eps0 = 1.0;
  Hydro1D<TestEosIdeal> h(TestEosIdeal{}, nx, dx, prm);
  std::vector<Real> e(nx, 0.0), r(nx, 0.0), u(nx, 0.0);
  for (int i = 0; i < n0; ++i) e[i] = eps0;  // net baryon free: rho = 0
  h.set_state(e, r, u);
  const int nsteps = 100;
  for (int s = 0; s < nsteps; ++s) h.step();
  const Real t = nsteps * prm.lambda * dx;
  VacuumRun out;
  out.fails = h.update_primitives() + h.n_recover_fail_total;
  out.bad_cells = h.count_E_lt_M();
  out.clamps = h.n_clamped_total;
  const auto E = h.line(h.U[2]), eps = h.line(h.eps);
  Real dT = 0, de = 0, nT = 0, ne = 0;
  const Real dz = dx / t;
  for (int i = 0; i < nx; ++i) {
    const Real zeta = ((i + 0.5) * dx - n0 * dx) / t;
    if (zeta >= -2.0 && zeta <= 3.0) {
      const Real aT = exact::vacuum_T00(zeta, eps0), ae = exact::vacuum_eps(zeta, eps0);
      dT += std::fabs(E[i] - aT) * dz; nT += aT * dz;
      de += std::fabs(eps[i] - ae) * dz; ne += ae * dz;
    }
    if (zeta > 1.0) {
      out.leak += E[i] * dz;
      if (E[i] > out.emax_acausal) out.emax_acausal = E[i];
    }
  }
  out.l1_T00 = dT; out.l1_eps = de; out.rel_T00 = dT / nT; out.rel_eps = de / ne;
  // normalise the acausal T00 to the integral of T00 over -1 <= zeta <= 1
  Real tot = 0;
  for (int i = 0; i < nx; ++i) {
    const Real zeta = ((i + 0.5) * dx - n0 * dx) / t;
    if (zeta >= -1.0 && zeta <= 1.0) tot += E[i] * dz;
  }
  out.leak_frac = out.leak / tot;
  return out;
}

}  // namespace

TEST_CASE("primitive recovery: round trip (eps, rho, v) -> (R, M, E) -> (eps, rho, v)") {
  const TestEosIdeal eos;
  Real worst = 0.0;
  int nfail = 0, n = 0;
  for (Real v : {0.0, 1e-6, 0.1, 0.5, 0.9, 0.99, 0.999, -0.7})
    for (Real eps : {1e-6, 1e-2, 1.0, 50.0})
      for (Real rho : {0.0, 0.3, 4.0}) {
        const Real p = eos.p0(eps, rho), g2 = 1.0 / (1.0 - v * v), g = std::sqrt(g2);
        const Real R = g * rho, M = (eps + p) * g2 * v, E = (eps + p) * g2 - p;
        Real e2, r2, v2, p2;
        nfail += recover_primitives(eos, R, M, E, 1e-12, e2, r2, v2, p2);
        const Real err = std::fabs(e2 - eps) / eps + std::fabs(r2 - rho) / std::fmax(rho, 1e-12) + std::fabs(v2 - v);
        if (err > worst) worst = err;
        ++n;
      }
  std::printf("[WP1] primitive recovery round trip: %d states, failures %d, worst combined error %.3e\n", n, nfail, worst);
  CHECK(nfail == 0);
  CHECK(worst < 1e-9);
}

TEST_CASE("exact shock tube solution satisfies the Rankine-Hugoniot conditions and the rarefaction invariant") {
  const exact::ShockTube ex(10.0, 1.0);
  const Real e = ex.eps_star, v = ex.v_star, V = ex.V_shock;
  const Real w = (e + e / 3.0) / (1.0 - v * v), eR = 1.0, pR = eR / 3.0;
  const Real T00s = w - e / 3.0, T0xs = w * v, Txxs = w * v * v + e / 3.0;
  const Real r1 = V * (T00s - eR) - T0xs;
  const Real r2 = V * T0xs - (Txxs - pR);
  std::printf("[WP1] shock tube exact: eps* = %.10f  v* = %.10f  V_shock = %.10f  zeta_tail = %.10f  RH residuals %.2e %.2e\n",
              e, v, V, ex.zeta_tail, r1, r2);
  CHECK(std::fabs(r1) < 1e-12);
  CHECK(std::fabs(r2) < 1e-12);
  CHECK(V < 1.0);
  CHECK(ex.zeta_tail < V);
  CHECK(std::fabs(ex.eps(ex.zeta_tail - 1e-12) - e) < 1e-9);  // fan joins the plateau continuously
}

TEST_CASE("T1.2 shock tube p = eps/3: L1 error decreases with resolution; E and R conserved") {
  const Config cfg = default_config();
  const ShastaParams prm = ShastaParams::from_config(cfg);
  Real err[3];
  const int nxs[3] = {200, 400, 800};
  for (int k = 0; k < 3; ++k) {
    const ShockRun r = run_shock(nxs[k], prm);
    err[k] = r.l1;
    std::printf("[WP1] T1.2 nx=%d t=%.3f: L1(eps) = %.6e GeV/fm^2  rel dE = %.3e  rel dR = %.3e  recovery fails = %ld  E<|M| cells = %ld  clamps = %ld\n",
                nxs[k], r.t, r.l1, r.rel_E, r.rel_R, r.fails, r.bad_cells, r.clamps);
    CHECK(r.rel_E < 1e-11);
    CHECK(r.rel_R < 1e-11);
    CHECK(r.fails == 0);
    CHECK(r.bad_cells == 0);
  }
  std::printf("[WP1] T1.2 L1: %.6e (200) %.6e (400) %.6e (800); order 200->400 = %.3f, 400->800 = %.3f\n", err[0], err[1], err[2],
              std::log2(err[0] / err[1]), std::log2(err[1] / err[2]));
  CHECK(err[1] < err[0]);
  CHECK(err[2] < err[1]);
}

TEST_CASE("T1.3 expansion of semi-infinite matter into vacuum: similarity solution after 100 steps") {
  const Config cfg = default_config();
  for (int variant = 0; variant < 2; ++variant) {
    ShastaParams prm = ShastaParams::from_config(cfg);
    prm.phoenical = (variant == 0);
    const VacuumRun r = run_vacuum(prm);
    std::printf("[WP1] T1.3 %s: L1(T00) = %.6e (rel %.4f)  L1(eps) = %.6e (rel %.4f)  integral of T00 at zeta>1 = %.6e (%.4e of int_{-1}^{1} T00)  max E at zeta>1 = %.3e  recovery fails = %ld  E<|M| cells = %ld  clamps = %ld\n",
                prm.phoenical ? "phoenical" : "explicit", r.l1_T00, r.rel_T00, r.l1_eps, r.rel_eps, r.leak, r.leak_frac,
                r.emax_acausal, r.fails, r.bad_cells, r.clamps);
    CHECK(r.fails == 0);
    CHECK(r.bad_cells == 0);
    CHECK(r.rel_T00 < 0.1);  // regression guard around the measured value, not an acceptance criterion
    CHECK(r.leak_frac < 0.05);
  }
}

TEST_CASE("T1.4 configuration variants: antidiffusion, eta_ad, lambda") {
  int ran = 0;
  for (const char* ad : {"phoenical", "explicit"})
    for (const char* eta : {"0.125", "0.1"})
      for (const char* lam : {"0.4", "0.2"}) {
        Config cfg = default_config();
        cfg.set_from_string(std::string("shasta.antidiffusion=") + ad);
        cfg.set_from_string(std::string("shasta.eta_ad=") + eta);
        cfg.set_from_string(std::string("shasta.lambda=") + lam);
        const ShastaParams prm = ShastaParams::from_config(cfg);
        CHECK(prm.phoenical == (std::string(ad) == "phoenical"));
        CHECK(prm.eta_ad == doctest::Approx(std::atof(eta)));
        CHECK(prm.lambda == doctest::Approx(std::atof(lam)));
        const ShockRun r = run_shock(400, prm);
        std::printf("[WP1] T1.4 %-9s eta_ad=%-5s lambda=%s: T1.2 L1(eps) at 400 cells = %.6e  (fails %ld, E<|M| %ld)\n", ad, eta,
                    lam, r.l1, r.fails, r.bad_cells);
        CHECK(r.fails == 0);
        CHECK(r.bad_cells == 0);
        CHECK(r.rel_E < 1e-11);
        ++ran;
      }
  CHECK(ran == 8);
  Config bad = default_config();
  bad.set_from_string("shasta.antidiffusion=bogus");
  CHECK_THROWS_AS(ShastaParams::from_config(bad), ConfigError);
}

TEST_CASE("T1.5 shock tube profile at 400 cells reproduces the stored reference (backend independence)") {
  const ShastaParams prm = ShastaParams::from_config(default_config());
  const ShockRun r = run_shock(400, prm);
  const std::string dir = SPF_TEST_DATA_DIR;
  const std::string ref_path = dir + "/shock400_reference.csv";
  // profile of this build, for comparison between builds
  {
    std::string name = std::string("shock400_") + ExecSpace::name() + ".csv";
    std::ofstream out(name);
    out.precision(17);
    out << "# i eps rho v\n";
    for (size_t i = 0; i < r.eps.size(); ++i) out << i << ' ' << r.eps[i] << ' ' << r.rho[i] << ' ' << r.v[i] << '\n';
  }
  if (std::getenv("SPF_WRITE_REFERENCE")) {
    std::ofstream out(ref_path);
    out.precision(17);
    out << "# i eps rho v   (p = eps/3, shock tube eps 10|1, rho 1|0.5, 400 cells, default shasta parameters, outflow)\n";
    for (size_t i = 0; i < r.eps.size(); ++i) out << i << ' ' << r.eps[i] << ' ' << r.rho[i] << ' ' << r.v[i] << '\n';
  }
  std::ifstream in(ref_path);
  REQUIRE(in.good());
  std::string line;
  std::vector<Real> e, rh, v;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    long i; Real a, b, c;
    ss >> i >> a >> b >> c;
    e.push_back(a); rh.push_back(b); v.push_back(c);
  }
  REQUIRE(e.size() == r.eps.size());
  Real worst_e = 0, worst_r = 0, worst_v = 0, vmax = 0;
  for (Real x : v) vmax = std::fmax(vmax, std::fabs(x));
  for (size_t i = 0; i < e.size(); ++i) {
    worst_e = std::fmax(worst_e, std::fabs(r.eps[i] - e[i]) / std::fabs(e[i]));
    worst_r = std::fmax(worst_r, std::fabs(r.rho[i] - rh[i]) / std::fabs(rh[i]));
    worst_v = std::fmax(worst_v, std::fabs(r.v[i] - v[i]) / vmax);
  }
  std::printf("[WP1] T1.5 backend %s: max relative deviation from reference: eps %.3e  rho %.3e  v (relative to max|v|) %.3e\n",
              ExecSpace::name(), worst_e, worst_r, worst_v);
  CHECK(worst_e < 1e-12);
  CHECK(worst_r < 1e-12);
  CHECK(worst_v < 1e-12);
}
