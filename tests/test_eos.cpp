// Acceptance tests of the equation-of-state structs (WP2a).  Every compared value is evaluated inside a Kokkos kernel
// (functors, so that the same source compiles with nvcc); measured numbers are printed as "[T2.x] ...".
#include <doctest/doctest.h>

#include "spf/config.hpp"
#include "spf/eos/eos.hpp"
#include "spf/types.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using spf::ExecSpace;
using spf::Real;
using VecD = std::vector<Real>;

static_assert(spf::EquationOfState<spf::EosIdeal>);
static_assert(spf::EquationOfState<spf::EosGammaLaw>);
static_assert(spf::EquationOfState<spf::EosColdThermal>);
static_assert(std::is_trivially_copyable_v<spf::EosIdeal>);
static_assert(std::is_trivially_copyable_v<spf::EosGammaLaw>);
static_assert(std::is_trivially_copyable_v<spf::EosColdThermal>);

namespace {

const std::string kSrc = SPF_SOURCE_DIR;
constexpr Real kRhoS = 0.153, kEpsS = 0.940 * 0.153;

// The table owns Kokkos Views, so it must be released before Kokkos::finalize: each test case holds a TableGuard.
std::unique_ptr<spf::ColdCurveTable>& table_ptr() {
  static std::unique_ptr<spf::ColdCurveTable> p;
  return p;
}
struct TableGuard {
  TableGuard() { table_ptr() = std::make_unique<spf::ColdCurveTable>(spf::ColdCurveTable::load(kSrc + "/data/eos/cold_curve.dat")); }
  ~TableGuard() { table_ptr().reset(); }
};
const spf::ColdCurveTable& table() { return *table_ptr(); }
spf::EosColdThermal two_phase() { return spf::EosColdThermal(table(), spf::ColdVariant::two_phase); }
spf::EosColdThermal maxwell() { return spf::EosColdThermal(table(), spf::ColdVariant::maxwell); }

using DView = Kokkos::View<Real*>;

DView to_device(const VecD& v) {
  DView d("in", v.size());
  auto h = Kokkos::create_mirror_view(d);
  for (size_t i = 0; i < v.size(); ++i) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}
VecD to_host(const DView& d) {
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);
  return VecD(h.data(), h.data() + h.extent(0));
}

// ---- generic evaluation of the interface ----
template <class E>
struct EvalK {
  E eos;
  DView eps, rho, p, dpe, dpr, vs;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    p(i) = eos.p0(eps(i), rho(i));
    dpe(i) = eos.dp_deps(eps(i), rho(i));
    dpr(i) = eos.dp_drho(eps(i), rho(i));
    vs(i) = eos.vs2(eps(i), rho(i));
  }
};
struct Out {
  VecD p, dpe, dpr, vs;
};
template <class E>
Out eval(const E& eos, const VecD& eps, const VecD& rho) {
  const int n = static_cast<int>(eps.size());
  EvalK<E> k{eos, to_device(eps), to_device(rho), DView("p", n), DView("a", n), DView("b", n), DView("v", n)};
  Kokkos::parallel_for("eval_eos", Kokkos::RangePolicy<ExecSpace>(0, n), k);
  return {to_host(k.p), to_host(k.dpe), to_host(k.dpr), to_host(k.vs)};
}

// ---- cold curve quantities ----
struct ColdK {
  spf::EosColdThermal eos;
  DView rho, ec, mu, pc, dpc;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    ec(i) = eos.eps_cold(rho(i));
    mu(i) = eos.mu_cold(rho(i));
    pc(i) = eos.p_cold(rho(i));
    dpc(i) = eos.dp_cold(rho(i));
  }
};
struct Cold {
  VecD ec, mu, pc, dpc;
};
Cold cold(const spf::EosColdThermal& eos, const VecD& rho) {
  const int n = static_cast<int>(rho.size());
  ColdK k{eos, to_device(rho), DView("a", n), DView("b", n), DView("c", n), DView("d", n)};
  Kokkos::parallel_for("eval_cold", Kokkos::RangePolicy<ExecSpace>(0, n), k);
  return {to_host(k.ec), to_host(k.mu), to_host(k.pc), to_host(k.dpc)};
}

// ---- spinodal and coexistence densities from the interpolant (single-thread kernel, fixed iteration counts) ----
struct RootsK {
  spf::EosColdThermal eos;
  Real lo, hi;  // scan interval [fm^-3]
  Real guess1, guess2;
  DView out;  // rhoA, rhoB, rho1, rho2, mu, p, newton residual
  KOKKOS_INLINE_FUNCTION void operator()(const int) const {
    const int n = 6000;
    Real prev = eos.dp_cold(lo), rA = -1.0, rB = -1.0;
    for (int i = 1; i <= n; ++i) {
      const Real x = lo + (hi - lo) * i / n, cur = eos.dp_cold(x);
      if (prev > 0.0 && cur <= 0.0 && rA < 0.0) rA = bisect(x - (hi - lo) / n, x);
      if (prev <= 0.0 && cur > 0.0) rB = bisect(x - (hi - lo) / n, x);
      prev = cur;
    }
    Real r1 = guess1, r2 = guess2, res = 0.0;
    for (int it = 0; it < 60; ++it) {
      const Real m1 = eos.mu_cold(r1), m2 = eos.mu_cold(r2);
      const Real f1 = m1 - m2, f2 = eos.p_cold(r1) - eos.p_cold(r2);
      const Real a = eos.dp_cold(r1) / r1, b = -eos.dp_cold(r2) / r2;  // d mu / d rho = p_c' / rho
      const Real c = eos.dp_cold(r1), d = -eos.dp_cold(r2);
      const Real det = a * d - b * c;
      r1 -= (f1 * d - b * f2) / det;
      r2 -= (a * f2 - c * f1) / det;
      res = Kokkos::fabs(f1) + Kokkos::fabs(f2);
    }
    out(0) = rA;
    out(1) = rB;
    out(2) = r1;
    out(3) = r2;
    out(4) = eos.mu_cold(r1);
    out(5) = eos.p_cold(r1);
    out(6) = res;
  }
  KOKKOS_INLINE_FUNCTION Real bisect(Real a, Real b) const {
    const Real fa = eos.dp_cold(a);
    for (int it = 0; it < 100; ++it) {
      const Real m = 0.5 * (a + b), fm = eos.dp_cold(m);
      if ((fm > 0.0) == (fa > 0.0)) a = m;
      else b = m;
    }
    return 0.5 * (a + b);
  }
};

// ---- finite-difference isentropic sound speed ----
template <class E>
struct IsentropeK {
  E eos;
  Real delta;
  DView eps, rho, vs_fd;
  KOKKOS_INLINE_FUNCTION Real slope(Real e, Real r) const { return (e + eos.p0(e, r)) / r; }
  // RK4 step of d eps / d rho = (eps + p0)/rho from (r, e) by dr; returns the new eps
  KOKKOS_INLINE_FUNCTION Real rk4(Real e, Real r, Real dr) const {
    const Real k1 = slope(e, r);
    const Real k2 = slope(e + 0.5 * dr * k1, r + 0.5 * dr);
    const Real k3 = slope(e + 0.5 * dr * k2, r + 0.5 * dr);
    const Real k4 = slope(e + dr * k3, r + dr);
    return e + dr * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
  }
  KOKKOS_INLINE_FUNCTION Real deriv(Real e0, Real r0, Real d) const {
    const Real ep = rk4(e0, r0, d), em = rk4(e0, r0, -d);
    return (eos.p0(ep, r0 + d) - eos.p0(em, r0 - d)) / (2.0 * d);  // dp/drho along the isentrope
  }
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    const Real e0 = eps(i), r0 = rho(i), h = e0 + eos.p0(e0, r0);
    const Real d1 = deriv(e0, r0, delta), d2 = deriv(e0, r0, 0.5 * delta);
    const Real richardson = (4.0 * d2 - d1) / 3.0;
    vs_fd(i) = r0 / h * richardson;
  }
};
template <class E>
VecD fd_vs2(const E& eos, const VecD& eps, const VecD& rho, Real delta) {
  const int n = static_cast<int>(eps.size());
  IsentropeK<E> k{eos, delta, to_device(eps), to_device(rho), DView("v", n)};
  Kokkos::parallel_for("isentrope_fd", Kokkos::RangePolicy<ExecSpace>(0, n), k);
  return to_host(k.vs_fd);
}

Real rel(Real a, Real b) { return std::fabs(a - b) / std::max(std::fabs(b), 1e-300); }

nlohmann::json reference() {
  std::ifstream in(kSrc + "/tools/eos/eos_reference_values.json");
  REQUIRE(in.good());
  return nlohmann::json::parse(in);
}

// density at the centre of grid cell n (keeps finite-difference stencils inside one polynomial piece)
Real cell_centre(Real rho_over_rhos) {
  const Real h = table().d_rho;
  return (std::floor(rho_over_rhos * kRhoS / h) + 0.5) * h;
}

}  // namespace

TEST_CASE("T2.0 concept and trivial copyability") {
  CHECK(spf::EquationOfState<spf::EosIdeal>);
  CHECK(spf::EquationOfState<spf::EosGammaLaw>);
  CHECK(spf::EquationOfState<spf::EosColdThermal>);
  CHECK(std::is_trivially_copyable_v<spf::EosColdThermal>);
  std::printf("[T2.0] concept satisfied and trivially copyable for ideal, gamma_law, cold_thermal\n");
}

TEST_CASE("table loader and model parser") {
  TableGuard guard;
  const auto& t = table();
  CHECK(t.n_nodes == 3201);
  CHECK(t.valid_range().rho_max == doctest::Approx(16.0 * kRhoS));
  CHECK(two_phase().valid_range().rho_max == doctest::Approx(16.0 * kRhoS));
  CHECK_THROWS_AS(spf::ColdCurveTable::load(kSrc + "/data/eos/no_such_file.dat"), std::runtime_error);
  CHECK(spf::parse_eos_model("cold_thermal_maxwell") == spf::EosModel::cold_thermal_maxwell);
  CHECK(spf::parse_eos_model("table_two_phase") == spf::EosModel::table_two_phase);
  CHECK_THROWS_AS(spf::parse_eos_model("nonsense"), spf::ConfigError);
  auto cfg = spf::Config::from_file(kSrc + "/config/defaults.json");
  CHECK(spf::eos_model_from_config(cfg) == spf::EosModel::table_two_phase);
  CHECK(cfg.get<double>("eos.gamma_th") == doctest::Approx(4.0 / 3.0));
  CHECK(cfg.get<double>("eos.gamma_law_gamma") == doctest::Approx(5.0 / 3.0));
  CHECK(cfg.get<double>("eos.gamma_law_mass_GeV") == doctest::Approx(0.940));
  CHECK(cfg.get<std::string>("eos.cold_curve_path") == "data/eos/cold_curve.dat");
  for (auto m : {spf::EosModel::ideal, spf::EosModel::gamma_law, spf::EosModel::cold_thermal_two_phase,
                 spf::EosModel::cold_thermal_maxwell, spf::EosModel::table_two_phase, spf::EosModel::table_maxwell})
    CHECK(spf::parse_eos_model(spf::eos_model_name(m)) == m);
  // outside the table range the density is clamped
  auto eos = two_phase();
  const Real rmax = t.valid_range().rho_max;
  auto c = cold(eos, {rmax, 2.0 * rmax, -1.0, 0.0});
  CHECK(c.ec[1] == c.ec[0]);
  CHECK(c.ec[2] == c.ec[3]);
  CHECK(c.ec[3] == 0.0);
}

TEST_CASE("analytic equations of state: closed forms") {
  spf::EosIdeal id;
  spf::EosGammaLaw gl(5.0 / 3.0, 0.940);
  const VecD eps = {2.0, 1.5, 0.5}, rho = {1.0, 0.8, 0.9};  // third point: eps < m rho (clamp)
  auto a = eval(id, eps, rho);
  auto b = eval(gl, eps, rho);
  for (int i = 0; i < 3; ++i) {
    CHECK(a.p[i] == doctest::Approx(eps[i] / 3.0));
    CHECK(a.vs[i] == doctest::Approx(1.0 / 3.0));
    CHECK(a.dpr[i] == 0.0);
  }
  CHECK(b.p[0] == doctest::Approx((2.0 / 3.0) * (2.0 - 0.940)));
  CHECK(b.p[2] == 0.0);
  CHECK(b.dpe[2] == 0.0);
  CHECK(b.dpr[0] == doctest::Approx(-(2.0 / 3.0) * 0.940));
}

TEST_CASE("T2.1 cold curve: coexistence and spinodal densities") {
  TableGuard guard;
  const auto ref = reference()["T0_spline"];
  const Real rA = ref["rhoA"], rB = ref["rhoB"], r1 = ref["rho1"], r2 = ref["rho2"];
  const Real mu_ref = Real(ref["mu_coex"]) / 1000.0, p_ref = Real(ref["p_coex"]) / 1000.0;
  DView out("out", 7);
  RootsK k{two_phase(), 4.2 * kRhoS, 10.5 * kRhoS, 4.4 * kRhoS, 10.4 * kRhoS, out};
  Kokkos::parallel_for("roots", Kokkos::RangePolicy<ExecSpace>(0, 1), k);
  const VecD o = to_host(out);
  const Real dA = std::fabs(o[0] / kRhoS - rA), dB = std::fabs(o[1] / kRhoS - rB);
  const Real d1 = std::fabs(o[2] / kRhoS - r1), d2 = std::fabs(o[3] / kRhoS - r2);
  const Real dmu = rel(o[4], mu_ref), dp = rel(o[5], p_ref);
  std::printf("[T2.1] rhoA=%.5f rhoB=%.5f rho1=%.5f rho2=%.5f [rho_s]; mu_coex=%.6f GeV p_coex=%.6f GeV/fm3; "
              "max |d rho|=%.2e rho_s, rel mu=%.2e rel p=%.2e, Newton residual=%.1e\n",
              o[0] / kRhoS, o[1] / kRhoS, o[2] / kRhoS, o[3] / kRhoS, o[4], o[5], std::max({dA, dB, d1, d2}), dmu, dp, o[6]);
  CHECK(o[6] < 1e-12);
  CHECK(dA < 0.01);
  CHECK(dB < 0.01);
  CHECK(d1 < 0.01);
  CHECK(d2 < 0.01);
  CHECK(dmu < 1e-3);
  CHECK(dp < 1e-3);
}

TEST_CASE("T2.2 maxwell variant") {
  TableGuard guard;
  const auto& t = table();
  auto tp = two_phase();
  auto mx = maxwell();
  VecD rin, rout;
  for (int i = 1; i <= 20; ++i) rin.push_back(t.rho_1 + (t.rho_2 - t.rho_1) * i / 21.0);
  for (int i = 0; i < 10; ++i) rout.push_back(0.05 + (t.rho_1 - 0.1) * i / 9.0);
  for (int i = 0; i < 10; ++i) rout.push_back(t.rho_2 + 0.02 + (16.0 * kRhoS - t.rho_2 - 0.04) * i / 9.0);
  auto cin = cold(mx, rin);
  Real max_dp = 0.0, max_p = 0.0, max_dmu = 0.0;
  for (size_t i = 0; i < rin.size(); ++i) {
    max_dp = std::max(max_dp, std::fabs(cin.dpc[i]));
    max_p = std::max(max_p, rel(cin.pc[i], t.p_coex));
    max_dmu = std::max(max_dmu, rel(cin.mu[i], t.mu_coex));
  }
  // thermal energy chosen above the cold curve in both variants
  VecD eps_out;
  auto cout_ = cold(tp, rout);
  for (size_t i = 0; i < rout.size(); ++i) eps_out.push_back(cout_.ec[i] + 0.05 + 0.1 * i);
  auto a = eval(mx, eps_out, rout), b = eval(tp, eps_out, rout);
  Real max_out = 0.0;
  for (size_t i = 0; i < rout.size(); ++i)
    max_out = std::max({max_out, rel(a.p[i], b.p[i]), rel(a.dpe[i], b.dpe[i]), rel(a.dpr[i], b.dpr[i])});
  // continuity of p0 at fixed eps across rho_1 and rho_2
  const Real e_fix = 3.0, eps_e = 1e-9;
  auto cont = eval(mx, VecD(4, e_fix), {t.rho_1 - eps_e, t.rho_1 + eps_e, t.rho_2 - eps_e, t.rho_2 + eps_e});
  const Real j1 = rel(cont.p[0], cont.p[1]), j2 = rel(cont.p[2], cont.p[3]);
  std::printf("[T2.2] max|dp_c/drho|=%.1e (inside), max rel(p_c-p_coex)=%.2e, rel(mu-mu_coex)=%.2e, "
              "outside max rel diff to two_phase=%.2e, p0 jump rel at rho_1=%.2e at rho_2=%.2e\n",
              max_dp, max_p, max_dmu, max_out, j1, j2);
  CHECK(max_dp == 0.0);
  CHECK(max_p < 1e-12);
  CHECK(max_dmu < 1e-12);
  CHECK(max_out < 1e-13);
  CHECK(j1 < 1e-6);
  CHECK(j2 < 1e-6);
}

TEST_CASE("T2.3 vs2 equals finite-difference isentropic derivative") {
  TableGuard guard;
  const Real delta = 2e-5;  // fm^-3; RK4 isentrope, central difference at +-delta and +-delta/2, Richardson extrapolation
  struct Case {
    const char* name;
    Real worst;
  };
  std::vector<Case> res;
  // ideal and gamma law
  {
    VecD eps, rho;
    for (int i = 0; i < 10; ++i) {
      rho.push_back(0.1 + 0.15 * i);
      eps.push_back(0.940 * rho.back() * (1.05 + 0.3 * i) + 0.1);
    }
    spf::EosIdeal id;
    spf::EosGammaLaw gl(5.0 / 3.0, 0.940);
    for (int w = 0; w < 2; ++w) {
      VecD vs = w == 0 ? eval(id, eps, rho).vs : eval(gl, eps, rho).vs;
      VecD fd = w == 0 ? fd_vs2(id, eps, rho, delta) : fd_vs2(gl, eps, rho, delta);
      Real worst = 0.0;
      for (int i = 0; i < 10; ++i) worst = std::max(worst, rel(fd[i], vs[i]));
      res.push_back({w == 0 ? "ideal" : "gamma_law", worst});
    }
  }
  // cold thermal, both variants; points at cell centres, thermal energy above the cold curve
  const VecD ratios = {0.8, 2.2, 3.4, 5.0, 5.6, 6.5, 7.4, 8.4, 9.5, 12.0};
  VecD rho_tp;
  for (Real r : ratios) rho_tp.push_back(cell_centre(r));
  VecD rho_mx = {0.80, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 0.45, 1.9};  // 8 inside the coexistence interval, 2 outside
  for (auto& r : rho_mx) r = (std::floor(r / table().d_rho) + 0.5) * table().d_rho;
  for (int w = 0; w < 2; ++w) {
    auto eos = w == 0 ? two_phase() : maxwell();
    const VecD& rho = w == 0 ? rho_tp : rho_mx;
    auto c = cold(eos, rho);
    VecD eps;
    for (size_t i = 0; i < rho.size(); ++i) eps.push_back(c.ec[i] + 0.1 + 0.2 * i);
    auto vs = eval(eos, eps, rho).vs;
    auto fd = fd_vs2(eos, eps, rho, delta);
    Real worst = 0.0;
    for (size_t i = 0; i < rho.size(); ++i) worst = std::max(worst, rel(fd[i], vs[i]));
    res.push_back({w == 0 ? "cold_thermal_two_phase" : "cold_thermal_maxwell", worst});
  }
  for (const auto& r : res) {
    std::printf("[T2.3] %s: max rel(vs2 - FD) = %.2e (10 points, delta = %.0e fm^-3)\n", r.name, r.worst, delta);
    CHECK(r.worst < 1e-6);
  }
}

TEST_CASE("T2.4 reference points of the development EoS") {
  TableGuard guard;
  const auto ref = reference();
  const auto pts = ref["cold_thermal_points"];
  VecD eps, rho;
  for (const auto& p : pts) {
    rho.push_back(Real(p["rho_over_rho_s"]) * kRhoS);
    eps.push_back(Real(p["eps_over_eps_s"]) * kEpsS);
  }
  auto o = eval(two_phase(), eps, rho);
  Real wp = 0.0, wh = 0.0, wv = 0.0;
  for (size_t i = 0; i < pts.size(); ++i) {
    const Real p_ref = Real(pts[i]["p0"]) / 1000.0, h_ref = Real(pts[i]["h"]) / 1000.0, v_ref = pts[i]["vs2"];
    wp = std::max(wp, rel(o.p[i], p_ref));
    wh = std::max(wh, rel(eps[i] + o.p[i], h_ref));
    wv = std::max(wv, std::fabs(o.vs[i] - v_ref));
    if (std::fabs(Real(pts[i]["rho_over_rho_s"]) - 5.8) < 1e-9 && std::fabs(Real(pts[i]["eth_over_eps_s"]) - 2.0) < 1e-9)
      std::printf("[T2.4] (5.8 rho_s, 2.0 eps_s): vs2=%.5f (ref -0.05428), h=%.5f GeV/fm3 (ref 1.98840)\n", o.vs[i],
                  eps[i] + o.p[i]);
  }
  std::printf("[T2.4] %zu points: max rel dp0=%.2e, max rel dh=%.2e, max |dvs2|=%.2e\n", pts.size(), wp, wh, wv);
  CHECK(pts.size() == 21);
  CHECK(wp < 1e-5);
  CHECK(wh < 1e-5);
  CHECK(wv < 2e-3);
}

TEST_CASE("T2.5 saturation point") {
  TableGuard guard;
  const auto ref = reference();
  const Real ea_ref = Real(ref["hadron_interaction"]["E_over_A_saturation"]) / 1000.0;
  auto c = cold(two_phase(), {kRhoS});
  const Real pc = c.pc[0], ea = c.ec[0] / kRhoS - 0.940;
  std::printf("[T2.5] p_c(rho_s)=%.3e GeV/fm3; eps_c/rho_s - 0.940 = %.6f GeV (reference %.6f, brief -0.01187)\n", pc, ea, ea_ref);
  CHECK(std::fabs(pc) < 1e-4);
  CHECK(std::fabs(ea - (-0.01187)) < 1e-4);
  CHECK(std::fabs(ea - ea_ref) < 1e-4);
}

TEST_CASE("T2.6 clamp below the cold curve") {
  TableGuard guard;
  VecD rho;
  for (Real r : {0.5, 2.0, 4.0, 6.0, 7.5, 9.0, 11.0, 14.0}) rho.push_back(r * kRhoS * 1.0001);
  for (int w = 0; w < 2; ++w) {
    auto eos = w == 0 ? two_phase() : maxwell();
    auto c = cold(eos, rho);
    VecD eps_lo, eps_hi;
    for (size_t i = 0; i < rho.size(); ++i) {
      eps_lo.push_back(c.ec[i] - 0.1 - 0.05 * i);
      eps_hi.push_back(c.ec[i] + 0.1);
    }
    auto lo = eval(eos, eps_lo, rho), hi = eval(eos, eps_hi, rho);
    Real dpmax = 0.0, dpe = 0.0, dpr = 0.0;
    for (size_t i = 0; i < rho.size(); ++i) {
      dpmax = std::max(dpmax, std::fabs(lo.p[i] - c.pc[i]));
      dpe = std::max(dpe, std::fabs(lo.dpe[i]));
      dpr = std::max(dpr, std::fabs(lo.dpr[i] - c.dpc[i]));
      CHECK(hi.dpe[i] == doctest::Approx(4.0 / 3.0 - 1.0));
      CHECK(hi.p[i] == doctest::Approx(c.pc[i] + 0.1 / 3.0));
    }
    std::printf("[T2.6] %s: max|p0 - p_c|=%.1e, max|dp_deps|=%.1e, max|dp_drho - p_c'|=%.1e in the clamped region\n",
                w == 0 ? "two_phase" : "maxwell", dpmax, dpe, dpr);
    CHECK(dpmax == 0.0);
    CHECK(dpe == 0.0);
    CHECK(dpr == 0.0);
  }
}
