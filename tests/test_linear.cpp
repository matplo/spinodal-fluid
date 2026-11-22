// WP8: linear-response library (arXiv:1007.1448). Tests T8.1-T8.5, docs/validation.md "WP8".
#include "spf/linear/amplification.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>

using namespace spf::linear;

namespace {
const double rho_s = 0.153, m_N = 0.940;
const ToyVdw toy(5.0 * rho_s, 0.100);  // rho_c = 5 rho_s, T_c = 100 MeV
// toy-EoS temperature of tests T8.1-T8.3 (0.35 T_c): both an isentropic spinodal and a band vT2 < 0 < vs2 exist there
const double kTspin = 0.035;
double Cgrad() { return gradient_C(0.033, m_N, rho_s); }

// rho at fixed T: interval where vs2 < 0 and interval where vT2 < 0 < vs2 (scan over 0.5..12 rho_s, step 0.001 rho_s)
struct Region { double rho_vs = 0, rho_vT = 0; };
Region find_points(double T) {
  Region r;
  double best = 0;
  for (double x = 0.5; x < 12.0; x += 0.001) {
    const auto p = toy.at(x * rho_s, T);
    if (p.vs2 < best) { best = p.vs2; r.rho_vs = x * rho_s; }
  }
  // the lower-density band vT2 < 0 < vs2 (below the isentropic core)
  double lo = 1e9, hi = -1e9;
  for (double x = 0.5; x * rho_s < r.rho_vs; x += 0.001) {
    const auto p = toy.at(x * rho_s, T);
    if (p.vT2 < 0 && p.vs2 > 0) { lo = std::min(lo, x); hi = std::max(hi, x); }
  }
  r.rho_vT = 0.5 * (lo + hi) * rho_s;
  return r;
}
}  // namespace

TEST_CASE("T8.0 constants, transport, cubic solver, spline particle density") {
  CHECK(std::abs(c0_constant() - 0.12779) < 5e-6);
  std::printf("T8.0 c0 = %.6f\n", c0_constant());
  const ThermoPoint p = toy.at(6.5 * rho_s, 0.07);
  Transport tr; tr.eta0 = 2; tr.kappa0 = 3;
  const double d = std::cbrt(1.0 / p.n);
  CHECK(lambda_visc(tr, p) == doctest::Approx(4.0 / 3.0 * 2 * tr.c0 * d).epsilon(1e-14));
  CHECK(lambda_heat(tr, p) == doctest::Approx(3 * tr.c0 * d).epsilon(1e-14));
  CHECK(eta_of(tr, p) == doctest::Approx(2 * tr.c0 * d * p.h).epsilon(1e-14));
  CHECK(kappa_of(tr, p) / p.cv == doctest::Approx(lambda_heat(tr, p)).epsilon(1e-14));
  CHECK(4.0 / 3.0 * eta_of(tr, p) / p.h == doctest::Approx(lambda_visc(tr, p)).epsilon(1e-14));
  // random parameters: residual and Vieta relations of the three roots, and agreement with the original (uncleared) equation
  std::mt19937_64 gen(7);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  double worst_res = 0, worst_vieta = 0, worst_orig = 0;
  for (int i = 0; i < 20000; ++i) {
    DispersionParams dp;
    dp.vT2 = (U(gen) - 0.7) * 0.5; dp.vs2 = dp.vT2 + U(gen) * 0.4 - 0.1; dp.B = U(gen) * 0.05;
    dp.lam_visc = U(gen) < 0.3 ? 0.0 : U(gen) * 2; dp.lam_heat = std::pow(10.0, -4.0 + 5.0 * U(gen));
    const double k = std::pow(10.0, -2.0 + 2.5 * U(gen));
    std::complex<double> a2, a1, a0;
    cubic_coefficients(dp, k, a2, a1, a0);
    const auto r = omega_roots(dp, k);
    REQUIRE(r.size() == 3);
    const double scale = std::max({1.0, std::abs(a2), std::abs(a1), std::abs(a0)});
    for (const auto& w : r) {
      const auto f = ((w + a2) * w + a1) * w + a0;
      worst_res = std::max(worst_res, std::abs(f) / (scale * std::max(1.0, std::pow(std::abs(w), 3))));
      // original form of Eq. (33): w^2 - [vT2 k^2 + B k^4 - i nu w + (vs2 - vT2) k^2 / (1 + i q/w)]
      const double k2 = k * k, nu = dp.lam_visc * k2, q = dp.lam_heat * k2;
      const auto g = w * w - (dp.vT2 * k2 + dp.B * k2 * k2 - std::complex<double>(0, 1) * nu * w + (dp.vs2 - dp.vT2) * k2 / (1.0 + std::complex<double>(0, 1) * q / w));
      worst_orig = std::max(worst_orig, std::abs(g) / (scale * std::max(1.0, std::pow(std::abs(w), 2))));
    }
    worst_vieta = std::max(worst_vieta, std::abs(r[0] + r[1] + r[2] + a2) / scale);
    worst_vieta = std::max(worst_vieta, std::abs(r[0] * r[1] * r[2] + a0) / scale);
  }
  std::printf("T8.0 cubic: worst scaled residual %.3e, Vieta %.3e, original Eq.(33) %.3e\n", worst_res, worst_vieta, worst_orig);
  CHECK(worst_res < 1e-13);
  CHECK(worst_vieta < 1e-12);
  CHECK(worst_orig < 1e-10);
  CHECK(particle_density_spline("linear_in_rho", 6.0, 4.0, 8.0, 0.2, 0.6) == doctest::Approx(0.4));
  CHECK(particle_density_spline("hadron_side", 6.0, 4.0, 8.0, 0.2, 0.6) == 0.2);
  CHECK(particle_density_spline("plasma_side", 6.0, 4.0, 8.0, 0.2, 0.6) == 0.6);
  CHECK_THROWS(particle_density_spline("x", 6.0, 4.0, 8.0, 0.2, 0.6));
}

TEST_CASE("T8.1 ideal limit: gamma^2 = |vs2| k^2 - B k^4 inside the isentropic spinodal, real omega outside") {
  const Region reg = find_points(kTspin);
  const ThermoPoint pin = toy.at(reg.rho_vs, kTspin);
  REQUIRE(pin.vs2 < 0);
  DispersionParams d = make_dispersion(pin, Cgrad(), Transport{0, 0});
  double worst = 0, worst_outside = 0;
  int nin = 0;
  for (double k = 0.01; k < 6.0; k *= 1.07) {
    const double g2 = -pin.vs2 * k * k - d.B * std::pow(k, 4);
    const auto r = omega_roots(d, k);
    if (g2 > 0) {
      ++nin;
      worst = std::max(worst, std::abs(growth_rate(d, k) - std::sqrt(g2)) / std::sqrt(g2));
      worst = std::max(worst, std::abs(r[0].imag() + std::sqrt(g2)) / std::sqrt(g2));  // the damped partner
      worst = std::max(worst, std::max(std::abs(r[0].real()), std::abs(r[1].real())));
    } else {
      worst_outside = std::max(worst_outside, std::max(std::abs(r[0].imag()), std::abs(r[1].imag())));
      CHECK(growth_rate(d, k) == 0.0);
    }
  }
  // stable background: purely real omega at all k
  const ThermoPoint pst = toy.at(1.0 * rho_s, kTspin);
  REQUIRE(pst.vs2 > 0);
  const DispersionParams ds = make_dispersion(pst, Cgrad(), Transport{0, 0});
  double worst_stable = 0;
  for (double k = 0.01; k < 6.0; k *= 1.07) for (const auto& w : omega_roots(ds, k)) worst_stable = std::max(worst_stable, std::abs(w.imag()));
  std::printf("T8.1 point rho = %.4f rho_s, vs2 = %.5f: %d unstable k, worst relative deviation %.3e; outside-band |Im w| %.3e; stable background |Im w| %.3e\n",
              reg.rho_vs / rho_s, pin.vs2, nin, worst, worst_outside, worst_stable);
  CHECK(worst < 1e-12);
  CHECK(worst_outside == 0.0);
  CHECK(worst_stable == 0.0);
}

TEST_CASE("T8.2 small viscosity shifts Im omega by -(1/2) lam_visc k^2") {
  const Region reg = find_points(kTspin);
  double worst = 0;
  for (const double rho : {reg.rho_vs, 0.5 * (reg.rho_vs + 1.0 * rho_s), 1.0 * rho_s}) {
    const ThermoPoint p = toy.at(rho, kTspin);
    DispersionParams d0 = make_dispersion(p, Cgrad(), Transport{0, 0});
    for (const double k : {0.3, 0.8, 1.5}) {
      const double lam = 1e-7;
      DispersionParams d1 = d0; d1.lam_visc = lam;
      const auto r0 = omega_roots(d0, k), r1 = omega_roots(d1, k);
      // pair the roots by real part / imaginary part ordering
      std::vector<std::complex<double>> a = r0, b = r1;
      auto key = [](const std::complex<double>& x, const std::complex<double>& y) { return x.real() != y.real() ? x.real() < y.real() : x.imag() < y.imag(); };
      std::sort(a.begin(), a.end(), key); std::sort(b.begin(), b.end(), key);
      const bool unstable = p.vs2 * k * k + d0.B * std::pow(k, 4) < 0;
      for (int j = 0; j < 2; ++j) {
        // for an unstable (purely imaginary) pair the growing root moves by -nu/2 + O(nu^2); the damped one likewise
        const double slope = (b[j].imag() - a[j].imag()) / lam;
        if (unstable && std::abs(a[j].imag()) < 1e-12) continue;
        worst = std::max(worst, std::abs(slope / (-0.5 * k * k) - 1.0));
      }
      if (unstable) {  // imaginary roots: both shift by -nu/2 to first order
        const double sg = (growth_rate(d1, k) - growth_rate(d0, k)) / lam;
        worst = std::max(worst, std::abs(sg / (-0.5 * k * k) - 1.0));
      }
    }
  }
  std::printf("T8.2 worst relative deviation of the numerical slope from -k^2/2: %.3e\n", worst);
  CHECK(worst < 1e-4);
}

TEST_CASE("T8.3 heat conduction: instability boundary moves from vs2 = 0 to vT2 = 0, growth increases at large k (toy EoS)") {
  const double T = kTspin;
  const Region reg = find_points(T);
  Transport ideal{0, 0}, heat{0, 1};
  // (a) k -> 0: sign of gamma follows vs2 (kappa = 0) or vT2 (kappa > 0) over a density scan
  int n_vs_ok = 0, n_vT_ok = 0, n = 0, n_between = 0;
  const double k0 = 1e-3;
  for (double x = 0.5; x < 11.0; x += 0.05) {
    const ThermoPoint p = toy.at(x * rho_s, T);
    if (std::abs(p.vs2) < 1e-6 || std::abs(p.vT2) < 1e-6) continue;
    ++n;
    const double g0 = growth_rate(make_dispersion(p, Cgrad(), ideal), k0);
    const double g1 = growth_rate(make_dispersion(p, Cgrad(), heat), k0);
    if ((g0 > 0) == (p.vs2 < 0)) ++n_vs_ok;
    if ((g1 > 0) == (p.vT2 < 0)) ++n_vT_ok;
    if (p.vT2 < 0 && p.vs2 > 0) ++n_between;
  }
  std::printf("T8.3a scan at T = 35 MeV: %d points; sign(gamma, kappa=0) = sign(-vs2) at %d; sign(gamma, kappa>0) = sign(-vT2) at %d; %d points with vT2<0<vs2\n", n, n_vs_ok, n_vT_ok, n_between);
  CHECK(n_vs_ok == n);
  CHECK(n_vT_ok == n);
  CHECK(n_between > 10);
  // (b) quantitative small-k law in the band vT2 < 0 < vs2: gamma -> lam_heat k^2 |vT2|/vs2
  const ThermoPoint pb = toy.at(reg.rho_vT, T);
  REQUIRE(pb.vT2 < 0);
  REQUIRE(pb.vs2 > 0);
  const DispersionParams db = make_dispersion(pb, Cgrad(), heat);
  const double kk = 1e-3;
  const double g = growth_rate(db, kk), pred = db.lam_heat * kk * kk * (-pb.vT2) / pb.vs2;
  std::printf("T8.3b rho = %.4f rho_s: vT2 = %.5f, vs2 = %.5f, lam_heat = %.5f fm; gamma(k=1e-3) = %.6e vs lam k^2|vT2|/vs2 = %.6e (rel diff %.2e)\n",
              reg.rho_vT / rho_s, pb.vT2, pb.vs2, db.lam_heat, g, pred, std::abs(g / pred - 1));
  CHECK(std::abs(g / pred - 1.0) < 1e-5);
  CHECK(growth_rate(make_dispersion(pb, Cgrad(), ideal), kk) <= 0.0);
  // (c) large k at the point of strongest ideal instability: gamma(kappa>0) >= gamma(kappa=0) for k above the ideal maximum
  const ThermoPoint pc = toy.at(reg.rho_vs, T);
  const DispersionParams di = make_dispersion(pc, Cgrad(), ideal), dh = make_dispersion(pc, Cgrad(), heat);
  const GammaMax mi = max_growth(di, 0.01, 6.0, 1200), mh = max_growth(dh, 0.01, 6.0, 1200);
  int ncmp = 0, nbad = 0;
  double kcut_i = 0, kcut_h = 0;
  for (int i = 0; i < 1200; ++i) {
    const double k = 0.01 + (6.0 - 0.01) * i / 1199.0;
    const double gi = growth_rate(di, k), gh = growth_rate(dh, k);
    if (gi > 0) kcut_i = k;
    if (gh > 0) kcut_h = k;
    if (k > mi.k && (gi > 0 || gh > 0)) { ++ncmp; if (gh < gi - 1e-12) ++nbad; }
  }
  std::printf("T8.3c rho = %.4f rho_s (min vs2 = %.5f): ideal gamma_max = %.5f at k = %.4f, kappa gamma_max = %.5f at k = %.4f; "
              "k_cut ideal %.4f, with heat %.4f; k > k_max(ideal) with gamma > 0 in either case: gamma_heat < gamma_ideal at %d of %d points\n",
              reg.rho_vs / rho_s, pc.vs2, mi.gamma, mi.k, mh.gamma, mh.k, kcut_i, kcut_h, nbad, ncmp);
  CHECK(nbad == 0);
  CHECK(kcut_h > kcut_i);
  CHECK(mh.gamma >= mi.gamma);
}

TEST_CASE("T8.4 Lalime equation and G_k") {
  const double gam = 0.37, D = 0.021, s0 = 1.3;
  std::vector<double> t, g;
  for (int i = 0; i <= 40; ++i) { t.push_back(0.25 * i); g.push_back(gam); }
  const auto s = lalime_solve(t, g, [&](std::size_t, double, double) { return D; }, s0);
  double worst = 0;
  for (std::size_t i = 0; i < t.size(); ++i) {
    const double ex = lalime_stationary(gam, D, s0, t[i] - t[0]);
    worst = std::max(worst, std::abs(s[i] - ex) / ex);
  }
  // stable constant gamma: relaxation to -D/gamma
  std::vector<double> g2(t.size(), -0.8);
  const auto s2 = lalime_solve(t, g2, [&](std::size_t, double, double) { return D; }, 5.0);
  const double eq = D / 0.8;
  const double worst_eq = std::abs(s2.back() - lalime_stationary(-0.8, D, 5.0, t.back() - t.front())) / eq;
  const double relax = std::abs(s2.back() - eq) / ((5.0 - eq) * std::exp(-1.6 * (t.back() - t.front())));  // = 1 for exact relaxation
  CHECK(std::abs(relax - 1.0) < 1e-9);
  // G_k = exp(gamma t_unstable): gamma = +0.37 for 4 < t < 8, negative elsewhere with a zero-crossing inside a step
  std::vector<double> tt, gg;
  for (int i = 0; i <= 80; ++i) { tt.push_back(0.25 * i); const double x = 0.25 * i; gg.push_back(x >= 4.0 && x <= 8.0 ? 0.37 : -0.5); }
  // piecewise constant with jumps on the grid nodes -> use exact node values; zero-crossing steps contribute 0.5*gp*frac*dt
  const double lnG = log_amplification(tt, gg);
  // pure plateau test: a trajectory with exactly constant unstable gamma
  std::vector<double> tp, gp;
  for (int i = 0; i <= 32; ++i) { tp.push_back(0.25 * i); gp.push_back(0.37); }
  const double Gp = std::exp(log_amplification(tp, gp));
  const double err_G = std::abs(Gp / std::exp(0.37 * 8.0) - 1.0);
  // linear ramp crossing zero: gamma = 0.4 (t - 3) on [0, 8]; positive part integral = 0.4 * 5^2 / 2 exactly
  std::vector<double> tr_, gr_;
  for (int i = 0; i <= 8; ++i) { tr_.push_back(i); gr_.push_back(0.4 * (i - 3.0)); }
  const double ramp = log_amplification(tr_, gr_), ramp_ex = 0.4 * 25.0 / 2.0;
  // unit steps with the zero crossing inside a step: gamma = 0.4 (t - 3.5)
  for (int i = 0; i <= 8; ++i) gr_[i] = 0.4 * (i - 3.5);
  const double ramp2 = log_amplification(tr_, gr_), ramp2_ex = 0.4 * 4.5 * 4.5 / 2.0;
  std::printf("T8.4 Lalime vs Eq.(48): %.3e; relaxation to -D/gamma: %.3e; G_k plateau: %.3e; ramp (node crossing) %.3e, ramp (in-step crossing) %.3e; stepped lnG = %.4f\n",
              worst, worst_eq, err_G, std::abs(ramp / ramp_ex - 1), std::abs(ramp2 / ramp2_ex - 1), lnG);
  CHECK(worst < 1e-10);
  CHECK(worst_eq < 1e-12);
  CHECK(err_G < 1e-10);
  CHECK(std::abs(ramp / ramp_ex - 1) < 1e-12);
  CHECK(std::abs(ramp2 / ramp2_ex - 1) < 1e-12);
  // default diffusion model: starts at equilibrium and stays there for a stable trajectory
  const auto s3 = lalime_solve(t, g2, einstein_hold_diffusion(2.0), 2.0);
  CHECK(std::abs(s3.back() / 2.0 - 1.0) < 1e-12);
  // cumulative Gamma of constant gamma
  const auto Gc = cumulative_gamma(t, g);
  CHECK(Gc.back() == doctest::Approx(gam * 10.0).epsilon(1e-13));
}

TEST_CASE("T8.5 ideal-fluid numbers of the paper's Fig. 2 point") {
  const double t0 = 1.38, k0 = 2.0;  // paper: fastest growth time (fm/c) and wave number (fm^-1)
  const double gmax = 1.0 / t0;
  // ideal: gamma^2 = |vs2| k^2 - B k^4 is maximal at k0^2 = |vs2|/(2B) with gamma_max^2 = vs2^2/(4B)
  const double B = gmax * gmax / std::pow(k0, 4);
  const double vs2 = 2.0 * B * k0 * k0;
  std::printf("T8.5 re-derived: B = C rho0^2/h0 = %.5f fm^2, |vs| = %.5f\n", B, std::sqrt(vs2));
  CHECK(B == doctest::Approx(0.0328).epsilon(2e-3));
  CHECK(std::sqrt(vs2) == doctest::Approx(0.512).epsilon(2e-3));
  DispersionParams d; d.vs2 = -vs2; d.vT2 = -vs2; d.B = B;
  const GammaMax m = max_growth(d, 0.5, 4.0, 3501);
  std::printf("T8.5 library ideal curve: gamma_max = %.5f /fm (t = %.4f fm/c) at k = %.4f /fm\n", m.gamma, 1.0 / m.gamma, m.k);
  CHECK(m.gamma == doctest::Approx(gmax).epsilon(1e-6));
  CHECK(m.k == doctest::Approx(k0).epsilon(1e-3));
  // implied finite-range parameter a for h0 = m_N rho0 (cold limit) and for h0 = 0.2.. ; reported only
  const double rho0 = 6.5 * rho_s;
  const double a_cold = std::sqrt(B * (m_N * rho0) * rho_s / (m_N * rho0 * rho0));
  std::printf("T8.5 B = a^2 m_N rho0^2/(rho_s h0): with h0 = m_N rho0 the implied a = %.4f fm (gradient.a_fm default 0.033 fm gives B = %.5f fm^2)\n",
              a_cold, gradient_C(0.033, m_N, rho_s) * rho0 * rho0 / (m_N * rho0));

}

// T8.5b: expected failure, kept on purpose (rule: a failing acceptance test is reported, not loosened). The tolerance 0.01 was fixed before
// the comparison. Measured: max distance 0.0105 (k = 2.830, where the digitised curve ends 0.011 fm^-1 beyond the library cut-off 2.8284).
// The cut-off follows from the rounded paper numbers (t0 = 1.38 fm/c, k0 = 2.0 fm^-1): k_cut = sqrt(|vs2|/B) = sqrt(2) k0, so 0.4 % in k0
// moves it by 0.011. A first attempt with the vertical deviation at fixed k gave 0.115 (ill-conditioned where the curve is vertical).
TEST_CASE("T8.5b digitised ideal curve of Fig. 2 vs library with the rounded paper numbers" * doctest::should_fail()) {
#ifdef SPF_SOURCE_DIR
  {
    const double t0 = 1.38, k0 = 2.0;
    const double B = std::pow(1.0 / t0, 2) / std::pow(k0, 4), vs2 = 2.0 * B * k0 * k0;
    DispersionParams d; d.vs2 = -vs2; d.vT2 = -vs2; d.B = B;
    // digitised ideal curve of Fig. 2 (tools/digitise_1007_1448.py): compare with the library at the derived (B, vs2); tolerance 0.01 for the distance in the (k [fm^-1], gamma [c/fm]) plane
    // (about 1.4 % of the peak; the digitisation resolution is 0.007 c/fm per PDF point, the paper quotes t0 to three digits)
    std::ifstream in(std::string(SPF_SOURCE_DIR) + "/tools/data/fig2_gamma_k_eta0_0_kappa0_0.csv");
    REQUIRE(bool(in));
    std::vector<std::pair<double, double>> lib;  // library curve sampled finely; compared by distance in the (k, gamma) plane
    for (double k = 0.0; k < 3.2; k += 2e-4) lib.emplace_back(k, std::max(growth_rate(d, k), 0.0));
    std::string line; double worst = 0, worst_g = 0, gdig_max = 0, kdig = 0, kend = 0; int n = 0;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#' || !std::isdigit(static_cast<unsigned char>(line[0]))) continue;
      std::istringstream ss(line); char c; double k, gg;
      ss >> k >> c >> gg;
      ++n;
      if (gg > gdig_max) { gdig_max = gg; kdig = k; }
      if (gg > 1e-3) kend = k;
      worst_g = std::max(worst_g, std::abs(gg - std::max(growth_rate(d, k), 0.0)));  // vertical deviation: ill-conditioned at the cut-off
      double dmin = 1e9;
      for (const auto& q : lib) dmin = std::min(dmin, std::hypot(q.first - k, q.second - gg));
      worst = std::max(worst, dmin);
    }
    std::printf("T8.5 digitised Fig. 2 (0,0) curve: %d points, gamma_max = %.4f at k = %.3f, last gamma > 0.001 at k = %.3f (library k_cut = %.4f); max distance of a digitised point from the library curve = %.4f (vertical deviation at fixed k, ill-conditioned at the cut-off: %.4f)\n",
                n, gdig_max, kdig, kend, std::sqrt(vs2 / B), worst, worst_g);
    CHECK(n > 100);
    CHECK(worst < 0.01);
  }
#endif
}

// T8.5c: the digitised (eta0, kappa0) = (1, 0) curve of Fig. 2 against the viscous quadratic. With the ideal numbers (B, vs2) from T8.5 the model
// gamma (gamma + lam k^2) = |vs2| k^2 - B k^4 gives a constant lam = (|S|/gamma - gamma)/k^2 at every point if the library's functional form holds.
// Check: relative scatter of lam over the digitised points with 0.4 < k < 2.5 and gamma > 0.05 below 10 % (fixed before the run); the implied
// interparticle spacing d = 3 lam / (4 c0 eta0) is reported (no check: it needs the finite-temperature table).
TEST_CASE("T8.5c viscous Fig. 2 curve gives a constant lam_visc") {
#ifdef SPF_SOURCE_DIR
  const double t0 = 1.38, k0 = 2.0, B = std::pow(1.0 / t0, 2) / std::pow(k0, 4), vs2 = 2.0 * B * k0 * k0;
  std::ifstream in(std::string(SPF_SOURCE_DIR) + "/tools/data/fig2_gamma_k_eta0_1_kappa0_0.csv");
  REQUIRE(bool(in));
  std::string line; std::vector<double> lam, kk;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#' || !std::isdigit(static_cast<unsigned char>(line[0]))) continue;
    std::istringstream ss(line); char c; double k, g;
    ss >> k >> c >> g;
    if (k < 0.4 || k > 2.5 || g < 0.05) continue;
    lam.push_back((vs2 * k * k - B * std::pow(k, 4)) / g - g);
    lam.back() /= k * k; kk.push_back(k);
  }
  REQUIRE(lam.size() > 20);
  double m = 0, s2 = 0;
  for (double v : lam) m += v;
  m /= lam.size();
  for (double v : lam) s2 += (v - m) * (v - m);
  const double sd = std::sqrt(s2 / lam.size());
  // round trip through the library: gamma(lam_visc = m) at the digitised k values reproduces the curve
  DispersionParams d; d.vs2 = -vs2; d.vT2 = -vs2; d.B = B; d.lam_visc = m;
  const GammaMax gm = max_growth(d, 0.1, 4.0, 3901);
  const double c0 = c0_constant();
  std::printf("T8.5c %zu points: lam_visc = %.4f fm, relative scatter %.3f; library with this lam: gamma_max = %.4f at k = %.3f (digitised 0.5067 at ~1.7); implied d = %.3f fm (n = %.3f fm^-3) for eta0 = 1\n",
              lam.size(), m, sd / m, gm.gamma, gm.k, 0.75 * m / c0, std::pow(0.75 * m / c0, -3));
  CHECK(sd / m < 0.10);
#endif
}

// T8.5d: prediction of the three dissipative Fig. 2 curves from quantities read off the ideal and the viscous curves; the thermodynamic inputs
// v_T^2 and c_v of the finite-temperature table are NOT needed: lam_heat = kappa0 c0 d does not contain c_v, and v_T^2 follows from the
// heat-conduction cut-off (gamma = 0 means w = 0, i.e. A = v_T^2 k^2 + B k^4 = 0, so v_T^2 = -B k_cut^2, k_cut of the digitised (0,1) curve).
// Inputs: B, vs2 (ideal numbers of T8.5), lam_visc (T8.5c, from the (1,0) curve), d = 3 lam_visc/(4 c0 eta0), then
// lam_heat = kappa0 c0 d (= 0.75 lam_visc for eta0 = kappa0 = 1; this ratio is the prediction), v_T^2 from the (0,1) cut-off.
// Compared curves: (1,0) [round trip of T8.5c], (0,1) and (1,1) [predictions]; tolerance 0.02 for the maximum distance of a digitised point
// from the library curve in the (k, gamma) plane (fixed before the run).
namespace {
std::vector<std::pair<double, double>> read_curve(const std::string& name) {
  std::vector<std::pair<double, double>> v;
  std::ifstream in(std::string(SPF_SOURCE_DIR) + "/tools/data/" + name);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#' || !std::isdigit(static_cast<unsigned char>(line[0]))) continue;
    std::istringstream ss(line); char c; double k, g;
    ss >> k >> c >> g;
    v.emplace_back(k, g);
  }
  return v;
}
double curve_distance(const DispersionParams& d, const std::vector<std::pair<double, double>>& dig, double kmax) {
  std::vector<std::pair<double, double>> lib;
  for (double k = 0.0; k < kmax; k += 4e-4) lib.emplace_back(k, std::max(growth_rate(d, k), 0.0));
  double worst = 0;
  for (const auto& p : dig) {
    double dmin = 1e9;
    for (const auto& q : lib) dmin = std::min(dmin, std::hypot(q.first - p.first, q.second - p.second));
    worst = std::max(worst, dmin);
  }
  return worst;
}
}  // namespace

TEST_CASE("T8.5d Fig. 2 heat-conduction curves predicted from the ideal and viscous curves") {
#ifdef SPF_SOURCE_DIR
  const double t0 = 1.38, k0 = 2.0, B = std::pow(1.0 / t0, 2) / std::pow(k0, 4), vs2 = 2.0 * B * k0 * k0;
  const auto c10 = read_curve("fig2_gamma_k_eta0_1_kappa0_0.csv"), c01 = read_curve("fig2_gamma_k_eta0_0_kappa0_1.csv"), c11 = read_curve("fig2_gamma_k_eta0_1_kappa0_1.csv");
  REQUIRE(c10.size() > 100);
  // lam_visc as in T8.5c
  double m = 0; int n = 0;
  for (const auto& p : c10) {
    if (p.first < 0.4 || p.first > 2.5 || p.second < 0.05) continue;
    m += ((vs2 * p.first * p.first - B * std::pow(p.first, 4)) / p.second - p.second) / (p.first * p.first); ++n;
  }
  const double lam_visc = m / n, c0 = c0_constant();
  const double d_fm = 0.75 * lam_visc / c0, lam_heat = 1.0 * c0 * d_fm;
  // cut-off of the (0,1) curve: last digitised point with gamma > 0.001 plus the interpolated end on the axis
  double kcut = 0;
  for (const auto& p : c01) if (p.second > 1e-3) kcut = std::max(kcut, p.first);
  // refine with the axis end point (gamma ~ 0)
  for (const auto& p : c01) if (p.second <= 1e-3 && p.first > 3.0) kcut = std::max(kcut, p.first);
  const double vT2 = -B * kcut * kcut;
  DispersionParams d10; d10.vs2 = -vs2; d10.vT2 = -vs2; d10.B = B; d10.lam_visc = lam_visc;
  DispersionParams d01; d01.vs2 = -vs2; d01.vT2 = vT2; d01.B = B; d01.lam_heat = lam_heat;
  DispersionParams d11 = d01; d11.lam_visc = lam_visc;
  const double e10 = curve_distance(d10, c10, 3.2), e01 = curve_distance(d01, c01, 3.8), e11 = curve_distance(d11, c11, 3.8);
  const GammaMax g01 = max_growth(d01, 0.1, 3.4, 3301), g11 = max_growth(d11, 0.1, 3.4, 3301);
  double pk01 = 0, pk11 = 0;
  for (const auto& p : c01) pk01 = std::max(pk01, p.second);
  for (const auto& p : c11) pk11 = std::max(pk11, p.second);
  std::printf("T8.5d inputs: lam_visc = %.4f fm, d = %.4f fm, lam_heat = %.4f fm, vT2 = -B kcut^2 = %.4f (kcut = %.3f); "
              "max distance: (1,0) %.4f, (0,1) %.4f, (1,1) %.4f; peaks (library vs digitised): (0,1) %.4f at k = %.3f vs %.4f, (1,1) %.4f at k = %.3f vs %.4f\n",
              lam_visc, d_fm, lam_heat, vT2, kcut, e10, e01, e11, g01.gamma, g01.k, pk01, g11.gamma, g11.k, pk11);
  CHECK(e10 < 0.02);
  CHECK(e01 < 0.02);
  CHECK(e11 < 0.02);
#endif
}
