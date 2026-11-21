// WP3: 3+1D solver. T3.1 embedded 1D problems, T3.2 conservation, T3.3 blast octants, T3.4 sound wave in the cold-thermal EoS,
// T3.5 convergence of a smooth wave, T3.6 p_extra hook; plus recovery with p_extra and the snapshot writer.
#include <doctest/doctest.h>

#include "spf/config.hpp"
#include "spf/eos/eos.hpp"
#include "spf/eos/table.hpp"
#include "spf/hydro1d.hpp"
#include "spf/hydro3d.hpp"
#include "test_eos_ideal.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace spf;

namespace {

Config default_config() { return Config::from_file(SPF_DEFAULT_CONFIG); }

Hydro3DParams params_for(int nx, int ny, int nz, Real dx, const char* order, Boundary bc = Boundary::Periodic) {
  Config cfg = default_config();
  cfg.set_from_string("gradient.form=off");  // the WP3 tests are for the solver without the finite-range term (WP4 has its own tests)
  cfg.set_from_string(std::string("shasta.sweep_order=") + order);
  Hydro3DParams hp = Hydro3DParams::from_config(cfg);
  hp.nx = nx; hp.ny = ny; hp.nz = nz; hp.dx = dx;
  hp.shasta.bc = bc;
  return hp;
}

// max over cells of |a - b| / max(|a|, |b|), 0 for 0/0
Real max_rel_dev(const std::vector<Real>& a, const std::vector<Real>& b) {
  REQUIRE(a.size() == b.size());
  Real w = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const Real m = std::fmax(std::fabs(a[i]), std::fabs(b[i]));
    if (m > 0.0) w = std::fmax(w, std::fabs(a[i] - b[i]) / m);
  }
  return w;
}

// ------------------------------------------------------------------ T3.1: 1D problems embedded along dir
struct Line { std::vector<Real> eps, rho, v; };

Line run_1d(bool shock) {
  ShastaParams prm = ShastaParams::from_config(default_config());
  prm.bc = Boundary::Outflow;
  const int nx = shock ? 400 : 300;
  const Real dx = shock ? 100.0 / 400 : 1.0;
  Hydro1D<TestEosIdeal> h(TestEosIdeal{}, nx, dx, prm);
  std::vector<Real> e(nx), r(nx), u(nx, 0.0);
  for (int i = 0; i < nx; ++i) {
    if (shock) { const bool left = (i + 0.5) * dx < 50.0; e[i] = left ? 10.0 : 1.0; r[i] = left ? 1.0 : 0.5; }
    else { e[i] = i < 150 ? 1.0 : 0.0; r[i] = 0.0; }
  }
  h.set_state(e, r, u);
  const long nsteps = shock ? std::lround(40.0 / (prm.lambda * dx)) : 100;
  for (long s = 0; s < nsteps; ++s) h.step();
  h.update_primitives();
  return {h.line(h.eps), h.line(h.rho), h.line(h.v)};
}

Line run_embedded(bool shock, int dir, const char* order) {
  const int nl = shock ? 400 : 300;
  const Real dx = shock ? 100.0 / 400 : 1.0;
  int n[3] = {4, 4, 4};
  n[dir] = nl;
  Hydro3DParams hp = params_for(n[0], n[1], n[2], dx, order, Boundary::Periodic);
  Hydro3D<TestEosIdeal> h(TestEosIdeal{}, hp);
  h.bc[dir] = Boundary::Outflow;
  h.set_state_fn([&](int ix, int iy, int iz, Real& e, Real& r, Real& ux, Real& uy, Real& uz) {
    const int c[3] = {ix, iy, iz};
    const int i = c[dir];
    ux = uy = uz = 0.0;
    if (shock) { const bool left = (i + 0.5) * dx < 50.0; e = left ? 10.0 : 1.0; r = left ? 1.0 : 0.5; }
    else { e = i < 150 ? 1.0 : 0.0; r = 0.0; }
  });
  const long nsteps = shock ? std::lround(40.0 / (hp.shasta.lambda * dx)) : 100;
  for (long s = 0; s < nsteps; ++s) h.step();
  const long nf = h.update_primitives() + h.n_recover_fail_total;
  CHECK(nf == 0);
  // line through the transverse cell (1, 1)
  auto pick = [&](const Field3& F) {
    const std::vector<Real> all = h.interior_copy(F);
    std::vector<Real> out(nl);
    for (int i = 0; i < nl; ++i) {
      int c[3] = {1, 1, 1};
      c[dir] = i;
      out[i] = all[(static_cast<size_t>(c[0]) * n[1] + c[1]) * n[2] + c[2]];
    }
    return out;
  };
  const Field3& vd = dir == 0 ? h.vx : (dir == 1 ? h.vy : h.vz);
  Line l{pick(h.eps), pick(h.rho), pick(vd)};
  // transverse uniformity and zero transverse velocity
  const std::vector<Real> a = h.interior_copy(h.eps);
  Real spread = 0.0;
  for (int i = 0; i < nl; ++i)
    for (int q = 0; q < 16; ++q) {
      int c[3] = {q / 4, q % 4, 0};
      // map transverse pair onto the two other axes
      int cc[3];
      int t = 0;
      for (int d = 0; d < 3; ++d) cc[d] = (d == dir) ? i : (t++ == 0 ? c[0] : c[1]);
      const size_t id = (static_cast<size_t>(cc[0]) * n[1] + cc[1]) * n[2] + cc[2];
      int ref[3] = {1, 1, 1};
      ref[dir] = i;
      const size_t ir = (static_cast<size_t>(ref[0]) * n[1] + ref[1]) * n[2] + ref[2];
      spread = std::fmax(spread, std::fabs(a[id] - a[ir]));
    }
  CHECK(spread == 0.0);
  return l;
}

}  // namespace

TEST_CASE("T3.1 1D shock tube and vacuum expansion embedded along x, y, z reproduce the 1D solver") {
  for (int shock = 1; shock >= 0; --shock) {
    const Line ref = run_1d(shock != 0);
    for (int dir = 0; dir < 3; ++dir) {
      const Line f = run_embedded(shock != 0, dir, "fixed");
      const Line p = run_embedded(shock != 0, dir, "permuted");
      const Real df = std::fmax(std::fmax(max_rel_dev(f.eps, ref.eps), max_rel_dev(f.rho, ref.rho)), max_rel_dev(f.v, ref.v));
      const Real dp = std::fmax(std::fmax(max_rel_dev(p.eps, ref.eps), max_rel_dev(p.rho, ref.rho)), max_rel_dev(p.v, ref.v));
      std::printf("[WP3] T3.1 %s along %c: max relative deviation from the 1D solver: fixed %.3e  permuted %.3e\n",
                  shock ? "shock tube (400 cells)" : "vacuum expansion (300 cells)", "xyz"[dir], df, dp);
      CHECK(df < 1e-12);
    }
  }
}

namespace {
// Round trip with p_extra evaluated in a kernel: the EoS is a device function (device_rules.md, rule 9).
struct RoundTripPexK {
  EosIdeal eos;
  Kokkos::View<Real*> in, out;  // in: v, eps, pex per state (rho fixed); out: eps, rho, c, p, p_true, fail
  Real rho;
  KOKKOS_INLINE_FUNCTION void operator()(const int s) const {
    const Real v = in(3 * s), eps = in(3 * s + 1), pex = in(3 * s + 2);
    const Real p = eos.p0(eps, rho) + pex;
    const Real g2 = 1.0 / (1.0 - v * v), g = Kokkos::sqrt(g2);
    const Real R = g * rho, M = (eps + p) * g2 * v, E = (eps + p) * g2 - p;
    Real e2, r2, c2, p2;
    const int fl = recover_core(eos, R, Kokkos::fabs(M), E, 1e-12, pex, e2, r2, c2, p2);
    out(6 * s) = e2; out(6 * s + 1) = r2; out(6 * s + 2) = c2; out(6 * s + 3) = p2; out(6 * s + 4) = p; out(6 * s + 5) = fl;
  }
};
}  // namespace

TEST_CASE("recovery with an additive extra pressure: round trip") {
  const Real rho = 0.3;
  std::vector<Real> st;
  for (Real v : {0.0, 0.1, 0.5, 0.9, 0.99})
    for (Real eps : {1e-2, 1.0, 50.0})
      for (Real pex : {-0.01, 0.0, 0.3, 5.0}) {
        const Real p = eps / 3.0 + pex;
        if (p < 0.0 && eps + p <= 0.0) continue;
        st.push_back(v); st.push_back(eps); st.push_back(pex);
      }
  const int n = static_cast<int>(st.size() / 3);
  Kokkos::View<Real*> in("rtp_in", 3 * n), out("rtp_out", 6 * n);
  auto hin = Kokkos::create_mirror_view(in);
  for (size_t i = 0; i < st.size(); ++i) hin(i) = st[i];
  Kokkos::deep_copy(in, hin);
  Kokkos::parallel_for("rtp", Kokkos::RangePolicy<ExecSpace>(0, n), RoundTripPexK{EosIdeal{}, in, out, rho});
  auto hout = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(hout, out);
  Real worst = 0.0;
  int nfail = 0;
  for (int s = 0; s < n; ++s) {
    const Real v = st[3 * s], eps = st[3 * s + 1];
    nfail += static_cast<int>(hout(6 * s + 5));
    const Real p = hout(6 * s + 4);
    worst = std::fmax(worst, std::fabs(hout(6 * s) - eps) / eps + std::fabs(hout(6 * s + 1) - rho) / rho + std::fabs(hout(6 * s + 2) - v) +
                                 std::fabs(hout(6 * s + 3) - p) / (std::fabs(p) + 1e-12));
  }
  std::printf("[WP3] recovery with p_extra: %d states, failures %d, worst combined error %.3e\n", n, nfail, worst);
  CHECK(nfail == 0);
  CHECK(worst < 1e-9);
}

// ------------------------------------------------------------------ T3.2
TEST_CASE("T3.2 periodic 48^3 box, smooth random state: conservation over 500 steps") {
  const int n = 48;
  Hydro3DParams hp = params_for(n, n, n, 0.2, "permuted");
  Hydro3D<EosIdeal> h(EosIdeal{}, hp);
  h.set_state(smooth_random_state(n, n, n));
  const Totals t0 = h.totals();
  h.run(500);
  const Totals t1 = h.totals();
  const long nf = h.update_primitives() + h.n_recover_fail_total;
  const Real dR = std::fabs(t1.R - t0.R) / t0.R, dE = std::fabs(t1.E - t0.E) / t0.E;
  const Real dMx = std::fabs(t1.Mx - t0.Mx) / t0.E, dMy = std::fabs(t1.My - t0.My) / t0.E, dMz = std::fabs(t1.Mz - t0.Mz) / t0.E;
  std::printf("[WP3] T3.2 48^3, 500 steps: rel change R %.3e  E %.3e  Mx/E %.3e  My/E %.3e  Mz/E %.3e  recovery failures %ld  E<|M| cells %ld  clamps %ld\n",
              dR, dE, dMx, dMy, dMz, nf, h.count_E_lt_M(), h.n_clamped_total);
  std::printf("[WP3] T3.2 totals before: R %.12e E %.12e Mx %.6e My %.6e Mz %.6e\n", t0.R, t0.E, t0.Mx, t0.My, t0.Mz);
  CHECK(dR < 1e-11);
  CHECK(dE < 1e-11);
  CHECK(dMx < 1e-11);
  CHECK(dMy < 1e-11);
  CHECK(dMz < 1e-11);
  CHECK(nf == 0);
}

// ------------------------------------------------------------------ T3.3
namespace {
struct BlastResult { std::vector<std::vector<Real>> F; Real asym = 0.0; };

BlastResult run_blast(const char* order, int n, int nsteps, Real& asym_out) {
  Hydro3DParams hp = params_for(n, n, n, 0.2, order, Boundary::Outflow);
  Hydro3D<EosIdeal> h(EosIdeal{}, hp);
  const Real r2max = 2.0 * 2.0;
  h.set_state_fn([&](int ix, int iy, int iz, Real& e, Real& r, Real& ux, Real& uy, Real& uz) {
    const Real c = 0.5 * (n - 1);  // exact half-integer offsets: reflections are exact
    const Real dx2 = ((ix - c) * (ix - c) + (iy - c) * (iy - c) + (iz - c) * (iz - c)) * 0.2 * 0.2;
    e = dx2 < r2max ? 10.0 : 1.0;
    r = 0.3; ux = uy = uz = 0.0;
  });
  h.run(nsteps);
  CHECK(h.n_recover_fail_total == 0);
  BlastResult res;
  for (int c = 0; c < 5; ++c) res.F.push_back(h.interior_copy(h.U[c]));
  Real worst = 0.0;
  for (int c = 0; c < 5; ++c) {
    Real fmax = 0.0;
    for (Real x : res.F[c]) fmax = std::fmax(fmax, std::fabs(x));
    for (int s = 1; s < 8; ++s)
      for (int ix = 0; ix < n; ++ix)
        for (int iy = 0; iy < n; ++iy)
          for (int iz = 0; iz < n; ++iz) {
            const int jx = (s & 1) ? n - 1 - ix : ix, jy = (s & 2) ? n - 1 - iy : iy, jz = (s & 4) ? n - 1 - iz : iz;
            // momentum component c-1 changes sign under the reflection of its own axis
            Real par = 1.0;
            if (c == 1 && (s & 1)) par = -1.0;
            if (c == 2 && (s & 2)) par = -1.0;
            if (c == 3 && (s & 4)) par = -1.0;
            const Real a = res.F[c][(static_cast<size_t>(ix) * n + iy) * n + iz];
            const Real b = par * res.F[c][(static_cast<size_t>(jx) * n + jy) * n + jz];
            worst = std::fmax(worst, std::fabs(a - b) / fmax);
          }
  }
  asym_out = worst;
  return res;
}
}  // namespace

TEST_CASE("T3.3 spherical blast on 64^3: octant symmetry, fixed versus permuted sweep order") {
  const int n = 64, nsteps = 40;
  Real asym_fixed, asym_perm;
  const BlastResult f = run_blast("fixed", n, nsteps, asym_fixed);
  const BlastResult p = run_blast("permuted", n, nsteps, asym_perm);
  Real diff = 0.0;
  const char* names[5] = {"R", "Mx", "My", "Mz", "E"};
  Real per_field[5], l1_field[5];
  for (int c = 0; c < 5; ++c) {
    Real fmax = 0.0, d = 0.0, sum_d = 0.0, sum_a = 0.0;
    for (size_t i = 0; i < f.F[c].size(); ++i) {
      fmax = std::fmax(fmax, std::fabs(f.F[c][i]));
      d = std::fmax(d, std::fabs(f.F[c][i] - p.F[c][i]));
      sum_d += std::fabs(f.F[c][i] - p.F[c][i]);
      sum_a += std::fabs(f.F[c][i]);
    }
    per_field[c] = d / fmax;
    l1_field[c] = sum_d / sum_a;
    diff = std::fmax(diff, per_field[c]);
  }
  std::printf("[WP3] T3.3 largest octant asymmetry (max |a - mirror(b)| / max|field|, all 7 reflections, fields R Mx My Mz E): fixed %.3e  permuted %.3e\n",
              asym_fixed, asym_perm);
  std::printf("[WP3] T3.3 largest difference fixed vs permuted (relative to max|field|): %.3e  [", diff);
  for (int c = 0; c < 5; ++c) std::printf("%s %.3e%s", names[c], per_field[c], c < 4 ? ", " : "]\n");
  std::printf("[WP3] T3.3 L1 difference fixed vs permuted (sum |a-b| / sum |a|, fixed order as reference):  [");
  for (int c = 0; c < 5; ++c) std::printf("%s %.3e%s", names[c], l1_field[c], c < 4 ? ", " : "]\n");
  CHECK(asym_fixed < 1e-10);
}

// ------------------------------------------------------------------ T3.4
namespace {
constexpr Real kRhoS = 0.153, kMN = 0.940;

struct ColdProbe { Real eps_c, vs2, p0; };

ColdProbe probe_cold(const EosColdThermal& eos, Real rho, Real eps_over_cold_shift) {
  Kokkos::View<Real*> out("probe", 3);
  Kokkos::parallel_for("probe_cold", Kokkos::RangePolicy<ExecSpace>(0, 1), KOKKOS_LAMBDA(const int) {
    const Real ec = eos.eps_cold(rho);
    const Real e = ec + eps_over_cold_shift;
    out(0) = ec; out(1) = eos.vs2(e, rho); out(2) = eos.p0(e, rho);
  });
  auto h = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(h, out);
  return {h(0), h(1), h(2)};
}
}  // namespace

TEST_CASE("T3.4 sound wave in the cold-thermal two-phase EoS at 2 rho_s: phase velocity") {
  const ColdCurveTable table = ColdCurveTable::load(std::string(SPF_SOURCE_DIR) + "/data/eos/cold_curve.dat");
  const EosColdThermal eos(table, ColdVariant::two_phase, 4.0 / 3.0);
  const Real rho0 = 2.0 * kRhoS, eps_s = kMN * kRhoS;
  const ColdProbe pr = probe_cold(eos, rho0, 0.5 * eps_s);
  const Real eps0 = pr.eps_c + 0.5 * eps_s, p0 = pr.p0, w0 = eps0 + p0, cs = std::sqrt(pr.vs2);
  std::printf("[WP3] T3.4 background: rho = %.6f fm^-3, eps_c = %.9f, eps = %.9f GeV/fm^3, p = %.9f, vs2 = %.9f, vs = %.9f\n", rho0, pr.eps_c,
              eps0, p0, pr.vs2, cs);
  REQUIRE(pr.vs2 > 0.0);

  const int nx = 64;
  const Real dx = 0.2, L = nx * dx;
  const Real amp = 1e-4;
  bool all_ok = true;
  for (int m = 1; m <= 4; ++m) {
    const Real k = 2.0 * M_PI * m / L;
    Hydro3DParams hp = params_for(nx, 4, 4, dx, "permuted");
    Hydro3D<EosColdThermal> h(eos, hp);
    h.set_state_fn([&](int ix, int, int, Real& e, Real& r, Real& ux, Real& uy, Real& uz) {
      const Real x = (ix + 0.5) * dx;
      const Real de = amp * eps0 * std::cos(k * x);
      e = eps0 + de;
      r = rho0 * (1.0 + de / w0);
      ux = cs * de / w0;
      uy = uz = 0.0;
    });
    const Real T = 2.0 * M_PI / (cs * k);
    const long nsteps = std::lround(T / (hp.shasta.lambda * dx));
    std::vector<Real> t, ph;
    Real amp0 = 0.0, amp1 = 0.0;
    for (long s = 0; s <= nsteps; ++s) {
      if (s > 0) h.step();
      h.update_primitives();
      const std::vector<Real> vx = h.interior_copy(h.vx);
      std::complex<Real> a = 0.0;
      for (int ix = 0; ix < nx; ++ix) {
        // average over the transverse cells (uniform)
        const Real x = (ix + 0.5) * dx;
        a += vx[static_cast<size_t>(ix) * 16] * std::exp(std::complex<Real>(0.0, -k * x));
      }
      Real phase = std::arg(a);
      if (!ph.empty()) {
        while (phase - ph.back() > M_PI) phase -= 2.0 * M_PI;
        while (phase - ph.back() < -M_PI) phase += 2.0 * M_PI;
      }
      ph.push_back(phase);
      t.push_back(h.time);
      if (s == 0) amp0 = std::abs(a);
      amp1 = std::abs(a);
    }
    // least-squares slope of the unwrapped phase: phase = phase0 - omega t
    Real st = 0, sp = 0, stt = 0, stp = 0;
    const size_t N = t.size();
    for (size_t i = 0; i < N; ++i) { st += t[i]; sp += ph[i]; stt += t[i] * t[i]; stp += t[i] * ph[i]; }
    const Real slope = (N * stp - st * sp) / (N * stt - st * st);
    const Real omega = -slope;
    const Real ratio = omega / (cs * k);
    std::printf("[WP3] T3.4 m = %d  k = %.6f fm^-1  k dx = %.4f  steps %ld  phase velocity measured/analytic = %.6f  amplitude after one period / initial = %.4f  recovery failures %ld\n",
                m, k, k * dx, nsteps, ratio, amp1 / amp0, h.n_recover_fail_total);
    const bool ok = std::fabs(ratio - 1.0) < 0.02;
    all_ok = all_ok && ok;
    CHECK(ok);
    CHECK(h.n_recover_fail_total == 0);
  }
  (void)all_ok;
}

// ------------------------------------------------------------------ T3.5
namespace {
Real sound_wave_error(int N, Real* lam_used) {
  const Real L = 16.0, dx = L / N, cs = std::sqrt(1.0 / 3.0), amp = 1e-6, eps0 = 1.5, rho0 = 0.3;
  const Real w0 = eps0 * (4.0 / 3.0), k = 2.0 * M_PI / L;
  Hydro3DParams hp = params_for(N, 4, 4, dx, "permuted");
  Hydro3D<EosIdeal> h(EosIdeal{}, hp);
  h.set_state_fn([&](int ix, int, int, Real& e, Real& r, Real& ux, Real& uy, Real& uz) {
    const Real x = (ix + 0.5) * dx, de = amp * eps0 * std::sin(k * x);
    e = eps0 + de; r = rho0 * (1.0 + de / w0); ux = cs * de / w0; uy = uz = 0.0;
  });
  const Real T = L / cs;
  const long nsteps = static_cast<long>(std::ceil(T / (hp.shasta.lambda * dx)));
  const Real lam = T / (nsteps * dx);
  *lam_used = lam;
  for (long s = 0; s < nsteps; ++s) h.step(lam);
  h.update_primitives();
  CHECK(h.n_recover_fail_total == 0);
  const std::vector<Real> e = h.interior_copy(h.eps);
  Real l1 = 0.0, l1ref = 0.0;
  for (int ix = 0; ix < N; ++ix) {
    const Real x = (ix + 0.5) * dx;
    const Real ex = amp * eps0 * std::sin(k * (x - cs * h.time));
    l1 += std::fabs((e[static_cast<size_t>(ix) * 16] - eps0) - ex) * dx;
    l1ref += std::fabs(ex) * dx;
  }
  return l1 / l1ref;
}
}  // namespace

TEST_CASE("T3.5 smooth-flow convergence: travelling sound wave, p = eps/3, one period") {
  const int Ns[3] = {32, 64, 128};
  Real err[3], lam[3];
  for (int i = 0; i < 3; ++i) {
    err[i] = sound_wave_error(Ns[i], &lam[i]);
    std::printf("[WP3] T3.5 N = %d  lambda used = %.5f  relative L1 error of eps fluctuation = %.6e\n", Ns[i], lam[i], err[i]);
  }
  const Real o1 = std::log2(err[0] / err[1]), o2 = std::log2(err[1] / err[2]);
  std::printf("[WP3] T3.5 measured order: 32->64 %.3f  64->128 %.3f\n", o1, o2);
  CHECK(err[1] < err[0]);
  CHECK(err[2] < err[1]);
}

// ------------------------------------------------------------------ T3.6
namespace {
bool same_bits(const std::vector<Real>& a, const std::vector<Real>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(Real)) == 0;
}
}  // namespace

TEST_CASE("T3.6 zero p_extra field changes nothing; a nonzero one does") {
  const int n = 16;
  auto run = [&](bool use, bool nonzero) {
    Hydro3DParams hp = params_for(n, n, n, 0.2, "permuted");
    Hydro3D<EosIdeal> h(EosIdeal{}, hp);
    h.set_state(smooth_random_state(n, n, n));
    h.use_p_extra = use;
    if (nonzero) {
      auto hp_ = Kokkos::create_mirror_view(h.p_extra);
      const int g = h.grid.g[0];
      for (int i = 0; i < h.grid.size(0); ++i)
        for (int j = 0; j < h.grid.size(1); ++j)
          for (int k = 0; k < h.grid.size(2); ++k) {
            const int a = ((i - g) % n + n) % n, b = ((j - g) % n + n) % n, c = ((k - g) % n + n) % n;
            hp_(i, j, k) = 0.05 * std::sin(2.0 * M_PI * a / n) * std::cos(2.0 * M_PI * (b + c) / n);
          }
      Kokkos::deep_copy(h.p_extra, hp_);
    }
    h.run(20);
    std::vector<std::vector<Real>> out;
    for (int c = 0; c < 5; ++c) out.push_back(h.interior_copy(h.U[c]));
    return out;
  };
  const auto off = run(false, false), zero = run(true, false), nz = run(true, true);
  bool identical = true, differs = false;
  Real dmax = 0.0;
  for (int c = 0; c < 5; ++c) {
    identical = identical && same_bits(off[c], zero[c]);
    if (!same_bits(off[c], nz[c])) differs = true;
    Real fmax = 0.0, d = 0.0;
    for (size_t i = 0; i < off[c].size(); ++i) { fmax = std::fmax(fmax, std::fabs(off[c][i])); d = std::fmax(d, std::fabs(off[c][i] - nz[c][i])); }
    dmax = std::fmax(dmax, d / fmax);
  }
  std::printf("[WP3] T3.6 zero p_extra bitwise identical to hook bypassed: %s;  nonzero p_extra (amplitude 0.05 GeV/fm^3) changes the result by up to %.3e of max|field|\n",
              identical ? "yes" : "NO", dmax);
  CHECK(identical);
  CHECK(differs);
}

// ------------------------------------------------------------------ set_state with table-backed equations of state
// Regression for the host-side eos.p0 call in set_state / set_state_fn: EosColdThermal and EosTable reach their data through device
// pointers, so a host call dereferences device memory on the CUDA backend. The expected conserved fields are computed in a kernel.
namespace {
template <class EoS>
struct ExpectedConsK {
  EoS eos;
  Kokkos::View<Real*> prim, out;  // prim: eps rho vx vy vz; out: R Mx My Mz E (per cell)
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    const Real e = prim(5 * i), r = prim(5 * i + 1), ux = prim(5 * i + 2), uy = prim(5 * i + 3), uz = prim(5 * i + 4);
    for (int c = 0; c < 5; ++c) out(5 * i + c) = 0.0;
    if (e <= 0.0) return;
    const Real p = eos.p0(e, r);
    const Real g = 1.0 / Kokkos::sqrt(1.0 - (ux * ux + uy * uy + uz * uz));
    const Real w = (e + p) * g * g;
    out(5 * i) = g * r; out(5 * i + 1) = w * ux; out(5 * i + 2) = w * uy; out(5 * i + 3) = w * uz; out(5 * i + 4) = w - p;
  }
};

template <class EoS>
std::vector<Real> expected_conserved(const EoS& eos, const std::vector<Real>& prim) {
  const int n = static_cast<int>(prim.size() / 5);
  Kokkos::View<Real*> d("exp_prim", prim.size()), o("exp_out", prim.size());
  auto h = Kokkos::create_mirror_view(d);
  for (size_t i = 0; i < prim.size(); ++i) h(i) = prim[i];
  Kokkos::deep_copy(d, h);
  Kokkos::parallel_for("expected_cons", Kokkos::RangePolicy<ExecSpace>(0, n), ExpectedConsK<EoS>{eos, d, o});
  auto ho = Kokkos::create_mirror_view(o);
  Kokkos::deep_copy(ho, o);
  std::vector<Real> r(prim.size());
  for (size_t i = 0; i < r.size(); ++i) r[i] = ho(i);
  return r;
}

// primitive state of cell number q (some cells vacuum)
void prim_of(int q, Real* P) {
  const Real a = std::sin(0.7 * q + 0.3), b = std::cos(1.3 * q), c = std::sin(2.1 * q + 1.0);
  P[0] = 1.4 + 0.6 * a;  P[1] = 0.28 + 0.1 * b;
  P[2] = 0.17 * a; P[3] = 0.17 * b; P[4] = 0.17 * c;
  if (q % 11 == 5) { P[0] = 0.0; P[1] = 0.0; P[2] = P[3] = P[4] = 0.0; }  // vacuum cell
}

template <class EoS>
void check_set_state(const EoS& eos, const char* label) {
  // ---- 3D
  const int nx = 5, ny = 4, nz = 3;
  Hydro3DParams hp = params_for(nx, ny, nz, 0.2, "fixed");
  Hydro3D<EoS> h(eos, hp);
  std::vector<Real> prim(5 * nx * ny * nz);
  for (int ix = 0; ix < nx; ++ix)
    for (int iy = 0; iy < ny; ++iy)
      for (int iz = 0; iz < nz; ++iz) prim_of((ix * ny + iy) * nz + iz, &prim[5 * ((ix * ny + iy) * nz + iz)]);
  h.set_state_fn([&](int ix, int iy, int iz, Real& e, Real& r, Real& ux, Real& uy, Real& uz) {
    const Real* P = &prim[5 * ((ix * ny + iy) * nz + iz)];
    e = P[0]; r = P[1]; ux = P[2]; uy = P[3]; uz = P[4];
  });
  const std::vector<Real> ex = expected_conserved(eos, prim);
  Real worst3 = 0.0, scale = 0.0;
  for (int c = 0; c < 5; ++c) {
    const std::vector<Real> got = h.interior_copy(h.U[c]);
    Real fmax = 0.0, d = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
      fmax = std::fmax(fmax, std::fabs(ex[5 * i + c]));
      d = std::fmax(d, std::fabs(got[i] - ex[5 * i + c]));
    }
    worst3 = std::fmax(worst3, d / fmax);
    scale = std::fmax(scale, fmax);
  }
  // ---- 1D (same primitives along x; velocity along x only)
  const int n1 = 24;
  std::vector<Real> e1(n1), r1(n1), v1(n1), prim1(5 * n1);
  for (int i = 0; i < n1; ++i) {
    Real P[5];
    prim_of(i, P);
    P[3] = P[4] = 0.0;
    for (int c = 0; c < 5; ++c) prim1[5 * i + c] = P[c];
    e1[i] = P[0]; r1[i] = P[1]; v1[i] = P[2];
  }
  Hydro1D<EoS> h1(eos, n1, 0.2, ShastaParams::from_config(default_config()));
  h1.set_state(e1, r1, v1);
  const std::vector<Real> ex1 = expected_conserved(eos, prim1);
  Real worst1 = 0.0;
  const int map1[3] = {0, 1, 4};  // R, M, E in the 5-component layout
  for (int c = 0; c < 3; ++c) {
    const std::vector<Real> got = h1.line(h1.U[c]);
    Real fmax = 0.0, d = 0.0;
    for (int i = 0; i < n1; ++i) {
      fmax = std::fmax(fmax, std::fabs(ex1[5 * i + map1[c]]));
      d = std::fmax(d, std::fabs(got[i] - ex1[5 * i + map1[c]]));
    }
    worst1 = std::fmax(worst1, d / fmax);
  }
  // ---- a short run from the state must not fail in the recovery
  Hydro3DParams hp2 = params_for(8, 8, 8, 0.2, "permuted");
  Hydro3D<EoS> h2(eos, hp2);
  h2.set_state_fn([&](int ix, int iy, int iz, Real& e, Real& r, Real& ux, Real& uy, Real& uz) {
    Real P[5];
    prim_of((ix * 8 + iy) * 8 + iz, P);
    if (P[0] <= 0.0) { P[0] = 1.4; P[1] = 0.28; }  // keep the smoke run free of vacuum
    e = P[0]; r = P[1]; ux = P[2]; uy = P[3]; uz = P[4];
  });
  h2.run(3);
  std::printf("[WP3] set_state with %s: 3D max |U - expected| / max|field| = %.3e, 1D %.3e (expected values computed in a kernel); 3 steps: recovery failures %ld\n",
              label, worst3, worst1, h2.n_recover_fail_total);
  CHECK(scale > 0.0);
  CHECK(worst3 < 1e-13);
  CHECK(worst1 < 1e-13);
  CHECK(h2.n_recover_fail_total == 0);
}
}  // namespace

TEST_CASE("set_state / set_state_fn with EosColdThermal and EosTable: conserved fields computed on the device") {
  const ColdCurveTable table = ColdCurveTable::load(std::string(SPF_SOURCE_DIR) + "/data/eos/cold_curve.dat");
  const EosColdThermal cold(table, ColdVariant::two_phase, 4.0 / 3.0);
  check_set_state(cold, "EosColdThermal (two_phase)");
  const EosColdThermal maxw(table, ColdVariant::maxwell, 4.0 / 3.0);
  check_set_state(maxw, "EosColdThermal (maxwell)");
  const EosTableData tab = EosTableData::load(std::string(SPF_TEST_DATA_DIR) + "/synthetic_eos");
  check_set_state(tab.eos(), "EosTable (synthetic, bilinear)");
  check_set_state(tab.eos(TableInterp::bicubic, false), "EosTable (synthetic, bicubic)");
  check_set_state(EosIdeal{}, "EosIdeal");
}

// ------------------------------------------------------------------ snapshot writer, config
TEST_CASE("snapshot: raw little-endian doubles and JSON sidecar round trip") {
  const int nx = 6, ny = 5, nz = 4;
  Hydro3DParams hp = params_for(nx, ny, nz, 0.3, "fixed");
  Hydro3D<EosIdeal> h(EosIdeal{}, hp);
  h.set_state(smooth_random_state(nx, ny, nz));
  h.run(3);
  Config cfg = default_config();
  cfg.set_from_string("shasta.sweep_order=fixed");
  h.write_snapshot("wp3_snapshot_test", {"E", "eps", "vx"}, cfg.json());
  std::ifstream js("wp3_snapshot_test.json");
  REQUIRE(js.good());
  const nlohmann::json j = nlohmann::json::parse(js);
  CHECK(j.at("shape")[0] == nx);
  CHECK(j.at("shape")[1] == ny);
  CHECK(j.at("shape")[2] == nz);
  CHECK(j.at("step") == 3);
  CHECK(j.at("dx_fm").get<double>() == doctest::Approx(0.3));
  CHECK(j.at("time_fm").get<double>() == doctest::Approx(3 * 0.4 * 0.3));
  CHECK(j.at("config").at("shasta").at("sweep_order") == "fixed");
  std::ifstream bin("wp3_snapshot_test.bin", std::ios::binary);
  const size_t n = static_cast<size_t>(nx) * ny * nz;
  std::vector<Real> buf(3 * n);
  bin.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(3 * n * sizeof(Real)));
  REQUIRE(bin.gcount() == static_cast<std::streamsize>(3 * n * sizeof(Real)));
  const std::vector<Real> E = h.interior_copy(h.U[4]), eps = h.interior_copy(h.eps), vx = h.interior_copy(h.vx);
  CHECK(std::equal(E.begin(), E.end(), buf.begin()));
  CHECK(std::equal(eps.begin(), eps.end(), buf.begin() + n));
  CHECK(std::equal(vx.begin(), vx.end(), buf.begin() + 2 * n));
  CHECK_THROWS_AS(h.write_snapshot("wp3_snapshot_bad", {"nope"}, cfg.json()), std::invalid_argument);
}

TEST_CASE("Hydro3DParams from configuration") {
  Config cfg = default_config();
  Hydro3DParams hp = Hydro3DParams::from_config(cfg);
  CHECK(hp.nx == 64);
  CHECK(hp.ny == 64);
  CHECK(hp.nz == 64);
  CHECK(hp.snapshot_every == 0);
  CHECK(hp.order == SweepOrder::Permuted);
  cfg.set_from_string("shasta.sweep_order=fixed");
  cfg.set_from_string("grid.nx=32");
  hp = Hydro3DParams::from_config(cfg);
  CHECK(hp.order == SweepOrder::Fixed);
  CHECK(hp.nx == 32);
  cfg.set_from_string("shasta.sweep_order=bogus");
  CHECK_THROWS_AS(Hydro3DParams::from_config(cfg), ConfigError);
}
