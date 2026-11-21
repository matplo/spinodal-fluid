#pragma once
// Spinodal growth rates: uniform background plus one harmonic perturbation in a periodic box elongated along x
// (ny = nz = 4), time evolution with Hydro3D, Fourier amplitude of the rest-frame density versus time, and the fits
// A0 cosh(gamma t) (unstable) and A0 cos(omega t) (stable). Linear theory: docs/equations.md, Sect. 10.
// Host-side driver code (the only kernels are the named functors of hydro3d.hpp and gradient.hpp), used by apps/spf_growth.cpp
// and tests/test_gradient.cpp.
#include "spf/eos/eos.hpp"
#include "spf/hydro3d.hpp"

#include <algorithm>
#include <functional>
#include <cmath>
#include <limits>
#include <vector>

namespace spf {

namespace detail {
// Pressure, vs2 and (for the cold-thermal EoS) the cold energy density at one point, evaluated on the device.
template <class EoS>
struct ProbeK {
  EoS eos;
  Real eps, rho;
  Kokkos::View<Real*> out;
  KOKKOS_INLINE_FUNCTION void operator()(const int) const {
    out(0) = eos.p0(eps, rho);
    out(1) = eos.vs2(eps, rho);
  }
};
struct ColdProbeK {
  EosColdThermal eos;
  Real rho;
  Kokkos::View<Real*> out;
  KOKKOS_INLINE_FUNCTION void operator()(const int) const { out(0) = eos.eps_cold(rho); }
};
}  // namespace detail

struct EosPoint {
  Real eps = 0, rho = 0, p0 = 0, vs2 = 0, h = 0;
};

// Probe an EoS at (eps, rho) through a one-thread kernel (docs/device_rules.md, rule 9).
template <class EoS>
EosPoint probe_eos(const EoS& eos, Real eps, Real rho) {
  Kokkos::View<Real*> out("probe", 2);
  Kokkos::parallel_for("spf_probe_eos", Kokkos::RangePolicy<ExecSpace>(0, 1), detail::ProbeK<EoS>{eos, eps, rho, out});
  auto h = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(h, out);
  EosPoint pt;
  pt.eps = eps; pt.rho = rho; pt.p0 = h(0); pt.vs2 = h(1); pt.h = eps + h(0);
  return pt;
}

inline Real cold_energy_density(const EosColdThermal& eos, Real rho) {
  Kokkos::View<Real*> out("probe_cold", 1);
  Kokkos::parallel_for("spf_probe_cold", Kokkos::RangePolicy<ExecSpace>(0, 1), detail::ColdProbeK{eos, rho, out});
  auto h = Kokkos::create_mirror_view(out);
  Kokkos::deep_copy(h, out);
  return h(0);
}

// Linear theory: omega^2(k) = vs2 k^2 + a^2 (eps_s / h) (rho / rho_s)^2 k^4  (isentropic perturbation, rest frame).
inline Real omega2_analytic(const GradientParams& g, const EosPoint& pt, Real k) {
  const Real r = pt.rho / g.rho_s;
  return pt.vs2 * k * k + g.a_fm * g.a_fm * (g.eps_s() / pt.h) * r * r * k * k * k * k;
}
// Growth rate of an unstable mode (0 where omega^2 >= 0).
inline Real gamma_analytic(const GradientParams& g, const EosPoint& pt, Real k) {
  const Real w2 = omega2_analytic(g, pt, k);
  return w2 < 0.0 ? std::sqrt(-w2) : 0.0;
}
// Cut-off wave number (gamma = 0, k > 0); 0 if vs2 >= 0 or a = 0.
inline Real kcut_analytic(const GradientParams& g, const EosPoint& pt) {
  if (pt.vs2 >= 0.0 || g.a_fm <= 0.0) return 0.0;
  const Real r = pt.rho / g.rho_s;
  return std::sqrt(-pt.vs2 * pt.h / (g.eps_s() * r * r)) / g.a_fm;
}
// Wave number and rate of the maximum of gamma(k) (a > 0, vs2 < 0): d(gamma^2)/dk = 0 gives k^2 = |vs2| h / (2 a^2 eps_s r^2).
inline void gamma_max_analytic(const GradientParams& g, const EosPoint& pt, Real& kmax, Real& gmax) {
  kmax = gmax = 0.0;
  if (pt.vs2 >= 0.0 || g.a_fm <= 0.0) return;
  const Real r = pt.rho / g.rho_s;
  kmax = std::sqrt(-pt.vs2 * pt.h / (2.0 * g.eps_s() * r * r)) / g.a_fm;
  gmax = gamma_analytic(g, pt, kmax);
}

struct GrowthSetup {
  EosPoint bg;              // background
  Real amp_rel = 1e-4;      // delta rho / rho0 (isentropic: delta eps = (h / rho) delta rho)
  bool paper = false;       // (delta rho, delta eps) = (0.1 rho_s, 0.2 eps_s) instead; bg must be (6 rho_s, 10 eps_s)
  Real rho_s = 0.153, eps_s = 0.94 * 0.153;
  Real t_max = 60.0;        // fm/c
  Real stop_factor = 3.6;   // unstable mode: stop when the amplitude exceeds this multiple of the initial one
  Real stable_periods = 1.5;
};

struct ModeSeries {
  std::vector<Real> t, a_cos, a_abs;  // Fourier amplitude of rho at mode m: signed cosine projection and modulus
  Real k = 0.0, L = 0.0;
  long steps = 0, recover_fail = 0;
  Real min_pgrad_ratio = 0.0;         // most negative p_grad / p0 sampled (every 10 steps)
};

// Evolve one mode m in a box of nx = round(L/dx) cells along x (hp.ny = hp.nz = 4 assumed set by the caller through nx only).
// hp carries dx, the gradient parameters and lambda. The mode is classified by the analytic omega^2 (only to choose the stopping rule).
template <class EoS>
ModeSeries run_mode(const EoS& eos, Hydro3DParams hp, const GrowthSetup& s, Real L, int m) {
  const int nx = static_cast<int>(std::lround(L / hp.dx));
  hp.nx = nx; hp.ny = 4; hp.nz = 4;
  const Real Lx = nx * hp.dx;
  const Real k = 2.0 * M_PI * m / Lx;
  Hydro3D<EoS> h(eos, hp);
  Real d_eps, d_rho;
  if (s.paper) { d_rho = 0.1 * s.rho_s; d_eps = 0.2 * s.eps_s; }
  else { d_rho = s.amp_rel * s.bg.rho; d_eps = s.bg.h / s.bg.rho * d_rho; }
  const Real dx = hp.dx, eps0 = s.bg.eps, rho0 = s.bg.rho;
  h.set_state_fn([&](int ix, int, int, Real& e, Real& r, Real& ux, Real& uy, Real& uz) {
    const Real c = std::cos(k * (ix + 0.5) * dx);
    e = eps0 + d_eps * c;
    r = rho0 + d_rho * c;
    ux = uy = uz = 0.0;
  });
  const GradientParams gp = hp.gradient;
  const Real w2 = omega2_analytic(gp, s.bg, k);
  const Real t_end = w2 > 0.0 ? s.stable_periods * 2.0 * M_PI / std::sqrt(w2) : s.t_max;
  ModeSeries out;
  out.k = k; out.L = Lx;
  auto sample = [&]() {
    h.update_primitives();
    const std::vector<Real> rho = h.interior_copy(h.rho);
    Real sc = 0, ss = 0;
    for (int ix = 0; ix < nx; ++ix) {
      const Real x = (ix + 0.5) * dx, r = rho[static_cast<size_t>(ix) * 16];
      sc += r * std::cos(k * x);
      ss += r * std::sin(k * x);
    }
    out.t.push_back(h.time);
    out.a_cos.push_back(2.0 * sc / nx);
    out.a_abs.push_back(2.0 * std::hypot(sc, ss) / nx);
  };
  sample();
  const Real a0 = out.a_abs[0];
  while (h.time < t_end) {
    h.step();
    sample();
    if (h.gradient_on() && h.steps % 10 == 0) h.sample_pgrad_ratio();
    if (w2 < 0.0 && out.a_abs.back() > s.stop_factor * a0) break;
    if (h.n_recover_fail_total > 1000) break;
  }
  if (h.gradient_on()) h.sample_pgrad_ratio();
  out.steps = h.steps;
  out.recover_fail = h.n_recover_fail_total;
  out.min_pgrad_ratio = h.min_pgrad_ratio_seen;
  return out;
}

// ------------------------------------------------------------------ fits
namespace fit {

inline Real golden_min(Real lo, Real hi, const std::function<Real(Real)>& f, int iters = 80) {
  const Real g = 0.6180339887498949;
  Real a = lo, b = hi, c = b - g * (b - a), d = a + g * (b - a), fc = f(c), fd = f(d);
  for (int i = 0; i < iters; ++i) {
    if (fc < fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = f(c); }
    else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = f(d); }
  }
  return 0.5 * (a + b);
}

struct Result {
  bool ok = false;
  Real rate = std::nan("");  // gamma or omega
  Real A0 = std::nan("");
  Real resid = std::nan(""); // rms of (A_i - model_i) / A_i (cosh) or / max|A| (cos)
  Real damp = 0.0;           // damping constant of the damped cosine fit
  int n = 0;
};

// A(t) = A0 cosh(gamma t) on the samples between lo and hi times the first amplitude (contiguous window from the first
// sample >= lo*A(0) to the last sample before A exceeds hi*A(0)). A0 is fitted too (closed form for a given gamma).
inline Result cosh_fit(const std::vector<Real>& t, const std::vector<Real>& A, Real lo = 1.05, Real hi = 3.0) {
  Result r;
  const Real a0 = A[0];
  size_t i0 = 0, i1 = 0;
  bool started = false;
  for (size_t i = 0; i < A.size(); ++i) {
    if (!started && A[i] >= lo * a0) { i0 = i; started = true; }
    if (started) {
      if (A[i] > hi * a0) break;
      i1 = i;
    }
  }
  if (!started || i1 < i0 + 2) return r;
  auto chi2 = [&](Real gam, Real* A0out) {
    Real sy = 0, syy = 0;
    for (size_t i = i0; i <= i1; ++i) { const Real y = A[i] / std::cosh(gam * t[i]); sy += y; syy += y * y; }
    const Real A0f = syy / sy;  // 1/A0 = sum y / sum y^2
    Real c = 0;
    for (size_t i = i0; i <= i1; ++i) { const Real q = A[i] / (A0f * std::cosh(gam * t[i])) - 1.0; c += q * q; }
    if (A0out) *A0out = A0f;
    return c;
  };
  const Real gam = golden_min(0.0, 5.0, [&](Real g) { return chi2(g, nullptr); }, 100);
  Real A0f = 0;
  chi2(gam, &A0f);
  Real rs = 0;
  for (size_t i = i0; i <= i1; ++i) { const Real q = (A[i] - A0f * std::cosh(gam * t[i])) / A[i]; rs += q * q; }
  r.ok = true; r.rate = gam; r.A0 = A0f; r.n = static_cast<int>(i1 - i0 + 1);
  r.resid = std::sqrt(rs / r.n);
  return r;
}

// A(t) = A0 cos(omega t) over the whole series (plain), and the damped variant A0 exp(-d t) cos(omega t) (omega, d free).
// The starting frequency comes from the first zero crossing (a cosine starts at its maximum); both fits are closed form in A0.
inline Result cos_fit(const std::vector<Real>& t, const std::vector<Real>& A, bool damped = false) {
  Result r;
  const size_t n = A.size();
  size_t iz = 0;
  for (size_t i = 1; i < n; ++i)
    if (A[i] * A[0] <= 0.0) { iz = i; break; }
  if (iz == 0) return r;
  const Real tz = t[iz - 1] + (t[iz] - t[iz - 1]) * A[iz - 1] / (A[iz - 1] - A[iz]);
  const Real w0 = M_PI / (2.0 * tz);
  Real amax = 0;
  for (Real a : A) amax = std::fmax(amax, std::fabs(a));
  auto sse = [&](Real w, Real d, Real* A0out) {
    Real sa = 0, sc = 0;
    std::vector<Real> c(n);
    for (size_t i = 0; i < n; ++i) { c[i] = std::exp(-d * t[i]) * std::cos(w * t[i]); sa += A[i] * c[i]; sc += c[i] * c[i]; }
    const Real A0f = sa / sc;
    Real e = 0;
    for (size_t i = 0; i < n; ++i) { const Real q = A[i] - A0f * c[i]; e += q * q; }
    if (A0out) *A0out = A0f;
    return e;
  };
  Real w, d = 0.0;
  if (!damped) {
    w = golden_min(0.5 * w0, 1.5 * w0, [&](Real x) { return sse(x, 0.0, nullptr); }, 100);
  } else {
    d = golden_min(0.0, 2.0, [&](Real dd) {
      const Real ww = golden_min(0.5 * w0, 1.5 * w0, [&](Real x) { return sse(x, dd, nullptr); }, 60);
      return sse(ww, dd, nullptr);
    }, 60);
    w = golden_min(0.5 * w0, 1.5 * w0, [&](Real x) { return sse(x, d, nullptr); }, 100);
  }
  Real A0f = 0;
  const Real e = sse(w, d, &A0f);
  r.ok = true; r.rate = w; r.A0 = A0f; r.damp = d; r.n = static_cast<int>(n);
  r.resid = std::sqrt(e / n) / amax;
  return r;
}

}  // namespace fit
}  // namespace spf
