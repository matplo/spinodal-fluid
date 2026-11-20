#pragma once
// Exact solutions for p = eps/3 used by the WP1 tests (host code only). Derivations: docs/equations.md.
#include <cmath>

namespace exact {

constexpr double cs2 = 1.0 / 3.0;
inline double cs() { return std::sqrt(cs2); }

// Rarefaction wave of a semi-infinite fluid (eps0 at zeta < 0, vacuum or lower pressure beyond), simple wave R+ = 0:
// v(eps) = tanh( -cs/(1+cs^2) ln(eps/eps0) ),   zeta(eps) = (v - cs)/(1 - v cs).
inline double rarefaction_v(double eps, double eps0) {
  return std::tanh(-cs() / (1.0 + cs2) * std::log(eps / eps0));
}
// eps(zeta) inside the fan, -cs <= zeta <= 1 (paper Eq. 53), in units of eps0.
inline double fan_eps(double zeta, double eps0) {
  const double c = cs();
  return eps0 * std::pow((1.0 - c) / (1.0 + c) * (1.0 - zeta) / (1.0 + zeta), (1.0 + cs2) / (2.0 * c));
}
inline double T00(double eps, double v) {
  const double g2 = 1.0 / (1.0 - v * v);
  return (eps + eps * cs2) * g2 - eps * cs2;
}

// Expansion of semi-infinite matter into vacuum (paper Sect. 4.2): eps and T^00 as functions of zeta = x/t.
inline double vacuum_eps(double zeta, double eps0) {
  if (zeta <= -cs()) return eps0;
  if (zeta >= 1.0) return 0.0;
  return fan_eps(zeta, eps0);
}
inline double vacuum_T00(double zeta, double eps0) {
  if (zeta <= -cs()) return eps0;
  if (zeta >= 1.0) return 0.0;
  const double e = fan_eps(zeta, eps0);
  const double v = (zeta + cs()) / (1.0 + zeta * cs());
  return T00(e, v);
}

// Riemann problem, both states at rest, high pressure on the left: left rarefaction, contact, right shock.
struct ShockTube {
  double eps_L, eps_R;
  double eps_star, v_star, V_shock, zeta_tail;

  // fluid velocity behind a shock moving into matter at rest with eps_R (Taub adiabat), p = eps/3
  static double shock_v(double e, double eR) {
    const double p = e * cs2, pR = eR * cs2;
    return std::sqrt((p - pR) * (e - eR) / ((eR + p) * (e + pR)));
  }
  static double shock_speed(double e, double eR) {
    const double p = e * cs2, pR = eR * cs2;
    return std::sqrt((p - pR) * (e + pR) / ((e - eR) * (eR + p)));
  }

  ShockTube(double eL, double eR) : eps_L(eL), eps_R(eR) {
    double lo = eps_R, hi = eps_L;  // f > 0 at lo, f < 0 at hi
    for (int it = 0; it < 200; ++it) {
      const double mid = 0.5 * (lo + hi);
      const double f = rarefaction_v(mid, eps_L) - shock_v(mid, eps_R);
      if (f > 0.0) lo = mid; else hi = mid;
    }
    eps_star = 0.5 * (lo + hi);
    v_star = rarefaction_v(eps_star, eps_L);
    V_shock = shock_speed(eps_star, eps_R);
    zeta_tail = (v_star - cs()) / (1.0 - v_star * cs());
  }

  double eps(double zeta) const {
    if (zeta < -cs()) return eps_L;
    if (zeta < zeta_tail) return fan_eps(zeta, eps_L);
    if (zeta < V_shock) return eps_star;
    return eps_R;
  }
};

}  // namespace exact
