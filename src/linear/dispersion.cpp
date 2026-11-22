#include "spf/linear/dispersion.hpp"

#include <algorithm>
#include <cmath>

namespace spf::linear {

using cd = std::complex<double>;

double c0_constant(double g_g, double g_q) {
  const double zeta3 = 1.2020569031595942854;
  return std::cbrt((g_g + 1.5 * g_q) * zeta3 / (kPi * kPi)) / (4.0 * kPi);
}
double spacing(double n) { return std::cbrt(1.0 / n); }
double eta_of(const Transport& tr, const ThermoPoint& p) { return tr.eta0 * tr.c0 * spacing(p.n) * p.h; }
double kappa_of(const Transport& tr, const ThermoPoint& p) { return tr.kappa0 * tr.c0 * spacing(p.n) * p.cv; }
double lambda_visc(const Transport& tr, const ThermoPoint& p) { return 4.0 / 3.0 * tr.eta0 * tr.c0 * spacing(p.n); }
double lambda_heat(const Transport& tr, const ThermoPoint& p) { return tr.kappa0 * tr.c0 * spacing(p.n); }
double gradient_C(double a_fm, double m_N, double rho_s) { return a_fm * a_fm * m_N / rho_s; }

DispersionParams make_dispersion(const ThermoPoint& p, double C, const Transport& tr) {
  DispersionParams d;
  d.vT2 = p.vT2; d.vs2 = p.vs2;
  d.B = C * p.rho * p.rho / p.h;
  d.lam_visc = lambda_visc(tr, p);
  d.lam_heat = lambda_heat(tr, p);
  return d;
}

void cubic_coefficients(const DispersionParams& d, double k, cd& a2, cd& a1, cd& a0) {
  const double k2 = k * k, k4 = k2 * k2;
  const double nu = d.lam_visc * k2, q = d.lam_heat * k2;
  const double A = d.vT2 * k2 + d.B * k4, S = d.vs2 * k2 + d.B * k4;
  a2 = cd(0.0, nu + q);
  a1 = cd(-(S + q * nu), 0.0);
  a0 = cd(0.0, -q * A);
}

namespace {
cd newton_polish(cd w, cd a2, cd a1, cd a0) {
  for (int it = 0; it < 6; ++it) {
    const cd f = ((w + a2) * w + a1) * w + a0;
    const cd fp = (3.0 * w + 2.0 * a2) * w + a1;
    if (std::abs(fp) < 1e-300) break;
    const cd dw = f / fp;
    w -= dw;
    if (std::abs(dw) <= 1e-16 * std::max(1.0, std::abs(w))) break;
  }
  return w;
}
}  // namespace

std::vector<cd> omega_roots(const DispersionParams& d, double k) {
  const double k2 = k * k, k4 = k2 * k2;
  if (d.lam_heat <= 0.0) {
    const double nu = d.lam_visc * k2, S = d.vs2 * k2 + d.B * k4;
    // w^2 + i nu w - S = 0
    const double disc = 4.0 * S - nu * nu;
    if (disc >= 0.0) {
      const double r = 0.5 * std::sqrt(disc);
      return {cd(-r, -0.5 * nu), cd(r, -0.5 * nu)};
    }
    const double s = std::sqrt(-disc);               // roots are purely imaginary
    const double plus = -2.0 * S / (nu + s);          // (-nu + s)/2 written without cancellation
    const double minus = -0.5 * (nu + s);
    return {cd(0.0, minus), cd(0.0, plus)};
  }
  cd a2, a1, a0;
  cubic_coefficients(d, k, a2, a1, a0);
  // depressed cubic z^3 + P z + Q = 0, w = z - a2/3
  const cd P = a1 - a2 * a2 / 3.0;
  const cd Q = 2.0 * a2 * a2 * a2 / 27.0 - a2 * a1 / 3.0 + a0;
  const cd sq = std::sqrt(Q * Q / 4.0 + P * P * P / 27.0);
  cd u3 = -Q / 2.0 + sq;
  const cd u3b = -Q / 2.0 - sq;
  if (std::abs(u3b) > std::abs(u3)) u3 = u3b;
  std::vector<cd> roots;
  const cd om = cd(-0.5, 0.5 * std::sqrt(3.0));
  if (std::abs(u3) < 1e-300) {
    // P = Q = 0 : triple root
    for (int j = 0; j < 3; ++j) roots.push_back(newton_polish(-a2 / 3.0, a2, a1, a0));
    return roots;
  }
  const cd u = std::pow(u3, 1.0 / 3.0);
  cd uj = u;
  for (int j = 0; j < 3; ++j) {
    const cd z = uj - P / (3.0 * uj);
    roots.push_back(newton_polish(z - a2 / 3.0, a2, a1, a0));
    uj *= om;
  }
  return roots;
}

std::complex<double> dominant_root(const DispersionParams& d, double k) {
  const auto r = omega_roots(d, k);
  return *std::max_element(r.begin(), r.end(), [](const cd& x, const cd& y) { return x.imag() < y.imag(); });
}
double growth_rate(const DispersionParams& d, double k) { return dominant_root(d, k).imag(); }

GammaMax max_growth(const DispersionParams& d, double k_min, double k_max, int nk) {
  GammaMax best;
  best.gamma = -1e300;
  for (int i = 0; i < nk; ++i) {
    const double k = nk > 1 ? k_min + (k_max - k_min) * i / (nk - 1) : k_min;
    const double g = growth_rate(d, k);
    if (g > best.gamma) { best.gamma = g; best.k = k; }
  }
  return best;
}

}  // namespace spf::linear
