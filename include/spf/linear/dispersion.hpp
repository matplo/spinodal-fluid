#pragma once
// Dispersion relation of finite-range dissipative fluid dynamics, arXiv:1007.1448 Eq. (33):
//   w^2 = vT2 k^2 + B k^4 - i nu w + (vs2 - vT2) k^2 / (1 + i q/w),   B = C rho0^2/h0,  nu = lam_visc k^2,  q = lam_heat k^2,
// with exp(-i w t): Im w > 0 is growth. lam_visc = xi/h0 = (4/3) eta0 c0 d, lam_heat = kappa/c_v = kappa0 c0 d (Eqs. 36, 38).
// Clearing the denominator (docs/equations.md, Sect. 12):  w^3 + i(nu+q) w^2 - (S + q nu) w - i q A = 0,
//   A = vT2 k^2 + B k^4,  S = vs2 k^2 + B k^4.  For q = 0 the factor w is spurious and the quadratic w^2 + i nu w - S = 0 is solved.
#include "spf/linear/thermo.hpp"

#include <complex>
#include <vector>

namespace spf::linear {

constexpr double kPi = 3.14159265358979323846;

// Eq. (34): c0 = (1/4pi) [ (g_g + 3/2 g_q) zeta(3)/pi^2 ]^(1/3).  Default g_g = 16, g_q = 12 gives 0.12779.
double c0_constant(double g_g = 16.0, double g_q = 12.0);

struct Transport {
  double eta0 = 1.0;    // shear-viscosity strength, Eq. (35)
  double kappa0 = 1.0;  // heat-conduction strength, Eq. (37)
  double c0 = c0_constant();
};

// Interparticle spacing d = n^(-1/3) (fm).
double spacing(double n);
// Eq. (35): eta = eta0 (c0/c) d h  (GeV/fm^2);  Eq. (37): kappa = kappa0 c0 c d c_v (fm^-2 per GeV... units of c_v * fm).
double eta_of(const Transport& tr, const ThermoPoint& p);
double kappa_of(const Transport& tr, const ThermoPoint& p);
// Eq. (36): lam_visc = (4/3) eta0 c0 d;  Eq. (38): lam_heat = kappa0 c0 d.   (fm; zeta neglected: xi = 4/3 eta)
double lambda_visc(const Transport& tr, const ThermoPoint& p);
double lambda_heat(const Transport& tr, const ThermoPoint& p);

// Finite-range strength C = a^2 eps_s / rho_s^2 = a^2 m_N / rho_s (same as GradientParams::coef()); GeV fm^5.
double gradient_C(double a_fm, double m_N_GeV = 0.940, double rho_s_fm3 = 0.153);

// Everything the dispersion relation needs at one phase point.
struct DispersionParams {
  double vT2 = 0, vs2 = 0;
  double B = 0;         // C rho0^2 / h0, fm^2
  double lam_visc = 0;  // xi/h0, fm
  double lam_heat = 0;  // kappa/c_v, fm
};
DispersionParams make_dispersion(const ThermoPoint& p, double C, const Transport& tr);

// Coefficients (a2, a1, a0) of w^3 + a2 w^2 + a1 w + a0 for q > 0.
void cubic_coefficients(const DispersionParams& d, double k, std::complex<double>& a2, std::complex<double>& a1, std::complex<double>& a0);
// All eigenfrequencies at wave number k: 2 roots if lam_heat == 0, else 3 (Cardano + Newton polish).
std::vector<std::complex<double>> omega_roots(const DispersionParams& d, double k);
// gamma_k = max Im w over the roots (stable evaluation of the quadratic when lam_heat == 0).
double growth_rate(const DispersionParams& d, double k);
// Largest Im w over the roots, together with the root.
std::complex<double> dominant_root(const DispersionParams& d, double k);

// Maximum of gamma_k over a k grid.
struct GammaMax { double k = 0, gamma = 0; };
GammaMax max_growth(const DispersionParams& d, double k_min, double k_max, int nk);

}  // namespace spf::linear
