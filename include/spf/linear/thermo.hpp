#pragma once
// Linear-response library (WP8), arXiv:1007.1448 Sects. III-V. Host-side code; no Kokkos kernels.
// Units: GeV, fm, c = 1. rho, n in fm^-3; T in GeV; h in GeV/fm^3; c_v = (d eps / d T)_rho in fm^-3 (T in GeV).
// Equation: docs/equations.md, Sect. 12 "Linear response".
#include <cmath>
#include <string>

namespace spf::linear {

// Thermodynamic quantities at one phase point (rho, T); any equation of state can fill this.
struct ThermoPoint {
  double rho = 0;  // net baryon density, fm^-3
  double T = 0;    // temperature, GeV
  double h = 0;    // enthalpy density eps + p, GeV/fm^3
  double vT2 = 0;  // isothermal sound speed squared (rho/h)(dp/drho)_T  (Eq. 5); may be negative
  double vs2 = 0;  // isentropic sound speed squared (Eq. 6); may be negative
  double cv = 0;   // specific heat (d eps/dT)_rho, fm^-3
  double n = 0;    // particle number density entering d = n^(-1/3), fm^-3
};

// Particle density across the spline (two-phase) region, config key linear.particle_density_in_spline_region:
//   "linear_in_rho": n = n_hadron + (n_plasma - n_hadron) (rho - rho_lo)/(rho_hi - rho_lo), clamped to [0, 1] in the weight;
//   "hadron_side":   n = n_hadron;   "plasma_side": n = n_plasma.
// n_hadron, n_plasma: particle densities (pions + nucleons + antinucleons; gluons + quarks + antiquarks) at the same T,
// rho_lo, rho_hi: boundaries of the spline region at that T. Throws std::invalid_argument for an unknown mode.
double particle_density_spline(const std::string& mode, double rho, double rho_lo, double rho_hi, double n_hadron, double n_plasma);

// Analytic toy equation of state with a spinodal region: van der Waals fluid with constant specific heat per particle,
//   p = rho T/(1 - b rho) - a rho^2,   eps = m rho + cv0 rho T - a rho^2,   n = rho.
// Then (dp/dT)_rho = rho/(1 - b rho), c_v = cv0 rho, (dp/drho)_T = T/(1 - b rho)^2 - 2 a rho, and
//   vT2 = (rho/h)(dp/drho)_T,   vs2 = vT2 + (T/(h c_v)) (dp/dT)^2 = vT2 + T rho/(h cv0 (1 - b rho)^2)   (Eq. 6).
// Critical point: rho_c = 1/(3b), T_c = 8 a/(27 b). The constructor takes (rho_c, T_c) and fixes b and a.
struct ToyVdw {
  double m = 0.940;    // GeV (rest energy per particle)
  double cv0 = 1.5;    // specific heat per particle
  double b = 0, a = 0;
  ToyVdw(double rho_c_fm3, double T_c_GeV, double m_GeV = 0.940, double cv0_ = 1.5);
  ThermoPoint at(double rho, double T) const;
  double pressure(double rho, double T) const { return rho * T / (1.0 - b * rho) - a * rho * rho; }
};

}  // namespace spf::linear
