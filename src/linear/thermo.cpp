#include "spf/linear/thermo.hpp"

#include <algorithm>
#include <stdexcept>

namespace spf::linear {

double particle_density_spline(const std::string& mode, double rho, double rho_lo, double rho_hi, double n_hadron, double n_plasma) {
  if (mode == "hadron_side") return n_hadron;
  if (mode == "plasma_side") return n_plasma;
  if (mode == "linear_in_rho") {
    const double w = rho_hi > rho_lo ? std::clamp((rho - rho_lo) / (rho_hi - rho_lo), 0.0, 1.0) : 0.0;
    return n_hadron + (n_plasma - n_hadron) * w;
  }
  throw std::invalid_argument("linear.particle_density_in_spline_region: unknown mode '" + mode + "'");
}

ToyVdw::ToyVdw(double rho_c, double T_c, double m_GeV, double cv0_) : m(m_GeV), cv0(cv0_) {
  b = 1.0 / (3.0 * rho_c);
  a = 27.0 * b * T_c / 8.0;
}

ThermoPoint ToyVdw::at(double rho, double T) const {
  ThermoPoint p;
  const double om = 1.0 - b * rho;
  const double p0 = pressure(rho, T);
  const double eps = m * rho + cv0 * rho * T - a * rho * rho;
  p.rho = rho; p.T = T; p.h = eps + p0; p.cv = cv0 * rho; p.n = rho;
  const double dpdrho = T / (om * om) - 2.0 * a * rho;
  p.vT2 = rho / p.h * dpdrho;
  p.vs2 = p.vT2 + T * rho / (p.h * cv0 * om * om);
  return p;
}

}  // namespace spf::linear
