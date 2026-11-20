#pragma once
// Analytic equations of state: conformal gas and gamma law.  Units: see concept.hpp.
#include "spf/eos/concept.hpp"

namespace spf {

// p0 = eps / 3, independent of rho.
struct EosIdeal {
  KOKKOS_INLINE_FUNCTION Real p0(Real eps, Real /*rho*/) const { return eps / 3.0; }
  KOKKOS_INLINE_FUNCTION Real dp_deps(Real /*eps*/, Real /*rho*/) const { return 1.0 / 3.0; }
  KOKKOS_INLINE_FUNCTION Real dp_drho(Real /*eps*/, Real /*rho*/) const { return 0.0; }
  KOKKOS_INLINE_FUNCTION Real vs2(Real /*eps*/, Real /*rho*/) const { return 1.0 / 3.0; }
  static const char* name() { return "ideal"; }
};

// p0 = (Gamma - 1) (eps - m rho), clamped to 0 when eps < m rho.  m = mass per baryon [GeV].
struct EosGammaLaw {
  Real gamma = 5.0 / 3.0;
  Real mass = 0.940;

  EosGammaLaw() = default;
  EosGammaLaw(Real gamma_, Real mass_) : gamma(gamma_), mass(mass_) {}

  KOKKOS_INLINE_FUNCTION Real p0(Real eps, Real rho) const {
    const Real th = eps - mass * rho;
    return th > 0.0 ? (gamma - 1.0) * th : 0.0;
  }
  KOKKOS_INLINE_FUNCTION Real dp_deps(Real eps, Real rho) const { return (eps - mass * rho > 0.0) ? gamma - 1.0 : 0.0; }
  KOKKOS_INLINE_FUNCTION Real dp_drho(Real eps, Real rho) const {
    return (eps - mass * rho > 0.0) ? -(gamma - 1.0) * mass : 0.0;
  }
  KOKKOS_INLINE_FUNCTION Real vs2(Real eps, Real rho) const {
    const Real h = eps + p0(eps, rho);
    return h > 0.0 ? dp_deps(eps, rho) + rho / h * dp_drho(eps, rho) : dp_deps(eps, rho);
  }
  static const char* name() { return "gamma_law"; }
};

}  // namespace spf
