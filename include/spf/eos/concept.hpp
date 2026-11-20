#pragma once
// Equation-of-state interface (docs/device_rules.md, rule 4): a C++20 concept, no base class, no virtual function.
//
// Units: GeV and fm with c = 1.  Energy density and pressure in GeV/fm^3, net baryon density in fm^-3.
//
//   p0(eps, rho)       pressure of uniform matter [GeV/fm^3]; eps = energy density in the local rest frame,
//                      rho = net baryon density in the local rest frame
//   dp_deps(eps, rho)  (dp0/deps) at fixed rho [dimensionless]
//   dp_drho(eps, rho)  (dp0/drho) at fixed eps [GeV]
//   vs2(eps, rho)      isentropic sound speed squared = dp_deps + rho / (eps + p0) * dp_drho [dimensionless];
//                      negative inside the spinodal (isentropic) region
//
// All four members are const and KOKKOS_INLINE_FUNCTION, so an EoS object is passed to a kernel by value.
// Table data are reached through raw pointers into memory owned by a host-side object that must outlive the
// kernels (see cold_thermal.hpp); this keeps the struct trivially copyable.
#include "spf/types.hpp"

#include <concepts>
#include <type_traits>

namespace spf {

template <class E>
concept EquationOfState = std::is_trivially_copyable_v<E> && requires(const E& e, Real eps, Real rho) {
  { e.p0(eps, rho) } -> std::convertible_to<Real>;
  { e.dp_deps(eps, rho) } -> std::convertible_to<Real>;
  { e.dp_drho(eps, rho) } -> std::convertible_to<Real>;
  { e.vs2(eps, rho) } -> std::convertible_to<Real>;
};

}  // namespace spf
