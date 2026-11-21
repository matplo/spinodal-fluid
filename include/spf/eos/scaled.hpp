#pragma once
// Scaled equation of state of Steinheimer and Randrup, Phys. Rev. C 87, 054903 (2013), arXiv:1302.2956, Sect. V C, Eqs. (17)-(19):
// the entropy density of the scaled family is s_bc(eps, rho) = s_0(b eps, c rho), from which
//
//   T_bc(eps, rho)  = T_0(b eps, c rho) / b,        mu_bc(eps, rho) = (c / b) mu_0(b eps, c rho),
//   p_bc(eps, rho)  = p_0(b eps, c rho) / b,        vs2_bc(eps, rho) = vs2_0(b eps, c rho).
//
// Derivatives follow by the chain rule: dp_deps_bc = dp_deps_0(b eps, c rho), dp_drho_bc = (c / b) dp_drho_0(b eps, c rho).
// The critical point moves to (eps_crit / b, rho_crit / c); b = c = 1/1.3 stretches the unstable region by 30 % in both directions.
// T, mu and s are available when the base provides them (EosTable does; the analytic equations of state do not).
#include "spf/eos/concept.hpp"

namespace spf {

namespace detail {
template <class B>
concept HasTmuS = requires(const B& e, Real eps, Real rho) {
  { e.T(eps, rho) } -> std::convertible_to<Real>;
  { e.mu(eps, rho) } -> std::convertible_to<Real>;
  { e.s(eps, rho) } -> std::convertible_to<Real>;
};
}  // namespace detail

template <EquationOfState Base>
struct EosScaled {
  Base base{};
  Real b = 1.0, c = 1.0;

  EosScaled() = default;
  EosScaled(const Base& base_, Real b_, Real c_) : base(base_), b(b_), c(c_) {}

  KOKKOS_INLINE_FUNCTION Real p0(Real eps, Real rho) const { return base.p0(b * eps, c * rho) / b; }
  KOKKOS_INLINE_FUNCTION Real dp_deps(Real eps, Real rho) const { return base.dp_deps(b * eps, c * rho); }
  KOKKOS_INLINE_FUNCTION Real dp_drho(Real eps, Real rho) const { return (c / b) * base.dp_drho(b * eps, c * rho); }
  KOKKOS_INLINE_FUNCTION Real vs2(Real eps, Real rho) const { return base.vs2(b * eps, c * rho); }

  KOKKOS_INLINE_FUNCTION Real T(Real eps, Real rho) const
    requires detail::HasTmuS<Base>
  {
    return base.T(b * eps, c * rho) / b;
  }
  KOKKOS_INLINE_FUNCTION Real mu(Real eps, Real rho) const
    requires detail::HasTmuS<Base>
  {
    return (c / b) * base.mu(b * eps, c * rho);
  }
  KOKKOS_INLINE_FUNCTION Real s(Real eps, Real rho) const
    requires detail::HasTmuS<Base>
  {
    return base.s(b * eps, c * rho);
  }
};

}  // namespace spf
