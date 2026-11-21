#pragma once
// Test-only equation of state p = eps/3 with the calling convention of the solver kernels.
#include "spf/types.hpp"

struct TestEosIdeal {
  KOKKOS_INLINE_FUNCTION spf::Real p0(spf::Real eps, spf::Real /*rho*/) const { return eps / 3.0; }
  KOKKOS_INLINE_FUNCTION spf::Real dp_deps(spf::Real, spf::Real) const { return 1.0 / 3.0; }
  KOKKOS_INLINE_FUNCTION spf::Real dp_drho(spf::Real, spf::Real) const { return 0.0; }
  KOKKOS_INLINE_FUNCTION spf::Real vs2(spf::Real, spf::Real) const { return 1.0 / 3.0; }
};
