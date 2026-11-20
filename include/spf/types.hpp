#pragma once
// Basic types shared by all kernels. See docs/device_rules.md.
#include <Kokkos_Core.hpp>

namespace spf {

using Real = double;

// One View per field, indexed (ix, iy, iz), ghost layers included.
// Lives in the default memory space: host memory for the Serial/OpenMP backends, device memory for CUDA.
using Field3 = Kokkos::View<Real***>;
using Field3Host = Field3::host_mirror_type;
using ExecSpace = Kokkos::DefaultExecutionSpace;

}  // namespace spf
