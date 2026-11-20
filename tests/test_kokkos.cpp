// Backend smoke tests written in the style required of every kernel (docs/device_rules.md):
// Views for data, KOKKOS_LAMBDA kernels, one output View per stage, parallel_reduce for sums.
#include <doctest/doctest.h>

#include "spf/types.hpp"

#include <cmath>

using spf::ExecSpace;
using spf::Field3;
using spf::Real;

TEST_CASE("parallel_reduce on the default execution space") {
  const long n = 1000000;
  long sum = 0;
  Kokkos::parallel_reduce(
      "sum_integers", Kokkos::RangePolicy<ExecSpace>(0, n), KOKKOS_LAMBDA(const long i, long& acc) { acc += i; }, sum);
  CHECK(sum == n * (n - 1) / 2);
}

TEST_CASE("3D stencil: second-order Laplacian of a quadratic field is exact") {
  const int n = 24;
  const Real dx = 0.2;
  Field3 f("f", n, n, n), lap("lap", n, n, n);
  Kokkos::parallel_for(
      "fill", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {n, n, n}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        const Real x = i * dx, y = j * dx, z = k * dx;
        f(i, j, k) = 1.5 * x * x - 0.5 * y * y + 2.0 * z * z + x * y;
      });
  const Real inv_dx2 = 1.0 / (dx * dx);
  Kokkos::parallel_for(
      "laplacian", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({1, 1, 1}, {n - 1, n - 1, n - 1}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) {
        lap(i, j, k) = (f(i + 1, j, k) + f(i - 1, j, k) + f(i, j + 1, k) + f(i, j - 1, k) + f(i, j, k + 1) + f(i, j, k - 1) -
                        6.0 * f(i, j, k)) *
                       inv_dx2;
      });
  Real max_err = 0.0;
  Kokkos::parallel_reduce(
      "max_err", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({1, 1, 1}, {n - 1, n - 1, n - 1}),
      KOKKOS_LAMBDA(const int i, const int j, const int k, Real& m) {
        const Real e = Kokkos::fabs(lap(i, j, k) - 6.0);  // 2*1.5 - 2*0.5 + 2*2.0
        if (e > m) m = e;
      },
      Kokkos::Max<Real>(max_err));
  CHECK(max_err < 1e-9);
}

TEST_CASE("host mirror round trip") {
  const int n = 8;
  Field3 f("f", n, n, n);
  Kokkos::parallel_for(
      "fill", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {n, n, n}),
      KOKKOS_LAMBDA(const int i, const int j, const int k) { f(i, j, k) = 100.0 * i + 10.0 * j + k; });
  auto h = Kokkos::create_mirror_view(f);
  Kokkos::deep_copy(h, f);
  CHECK(h(3, 2, 1) == doctest::Approx(321.0));
  h(3, 2, 1) = -1.0;
  Kokkos::deep_copy(f, h);
  Real total = 0.0;
  Kokkos::parallel_reduce(
      "sum", Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {n, n, n}),
      KOKKOS_LAMBDA(const int i, const int j, const int k, Real& acc) { acc += f(i, j, k); }, total);
  // sum over the grid of 100 i + 10 j + k = 111 * (0+...+7) * 64 = 198912; one entry changed from 321 to -1
  CHECK(total == doctest::Approx(198912.0 - 322.0));
}
