#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <Kokkos_Core.hpp>

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  doctest::Context context(argc, argv);
  return context.run();
}
