// Prints the build/runtime description, runs a small reduction on the default execution space,
// and optionally echoes the effective configuration.
//
//   spf_info [--config defaults.json] [--override file.json]... [--echo dir] [section.key=value]...
#include "spf/config.hpp"
#include "spf/types.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  std::string defaults = SPF_DEFAULT_CONFIG, echo_dir;
  std::vector<std::string> override_files, assignments;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--config" && i + 1 < argc) {
      defaults = argv[++i];
    } else if (a == "--override" && i + 1 < argc) {
      override_files.push_back(argv[++i]);
    } else if (a == "--echo" && i + 1 < argc) {
      echo_dir = argv[++i];
    } else if (a.find('=') != std::string::npos && a.rfind("--", 0) != 0) {
      assignments.push_back(a);
    } else {
      std::cerr << "unrecognised argument: " << a << "\n";
      return 2;
    }
  }
  try {
    spf::Config cfg = spf::Config::from_file(defaults);
    for (const auto& f : override_files) cfg.merge_file(f);
    for (const auto& s : assignments) cfg.set_from_string(s);

    std::cout << spf::runtime_info().dump(2) << "\n";

    // sum_{i>=1} 1/i^2 -> pi^2/6 ; exercises parallel_reduce on the default execution space
    const long n = 200000000;
    Kokkos::Timer timer;
    double sum = 0.0;
    Kokkos::parallel_reduce(
        "spf_info_reduction", Kokkos::RangePolicy<spf::ExecSpace>(0, n),
        KOKKOS_LAMBDA(const long i, double& acc) {
          const double x = static_cast<double>(i + 1);
          acc += 1.0 / (x * x);
        },
        sum);
    Kokkos::fence();
    const double seconds = timer.seconds();
    const double exact = std::acos(-1.0) * std::acos(-1.0) / 6.0;
    const double expected = exact - 1.0 / n;  // tail of the series; rounding in the summation stays below 1e-7
    std::printf("reduction: n = %ld  sum = %.12f  expected = %.12f  diff = %.2e  %.3f s\n", n, sum, expected, sum - expected,
                seconds);
    if (std::fabs(sum - expected) > 1e-7) {
      std::cerr << "error: reduction result outside tolerance\n";
      return 1;
    }

    if (!echo_dir.empty()) {
      cfg.echo(echo_dir);
      std::cout << "wrote " << echo_dir << "/config_used.json\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
