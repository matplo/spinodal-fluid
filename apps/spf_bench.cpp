// Benchmark of the 3D solver: periodic box, smooth random initial state, cell updates per second.
//   spf_bench N eos steps [key=value ...]     eos = ideal | cold_thermal_two_phase
// Timed: `steps` full time steps after one untimed warm-up step; setup is excluded. Reports the conservation totals before
// the warm-up step and after the timed loop, and the recovery-failure and clamp counters.
#include "spf/eos/eos.hpp"
#include "spf/hydro3d.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <sys/resource.h>

namespace {

template <class EoS>
int run(const EoS& eos, const spf::Config& cfg, int n, long steps, const char* eos_name) {
  spf::Hydro3DParams hp = spf::Hydro3DParams::from_config(cfg);
  hp.nx = hp.ny = hp.nz = n;
  spf::Hydro3D<EoS> h(eos, hp);
  h.set_state(spf::smooth_random_state(n, n, n));
  const spf::Totals t0 = h.totals();
  h.step();  // warm-up
  Kokkos::fence();
  Kokkos::Timer timer;
  for (long s = 0; s < steps; ++s) h.step();
  Kokkos::fence();
  const double wall = timer.seconds();
  const spf::Totals t1 = h.totals();
  const double cells = std::pow(static_cast<double>(n), 3);
  std::printf("backend=%s concurrency=%d eos=%s N=%d steps=%ld (+1 warm-up) sweep_order=%s\n", spf::ExecSpace::name(),
              static_cast<int>(spf::ExecSpace().concurrency()), eos_name, n, steps,
              hp.order == spf::SweepOrder::Permuted ? "permuted" : "fixed");
  std::printf("wall_s=%.6f  time_per_step_s=%.6f  cell_updates_per_s=%.6e\n", wall, wall / static_cast<double>(steps),
              cells * static_cast<double>(steps) / wall);
  std::printf("totals_before: R=%.15e E=%.15e Mx=%.15e My=%.15e Mz=%.15e\n", t0.R, t0.E, t0.Mx, t0.My, t0.Mz);
  std::printf("totals_after:  R=%.15e E=%.15e Mx=%.15e My=%.15e Mz=%.15e\n", t1.R, t1.E, t1.Mx, t1.My, t1.Mz);
  std::printf("rel_change: R=%.3e E=%.3e  (Mi/E: %.3e %.3e %.3e)\n", std::fabs(t1.R - t0.R) / t0.R, std::fabs(t1.E - t0.E) / t0.E,
              std::fabs(t1.Mx - t0.Mx) / t0.E, std::fabs(t1.My - t0.My) / t0.E, std::fabs(t1.Mz - t0.Mz) / t0.E);
  std::printf("recover_failures=%ld clamped=%ld\n", h.n_recover_fail_total, h.n_clamped_total);
  struct rusage ru;
  getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
  const double rss_mib = static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);  // bytes
#else
  const double rss_mib = static_cast<double>(ru.ru_maxrss) / 1024.0;  // KiB
#endif
  std::printf("peak_host_rss_MiB=%.1f\n", rss_mib);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    try {
      if (argc < 4) {
        std::fprintf(stderr, "usage: spf_bench N eos steps [key=value ...]\n");
        rc = 2;
      } else {
        const int n = std::atoi(argv[1]);
        const std::string eos = argv[2];
        const long steps = std::atol(argv[3]);
        spf::Config cfg = spf::Config::from_file(SPF_DEFAULT_CONFIG);
        for (int a = 4; a < argc; ++a) {
          const std::string s = argv[a];
          if (s.find('=') != std::string::npos && s.rfind("--", 0) != 0) cfg.set_from_string(s);
        }
        if (eos == "ideal") {
          rc = run(spf::EosIdeal{}, cfg, n, steps, "ideal");
        } else if (eos == "cold_thermal_two_phase") {
          const std::string path = std::string(SPF_SOURCE_DIR) + "/" + cfg.get<std::string>("eos.cold_curve_path");
          const spf::ColdCurveTable table = spf::ColdCurveTable::load(path);
          const spf::EosColdThermal e(table, spf::ColdVariant::two_phase, cfg.get<spf::Real>("eos.gamma_th"));
          rc = run(e, cfg, n, steps, "cold_thermal_two_phase");
        } else {
          std::fprintf(stderr, "unknown eos '%s'\n", eos.c_str());
          rc = 2;
        }
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      rc = 1;
    }
  }
  Kokkos::finalize();
  return rc;
}
