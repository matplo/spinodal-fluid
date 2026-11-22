// WP7: collision-run production driver (apps/spf_collide.cpp) and the box / dx key logic.
// The driver is run as a program (SPF_COLLIDE_EXE) on a small Glauber-Gaussians state, so that no SMASH output is needed; the backend is that of
// the build (Serial, OpenMP or CUDA).
#include <doctest/doctest.h>

#include "spf/config.hpp"

#include <sys/wait.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace spf;

namespace {

std::string scratch(const std::string& name) {
  const std::string d = (std::filesystem::temp_directory_path() / ("spf_wp7_" + name)).string();
  std::filesystem::remove_all(d);
  std::filesystem::create_directories(d);
  return d;
}

nlohmann::json read_json(const std::string& path) {
  std::ifstream in(path);
  REQUIRE(in.good());
  return nlohmann::json::parse(in);
}

// Small box: 24 fm cube, 3 A GeV Glauber nucleons of width 1 fm; cold-thermal two-phase EoS (no table needed).
int run_driver(const std::string& out, const std::string& extra, const std::string& dx, const char* log = "driver.log") {
  const std::string cmd = std::string("\"") + SPF_COLLIDE_EXE + "\" --out " + out + " --config " + SPF_SOURCE_DIR +
                          "/config/collisions/common.json ic.generator=glauber_gaussians ic.e_lab_gev=3 eos.model=cold_thermal_two_phase" +
                          " grid.dx_fm=" + dx + " grid.Lx_fm=24 grid.Ly_fm=24 grid.Lz_fm=24 run.n_events=1 " + extra + " > " + out + "/" + log + " 2>&1";
  const int st = std::system(cmd.c_str());
  return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

}  // namespace

TEST_CASE("T7.1 box lengths and dx: the cell counts follow box / dx") {
  Config cfg = Config::from_file(SPF_DEFAULT_CONFIG);
  // default: no box lengths, the explicit cell counts are used
  GridCells g0 = grid_cells(cfg);
  CHECK(g0.n[0] == cfg.get<int>("grid.nx"));
  cfg.set_from_string("grid.Lx_fm=40");
  cfg.set_from_string("grid.Ly_fm=40");
  cfg.set_from_string("grid.Lz_fm=48");
  cfg.set_from_string("grid.dx_fm=0.2");
  GridCells g = grid_cells(cfg);
  printf("[T7.1] dx = 0.2: %d x %d x %d; ", g.n[0], g.n[1], g.n[2]);
  CHECK(g.n[0] == 200);
  CHECK(g.n[1] == 200);
  CHECK(g.n[2] == 240);
  cfg.set_from_string("grid.dx_fm=0.1");  // the only key that changes
  g = grid_cells(cfg);
  printf("dx = 0.1: %d x %d x %d; ", g.n[0], g.n[1], g.n[2]);
  CHECK(g.n[0] == 400);
  CHECK(g.n[1] == 400);
  CHECK(g.n[2] == 480);
  CHECK(std::fabs(g.length_fm[2] - 48.0) < 1e-12);
  cfg.set_from_string("grid.dx_fm=0.25");
  g = grid_cells(cfg);
  printf("dx = 0.25: %d x %d x %d\n", g.n[0], g.n[1], g.n[2]);
  CHECK(g.n[0] == 160);
  CHECK(g.n[2] == 192);
  cfg.set_from_string("grid.dx_fm=0.3");  // 40 / 0.3 is not an integer
  CHECK_THROWS_AS(grid_cells(cfg), ConfigError);
  cfg.set_from_string("grid.dx_fm=-0.2");
  CHECK_THROWS_AS(grid_cells(cfg), ConfigError);
}

TEST_CASE("T7.2 driver: Glauber-Gaussians event runs to the end time with 0 recovery failures; summary fields") {
  const std::string out = scratch("run");
  REQUIRE(run_driver(out, "run.t_max_after_init_fm=4.0", "0.4") == 0);
  const auto j = read_json(out + "/event_000/summary.json");
  for (const char* key : {"event", "status", "e_lab_gev", "t_init_fm", "steps", "wall_total_s", "wall_evolve_s", "wall_per_step_s", "backend",
                          "recovery_failures", "floor_activations", "limiter_cells_run", "ic_limiter_cells", "baryon_leaving_fraction",
                          "energy_leaving_fraction", "fluid_end_time_fm", "max_enhancement", "grid", "code", "peak_rss_MB"})
    CHECK_MESSAGE(j.contains(key), "missing summary field ", key);
  CHECK(j.at("status") == "t_max");
  CHECK(j.at("recovery_failures").get<long>() == 0);
  CHECK(j.at("steps").get<long>() == 25);  // 4 fm / (0.4 * 0.4 fm)
  CHECK(j.at("grid").at("nx").get<int>() == 60);
  CHECK(std::fabs(j.at("baryon_leaving_fraction").get<double>()) < 1e-4);
  CHECK(std::fabs(j.at("e_lab_gev").get<double>() - 3.0) < 1e-12);
  CHECK(std::filesystem::exists(out + "/event_000/observables.csv"));
  CHECK(std::filesystem::exists(out + "/config_used.json"));
  CHECK(std::filesystem::exists(out + "/event_000/config_used.json"));
  const auto echo = read_json(out + "/config_used.json");
  CHECK(echo.at("runtime").contains("git"));
  printf("[T7.2] backend %s, status %s, %ld steps, failures %ld, floor %ld, baryon leaving %.3e, energy leaving %.3e, wall %.2f s\n",
         j.at("backend").get<std::string>().c_str(), j.at("status").get<std::string>().c_str(), j.at("steps").get<long>(),
         j.at("recovery_failures").get<long>(), j.at("floor_activations").get<long>(), j.at("baryon_leaving_fraction").get<double>(),
         j.at("energy_leaving_fraction").get<double>(), j.at("wall_total_s").get<double>());
  // resume: a second call finds the finished event and does not run it again
  REQUIRE(run_driver(out, "run.t_max_after_init_fm=4.0", "0.4", "driver2.log") == 0);
  std::ifstream l2(out + "/driver2.log");
  const std::string log((std::istreambuf_iterator<char>(l2)), std::istreambuf_iterator<char>());
  CHECK(log.find("skipped") != std::string::npos);
  std::filesystem::remove_all(out);
}

TEST_CASE("T7.3 driver: changing only grid.dx_fm changes the cell counts and not the box") {
  const std::string o1 = scratch("dx04"), o2 = scratch("dx08");
  REQUIRE(run_driver(o1, "run.t_max_after_init_fm=1.0", "0.4") == 0);
  REQUIRE(run_driver(o2, "run.t_max_after_init_fm=1.0", "0.8") == 0);
  const auto a = read_json(o1 + "/event_000/summary.json"), b = read_json(o2 + "/event_000/summary.json");
  CHECK(a.at("grid").at("nx").get<int>() == 60);
  CHECK(b.at("grid").at("nx").get<int>() == 30);
  CHECK(a.at("grid").at("Lx_fm").get<double>() == doctest::Approx(b.at("grid").at("Lx_fm").get<double>()));
  CHECK(a.at("recovery_failures").get<long>() == 0);
  CHECK(b.at("recovery_failures").get<long>() == 0);
  printf("[T7.3] dx = 0.4: %d cells per side, dx = 0.8: %d cells per side, box %.1f fm in both\n", a.at("grid").at("nx").get<int>(),
         b.at("grid").at("nx").get<int>(), a.at("grid").at("Lx_fm").get<double>());
  std::filesystem::remove_all(o1);
  std::filesystem::remove_all(o2);
}

TEST_CASE("T7.4 driver: stops the event with status recovery_failure when the failure count exceeds the threshold") {
  const std::string o1 = scratch("fail"), o2 = scratch("tol");
  // without the floor on the gradient pressure the dilute fringe fails the primitive recovery (WP5, T5.4)
  REQUIRE(run_driver(o1, "gradient.pex_floor_kappa=-1 run.t_max_after_init_fm=4.0 run.max_recovery_failures=0", "0.4") == 3);
  const auto a = read_json(o1 + "/event_000/summary.json");
  CHECK(a.at("status") == "recovery_failure");
  CHECK(a.at("recovery_failures").get<long>() > 0);
  CHECK(a.at("steps").get<long>() < 25);
  // the same event with a threshold above the failure count runs to the end time and still reports the failures
  REQUIRE(run_driver(o2, "gradient.pex_floor_kappa=-1 run.t_max_after_init_fm=4.0 run.max_recovery_failures=100000000", "0.4") == 0);
  const auto b = read_json(o2 + "/event_000/summary.json");
  CHECK(b.at("status") == "t_max");
  CHECK(b.at("recovery_failures").get<long>() > 0);
  printf("[T7.4] threshold 0: status %s after %ld steps with %ld failures; threshold 1e8: status %s, %ld steps, %ld failures\n",
         a.at("status").get<std::string>().c_str(), a.at("steps").get<long>(), a.at("recovery_failures").get<long>(),
         b.at("status").get<std::string>().c_str(), b.at("steps").get<long>(), b.at("recovery_failures").get<long>());
  std::filesystem::remove_all(o1);
  std::filesystem::remove_all(o2);
}
