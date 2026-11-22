// Post-processing of Hydro3D snapshots (WP6).
//   spf_observe [--out DIR] [key=value ...] snapshot [snapshot ...]
// A snapshot is the base name (or the .json / .bin file) written by Hydro3D::write_snapshot; it must contain the fields R rho eps vx vy.
// The configuration is the one echoed in the first snapshot, overridden by key=value arguments (e.g. obs.hist_times_fm=[2.0,4.0],
// obs.clump_connectivity=26). The equation of state for the unstable-fraction mask is built from eos.model of that configuration.
// Output (DIR, default "obs_out"): the files of ObservableRecorder (docs/observables.md), one record per snapshot.
#include "spf/eos/eos.hpp"
#include "spf/observables.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace {

// Defaults file first (so that every key exists), then the echo of the snapshot, then the command line.
spf::Config build_config(const nlohmann::json& echo, const std::vector<std::string>& overrides) {
  spf::Config cfg = spf::Config::from_file(SPF_DEFAULT_CONFIG);
  // the echo is the plain configuration (Config::json()); a config_used.json file ({config, runtime}) is accepted too
  nlohmann::json e = (echo.contains("config") && !echo.contains("grid")) ? echo.at("config") : echo;
  cfg.merge_json(e);
  for (const auto& s : overrides) cfg.set_from_string(s);
  return cfg;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    try {
      std::string out = "obs_out";
      std::vector<std::string> overrides, snaps;
      for (int a = 1; a < argc; ++a) {
        const std::string s = argv[a];
        if (s == "--out" && a + 1 < argc) out = argv[++a];
        else if (s.rfind("--", 0) == 0) { /* Kokkos flags are consumed by initialize */ }
        else if (s.find('=') != std::string::npos && s.find('/') == std::string::npos) overrides.push_back(s);
        else snaps.push_back(s);
      }
      if (snaps.empty()) {
        std::fprintf(stderr, "usage: spf_observe [--out DIR] [key=value ...] snapshot [snapshot ...]\n");
        rc = 2;
      } else {
        const spf::LoadedSnapshot first = spf::load_snapshot_fields(snaps.front());
        const spf::Config cfg = build_config(first.config, overrides);
        const spf::EosModel m = spf::eos_model_from_config(cfg);
        const std::string src = SPF_SOURCE_DIR;
        if (m == spf::EosModel::ideal) {
          spf::observe_snapshots(cfg, spf::EosIdeal{}, snaps, out);
        } else if (m == spf::EosModel::gamma_law) {
          spf::observe_snapshots(cfg, spf::EosGammaLaw(cfg.get<spf::Real>("eos.gamma_law_gamma"), cfg.get<spf::Real>("eos.gamma_law_mass_GeV")),
                                 snaps, out);
        } else if (m == spf::EosModel::cold_thermal_two_phase || m == spf::EosModel::cold_thermal_maxwell) {
          std::string path = cfg.get<std::string>("eos.cold_curve_path");
          if (!path.empty() && path[0] != '/') path = src + "/" + path;
          const spf::ColdCurveTable table = spf::ColdCurveTable::load(path);
          const spf::ColdVariant v = m == spf::EosModel::cold_thermal_two_phase ? spf::ColdVariant::two_phase : spf::ColdVariant::maxwell;
          const spf::EosColdThermal e(table, v, cfg.get<spf::Real>("eos.gamma_th"));
          const spf::Real b = cfg.get<spf::Real>("eos.scale_b"), c = cfg.get<spf::Real>("eos.scale_c");
          if (b == 1.0 && c == 1.0) spf::observe_snapshots(cfg, e, snaps, out);
          else spf::observe_snapshots(cfg, spf::EosScaled<spf::EosColdThermal>(e, b, c), snaps, out);
        } else {
          const spf::EosTableData data = spf::EosTableData::from_config(cfg);
          const spf::EosTable e = data.eos(cfg);
          const spf::Real b = cfg.get<spf::Real>("eos.scale_b"), c = cfg.get<spf::Real>("eos.scale_c");
          if (b == 1.0 && c == 1.0) spf::observe_snapshots(cfg, e, snaps, out);
          else spf::observe_snapshots(cfg, spf::EosScaled<spf::EosTable>(e, b, c), snaps, out);
        }
        std::printf("spf_observe: %zu snapshot(s) -> %s\n", snaps.size(), out.c_str());
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      rc = 1;
    }
  }
  Kokkos::finalize();
  return rc;
}
