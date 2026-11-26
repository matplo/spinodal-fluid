// WP7: production driver for collision runs.
//   spf_collide [--out DIR] [--config FILE]... [key=value ...]
// One process runs a list of events (run.event_first, run.n_events; run.n_events = -1: every event of the OSCAR file) of one initial-state
// source (ic.generator, ic.oscar_path, ...): for each event it deposits the initial state (WP5), evolves it with Hydro3D, records the
// observables of WP6 with the ObservableRecorder into <DIR>/event_<NNN>/ and writes <DIR>/event_<NNN>/summary.json.
// An event ends at the fluid end (all cells eps < obs.fluid_end_eps_eps_s eps_s, counted only after some record had a cell at or above
// that value; run.stop_at_fluid_end), at run.t_max_after_init_fm after the start of the fluid stage, or when the number of primitive-recovery
// failures exceeds run.max_recovery_failures. Events whose summary.json exists are skipped (run.resume) unless they ended in failure and
// run.retry_failed is true. Exit code 0: every event ended with status fluid_end or t_max; 3: some event failed (see summary.json).
// With particlisation.enabled (WP10a, config/collisions/common.json) the initial slice is emitted and removed at t_init, the event runs until no
// cell has eps >= eps_sw (cap run.t_max_after_init_fm) and the surface is written to event_NNN/event_NNN.surface.{bin,json}
// (docs/particlisation_format.md).
// Documentation: docs/running_collisions.md.
#include "spf/corona.hpp"
#include "spf/eos/dispatch.hpp"
#include "spf/hydro3d.hpp"
#include "spf/initial_state.hpp"
#include "spf/observables.hpp"
#include "spf/surface.hpp"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
double seconds_since(const Clock::time_point& t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

double peak_rss_mb() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
  return ru.ru_maxrss / 1.0e6;  // bytes
#else
  return ru.ru_maxrss / 1.0e3;  // kilobytes
#endif
}

std::string event_dir_name(int id) {
  char b[32];
  std::snprintf(b, sizeof b, "event_%03d", id);
  return b;
}

// eos.table_path empty: take it from eos.table_path_two_phase / eos.table_path_maxwell according to eos.model; relative paths are
// resolved against the source directory (the tables of data/eos/generated live there).
void resolve_eos_paths(spf::Config& cfg, const std::string& src) {
  const spf::EosModel m = spf::eos_model_from_config(cfg);
  auto absolutise = [&](const std::string& p) { return (!p.empty() && p[0] != '/') ? src + "/" + p : p; };
  if (spf::eos_model_uses_table(m)) {
    std::string p = cfg.get<std::string>("eos.table_path");
    if (p.empty()) p = cfg.get<std::string>(m == spf::EosModel::table_two_phase ? "eos.table_path_two_phase" : "eos.table_path_maxwell");
    cfg.set_from_string("eos.table_path=" + nlohmann::json(absolutise(p)).dump());
  }
}

// Sums of the computational-frame baryon density R dV over the slabs of one axis (x, y or z).
nlohmann::json marginals(const std::vector<double>& R, int nx, int ny, int nz, double dx) {
  std::vector<double> mx(nx, 0.0), my(ny, 0.0), mz(nz, 0.0);
  size_t q = 0;
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k, ++q) { mx[i] += R[q]; my[j] += R[q]; mz[k] += R[q]; }
  const double dV = dx * dx * dx;
  for (auto* v : {&mx, &my, &mz}) for (auto& x : *v) x *= dV;
  return {{"x", mx}, {"y", my}, {"z", mz}};
}

// sha256 recorded in the header of the EoS table in use (empty for the analytic and cold-thermal models)
std::string eos_table_sha256(const spf::Config& cfg) {
  if (!spf::eos_model_uses_table(spf::eos_model_from_config(cfg))) return "";
  std::string p = cfg.get<std::string>("eos.table_path");
  if (p.size() > 5 && p.compare(p.size() - 5, 5, ".json") == 0) p.resize(p.size() - 5);
  std::ifstream in(p + ".json");
  const nlohmann::json h = nlohmann::json::parse(in, nullptr, false);
  return (!h.is_discarded() && h.contains("sha256")) ? h["sha256"].get<std::string>() : "";
}

bool status_is_ok(const std::string& s) { return s == "fluid_end" || s == "t_max"; }

template <class EoS>
nlohmann::json run_event(const spf::Config& cfg, const EoS& eos, int id, const std::string& edir, const std::string& build_info_backend) {
  using spf::Real;
  nlohmann::json J;
  J["event"] = id;
  J["backend"] = build_info_backend;
  J["status"] = "error";
  const auto t_all = Clock::now();
  std::filesystem::create_directories(edir);
  const spf::GridCells gc = spf::grid_cells(cfg);
  const spf::Grid grid = spf::Grid::make_3d(gc.n[0], gc.n[1], gc.n[2], gc.dx);
  J["grid"] = {{"nx", gc.n[0]}, {"ny", gc.n[1]}, {"nz", gc.n[2]}, {"dx_fm", gc.dx},
               {"Lx_fm", gc.length_fm[0]}, {"Ly_fm", gc.length_fm[1]}, {"Lz_fm", gc.length_fm[2]},
               {"cells", static_cast<double>(gc.n[0]) * gc.n[1] * gc.n[2]}, {"boundary", cfg.get<std::string>("grid.boundary")}};
  J["eos_model"] = cfg.get<std::string>("eos.model");
  J["e_lab_gev"] = cfg.get<double>("ic.e_lab_gev");
  try {
    // ------------------------------------------------------------ initial state
    const auto t_ic = Clock::now();
    spf::InitialState st = spf::build_initial_state(cfg, grid);
    const Real v3 = grid.dx * grid.dx * grid.dx;
    // ------------------------------------------------------------ corona (WP10e, decision S31): core/corona split before the deposition
    const spf::ParticlisationParams pp = spf::ParticlisationParams::from_config(cfg);
    const spf::CoronaParams cp = spf::CoronaParams::from_config(cfg);
    nlohmann::json corona_json = {{"mode", cp.mode_name()}, {"applied", false}, {"n_corona", 0}};
    if (pp.enabled && cp.mode == spf::CoronaMode::CoreCorona) {
      const auto t_cor = Clock::now();
      if (st.particles.empty()) throw std::runtime_error("particlisation.corona.mode = core_corona needs a hadron list (ic.generator smash_cascade, oscar_file or glauber_gaussians)");
      spf::Config rc = cfg;
      const std::string ref_model = cp.reference_eos;
      const bool same_model = ref_model == cfg.get<std::string>("eos.model");
      std::string ref_table = cfg.get<std::string>("particlisation.corona.reference_table_path");
      rc.set_from_string("eos.model=" + ref_model);
      if (!ref_table.empty()) rc.set_from_string("eos.table_path=" + nlohmann::json(ref_table).dump());
      else if (!same_model) rc.set_from_string("eos.table_path=\"\"");
      resolve_eos_paths(rc, SPF_SOURCE_DIR);
      spf::CoronaSplit split;
      spf::with_eos(rc, SPF_SOURCE_DIR, [&](const auto& er) { split = spf::classify_corona(cfg, grid, st, er, pp.eps_sw()); });
      const std::string odir0 = pp.output_dir.empty() ? edir : pp.output_dir;
      std::filesystem::create_directories(odir0);
      const std::string cpath = odir0 + "/" + event_dir_name(id) + ".corona.oscar";
      spf::write_corona_oscar(cpath, st.particles, split.is_core, id, "spf_collide corona hadrons (WP10e): SMASH hadrons with eps(x_i) < eps_sw at t_init");
      const spf::DepositResult all_dep = st.dep;  // deposition of the full list (classification state)
      const double E_all_grid = all_dep.grid.E, B_all_grid = all_dep.grid.B;
      spf::redeposit_subset(cfg, grid, st, split.is_core);
      std::map<long, long> species;
      for (size_t i = 0; i < st.particles.size(); ++i) if (!split.is_core[i]) species[st.particles[i].pdg] += 1;
      nlohmann::json sp = nlohmann::json::object();
      for (const auto& kv : species) sp[std::to_string(kv.first)] = kv.second;
      const spf::CoronaTotals all = spf::corona_totals(st.particles, split.is_core, true);
      corona_json = {{"mode", cp.mode_name()}, {"applied", true}, {"reference_eos", ref_model}, {"eps_sw_GeV_fm3", pp.eps_sw()},
                     {"n_particles_total", static_cast<long>(st.particles.size())}, {"n_core", split.core.n}, {"n_corona", split.corona.n},
                     {"corona", spf::corona_totals_json(split.corona)}, {"core", spf::corona_totals_json(split.core)},
                     {"corona_species_pdg", sp}, {"file", event_dir_name(id) + ".corona.oscar"},
                     {"deposited_all_E_GeV", E_all_grid}, {"deposited_all_B", B_all_grid},
                     {"deposited_core_E_GeV", st.dep.grid.E}, {"deposited_core_B", st.dep.grid.B},
                     {"corona_index_comment", "hadron order of the OSCAR input list; corona.oscar keeps that order"}};
      (void)all;
      J["wall_corona_s"] = seconds_since(t_cor);
    }
    J["corona"] = corona_json;
    const spf::Hydro3DParams hp = spf::Hydro3DParams::from_config(cfg);
    spf::Hydro3D<EoS> h(eos, hp);
    h.set_conserved(st.dep.R, st.dep.Mx, st.dep.My, st.dep.Mz, st.dep.E);
    const long fail_init = h.update_primitives();
    const long n0 = h.n_recover_fail_total;
    const auto tot0 = h.totals();
    Kokkos::fence();
    J["wall_ic_s"] = seconds_since(t_ic);
    J["generator"] = st.generator;
    J["t_init_fm"] = st.t_init_fm;
    J["oscar_time_fm"] = st.oscar_time_fm;
    J["n_hadrons"] = st.n_hadrons;
    J["baryon_in_particles"] = st.baryon_total;
    J["baryon_on_grid_initial"] = tot0.R * v3;
    J["energy_on_grid_initial_GeV"] = tot0.E * v3;
    J["ic_outside_fraction_B"] = st.dep.outside_fraction_B;
    J["ic_limiter_cells"] = st.limiter_cells;
    J["recovery_failures_initial_state"] = fail_init;
    const bool cell_diag = cfg.get<bool>("run.cell_diagnostics");
    h.track_vacuum = cell_diag;
    // ------------------------------------------------------------ particlisation surface (WP10a): initial slice before the first record
    std::unique_ptr<spf::SurfaceFinder<EoS>> sf;
    auto tot_ref = tot0;  // reference of the leaving fractions: the state that is evolved
    if (pp.enabled) {
      const auto t_sf = Clock::now();
      typename spf::SurfaceFinder<EoS>::Meta meta;
      meta.t_init = st.t_init_fm;
      meta.origin = spf::centred_origin(grid);
      meta.json = {{"event_id", id}, {"E_lab_GeV", cfg.get<double>("ic.e_lab_gev")}, {"eos_model", cfg.get<std::string>("eos.model")},
                   {"eos_table_sha256", eos_table_sha256(cfg)}, {"eos_table_interpolation", cfg.get<std::string>("eos.table_interpolation")},
                   {"eos_scale_b", cfg.get<double>("eos.scale_b")}, {"eos_scale_c", cfg.get<double>("eos.scale_c")},
                   {"gradient_form", cfg.get<std::string>("gradient.form")}, {"generator", st.generator},
                   {"oscar_path", cfg.get<std::string>("ic.oscar_path")}, {"oscar_time_fm", st.oscar_time_fm}};
      std::string odir = pp.output_dir.empty() ? edir : pp.output_dir;
      sf = std::make_unique<spf::SurfaceFinder<EoS>>(eos, pp, grid, odir + "/" + event_dir_name(id), meta);
      sf->begin(h);
      tot_ref = h.totals();
      Kokkos::fence();
      J["wall_surface_begin_s"] = seconds_since(t_sf);
    }
    const bool want_marg = cfg.get<bool>("run.marginals");
    nlohmann::json marg;
    if (want_marg) marg["start"] = marginals(h.interior_copy(h.U[0]), gc.n[0], gc.n[1], gc.n[2], gc.dx);

    // ------------------------------------------------------------ evolution
    spf::ObservableRecorder<EoS> rec(cfg, eos, edir);
    const long max_fail = cfg.get<long>("run.max_recovery_failures");
    const double t_max = cfg.get<double>("run.t_max_after_init_fm");
    const bool stop_fe = cfg.get<bool>("run.stop_at_fluid_end");
    const double eps_thr = cfg.get<double>("obs.fluid_end_eps_eps_s");
    const long nmax = static_cast<long>(std::ceil(t_max / h.dt() - 1e-9));
    std::string status = "t_max";
    double fluid_end_time = std::nan("");
    bool armed = false;
    double wall_evolve = 0.0, wall_obs = 0.0;
    // optional diagnostic (run.cell_diagnostics): floor and vacuum-cut cells, one line per step in cell_diagnostics.csv
    std::ofstream dcsv;
    nlohmann::json diag_sum;
    double dmax_nfloor = 0, dmax_Ef = 0, dmax_Rf = 0, t_Ef = 0, t_Rf = 0, t_nfloor = 0;
    nlohmann::json diag0;
    auto diagnose = [&]() {
      if (!cell_diag) return;
      const auto d = h.diagnose_cells();
      const double Et = d.E_tot != 0.0 ? d.E_tot : 1.0, Rt = d.R_tot != 0.0 ? d.R_tot : 1.0;
      const double Ef = d.E_floor / Et, Rf = d.R_floor / Rt;
      const double E0 = tot0.E != 0.0 ? tot0.E * v3 : 1.0, R0 = tot0.R != 0.0 ? tot0.R * v3 : 1.0;
      if (!dcsv.is_open()) {
        dcsv.open(edir + "/cell_diagnostics.csv");
        dcsv << "step,t_fm,n_floor_cells,n_failed_cells,E_floor_fraction,R_floor_fraction,n_vac_cells,E_vac_fraction,R_vac_fraction,"
                "vac_removed_cells_cumulative,vac_removed_E_fraction_cumulative,vac_removed_R_fraction_cumulative\n";
        diag0 = {{"n_floor_cells", d.n_floor}, {"E_floor_fraction", Ef}, {"R_floor_fraction", Rf}, {"n_vac_cells", d.n_vac},
                 {"E_vac_fraction", d.E_vac / Et}, {"R_vac_fraction", d.R_vac / Rt}};
      }
      char b[512];
      std::snprintf(b, sizeof b, "%ld,%.10g,%ld,%ld,%.10e,%.10e,%ld,%.10e,%.10e,%ld,%.10e,%.10e\n", h.steps, h.time, d.n_floor, d.n_fail, Ef, Rf, d.n_vac,
                    d.E_vac / Et, d.R_vac / Rt, h.vac_removed_cells, h.vac_removed_E / E0, h.vac_removed_R / R0);
      dcsv << b;
      if (d.n_floor > dmax_nfloor) { dmax_nfloor = d.n_floor; t_nfloor = h.time; }
      if (Ef > dmax_Ef) { dmax_Ef = Ef; t_Ef = h.time; }
      if (Rf > dmax_Rf) { dmax_Rf = Rf; t_Rf = h.time; }
    };
    auto observe = [&]() {
      const auto t0 = Clock::now();
      const size_t before = rec.history().size();
      rec.on_step(h);
      wall_obs += seconds_since(t0);
      if (rec.history().size() > before) {
        const double e = rec.history().back().eps_max_eps_s;
        if (e >= eps_thr) armed = true;
        else if (armed && std::isnan(fluid_end_time)) fluid_end_time = rec.history().back().time;
      }
    };
    if (fail_init > max_fail) {
      status = "recovery_failure";
    } else {
      diagnose();
      observe();
      long s = 0;
      for (; s < nmax; ++s) {
        const auto t0 = Clock::now();
        h.step();
        Kokkos::fence();
        wall_evolve += seconds_since(t0);
        if (h.n_recover_fail_total - n0 > max_fail) { status = "recovery_failure"; ++s; break; }
        diagnose();
        observe();
        if (sf) {
          // with particlisation the event runs until the surface is complete (no cell at or above eps_sw), capped by t_max
          if (sf->after_step(h)) { status = "fluid_end"; ++s; break; }
          continue;
        }
        if (stop_fe && !std::isnan(fluid_end_time)) { status = "fluid_end"; ++s; break; }
      }
    }
    if (h.steps % rec.params().every_steps != 0 && !rec.history().empty() && rec.history().back().step != h.steps) {
      // the last step is always recorded, whatever the recording interval
      rec.record(spf::ObservableRecorder<EoS>::fields_of(h), h.time, h.steps);
    }
    rec.finalize();
    const long fail_run = h.n_recover_fail_total - n0;
    const auto tot1 = h.totals();
    // ------------------------------------------------------------ summary
    J["status"] = status;
    J["steps"] = h.steps;
    J["dt_fm"] = h.dt();
    J["time_after_init_fm"] = h.time;
    J["t_max_after_init_fm"] = t_max;
    J["fluid_end_time_fm"] = std::isnan(fluid_end_time) ? nlohmann::json(nullptr) : nlohmann::json(fluid_end_time);
    J["recovery_failures"] = fail_run + fail_init;
    J["recovery_failures_during_run"] = fail_run;
    J["max_recovery_failures_allowed"] = max_fail;
    J["floor_activations"] = h.n_pex_floor_total;
    J["limiter_cells_run"] = h.n_clamped_total;
    J["baryon_on_grid_final"] = tot1.R * v3;
    J["energy_on_grid_final_GeV"] = tot1.E * v3;
    J["baryon_leaving_fraction"] = tot_ref.R != 0.0 ? 1.0 - tot1.R / tot_ref.R : 0.0;
    J["energy_leaving_fraction"] = tot_ref.E != 0.0 ? 1.0 - tot1.E / tot_ref.E : 0.0;
    J["momentum_z_final_GeV"] = tot1.Mz * v3;
    {
      const auto& H = rec.history();
      double emax = 0.0, ufmax = 0.0, tuf = 0.0;
      for (const auto& r : H) { emax = std::max(emax, r.eps_max_eps_s); if (r.unstable_fraction > ufmax) { ufmax = r.unstable_fraction; tuf = r.time; } }
      J["eps_max_initial_eps_s"] = H.empty() ? 0.0 : H.front().eps_max_eps_s;
      J["eps_max_peak_eps_s"] = emax;
      J["unstable_fraction_peak"] = ufmax;
      J["unstable_fraction_peak_time_fm"] = tuf;
      J["unstable_fraction_final"] = H.empty() ? 0.0 : H.back().unstable_fraction;
      nlohmann::json me;
      for (int n = 2; n <= spf::kObsMaxMoment; ++n) {
        double best = -1, tb = 0;
        for (const auto& r : H) {
          const double e = H.front().moment[n] != 0.0 ? r.moment[n] / H.front().moment[n] : 0.0;
          if (e > best) { best = e; tb = r.time; }
        }
        me["N" + std::to_string(n)] = {{"value", best}, {"time_fm", tb}};
      }
      J["max_enhancement"] = me;
    }
    if (cell_diag) {
      const double E0 = tot0.E != 0.0 ? tot0.E * v3 : 1.0, R0 = tot0.R != 0.0 ? tot0.R * v3 : 1.0;
      J["cell_diagnostics"] = {{"definition", "floor cells: cells whose last recovery used the floor on the extra pressure (state after each step); fractions of the "
                                              "energy / baryon number on the grid in those cells; vac: cells with E < shasta.e_vac_GeV_fm3"},
                               {"e_vac_GeV_fm3", cfg.get<double>("shasta.e_vac_GeV_fm3")},
                               {"initial_state", diag0},
                               {"max_floor_cells", dmax_nfloor}, {"time_of_max_floor_cells_fm", t_nfloor},
                               {"max_E_floor_fraction", dmax_Ef}, {"time_of_max_E_floor_fraction_fm", t_Ef},
                               {"max_R_floor_fraction", dmax_Rf}, {"time_of_max_R_floor_fraction_fm", t_Rf},
                               {"vacuum_cut_removed_cells_total", h.vac_removed_cells},
                               {"vacuum_cut_removed_E_fraction_of_initial", h.vac_removed_E / E0},
                               {"vacuum_cut_removed_R_fraction_of_initial", h.vac_removed_R / R0}};
    }
    if (sf) {
      const auto t_fin = Clock::now();
      sf->final_slice(h);  // deferred initial slice (corona.mode = keep): emit what has not crossed the surface
      const double wall_final = seconds_since(t_fin);
      nlohmann::json extra = {{"status", status}, {"steps", h.steps}, {"time_after_init_fm", h.time},
                              {"t_max_after_init_fm", t_max}, {"hit_t_max", !sf->closed()}, {"corona", corona_json},
                              {"corona_mode", cp.mode_name()}, {"wall_final_slice_s", wall_final}};
      sf->finish(h, extra);
      const auto& S = sf->summary(0);
      J["particlisation"] = {{"eps_sw_eps_s", pp.eps_sw_eps_s}, {"method", S["method"]}, {"closed", sf->closed()}, {"hit_t_max", !sf->closed()},
                             {"surface_file", sf->output_base(0) + ".surface.bin"}, {"n_outputs", sf->n_outputs()},
                             {"n_elements", S["n_elements"]}, {"n_elements_initial_slice", S["n_elements_initial_slice"]},
                             {"n_elements_surface", S["n_elements_surface"]}, {"n_elements_final_slice", S["n_elements_final_slice"]},
                             {"initial_slice", S["initial_slice"]}, {"corona_mode", cp.mode_name()}, {"closure", S["closure"]},
                             {"end_time_after_init_fm", h.time}, {"wall_surface_s", sf->wall_s()},
                             {"diagnostics", S["diagnostics"]},
                             {"leaving_fractions_relative_to", "state after the initial slice was removed"},
                             {"energy_on_grid_after_initial_slice_GeV", tot_ref.E * v3}, {"baryon_on_grid_after_initial_slice", tot_ref.R * v3}};
      if (!sf->closed()) std::printf("event %03d: surface not closed at t_max = %.2f fm/c (cells at or above eps_sw remain)\n", id, t_max);
    }
    J["wall_evolve_s"] = wall_evolve;
    J["wall_observe_s"] = wall_obs;
    J["wall_per_step_s"] = h.steps > 0 ? wall_evolve / h.steps : 0.0;
    J["peak_rss_MB"] = peak_rss_mb();
    if (want_marg) {
      marg["end"] = marginals(h.interior_copy(h.U[0]), gc.n[0], gc.n[1], gc.n[2], gc.dx);
      marg["dx_fm"] = gc.dx;
      marg["n"] = {gc.n[0], gc.n[1], gc.n[2]};
      std::ofstream(edir + "/marginals.json") << marg.dump() << "\n";
    }
  } catch (const std::exception& e) {
    J["status"] = "error";
    J["message"] = e.what();
  }
  J["wall_total_s"] = seconds_since(t_all);
  J["code"] = spf::runtime_info();
  std::ofstream(edir + "/summary.json") << J.dump(2) << "\n";
  return J;
}

template <class EoS>
int run_all(const spf::Config& cfg, const EoS& eos, const std::string& out, const std::vector<int>& ids) {
  const std::string backend = spf::runtime_info()["execution_space"].get<std::string>();
  const bool resume = cfg.get<bool>("run.resume"), retry = cfg.get<bool>("run.retry_failed");
  int n_bad = 0;
  nlohmann::json index = nlohmann::json::array();
  for (int id : ids) {
    const std::string edir = out + "/" + event_dir_name(id);
    const std::string sj = edir + "/summary.json";
    nlohmann::json J;
    bool skipped = false;
    if (resume && std::filesystem::exists(sj)) {
      std::ifstream in(sj);
      J = nlohmann::json::parse(in, nullptr, false);
      if (!J.is_discarded() && J.contains("status") && (status_is_ok(J["status"].get<std::string>()) || !retry)) skipped = true;
    }
    if (!skipped) {
      spf::Config ce = cfg;
      const std::string gen = cfg.get<std::string>("ic.generator");
      if (gen == "smash_cascade" || gen == "oscar_file") ce.set_from_string("ic.oscar_event=" + std::to_string(id));
      else ce.set_from_string("ic.seed=" + std::to_string(cfg.get<long>("ic.seed") + id));
      J = run_event(ce, eos, id, edir, backend);
    }
    const std::string st = J.value("status", std::string("error"));
    if (!status_is_ok(st)) ++n_bad;
    std::printf("event %03d: %s%s  steps=%ld  fail=%ld  wall=%.1f s\n", id, st.c_str(), skipped ? " (skipped, finished earlier)" : "",
                J.value("steps", 0L), J.value("recovery_failures", 0L), J.value("wall_total_s", 0.0));
    std::fflush(stdout);
    index.push_back({{"event", id}, {"status", st}, {"skipped", skipped}});
  }
  std::ofstream(out + "/run_index.json") << index.dump(1) << "\n";
  return n_bad == 0 ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    try {
      std::string out = "collide_out";
      std::vector<std::string> cfgfiles, overrides;
      for (int a = 1; a < argc; ++a) {
        const std::string s = argv[a];
        if (s == "--out" && a + 1 < argc) out = argv[++a];
        else if (s == "--config" && a + 1 < argc) cfgfiles.push_back(argv[++a]);
        else if (s.rfind("--", 0) == 0) { /* Kokkos flags */ }
        else overrides.push_back(s);
      }
      spf::Config cfg = spf::Config::from_file(SPF_DEFAULT_CONFIG);
      for (const auto& f : cfgfiles) cfg.merge_file(f);
      for (const auto& s : overrides) cfg.set_from_string(s);
      const std::string src = SPF_SOURCE_DIR;
      resolve_eos_paths(cfg, src);
      if (cfg.get<std::string>("ic.generator") == "smash_cascade" || cfg.get<std::string>("ic.generator") == "oscar_file") {
        std::string p = cfg.get<std::string>("ic.oscar_path");
        if (!p.empty() && p[0] != '/') p = std::filesystem::absolute(p).string();
        cfg.set_from_string("ic.oscar_path=" + nlohmann::json(p).dump());
      }
      // the explicit cell counts that follow from grid.dx_fm and the box lengths go into the echo
      const spf::GridCells gc = spf::grid_cells(cfg);
      cfg.set_from_string("grid.nx=" + std::to_string(gc.n[0]));
      cfg.set_from_string("grid.ny=" + std::to_string(gc.n[1]));
      cfg.set_from_string("grid.nz=" + std::to_string(gc.n[2]));
      cfg.echo(out);
      // event list
      std::vector<int> ids;
      const int first = cfg.get<int>("run.event_first");
      int n = cfg.get<int>("run.n_events");
      const std::string gen = cfg.get<std::string>("ic.generator");
      if (gen == "smash_cascade" || gen == "oscar_file") {
        const spf::OscarFile f = spf::read_oscar2013(cfg.get<std::string>("ic.oscar_path"));
        std::set<int> all;
        for (const auto& b : f.blocks) if (b.kind == "out") all.insert(b.event);
        std::vector<int> v(all.begin(), all.end());
        for (int i = first; i < static_cast<int>(v.size()) && (n < 0 || i < first + n); ++i) ids.push_back(v[i]);
      } else {
        if (n < 0) throw spf::ConfigError("run.n_events = -1 is only valid for OSCAR input");
        for (int i = first; i < first + n; ++i) ids.push_back(i);
      }
      std::printf("spf_collide: %zu events, %d x %d x %d cells of %.3f fm, eos %s, out %s\n", ids.size(), gc.n[0], gc.n[1], gc.n[2], gc.dx,
                  cfg.get<std::string>("eos.model").c_str(), out.c_str());
      spf::with_eos(cfg, src, [&](const auto& eos) { rc = run_all(cfg, eos, out, ids); });
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      rc = 1;
    }
  }
  Kokkos::finalize();
  return rc;
}
