// WP5: initial state for the fluid stage.
//   spf_ic [--out DIR] [--config FILE] [--evolve T_fm] [key=value ...]
// ic.generator = smash_cascade | oscar_file | glauber_gaussians | analytic (see docs/initial_state.md). Deposits the hadrons on the
// grid, applies the E >= |M| limiter, hands the conserved fields to Hydro3D, recovers the primitives with the equation of state of
// eos.model, writes the Hydro3D snapshot <DIR>/ic_state.{bin,json} and the summary <DIR>/ic_summary.json.
#include "spf/eos/dispatch.hpp"
#include "spf/hydro3d.hpp"
#include "spf/initial_state.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

template <class EoS>
nlohmann::json run(const spf::Config& cfg, const EoS& eos, const spf::Grid& grid, const spf::InitialState& st, const std::string& out, double evolve_fm) {
  using spf::Real;
  const spf::Hydro3DParams hp = spf::Hydro3DParams::from_config(cfg);
  spf::Hydro3D<EoS> h(eos, hp);
  h.set_conserved(st.dep.R, st.dep.Mx, st.dep.My, st.dep.Mz, st.dep.E);
  const long fail = h.update_primitives();
  const std::vector<std::string> fields = {"R", "Mx", "My", "Mz", "E", "eps", "rho", "vx", "vy", "vz", "p"};
  h.write_snapshot(out + "/ic_state", fields, cfg.json());
  const auto eps = h.interior_copy(h.eps), rho = h.interior_copy(h.rho);
  const int nx = grid.n[0], ny = grid.n[1], nz = grid.n[2];
  const auto o = spf::centred_origin(grid);
  const Real rho_s = cfg.get<Real>("constants.rho_s_fm3"), eps_s = cfg.get<Real>("constants.m_N_GeV") * rho_s;
  double sc_eps = 0, sc_rho = 0, mx_eps = 0, mx_rho = 0;
  long nc = 0;
  size_t q = 0;
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k, ++q) {
        const double x = o[0] + i * grid.dx, y = o[1] + j * grid.dx, z = o[2] + k * grid.dx;
        mx_eps = std::fmax(mx_eps, eps[q]);
        mx_rho = std::fmax(mx_rho, rho[q]);
        if (x * x + y * y + z * z < 1.0) { sc_eps += eps[q]; sc_rho += rho[q]; ++nc; }
      }
  nlohmann::json j;
  j["eos"] = spf::eos_model_name(spf::eos_model_from_config(cfg));
  j["recovery_failures"] = fail;
  j["central_cells"] = nc;
  j["central_rho_fm3"] = nc ? sc_rho / nc : 0.0;
  j["central_eps_GeV_fm3"] = nc ? sc_eps / nc : 0.0;
  j["central_rho_over_rho_s"] = nc ? sc_rho / nc / rho_s : 0.0;
  j["central_eps_over_eps_s"] = nc ? sc_eps / nc / eps_s : 0.0;
  j["max_rho_over_rho_s"] = mx_rho / rho_s;
  j["max_eps_over_eps_s"] = mx_eps / eps_s;
  if (evolve_fm > 0.0) {
    // T5.4: run the fluid for evolve_fm fm/c; failures and limiter activations are counted over all sweeps and stages
    const long n0 = h.n_recover_fail_total, c0 = h.n_clamped_total;
    const long nsteps = static_cast<long>(std::ceil(evolve_fm / h.dt() - 1e-9));
    const auto t0 = h.totals();
    h.run(nsteps);
    const auto t1 = h.totals();
    h.write_snapshot(out + "/ic_evolved", fields, cfg.json());
    const Real v = grid.dx * grid.dx * grid.dx;
    const long fail_final = h.update_primitives();
    j["evolve"] = {{"requested_fm", evolve_fm}, {"steps", nsteps}, {"time_fm", h.time},
                   {"recovery_failures_during_run", h.n_recover_fail_total - n0 - fail_final},
                   {"recovery_failures_final_state", fail_final},
                   {"limiter_activations_during_run", h.n_clamped_total - c0},
                   {"E_over_E0", t1.E / t0.E}, {"R_over_R0", t1.R / t0.R}, {"R_end_x_dx3", t1.R * v}, {"E_end_x_dx3", t1.E * v}};
    if (h.n_recover_fail_total - n0 > 0) {
      // location of failing cells in the final state is not tracked per cell; report extremes of the final primitives
      const auto e2 = h.interior_copy(h.eps), r2 = h.interior_copy(h.rho);
      double me = 0, mr = 0;
      for (size_t i = 0; i < e2.size(); ++i) { me = std::fmax(me, e2[i]); mr = std::fmax(mr, r2[i]); }
      j["evolve"]["final_max_eps_over_eps_s"] = me / eps_s;
      j["evolve"]["final_max_rho_over_rho_s"] = mr / rho_s;
    }
  }
  return j;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    try {
      std::string out = "ic_out", cfgfile;
      double evolve_fm = 0.0;
      std::vector<std::string> overrides;
      for (int a = 1; a < argc; ++a) {
        const std::string s = argv[a];
        if (s == "--out" && a + 1 < argc) out = argv[++a];
        else if (s == "--config" && a + 1 < argc) cfgfile = argv[++a];
        else if (s == "--evolve" && a + 1 < argc) evolve_fm = std::stod(argv[++a]);
        else if (s.rfind("--", 0) == 0) { /* Kokkos flags */ }
        else overrides.push_back(s);
      }
      spf::Config cfg = spf::Config::from_file(SPF_DEFAULT_CONFIG);
      if (!cfgfile.empty()) cfg.merge_file(cfgfile);
      for (const auto& s : overrides) cfg.set_from_string(s);
      cfg.echo(out);
      const spf::Grid grid = spf::Grid::make_3d(cfg.get<int>("grid.nx"), cfg.get<int>("grid.ny"), cfg.get<int>("grid.nz"), cfg.get<double>("grid.dx_fm"));
      const spf::InitialState st = spf::build_initial_state(cfg, grid);
      nlohmann::json j;
      spf::with_eos(cfg, SPF_SOURCE_DIR, [&](const auto& eos) { j = run(cfg, eos, grid, st, out, evolve_fm); });
      j["generator"] = st.generator;
      j["e_lab_gev"] = st.kin.e_lab;
      j["gamma_cm"] = st.kin.gamma_cm;
      j["sqrt_snn_gev"] = st.kin.sqrt_snn;
      j["t_init_radius_fm"] = st.radius_fm;
      j["t_init_fm"] = st.t_init_fm;
      j["oscar_time_fm"] = st.oscar_time_fm;
      j["n_hadrons"] = st.n_hadrons;
      j["sigma_init_fm"] = cfg.get<double>("ic.sigma_init_fm");
      j["gaussian_contraction"] = cfg.get<std::string>("ic.gaussian_contraction");
      const auto& d = st.dep;
      j["particles"] = {{"B", d.particles.B}, {"E", d.particles.E}, {"Px", d.particles.Px}, {"Py", d.particles.Py}, {"Pz", d.particles.Pz}};
      j["grid_sums"] = {{"B", d.grid.B}, {"E", d.grid.E}, {"Px", d.grid.Px}, {"Py", d.grid.Py}, {"Pz", d.grid.Pz}};
      j["outside_fraction_B"] = d.outside_fraction_B;
      j["outside_fraction_E"] = d.outside_fraction_E;
      j["n_particles_clipped"] = d.n_particles_clipped;
      j["n_particles_empty"] = d.n_particles_empty;
      j["limiter_cells"] = st.limiter_cells;
      j["baryon_total"] = st.baryon_total;
      j["baryon_abs_y_lt_0p5"] = st.baryon_y_window;
      j["grid"] = {{"nx", grid.n[0]}, {"ny", grid.n[1]}, {"nz", grid.n[2]}, {"dx_fm", grid.dx}};
      std::ofstream(out + "/ic_summary.json") << j.dump(2) << "\n";
      std::printf("%s\n", j.dump(2).c_str());
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      rc = 1;
    }
  }
  Kokkos::finalize();
  return rc;
}
