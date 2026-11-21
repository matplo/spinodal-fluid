// Spinodal growth rates and stable-side oscillation frequencies of the finite-range fluid (WP4).
//   spf_growth [key=value ...]     eos.model = cold_thermal_two_phase | cold_thermal_maxwell
// Periodic box elongated along x (L = growth.L_fm, ny = nz = 4), uniform background rho0 = growth.rho0_rho_s * rho_s,
// eps0 = eps_c(rho0) + growth.eth_eps_s * eps_s, isentropic harmonic perturbation of relative amplitude growth.amp_rel, no flow.
// One run per mode m with k = 2 pi m / L, k from growth.k_min_fm to growth.kdx_max / dx. The scan runs over growth.a_list and
// growth.dx_list (comma-separated; empty: gradient.a_fm and grid.dx_fm). Output CSV: one row per mode; series CSV: A(t) of the modes
// listed in growth.series_modes. Fits and analytic values: docs/equations.md, Sect. 10.
#include "spf/eos/eos.hpp"
#include "spf/growth.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<double> parse_list(const std::string& s, double fallback) {
  std::vector<double> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) if (!tok.empty()) v.push_back(std::stod(tok));
  if (v.empty()) v.push_back(fallback);
  return v;
}

template <class EoS>
int run(const EoS& eos, const spf::EosPoint& bg, const spf::Config& cfg) {
  using spf::Real;
  spf::Hydro3DParams hp0 = spf::Hydro3DParams::from_config(cfg);
  spf::GrowthSetup s;
  s.bg = bg;
  s.amp_rel = cfg.get<Real>("growth.amp_rel");
  s.paper = cfg.get<bool>("growth.paper_perturbation");
  s.rho_s = hp0.gradient.rho_s;
  s.eps_s = hp0.gradient.eps_s();
  s.t_max = cfg.get<Real>("growth.t_max_fm");
  s.stop_factor = cfg.get<Real>("growth.stop_factor");
  s.stable_periods = cfg.get<Real>("growth.stable_periods");
  const Real L = cfg.get<Real>("growth.L_fm"), kmin = cfg.get<Real>("growth.k_min_fm"), kdx_max = cfg.get<Real>("growth.kdx_max");
  const Real fit_lo = cfg.get<Real>("growth.fit_lo"), fit_hi = cfg.get<Real>("growth.fit_hi");
  const std::vector<double> alist = parse_list(cfg.get<std::string>("growth.a_list"), hp0.gradient.a_fm);
  const std::vector<double> dxlist = parse_list(cfg.get<std::string>("growth.dx_list"), hp0.dx);
  std::vector<int> series_modes;
  for (double d : parse_list(cfg.get<std::string>("growth.series_modes"), -1.0)) if (d > 0) series_modes.push_back(static_cast<int>(d));
  const std::string dir = cfg.get<std::string>("output.dir");
  std::ofstream csv(dir + "/" + cfg.get<std::string>("growth.output_csv"));
  std::ofstream ser(dir + "/" + cfg.get<std::string>("growth.series_csv"));
  csv.precision(10);
  ser.precision(10);
  csv << "dx_fm,a_fm,m,k_fm,L_fm,kdx,kind,gamma_fit,omega_fit,omega_damped_fit,damping_fit,analytic_gamma,analytic_omega2,"
         "ratio,omega2_ratio,A0_fit_over_A0,fit_resid,n_fit,steps,recover_failures,min_pgrad_over_p0\n";
  ser << "dx_fm,a_fm,m,k_fm,t_fm,A_cos,A_abs\n";
  std::printf("background: eps = %.9f GeV/fm^3, rho = %.9f fm^-3, p0 = %.9f, h = %.9f, vs2 = %.9f\n", bg.eps, bg.rho, bg.p0, bg.h, bg.vs2);
  for (double dx : dxlist) {
    for (double a : alist) {
      spf::Hydro3DParams hp = hp0;
      hp.dx = dx;
      hp.gradient.a_fm = a;
      Real kmax_an = 0, gmax_an = 0;
      spf::gamma_max_analytic(hp.gradient, bg, kmax_an, gmax_an);
      std::printf("dx = %.3f a = %.3f: analytic gamma_max = %.6f at k = %.6f, k_cut = %.6f\n", dx, a, gmax_an, kmax_an,
                  spf::kcut_analytic(hp.gradient, bg));
      const Real kmax = kdx_max / dx;
      const int nx = static_cast<int>(std::lround(L / dx));
      for (int m = 1; 2.0 * M_PI * m / (nx * dx) <= kmax + 1e-12; ++m) {
        const Real k = 2.0 * M_PI * m / (nx * dx);
        if (k < kmin) continue;
        const spf::ModeSeries r = spf::run_mode(eos, hp, s, L, m);
        const Real w2 = spf::omega2_analytic(hp.gradient, bg, k);
        const bool unstable = w2 < 0.0;
        Real gam = std::nan(""), om = std::nan(""), omd = std::nan(""), dd = std::nan(""), a0r = std::nan(""), res = std::nan("");
        int nfit = 0;
        if (unstable) {
          const auto f = spf::fit::cosh_fit(r.t, r.a_abs, fit_lo, fit_hi);
          if (f.ok) { gam = f.rate; a0r = f.A0 / r.a_abs[0]; res = f.resid; nfit = f.n; }
        } else {
          const auto f = spf::fit::cos_fit(r.t, r.a_cos, false);
          const auto fd = spf::fit::cos_fit(r.t, r.a_cos, true);
          if (f.ok) { om = f.rate; a0r = f.A0 / r.a_cos[0]; res = f.resid; nfit = f.n; }
          if (fd.ok) { omd = fd.rate; dd = fd.damp; }
        }
        const Real an_g = unstable ? std::sqrt(-w2) : 0.0;
        const Real ratio = unstable ? gam / an_g : om / std::sqrt(w2);
        const Real w2ratio = unstable ? std::nan("") : om * om / w2;
        csv << dx << ',' << a << ',' << m << ',' << k << ',' << r.L << ',' << k * dx << ',' << (unstable ? "unstable" : "stable") << ',' << gam
            << ',' << om << ',' << omd << ',' << dd << ',' << an_g << ',' << w2 << ',' << ratio << ',' << w2ratio << ',' << a0r << ',' << res
            << ',' << nfit << ',' << r.steps << ',' << r.recover_fail << ',' << r.min_pgrad_ratio << '\n';
        std::printf("dx %.2f a %.3f m %3d k %8.5f %s: %s = %.6f analytic %.6f ratio %.4f resid %.2e  fail %ld  min(pg/p0) %.3e\n", dx, a, m, k,
                    unstable ? "unstable" : "stable  ", unstable ? "gamma" : "omega", unstable ? gam : om,
                    unstable ? an_g : std::sqrt(w2), ratio, res, r.recover_fail, r.min_pgrad_ratio);
        if (std::find(series_modes.begin(), series_modes.end(), m) != series_modes.end())
          for (size_t i = 0; i < r.t.size(); ++i)
            ser << dx << ',' << a << ',' << m << ',' << k << ',' << r.t[i] << ',' << r.a_cos[i] << ',' << r.a_abs[i] << '\n';
      }
    }
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    try {
      spf::Config cfg = spf::Config::from_file(SPF_DEFAULT_CONFIG);
      for (int a = 1; a < argc; ++a) {
        const std::string s = argv[a];
        if (s == "--override" && a + 1 < argc) cfg.merge_file(argv[++a]);
        else if (s.find('=') != std::string::npos && s.rfind("--", 0) != 0) cfg.set_from_string(s);
      }
      cfg.echo(cfg.get<std::string>("output.dir"));
      const std::string model = cfg.get<std::string>("eos.model");
      if (model != "cold_thermal_two_phase" && model != "cold_thermal_maxwell") {
        std::fprintf(stderr, "spf_growth: set eos.model=cold_thermal_two_phase or cold_thermal_maxwell (got '%s')\n", model.c_str());
        rc = 2;
      } else {
        const std::string path = std::string(SPF_SOURCE_DIR) + "/" + cfg.get<std::string>("eos.cold_curve_path");
        const spf::ColdCurveTable table = spf::ColdCurveTable::load(path);
        const spf::EosColdThermal eos(table, model == "cold_thermal_two_phase" ? spf::ColdVariant::two_phase : spf::ColdVariant::maxwell,
                                      cfg.get<spf::Real>("eos.gamma_th"));
        const spf::Real rho_s = cfg.get<spf::Real>("constants.rho_s_fm3"), eps_s = cfg.get<spf::Real>("constants.m_N_GeV") * rho_s;
        spf::Real rho0 = cfg.get<spf::Real>("growth.rho0_rho_s") * rho_s;
        spf::Real eps0;
        if (cfg.get<bool>("growth.paper_perturbation")) {
          rho0 = 6.0 * rho_s;
          eps0 = 10.0 * eps_s;
        } else {
          eps0 = spf::cold_energy_density(eos, rho0) + cfg.get<spf::Real>("growth.eth_eps_s") * eps_s;
        }
        const spf::EosPoint bg = spf::probe_eos(eos, eps0, rho0);
        rc = run(eos, bg, cfg);
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      rc = 1;
    }
  }
  Kokkos::finalize();
  return rc;
}
