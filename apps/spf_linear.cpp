// Linear response along a phase trajectory (WP8, arXiv:1007.1448).
//   spf_linear [--override file.json] [key=value ...]
// Input  linear.trajectory_csv : columns t [fm/c], rho/rho_s, T [MeV]   ('#' lines and one header line are skipped)
// Optional linear.thermo_csv   : one row per trajectory row, columns t, h [GeV/fm^3], vT2, vs2, cv [fm^-3], n [fm^-3]
//                                (thermodynamics from an external equation of state); default: the analytic toy van der Waals EoS.
// Parameters: gradient.a_fm, linear.eta0, linear.kappa0, k grid linear.k_min_fm / k_max_fm / nk, linear.sigma_eq2.
// Output in output.dir: gamma_k_t.csv (t, k, gamma_k, Gamma_k), G_k.csv (k, G_k, ln_G_k, t_unstable, Gamma_end, sigma2_end), config_used.json.
#include "spf/config.hpp"
#include "spf/linear/amplification.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::vector<std::vector<double>> read_rows(const std::string& path, std::size_t ncol) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<std::vector<double>> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    for (char& c : line) if (c == ',' || c == ';' || c == '\t') c = ' ';
    std::istringstream ss(line);
    std::vector<double> v;
    double x;
    while (ss >> x) v.push_back(x);
    if (v.size() < ncol) continue;  // header line
    rows.push_back(v);
  }
  return rows;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    spf::Config cfg = spf::Config::from_file(SPF_DEFAULT_CONFIG);
    for (int a = 1; a < argc; ++a) {
      const std::string s = argv[a];
      if (s == "--override" && a + 1 < argc) cfg.merge_file(argv[++a]);
      else if (s.find('=') != std::string::npos && s.rfind("--", 0) != 0) cfg.set_from_string(s);
    }
    namespace L = spf::linear;
    const std::string out = cfg.get<std::string>("output.dir");
    std::filesystem::create_directories(out);
    cfg.echo(out);
    const std::string trajfile = cfg.get<std::string>("linear.trajectory_csv");
    if (trajfile.empty()) { std::fprintf(stderr, "spf_linear: set linear.trajectory_csv=<file with t, rho/rho_s, T_MeV>\n"); return 2; }
    const double rho_s = cfg.get<double>("constants.rho_s_fm3"), m_N = cfg.get<double>("constants.m_N_GeV");
    std::vector<L::TrajectoryPoint> traj;
    for (const auto& r : read_rows(trajfile, 3)) traj.push_back({r[0], r[1] * rho_s, r[2] * 1e-3});
    if (traj.size() < 2) throw std::runtime_error("trajectory needs at least two rows");

    L::Transport tr;
    tr.eta0 = cfg.get<double>("linear.eta0");
    tr.kappa0 = cfg.get<double>("linear.kappa0");
    tr.c0 = L::c0_constant(cfg.get<double>("linear.g_gluon"), cfg.get<double>("linear.g_quark"));
    const double C = L::gradient_C(cfg.get<double>("gradient.a_fm"), m_N, rho_s);

    L::ThermoProvider thermo;
    const std::string thermofile = cfg.get<std::string>("linear.thermo_csv");
    std::vector<std::vector<double>> trows;
    if (!thermofile.empty()) {
      trows = read_rows(thermofile, 6);
      if (trows.size() != traj.size()) throw std::runtime_error("thermo_csv must have one row per trajectory row");
      // lookup by nearest trajectory index: the provider is called in trajectory order with the stored (rho, T)
      thermo = [&](double rho, double T) {
        for (std::size_t i = 0; i < traj.size(); ++i)
          if (traj[i].rho == rho && traj[i].T == T) {
            L::ThermoPoint p;
            p.rho = rho; p.T = T; p.h = trows[i][1]; p.vT2 = trows[i][2]; p.vs2 = trows[i][3]; p.cv = trows[i][4]; p.n = trows[i][5];
            return p;
          }
        throw std::runtime_error("thermo lookup failed");
      };
    } else {
      static const L::ToyVdw toy(cfg.get<double>("linear.toy_rho_c_rho_s") * rho_s, cfg.get<double>("linear.toy_T_c_MeV") * 1e-3,
                                 cfg.get<double>("linear.toy_mass_GeV"), cfg.get<double>("linear.toy_cv0"));
      thermo = [](double rho, double T) { return toy.at(rho, T); };
    }

    const double kmin = cfg.get<double>("linear.k_min_fm"), kmax = cfg.get<double>("linear.k_max_fm");
    const int nk = cfg.get<int>("linear.nk");
    const double sig2 = cfg.get<double>("linear.sigma_eq2");
    std::vector<double> t(traj.size());
    for (std::size_t i = 0; i < traj.size(); ++i) t[i] = traj[i].t;

    std::ofstream fg(out + "/gamma_k_t.csv"), fG(out + "/G_k.csv");
    fg << "t_fm,k_fm,gamma_k_per_fm,Gamma_k\n";
    fG << "k_fm,G_k,ln_G_k,t_unstable_fm,Gamma_end,sigma2_end\n";
    fg.precision(10); fG.precision(10);
    for (int ik = 0; ik < nk; ++ik) {
      const double k = nk > 1 ? kmin + (kmax - kmin) * ik / (nk - 1) : kmin;
      const auto g = L::gamma_history(thermo, traj, C, tr, k);
      const auto Gam = L::cumulative_gamma(t, g);
      for (std::size_t i = 0; i < t.size(); ++i) fg << t[i] << ',' << k << ',' << g[i] << ',' << Gam[i] << '\n';
      const double lnG = L::log_amplification(t, g);
      const auto sig = L::lalime_solve(t, g, L::einstein_hold_diffusion(sig2), sig2);
      fG << k << ',' << std::exp(lnG) << ',' << lnG << ',' << L::unstable_time(t, g) << ',' << Gam.back() << ',' << sig.back() << '\n';
    }
    std::printf("spf_linear: %zu trajectory points, %d modes -> %s\n", traj.size(), nk, out.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "spf_linear: %s\n", e.what());
    return 1;
  }
  return 0;
}
