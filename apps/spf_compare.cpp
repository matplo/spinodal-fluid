// Backend comparison of the T3.2 state (WP3g, G2).
//   spf_compare write <basename> [steps=100]   run the T3.2 smooth random 48^3 state with default parameters and write a snapshot
//                                              (R Mx My Mz E eps rho vx vy vz p) plus <basename>_totals.json
//   spf_compare diff <baseA> <baseB> [E_min=1e-6]   per-field deviation between two snapshots on cells with E > E_min:
//                                              max |a-b| / max(|a|,|b|) per cell, and max |a-b| / max|field|; totals deviation
#include "spf/eos/eos.hpp"
#include "spf/hydro3d.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::vector<std::vector<double>> read_snapshot(const std::string& base, nlohmann::json& side) {
  std::ifstream js(base + ".json");
  if (!js) throw std::runtime_error("cannot open " + base + ".json");
  side = nlohmann::json::parse(js);
  const size_t n = static_cast<size_t>(side.at("shape")[0].get<int>()) * side.at("shape")[1].get<int>() * side.at("shape")[2].get<int>();
  std::ifstream bin(base + ".bin", std::ios::binary);
  std::vector<std::vector<double>> out;
  for (const auto& f : side.at("fields")) {
    std::vector<double> v(n);
    bin.seekg(static_cast<std::streamoff>(f.at("byte_offset").get<size_t>()));
    bin.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(double)));
    out.push_back(std::move(v));
  }
  return out;
}

int write_mode(const std::string& base, long steps) {
  const int n = 48;
  spf::Config cfg = spf::Config::from_file(SPF_DEFAULT_CONFIG);
  spf::Hydro3DParams hp = spf::Hydro3DParams::from_config(cfg);
  hp.nx = hp.ny = hp.nz = n;
  spf::Hydro3D<spf::EosIdeal> h(spf::EosIdeal{}, hp);
  h.set_state(spf::smooth_random_state(n, n, n));
  h.run(steps);
  h.write_snapshot(base, {"R", "Mx", "My", "Mz", "E", "eps", "rho", "vx", "vy", "vz", "p"}, cfg.json());
  const spf::Totals t = h.totals();
  nlohmann::json j;
  j["backend"] = spf::ExecSpace::name();
  j["steps"] = steps;
  j["R"] = t.R; j["E"] = t.E; j["Mx"] = t.Mx; j["My"] = t.My; j["Mz"] = t.Mz;
  j["recover_failures"] = h.n_recover_fail_total;
  std::ofstream(base + "_totals.json") << j.dump(2) << "\n";
  std::printf("wrote %s (%s, %ld steps), recovery failures %ld\n", base.c_str(), spf::ExecSpace::name(), steps, h.n_recover_fail_total);
  return 0;
}

int diff_mode(const std::string& a, const std::string& b, double e_min) {
  nlohmann::json sa, sb;
  const auto A = read_snapshot(a, sa);
  const auto B = read_snapshot(b, sb);
  if (A.size() != B.size() || sa.at("shape") != sb.at("shape")) throw std::runtime_error("snapshots differ in layout");
  size_t iE = 0;
  for (size_t f = 0; f < A.size(); ++f)
    if (sa.at("fields")[f].at("name") == "E") iE = f;
  size_t ncell = 0;
  for (size_t i = 0; i < A[iE].size(); ++i)
    if (A[iE][i] > e_min && B[iE][i] > e_min) ++ncell;
  std::printf("cells with E > %.1e in both snapshots: %zu of %zu\n", e_min, ncell, A[iE].size());
  double worst = 0.0;
  for (size_t f = 0; f < A.size(); ++f) {
    double rel_cell = 0.0, rel_norm = 0.0, fmax = 0.0;
    for (size_t i = 0; i < A[f].size(); ++i) {
      if (!(A[iE][i] > e_min) || !(B[iE][i] > e_min)) continue;
      const double d = std::fabs(A[f][i] - B[f][i]), m = std::fmax(std::fabs(A[f][i]), std::fabs(B[f][i]));
      if (m > 0.0) rel_cell = std::fmax(rel_cell, d / m);
      rel_norm = std::fmax(rel_norm, d);
      fmax = std::fmax(fmax, std::fabs(A[f][i]));
    }
    rel_norm /= fmax;
    std::printf("field %-4s max per-cell relative deviation %.3e   max |a-b| / max|field| %.3e\n", sa.at("fields")[f].at("name").get<std::string>().c_str(),
                rel_cell, rel_norm);
    worst = std::fmax(worst, rel_cell);
  }
  std::ifstream ta(a + "_totals.json"), tb(b + "_totals.json");
  const nlohmann::json ja = nlohmann::json::parse(ta), jb = nlohmann::json::parse(tb);
  double wt = 0.0;
  for (const char* k : {"R", "E"}) {
    const double d = std::fabs(ja.at(k).get<double>() - jb.at(k).get<double>()) / std::fabs(ja.at(k).get<double>());
    std::printf("total %-2s relative deviation %.3e\n", k, d);
    wt = std::fmax(wt, d);
  }
  const double Et = ja.at("E").get<double>();
  for (const char* k : {"Mx", "My", "Mz"}) {
    const double d = std::fabs(ja.at(k).get<double>() - jb.at(k).get<double>()) / Et;
    std::printf("total %-2s deviation relative to total E %.3e\n", k, d);
    wt = std::fmax(wt, d);
  }
  std::printf("worst per-cell relative deviation %.3e, worst totals deviation %.3e -> %s (limit 1e-11)\n", worst, wt,
              (worst < 1e-11 && wt < 1e-11) ? "PASS" : "FAIL");
  return (worst < 1e-11 && wt < 1e-11) ? 0 : 3;
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  int rc = 0;
  {
    try {
      const std::string mode = argc > 1 ? argv[1] : "";
      if (mode == "write" && argc >= 3) rc = write_mode(argv[2], argc > 3 ? std::atol(argv[3]) : 100);
      else if (mode == "diff" && argc >= 4) rc = diff_mode(argv[2], argv[3], argc > 4 ? std::atof(argv[4]) : 1e-6);
      else { std::fprintf(stderr, "usage: spf_compare write <base> [steps] | diff <baseA> <baseB> [E_min]\n"); rc = 2; }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "error: %s\n", e.what());
      rc = 1;
    }
  }
  Kokkos::finalize();
  return rc;
}
