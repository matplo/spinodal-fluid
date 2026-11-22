// Host side of the observables (WP6): parameters, clump finder, histograms, JSON output, snapshot reader, CSV format.
#include "spf/observables.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>

namespace spf {

ObsParams ObsParams::from_config(const Config& cfg) {
  ObsParams p;
  const std::string md = cfg.get<std::string>("obs.moment_density");
  if (md == "computational_frame") p.rest_frame = false;
  else if (md == "rest_frame") p.rest_frame = true;
  else throw ConfigError("obs.moment_density must be 'computational_frame' or 'rest_frame', got '" + md + "'");
  p.connectivity = cfg.get<int>("obs.clump_connectivity");
  if (p.connectivity != 6 && p.connectivity != 26) throw ConfigError("obs.clump_connectivity must be 6 or 26");
  p.periodic = cfg.get<std::string>("grid.boundary") == "periodic";
  p.rho_s = cfg.get<Real>("constants.rho_s_fm3");
  p.eps_s = cfg.get<Real>("constants.m_N_GeV") * p.rho_s;
  p.clump_rho_min_rho_s = cfg.get<Real>("obs.clump_rho_min_rho_s");
  p.clump_hist_max = cfg.get<int>("obs.clump_hist_max");
  p.fluid_end_eps_eps_s = cfg.get<Real>("obs.fluid_end_eps_eps_s");
  p.every_steps = cfg.get<long>("obs.every_steps");
  if (p.every_steps < 1) throw ConfigError("obs.every_steps must be >= 1");
  const Real rmax = cfg.get<Real>("obs.phase_rho_max_rho_s"), emax = cfg.get<Real>("obs.phase_eps_max_eps_s");
  p.d_rho = cfg.get<Real>("obs.phase_rho_step_rho_s");
  p.d_eps = cfg.get<Real>("obs.phase_eps_step_eps_s");
  if (!(p.d_rho > 0.0 && p.d_eps > 0.0 && rmax > 0.0 && emax > 0.0)) throw ConfigError("obs.phase_* ranges and steps must be positive");
  p.n_rho = static_cast<int>(std::lround(rmax / p.d_rho));
  p.n_eps = static_cast<int>(std::lround(emax / p.d_eps));
  p.hist_times = cfg.get<std::vector<Real>>("obs.hist_times_fm");
  return p;
}

double PhaseHist::total() const {
  double s = outside;
  for (double b : baryon) s += b;
  return s;
}

namespace {
struct UnionFind {
  std::vector<long> parent;
  explicit UnionFind(size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), 0L); }
  long find(long a) {
    while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
    return a;
  }
  void unite(long a, long b) {
    a = find(a); b = find(b);
    if (a != b) parent[std::max(a, b)] = std::min(a, b);
  }
};
}  // namespace

ClumpResult find_clumps(const ObsFields& f, const ObsParams& prm) {
  const Grid& g = f.grid;
  const int nx = g.n[0], ny = g.n[1], nz = g.n[2];
  const Field3& Dsrc = prm.rest_frame ? f.rho : f.R;
  auto hD = Kokkos::create_mirror_view(Dsrc);
  auto hR = Kokkos::create_mirror_view(f.R);
  Kokkos::deep_copy(hD, Dsrc);
  Kokkos::deep_copy(hR, f.R);
  const Real thr = prm.clump_rho_min_rho_s * prm.rho_s;
  const size_t n = static_cast<size_t>(nx) * ny * nz;
  auto id = [&](int i, int j, int k) { return (static_cast<long>(i) * ny + j) * nz + k; };
  std::vector<char> above(n, 0);
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k) above[id(i, j, k)] = hD(g.g[0] + i, g.g[1] + j, g.g[2] + k) > thr ? 1 : 0;
  // forward half of the neighbour set: 3 offsets (6-connectivity) or 13 (26-connectivity)
  std::vector<std::array<int, 3>> off;
  for (int a = -1; a <= 1; ++a)
    for (int b = -1; b <= 1; ++b)
      for (int c = -1; c <= 1; ++c) {
        const int order = std::abs(a) + std::abs(b) + std::abs(c);
        if (order == 0) continue;
        if (prm.connectivity == 6 && order != 1) continue;
        if (a < 0 || (a == 0 && b < 0) || (a == 0 && b == 0 && c < 0)) continue;  // keep one of each pair
        off.push_back({a, b, c});
      }
  UnionFind uf(n);
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k) {
        if (!above[id(i, j, k)]) continue;
        for (const auto& o : off) {
          int a = i + o[0], b = j + o[1], c = k + o[2];
          if (prm.periodic) {
            a = (a + nx) % nx; b = (b + ny) % ny; c = (c + nz) % nz;
          } else if (a < 0 || a >= nx || b < 0 || b >= ny || c < 0 || c >= nz) {
            continue;
          }
          if (above[id(a, b, c)]) uf.unite(id(i, j, k), id(a, b, c));
        }
      }
  const double dV = g.dx * g.dx * g.dx;
  std::vector<long> label(n, -1);
  ClumpResult out;
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k) {
        const long q = id(i, j, k);
        if (!above[q]) continue;
        const long root = uf.find(q);
        if (label[root] < 0) {
          label[root] = static_cast<long>(out.baryon.size());
          out.baryon.push_back(0.0);
          out.cells.push_back(0);
        }
        out.baryon[label[root]] += hR(g.g[0] + i, g.g[1] + j, g.g[2] + k) * dV;
        out.cells[label[root]] += 1;
      }
  return out;
}

std::vector<long> clump_histogram(const std::vector<double>& baryon, int max_bin, long& overflow) {
  std::vector<long> h(static_cast<size_t>(max_bin) + 1, 0);
  overflow = 0;
  for (double b : baryon) {
    const long k = std::lround(b);
    if (k > max_bin) ++overflow;
    else h[static_cast<size_t>(std::max(0L, k))] += 1;
  }
  return h;
}

nlohmann::json clump_json(const ClumpResult& c, const ObsParams& prm, Real time, long step) {
  long over = 0;
  nlohmann::json j;
  j["time_fm"] = time;
  j["step"] = step;
  j["threshold_rho_s"] = prm.clump_rho_min_rho_s;
  j["density"] = prm.rest_frame ? "rest_frame" : "computational_frame";
  j["connectivity"] = prm.connectivity;
  j["periodic_wrap"] = prm.periodic;
  j["n_clumps"] = c.baryon.size();
  j["bin_definition"] = "bin k counts clumps with round(B) = k; bin 0 includes B < 0.5";
  j["histogram"] = clump_histogram(c.baryon, prm.clump_hist_max, over);
  j["overflow"] = over;
  j["baryon_per_clump"] = c.baryon;
  j["cells_per_clump"] = c.cells;
  return j;
}

nlohmann::json phase_json(const PhaseHist& p, const ObsParams& prm, Real time, long step, Real unstable_fraction) {
  nlohmann::json j;
  j["time_fm"] = time;
  j["step"] = step;
  j["rho_s_fm3"] = prm.rho_s;
  j["eps_s_GeV_fm3"] = prm.eps_s;
  j["n_rho"] = p.n_rho;
  j["n_eps"] = p.n_eps;
  j["d_rho_over_rho_s"] = p.d_rho;
  j["d_eps_over_eps_s"] = p.d_eps;
  j["layout"] = "baryon[ir * n_eps + ie]: baryon number (weights R dV) with rho/rho_s in [ir d_rho, (ir+1) d_rho), eps/eps_s in [ie d_eps, (ie+1) d_eps)";
  j["baryon"] = p.baryon;
  j["outside"] = p.outside;
  j["total"] = p.total();
  j["unstable_fraction"] = unstable_fraction;
  return j;
}

LoadedSnapshot load_snapshot_fields(const std::string& base_in) {
  std::string base = base_in;
  for (const char* suf : {".json", ".bin"}) {
    const std::string s = suf;
    if (base.size() > s.size() && base.compare(base.size() - s.size(), s.size(), s) == 0) base.resize(base.size() - s.size());
  }
  std::ifstream js(base + ".json");
  if (!js) throw std::runtime_error("cannot open " + base + ".json");
  const nlohmann::json side = nlohmann::json::parse(js);
  const int nx = side.at("shape")[0].get<int>(), ny = side.at("shape")[1].get<int>(), nz = side.at("shape")[2].get<int>();
  const size_t n = static_cast<size_t>(nx) * ny * nz;
  std::ifstream bin(base + ".bin", std::ios::binary);
  if (!bin) throw std::runtime_error("cannot open " + base + ".bin");
  LoadedSnapshot s;
  s.time = side.at("time_fm").get<Real>();
  s.step = side.at("step").get<long>();
  s.config = side.at("config");
  s.fields.grid = Grid::make_3d(nx, ny, nz, side.at("dx_fm").get<Real>());
  const Grid& g = s.fields.grid;
  Field3* targets[5] = {&s.fields.R, &s.fields.rho, &s.fields.eps, &s.fields.vx, &s.fields.vy};
  const char* names[5] = {"R", "rho", "eps", "vx", "vy"};
  for (int q = 0; q < 5; ++q) {
    bool found = false;
    for (const auto& fl : side.at("fields")) {
      if (fl.at("name").get<std::string>() != names[q]) continue;
      found = true;
      std::vector<double> v(n);
      bin.seekg(static_cast<std::streamoff>(fl.at("byte_offset").get<size_t>()));
      bin.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(double)));
      if (!bin) throw std::runtime_error("short read of field '" + std::string(names[q]) + "' in " + base + ".bin");
      *targets[q] = g.make_field(names[q]);
      auto h = Kokkos::create_mirror_view(*targets[q]);
      Kokkos::deep_copy(h, 0.0);
      size_t p = 0;
      for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
          for (int k = 0; k < nz; ++k) h(g.g[0] + i, g.g[1] + j, g.g[2] + k) = v[p++];
      Kokkos::deep_copy(*targets[q], h);
    }
    if (!found) throw std::runtime_error(base + ".json: snapshot has no field '" + names[q] + "' (needed: R rho eps vx vy)");
  }
  return s;
}

std::string obs_csv_header() { return "t_fm,A,mom2,mom3,mom4,mom5,mom6,mom7,U_perp,unstable_fraction,eps_max_eps_s,step"; }

std::string obs_csv_line(const ObsResult& r) {
  char buf[512];
  std::snprintf(buf, sizeof buf, "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%ld", r.time, r.A, r.moment[2],
                r.moment[3], r.moment[4], r.moment[5], r.moment[6], r.moment[7], r.u_perp, r.unstable_fraction, r.eps_max_eps_s, r.step);
  return buf;
}

}  // namespace spf
