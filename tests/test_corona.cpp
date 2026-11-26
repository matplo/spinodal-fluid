// WP10e: corona treatment (decision S31).  Tests C1 - C6.  Measured values are printed with [C<n>].
// The fluid runs go through the driver spf_collide and the sampler through spf_sample (the test tables of data/eos/test, Glauber-Gaussians
// Pb+Pb at 3 A GeV, box 24 fm).
#include <doctest/doctest.h>

#include "spf/config.hpp"
#include "spf/corona.hpp"
#include "spf/eos/eos.hpp"
#include "spf/initial_state.hpp"
#include "spf/particlisation/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using spf::Real;

namespace {

nlohmann::json read_json(const std::string& p) {
  std::ifstream in(p);
  REQUIRE_MESSAGE(in.good(), "cannot read ", p);
  return nlohmann::json::parse(in);
}
std::string slurp(const std::string& p) {
  std::ifstream in(p, std::ios::binary);
  std::stringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string table(const char* which) { return std::string(SPF_SOURCE_DIR) + "/data/eos/test/eos_" + which + "_test"; }

// Runs spf_collide for one event; returns the event directory.  extra: key=value assignments.
std::string run_collide(const std::string& tag, const std::string& model, double dx, const std::string& extra) {
  const std::string out = (fs::temp_directory_path() / ("spf_wp10e_" + tag)).string();
  fs::remove_all(out);
  fs::create_directories(out);
  const std::string tab = model == "table_maxwell" ? table("maxwell") : table("two_phase");
  char dxs[32];
  std::snprintf(dxs, sizeof dxs, "%.3f", dx);
  const std::string cmd = std::string("\"") + SPF_COLLIDE_EXE + "\" --out " + out + " --config " + SPF_SOURCE_DIR +
                          "/config/collisions/common.json ic.generator=glauber_gaussians ic.e_lab_gev=3 eos.model=" + model + " eos.table_path=\"" + tab +
                          "\" grid.dx_fm=" + dxs + " grid.Lx_fm=24 grid.Ly_fm=24 grid.Lz_fm=24 run.n_events=1 ic.max_outside_fraction=1e-3 " + extra +
                          " > " + out + "/driver.log 2>&1";
  REQUIRE(std::system(cmd.c_str()) == 0);
  return out + "/event_000";
}

double jv(const nlohmann::json& j, const char* a, const char* b) { return j.at(a).at(b).get<double>(); }

// in-process initial state of the event used by C1 / C2 (same settings as the driver run)
struct Event {
  spf::Config cfg;
  spf::Grid grid;
  spf::InitialState st;
  Event(double dx, const std::string& model) : cfg(spf::Config::from_file(SPF_DEFAULT_CONFIG)), grid(spf::Grid::make_3d(2, 2, 2, 1.0)) {
    cfg.merge_file(std::string(SPF_SOURCE_DIR) + "/config/collisions/common.json");
    for (const char* a : {"ic.generator=glauber_gaussians", "ic.e_lab_gev=3", "ic.max_outside_fraction=1e-3", "grid.Lx_fm=24", "grid.Ly_fm=24", "grid.Lz_fm=24"})
      cfg.set_from_string(a);
    cfg.set_from_string("grid.dx_fm=" + std::to_string(dx));
    cfg.set_from_string("eos.model=" + model);
    cfg.set_from_string("eos.table_path=" + nlohmann::json(model == "table_maxwell" ? table("maxwell") : table("two_phase")).dump());
    const spf::GridCells gc = spf::grid_cells(cfg);
    grid = spf::Grid::make_3d(gc.n[0], gc.n[1], gc.n[2], gc.dx);
    st = spf::build_initial_state(cfg, grid);
  }
};

spf::CoronaSplit split_with(Event& ev, const char* ref_table, const char* ref_model) {
  spf::Config rc = ev.cfg;
  rc.set_from_string(std::string("eos.model=") + ref_model);
  rc.set_from_string("eos.table_path=" + nlohmann::json(table(ref_table)).dump());
  const spf::EosTableData td = spf::EosTableData::from_config(rc);
  const spf::EosTable eos = td.eos(rc);
  const Real eps_sw = ev.cfg.get<Real>("particlisation.eps_sw_eps_s") * ev.cfg.get<Real>("constants.m_N_GeV") * ev.cfg.get<Real>("constants.rho_s_fm3");
  return spf::classify_corona(ev.cfg, ev.grid, ev.st, eos, eps_sw);
}

std::vector<long> corona_indices(const spf::CoronaSplit& s) {
  std::vector<long> v;
  for (size_t i = 0; i < s.is_core.size(); ++i) if (!s.is_core[i]) v.push_back(static_cast<long>(i));
  return v;
}

}  // namespace

// C1: the split is a partition: sums of E, P over core + corona equal the full list to 1e-12 relative, B and Q exactly (integers); the
// deposition of the core hadrons carries the core baryon number.
TEST_CASE("C1 core + corona partition: E, P to 1e-12, B and Q exact") {
  Event ev(0.5, "table_two_phase");
  const spf::CoronaSplit s = split_with(ev, "two_phase", "table_two_phase");
  const std::vector<char> all_core(ev.st.particles.size(), 1);
  const spf::CoronaTotals full = spf::corona_totals(ev.st.particles, all_core, true);
  const double dE = std::fabs(s.core.E + s.corona.E - full.E) / full.E;
  const double dPz = std::fabs(s.core.Pz + s.corona.Pz - full.Pz) / full.E;
  const long long dB = s.core.B + s.corona.B - full.B, dQ = s.core.Q + s.corona.Q - full.Q;
  CHECK(s.core.n + s.corona.n == full.n);
  CHECK(s.core.n > 0);
  CHECK(s.corona.n > 0);
  CHECK(dE <= 1e-12);
  CHECK(dPz <= 1e-12);
  CHECK(dB == 0);
  CHECK(dQ == 0);
  spf::InitialState core_st = ev.st;
  spf::redeposit_subset(ev.cfg, ev.grid, core_st, s.is_core);
  const double dBdep = std::fabs(core_st.dep.grid.B - static_cast<double>(s.core.B));
  const double dEdep = std::fabs(core_st.dep.grid.E - s.core.E) / s.core.E;
  std::printf("[C1] %ld hadrons: core %ld, corona %ld; |dE|/E %.2e, |dPz|/E %.2e, dB %lld, dQ %lld; deposited core B %.12g (core hadrons %lld), E relative deviation %.2e\n",
              full.n, s.core.n, s.corona.n, dE, dPz, dB, dQ, core_st.dep.grid.B, s.core.B, dEdep);
  CHECK(dBdep <= 1e-9);
  CHECK(dEdep <= 1e-9);
}

// C2: the split of an event does not depend on the equation of state of the run (the reference equation of state is fixed) and not on the
// backend (compared with the list of corona hadrons stored from the Serial backend, tests/data/corona_split_glauber_dx0.5.json).
TEST_CASE("C2 split identical for table_two_phase and table_maxwell runs, and equal to the Serial reference") {
  const std::string a = run_collide("c2a", "table_two_phase", 0.5, "particlisation.corona.reference_eos=table_two_phase");
  const std::string b = run_collide("c2b", "table_maxwell", 0.5,
                                    std::string("particlisation.corona.reference_eos=table_two_phase particlisation.corona.reference_table_path=\"") + table("two_phase") + "\"");
  const std::string fa = slurp(a + "/event_000.corona.oscar"), fb = slurp(b + "/event_000.corona.oscar");
  CHECK(!fa.empty());
  const bool same = fa == fb;
  const auto sa = read_json(a + "/event_000.surface.json"), sb = read_json(b + "/event_000.surface.json");
  CHECK(same);
  CHECK(sa["corona"]["n_corona"] == sb["corona"]["n_corona"]);
  Event ev(0.5, "table_two_phase");
  const spf::CoronaSplit s = split_with(ev, "two_phase", "table_two_phase");
  const std::vector<long> idx = corona_indices(s);
  if (const char* dump = std::getenv("SPF_CORONA_DUMP")) {
    nlohmann::json j = {{"corona_indices", idx}, {"n_particles", ev.st.particles.size()}, {"dx_fm", 0.5}, {"reference_eos", "table_two_phase (data/eos/test)"}};
    std::ofstream(dump) << j.dump() << "\n";
  }
  const auto ref = read_json(std::string(SPF_TEST_DATA_DIR) + "/corona_split_glauber_dx0.5.json");
  std::set<long> r(ref["corona_indices"].begin(), ref["corona_indices"].end()), m(idx.begin(), idx.end());
  long diff = 0;
  for (long i : r) if (!m.count(i)) ++diff;
  for (long i : m) if (!r.count(i)) ++diff;
  // distance of the hadrons to the threshold: how close the closest one is (rounding differences between backends matter only there)
  double closest = 1e300;
  const double eps_sw = ev.cfg.get<double>("constants.m_N_GeV") * ev.cfg.get<double>("constants.rho_s_fm3") * 4.0;
  for (double e : s.eps) closest = std::min(closest, std::fabs(e - eps_sw) / eps_sw);
  std::printf("[C2] corona files of the two-phase and Maxwell runs byte-identical: %d (%ld corona hadrons); in-process split vs Serial reference list: %ld differing hadrons; "
              "closest hadron to eps_sw: %.2e relative\n", static_cast<int>(same), static_cast<long>(idx.size()), diff, closest);
  CHECK(diff == 0);
}

// C3 and C4 share one chain: core_corona, table_two_phase, dx = 0.5 fm, sampler with corona append, 4 samples.
namespace {
struct Chain {
  std::string dir;
  nlohmann::json surf, samp;
  std::string oscar;
};
const Chain& chain() {
  static Chain c;
  static bool done = false;
  if (done) return c;
  c.dir = run_collide("c3", "table_two_phase", 0.5, "particlisation.corona.mode=core_corona");
  c.surf = read_json(c.dir + "/event_000.surface.json");
  const std::string out = c.dir + "/sample";
  fs::create_directories(out);
  const std::string cmd = std::string("\"") + SPF_SAMPLE_EXE + "\" --surface " + c.dir + "/event_000 --out " + out + " sampler.n_samples=4 > " + out + "/log 2>&1";
  REQUIRE(std::system(cmd.c_str()) == 0);
  c.samp = read_json(out + "/sampled.json");
  c.oscar = out + "/sampled.oscar";
  done = true;
  return c;
}
}  // namespace

// C3: initial hadrons = corona + surface elements + left inside.  The corona is exact (its hadrons are written unchanged), so the only
// error is the closure of the fluid part (decision S23: first order in dx, -6e-2 at dx = 0.5 fm on the test table): the deviation of the chain,
// relative to the full initial totals, is the closure times the core share, and is checked against the closure of the surface file.
TEST_CASE("C3 conservation chain: corona + surface + inside equal the initial totals within the closure") {
  const Chain& c = chain();
  const auto& co = c.surf["corona"];
  const double E_all = co["corona"]["E_GeV"].get<double>() + co["core"]["E_GeV"].get<double>();
  const double B_all = co["corona"]["B"].get<double>() + co["core"]["B"].get<double>();
  const double E_chain = co["corona"]["E_GeV"].get<double>() + jv(c.surf, "totals_all_elements", "E_GeV") + jv(c.surf, "totals_inside_fluid_at_end", "E_GeV");
  const double B_chain = co["corona"]["B"].get<double>() + jv(c.surf, "totals_all_elements", "B") + jv(c.surf, "totals_inside_fluid_at_end", "B");
  const double devE = (E_chain - E_all) / E_all, devB = (B_chain - B_all) / B_all;
  const double clE = jv(c.surf, "closure", "E"), clB = jv(c.surf, "closure", "B");
  std::printf("[C3] chain deviation (corona + elements + inside - all hadrons) / all hadrons: E %.3e, B %.3e; surface-file closure E %.3e, B %.3e; core share of E %.3f\n",
              devE, devB, clE, clB, co["core"]["E_GeV"].get<double>() / E_all);
  CHECK(std::fabs(devE) <= std::fabs(clE) + 1e-9);
  CHECK(std::fabs(devB) <= std::fabs(clB) + 1e-9);
  CHECK(std::fabs(co["deposited_core_B"].get<double>() - co["core"]["B"].get<double>()) <= 1e-6);
  CHECK(co["n_core"].get<long>() + co["n_corona"].get<long>() == co["n_particles_total"].get<long>());
}

// C4: sampler with corona append: B and Q of the thermal part equal the targets (exact), the event B is thermal B + corona B, the corona block
// is the same in every sample and marked by the index range in the summary.
TEST_CASE("C4 sampler with corona append: exact B and Q of the thermal part, event B = thermal + corona") {
  const Chain& c = chain();
  const long n_corona = c.surf["corona"]["n_corona"].get<long>();
  const long long B_cor = c.surf["corona"]["corona"]["B"].get<long long>();
  const double B_dep = c.surf["corona"]["deposited_all_B"].get<double>();
  long bad = 0;
  for (const auto& s : c.samp["samples"]) {
    if (s["sum"]["B"].get<long long>() != s["targets"]["B"].get<long long>()) ++bad;
    if (s["sum"]["Q"].get<long long>() != s["targets"]["Q"].get<long long>()) ++bad;
    if (s["corona"]["n"].get<long>() != n_corona) ++bad;
    if (s["corona"]["end_index"].get<long>() - s["corona"]["first_index"].get<long>() != n_corona) ++bad;
    if (s["event_totals"]["B"].get<long long>() != s["sum"]["B"].get<long long>() + B_cor) ++bad;
    if (s["n_particles_total"].get<long>() != s["n_particles"].get<long>() + n_corona) ++bad;
  }
  CHECK(bad == 0);
  const long long B_evt = c.samp["samples"][0]["event_totals"]["B"].get<long long>();
  // the thermal target is the rounded surface integral, so the event B differs from the deposited B by the closure of the fluid part
  const double clB = jv(c.surf, "closure", "B"), B_core = c.surf["corona"]["core"]["B"].get<double>();
  std::printf("[C4] %zu samples: thermal B and Q at the targets in every sample (violations %ld); event B %lld = thermal %lld + corona %lld; deposited event B %.3f "
              "(difference %.2f, closure-allowed %.2f); corona block %ld hadrons, indices [%ld, %ld)\n",
              c.samp["samples"].size(), bad, B_evt, c.samp["samples"][0]["sum"]["B"].get<long long>(), B_cor, B_dep, B_evt - B_dep, std::fabs(clB) * B_core + 1.0, n_corona,
              c.samp["samples"][0]["corona"]["first_index"].get<long>(), c.samp["samples"][0]["corona"]["end_index"].get<long>());
  CHECK(std::fabs(B_evt - B_dep) <= std::fabs(clB) * B_core + 1.0);
  std::ifstream in(c.oscar);
  std::string line;
  long declared = -1;
  while (std::getline(in, line))
    if (line.rfind("# event 0 ensemble 0 out", 0) == 0) { declared = std::stol(line.substr(line.rfind(' ') + 1)); break; }
  CHECK(declared == c.samp["samples"][0]["n_particles_total"].get<long>());
}

// C5: keep (deferred initial slice + final slice) against thermal (remove), closure of E and B at dx = 0.2 fm.
TEST_CASE("C5 keep + final slice: closure of E and B, target 1e-2 at dx = 0.2 fm, compared with thermal") {
  const std::string k = run_collide("c5k", "table_two_phase", 0.2, "particlisation.corona.mode=keep");
  const std::string t = run_collide("c5t", "table_two_phase", 0.2, "particlisation.corona.mode=thermal");
  const auto sk = read_json(k + "/event_000.surface.json"), st = read_json(t + "/event_000.surface.json");
  const double kE = jv(sk, "closure", "E"), kB = jv(sk, "closure", "B"), tE = jv(st, "closure", "E"), tB = jv(st, "closure", "B");
  std::printf("[C5] dx = 0.2 fm: keep + final slice closure E %.3e B %.3e (final slice %ld elements, weights E %.4f B %.4f); thermal closure E %.3e B %.3e\n", kE, kB,
              sk["n_elements_final_slice"].get<long>(), sk["final_slice_info"]["weight_E"].get<double>(), sk["final_slice_info"]["weight_B"].get<double>(), tE, tB);
  CHECK(sk["n_elements_initial_slice"].get<long>() == 0);
  CHECK(std::fabs(kE) <= 1e-2);
  CHECK(std::fabs(kB) <= 1e-2);
}

// Chain at lower switching densities: the surface partly on the hadron branch (spline region).
TEST_CASE("C6 chain at eps_sw = 3 and 2 eps_s: surface and sampler run; spline-region share reported") {
  for (const char* tag : {"3", "2"}) {
    const std::string d = run_collide(std::string("c6_") + tag, "table_two_phase", 0.5, std::string("particlisation.eps_sw_eps_s=") + tag + " particlisation.corona.mode=core_corona");
    const auto S = read_json(d + "/event_000.surface.json");
    const std::string out = d + "/sample";
    fs::create_directories(out);
    const std::string cmd = std::string("\"") + SPF_SAMPLE_EXE + "\" --surface " + d + "/event_000 --out " + out + " sampler.n_samples=2 > " + out + "/log 2>&1";
    REQUIRE(std::system(cmd.c_str()) == 0);
    const auto J = read_json(out + "/sampled.json");
    const auto& prep = J["preparation"];
    std::printf("[C6] eps_sw = %s eps_s: %ld elements, spline-flagged %s of %s; corona %ld hadrons; sample 0: %ld particles, B %lld (target %lld)\n", tag,
                S["n_elements"].get<long>(), prep.value("n_spline_flagged", nlohmann::json(0)).dump().c_str(), prep.value("n_elements", nlohmann::json(0)).dump().c_str(),
                S["corona"]["n_corona"].get<long>(), J["samples"][0]["n_particles_total"].get<long>(), J["samples"][0]["sum"]["B"].get<long long>(),
                J["samples"][0]["targets"]["B"].get<long long>());
    CHECK(J["samples"][0]["ok"].get<bool>());
    CHECK(S["closed"].get<bool>());
  }
}
