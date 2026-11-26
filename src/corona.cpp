#include "spf/corona.hpp"

#include "spf/particlisation/species.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace spf {

CoronaParams CoronaParams::from_config(const Config& cfg) {
  CoronaParams p;
  const std::string m = cfg.get<std::string>("particlisation.corona.mode");
  if (m == "core_corona") p.mode = CoronaMode::CoreCorona;
  else if (m == "thermal") p.mode = CoronaMode::Thermal;
  else if (m == "keep") p.mode = CoronaMode::Keep;
  else throw ConfigError("particlisation.corona.mode must be 'core_corona', 'thermal' or 'keep', got '" + m + "'");
  p.reference_eos = cfg.get<std::string>("particlisation.corona.reference_eos");
  return p;
}

const char* CoronaParams::mode_name() const {
  return mode == CoronaMode::CoreCorona ? "core_corona" : (mode == CoronaMode::Thermal ? "thermal" : "keep");
}

int corona_charge(const Particle& p) {
  if (p.charge != 0) return p.charge;
  const long a = p.pdg < 0 ? -p.pdg : p.pdg;
  return a >= 1000000000L ? 0 : particlisation::pdg_charge(p.pdg);
}

double corona_mass(const Particle& p) {
  if (p.mass > 0.0) return p.mass;
  const double m2 = p.E * p.E - p.px * p.px - p.py * p.py - p.pz * p.pz;
  return m2 > 0.0 ? std::sqrt(m2) : 0.0;
}

CoronaTotals corona_totals(const std::vector<Particle>& all, const std::vector<char>& mask, bool select) {
  CoronaTotals t;
  for (size_t i = 0; i < all.size(); ++i) {
    if ((mask[i] != 0) != select) continue;
    const Particle& p = all[i];
    ++t.n;
    t.E += p.E; t.Px += p.px; t.Py += p.py; t.Pz += p.pz;
    t.B += p.B;
    t.Q += corona_charge(p);
  }
  return t;
}

nlohmann::json corona_totals_json(const CoronaTotals& t) {
  return {{"n", t.n}, {"E_GeV", t.E}, {"Px_GeV", t.Px}, {"Py_GeV", t.Py}, {"Pz_GeV", t.Pz}, {"B", t.B}, {"Q", t.Q}};
}

double interp_trilinear(const std::vector<Real>& f, const int n[3], Real dx, const Real origin[3], Real x, Real y, Real z) {
  const Real u[3] = {(x - origin[0]) / dx, (y - origin[1]) / dx, (z - origin[2]) / dx};
  int i0[3];
  Real w[3];
  for (int d = 0; d < 3; ++d) {
    if (u[d] < -0.5 || u[d] > n[d] - 0.5) return 0.0;
    Real uc = std::min<Real>(std::max<Real>(u[d], 0.0), n[d] - 1.0);
    i0[d] = std::min(static_cast<int>(std::floor(uc)), std::max(n[d] - 2, 0));
    w[d] = n[d] > 1 ? uc - i0[d] : 0.0;
  }
  double s = 0.0;
  for (int a = 0; a < 2; ++a)
    for (int b = 0; b < 2; ++b)
      for (int c = 0; c < 2; ++c) {
        const int i = std::min(i0[0] + a, n[0] - 1), j = std::min(i0[1] + b, n[1] - 1), k = std::min(i0[2] + c, n[2] - 1);
        const double wt = (a ? w[0] : 1 - w[0]) * (b ? w[1] : 1 - w[1]) * (c ? w[2] : 1 - w[2]);
        s += wt * f[(static_cast<size_t>(i) * n[1] + j) * n[2] + k];
      }
  return s;
}

CoronaTotals write_corona_oscar(const std::string& path, const std::vector<Particle>& all, const std::vector<char>& is_core, int event_id,
                                const std::string& comment) {
  std::ofstream os(path);
  if (!os) throw std::runtime_error("cannot open " + path);
  const CoronaTotals t = corona_totals(all, is_core, false);
  os << "#!OSCAR2013 particle_lists t x y z mass p0 px py pz pdg ID charge\n";
  os << "# Units: fm fm fm fm GeV GeV GeV GeV GeV none none e\n";
  os << "# " << comment << "\n";
  os << "# event " << event_id << " ensemble 0 out " << t.n << "\n";
  char buf[512];
  for (size_t i = 0; i < all.size(); ++i) {
    if (is_core[i]) continue;
    const Particle& p = all[i];
    std::snprintf(buf, sizeof buf, "%.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %ld %ld %d\n", p.t, p.x, p.y, p.z, corona_mass(p), p.E,
                  p.px, p.py, p.pz, p.pdg, p.id, corona_charge(p));
    os << buf;
  }
  os << "# event " << event_id << " ensemble 0 end 0 impact   0.000 scattering_projectile_target yes\n";
  return t;
}

}  // namespace spf
