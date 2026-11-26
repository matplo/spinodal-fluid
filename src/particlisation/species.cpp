#include "spf/particlisation/species.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>

namespace spf::particlisation {

MeanField MeanField::make(double rho_s_fm3, double mN_GeV, double K_MeV) {
  MeanField f;
  f.rho_s = rho_s_fm3;
  f.mN_MeV = 1000.0 * mN_GeV;
  f.K_MeV = K_MeV;
  const double hc = 197.3269804, gN = 4.0, pi = 3.14159265358979323846, mN = f.mN_MeV;
  auto ekin0 = [&](double rho) {
    const double kF = std::cbrt(6 * pi * pi * rho / gN) * hc;
    return gN / (16 * pi * pi) * (kF * std::sqrt(kF * kF + mN * mN) * (2 * kF * kF + mN * mN) - std::pow(mN, 4) * std::asinh(kF / mN)) / (hc * hc * hc);
  };
  const double h = 1e-4, rs = rho_s_fm3;
  auto EA = [&](double r) { return ekin0(r) / r - mN; };
  const double d1 = (EA(rs + h) - EA(rs - h)) / (2 * h);
  const double d2 = (EA(rs + h) - 2 * EA(rs) + EA(rs - h)) / (h * h);
  const double KN = 9 * rs * rs * d2;
  f.B_MeV = (K_MeV - KN) / 18.0;
  f.A_MeV = rs * d1 + 2 * f.B_MeV;
  return f;
}
double MeanField::w(double rho) const {
  const double c = rho / rho_s;
  return 1e-3 * (-A_MeV * c + B_MeV * c * c) * rho;
}
double MeanField::dw(double rho) const {
  const double c = rho / rho_s;
  return 1e-3 * (-2 * A_MeV * c + 3 * B_MeV * c * c);
}

SpeciesSet make_eos_consistent_set(double mN, double mpi, bool qs) {
  SpeciesSet s;
  s.kind = "eos_consistent";
  Species N;
  N.name = "N";
  N.mass = mN;
  N.g = 4;
  N.stat = qs ? Stat::fermi : Stat::boltzmann;
  N.baryon = 1;
  N.mu_kind = MuKind::kinetic_plus;
  N.members = {{2212, 1}, {2112, 0}};
  Species Nb = N;
  Nb.name = "Nbar";
  Nb.baryon = -1;
  Nb.mu_kind = MuKind::kinetic_minus;
  Nb.members = {{-2212, -1}, {-2112, 0}};
  Species pi;
  pi.name = "pi";
  pi.mass = mpi;
  pi.g = 3;
  pi.stat = qs ? Stat::bose : Stat::boltzmann;
  pi.baryon = 0;
  pi.mu_kind = MuKind::zero;
  pi.members = {{211, 1}, {111, 0}, {-211, -1}};
  s.species = {N, Nb, pi};
  return s;
}

namespace {
int qcharge3(int q) { return (q == 2 || q == 4 || q == 6) ? 2 : -1; }
}  // namespace

int pdg_baryon(long long pdg) {
  const long long a = std::llabs(pdg);
  if (a >= 1000000000LL) return static_cast<int>((a / 10) % 1000) * (pdg < 0 ? -1 : 1);
  if ((a / 1000) % 10 != 0) return pdg < 0 ? -1 : 1;
  return 0;
}
int pdg_degeneracy(long long pdg) {
  const int j = static_cast<int>(std::llabs(pdg) % 10);
  return j > 0 ? j : 1;
}
int pdg_charge(long long pdg) {
  const long long a = std::llabs(pdg);
  const int sgn = pdg < 0 ? -1 : 1;
  const int q1 = static_cast<int>((a / 1000) % 10), q2 = static_cast<int>((a / 100) % 10), q3 = static_cast<int>((a / 10) % 10);
  if (q1 != 0) return sgn * (qcharge3(q1) + qcharge3(q2) + qcharge3(q3)) / 3;
  const int c3 = (q2 % 2 == 1) ? (-qcharge3(q2) + qcharge3(q3)) : (qcharge3(q2) - qcharge3(q3));
  return sgn * c3 / 3;
}

SpeciesSet read_hrg_set(const std::string& path, double max_mass) {
  std::ifstream in(path);
  if (!in) throw ConfigError("sampler.hrg_particles_file: cannot open '" + path + "'");
  SpeciesSet set;
  set.kind = "hrg";
  std::set<long long> present;
  struct Row {
    std::string name;
    double m;
    long long pdg;
  };
  std::vector<Row> rows;
  std::string line;
  while (std::getline(in, line)) {
    const size_t h = line.find('#');
    if (h != std::string::npos) line = line.substr(0, h);
    std::istringstream is(line);
    std::string name, parity;
    double m, width;
    if (!(is >> name >> m >> width >> parity)) continue;
    long long pdg;
    while (is >> pdg) {
      if (std::llabs(pdg) >= 1000000000LL) continue;
      if (!(m > 0.0)) continue;
      if (max_mass > 0.0 && m > max_mass) continue;
      rows.push_back({name, m, pdg});
      present.insert(pdg);
    }
  }
  std::set<long long> done;
  auto push = [&](const std::string& name, double m, long long pdg) {
    if (!done.insert(pdg).second) return;
    Species sp;
    sp.name = name;
    sp.mass = m;
    sp.g = pdg_degeneracy(pdg);
    sp.baryon = pdg_baryon(pdg);
    sp.stat = sp.baryon != 0 ? Stat::fermi : Stat::bose;
    sp.mu_kind = sp.baryon != 0 ? MuKind::baryon : MuKind::zero;
    sp.members = {{static_cast<int>(pdg), pdg_charge(pdg)}};
    set.species.push_back(sp);
  };
  for (const Row& r : rows) {
    push(r.name, r.m, r.pdg);
    const long long a = std::llabs(r.pdg);
    const bool baryon = (a / 1000) % 10 != 0;
    const bool diag = !baryon && ((a / 100) % 10 == (a / 10) % 10);
    if (!diag && !present.count(-r.pdg)) push(r.name + "~", r.m, -r.pdg);
  }
  if (set.species.empty()) throw ConfigError("sampler.hrg_particles_file: no hadron found in '" + path + "'");
  return set;
}

SpeciesSet make_species_set(const Config& cfg) {
  const std::string kind = cfg.get<std::string>("sampler.species");
  if (kind == "eos_consistent")
    return make_eos_consistent_set(cfg.get<double>("sampler.nucleon_mass_GeV"), cfg.get<double>("sampler.pion_mass_GeV"),
                                   cfg.get<bool>("sampler.quantum_statistics"));
  if (kind == "hrg") return read_hrg_set(cfg.get<std::string>("sampler.hrg_particles_file"), cfg.get<double>("sampler.hrg_max_mass_GeV"));
  throw ConfigError("unknown sampler.species '" + kind + "' (eos_consistent | hrg)");
}

}  // namespace spf::particlisation
