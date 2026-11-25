#pragma once
// Hadron content of the sampler (sampler.species) and the mean field of the hadron phase of the equation of state.
#include "spf/config.hpp"
#include "spf/particlisation/thermal.hpp"

#include <string>
#include <vector>

namespace spf::particlisation {

// Hadron-phase mean field w(rho) exactly as in tools/eos/finite_T/eos2b.py (units there: MeV, fm; here the accessors return GeV):
//   w(rho) = rho (-A c + B c^2),  c = rho/rho_s,  dw/drho = -2 A c + 3 B c^2,  d2w/drho2 = (-2 A + 6 B c)/rho_s,
//   B = (K - 9 rho_s^2 d2) / 18,  A = rho_s d1 + 2 B,  d1, d2 = first and second derivative at rho_s of E/A = e_kin0(rho)/rho - m_N
//   (central differences, step 1e-4 fm^-3, as in eos2b), e_kin0 the T = 0 relativistic Fermi gas of g = 4 nucleons, K = 300 MeV.
struct MeanField {
  double rho_s = 0.153, mN_MeV = 940.0, K_MeV = 300.0, A_MeV = 0, B_MeV = 0;
  static MeanField make(double rho_s_fm3, double mN_GeV, double K_MeV);
  double w(double rho) const;    // GeV/fm^3
  double dw(double rho) const;   // GeV   (d w / d rho)
};

enum class MuKind { zero, kinetic_plus, kinetic_minus, baryon };

struct Member {
  int pdg;
  int charge;  // units of e
};

struct Species {
  std::string name;
  double mass = 0;  // GeV
  double g = 1;     // degeneracy
  Stat stat = Stat::boltzmann;
  int baryon = 0;
  MuKind mu_kind = MuKind::zero;
  std::vector<Member> members;  // one member: fixed species; several: charge states assigned by label (equal probability)
};

struct SpeciesSet {
  std::string kind;  // "eos_consistent" | "hrg"
  std::vector<Species> species;
};

// eos_consistent: N (g = 4: p, n), Nbar (g = 4: pbar, nbar), both Fermi-Dirac with the kinetic mu_0 = mu_B - dw/drho; pion (g = 3, Bose, mu = 0).
SpeciesSet make_eos_consistent_set(double mN_GeV, double mpi_GeV, bool quantum_statistics = true);
// hrg: every hadron of a SMASH particles.txt (pole masses, mu = B mu_B, quantum statistics, antiparticles added); nuclei (|pdg| > 10^9) are skipped.
// max_mass_GeV > 0 drops heavier species. Throws ConfigError when the file cannot be read.
SpeciesSet read_hrg_set(const std::string& particles_txt, double max_mass_GeV);
SpeciesSet make_species_set(const Config& cfg);

// PDG-code rules used for the hrg set (exposed for tests): charge in units of e, baryon number, 2J+1.
int pdg_charge(long long pdg);
int pdg_baryon(long long pdg);
int pdg_degeneracy(long long pdg);

}  // namespace spf::particlisation
