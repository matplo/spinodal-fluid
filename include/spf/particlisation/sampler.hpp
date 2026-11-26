#pragma once
// Cooper-Frye particle sampler (host only). Reads a switching surface (surface.hpp), writes OSCAR2013 particle lists.
// Method and conventions: docs/particlisation_format.md and docs/validation.md (WP10b).
//
// Per sample and species s:   candidates ~ Poisson(Lambda_s), Lambda_s = Sum_e n_s(T_e, mu_s,e) dSigma_max,e,
//   dSigma_max = |u.dsig| + |dsig*_vec| (local rest frame, LRF); each candidate takes an element with probability lambda_e / Lambda_s
//   (Poisson superposition: identical to one Poisson draw per element), an LRF momentum from the thermal distribution, and is
//   accepted with probability max(p*.dsig*, 0) / (E* dSigma_max); accepted momenta are boosted to the computational frame.
// Determinism: the table lambda_e is a pure function of the surface; each sample owns one generator seeded from (sampler.seed, event id,
// sample id); samples are independent of each other and of the thread count, the output is bit-identical for any number of threads.
#include "spf/config.hpp"
#include "spf/particlisation/rng.hpp"
#include "spf/particlisation/species.hpp"
#include "spf/particlisation/surface.hpp"

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace spf::particlisation {

struct SamplerOptions {
  std::string conservation = "global_B_Q";  // global_B_Q | poisson
  std::string global_method = "exact_counts";  // exact_counts (eos_consistent; no rejection) | rejection (reference: baryon-stage and label rejection)
  std::string spline_policy = "use_table";  // use_table | skip
  std::string mu_rule = "rho_matched";  // rho_matched (S24: mu_kin from n_N - n_Nbar = rho) | table (mu_kin = mu_B - dw/drho); eos_consistent set only
  double mu_T_floor_GeV = 0.005;        // rho_matched: elements with T < floor are solved and sampled at T = floor and counted in PrepReport::n_mu_floor
  int quad_nodes = 16;
  double spline_tol_rel = 0.02, spline_tol_abs_fm3 = 1e-6;
  double T_min_GeV = 1e-3;
  long long max_trials = 200000, max_label_trials = 1000000;
  double charge_Z = 82, charge_A = 208;
  std::uint64_t seed = 12345;
  double mean_field_K_MeV = 300, rho_s = 0.153, mN_GeV = 0.94;
  bool smear_position = true, sort_by_time = true;
  static SamplerOptions from_config(const Config& cfg);
};

struct Particle {
  double t, x, y, z, m, p0, px, py, pz;
  int pdg, charge, baryon;
  int species;  // index in the species set
  int id = 0;
};

struct PrepReport {
  std::size_t n_elements = 0, n_used = 0, n_spline_flagged = 0, n_skipped_spline = 0, n_skipped_cold = 0, n_negative_u_dsig = 0;
  double B_integral_all = 0, B_integral_sampled = 0;  // Int N.dsig over all elements / over the sampled ones
  double skipped_B_flux = 0, skipped_E_flux_eps_only = 0;  // skipped elements (spline policy skip, cold elements): N.dsig and eps u^0 (u.dsig)
  std::size_t n_mu_floor = 0, n_mu_failed = 0;              // rho_matched: elements solved at the temperature floor; elements where the solver failed (table mu used)
  double max_mu_rel_residual = 0;                           // rho_matched: max over elements of |n_N - n_Nbar - rho| / |rho| at the solution
  double flagged_B_flux = 0;                                // elements flagged as spline region (also under use_table)
  double sum_lambda_total = 0;                              // Sum_s Lambda_s (expected candidates per trial)
  std::vector<double> expected_accepted;                    // per species: expected number of accepted particles (Poisson mean of the accepted count)
  double expected_net_B_accepted = 0;                       // sum_s B_s expected_accepted_s (equals B_integral_sampled when no element has negative p.dsig)
  double expected_negative_fraction_number = 0;             // exact (quadrature) sum w^- / (sum w^+ + sum w^-), number weighted, all species
  long long B_target = 0, Q_target = 0;
  nlohmann::json to_json() const;
};
struct ExactCountTable;  // internal: conditional law of the antinucleon number (global_B_Q, exact_counts)

struct SampleResult {
  std::vector<Particle> particles;
  nlohmann::json summary;
  bool ok = true;
};

class Sampler {
 public:
  Sampler(const SamplerOptions& opt, Surface surf, SpeciesSet set);
  const PrepReport& report() const { return rep_; }
  const SpeciesSet& species() const { return set_; }
  const Surface& surface() const { return surf_; }
  const SamplerOptions& options() const { return opt_; }
  const MeanField& mean_field() const { return mf_; }
  double element_mu_kin(std::size_t i) const { return mukin_[i]; }  // kinetic nucleon chemical potential used for element i [GeV]
  // Expected number of accepted particles per species is not tabulated; Lambda_s (expected candidates) is.
  double candidates_mean(std::size_t species_index) const { return lam_tot_[species_index]; }
  SampleResult sample(int sample_id) const;
  // Samples 0..n-1, in parallel over samples (OpenMP when available); the result vector is ordered by sample id.
  std::vector<SampleResult> sample_all(int n) const;

 private:
  struct Tally;
  bool one_candidate(std::size_t s, Rng& rng, std::vector<Particle>& out, Tally& tl) const;
  bool run_species_n(std::size_t s, std::uint64_t n, Rng& rng, std::vector<Particle>& out, Tally& tl) const;
  void build_exact_table();
  void run_species(std::size_t s, Rng& rng, std::vector<Particle>& out, Tally& tl) const;
  double species_mu(const Species& sp, std::size_t i) const;
  std::vector<double> mukin_;  // per element: kinetic nucleon chemical potential (rho_matched or table rule)
  std::vector<double> Tuse_;   // per element: temperature used for sampling (T, or the floor)

  SamplerOptions opt_;
  Surface surf_;
  SpeciesSet set_;
  MeanField mf_;
  std::vector<std::vector<double>> cum_;  // per species: prefix sums over elements of lambda
  std::vector<double> lam_tot_;
  PrepReport rep_;
  std::vector<double> acc_tot_, neg_tot_;  // per species: expected accepted number, expected rejected-negative number
  // exact_counts: law of the antinucleon number nNb given (B, Q); pion charge pmf; truncation windows
  long long tab_lo_ = 0;
  std::vector<double> tab_cdf_, spmf_;
  long long spmf_J_ = 0;
  double mu_pi_ch_ = 0;  // mean of each charged-pion state
  bool tab_ok_ = false;
};

// OSCAR2013 particle lists (the format SMASH's List modus reads): one event block per sample.
void write_oscar2013(std::ostream& os, const std::vector<SampleResult>& samples, const std::string& generator);

// Conditional charge counts (eos_consistent, global_B_Q, exact_counts). The accepted numbers of the charge states are independent Poisson variables
// (isospin symmetry: p, n each with mean Abar_N/2, pbar, nbar Abar_Nbar/2, pi+, pi0, pi- Abar_pi/3), so the event is the Poisson process conditioned on
// B = Np + Nn - Npb - Nnb and Q = Np - Npb + Npi+ - Npi-. Given the numbers nN = Np + Nn and nNb = Npb + Nnb of nucleons and antinucleons, this draws
// a = Np, b = Npb, pp = Npi+, pm = Npi- with the exact conditional law given Q (charged pions Poisson with mean mu_pi_ch each, so that
// P(a, b, pp, pm | nN, nNb, Q) ~ Bin(a; nN, 1/2) Bin(b; nNb, 1/2) Pois(pp) Pois(pm) on a - b + pp - pm = Q). Returns false when the weight vanishes.
bool draw_charge_counts(Rng& rng, long long nN, long long nNb, double mu_pi_ch, long long Q, long long& a, long long& b, long long& pp, long long& pm);

// Threads used by the sampler's parallel loops (1 without OpenMP).
void set_threads(int n);
int max_threads();

// LRF decomposition of an element (exposed for tests): d0 = u.dsig, sv = dsig* spatial components (upper index), dmax = |d0| + |sv|.
struct ElementKin {
  double d0, sv[3], dmax;
};
ElementKin element_kinematics(const double* rec);

}  // namespace spf::particlisation
