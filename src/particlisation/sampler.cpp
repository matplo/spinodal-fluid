#include "spf/particlisation/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace spf::particlisation {

void set_threads(int n) {
#ifdef _OPENMP
  omp_set_num_threads(n);
#else
  (void)n;
#endif
}
int max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

SamplerOptions SamplerOptions::from_config(const Config& c) {
  SamplerOptions o;
  o.conservation = c.get<std::string>("sampler.conservation");
  o.spline_policy = c.get<std::string>("sampler.spline_policy");
  o.mu_rule = c.get<std::string>("sampler.mu_rule");
  if (o.mu_rule != "rho_matched" && o.mu_rule != "table") throw ConfigError("unknown sampler.mu_rule '" + o.mu_rule + "' (rho_matched | table)");
  o.mu_T_floor_GeV = c.get<double>("sampler.mu_T_floor_GeV");
  o.global_method = c.get<std::string>("sampler.global_method");
  if (o.global_method != "exact_counts" && o.global_method != "rejection") throw ConfigError("unknown sampler.global_method '" + o.global_method + "' (exact_counts | rejection)");
  if (o.conservation != "global_B_Q" && o.conservation != "poisson") throw ConfigError("unknown sampler.conservation '" + o.conservation + "' (global_B_Q | poisson)");
  if (o.spline_policy != "use_table" && o.spline_policy != "skip") throw ConfigError("unknown sampler.spline_policy '" + o.spline_policy + "' (use_table | skip)");
  o.quad_nodes = c.get<int>("sampler.quad_nodes");
  o.spline_tol_rel = c.get<double>("sampler.spline_tol_rel");
  o.spline_tol_abs_fm3 = c.get<double>("sampler.spline_tol_abs_fm3");
  o.T_min_GeV = c.get<double>("sampler.T_min_GeV");
  o.max_trials = c.get<long long>("sampler.max_trials");
  o.max_label_trials = c.get<long long>("sampler.max_label_trials");
  o.charge_Z = c.get<double>("sampler.charge_Z");
  o.charge_A = c.get<double>("sampler.charge_A");
  o.seed = static_cast<std::uint64_t>(c.get<long long>("sampler.seed"));
  o.mean_field_K_MeV = c.get<double>("sampler.mean_field_K_MeV");
  o.rho_s = c.get<double>("constants.rho_s_fm3");
  o.mN_GeV = c.get<double>("sampler.nucleon_mass_GeV");
  o.smear_position = c.get<std::string>("sampler.position") == "cell_uniform";
  o.sort_by_time = c.get<bool>("sampler.sort_by_time");
  return o;
}

nlohmann::json PrepReport::to_json() const {
  nlohmann::json j;
  j["n_elements"] = n_elements;
  j["n_sampled_elements"] = n_used;
  j["n_spline_region_flagged"] = n_spline_flagged;
  j["n_skipped_spline"] = n_skipped_spline;
  j["mu_rule_n_at_T_floor"] = n_mu_floor;
  j["mu_rule_n_solver_failed"] = n_mu_failed;
  j["mu_rule_max_rel_residual"] = max_mu_rel_residual;
  j["n_skipped_cold"] = n_skipped_cold;
  j["n_elements_negative_u_dsig"] = n_negative_u_dsig;
  j["B_integral_all_elements"] = B_integral_all;
  j["B_integral_sampled_elements"] = B_integral_sampled;
  j["skipped_B_flux"] = skipped_B_flux;
  j["skipped_E_flux_pressure_neglected"] = skipped_E_flux_eps_only;
  j["spline_flagged_B_flux"] = flagged_B_flux;
  j["expected_candidates_per_trial"] = sum_lambda_total;
  j["expected_accepted_per_species"] = expected_accepted;
  j["expected_net_B_accepted"] = expected_net_B_accepted;
  j["expected_negative_fraction_number"] = expected_negative_fraction_number;
  j["B_target"] = B_target;
  j["Q_target"] = Q_target;
  return j;
}

ElementKin element_kinematics(const double* r) {
  const double* ds = r + kDs;
  const double* u = r + kU;
  ElementKin k;
  const double V[3] = {-ds[1], -ds[2], -ds[3]};  // upper-index spatial components of dsig
  const double uv = u[1] * V[0] + u[2] * V[1] + u[3] * V[2];
  k.d0 = u[0] * ds[0] - uv;  // = u^mu dsig_mu
  const double f = ds[0] - uv / (1.0 + u[0]);
  double s2 = 0;
  for (int i = 0; i < 3; ++i) {
    k.sv[i] = V[i] - u[1 + i] * f;
    s2 += k.sv[i] * k.sv[i];
  }
  k.dmax = std::fabs(k.d0) + std::sqrt(s2);
  return k;
}

double Sampler::species_mu(const Species& sp, std::size_t i) const {
  const double* rec = surf_.at(i);
  switch (sp.mu_kind) {
    case MuKind::zero: return 0.0;
    case MuKind::kinetic_plus: return mukin_[i];
    case MuKind::kinetic_minus: return -mukin_[i];
    case MuKind::baryon: return sp.baryon * rec[kMuB];
  }
  return 0.0;
}

Sampler::Sampler(const SamplerOptions& opt, Surface surf, SpeciesSet set) : opt_(opt), surf_(std::move(surf)), set_(std::move(set)) {
  mf_ = MeanField::make(opt_.rho_s, opt_.mN_GeV, opt_.mean_field_K_MeV);
  const std::size_t n = surf_.size(), ns = set_.species.size();
  rep_.n_elements = n;
  cum_.assign(ns, std::vector<double>(n, 0.0));
  std::vector<unsigned char> flag(n, 0);  // 0 sampled, 1 skipped spline, 2 skipped cold / invalid
  std::vector<unsigned char> sflag(n, 0);  // 1: spline region (also if used)
  const Species nucleon = make_eos_consistent_set(opt_.mN_GeV, 0.14, true).species[0];
  const bool skip_spline = opt_.spline_policy == "skip";
  // nucleon species of the set (kinetic mu): used by the rho_matched rule
  Species nuc = nucleon;
  for (const auto& sp : set_.species) if (sp.mu_kind == MuKind::kinetic_plus) nuc = sp;
  const bool rho_matched = opt_.mu_rule == "rho_matched" && set_.kind == "eos_consistent";
  mukin_.assign(n, 0.0);
  Tuse_.assign(n, 0.0);
  std::vector<double> mres(n, 0.0);
  std::vector<unsigned char> mfl(n, 0);  // 1: solved at the temperature floor, 2: solver failed (table mu used)
  const bool want_acc = set_.kind == "eos_consistent";
  std::vector<std::vector<double>> accA, negA;
  if (want_acc) {
    accA.assign(ns, std::vector<double>(n, 0.0));
    negA.assign(ns, std::vector<double>(n, 0.0));
  }

#pragma omp parallel
  {
    double cT = -1, cMu = -1, cRho = -1;  // one-state cache: neighbouring elements of a uniform region share (T, mu_B, rho)
    double cmu = 0, cTu = 0, cres = 0;
    unsigned char cfl = 0;
    std::vector<double> dens(ns, 0.0);
    unsigned char cspl = 0;
#pragma omp for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
      const double* r = surf_.at(static_cast<std::size_t>(i));
      const ElementKin k = element_kinematics(r);
      const double T = r[kTemp];
      if (!(T >= opt_.T_min_GeV) || !(k.dmax > 0.0)) {
        flag[i] = 2;
        continue;
      }
      if (!(T == cT && r[kMuB] == cMu && r[kRho] == cRho)) {
        const double mu0 = r[kMuB] - mf_.dw(r[kRho]);
        cmu = mu0;
        cTu = T;
        cfl = 0;
        cres = 0;
        if (rho_matched) {
          const double Tf = std::max(T, opt_.mu_T_floor_GeV);
          const MuSolve ms = solve_mu_net_density(nuc.mass, nuc.g, Tf, r[kRho], nuc.stat, opt_.quad_nodes, mu0);
          if (ms.status == 2) {
            cfl = 2;  // solver failed: table mu at the true T
          } else {
            cmu = ms.mu;
            cTu = Tf;
            cres = std::fabs(ms.rel_residual);
            if (Tf != T) cfl = 1;
          }
        }
        const double nN = thermal_density(nucleon.mass, nucleon.g, T, mu0, nucleon.stat, opt_.quad_nodes);
        const double nNb = thermal_density(nucleon.mass, nucleon.g, T, -mu0, nucleon.stat, opt_.quad_nodes);
        cspl = std::fabs((nN - nNb) - r[kRho]) > opt_.spline_tol_rel * r[kRho] + opt_.spline_tol_abs_fm3 ? 1 : 0;
        for (std::size_t s = 0; s < ns; ++s) {
          const Species& sp = set_.species[s];
          double mu = 0;
          switch (sp.mu_kind) {
            case MuKind::zero: mu = 0.0; break;
            case MuKind::kinetic_plus: mu = cmu; break;
            case MuKind::kinetic_minus: mu = -cmu; break;
            case MuKind::baryon: mu = sp.baryon * r[kMuB]; break;
          }
          dens[s] = thermal_density(sp.mass, sp.g, cTu, mu, sp.stat, opt_.quad_nodes);
        }
        cT = T;
        cMu = r[kMuB];
        cRho = r[kRho];
      }
      mukin_[i] = cmu;
      Tuse_[i] = cTu;
      mfl[i] = cfl;
      mres[i] = cres;
      sflag[i] = cspl;
      if (cspl && skip_spline) {
        flag[i] = 1;
        continue;
      }
      const double slen = k.dmax - std::fabs(k.d0);
      const bool spacelike = slen > std::fabs(k.d0);
      for (std::size_t s = 0; s < ns; ++s) {
        const double lam = dens[s] * k.dmax;
        cum_[s][i] = lam;
        if (want_acc) {
          const Species& sp = set_.species[s];
          const double alpha = spacelike ? thermal_density_acceptance(sp.mass, sp.g, Tuse_[i], species_mu(sp, static_cast<std::size_t>(i)), sp.stat, opt_.quad_nodes, k.d0, slen).alpha
                                         : std::max(k.d0, 0.0) / k.dmax;
          accA[s][i] = lam * alpha;
          negA[s][i] = lam * (alpha - k.d0 / k.dmax);
        }
      }
    }
  }
  // serial pass: fluxes, prefix sums (fixed summation order, independent of the thread count)
  for (std::size_t i = 0; i < n; ++i) {
    const double* r = surf_.at(i);
    const ElementKin k = element_kinematics(r);
    const double Bf = r[kRho] * k.d0;
    rep_.B_integral_all += Bf;
    if (k.d0 < 0) ++rep_.n_negative_u_dsig;
    if (flag[i] != 2) {
      if (mfl[i] == 1) ++rep_.n_mu_floor;
      if (mfl[i] == 2) ++rep_.n_mu_failed;
      rep_.max_mu_rel_residual = std::max(rep_.max_mu_rel_residual, mres[i]);
    }
    if (sflag[i]) {
      ++rep_.n_spline_flagged;
      rep_.flagged_B_flux += Bf;
    }
    if (flag[i] == 0) {
      ++rep_.n_used;
      rep_.B_integral_sampled += Bf;
    } else {
      if (flag[i] == 1) ++rep_.n_skipped_spline;
      if (flag[i] == 2) ++rep_.n_skipped_cold;
      rep_.skipped_B_flux += Bf;
      rep_.skipped_E_flux_eps_only += r[kEps] * r[kU] * k.d0;
    }
  }
  lam_tot_.assign(ns, 0.0);
  for (std::size_t s = 0; s < ns; ++s) {
    double acc = 0;
    for (std::size_t i = 0; i < n; ++i) {
      acc += cum_[s][i];
      cum_[s][i] = acc;
    }
    lam_tot_[s] = acc;
    rep_.sum_lambda_total += acc;
  }
  acc_tot_.assign(ns, 0.0);
  neg_tot_.assign(ns, 0.0);
  if (want_acc) {
    double pos = 0, neg = 0;
    for (std::size_t s = 0; s < ns; ++s) {
      double a = 0, g = 0;
      for (std::size_t i = 0; i < n; ++i) { a += accA[s][i]; g += negA[s][i]; }
      acc_tot_[s] = a;
      neg_tot_[s] = g;
      pos += a;
      neg += g;
      rep_.expected_net_B_accepted += set_.species[s].baryon * a;
    }
    rep_.expected_accepted = acc_tot_;
    rep_.expected_negative_fraction_number = pos + neg > 0 ? neg / (pos + neg) : 0.0;
  }
  rep_.B_target = std::llround(rep_.B_integral_sampled);
  rep_.Q_target = std::llround(static_cast<double>(rep_.B_target) * opt_.charge_Z / opt_.charge_A);
  if (opt_.conservation == "global_B_Q" && opt_.global_method == "exact_counts" && set_.kind == "eos_consistent") build_exact_table();
}

struct Sampler::Tally {
  double pos_n = 0, neg_n = 0, pos_E = 0, neg_E = 0, pos_B = 0, neg_B = 0;
  std::uint64_t ncand = 0;
  void add(const Tally& o) {
    pos_n += o.pos_n; neg_n += o.neg_n; pos_E += o.pos_E; neg_E += o.neg_E; pos_B += o.pos_B; neg_B += o.neg_B; ncand += o.ncand;
  }
};

bool Sampler::one_candidate(std::size_t s, Rng& rng, std::vector<Particle>& out, Tally& tl) const {
  const double L = lam_tot_[s];
  const Species& sp = set_.species[s];
  const std::vector<double>& cum = cum_[s];
  const double dx = surf_.dx();
  const double pi2 = 2.0 * 3.14159265358979323846;
  ++tl.ncand;
    const double r = rng.uniform() * L;
    std::size_t e = static_cast<std::size_t>(std::upper_bound(cum.begin(), cum.end(), r) - cum.begin());
    if (e >= cum.size()) e = cum.size() - 1;
    const double* rec = surf_.at(e);
    const ElementKin kin = element_kinematics(rec);
    const double mu = species_mu(sp, e);
    const double k = sample_momentum(sp.mass, Tuse_[e], mu, sp.stat, rng, opt_.quad_nodes);
    const double E = std::sqrt(k * k + sp.mass * sp.mass);
    const double ct = 2.0 * rng.uniform() - 1.0, st = std::sqrt(std::max(0.0, 1.0 - ct * ct)), ph = pi2 * rng.uniform();
    const double nx = st * std::cos(ph), ny = st * std::sin(ph), nz = ct;
    const double w = (E * kin.d0 - k * (nx * kin.sv[0] + ny * kin.sv[1] + nz * kin.sv[2])) / (E * kin.dmax);
    const double ps[3] = {k * nx, k * ny, k * nz};
    const double* u = rec + kU;
    const double up = u[1] * ps[0] + u[2] * ps[1] + u[3] * ps[2];
    const double p0 = u[0] * E + up;
    const double fac = E + up / (1.0 + u[0]);
    const double wp = std::max(w, 0.0), wn = std::max(-w, 0.0), ab = std::abs(sp.baryon);
    tl.pos_n += wp; tl.neg_n += wn;
    tl.pos_E += wp * p0; tl.neg_E += wn * p0;
    tl.pos_B += wp * ab; tl.neg_B += wn * ab;
    if (!(rng.uniform() < w)) return false;
    Particle p;
    p.px = ps[0] + u[1] * fac;
    p.py = ps[1] + u[2] * fac;
    p.pz = ps[2] + u[3] * fac;
    p.p0 = p0;
    p.m = sp.mass;
    p.t = rec[kT];
    p.x = rec[kX];
    p.y = rec[kY];
    p.z = rec[kZ];
    if (opt_.smear_position) {
      p.x += dx * (rng.uniform() - 0.5);
      p.y += dx * (rng.uniform() - 0.5);
      p.z += dx * (rng.uniform() - 0.5);
    }
    p.pdg = sp.members[0].pdg;
    p.charge = sp.members[0].charge;
    p.baryon = sp.baryon;
    p.species = static_cast<int>(s);
    out.push_back(p);
    return true;
}

void Sampler::run_species(std::size_t s, Rng& rng, std::vector<Particle>& out, Tally& tl) const {
  const double L = lam_tot_[s];
  if (!(L > 0.0)) return;
  const std::uint64_t nc = rng.poisson(L);
  for (std::uint64_t c = 0; c < nc; ++c) one_candidate(s, rng, out, tl);
}

// exactly n accepted particles of species s: candidates are drawn until n are accepted (the accepted ones are iid from the accepted intensity)
bool Sampler::run_species_n(std::size_t s, std::uint64_t n, Rng& rng, std::vector<Particle>& out, Tally& tl) const {
  if (n == 0) return true;
  if (!(lam_tot_[s] > 0.0)) return false;
  std::uint64_t got = 0, cap = 1000 * n + 1000000;
  while (got < n) {
    if (cap-- == 0) return false;
    got += one_candidate(s, rng, out, tl) ? 1 : 0;
  }
  return true;
}

namespace {
std::string pdg_name(int pdg) {
  switch (pdg) {
    case 2212: return "p";
    case 2112: return "n";
    case -2212: return "pbar";
    case -2112: return "nbar";
    case 211: return "pi+";
    case 111: return "pi0";
    case -211: return "pi-";
  }
  return "pdg_" + std::to_string(pdg);
}
}  // namespace

SampleResult Sampler::sample(int sample_id) const {
  const long long event = surf_.event_id();
  const std::uint64_t seed = derive_seed(opt_.seed, static_cast<std::uint64_t>(event), static_cast<std::uint64_t>(sample_id));
  Rng rng(seed);
  SampleResult R;
  const bool global = opt_.conservation == "global_B_Q";
  const bool labelled = set_.kind == "eos_consistent";
  const std::size_t ns = set_.species.size();
  Tally tl;
  long long trials_B = 0, trials_label = 0;
  std::string fail;
  std::vector<Particle>& P = R.particles;

  auto net_baryon = [&](const std::vector<Particle>& v) {
    long long b = 0;
    for (const auto& p : v) b += p.baryon;
    return b;
  };
  auto total_charge = [&](const std::vector<Particle>& v) {
    long long q = 0;
    for (const auto& p : v) q += p.charge;
    return q;
  };

  const bool exact = global && labelled && opt_.global_method == "exact_counts";
  std::vector<unsigned char> lab;
  if (!global) {
    for (std::size_t s = 0; s < ns; ++s) run_species(s, rng, P, tl);
  } else if (exact) {
    // Poisson process conditioned on (B, Q) without rejection: counts from the exact conditional law, then exactly that many accepted particles per group
    std::size_t iN = 0, iNb = 0, iPi = 0;
    for (std::size_t s = 0; s < ns; ++s) {
      if (set_.species[s].baryon > 0) iN = s;
      else if (set_.species[s].baryon < 0) iNb = s;
      else iPi = s;
    }
    long long a = 0, b = 0, pp = 0, pm = 0, nNb = 0, nN = 0, n0 = 0;
    if (!tab_ok_) {
      fail = "(B, Q) target has zero probability in the Poisson model";
    } else {
      const double r = rng.uniform();
      std::size_t k = static_cast<std::size_t>(std::upper_bound(tab_cdf_.begin(), tab_cdf_.end(), r) - tab_cdf_.begin());
      if (k >= tab_cdf_.size()) k = tab_cdf_.size() - 1;
      nNb = tab_lo_ + static_cast<long long>(k);
      nN = rep_.B_target + nNb;
      if (!draw_charge_counts(rng, nN, nNb, mu_pi_ch_, rep_.Q_target, a, b, pp, pm)) fail = "charge target not reachable";
      n0 = static_cast<long long>(rng.poisson(mu_pi_ch_));
    }
    if (fail.empty()) {
      if (!run_species_n(iN, static_cast<std::uint64_t>(nN), rng, P, tl) || !run_species_n(iNb, static_cast<std::uint64_t>(nNb), rng, P, tl) ||
          !run_species_n(iPi, static_cast<std::uint64_t>(pp + pm + n0), rng, P, tl))
        fail = "candidate cap reached";
    }
    if (fail.empty()) {
      trials_B = 1;
      trials_label = 1;
      lab.assign(P.size(), 0);
      std::vector<std::size_t> gN, gNb, gPi;
      for (std::size_t i = 0; i < P.size(); ++i) (P[i].baryon > 0 ? gN : (P[i].baryon < 0 ? gNb : gPi)).push_back(i);
      auto shuffle = [&](std::vector<std::size_t>& v) {
        for (std::size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[rng.below(i)]);
      };
      shuffle(gN);
      shuffle(gNb);
      shuffle(gPi);
      for (std::size_t i = 0; i < gN.size(); ++i) lab[gN[i]] = static_cast<long long>(i) < a ? 0 : 1;       // p, n
      for (std::size_t i = 0; i < gNb.size(); ++i) lab[gNb[i]] = static_cast<long long>(i) < b ? 0 : 1;     // pbar, nbar
      for (std::size_t i = 0; i < gPi.size(); ++i) lab[gPi[i]] = static_cast<long long>(i) < pp ? 0 : (static_cast<long long>(i) < pp + pm ? 2 : 1);  // pi+, pi-, pi0
    }
  } else if (labelled) {
    // reference method (sampler.global_method = rejection): baryon sector by whole-event rejection, pions once, label vector redrawn until Q = Q_target
    std::vector<Particle> trial;
    for (;;) {
      if (++trials_B > opt_.max_trials) { fail = "max_trials exceeded for the net baryon number"; break; }
      trial.clear();
      Tally t;
      for (std::size_t s = 0; s < ns; ++s)
        if (set_.species[s].baryon != 0) run_species(s, rng, trial, t);
      if (net_baryon(trial) == rep_.B_target) { P.swap(trial); tl.add(t); break; }
    }
    if (fail.empty())
      for (std::size_t s = 0; s < ns; ++s)
        if (set_.species[s].baryon == 0) run_species(s, rng, P, tl);
    if (fail.empty()) {
      lab.assign(P.size(), 0);
      for (;;) {
        ++trials_label;
        long long q = 0;
        for (std::size_t i = 0; i < P.size(); ++i) {
          const auto& mem = set_.species[P[i].species].members;
          lab[i] = static_cast<unsigned char>(rng.below(mem.size()));
          q += mem[lab[i]].charge;
        }
        if (q == rep_.Q_target) break;
        if (trials_label >= opt_.max_label_trials) { fail = "max_label_trials exceeded for the charge"; break; }
      }
    }
  } else {
    // generic species: joint rejection on (B, Q); the baryon sector is drawn first and a failed B check ends the trial (the sectors are independent)
    std::vector<Particle> trial;
    for (;;) {
      if (++trials_B > opt_.max_trials) { fail = "max_trials exceeded for (B, Q)"; break; }
      trial.clear();
      Tally t;
      for (std::size_t s = 0; s < ns; ++s)
        if (set_.species[s].baryon != 0) run_species(s, rng, trial, t);
      if (net_baryon(trial) != rep_.B_target) continue;
      for (std::size_t s = 0; s < ns; ++s)
        if (set_.species[s].baryon == 0) run_species(s, rng, trial, t);
      if (total_charge(trial) == rep_.Q_target) { P.swap(trial); tl.add(t); break; }
    }
  }

  // charge states of the isospin-summed species: poisson mode, independent uniform labels; global_B_Q: labels fixed above
  if (fail.empty() && labelled) {
    if (!global) {
      lab.assign(P.size(), 0);
      for (std::size_t i = 0; i < P.size(); ++i) lab[i] = static_cast<unsigned char>(rng.below(set_.species[P[i].species].members.size()));
      trials_label = 1;
    }
    for (std::size_t i = 0; i < P.size(); ++i) {
      const Member& m = set_.species[P[i].species].members[lab[i]];
      P[i].pdg = m.pdg;
      P[i].charge = m.charge;
    }
  }

  R.ok = fail.empty();
  if (!R.ok) P.clear();
  if (opt_.sort_by_time) std::stable_sort(P.begin(), P.end(), [](const Particle& a, const Particle& b) { return a.t < b.t; });
  int id = 0;
  for (auto& p : P) p.id = id++;

  nlohmann::json j;
  j["sample_id"] = sample_id;
  j["event_id"] = event;
  j["seed"] = seed;
  j["ok"] = R.ok;
  if (!R.ok) j["failure"] = fail;
  nlohmann::json counts = nlohmann::json::object();
  double E = 0, px = 0, py = 0, pz = 0;
  long long B = 0, Q = 0;
  for (const auto& p : P) {
    counts[pdg_name(p.pdg)] = counts.value(pdg_name(p.pdg), 0) + 1;
    E += p.p0; px += p.px; py += p.py; pz += p.pz;
    B += p.baryon;
    Q += p.charge;
  }
  j["n_particles"] = P.size();
  j["counts"] = counts;
  j["sum"] = {{"E_GeV", E}, {"Px_GeV", px}, {"Py_GeV", py}, {"Pz_GeV", pz}, {"B", B}, {"Q", Q}};
  j["targets"] = {{"B", rep_.B_target}, {"Q", rep_.Q_target}, {"conservation", opt_.conservation}, {"global_method", global ? opt_.global_method : std::string("none")}};
  nlohmann::json si = {{"B_all_elements", rep_.B_integral_all}, {"B_sampled_elements", rep_.B_integral_sampled}};
  nlohmann::json mm = {{"B", static_cast<double>(B) - rep_.B_integral_sampled}};
  // surface integrals: the key written by spf_collide is "totals_all_elements" ({E_GeV, Px_GeV, ...}); the stand-alone
  // sampler's "integrals" ({E, Px, ...}) is accepted as a fallback
  {
    const nlohmann::json* ig = nullptr;
    bool short_keys = false;
    if (surf_.meta.contains("totals_all_elements") && surf_.meta["totals_all_elements"].is_object()) ig = &surf_.meta["totals_all_elements"];
    else if (surf_.meta.contains("integrals") && surf_.meta["integrals"].is_object()) { ig = &surf_.meta["integrals"]; short_keys = true; }
    if (ig) {
      const char* keys[4][2] = {{"E", "E_GeV"}, {"Px", "Px_GeV"}, {"Py", "Py_GeV"}, {"Pz", "Pz_GeV"}};
      const double vals[4] = {E, px, py, pz};
      for (int i = 0; i < 4; ++i) {
        const char* k = short_keys ? keys[i][0] : keys[i][1];
        if (ig->contains(k) && (*ig)[k].is_number()) {
          si[keys[i][1]] = (*ig)[k];
          mm[keys[i][1]] = vals[i] - (*ig)[k].get<double>();
        }
      }
    }
  }
  j["surface_integrals"] = si;
  j["mismatch_sampled_minus_surface"] = mm;
  auto frac = [](double ng, double ps) { return ng + ps > 0 ? ng / (ng + ps) : 0.0; };
  j["negative_fraction"] = {{"definition", "sum of rejected negative contributions / sum of absolute contributions, over the candidates of the accepted trial"},
                            {"number", frac(tl.neg_n, tl.pos_n)}, {"energy", frac(tl.neg_E, tl.pos_E)}, {"baryon", frac(tl.neg_B, tl.pos_B)}};
  j["skipped"] = {{"policy", opt_.spline_policy}, {"n_elements_spline", rep_.n_skipped_spline}, {"n_elements_cold", rep_.n_skipped_cold},
                  {"B_flux", rep_.skipped_B_flux}, {"E_flux_pressure_neglected", rep_.skipped_E_flux_eps_only}};
  j["spline_region_flagged_elements"] = rep_.n_spline_flagged;
  j["candidates_in_accepted_trial"] = tl.ncand;
  j["efficiency"] = {{"trials_baryon_or_joint", trials_B}, {"trials_charge_labels", trials_label},
                     {"acceptance_efficiency", global ? 1.0 / (static_cast<double>(std::max(1LL, trials_B)) * static_cast<double>(std::max(1LL, trials_label))) : 1.0}};
  R.summary = j;
  return R;
}

namespace {
std::size_t draw_discrete(Rng& rng, const std::vector<double>& w) {
  double tot = 0;
  for (double x : w) tot += x;
  const double r = rng.uniform() * tot;
  double acc = 0;
  std::size_t last = 0;
  for (std::size_t i = 0; i < w.size(); ++i) {
    if (w[i] <= 0.0) continue;
    acc += w[i];
    last = i;
    if (r < acc) return i;
  }
  return last;
}
}  // namespace

namespace {
constexpr double kNegInf = -1e300;
double log_pois(long long k, double mu) {
  if (k < 0) return kNegInf;
  if (mu <= 0.0) return k == 0 ? 0.0 : kNegInf;
  return static_cast<double>(k) * std::log(mu) - mu - std::lgamma(static_cast<double>(k) + 1.0);
}
double log_binh(long long n, long long k) {
  if (k < 0 || k > n) return kNegInf;
  return std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0) - static_cast<double>(n) * std::log(2.0);
}
long long pois_window(double mu) { return static_cast<long long>(std::ceil(mu + 14.0 * std::sqrt(mu) + 30.0)); }
// log pmf of S = pi+ - pi-, both Poisson(mu), for s = -J..J (index s + J)
std::vector<double> log_skellam(double mu, long long J) {
  std::vector<double> out(static_cast<std::size_t>(2 * J + 1), kNegInf);
  for (long long sgn = -J; sgn <= J; ++sgn) {
    double mx = kNegInf;
    std::vector<double> t;
    for (long long j = std::max(0LL, sgn); j <= J + std::max(0LL, sgn); ++j) {
      const double l = log_pois(j, mu) + log_pois(j - sgn, mu);
      t.push_back(l);
      mx = std::max(mx, l);
    }
    if (mx <= kNegInf) continue;
    double sum = 0;
    for (double l : t) sum += std::exp(l - mx);
    out[static_cast<std::size_t>(sgn + J)] = mx + std::log(sum);
  }
  return out;
}
// log pmf of U = a - b, a ~ Bin(nN, 1/2), b ~ Bin(nNb, 1/2); index u + nNb
std::vector<double> log_pU(long long nN, long long nNb) {
  std::vector<double> out(static_cast<std::size_t>(nN + nNb + 1), kNegInf);
  for (long long u = -nNb; u <= nN; ++u) {
    double mx = kNegInf;
    std::vector<double> t;
    for (long long b = std::max(0LL, -u); b <= nNb && u + b <= nN; ++b) {
      const double l = log_binh(nN, u + b) + log_binh(nNb, b);
      t.push_back(l);
      mx = std::max(mx, l);
    }
    if (mx <= kNegInf) continue;
    double sum = 0;
    for (double l : t) sum += std::exp(l - mx);
    out[static_cast<std::size_t>(u + nNb)] = mx + std::log(sum);
  }
  return out;
}
double log_sum_exp(const std::vector<double>& v) {
  double mx = kNegInf;
  for (double x : v) mx = std::max(mx, x);
  if (mx <= kNegInf) return kNegInf;
  double s = 0;
  for (double x : v) s += std::exp(x - mx);
  return mx + std::log(s);
}
std::size_t draw_log_discrete(Rng& rng, const std::vector<double>& lw) {
  double mx = kNegInf;
  for (double x : lw) mx = std::max(mx, x);
  std::vector<double> w(lw.size());
  for (std::size_t i = 0; i < lw.size(); ++i) w[i] = lw[i] <= kNegInf ? 0.0 : std::exp(lw[i] - mx);
  return draw_discrete(rng, w);
}
}  // namespace

bool draw_charge_counts(Rng& rng, long long nN, long long nNb, double mu, long long Q, long long& a, long long& b, long long& pp, long long& pm) {
  const long long J = pois_window(mu);
  const std::vector<double> lS = log_skellam(mu, J), lU = log_pU(nN, nNb);
  std::vector<double> lw(lU.size(), kNegInf);
  bool any = false;
  for (long long u = -nNb; u <= nN; ++u) {
    const long long s = Q - u;
    if (s < -J || s > J) continue;
    const double l = lU[static_cast<std::size_t>(u + nNb)] + lS[static_cast<std::size_t>(s + J)];
    lw[static_cast<std::size_t>(u + nNb)] = l;
    any = any || l > kNegInf;
  }
  if (!any) return false;
  const long long u = static_cast<long long>(draw_log_discrete(rng, lw)) - nNb, s = Q - u;
  std::vector<double> la(static_cast<std::size_t>(nN) + 1, kNegInf);
  for (long long i = 0; i <= nN; ++i) {
    const long long j = i - u;
    if (j >= 0 && j <= nNb) la[static_cast<std::size_t>(i)] = log_binh(nN, i) + log_binh(nNb, j);
  }
  a = static_cast<long long>(draw_log_discrete(rng, la));
  b = a - u;
  std::vector<double> lp(static_cast<std::size_t>(J + std::max(0LL, s)) + 1, kNegInf);
  for (long long j = std::max(0LL, s); j < static_cast<long long>(lp.size()); ++j) lp[static_cast<std::size_t>(j)] = log_pois(j, mu) + log_pois(j - s, mu);
  pp = static_cast<long long>(draw_log_discrete(rng, lp));
  pm = pp - s;
  return true;
}

void Sampler::build_exact_table() {
  // Accepted numbers are Poisson: nucleons Abar_N, antinucleons Abar_Nbar, charged pions Abar_pi/3 per state. P(nNb | B, Q) ~ Pois(B + nNb; Abar_N) Pois(nNb; Abar_Nbar) G(nNb),
  // G = P(Q | nN, nNb) = Sum_s Skellam(s) P(a - b = Q - s). nNb runs over a window of the Poisson law (mean + 14 sigma + 30, probability left outside < 1e-40).
  std::size_t iN = 0, iNb = 0, iPi = 0;
  for (std::size_t s = 0; s < set_.species.size(); ++s) {
    if (set_.species[s].baryon > 0) iN = s;
    else if (set_.species[s].baryon < 0) iNb = s;
    else iPi = s;
  }
  const double AN = acc_tot_[iN], ANb = acc_tot_[iNb];
  mu_pi_ch_ = acc_tot_[iPi] / 3.0;
  spmf_J_ = pois_window(mu_pi_ch_);
  spmf_ = log_skellam(mu_pi_ch_, spmf_J_);
  const long long B = rep_.B_target, Q = rep_.Q_target;
  tab_lo_ = std::max(0LL, -B);
  const long long hi = std::max(tab_lo_, pois_window(ANb));
  std::vector<double> lw(static_cast<std::size_t>(hi - tab_lo_ + 1), kNegInf);
  for (long long nb = tab_lo_; nb <= hi; ++nb) {
    const long long nn = B + nb;
    const double lp = log_pois(nn, AN) + log_pois(nb, ANb);
    if (lp <= kNegInf) continue;
    const std::vector<double> lU = log_pU(nn, nb);
    std::vector<double> t;
    for (long long sgn = -spmf_J_; sgn <= spmf_J_; ++sgn) {
      const long long u = Q - sgn;
      if (u < -nb || u > nn) continue;
      const double l = spmf_[static_cast<std::size_t>(sgn + spmf_J_)] + lU[static_cast<std::size_t>(u + nb)];
      if (l > kNegInf) t.push_back(l);
    }
    const double lg = log_sum_exp(t);
    if (lg > kNegInf) lw[static_cast<std::size_t>(nb - tab_lo_)] = lp + lg;
  }
  const double lt = log_sum_exp(lw);
  tab_cdf_.assign(lw.size(), 0.0);
  tab_ok_ = lt > kNegInf;
  if (!tab_ok_) return;
  double acc = 0;
  for (std::size_t i = 0; i < lw.size(); ++i) {
    acc += lw[i] <= kNegInf ? 0.0 : std::exp(lw[i] - lt);
    tab_cdf_[i] = acc;
  }
}

std::vector<SampleResult> Sampler::sample_all(int n) const {
  std::vector<SampleResult> out(static_cast<std::size_t>(std::max(n, 0)));
#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = sample(i);
  return out;
}

void write_oscar2013(std::ostream& os, const std::vector<SampleResult>& samples, const std::string& generator) {
  os << "#!OSCAR2013 particle_lists t x y z mass p0 px py pz pdg ID charge\n";
  os << "# Units: fm fm fm fm GeV GeV GeV GeV GeV none none e\n";
  os << "# " << generator << "\n";
  char buf[512];
  for (std::size_t s = 0; s < samples.size(); ++s) {
    if (!samples[s].ok) continue;
    os << "# event " << s << " ensemble 0 out " << samples[s].particles.size() << "\n";
    for (const auto& p : samples[s].particles) {
      std::snprintf(buf, sizeof buf, "%.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %.15g %d %d %d\n", p.t, p.x, p.y, p.z, p.m, p.p0, p.px, p.py, p.pz, p.pdg, p.id, p.charge);
      os << buf;
    }
    os << "# event " << s << " ensemble 0 end 0 impact   0.000 scattering_projectile_target yes\n";
  }
}

std::vector<Particle> read_corona_oscar(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open corona file '" + path + "'");
  std::vector<Particle> out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    Particle p{};
    long long pdg = 0, id = 0;
    int ch = 0;
    if (!(is >> p.t >> p.x >> p.y >> p.z >> p.m >> p.p0 >> p.px >> p.py >> p.pz >> pdg >> id >> ch))
      throw std::runtime_error("malformed corona line in '" + path + "'");
    p.pdg = static_cast<int>(pdg);
    p.charge = ch;
    p.baryon = pdg_baryon(pdg);
    p.species = -1;
    p.id = static_cast<int>(id);
    out.push_back(p);
  }
  return out;
}

void append_corona(SampleResult& r, const std::vector<Particle>& corona) {
  if (!r.ok) return;
  const std::size_t first = r.particles.size();
  int next_id = 0;
  for (const auto& p : r.particles) next_id = std::max(next_id, p.id + 1);
  double cE = 0, cx = 0, cy = 0, cz = 0;
  long long cB = 0, cQ = 0;
  std::map<int, long long> counts;
  for (auto p : corona) {
    p.id = next_id++;
    cE += p.p0; cx += p.px; cy += p.py; cz += p.pz;
    cB += p.baryon; cQ += p.charge;
    counts[p.pdg] += 1;
    r.particles.push_back(p);
  }
  nlohmann::json cj = nlohmann::json::object();
  for (const auto& kv : counts) cj[std::to_string(kv.first)] = kv.second;
  r.summary["corona"] = {{"appended", true}, {"n", corona.size()}, {"first_index", first}, {"end_index", r.particles.size()},
                         {"sum", {{"E_GeV", cE}, {"Px_GeV", cx}, {"Py_GeV", cy}, {"Pz_GeV", cz}, {"B", cB}, {"Q", cQ}}}, {"counts_pdg", cj}};
  const auto& s = r.summary["sum"];
  r.summary["thermal_range"] = {{"first_index", 0}, {"end_index", first}};
  r.summary["event_totals"] = {{"E_GeV", s["E_GeV"].get<double>() + cE}, {"Px_GeV", s["Px_GeV"].get<double>() + cx},
                               {"Py_GeV", s["Py_GeV"].get<double>() + cy}, {"Pz_GeV", s["Pz_GeV"].get<double>() + cz},
                               {"B", s["B"].get<long long>() + cB}, {"Q", s["Q"].get<long long>() + cQ}};
  r.summary["n_particles_total"] = r.particles.size();
}

}  // namespace spf::particlisation
