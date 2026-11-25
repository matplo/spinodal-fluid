#include "spf/particlisation/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
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
  j["n_skipped_cold"] = n_skipped_cold;
  j["n_elements_negative_u_dsig"] = n_negative_u_dsig;
  j["B_integral_all_elements"] = B_integral_all;
  j["B_integral_sampled_elements"] = B_integral_sampled;
  j["skipped_B_flux"] = skipped_B_flux;
  j["skipped_E_flux_pressure_neglected"] = skipped_E_flux_eps_only;
  j["spline_flagged_B_flux"] = flagged_B_flux;
  j["expected_candidates_per_trial"] = sum_lambda_total;
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

double Sampler::species_mu(const Species& sp, const double* rec) const {
  switch (sp.mu_kind) {
    case MuKind::zero: return 0.0;
    case MuKind::kinetic_plus: return rec[kMuB] - mf_.dw(rec[kRho]);
    case MuKind::kinetic_minus: return -(rec[kMuB] - mf_.dw(rec[kRho]));
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

#pragma omp parallel
  {
    double cT = -1, cMu = -1, cRho = -1;  // one-state cache: neighbouring elements of a uniform region share (T, mu_B, rho)
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
        const double nN = thermal_density(nucleon.mass, nucleon.g, T, mu0, nucleon.stat, opt_.quad_nodes);
        const double nNb = thermal_density(nucleon.mass, nucleon.g, T, -mu0, nucleon.stat, opt_.quad_nodes);
        cspl = std::fabs((nN - nNb) - r[kRho]) > opt_.spline_tol_rel * r[kRho] + opt_.spline_tol_abs_fm3 ? 1 : 0;
        for (std::size_t s = 0; s < ns; ++s) {
          const Species& sp = set_.species[s];
          dens[s] = thermal_density(sp.mass, sp.g, T, species_mu(sp, r), sp.stat, opt_.quad_nodes);
        }
        cT = T;
        cMu = r[kMuB];
        cRho = r[kRho];
      }
      sflag[i] = cspl;
      if (cspl && skip_spline) {
        flag[i] = 1;
        continue;
      }
      for (std::size_t s = 0; s < ns; ++s) cum_[s][i] = dens[s] * k.dmax;
    }
  }
  // serial pass: fluxes, prefix sums (fixed summation order, independent of the thread count)
  for (std::size_t i = 0; i < n; ++i) {
    const double* r = surf_.at(i);
    const ElementKin k = element_kinematics(r);
    const double Bf = r[kRho] * k.d0;
    rep_.B_integral_all += Bf;
    if (k.d0 < 0) ++rep_.n_negative_u_dsig;
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
  rep_.B_target = std::llround(rep_.B_integral_sampled);
  rep_.Q_target = std::llround(static_cast<double>(rep_.B_target) * opt_.charge_Z / opt_.charge_A);
}

struct Sampler::Tally {
  double pos_n = 0, neg_n = 0, pos_E = 0, neg_E = 0, pos_B = 0, neg_B = 0;
  std::uint64_t ncand = 0;
  void add(const Tally& o) {
    pos_n += o.pos_n; neg_n += o.neg_n; pos_E += o.pos_E; neg_E += o.neg_E; pos_B += o.pos_B; neg_B += o.neg_B; ncand += o.ncand;
  }
};

void Sampler::run_species(std::size_t s, Rng& rng, std::vector<Particle>& out, Tally& tl) const {
  const double L = lam_tot_[s];
  if (!(L > 0.0)) return;
  const Species& sp = set_.species[s];
  const std::vector<double>& cum = cum_[s];
  const std::uint64_t nc = rng.poisson(L);
  tl.ncand += nc;
  const double dx = surf_.dx();
  const double pi2 = 2.0 * 3.14159265358979323846;
  for (std::uint64_t c = 0; c < nc; ++c) {
    const double r = rng.uniform() * L;
    std::size_t e = static_cast<std::size_t>(std::upper_bound(cum.begin(), cum.end(), r) - cum.begin());
    if (e >= cum.size()) e = cum.size() - 1;
    const double* rec = surf_.at(e);
    const ElementKin kin = element_kinematics(rec);
    const double mu = species_mu(sp, rec);
    const double k = sample_momentum(sp.mass, rec[kTemp], mu, sp.stat, rng, opt_.quad_nodes);
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
    if (!(rng.uniform() < w)) continue;
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
  }
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

  if (!global) {
    for (std::size_t s = 0; s < ns; ++s) run_species(s, rng, P, tl);
  } else if (labelled) {
    // baryon sector by whole-event rejection on the net baryon number, other species once
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

  // charge states of the isospin-summed species: independent uniform labels; in global mode the label vector is redrawn until Q = Q_target
  if (fail.empty() && labelled) {
    std::vector<unsigned char> lab(P.size());
    auto draw = [&]() {
      long long q = 0;
      for (std::size_t i = 0; i < P.size(); ++i) {
        const auto& mem = set_.species[P[i].species].members;
        lab[i] = static_cast<unsigned char>(rng.below(mem.size()));
        q += mem[lab[i]].charge;
      }
      return q;
    };
    for (;;) {
      ++trials_label;
      const long long q = draw();
      if (!global || q == rep_.Q_target) break;
      if (trials_label >= opt_.max_label_trials) { fail = "max_label_trials exceeded for the charge"; break; }
    }
    if (fail.empty())
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
  j["targets"] = {{"B", rep_.B_target}, {"Q", rep_.Q_target}, {"conservation", opt_.conservation}};
  nlohmann::json si = {{"B_all_elements", rep_.B_integral_all}, {"B_sampled_elements", rep_.B_integral_sampled}};
  nlohmann::json mm = {{"B", static_cast<double>(B) - rep_.B_integral_sampled}};
  if (surf_.meta.contains("integrals") && surf_.meta["integrals"].is_object()) {
    const auto& ig = surf_.meta["integrals"];
    const char* keys[4][2] = {{"E", "E_GeV"}, {"Px", "Px_GeV"}, {"Py", "Py_GeV"}, {"Pz", "Pz_GeV"}};
    const double vals[4] = {E, px, py, pz};
    for (int i = 0; i < 4; ++i)
      if (ig.contains(keys[i][0]) && ig[keys[i][0]].is_number()) {
        si[keys[i][1]] = ig[keys[i][0]];
        mm[keys[i][1]] = vals[i] - ig[keys[i][0]].get<double>();
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

}  // namespace spf::particlisation
