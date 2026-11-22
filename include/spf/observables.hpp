#pragma once
// Observables of the collision simulations (WP6): density moments, transverse flow, phase-plane occupancy, baryon fraction in a
// region mask, end of the fluid stage, clump finder; the ObservableRecorder and the snapshot post-processing share one code path.
// Definitions: docs/equations.md, Sect. 10; output files: docs/observables.md.
//
// Reductions are named functor structs (docs/device_rules.md, rules 2 and 6). The phase-plane histogram uses atomic additions
// (Kokkos::atomic_add on a device View), so its summation order is not reproducible between runs on the OpenMP and CUDA backends.
// The clump finder runs on the host from mirror copies (union-find over the cells above the threshold).
#include "spf/config.hpp"
#include "spf/hydro3d.hpp"
#include "spf/shasta.hpp"

#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace spf {

inline constexpr int kObsMaxMoment = 7;

// ---------------------------------------------------------------------------------------------------- configuration
struct ObsParams {
  bool rest_frame = false;            // obs.moment_density: false = computational frame (R), true = rest frame (rho)
  int connectivity = 6;               // obs.clump_connectivity
  bool periodic = true;               // clumps wrap around the box iff grid.boundary == "periodic"
  Real rho_s = 0.153, eps_s = 0.94 * 0.153;
  Real clump_rho_min_rho_s = 7.0;
  int clump_hist_max = 25;            // clump histogram: bins 0..clump_hist_max (bin k: round(B) = k), plus overflow
  Real fluid_end_eps_eps_s = 5.0;
  long every_steps = 1;
  int n_rho = 120, n_eps = 120;       // phase-plane bins
  Real d_rho = 0.1, d_eps = 0.25;     // bin widths in rho_s and eps_s
  std::vector<Real> hist_times;       // obs.hist_times_fm
  static ObsParams from_config(const Config& cfg);
};

// Views of the fields the observables need (ghost layers of the Grid included; only interior cells are read).
struct ObsFields {
  Grid grid;
  Field3 R;      // computational-frame baryon density gamma * rho [fm^-3]
  Field3 rho;    // rest-frame baryon density [fm^-3]
  Field3 eps;    // rest-frame energy density [GeV/fm^3]
  Field3 vx, vy; // laboratory-frame velocity (c = 1)
};

// ---------------------------------------------------------------------------------------------------- results
struct ObsResult {
  Real time = 0.0;
  long step = 0;
  Real A = 0.0;                           // integral of D d^3r (D = R or rho): baryon number in the computational frame
  Real baryon_number = 0.0;               // integral of R d^3r
  Real raw_moment[kObsMaxMoment + 1] = {};  // <D^N> = (1/A) integral D^N D d^3r  [fm^-3N]
  Real moment[kObsMaxMoment + 1] = {};      // <D^N> / <D>^N  (index 0 unused = 1, index 1 = 1)
  Real u_perp = 0.0;                      // Eq. 14-16 of arXiv:1302.2956, weights D
  Real unstable_fraction = 0.0;           // baryon fraction (weights R) in cells where the mask is true
  Real eps_max_eps_s = 0.0;               // max over cells of eps / eps_s
};

struct PhaseHist {
  int n_rho = 0, n_eps = 0;
  Real d_rho = 0, d_eps = 0;              // in units of rho_s, eps_s
  std::vector<double> baryon;             // [ir * n_eps + ie], baryon number (weights R dV)
  double outside = 0.0;                   // baryon number outside the histogram range
  double total() const;                   // inside + outside
};

struct ClumpResult {
  std::vector<double> baryon;             // integral of R d^3r per clump
  std::vector<long> cells;
};

// ---------------------------------------------------------------------------------------------------- device reductions
struct ObsSums {
  Real d[kObsMaxMoment + 1] = {};  // sum of D^(m+1), m = 0..7
  Real baryon = 0, unstable = 0, sxx = 0, sxv = 0, emax = 0;
  KOKKOS_INLINE_FUNCTION ObsSums& operator+=(const ObsSums& o) {
    for (int m = 0; m <= kObsMaxMoment; ++m) d[m] += o.d[m];
    baryon += o.baryon; unstable += o.unstable; sxx += o.sxx; sxv += o.sxv;
    if (o.emax > emax) emax = o.emax;
    return *this;
  }
};

}  // namespace spf

namespace Kokkos {
template <>
struct reduction_identity<spf::ObsSums> {
  KOKKOS_FORCEINLINE_FUNCTION static spf::ObsSums sum() { return spf::ObsSums(); }
};
}  // namespace Kokkos

namespace spf {

// Region mask hook. A mask is a trivially copyable struct with  KOKKOS_INLINE_FUNCTION bool operator()(Real eps, Real rho) const
// (rest-frame values, device function). Vs2Mask is the only one implemented: mechanically unstable cells, vs2(eps, rho) < 0.
// A mask from a finite-temperature table (coexistence region, isothermal spinodal) plugs into compute_observables /
// compute_phase_plane in the same way; none is implemented here (WP2b).
template <class EoS>
struct Vs2Mask {
  EoS eos;
  KOKKOS_INLINE_FUNCTION bool operator()(Real eps, Real rho) const { return eos.vs2(eps, rho) < 0.0; }
};

namespace detail {

// Pre-pass: sum of D over the interior (the scale of the density; the power sums use D / mean(D), which keeps the rounding error of
// the N-th normalised moment at the level of N * machine epsilon for a nearly uniform density).
struct SumDK {
  Field3 R, rho;
  int rest;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k, Real& a) const { a += rest ? rho(i, j, k) : R(i, j, k); }
};

template <class Mask>
struct ObsReduceK {
  Mask mask;
  Real inv_scale;  // 1 / mean(D); the sums d[], sxx, sxv are of D * inv_scale
  Field3 R, rho, eps, vx, vy;
  int rest;
  int g0, g1, n0, n1;
  Real dx;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k, ObsSums& a) const {
    const Real r = R(i, j, k), rr = rho(i, j, k), e = eps(i, j, k);
    const Real D = (rest ? rr : r) * inv_scale;
    const Real x = (static_cast<Real>(i - g0) + 0.5 - 0.5 * n0) * dx;
    const Real y = (static_cast<Real>(j - g1) + 0.5 - 0.5 * n1) * dx;
    Real pw = D;
    for (int m = 0; m <= kObsMaxMoment; ++m) { a.d[m] += pw; pw *= D; }
    a.baryon += r;
    if (r > 0.0 && mask(e, rr)) a.unstable += r;
    a.sxx += (x * x + y * y) * D;
    a.sxv += (vx(i, j, k) * x + vy(i, j, k) * y) * D;
    if (e > a.emax) a.emax = e;
  }
};

template <class Mask>
struct PhaseK {
  Mask mask;  // unused for the histogram itself, kept so that a future per-region histogram can reuse the functor
  Field3 R, rho, eps;
  Kokkos::View<Real*> hist;  // n_rho * n_eps + 1 (last entry: outside the range)
  int n_rho, n_eps;
  Real inv_drho, inv_deps;  // 1 / (bin width in fm^-3, GeV/fm^3)
  Real dV;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
    const Real r = R(i, j, k);
    if (r == 0.0) return;
    const Real fr = rho(i, j, k) * inv_drho, fe = eps(i, j, k) * inv_deps;
    int idx = n_rho * n_eps;
    if (fr >= 0.0 && fe >= 0.0 && fr < static_cast<Real>(n_rho) && fe < static_cast<Real>(n_eps))
      idx = static_cast<int>(fr) * n_eps + static_cast<int>(fe);
    Kokkos::atomic_add(&hist(idx), r * dV);
  }
};

}  // namespace detail

template <class Mask>
ObsResult compute_observables(const ObsFields& f, const ObsParams& prm, const Mask& mask, Real time, long step) {
  const Grid& g = f.grid;
  const int rest = prm.rest_frame ? 1 : 0;
  Real sumD = 0.0;
  Kokkos::parallel_reduce("spf_observables_scale", interior_policy(g), detail::SumDK{f.R, f.rho, rest}, sumD);
  const Real ncell = static_cast<Real>(g.n[0]) * g.n[1] * g.n[2];
  const Real scale = sumD > 0.0 ? sumD / ncell : 1.0;  // D -> D / scale
  ObsSums s;
  Kokkos::parallel_reduce("spf_observables", interior_policy(g),
                          detail::ObsReduceK<Mask>{mask, 1.0 / scale, f.R, f.rho, f.eps, f.vx, f.vy, prm.rest_frame ? 1 : 0, g.g[0], g.g[1], g.n[0],
                                                   g.n[1], g.dx},
                          s);
  const Real dV = g.dx * g.dx * g.dx;
  ObsResult o;
  o.time = time;
  o.step = step;
  o.A = s.d[0] * scale * dV;
  o.baryon_number = s.baryon * dV;
  o.eps_max_eps_s = s.emax / prm.eps_s;
  if (s.d[0] != 0.0) {
    {
      Real sc = 1.0;
      for (int n = 1; n <= kObsMaxMoment; ++n) { sc *= scale; o.raw_moment[n] = s.d[n] / s.d[0] * sc; }
    }
    for (int n = 1; n <= kObsMaxMoment; ++n) {
      // normalised moment from the scaled sums: (d[n]/d[0]) / (d[1]/d[0])^n
      const Real m1 = s.d[1] / s.d[0];
      Real den = 1.0;
      for (int q = 0; q < n; ++q) den *= m1;
      o.moment[n] = den != 0.0 ? (s.d[n] / s.d[0]) / den : 0.0;
    }
    if (s.sxx > 0.0) o.u_perp = s.sxv / (s.d[0] * std::sqrt(s.sxx / s.d[0]));
  }
  o.unstable_fraction = s.baryon != 0.0 ? s.unstable / s.baryon : 0.0;
  return o;
}

// Baryon number (weights R dV) in the (rho / rho_s, eps / eps_s) plane, rest-frame rho and eps.
template <class Mask>
PhaseHist compute_phase_plane(const ObsFields& f, const ObsParams& prm, const Mask& mask) {
  const Grid& g = f.grid;
  Kokkos::View<Real*> h("spf_phase_hist", static_cast<size_t>(prm.n_rho) * prm.n_eps + 1);
  Kokkos::parallel_for("spf_phase_plane", interior_policy(g),
                       detail::PhaseK<Mask>{mask, f.R, f.rho, f.eps, h, prm.n_rho, prm.n_eps, 1.0 / (prm.d_rho * prm.rho_s),
                                            1.0 / (prm.d_eps * prm.eps_s), g.dx * g.dx * g.dx});
  auto hh = Kokkos::create_mirror_view(h);
  Kokkos::deep_copy(hh, h);
  PhaseHist p;
  p.n_rho = prm.n_rho; p.n_eps = prm.n_eps; p.d_rho = prm.d_rho; p.d_eps = prm.d_eps;
  p.baryon.assign(hh.data(), hh.data() + static_cast<size_t>(prm.n_rho) * prm.n_eps);
  p.outside = hh(static_cast<size_t>(prm.n_rho) * prm.n_eps);
  return p;
}

// ---------------------------------------------------------------------------------------------------- host side (src/observables.cpp)
// Clump finder: cells with D > threshold_density (D = R or rho, as chosen by prm.rest_frame; threshold = clump_rho_min_rho_s rho_s) are
// connected by 6- or 26-neighbour adjacency (prm.connectivity) with union-find on the host; with prm.periodic the box wraps, so a clump
// crossing a face is one clump. Baryon number per clump = sum of R dV.
ClumpResult find_clumps(const ObsFields& f, const ObsParams& prm);
// Histogram with bins k = 0..max (clump with round(B) = k; B < 0.5 counts in bin 0) and the overflow count of clumps with round(B) > max.
std::vector<long> clump_histogram(const std::vector<double>& baryon, int max_bin, long& overflow);

nlohmann::json clump_json(const ClumpResult& c, const ObsParams& prm, Real time, long step);
nlohmann::json phase_json(const PhaseHist& p, const ObsParams& prm, Real time, long step, Real unstable_fraction);

// Snapshot written by Hydro3D::write_snapshot: needs the fields R, rho, eps, vx, vy. Returns the views on the default memory space.
struct LoadedSnapshot {
  ObsFields fields;
  Real time = 0.0;
  long step = 0;
  nlohmann::json config;  // configuration echo of the run that wrote it
};
LoadedSnapshot load_snapshot_fields(const std::string& base);  // base with or without the .json / .bin suffix

// ---------------------------------------------------------------------------------------------------- recorder
// Appends one CSV line per recorded time to <dir>/observables.csv, writes the clump and phase-plane histograms (JSON) at the first record at
// or after each time of obs.hist_times_fm (files clumps_step<step>.json and phase_step<step>.json), echoes the configuration to <dir>.
// A record is made when step % obs.every_steps == 0 (on_step) or unconditionally (record).
template <class EoS>
class ObservableRecorder {
 public:
  ObservableRecorder(const Config& cfg, const EoS& eos, const std::string& dir);
  // Record the current state of the solver if step % every_steps == 0 (call before the first step and after each step).
  void on_step(Hydro3D<EoS>& h) {
    if (h.steps % prm_.every_steps == 0) record(fields_of(h), h.time, h.steps);
  }
  void record(const ObsFields& f, Real time, long step);
  // Writes observables_summary.json (fluid-end time, maximum enhancement); idempotent.
  void finalize();
  const ObsParams& params() const { return prm_; }
  const std::vector<ObsResult>& history() const { return hist_; }
  // First recorded time at which all cells have eps < fluid_end_eps_eps_s eps_s; NaN if not reached. Resolution: every_steps * dt.
  Real fluid_end_time() const { return fluid_end_time_; }
  const std::string& dir() const { return dir_; }

  // Observed fields of the solver: runs update_primitives() (restoring the recovery-failure counter, which the extra call would double count).
  static ObsFields fields_of(Hydro3D<EoS>& h) {
    const long nf = h.n_recover_fail_total;
    h.update_primitives();
    h.n_recover_fail_total = nf;
    return ObsFields{h.grid, h.U[0], h.rho, h.eps, h.vx, h.vy};
  }

 private:
  void write_hists(const ObsFields& f, const ObsResult& r);
  Vs2Mask<EoS> mask_;
  ObsParams prm_;
  std::string dir_;
  std::ofstream csv_;
  std::vector<ObsResult> hist_;
  std::vector<Real> pending_;
  Real fluid_end_time_ = std::numeric_limits<Real>::quiet_NaN();
  long fluid_end_step_ = -1;
};

// Post-processing of snapshot files with the same recorder (every record is made: obs.every_steps is not applied).
template <class EoS>
void observe_snapshots(const Config& cfg, const EoS& eos, const std::vector<std::string>& bases, const std::string& outdir);

std::string obs_csv_header();
std::string obs_csv_line(const ObsResult& r);

}  // namespace spf

#include "spf/observables_impl.hpp"
