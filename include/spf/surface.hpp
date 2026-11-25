#pragma once
// WP10a: switching (particlisation) surface of a fluid event.  File format: docs/particlisation_format.md.
//
//   isoenergy    (particlisation.surface.method = isoenergy, default) the 4D hypersurface eps = eps_sw between consecutive full time steps,
//                found per hypercube by surf::find_pieces (surface_geometry.hpp) on the device, appended to a compact buffer
//   isochronous  (cross-check) at the times particlisation.surface.isochronous_times_fm after t_init every cell with eps > 0
//                (isochronous_cells = all, default; as the isochronous Cooper-Frye integral of arXiv:1311.0999) or with eps >= eps_sw
//                (isochronous_cells = above_eps_sw) gives one element with dSigma = (dV, 0, 0, 0)
//   initial slice at t_init every cell with 0 < eps < eps_sw gives an element with dSigma = (dV, 0, 0, 0) and is set to vacuum
//
// Kernels are named functors (docs/device_rules.md, rule 2); the equation of state is called only inside kernels (rule 9).
#include "spf/config.hpp"
#include "spf/hydro3d.hpp"
#include "spf/surface_geometry.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace spf {

inline constexpr int kSurfRecord = 16;  // values per element in the .bin file

enum class SurfaceMethod { Isoenergy, Isochronous };
enum class IsochronousCells { All, AboveEpsSw };
enum class FirstLevel { BeforeRemoval, AfterRemoval };

struct ParticlisationParams {
  bool enabled = false;
  Real eps_sw_eps_s = 4.0;
  SurfaceMethod method = SurfaceMethod::Isoenergy;
  std::vector<Real> isochronous_times_fm;
  IsochronousCells iso_cells = IsochronousCells::All;
  FirstLevel first_level = FirstLevel::BeforeRemoval;
  std::string output_dir;
  Real eps_s = 0.94 * 0.153;  // m_N rho_s from constants.*

  Real eps_sw() const { return eps_sw_eps_s * eps_s; }

  static ParticlisationParams from_config(const Config& cfg) {
    ParticlisationParams p;
    p.enabled = cfg.get<bool>("particlisation.enabled");
    p.eps_sw_eps_s = cfg.get<Real>("particlisation.eps_sw_eps_s");
    p.eps_s = cfg.get<Real>("constants.m_N_GeV") * cfg.get<Real>("constants.rho_s_fm3");
    const std::string m = cfg.get<std::string>("particlisation.surface.method");
    if (m == "isoenergy") p.method = SurfaceMethod::Isoenergy;
    else if (m == "isochronous") p.method = SurfaceMethod::Isochronous;
    else throw ConfigError("particlisation.surface.method must be 'isoenergy' or 'isochronous', got '" + m + "'");
    p.isochronous_times_fm = cfg.get<std::vector<Real>>("particlisation.surface.isochronous_times_fm");
    const std::string ic = cfg.get<std::string>("particlisation.surface.isochronous_cells");
    if (ic == "all") p.iso_cells = IsochronousCells::All;
    else if (ic == "above_eps_sw") p.iso_cells = IsochronousCells::AboveEpsSw;
    else throw ConfigError("particlisation.surface.isochronous_cells must be 'all' or 'above_eps_sw', got '" + ic + "'");
    const std::string fl = cfg.get<std::string>("particlisation.surface.first_level");
    if (fl == "before_removal") p.first_level = FirstLevel::BeforeRemoval;
    else if (fl == "after_removal") p.first_level = FirstLevel::AfterRemoval;
    else throw ConfigError("particlisation.surface.first_level must be 'before_removal' or 'after_removal', got '" + fl + "'");
    p.output_dir = cfg.get<std::string>("particlisation.surface.output_dir");
    if (!(p.eps_sw_eps_s > 0.0)) throw ConfigError("particlisation.eps_sw_eps_s must be positive");
    if (p.method == SurfaceMethod::Isochronous && p.isochronous_times_fm.empty())
      throw ConfigError("particlisation.surface.method = isochronous needs particlisation.surface.isochronous_times_fm");
    return p;
  }
};

// Integrals over surface elements (GeV, GeV, GeV, GeV, baryon number): int T^{0 mu} dSigma_mu, int T^{i mu} dSigma_mu, int N^mu dSigma_mu,
// with T^{mu nu} = (eps + p0) u^mu u^nu - p0 g^{mu nu}, N^mu = rho u^mu, p0 = eos.p0(eps, rho) of the stored eps, rho.
// E_neg, B_neg: the same restricted to elements with u^mu dSigma_mu < 0 (inflow).
struct SurfFlux {
  Real E = 0, Px = 0, Py = 0, Pz = 0, B = 0, E_neg = 0, B_neg = 0;
  KOKKOS_INLINE_FUNCTION SurfFlux& operator+=(const SurfFlux& o) {
    E += o.E; Px += o.Px; Py += o.Py; Pz += o.Pz; B += o.B; E_neg += o.E_neg; B_neg += o.B_neg;
    return *this;
  }
  SurfFlux operator+(const SurfFlux& o) const { SurfFlux r = *this; r += o; return r; }
};

// Cells with eps >= eps_sw (n_above), those on the outermost interior layer (n_edge: the surface would be cut by the box), max eps.
struct SurfStatus {
  long n_above = 0, n_edge = 0;
  Real eps_max = 0;
  KOKKOS_INLINE_FUNCTION SurfStatus& operator+=(const SurfStatus& o) {
    n_above += o.n_above; n_edge += o.n_edge;
    if (o.eps_max > eps_max) eps_max = o.eps_max;
    return *this;
  }
};

// Conserved totals (sums of cell values, multiply by dV) of the cells selected by eps >= eps_sw.
struct Sums5 {
  Real E = 0, Mx = 0, My = 0, Mz = 0, R = 0;
  KOKKOS_INLINE_FUNCTION Sums5& operator+=(const Sums5& o) {
    E += o.E; Mx += o.Mx; My += o.My; Mz += o.Mz; R += o.R;
    return *this;
  }
};

}  // namespace spf

namespace Kokkos {
template <>
struct reduction_identity<spf::SurfFlux> {
  KOKKOS_FORCEINLINE_FUNCTION static spf::SurfFlux sum() { return spf::SurfFlux(); }
};
template <>
struct reduction_identity<spf::SurfStatus> {
  KOKKOS_FORCEINLINE_FUNCTION static spf::SurfStatus sum() { return spf::SurfStatus(); }
};
template <>
struct reduction_identity<spf::Sums5> {
  KOKKOS_FORCEINLINE_FUNCTION static spf::Sums5 sum() { return spf::Sums5(); }
};
}  // namespace Kokkos

namespace spf {

using SurfBuffer = Kokkos::View<Real* [kSurfRecord], Kokkos::LayoutRight>;  // rows contiguous on every backend

namespace detail {

template <class EoS>
KOKKOS_INLINE_FUNCTION void eos_T_mu(const EoS& eos, Real eps, Real rho, Real& T, Real& mu) {
  if constexpr (requires(const EoS& e, Real a, Real b) {
                  e.T(a, b);
                  e.mu(a, b);
                }) {
    T = eos.T(eps, rho);
    mu = eos.mu(eps, rho);
  } else {
    T = 0.0;
    mu = 0.0;
  }
}

template <class EoS>
constexpr bool eos_has_T_mu() {
  return requires(const EoS& e, Real a, Real b) {
    e.T(a, b);
    e.mu(a, b);
  };
}

// One time level of the surface finder: eps, rho and the spatial components u^i = gamma v^i of the last update_primitives().
struct StoreLevelK {
  Field3 eps, rho, vx, vy, vz;
  Field3 le, lr, lx, ly, lz;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
    const Real a = vx(i, j, k), b = vy(i, j, k), c = vz(i, j, k);
    const Real v2 = a * a + b * b + c * c;
    const Real g = v2 < 1.0 ? 1.0 / Kokkos::sqrt(1.0 - v2) : 0.0;
    le(i, j, k) = eps(i, j, k);
    lr(i, j, k) = rho(i, j, k);
    lx(i, j, k) = g * a;
    ly(i, j, k) = g * b;
    lz(i, j, k) = g * c;
  }
};

// Common record writer: 16 values at row r of the buffer.
template <class EoS>
KOKKOS_INLINE_FUNCTION void write_record(const EoS& eos, const SurfBuffer& buf, long r, const Real x[4], const Real n[4], Real ux, Real uy,
                                         Real uz, Real e, Real rh) {
  Real T, mu;
  eos_T_mu(eos, e, rh, T, mu);
  const Real u0 = Kokkos::sqrt(1.0 + ux * ux + uy * uy + uz * uz);
  const Real v[kSurfRecord] = {x[0], x[1], x[2], x[3], n[0], n[1], n[2], n[3], u0, ux, uy, uz, e, rh, T, mu};
  for (int q = 0; q < kSurfRecord; ++q) buf(r, q) = v[q];
}

// Isoenergy surface between two levels.  Iterates over the hypercubes whose lower spatial corner is interior cell (i, j, k) with
// i < g + n - 1 (cubes between interior cell centres).  errs(0): hypercubes with inconsistent topology, errs(1): pieces dropped
// (more than kMaxPieces), errs(2): records beyond the buffer capacity (the caller enlarges the buffer and repeats the step).
template <class EoS>
struct SurfaceIsoK {
  EoS eos;
  Field3 e0, r0, x0, y0, z0, e1, r1, x1, y1, z1;
  Real eps_sw, t0, dt, dx;
  Real org[3];  // physical position of interior cell (0, 0, 0)
  int g[3];
  SurfBuffer buf;
  Kokkos::View<long> count;
  Kokkos::View<long[3]> errs;
  long cap;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
    Real f[16];
    Real fmin = 1e300, fmax = -1e300;
    for (int c = 0; c < 16; ++c) {
      const int bt = c & 1, bx = (c >> 1) & 1, by = (c >> 2) & 1, bz = (c >> 3) & 1;
      const Real e = bt ? e1(i + bx, j + by, k + bz) : e0(i + bx, j + by, k + bz);
      f[c] = e - eps_sw;
      fmin = f[c] < fmin ? f[c] : fmin;
      fmax = f[c] > fmax ? f[c] : fmax;
    }
    if (fmin >= 0.0 || fmax < 0.0) return;
    const Real L[4] = {dt, dx, dx, dx};
    surf::Piece pc[surf::kMaxPieces];
    int np = surf::find_pieces(f, L, pc, surf::kMaxPieces);
    if (np == -1) { Kokkos::atomic_add(&errs(0), 1L); return; }
    if (np == -2) { Kokkos::atomic_add(&errs(1), 1L); np = surf::kMaxPieces; }
    if (np <= 0) return;
    const long base = Kokkos::atomic_fetch_add(&count(), static_cast<long>(np));
    Real ve[16], vr[16], vx[16], vy[16], vz[16];
    for (int c = 0; c < 16; ++c) {
      const int bt = c & 1, ii = i + ((c >> 1) & 1), jj = j + ((c >> 2) & 1), kk = k + ((c >> 3) & 1);
      ve[c] = bt ? e1(ii, jj, kk) : e0(ii, jj, kk);
      vr[c] = bt ? r1(ii, jj, kk) : r0(ii, jj, kk);
      vx[c] = bt ? x1(ii, jj, kk) : x0(ii, jj, kk);
      vy[c] = bt ? y1(ii, jj, kk) : y0(ii, jj, kk);
      vz[c] = bt ? z1(ii, jj, kk) : z0(ii, jj, kk);
    }
    for (int p = 0; p < np; ++p) {
      const long r = base + p;
      if (r >= cap) { Kokkos::atomic_add(&errs(2), 1L); continue; }
      const Real* xc = pc[p].xc;
      const Real X[4] = {t0 + xc[0] * dt, org[0] + (i - g[0] + xc[1]) * dx, org[1] + (j - g[1] + xc[2]) * dx, org[2] + (k - g[2] + xc[3]) * dx};
      write_record(eos, buf, r, X, pc[p].n, surf::interp16(vx, xc), surf::interp16(vy, xc), surf::interp16(vz, xc), surf::interp16(ve, xc),
                   surf::interp16(vr, xc));
    }
  }
};

// One element per selected cell with dSigma = (dV, 0, 0, 0) at time t (initial slice: lo < eps < hi with lo = 0, hi = eps_sw, and the
// cell is flagged for removal; isochronous: eps > 0 or eps >= eps_sw).
template <class EoS>
struct CellElementsK {
  EoS eos;
  Field3 eps, rho, vx, vy, vz;
  Real lo, hi;  // select lo < eps < hi
  int hi_inclusive_above;  // 1: select eps >= lo instead (isochronous above_eps_sw)
  Real t, dx;
  Real org[3];
  int g[3];
  SurfBuffer buf;
  Kokkos::View<long> count;
  Kokkos::View<long[3]> errs;
  long cap;
  Kokkos::View<int***> flag;  // set to 1 for selected cells if flag_cells
  int flag_cells;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
    const Real e = eps(i, j, k);
    const bool sel = hi_inclusive_above ? (e >= lo) : (e > lo && e < hi);
    if (!sel) return;
    if (flag_cells) flag(i, j, k) = 1;
    const long r = Kokkos::atomic_fetch_add(&count(), 1L);
    if (r >= cap) { Kokkos::atomic_add(&errs(2), 1L); return; }
    const Real a = vx(i, j, k), b = vy(i, j, k), c = vz(i, j, k);
    const Real v2 = a * a + b * b + c * c;
    const Real gm = 1.0 / Kokkos::sqrt(1.0 - v2);
    const Real X[4] = {t, org[0] + (i - g[0]) * dx, org[1] + (j - g[1]) * dx, org[2] + (k - g[2]) * dx};
    const Real n[4] = {dx * dx * dx, 0.0, 0.0, 0.0};
    write_record(eos, buf, r, X, n, gm * a, gm * b, gm * c, e, rho(i, j, k));
  }
};

struct RemoveFlaggedK {
  Kokkos::View<int***> flag;
  Field3 R, Mx, My, Mz, E;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
    if (!flag(i, j, k)) return;
    R(i, j, k) = 0.0; Mx(i, j, k) = 0.0; My(i, j, k) = 0.0; Mz(i, j, k) = 0.0; E(i, j, k) = 0.0;
  }
};

template <class EoS>
struct SurfFluxK {
  EoS eos;
  SurfBuffer buf;
  KOKKOS_INLINE_FUNCTION void operator()(const long r, SurfFlux& a) const {
    const Real n0 = buf(r, 4), n1 = buf(r, 5), n2 = buf(r, 6), n3 = buf(r, 7);
    const Real u0 = buf(r, 8), u1 = buf(r, 9), u2 = buf(r, 10), u3 = buf(r, 11);
    const Real e = buf(r, 12), rh = buf(r, 13);
    const Real p = eos.p0(e, rh);
    const Real ud = u0 * n0 + u1 * n1 + u2 * n2 + u3 * n3;  // u^mu dSigma_mu
    const Real w = (e + p) * ud;
    // T^{nu mu} dSigma_mu = (e + p) u^nu (u.dSigma) - p g^{nu mu} dSigma_mu, g^{00} = 1, g^{ii} = -1
    const Real E = w * u0 - p * n0;
    a.E += E;
    a.Px += w * u1 + p * n1;
    a.Py += w * u2 + p * n2;
    a.Pz += w * u3 + p * n3;
    a.B += rh * ud;
    if (ud < 0.0) { a.E_neg += E; a.B_neg += rh * ud; }
  }
};

struct SurfStatusK {
  Field3 eps;
  Real eps_sw;
  int g[3], n[3];
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k, SurfStatus& a) const {
    const Real e = eps(i, j, k);
    if (e > a.eps_max) a.eps_max = e;
    if (e >= eps_sw) {
      a.n_above += 1;
      if (i == g[0] || j == g[1] || k == g[2] || i == g[0] + n[0] - 1 || j == g[1] + n[1] - 1 || k == g[2] + n[2] - 1) a.n_edge += 1;
    }
  }
};

struct InsideSumsK {
  Field3 eps, R, Mx, My, Mz, E;
  Real eps_sw;
  int all;  // 1: every interior cell
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k, Sums5& a) const {
    if (!all && !(eps(i, j, k) >= eps_sw)) return;
    a.E += E(i, j, k); a.Mx += Mx(i, j, k); a.My += My(i, j, k); a.Mz += Mz(i, j, k); a.R += R(i, j, k);
  }
};

}  // namespace detail

// Host side: one surface (or, for the isochronous method, one surface per listed time) of one fluid event.
//   begin(h)       at t_init (state just set): records the initial totals, emits the initial slice and removes those cells, stores the first level
//   after_step(h)  after every full step: emits the isoenergy elements between the previous and the current level (or the isochronous
//                  elements at a listed time); returns the status of the current level
//   finish(h)      totals left inside, JSON sidecar(s)
// Elements of one call are sorted on the host (lexicographic in the 16 values) before they are appended to the file, so the files do not depend
// on the order of the atomic appends.
template <class EoS>
class SurfaceFinder {
 public:
  struct Meta {
    nlohmann::json json;  // event id, E_lab, EoS model, table checksum, ... copied into the sidecar
    Real t_init = 0.0;    // absolute time of the fluid start (element times are t_init + fluid time)
    std::array<Real, 3> origin{0, 0, 0};  // physical position of interior cell (0, 0, 0)
  };

  SurfaceFinder(const EoS& eos, const ParticlisationParams& prm, const Grid& grid, const std::string& base, const Meta& meta)
      : eos_(eos), prm_(prm), grid_(grid), base_(base), meta_(meta) {
    for (int l = 0; l < 2; ++l)
      for (int c = 0; c < 5; ++c) lev_[l][c] = grid.make_field("surf_level");
    count_ = Kokkos::View<long>("surf_count");
    errs_ = Kokkos::View<long[3]>("surf_errs");
    cap_ = 1 << 16;
    buf_ = SurfBuffer("surf_buf", cap_);
    if (prm_.method == SurfaceMethod::Isochronous) {
      iso_done_.assign(prm_.isochronous_times_fm.size(), false);
      for (size_t q = 0; q < prm_.isochronous_times_fm.size(); ++q) outs_.push_back(Out{iso_base(q)});
    } else {
      outs_.push_back(Out{base_});
    }
    for (auto& o : outs_) {
      if (base_.empty()) continue;  // no files (tests)
      std::filesystem::path p(o.base + ".surface.bin");
      if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
      o.f.open(o.base + ".surface.bin", std::ios::binary | std::ios::trunc);
      if (!o.f) throw std::runtime_error("cannot open " + o.base + ".surface.bin");
    }
  }

  Real eps_sw() const { return prm_.eps_sw(); }
  const ParticlisationParams& params() const { return prm_; }
  bool closed() const { return closed_; }
  long n_elements_total() const { return outs_.empty() ? 0 : outs_[0].n_initial + outs_[0].n_surface; }
  // Totals (initial, surface, inside) of output q.
  const nlohmann::json& summary(size_t q = 0) const { return outs_.at(q).summary; }
  size_t n_outputs() const { return outs_.size(); }
  const std::string& output_base(size_t q = 0) const { return outs_.at(q).base; }
  double wall_s() const { return wall_; }
  long n_topology_errors() const { return err_total_[0]; }
  long n_pieces_dropped() const { return err_total_[1]; }
  long n_edge_cells_max() const { return n_edge_max_; }
  SurfStatus last_status() const { return status_; }

  void begin(Hydro3D<EoS>& h) {
    const auto t0 = clock_();
    recover(h);
    const Real dV = grid_.dx * grid_.dx * grid_.dx;
    {
      Sums5 s;
      Kokkos::parallel_reduce("surf_initial_totals", interior_policy(grid_),
                              detail::InsideSumsK{h.eps, h.U[0], h.U[1], h.U[2], h.U[3], h.U[4], prm_.eps_sw(), 1}, s);
      init_ = scale(s, dV);
    }
    if (prm_.first_level == FirstLevel::BeforeRemoval) store_level(h, 0);
    // initial slice
    Kokkos::View<int***> flag("surf_flag", grid_.size(0), grid_.size(1), grid_.size(2));
    auto launch = [&]() {
      detail::CellElementsK<EoS> k{eos_, h.eps, h.rho, h.vx, h.vy, h.vz, 0.0, prm_.eps_sw(), 0, meta_.t_init, grid_.dx,
                                   {meta_.origin[0], meta_.origin[1], meta_.origin[2]}, {grid_.g[0], grid_.g[1], grid_.g[2]},
                                   buf_, count_, errs_, cap_, flag, 1};
      Kokkos::parallel_for("surf_initial_slice", interior_policy(grid_), k);
    };
    const long n0 = run_with_capacity(launch);
    const SurfFlux fl = flux(n0);
    std::vector<Real> host = copy_sorted(n0);
    for (auto& o : outs_) {
      write(o, host);
      o.n_initial = n0;
      o.flux_initial = fl;
    }
    Kokkos::parallel_for("surf_remove", interior_policy(grid_), detail::RemoveFlaggedK{flag, h.U[0], h.U[1], h.U[2], h.U[3], h.U[4]});
    Kokkos::fence();
    recover(h);
    if (prm_.first_level == FirstLevel::AfterRemoval) store_level(h, 0);
    status_ = status(h);
    n_edge_max_ = std::max(n_edge_max_, status_.n_edge);
    t_prev_ = meta_.t_init + h.time;
    started_ = true;
    wall_ += seconds_(t0);
  }

  // Returns true when the surface is complete (isoenergy: no cell at or above eps_sw; isochronous: every listed time done).
  bool after_step(Hydro3D<EoS>& h) {
    if (!started_) throw std::logic_error("SurfaceFinder::after_step before begin");
    const auto t0 = clock_();
    recover(h);
    const Real t_now = meta_.t_init + h.time;
    if (prm_.method == SurfaceMethod::Isoenergy) {
      if (!closed_) {
        store_level(h, 1);
        const long n = iso_run(lev_[0], lev_[1], t_prev_, t_now - t_prev_);
        const SurfFlux fl = flux(n);
        write(outs_[0], copy_sorted(n));
        outs_[0].n_surface += n;
        outs_[0].flux_surface += fl;
        std::swap(lev_[0], lev_[1]);
      }
    } else {
      for (size_t q = 0; q < prm_.isochronous_times_fm.size(); ++q) {
        if (iso_done_[q]) continue;
        // the first full step at or after the listed time (time after t_init), within rounding of the step sum
        if (h.time + 1e-9 < prm_.isochronous_times_fm[q]) continue;
        auto launch = [&]() {
          const bool above = prm_.iso_cells == IsochronousCells::AboveEpsSw;
          detail::CellElementsK<EoS> k{eos_, h.eps, h.rho, h.vx, h.vy, h.vz, above ? prm_.eps_sw() : 0.0, 1e300, above ? 1 : 0, t_now,
                                       grid_.dx, {meta_.origin[0], meta_.origin[1], meta_.origin[2]}, {grid_.g[0], grid_.g[1], grid_.g[2]},
                                       buf_, count_, errs_, cap_, Kokkos::View<int***>(), 0};
          Kokkos::parallel_for("surf_isochronous", interior_policy(grid_), k);
        };
        const long n = run_with_capacity(launch);
        const SurfFlux fl = flux(n);
        write(outs_[q], copy_sorted(n));
        outs_[q].n_surface += n;
        outs_[q].flux_surface += fl;
        outs_[q].t_iso = t_now;
        // everything at or above eps_sw (all: every cell) is converted here: nothing is left inside; with above_eps_sw the cells with
        // 0 < eps < eps_sw at that time are not emitted (reported as totals_not_emitted)
        outs_[q].inside_set = true;
        if (prm_.iso_cells == IsochronousCells::AboveEpsSw) {
          const Real dV = grid_.dx * grid_.dx * grid_.dx;
          SurfFlux all = scale(all_sums(h), dV), ab = scale(inside_sums(h), dV);
          outs_[q].not_emitted = SurfFlux{all.E - ab.E, all.Px - ab.Px, all.Py - ab.Py, all.Pz - ab.Pz, all.B - ab.B, 0, 0};
        }
        iso_done_[q] = true;
      }
    }
    status_ = status(h);
    n_edge_max_ = std::max(n_edge_max_, status_.n_edge);
    t_prev_ = t_now;
    if (prm_.method == SurfaceMethod::Isoenergy) {
      if (status_.n_above == 0) closed_ = true;
    } else {
      closed_ = std::all_of(iso_done_.begin(), iso_done_.end(), [](bool b) { return b; });
    }
    wall_ += seconds_(t0);
    return closed_;
  }

  // Isoenergy elements between two prescribed levels (eps, rho, u^x, u^y, u^z on this grid, ghost layers included in the extents; level a at
  // absolute time t0, level b at t0 + dt), sorted; nothing is written to a file.  Used by the tests.
  std::vector<Real> elements_between(const Field3 a[5], const Field3 b[5], Real t0, Real dt, SurfFlux* fl = nullptr) {
    const long n = iso_run(a, b, t0, dt);
    if (fl) *fl = flux(n);
    return copy_sorted(n);
  }

  // Writes the sidecar(s).  extra: fields added to every sidecar (end status, step counts, ...).
  void finish(Hydro3D<EoS>& h, const nlohmann::json& extra = nlohmann::json::object()) {
    recover(h);
    const Real dV = grid_.dx * grid_.dx * grid_.dx;
    const SurfFlux in_end = scale(inside_sums(h), dV, true);
    for (size_t q = 0; q < outs_.size(); ++q) {
      Out& o = outs_[q];
      o.f.close();
      const SurfFlux inside = o.inside_set ? o.inside : in_end;
      const SurfFlux tot = o.flux_initial + o.flux_surface;
      nlohmann::json J = meta_.json;
      J["format_version"] = 1;
      J["byte_order"] = "little";
      J["dtype"] = "float64";
      J["values_per_element"] = kSurfRecord;
      J["columns"] = {"t", "x", "y", "z", "dsig_0", "dsig_1", "dsig_2", "dsig_3", "u0", "u1", "u2", "u3", "eps", "rho", "T", "mu_B"};
      J["units"] = {{"t", "fm"}, {"x", "fm"}, {"dsig", "fm^3"}, {"u", "1"}, {"eps", "GeV/fm^3"}, {"rho", "fm^-3"}, {"T", "GeV"}, {"mu_B", "GeV"}};
      J["n_elements"] = o.n_initial + o.n_surface;
      J["n_elements_initial_slice"] = o.n_initial;
      J["n_elements_surface"] = o.n_surface;
      J["eps_sw_GeV_fm3"] = prm_.eps_sw();
      J["eps_sw_eps_s"] = prm_.eps_sw_eps_s;
      J["method"] = prm_.method == SurfaceMethod::Isoenergy ? "isoenergy" : "isochronous";
      J["methods_in_file"] = {{"initial_slice", o.n_initial},
                              {prm_.method == SurfaceMethod::Isoenergy ? "isoenergy" : "isochronous", o.n_surface}};
      if (prm_.method == SurfaceMethod::Isochronous) {
        J["isochronous_time_after_t_init_fm"] = prm_.isochronous_times_fm[q];
        J["isochronous_time_fm"] = o.t_iso;
        J["isochronous_cells"] = prm_.iso_cells == IsochronousCells::All ? "all" : "above_eps_sw";
      }
      J["first_level"] = prm_.first_level == FirstLevel::BeforeRemoval ? "before_removal" : "after_removal";
      J["dx_fm"] = grid_.dx;
      J["dt_fm"] = h.dt();
      J["grid_shape"] = {grid_.n[0], grid_.n[1], grid_.n[2]};
      J["origin_fm"] = {meta_.origin[0], meta_.origin[1], meta_.origin[2]};
      J["t_init_fm"] = meta_.t_init;
      J["T_mu_from_eos"] = detail::eos_has_T_mu<EoS>();
      auto fj = [](const SurfFlux& s) {
        return nlohmann::json{{"E_GeV", s.E}, {"Px_GeV", s.Px}, {"Py_GeV", s.Py}, {"Pz_GeV", s.Pz}, {"B", s.B}};
      };
      J["totals_initial_on_grid"] = fj(init_);
      J["totals_initial_slice"] = fj(o.flux_initial);
      J["totals_surface"] = fj(o.flux_surface);
      J["totals_all_elements"] = fj(tot);
      J["inflow_part_of_totals"] = {{"E_GeV", o.flux_initial.E_neg + o.flux_surface.E_neg}, {"B", o.flux_initial.B_neg + o.flux_surface.B_neg}};
      J["totals_inside_fluid_at_end"] = fj(inside);
      if (prm_.method == SurfaceMethod::Isochronous && prm_.iso_cells == IsochronousCells::AboveEpsSw) J["totals_not_emitted"] = fj(o.not_emitted);
      const SurfFlux out_plus_in = tot + inside;
      auto rel = [](Real a, Real b) { return b != 0.0 ? (a - b) / b : a; };
      J["closure"] = {{"definition", "(elements + inside at end - initial on grid) / initial on grid; momenta relative to the initial energy"},
                      {"E", rel(out_plus_in.E, init_.E)}, {"B", rel(out_plus_in.B, init_.B)},
                      {"Px", init_.E != 0.0 ? (out_plus_in.Px - init_.Px) / init_.E : 0.0},
                      {"Py", init_.E != 0.0 ? (out_plus_in.Py - init_.Py) / init_.E : 0.0},
                      {"Pz", init_.E != 0.0 ? (out_plus_in.Pz - init_.Pz) / init_.E : 0.0}};
      J["closed"] = closed_;
      J["diagnostics"] = {{"hypercubes_with_inconsistent_topology", err_total_[0]}, {"pieces_dropped", err_total_[1]},
                          {"max_cells_above_eps_sw_on_box_edge", n_edge_max_}, {"buffer_capacity_final", cap_},
                          {"wall_surface_s", wall_}};
      for (auto it = extra.begin(); it != extra.end(); ++it) J[it.key()] = it.value();
      o.summary = J;
      if (!base_.empty()) std::ofstream(o.base + ".surface.json") << J.dump(2) << "\n";
    }
  }

  // Closure of output q (relative error of E and B), from the summary written by finish().
  Real closure_E(size_t q = 0) const { return outs_.at(q).summary["closure"]["E"].template get<Real>(); }
  Real closure_B(size_t q = 0) const { return outs_.at(q).summary["closure"]["B"].template get<Real>(); }

 private:
  struct Out {
    std::string base;
    std::ofstream f;
    long n_initial = 0, n_surface = 0;
    SurfFlux flux_initial, flux_surface, inside, not_emitted;
    bool inside_set = false;
    Real t_iso = 0.0;
    nlohmann::json summary;
  };

  EoS eos_;
  ParticlisationParams prm_;
  Grid grid_;
  std::string base_;
  Meta meta_;
  Field3 lev_[2][5];
  SurfBuffer buf_;
  Kokkos::View<long> count_;
  Kokkos::View<long[3]> errs_;
  long cap_;
  std::vector<Out> outs_;
  std::vector<bool> iso_done_;
  SurfFlux init_;
  SurfStatus status_;
  Real t_prev_ = 0.0;
  bool started_ = false, closed_ = false;
  long err_total_[2] = {0, 0};
  long n_edge_max_ = 0;
  double wall_ = 0.0;

  static std::chrono::steady_clock::time_point clock_() { return std::chrono::steady_clock::now(); }
  static double seconds_(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }

  std::string iso_base(size_t q) const {
    char b[64];
    std::snprintf(b, sizeof b, ".iso_%.4gfm", prm_.isochronous_times_fm[q]);
    return base_ + b;
  }

  // primitives of the current state; the recovery-failure counters of the solver are restored (an extra call would double count)
  void recover(Hydro3D<EoS>& h) {
    const long nf = h.n_recover_fail_total, nfl = h.n_pex_floor_total;
    h.update_primitives();
    h.n_recover_fail_total = nf;
    h.n_pex_floor_total = nfl;
  }

  void store_level(Hydro3D<EoS>& h, int l) {
    Kokkos::parallel_for("surf_store_level", interior_policy(grid_),
                         detail::StoreLevelK{h.eps, h.rho, h.vx, h.vy, h.vz, lev_[l][0], lev_[l][1], lev_[l][2], lev_[l][3], lev_[l][4]});
  }

  // Run a kernel that appends to the buffer; enlarge the buffer and repeat if it overflowed.  Returns the number of records.
  template <class F>
  long run_with_capacity(F&& launch) {
    for (int attempt = 0; attempt < 3; ++attempt) {
      Kokkos::deep_copy(count_, 0L);
      Kokkos::deep_copy(errs_, 0L);
      launch();
      Kokkos::fence();
      long n = 0;
      Kokkos::deep_copy(n, count_);
      auto he = Kokkos::create_mirror_view(errs_);
      Kokkos::deep_copy(he, errs_);
      if (n <= cap_) {
        err_total_[0] += he(0);
        err_total_[1] += he(1);
        return n;
      }
      cap_ = static_cast<long>(1.25 * n) + 1024;
      buf_ = SurfBuffer("surf_buf", cap_);
    }
    throw std::runtime_error("SurfaceFinder: buffer capacity not reached after enlarging");
  }

  long iso_run(const Field3 a[5], const Field3 b[5], Real t0, Real dt) {
    auto launch = [&]() {
      detail::SurfaceIsoK<EoS> k{eos_, a[0], a[1], a[2], a[3], a[4], b[0], b[1], b[2], b[3], b[4], prm_.eps_sw(), t0, dt, grid_.dx,
                                 {meta_.origin[0], meta_.origin[1], meta_.origin[2]}, {grid_.g[0], grid_.g[1], grid_.g[2]},
                                 buf_, count_, errs_, cap_};
      const int lo[3] = {grid_.g[0], grid_.g[1], grid_.g[2]};
      const int hi[3] = {grid_.g[0] + grid_.n[0] - 1, grid_.g[1] + grid_.n[1] - 1, grid_.g[2] + grid_.n[2] - 1};
      Kokkos::parallel_for("surf_isoenergy", make_policy(lo, hi), k);
    };
    return run_with_capacity(launch);
  }

  SurfFlux flux(long n) {
    SurfFlux s;
    if (n > 0)
      Kokkos::parallel_reduce("surf_flux", Kokkos::RangePolicy<ExecSpace>(0, n), detail::SurfFluxK<EoS>{eos_, buf_}, s);
    return s;
  }

  std::vector<Real> copy_sorted(long n) {
    std::vector<Real> out(static_cast<size_t>(n) * kSurfRecord);
    if (n == 0) return out;
    auto sub = Kokkos::subview(buf_, std::make_pair(0L, n), Kokkos::ALL);
    auto hm = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sub);
    std::vector<long> idx(static_cast<size_t>(n));
    for (long r = 0; r < n; ++r) idx[r] = r;
    std::sort(idx.begin(), idx.end(), [&](long a, long b) {
      for (int q = 0; q < kSurfRecord; ++q)
        if (hm(a, q) != hm(b, q)) return hm(a, q) < hm(b, q);
      return false;
    });
    for (long r = 0; r < n; ++r)
      for (int q = 0; q < kSurfRecord; ++q) out[static_cast<size_t>(r) * kSurfRecord + q] = hm(idx[r], q);
    return out;
  }

  static void write(Out& o, const std::vector<Real>& v) {
    if (!o.f.is_open()) return;
    static_assert(std::endian::native == std::endian::little, "surface files are little-endian");
    o.f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(Real)));
    if (!o.f) throw std::runtime_error("write error on " + o.base + ".surface.bin");
  }

  SurfStatus status(Hydro3D<EoS>& h) const {
    SurfStatus s;
    Kokkos::parallel_reduce("surf_status", interior_policy(grid_),
                            detail::SurfStatusK{h.eps, prm_.eps_sw(), {grid_.g[0], grid_.g[1], grid_.g[2]}, {grid_.n[0], grid_.n[1], grid_.n[2]}}, s);
    return s;
  }

  Sums5 inside_sums(Hydro3D<EoS>& h) const {
    Sums5 s;
    Kokkos::parallel_reduce("surf_inside", interior_policy(grid_),
                            detail::InsideSumsK{h.eps, h.U[0], h.U[1], h.U[2], h.U[3], h.U[4], prm_.eps_sw(), 0}, s);
    return s;
  }

  Sums5 all_sums(Hydro3D<EoS>& h) const {
    Sums5 s;
    Kokkos::parallel_reduce("surf_all", interior_policy(grid_),
                            detail::InsideSumsK{h.eps, h.U[0], h.U[1], h.U[2], h.U[3], h.U[4], prm_.eps_sw(), 1}, s);
    return s;
  }

  static SurfFlux scale(const Sums5& s, Real dV, bool = false) {
    SurfFlux f;
    f.E = s.E * dV; f.Px = s.Mx * dV; f.Py = s.My * dV; f.Pz = s.Mz * dV; f.B = s.R * dV;
    return f;
  }
};

}  // namespace spf
