#pragma once
// Three-dimensional relativistic ideal fluid: dimensional splitting of the SHASTA sweeps of shasta.hpp, half-step method per sweep
// (Rischke, Bernard, Maruhn, Nucl. Phys. A 595, 346 (1995), arXiv:nucl-th/9504018, Sect. 3). Sequence of operations: docs/equations.md, Sect. 7.
// Conserved fields R = gamma rho, M_i = (eps + p) gamma^2 v_i, E = (eps + p) gamma^2 - p; p = p0(eps, rho) + p_extra.
// The solver is a template on the equation-of-state type and uses only  eos.p0(eps, rho).
// Kernels are named functors (docs/device_rules.md, rule 2).
#include "spf/config.hpp"
#include "spf/gradient.hpp"
#include "spf/shasta.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace spf {

inline constexpr int kRecover3MaxIter = 60;
inline constexpr Real kRecover3Tol = 1e-12;

enum class SweepOrder { Permuted, Fixed };

struct Hydro3DParams {
  ShastaParams shasta;
  GradientParams gradient;  // finite-range pressure (gradient.hpp); form = off unless set from a configuration
  SweepOrder order = SweepOrder::Permuted;
  int nx = 64, ny = 64, nz = 64;
  Real dx = 0.2;  // fm
  long snapshot_every = 0;

  static Hydro3DParams from_config(const Config& cfg) {
    Hydro3DParams p;
    p.shasta = ShastaParams::from_config(cfg);
    p.gradient = GradientParams::from_config(cfg);
    const std::string so = cfg.get<std::string>("shasta.sweep_order");
    if (so == "permuted") p.order = SweepOrder::Permuted;
    else if (so == "fixed") p.order = SweepOrder::Fixed;
    else throw ConfigError("shasta.sweep_order must be 'permuted' or 'fixed', got '" + so + "'");
    const GridCells gc = grid_cells(cfg);  // grid.nx/ny/nz, or grid.Lx_fm/Ly_fm/Lz_fm divided by grid.dx_fm
    p.nx = gc.n[0];
    p.ny = gc.n[1];
    p.nz = gc.n[2];
    p.dx = gc.dx;
    p.snapshot_every = cfg.get<long>("output.snapshot_every");
    return p;
  }
};

// Primitive recovery for one cell from (R, |M|, E) with an additive extra pressure p_extra (fixed during the iteration).
// Solves v = |M| / (E + p0(E - |M| v, R sqrt(1 - v^2)) + p_extra) on [0, 1) with the bracketed regula-falsi (Illinois)
// iteration of recover_primitives (hydro1d.hpp), at most kRecover3MaxIter iterations. Output: speed c = |v| >= 0, eps, rho, p
// (total pressure p0 + p_extra). Vacuum cells (E < e_vac) give everything 0. Returns 0 on success, 1 on failure.
// p_extra must keep g(1) >= 0 (it does for p_extra >= 0 and for small negative values); otherwise the cell is flagged.
inline constexpr Real kRecover3MaTol = 1e-12;  // relative excess of |M| over E treated as rounding

template <class EoS>
// pex_floor_kappa >= 0: in a cell where the extra pressure would leave no root (g(1) < 0, e.g. a negative gradient pressure on a
// cell at the E = |M| limiter surface of a dilute fringe), it is raised to -kappa (E - |M| + p0(E - |M|, 0)); bit 1 of the return
// value (value 2) marks such a cell, bit 0 (value 1) a failure. Negative kappa: no floor.
KOKKOS_INLINE_FUNCTION int recover_core(const EoS& eos, Real R, Real Ma, Real E, Real e_vac, Real pex, Real& eps, Real& rho,
                                        Real& c_out, Real& p, Real pex_floor_kappa = -1.0) {
  eps = 0.0; rho = 0.0; c_out = 0.0; p = 0.0;
  if (E < e_vac) return 0;
  const Real Rp = R > 0.0 ? R : 0.0;
  if (Ma == 0.0) {
    eps = E; rho = Rp; p = eos.p0(eps, rho) + pex;
    return 0;
  }
  // The E >= |M| limiter (Enforce3K) rescales the momentum vector to |M| = E; recomputing |M| through the square root of
  // three components can exceed E by a few units in the last place, which would leave no root in [0, 1] and flag the cell.
  // Treat such an excess as rounding (relative 1e-12); a larger excess is a genuine violation and is still flagged.
  if (Ma > E && Ma <= E * (1.0 + kRecover3MaTol)) Ma = E;
  Real a = 0.0, ga = -Ma;
  Real b = 1.0;
  Real gb;
  int floored = 0;
  {
    const Real e1 = E - Ma > 0.0 ? E - Ma : 0.0;
    const Real p01 = eos.p0(e1, 0.0);
    gb = E + (p01 + pex) - Ma;
    if (gb < 0.0 && pex_floor_kappa >= 0.0) {
      pex = -pex_floor_kappa * (e1 + p01);
      gb = E + (p01 + pex) - Ma;
      floored = 2;
    }
  }
  int fail = 0;
  Real c = Ma / E;
  if (c > 1.0) c = 1.0;
  if (gb < 0.0) {
    fail = 1;
  } else {
    int side = 0;
    Real c_prev = -1.0;
    fail = 1;
    for (int it = 0; it < kRecover3MaxIter; ++it) {
      c = (a * gb - b * ga) / (gb - ga);
      if (!(c > a && c < b)) c = 0.5 * (a + b);
      Real ec = E - Ma * c;
      if (ec < 0.0) ec = 0.0;
      const Real rc = Rp * Kokkos::sqrt(1.0 - c * c);
      const Real gc = c * (E + (eos.p0(ec, rc) + pex)) - Ma;
      if (Kokkos::fabs(gc) <= 1e-15 * E || (it > 0 && Kokkos::fabs(c - c_prev) < kRecover3Tol) || (b - a) < kRecover3Tol) {
        fail = 0;
        break;
      }
      c_prev = c;
      if (gc * gb > 0.0) {
        b = c; gb = gc;
        if (side == -1) ga *= 0.5;
        side = -1;
      } else {
        a = c; ga = gc;
        if (side == 1) gb *= 0.5;
        side = 1;
      }
    }
  }
  c_out = c;
  eps = E - Ma * c;
  if (eps < 0.0) eps = 0.0;
  rho = Rp * Kokkos::sqrt(1.0 - c * c);
  p = eos.p0(eps, rho) + pex;
  return fail | floored;
}

// Host files: raw little-endian doubles (all requested fields concatenated, each in C order [ix][iy][iz], interior cells only)
// and a JSON sidecar <base>.json (shape, dx, time, step, field list with byte offsets, configuration echo). Implemented in src/hydro3d_io.cpp.
void write_snapshot_files(const std::string& base, const int shape[3], Real dx, Real time, long step,
                          const std::vector<std::string>& names, const std::vector<std::vector<Real>>& data,
                          const nlohmann::json& config_echo);

// Host-side initial state in local-rest-frame variables, index (ix * ny + iy) * nz + iz.
struct HostState {
  std::vector<Real> eps, rho, vx, vy, vz;
};

// Smooth random initial state (src/ic_random.cpp): a few low-k Fourier modes with a fixed seed.
//   eps in [1, 2] GeV/fm^3, rho in [0.2, 0.4] fm^-3, |v| <= 0.3, p = eps/3 compatible.
HostState smooth_random_state(int nx, int ny, int nz, std::uint64_t seed = 12345, int n_modes = 6);

namespace detail {

template <class EoS>
struct Recover3K {
  EoS eos;
  Real e_vac;
  Field3 R, Mx, My, Mz, E, pex;
  Field3 p, vd, pvd;
  Kokkos::View<int***> fail;
  int dir, use_pex;
  Field3 rho_rf;     // rest-frame density of the recovery (written when write_rho != 0)
  int write_rho;
  int force_only;    // 1: the recovery uses p0 alone; pex is added to the flux terms only (gradient.in_primitive_recovery)
  Real kappa;        // gradient.pex_floor_kappa (negative: no floor)
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
    const Real mx = Mx(i, j, k), my = My(i, j, k), mz = Mz(i, j, k);
    const Real Ma = Kokkos::sqrt(mx * mx + my * my + mz * mz);
    const Real pe = use_pex ? pex(i, j, k) : 0.0;
    const Real pe_rec = force_only ? 0.0 : pe;
    Real ee, rr, c, pp;
    const int fl = recover_core(eos, R(i, j, k), Ma, E(i, j, k), e_vac, pe_rec, ee, rr, c, pp, kappa);
    if (force_only) pp += pe;
    if (write_rho) rho_rf(i, j, k) = rr;
    const Real md = dir == 0 ? mx : (dir == 1 ? my : mz);
    const Real vv = Ma > 0.0 ? c * (md / Ma) : 0.0;
    p(i, j, k) = pp;
    vd(i, j, k) = vv;
    pvd(i, j, k) = pp * vv;
    fail(i, j, k) = fl;
  }
};

template <class EoS>
struct PrimOutK {
  EoS eos;
  Real e_vac;
  Field3 R, Mx, My, Mz, E, pex;
  Field3 eps, rho, vx, vy, vz, p;
  Kokkos::View<int***> fail;
  int use_pex;
  int force_only;
  Real kappa;        // gradient.pex_floor_kappa (negative: no floor)
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
    const Real mx = Mx(i, j, k), my = My(i, j, k), mz = Mz(i, j, k);
    const Real Ma = Kokkos::sqrt(mx * mx + my * my + mz * mz);
    const Real pe = use_pex ? pex(i, j, k) : 0.0;
    const Real pe_rec = force_only ? 0.0 : pe;
    Real ee, rr, c, pp;
    const int fl = recover_core(eos, R(i, j, k), Ma, E(i, j, k), e_vac, pe_rec, ee, rr, c, pp, kappa);
    if (force_only) pp += pe;
    eps(i, j, k) = ee; rho(i, j, k) = rr; p(i, j, k) = pp;
    vx(i, j, k) = Ma > 0.0 ? c * (mx / Ma) : 0.0;
    vy(i, j, k) = Ma > 0.0 ? c * (my / Ma) : 0.0;
    vz(i, j, k) = Ma > 0.0 ? c * (mz / Ma) : 0.0;
    fail(i, j, k) = fl;
  }
};

// Conserved fields from primitives of one cell; eps <= 0 is a vacuum cell. Device function: calls eos.p0.
template <class EoS>
struct ConsFromPrim3K {
  EoS eos;
  Field3 eps, rho, vx, vy, vz, R, Mx, My, Mz, E;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
    const Real e = eps(i, j, k), r = rho(i, j, k);
    if (e <= 0.0) return;  // U was zeroed
    const Real ux = vx(i, j, k), uy = vy(i, j, k), uz = vz(i, j, k);
    const Real pr = eos.p0(e, r);
    const Real v2 = ux * ux + uy * uy + uz * uz;
    const Real gam2 = 1.0 / (1.0 - v2);
    const Real gam = Kokkos::sqrt(gam2);
    const Real w = (e + pr) * gam2;
    R(i, j, k) = gam * r;
    Mx(i, j, k) = w * ux;
    My(i, j, k) = w * uy;
    Mz(i, j, k) = w * uz;
    E(i, j, k) = w - pr;
  }
};

struct CountFail3K {
  Kokkos::View<int***> fail;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k, long& acc) const { acc += fail(i, j, k) & 1; }
};

struct CountFloor3K {
  Kokkos::View<int***> fail;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k, long& acc) const { acc += (fail(i, j, k) >> 1) & 1; }
};

struct Enforce3K {
  Field3 R, Mx, My, Mz, E;
  Real e_vac;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k, long& acc) const {
    const Real e = E(i, j, k);
    if (e < e_vac) {
      R(i, j, k) = 0.0; Mx(i, j, k) = 0.0; My(i, j, k) = 0.0; Mz(i, j, k) = 0.0; E(i, j, k) = 0.0;
      return;
    }
    const Real mx = Mx(i, j, k), my = My(i, j, k), mz = Mz(i, j, k);
    const Real Ma = Kokkos::sqrt(mx * mx + my * my + mz * mz);
    if (Ma > e) {
      Mx(i, j, k) = (mx / Ma) * e; My(i, j, k) = (my / Ma) * e; Mz(i, j, k) = (mz / Ma) * e;
      acc += 1;
    }
  }
};

struct CountEltM3K {
  Field3 Mx, My, Mz, E;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k, long& acc) const {
    const Real mx = Mx(i, j, k), my = My(i, j, k), mz = Mz(i, j, k);
    if (E(i, j, k) < Kokkos::sqrt(mx * mx + my * my + mz * mz)) acc += 1;
  }
};

}  // namespace detail

struct Totals {
  Real R = 0, E = 0, Mx = 0, My = 0, Mz = 0;
};

template <class EoS>
class Hydro3D {
 public:
  EoS eos;
  Grid grid;
  ShastaParams prm;
  GradientParams grad;
  SweepOrder order = SweepOrder::Permuted;
  Boundary bc[3];
  Field3 U[5], Uh[5], Un[5];  // conserved (R, Mx, My, Mz, E) at n, n+1/2, n+1
  Field3 p_extra;             // extra pressure supplied by the caller, all cells incl. ghosts; zero here
  bool use_p_extra = true;    // false: the hook is bypassed (T3.6)
  Field3 rho_rf;              // rest-frame density of the most recent recovery inside a sweep (allocated if the gradient term is on)
  bool rho_rf_valid = false;  // false: step() first fills rho_rf from a recovery of U
  Real min_pgrad_ratio_seen = 0.0;  // most negative p_grad / p0 over the samples taken by sample_pgrad_ratio()
  Field3 eps, rho, vx, vy, vz, p_out;  // primitives of the last update_primitives() (allocated on first use)
  long n_clamped_total = 0, n_recover_fail_total = 0, steps = 0;
  long n_pex_floor_total = 0;  // recovered cells in which the floor on p_extra acted (gradient.pex_floor_kappa), summed
  Real time = 0.0;

  Hydro3D(const EoS& eos_, const Hydro3DParams& hp)
      : eos(eos_), grid(Grid::make_3d(hp.nx, hp.ny, hp.nz, hp.dx)), prm(hp.shasta), grad(hp.gradient), order(hp.order), work_(grid) {
    for (int d = 0; d < 3; ++d) bc[d] = prm.bc;
    const char* names[5] = {"R", "Mx", "My", "Mz", "E"};
    for (int c = 0; c < 5; ++c) {
      U[c] = grid.make_field(std::string("U_") + names[c]);
      Uh[c] = grid.make_field(std::string("Uh_") + names[c]);
      Un[c] = grid.make_field(std::string("Un_") + names[c]);
    }
    p_extra = grid.make_field("p_extra");
    if (grad.form != GradientForm::Off) rho_rf = grid.make_field("rho_rf");
    p_ = grid.make_field("p");
    vd_ = grid.make_field("vd");
    pvd_ = grid.make_field("pvd");
    zero_ = grid.make_field("zero");
    fail_ = Kokkos::View<int***>("fail", grid.size(0), grid.size(1), grid.size(2));
  }

  Real dt() const { return prm.lambda * grid.dx; }
  long idx_n() const { return static_cast<long>(grid.n[0]) * grid.n[1] * grid.n[2]; }

  // ------------------------------------------------------------------ state setting (host side)
  void set_state(const HostState& s) {
    const int ny = grid.n[1], nz = grid.n[2];
    set_state_fn([&](int ix, int iy, int iz, Real& e, Real& r, Real& ux, Real& uy, Real& uz) {
      const size_t id = (static_cast<size_t>(ix) * ny + iy) * nz + iz;
      e = s.eps[id]; r = s.rho[id]; ux = s.vx[id]; uy = s.vy[id]; uz = s.vz[id];
    });
  }

  // f(ix, iy, iz, eps, rho, vx, vy, vz) fills the local-rest-frame values of interior cell (ix, iy, iz). eps <= 0: vacuum cell.
  // The host only stages the primitive values; the conserved fields are computed in a kernel because eos.p0 is a device
  // function (docs/device_rules.md, rule 9). Uh[0..4] serve as device staging for (eps, rho, vx, vy, vz).
  template <class F>
  void set_state_fn(F&& f) {
    Field3Host h[5];
    for (int c = 0; c < 5; ++c) {
      h[c] = Kokkos::create_mirror_view(Uh[c]);
      Kokkos::deep_copy(h[c], 0.0);
    }
    const int gx = grid.g[0], gy = grid.g[1], gz = grid.g[2];
    for (int ix = 0; ix < grid.n[0]; ++ix)
      for (int iy = 0; iy < grid.n[1]; ++iy)
        for (int iz = 0; iz < grid.n[2]; ++iz) {
          Real e = 0, r = 0, ux = 0, uy = 0, uz = 0;
          f(ix, iy, iz, e, r, ux, uy, uz);
          h[0](gx + ix, gy + iy, gz + iz) = e;
          h[1](gx + ix, gy + iy, gz + iz) = r;
          h[2](gx + ix, gy + iy, gz + iz) = ux;
          h[3](gx + ix, gy + iy, gz + iz) = uy;
          h[4](gx + ix, gy + iy, gz + iz) = uz;
        }
    for (int c = 0; c < 5; ++c) {
      Kokkos::deep_copy(Uh[c], h[c]);
      Kokkos::deep_copy(U[c], 0.0);
    }
    // p_extra of the initial state is not included: the hook starts at zero
    Kokkos::parallel_for("hydro3d_cons_from_prim", interior_policy(grid),
                         detail::ConsFromPrim3K<EoS>{eos, Uh[0], Uh[1], Uh[2], Uh[3], Uh[4], U[0], U[1], U[2], U[3], U[4]});
    if (gradient_on()) {
      // Uh[1] still holds the staged rest-frame density: it is the "most recent recovery" for the first refresh
      Kokkos::deep_copy(rho_rf, Uh[1]);
      rho_rf_valid = true;
    }
    time = 0.0;
    steps = 0;
  }

  // Set the conserved fields directly (WP5 initial states). The five Views must have the extents of this grid (ghost layers
  // included; the ghost layers are copied as they are, normally zero). Time and step counter are reset; with the gradient term on,
  // rho_rf is filled lazily by the first step (init_rho_rf). No equation of state is used here.
  void set_conserved(const Field3& R_, const Field3& Mx_, const Field3& My_, const Field3& Mz_, const Field3& E_) {
    const Field3* src[5] = {&R_, &Mx_, &My_, &Mz_, &E_};
    for (int c = 0; c < 5; ++c) {
      if (src[c]->extent(0) != U[c].extent(0) || src[c]->extent(1) != U[c].extent(1) || src[c]->extent(2) != U[c].extent(2))
        throw std::invalid_argument("Hydro3D::set_conserved: field extents differ from the grid");
      Kokkos::deep_copy(U[c], *src[c]);
    }
    rho_rf_valid = false;
    time = 0.0;
    steps = 0;
  }

  // The finite-range pressure is applied (form != off and the hook is enabled).
  bool gradient_on() const { return grad.form != GradientForm::Off && use_p_extra; }
  bool force_only() const { return gradient_on() && grad.recovery == GradientRecovery::ForceOnly; }
  Real pex_kappa() const { return gradient_on() ? grad.pex_floor_kappa : -1.0; }

  // Fill rho_rf from a recovery of U (use after writing U directly, e.g. from an initial-condition generator).
  // Uses the current p_extra (zero at start).
  void init_rho_rf() {
    update_primitives();
    Kokkos::deep_copy(rho_rf, rho);
    rho_rf_valid = true;
  }

  // Minimum over the interior of p_grad / p0 for the current p_extra and U (diagnostic; side-effect free for the solution).
  // The running minimum is kept in min_pgrad_ratio_seen.
  Real sample_pgrad_ratio() {
    update_primitives();
    Real m = 1e300;
    Kokkos::parallel_reduce("hydro3d_min_pgrad_ratio", interior_policy(grid), detail::MinPgradRatioK{p_out, p_extra}, Kokkos::Min<Real>(m));
    if (m > 0.0) m = 0.0;
    if (m < min_pgrad_ratio_seen) min_pgrad_ratio_seen = m;
    return m;
  }

  // ------------------------------------------------------------------ diagnostics
  Totals totals() const {
    Totals t;
    t.R = field_sum(grid, U[0]);
    t.Mx = field_sum(grid, U[1]);
    t.My = field_sum(grid, U[2]);
    t.Mz = field_sum(grid, U[3]);
    t.E = field_sum(grid, U[4]);
    return t;
  }

  long count_E_lt_M() const {
    long n = 0;
    Kokkos::parallel_reduce("hydro3d_count_EltM", interior_policy(grid), detail::CountEltM3K{U[1], U[2], U[3], U[4]}, n);
    return n;
  }

  // Recover (eps, rho, v, p) of U on the interior (allocates the output fields on first use). Returns the failure count.
  long update_primitives() {
    if (!eps.is_allocated()) {
      eps = grid.make_field("eps"); rho = grid.make_field("rho");
      vx = grid.make_field("vx"); vy = grid.make_field("vy"); vz = grid.make_field("vz");
      p_out = grid.make_field("p_out");
    }
    Kokkos::parallel_for("hydro3d_primitives", interior_policy(grid),
                         detail::PrimOutK<EoS>{eos, prm.e_vac, U[0], U[1], U[2], U[3], U[4], p_extra, eps, rho, vx, vy, vz, p_out,
                                               fail_, use_p_extra ? 1 : 0, force_only() ? 1 : 0, pex_kappa()});
    return count_fail();
  }

  // ------------------------------------------------------------------ evolution
  // One time step dt = lam * dx: three sweeps (one per direction), each a full half-step SHASTA step of the five fields.
  void step(Real lam) {
    if (gradient_on() && !rho_rf_valid) init_rho_rf();
    int o[3] = {0, 1, 2};
    if (order == SweepOrder::Permuted) {
      const int s = static_cast<int>(steps % 3);
      for (int q = 0; q < 3; ++q) o[q] = (s + q) % 3;
    }
    for (int q = 0; q < 3; ++q) sweep(o[q], lam);
    ++steps;
    time += lam * grid.dx;
  }
  void step() { step(prm.lambda); }

  void run(long nsteps) {
    for (long s = 0; s < nsteps; ++s) step();
  }

  // Run nsteps steps; write a snapshot at step 0 and every snapshot_every steps (if > 0) as <dir>/<prefix>_<step>.{bin,json}.
  void run_with_snapshots(long nsteps, long snapshot_every, const std::string& dir, const std::string& prefix,
                          const std::vector<std::string>& fields, const nlohmann::json& config_echo) {
    if (snapshot_every > 0) write_snapshot(dir + "/" + prefix + "_" + std::to_string(steps), fields, config_echo);
    for (long s = 0; s < nsteps; ++s) {
      step();
      if (snapshot_every > 0 && steps % snapshot_every == 0)
        write_snapshot(dir + "/" + prefix + "_" + std::to_string(steps), fields, config_echo);
    }
  }

  // Fields: R, Mx, My, Mz, E, eps, rho, vx, vy, vz, p. Host mirror copies only here.
  void write_snapshot(const std::string& base, const std::vector<std::string>& names, const nlohmann::json& config_echo) {
    bool need_prim = false;
    for (const auto& n : names)
      if (n == "eps" || n == "rho" || n == "vx" || n == "vy" || n == "vz" || n == "p") need_prim = true;
    if (need_prim) update_primitives();
    std::vector<std::vector<Real>> data;
    for (const auto& n : names) data.push_back(interior_copy(field_by_name(n)));
    write_snapshot_files(base, grid.n, grid.dx, time, steps, names, data, config_echo);
  }

  // Failure flags of the most recent recovery (update_primitives or a sweep), as a host vector like interior_copy (1 = failed).
  std::vector<int> fail_flags_copy() const {
    auto h = Kokkos::create_mirror_view(fail_);
    Kokkos::deep_copy(h, fail_);
    std::vector<int> out(static_cast<size_t>(idx_n()));
    size_t q = 0;
    for (int ix = 0; ix < grid.n[0]; ++ix)
      for (int iy = 0; iy < grid.n[1]; ++iy)
        for (int iz = 0; iz < grid.n[2]; ++iz) out[q++] = h(grid.g[0] + ix, grid.g[1] + iy, grid.g[2] + iz);
    return out;
  }

  // Interior of a field as a host vector, C order [ix][iy][iz].
  std::vector<Real> interior_copy(const Field3& F) const {
    auto h = Kokkos::create_mirror_view(F);
    Kokkos::deep_copy(h, F);
    std::vector<Real> out(static_cast<size_t>(idx_n()));
    size_t q = 0;
    for (int ix = 0; ix < grid.n[0]; ++ix)
      for (int iy = 0; iy < grid.n[1]; ++iy)
        for (int iz = 0; iz < grid.n[2]; ++iz) out[q++] = h(grid.g[0] + ix, grid.g[1] + iy, grid.g[2] + iz);
    return out;
  }

  const Field3& field_by_name(const std::string& n) const {
    if (n == "R") return U[0];
    if (n == "Mx") return U[1];
    if (n == "My") return U[2];
    if (n == "Mz") return U[3];
    if (n == "E") return U[4];
    if (n == "eps") return eps;
    if (n == "rho") return rho;
    if (n == "vx") return vx;
    if (n == "vy") return vy;
    if (n == "vz") return vz;
    if (n == "p") return p_out;
    throw std::invalid_argument("unknown snapshot field '" + n + "'");
  }

 private:
  Field3 p_, vd_, pvd_, zero_;
  Kokkos::View<int***> fail_;
  ShastaWork work_;

  long count_fail() {
    long nf = 0;
    Kokkos::parallel_reduce("hydro3d_count_fail", interior_policy(grid), detail::CountFail3K{fail_}, nf);
    n_recover_fail_total += nf;
    if (pex_kappa() >= 0.0) {
      long nfl = 0;
      Kokkos::parallel_reduce("hydro3d_count_floor", interior_policy(grid), detail::CountFloor3K{fail_}, nfl);
      n_pex_floor_total += nfl;
    }
    return nf;
  }

  // Ghost layers along dir on the transverse interior only (a sweep reads nothing else).
  void fill_ghosts_dir(Field3* Q, int dir) const {
    const int g = grid.g[dir], n = grid.n[dir];
    for (int c = 0; c < 5; ++c) {
      for (int side = 0; side < 2; ++side) {
        int b[3] = {grid.g[0], grid.g[1], grid.g[2]};
        int e[3] = {grid.g[0] + grid.n[0], grid.g[1] + grid.n[1], grid.g[2] + grid.n[2]};
        b[dir] = side == 0 ? 0 : g + n;
        e[dir] = side == 0 ? g : g + n + g;
        detail::GhostFillK k{Q[c], dir, g, n, bc[dir] == Boundary::Periodic ? 1 : 0, side == 0 ? 1 : 0, b[dir]};
        Kokkos::parallel_for("spf_fill_ghost_dir", make_policy(b, e), k);
      }
    }
  }

  // Recover p, v_dir, p v_dir from Q on all cells along dir (ghosts included), transverse interior.
  void recover_dir(Field3* Q, int dir) {
    int b[3] = {grid.g[0], grid.g[1], grid.g[2]};
    int e[3] = {grid.g[0] + grid.n[0], grid.g[1] + grid.n[1], grid.g[2] + grid.n[2]};
    b[dir] = 0;
    e[dir] = grid.size(dir);
    Kokkos::parallel_for("hydro3d_recover", make_policy(b, e),
                         detail::Recover3K<EoS>{eos, prm.e_vac, Q[0], Q[1], Q[2], Q[3], Q[4], p_extra, p_, vd_, pvd_, fail_, dir,
                                                use_p_extra ? 1 : 0, rho_rf, gradient_on() ? 1 : 0, force_only() ? 1 : 0, pex_kappa()});
    count_fail();
  }

  // Ghost layers along dir of one field (transverse interior only).
  void fill_dir_one(const Field3& F, int dir) const {
    const int g = grid.g[dir], n = grid.n[dir];
    for (int side = 0; side < 2; ++side) {
      int b[3] = {grid.g[0], grid.g[1], grid.g[2]};
      int e[3] = {grid.g[0] + grid.n[0], grid.g[1] + grid.n[1], grid.g[2] + grid.n[2]};
      b[dir] = side == 0 ? 0 : g + n;
      e[dir] = side == 0 ? g : g + n + g;
      detail::GhostFillK k{F, dir, g, n, bc[dir] == Boundary::Periodic ? 1 : 0, side == 0 ? 1 : 0, b[dir]};
      Kokkos::parallel_for("spf_fill_ghost_dir1", make_policy(b, e), k);
    }
  }

  // p_extra <- p_grad(rho_rf): ghost-fill rho_rf in all three directions (sequentially, so that transverse ghosts and corners
  // are filled), evaluate the Laplacian stencil on the interior, then fill the ghost layers of p_extra along dir (a sweep reads
  // p_extra on its own line only, ghosts included).
  void refresh_p_grad(int dir) {
    for (int d = 0; d < 3; ++d) fill_ghost(grid, d, bc[d], rho_rf);
    compute_p_grad(grid, grad, rho_rf, p_extra);
    fill_dir_one(p_extra, dir);
  }

  void enforce(Field3* Q) {
    long nc = 0;
    Kokkos::parallel_reduce("hydro3d_enforce", interior_policy(grid), detail::Enforce3K{Q[0], Q[1], Q[2], Q[3], Q[4], prm.e_vac}, nc);
    n_clamped_total += nc;
  }

  void sweep(int dir, Real lam) {
    const Field3* f[5] = {&zero_, &zero_, &zero_, &zero_, &pvd_};
    f[1 + dir] = &p_;
    fill_ghosts_dir(U, dir);
    if (gradient_on()) refresh_p_grad(dir);  // from rho_rf of the previous recovery (half-step state of the previous sweep)
    recover_dir(U, dir);                     // writes rho_rf from U
    for (int c = 0; c < 5; ++c) shasta_step(grid, dir, 0.5 * lam, prm, U[c], vd_, *f[c], Uh[c], work_);
    enforce(Uh);
    fill_ghosts_dir(Uh, dir);
    if (gradient_on()) refresh_p_grad(dir);  // from rho_rf of the recovery of U just above
    recover_dir(Uh, dir);                    // writes rho_rf from U^{1/2}
    if (gradient_on() && grad.half_step == GradientHalfStep::Midpoint) {
      refresh_p_grad(dir);                   // gradient.half_step_refresh = midpoint: p_grad from the density of U^{1/2}
      recover_dir(Uh, dir);
    }
    for (int c = 0; c < 5; ++c) shasta_step(grid, dir, lam, prm, U[c], vd_, *f[c], Un[c], work_);
    enforce(Un);
    for (int c = 0; c < 5; ++c) {
      Field3 t = U[c]; U[c] = Un[c]; Un[c] = t;
    }
  }
};

}  // namespace spf
