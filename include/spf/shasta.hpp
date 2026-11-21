#pragma once
// SHASTA flux-corrected transport (Boris & Book; Rischke, Bernard, Maruhn, Nucl. Phys. A 595, 346 (1995),
// arXiv:nucl-th/9504018, Sect. 3.1) for  d_t U + d_x (U v + f) = 0  along one direction of a 3D grid.
// Scheme as implemented: docs/equations.md. Kernel rules: docs/device_rules.md.
#include "spf/config.hpp"
#include "spf/types.hpp"

#include <stdexcept>
#include <string>

namespace spf {

// Cells of ghost layer needed on each side along the sweep direction by the full stencil
// (transport+diffusion, antidiffusive flux, limiter, update); derivation in docs/equations.md.
inline constexpr int kShastaGhost = 3;

enum class Boundary { Periodic, Outflow };

struct ShastaParams {
  bool phoenical = true;    // false: explicit antidiffusion
  Real eta_ad = 0.125;      // antidiffusion coefficient
  Real lambda = 0.4;        // dt/dx
  Real e_vac = 1e-12;       // GeV/fm^3, cells with E below this are vacuum
  Boundary bc = Boundary::Periodic;

  static ShastaParams from_config(const Config& cfg) {
    ShastaParams p;
    const std::string ad = cfg.get<std::string>("shasta.antidiffusion");
    if (ad == "phoenical") p.phoenical = true;
    else if (ad == "explicit") p.phoenical = false;
    else throw ConfigError("shasta.antidiffusion must be 'phoenical' or 'explicit', got '" + ad + "'");
    p.eta_ad = cfg.get<Real>("shasta.eta_ad");
    p.lambda = cfg.get<Real>("shasta.lambda");
    p.e_vac = cfg.get<Real>("shasta.e_vac_GeV_fm3");
    const std::string bc = cfg.get<std::string>("grid.boundary");
    if (bc == "periodic") p.bc = Boundary::Periodic;
    else if (bc == "outflow") p.bc = Boundary::Outflow;
    else throw ConfigError("grid.boundary must be 'periodic' or 'outflow', got '" + bc + "'");
    return p;
  }
};

// Uniform grid: n interior cells and g ghost cells on each side, per direction.
struct Grid {
  int n[3] = {1, 1, 1};
  int g[3] = {0, 0, 0};
  Real dx = 1.0;

  // ny = nz = 1, ghost layers along x only
  static Grid make_1d(int nx, Real dx_) {
    Grid gr;
    gr.n[0] = nx; gr.g[0] = kShastaGhost; gr.dx = dx_;
    return gr;
  }
  static Grid make_3d(int nx, int ny, int nz, Real dx_) {
    Grid gr;
    gr.n[0] = nx; gr.n[1] = ny; gr.n[2] = nz;
    gr.g[0] = gr.g[1] = gr.g[2] = kShastaGhost;
    gr.dx = dx_;
    return gr;
  }
  int size(int d) const { return n[d] + 2 * g[d]; }
  Field3 make_field(const std::string& label) const { return Field3(label, size(0), size(1), size(2)); }
};

using Policy3 = Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>;

inline Policy3 make_policy(const int b[3], const int e[3]) {
  return Policy3({b[0], b[1], b[2]}, {e[0], e[1], e[2]});
}

// Interior cells only.
inline Policy3 interior_policy(const Grid& gr) {
  const int b[3] = {gr.g[0], gr.g[1], gr.g[2]};
  const int e[3] = {gr.g[0] + gr.n[0], gr.g[1] + gr.n[1], gr.g[2] + gr.n[2]};
  return make_policy(b, e);
}

// All cells including ghosts.
inline Policy3 full_policy(const Grid& gr) {
  const int b[3] = {0, 0, 0};
  const int e[3] = {gr.size(0), gr.size(1), gr.size(2)};
  return make_policy(b, e);
}

namespace detail {

// Kernels are named functors, not KOKKOS_LAMBDAs: an nvcc extended lambda defined in a non-template inline function of a
// header that is compiled in several translation units can end up with an unset host-side wrapper pointer in optimised
// builds (null call at the first launch). Functors do not use that mechanism.
struct GhostFillK {
  Field3 f;
  int dir, g, n, periodic, left, off;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
    int c[3] = {i, j, k};
    const int a = c[dir] - off;  // 0 .. g-1 within the ghost layer
    int src;
    if (left) src = periodic ? a + n : g;
    else src = periodic ? g + a : g + n - 1;
    int s[3] = {i, j, k};
    s[dir] = src;
    f(i, j, k) = f(s[0], s[1], s[2]);
  }
};

}  // namespace detail

// Fill the ghost layers along direction dir of one field (periodic or zero-gradient).
// Transverse ghost layers are included in the loop range, so apply the directions in sequence for 3D.
inline void fill_ghost(const Grid& gr, int dir, Boundary bc, const Field3& F) {
  const int g = gr.g[dir], n = gr.n[dir];
  if (g == 0) return;
  for (int side = 0; side < 2; ++side) {
    int b[3] = {0, 0, 0};
    int e[3] = {gr.size(0), gr.size(1), gr.size(2)};
    b[dir] = side == 0 ? 0 : g + n;
    e[dir] = side == 0 ? g : g + n + g;
    detail::GhostFillK k{F, dir, g, n, bc == Boundary::Periodic ? 1 : 0, side == 0 ? 1 : 0, b[dir]};
    Kokkos::parallel_for("spf_fill_ghost", make_policy(b, e), k);
  }
}

struct ShastaWork {
  Field3 ut, a, at;
  explicit ShastaWork(const Grid& gr) : ut(gr.make_field("shasta_ut")), a(gr.make_field("shasta_a")), at(gr.make_field("shasta_at")) {}
};

namespace detail {

struct TransportDiffusionK {
  Field3 U, v, f, ut;
  int o0, o1, o2;
  Real lam;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
    const Real e0 = lam * v(i, j, k);
    const Real ep = lam * v(i + o0, j + o1, k + o2);
    const Real em = lam * v(i - o0, j - o1, k - o2);
    const Real qp = (0.5 - e0) / (1.0 + (ep - e0));
    const Real qm = (0.5 + e0) / (1.0 - (em - e0));
    const Real u0 = U(i, j, k);
    const Real d_j = U(i + o0, j + o1, k + o2) - u0;
    const Real d_jm = u0 - U(i - o0, j - o1, k - o2);
    const Real df = -0.5 * (f(i + o0, j + o1, k + o2) - f(i - o0, j - o1, k - o2));
    ut(i, j, k) = 0.5 * qp * qp * d_j - 0.5 * qm * qm * d_jm + (qp + qm) * u0 + lam * df;
  }
};

struct AntidiffusiveFluxK {
  Field3 U, ut, a;
  int o0, o1, o2, ph;
  Real eta;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
    const Real dt_j = ut(i + o0, j + o1, k + o2) - ut(i, j, k);
    Real flux = dt_j;
    if (ph) {
      const Real d_jp = U(i + 2 * o0, j + 2 * o1, k + 2 * o2) - U(i + o0, j + o1, k + o2);
      const Real d_j = U(i + o0, j + o1, k + o2) - U(i, j, k);
      const Real d_jm = U(i, j, k) - U(i - o0, j - o1, k - o2);
      flux = dt_j - 0.125 * (d_jp - 2.0 * d_j + d_jm);
    }
    a(i, j, k) = eta * flux;
  }
};

struct LimiterK {
  Field3 ut, a, at;
  int o0, o1, o2;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
    const Real av = a(i, j, k);
    const Real s = (av >= 0.0) ? 1.0 : -1.0;
    const Real dt_p = ut(i + 2 * o0, j + 2 * o1, k + 2 * o2) - ut(i + o0, j + o1, k + o2);
    const Real dt_m = ut(i, j, k) - ut(i - o0, j - o1, k - o2);
    const Real m = Kokkos::fmin(Kokkos::fmin(s * dt_p, Kokkos::fabs(av)), s * dt_m);
    at(i, j, k) = s * Kokkos::fmax(0.0, m);
  }
};

struct UpdateK {
  Field3 ut, at, Uo;
  int o0, o1, o2;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
    Uo(i, j, k) = ut(i, j, k) - at(i, j, k) + at(i - o0, j - o1, k - o2);
  }
};

struct SumK {
  Field3 f;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k, Real& acc) const { acc += f(i, j, k); }
};
struct MaxK {
  Field3 f;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k, Real& acc) const {
    if (f(i, j, k) > acc) acc = f(i, j, k);
  }
};
struct MinK {
  Field3 f;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k, Real& acc) const {
    if (f(i, j, k) < acc) acc = f(i, j, k);
  }
};

}  // namespace detail

// One SHASTA step of one component U along direction dir:
//   Uout = shasta(U; v, f) with  lam = dt/dx  (use lam = dt/(2 dx) for the half step).
// U, v, f must carry valid ghost layers (width kShastaGhost along dir). Uout is written on the interior only
// and must not alias U. v is the velocity component along dir, f the flux term of the component (0, p, or p v).
inline void shasta_step(const Grid& gr, int dir, Real lam, const ShastaParams& prm, const Field3& U, const Field3& v,
                        const Field3& f, const Field3& Uout, ShastaWork& w) {
  const int o0 = dir == 0, o1 = dir == 1, o2 = dir == 2;
  const int g = gr.g[dir], n = gr.n[dir];
  const int ph = prm.phoenical ? 1 : 0;

  const int bI[3] = {gr.g[0], gr.g[1], gr.g[2]};
  const int eI[3] = {gr.g[0] + gr.n[0], gr.g[1] + gr.n[1], gr.g[2] + gr.n[2]};

  // stage 1: transported and diffused Utilde on [g-2, g+n+2)
  {
    int b[3] = {bI[0], bI[1], bI[2]}, e[3] = {eI[0], eI[1], eI[2]};
    b[dir] = g - 2; e[dir] = g + n + 2;
    Kokkos::parallel_for("shasta_transport_diffusion", make_policy(b, e),
                         detail::TransportDiffusionK{U, v, f, w.ut, o0, o1, o2, lam});
  }
  // stage 2: antidiffusive flux A_j on [g-1, g+n)
  {
    int b[3] = {bI[0], bI[1], bI[2]}, e[3] = {eI[0], eI[1], eI[2]};
    b[dir] = g - 1; e[dir] = g + n;
    Kokkos::parallel_for("shasta_antidiffusive_flux", make_policy(b, e),
                         detail::AntidiffusiveFluxK{U, w.ut, w.a, o0, o1, o2, ph, prm.eta_ad});
  }
  // stage 3: flux limiter on [g-1, g+n)
  {
    int b[3] = {bI[0], bI[1], bI[2]}, e[3] = {eI[0], eI[1], eI[2]};
    b[dir] = g - 1; e[dir] = g + n;
    Kokkos::parallel_for("shasta_limiter", make_policy(b, e), detail::LimiterK{w.ut, w.a, w.at, o0, o1, o2});
  }
  // stage 4: update on the interior
  Kokkos::parallel_for("shasta_update", make_policy(bI, eI), detail::UpdateK{w.ut, w.at, Uout, o0, o1, o2});
}

// Sum over the interior [field units].
inline Real field_sum(const Grid& gr, const Field3& F) {
  Real s = 0.0;
  Kokkos::parallel_reduce("spf_sum", interior_policy(gr), detail::SumK{F}, s);
  return s;
}
inline Real field_max(const Grid& gr, const Field3& F) {
  Real m;
  Kokkos::parallel_reduce("spf_max", interior_policy(gr), detail::MaxK{F}, Kokkos::Max<Real>(m));
  return m;
}
inline Real field_min(const Grid& gr, const Field3& F) {
  Real m;
  Kokkos::parallel_reduce("spf_min", interior_policy(gr), detail::MinK{F}, Kokkos::Min<Real>(m));
  return m;
}

}  // namespace spf
