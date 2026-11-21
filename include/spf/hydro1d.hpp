#pragma once
// One-dimensional relativistic ideal fluid with SHASTA transport and the half-step method.
// Conserved fields R = gamma rho, M = (eps+p) gamma^2 v, E = (eps+p) gamma^2 - p.
// The solver is a template on the equation-of-state type; it uses only  eos.p0(eps, rho).
#include "spf/shasta.hpp"

#include <vector>

namespace spf {

inline constexpr int kRecoverMaxIter = 60;
inline constexpr Real kRecoverTol = 1e-12;

// Primitive recovery for one cell from (R, Mx, E). Solves v = |M| / (E + p0(E - |M| v, R sqrt(1-v^2))) on [0, 1)
// with a bracketed regula-falsi (Illinois) iteration, at most kRecoverMaxIter iterations.
// Vacuum cells (E < e_vac) give eps = rho = v = p = 0. Returns 0 on success, 1 if the iteration did not converge
// or no root lies in [0, 1] (the best estimate is still returned).
template <class EoS>
KOKKOS_INLINE_FUNCTION int recover_primitives(const EoS& eos, Real R, Real M, Real E, Real e_vac, Real& eps, Real& rho,
                                              Real& v, Real& p) {
  eps = 0.0; rho = 0.0; v = 0.0; p = 0.0;
  if (E < e_vac) return 0;
  const Real Ma = Kokkos::fabs(M);
  const Real Rp = R > 0.0 ? R : 0.0;
  if (Ma == 0.0) {
    eps = E; rho = Rp; p = eos.p0(eps, rho);
    return 0;
  }
  // g(v) = v (E + p0(eps(v), rho(v))) - |M|
  Real a = 0.0, ga = -Ma;
  Real b = 1.0;
  Real gb;
  {
    const Real e1 = E - Ma > 0.0 ? E - Ma : 0.0;
    gb = E + eos.p0(e1, 0.0) - Ma;
  }
  int fail = 0;
  Real c = Ma / E;  // fallback: p0 = 0
  if (c > 1.0) c = 1.0;
  if (gb < 0.0) {
    fail = 1;  // no root in [0, 1]: E < |M| (should have been limited) or negative p0
  } else {
    int side = 0;
    Real c_prev = -1.0;
    fail = 1;
    for (int it = 0; it < kRecoverMaxIter; ++it) {
      c = (a * gb - b * ga) / (gb - ga);
      if (!(c > a && c < b)) c = 0.5 * (a + b);
      Real ec = E - Ma * c;
      if (ec < 0.0) ec = 0.0;
      const Real rc = Rp * Kokkos::sqrt(1.0 - c * c);
      const Real gc = c * (E + eos.p0(ec, rc)) - Ma;
      if (Kokkos::fabs(gc) <= 1e-15 * E || (it > 0 && Kokkos::fabs(c - c_prev) < kRecoverTol) ||
          (b - a) < kRecoverTol) {
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
  v = (M < 0.0 ? -c : c);
  eps = E - Ma * c;
  if (eps < 0.0) eps = 0.0;
  rho = Rp * Kokkos::sqrt(1.0 - c * c);
  p = eos.p0(eps, rho);
  return fail;
}

namespace detail {

template <class EoS>
struct RecoverK {
  EoS eos;
  Real e_vac;
  Field3 R, M, E, eps, rho, v, p, pv;
  Kokkos::View<int***> fail;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
    Real ee, rr, vv, pp;
    const int fl = recover_primitives(eos, R(i, j, k), M(i, j, k), E(i, j, k), e_vac, ee, rr, vv, pp);
    eps(i, j, k) = ee; rho(i, j, k) = rr; v(i, j, k) = vv; p(i, j, k) = pp;
    pv(i, j, k) = pp * vv;
    fail(i, j, k) = fl;
  }
};

struct CountFailK {
  Kokkos::View<int***> fail;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k, long& acc) const { acc += fail(i, j, k); }
};

struct EnforceK {
  Field3 R, M, E;
  Real e_vac;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k, long& acc) const {
    if (E(i, j, k) < e_vac) {
      R(i, j, k) = 0.0; M(i, j, k) = 0.0; E(i, j, k) = 0.0;
    } else if (Kokkos::fabs(M(i, j, k)) > E(i, j, k)) {
      M(i, j, k) = (M(i, j, k) > 0.0) ? E(i, j, k) : -E(i, j, k);
      acc += 1;
    }
  }
};

struct CountEltMK {
  Field3 M, E;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k, long& acc) const {
    if (E(i, j, k) < Kokkos::fabs(M(i, j, k))) acc += 1;
  }
};

}  // namespace detail

namespace detail {

// Conserved fields from primitives (eps, rho, v) of one cell; eps <= 0 is a vacuum cell. Device function: calls eos.p0.
template <class EoS>
struct ConsFromPrim1K {
  EoS eos;
  Field3 eps, rho, v, R, M, E;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
    const Real e = eps(i, j, k), r = rho(i, j, k), u = v(i, j, k);
    if (e <= 0.0) return;  // U was zeroed
    const Real pr = eos.p0(e, r);
    const Real gam2 = 1.0 / (1.0 - u * u);
    const Real gam = Kokkos::sqrt(gam2);
    const Real w = (e + pr) * gam2;
    R(i, j, k) = gam * r;
    M(i, j, k) = w * u;
    E(i, j, k) = w - pr;
  }
};

}  // namespace detail

template <class EoS>
struct Hydro1D {
  EoS eos;
  Grid grid;
  ShastaParams prm;
  Field3 U[3], Uh[3], Un[3];  // conserved (R, M, E) at n, n+1/2, n+1
  Field3 eps, rho, v, p, pv;  // primitives of the state last passed to update_primitives*
  Field3 zero;                // f = 0 for the baryon density
  Kokkos::View<int***> fail;  // per-cell primitive-recovery failure flag
  ShastaWork work;
  long n_clamped_total = 0;   // cumulative number of cells where E >= |M| had to be enforced
  long n_recover_fail_total = 0;
  long steps = 0;

  Hydro1D(const EoS& eos_, int nx, Real dx, const ShastaParams& prm_)
      : eos(eos_), grid(Grid::make_1d(nx, dx)), prm(prm_), work(grid) {
    const char* names[3] = {"R", "M", "E"};
    for (int c = 0; c < 3; ++c) {
      U[c] = grid.make_field(std::string("U_") + names[c]);
      Uh[c] = grid.make_field(std::string("Uh_") + names[c]);
      Un[c] = grid.make_field(std::string("Un_") + names[c]);
    }
    eps = grid.make_field("eps");
    rho = grid.make_field("rho");
    v = grid.make_field("v");
    p = grid.make_field("p");
    pv = grid.make_field("pv");
    zero = grid.make_field("zero");
    Kokkos::deep_copy(zero, 0.0);
    fail = Kokkos::View<int***>("fail", grid.size(0), grid.size(1), grid.size(2));
  }

  // Set the interior from local-rest-frame values (host vectors of length nx), then fill ghosts.
  // The host only stages the primitive values; the conserved fields are computed in a kernel because eos.p0 is a device
  // function (docs/device_rules.md, rule 9). Uh[0..2] serve as device staging for (eps, rho, v).
  void set_state(const std::vector<Real>& eps_h, const std::vector<Real>& rho_h, const std::vector<Real>& v_h) {
    auto he = Kokkos::create_mirror_view(Uh[0]);
    auto hr = Kokkos::create_mirror_view(Uh[1]);
    auto hv = Kokkos::create_mirror_view(Uh[2]);
    Kokkos::deep_copy(he, 0.0); Kokkos::deep_copy(hr, 0.0); Kokkos::deep_copy(hv, 0.0);
    const int g = grid.g[0];
    for (int i = 0; i < grid.n[0]; ++i) {
      he(g + i, 0, 0) = eps_h[i]; hr(g + i, 0, 0) = rho_h[i]; hv(g + i, 0, 0) = v_h[i];
    }
    Kokkos::deep_copy(Uh[0], he); Kokkos::deep_copy(Uh[1], hr); Kokkos::deep_copy(Uh[2], hv);
    for (int c = 0; c < 3; ++c) Kokkos::deep_copy(U[c], 0.0);
    Kokkos::parallel_for("hydro1d_cons_from_prim", interior_policy(grid),
                         detail::ConsFromPrim1K<EoS>{eos, Uh[0], Uh[1], Uh[2], U[0], U[1], U[2]});
    update_primitives();
  }

  // Fill the ghost layers of the conserved fields Q[0..2].
  void fill_ghosts(Field3* Q) const {
    for (int c = 0; c < 3; ++c) fill_ghost(grid, 0, prm.bc, Q[c]);
  }

  // Recover (eps, rho, v, p) and p v from Q on all cells including ghosts. Q must have valid ghosts.
  // Returns the number of interior cells whose recovery failed.
  long recover_from(Field3* Q) {
    Kokkos::parallel_for("hydro1d_recover", full_policy(grid),
                         detail::RecoverK<EoS>{eos, prm.e_vac, Q[0], Q[1], Q[2], eps, rho, v, p, pv, fail});
    long nf = 0;
    Kokkos::parallel_reduce("hydro1d_count_fail", interior_policy(grid), detail::CountFailK{fail}, nf);
    n_recover_fail_total += nf;
    return nf;
  }

  // Fill ghosts of U and recover primitives of the current state (for output and diagnostics).
  long update_primitives() {
    fill_ghosts(U);
    return recover_from(U);
  }

  // Enforce E >= |M| (restrict |M| to E) and zero vacuum cells (E < e_vac) on the interior of Q. Returns the clamp count.
  long enforce(Field3* Q) {
    long nc = 0;
    Kokkos::parallel_reduce("hydro1d_enforce", interior_policy(grid), detail::EnforceK{Q[0], Q[1], Q[2], prm.e_vac}, nc);
    n_clamped_total += nc;
    return nc;
  }

  // Advance by dt = lam * dx with the half-step method (U^n -> U^{n+1/2} with v^n, f^n over dt/2;
  // then U^n -> U^{n+1} with v^{n+1/2}, f^{n+1/2} over dt).
  void step(Real lam) {
    update_primitives();  // ghosts of U^n and primitives at n
    const Field3* fq[3] = {&zero, &p, &pv};
    for (int c = 0; c < 3; ++c) shasta_step(grid, 0, 0.5 * lam, prm, U[c], v, *fq[c], Uh[c], work);
    enforce(Uh);
    fill_ghosts(Uh);
    recover_from(Uh);     // primitives at n+1/2 (ghosts included)
    for (int c = 0; c < 3; ++c) shasta_step(grid, 0, lam, prm, U[c], v, *fq[c], Un[c], work);
    enforce(Un);
    for (int c = 0; c < 3; ++c) { Field3 t = U[c]; U[c] = Un[c]; Un[c] = t; }
    ++steps;
  }
  void step() { step(prm.lambda); }
  Real dt() const { return prm.lambda * grid.dx; }

  // Interior sums of R, M, E.
  Real total_R() const { return field_sum(grid, U[0]); }
  Real total_M() const { return field_sum(grid, U[1]); }
  Real total_E() const { return field_sum(grid, U[2]); }

  // Number of interior cells with E < |M| (after enforce this must be zero).
  long count_E_lt_M() const {
    long n = 0;
    Kokkos::parallel_reduce("hydro1d_count_EltM", interior_policy(grid), detail::CountEltMK{U[1], U[2]}, n);
    return n;
  }

  // Host copy of one interior line of a field.
  std::vector<Real> line(const Field3& F) const {
    auto h = Kokkos::create_mirror_view(F);
    Kokkos::deep_copy(h, F);
    std::vector<Real> out(grid.n[0]);
    for (int i = 0; i < grid.n[0]; ++i) out[i] = h(grid.g[0] + i, 0, 0);
    return out;
  }
};

}  // namespace spf
