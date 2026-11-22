#pragma once
// Finite-range (gradient) pressure of Steinheimer and Randrup, PRL 109, 212301 (2012), arXiv:1209.2462, Eq. 1, and
// PRC 87, 054903 (2013), arXiv:1302.2956, Sect. III:
//
//   p = p0(eps, rho) + p_grad,   p_grad = - a^2 (eps_s / rho_s^2) rho Lap(rho),   rho = local-rest-frame net baryon density.
//
// eps_s = m_N rho_s, so the prefactor is a^2 m_N / rho_s. Derivation of the linear dispersion relation: docs/equations.md, Sect. 10.
// The kernel is a named functor (docs/device_rules.md, rule 2); it reads a ghost-filled rest-frame density field and
// writes p_grad on the interior cells. Ghost width needed: 1 cell (second_order) or 2 cells (fourth_order); the grid of
// shasta.hpp has kShastaGhost = 3 cells, so both stencils fit.
#include "spf/config.hpp"
#include "spf/shasta.hpp"

#include <string>

namespace spf {

enum class GradientForm { Off, PaperPressure };
enum class GradientStencil { SecondOrder, FourthOrder };
enum class GradientRecovery { Lagged, ForceOnly };
// Refresh of p_grad before the recovery of the half-step state: Lagged = from the density of the recovery of U^n (one recovery lag, default);
// Midpoint = the half-step state is recovered once with that p_grad, p_grad is refreshed from its density, and it is recovered again,
// so that the full-step force is evaluated at the half-step density (one extra recovery per sweep).
enum class GradientHalfStep { Lagged, Midpoint };

inline constexpr int kGradientGhostSecond = 1;
inline constexpr int kGradientGhostFourth = 2;
static_assert(kShastaGhost >= kGradientGhostFourth, "the grid ghost width must hold the 4th-order gradient stencil");

struct GradientParams {
  GradientForm form = GradientForm::Off;  // the library default is off; the configuration default is paper_pressure
  Real a_fm = 0.033;                      // fm
  GradientStencil stencil = GradientStencil::SecondOrder;
  GradientRecovery recovery = GradientRecovery::Lagged;
  GradientHalfStep half_step = GradientHalfStep::Lagged;
  Real rho_s = 0.153;                     // fm^-3
  Real m_N = 0.940;                       // GeV

  // a^2 eps_s / rho_s^2 = a^2 m_N / rho_s   [GeV fm^-1... such that coef * rho * Lap(rho) is in GeV/fm^3]
  Real coef() const { return a_fm * a_fm * m_N / rho_s; }
  Real eps_s() const { return m_N * rho_s; }

  static GradientParams from_config(const Config& cfg) {
    GradientParams g;
    const std::string form = cfg.get<std::string>("gradient.form");
    if (form == "off") g.form = GradientForm::Off;
    else if (form == "paper_pressure") g.form = GradientForm::PaperPressure;
    else if (form == "korteweg_full") throw ConfigError("gradient.form = korteweg_full: not implemented, see plan WP9");
    else throw ConfigError("gradient.form must be 'paper_pressure', 'off' or 'korteweg_full', got '" + form + "'");
    g.a_fm = cfg.get<Real>("gradient.a_fm");
    if (g.a_fm < 0.0) throw ConfigError("gradient.a_fm must be >= 0");
    const std::string st = cfg.get<std::string>("gradient.stencil");
    if (st == "second_order") g.stencil = GradientStencil::SecondOrder;
    else if (st == "fourth_order") g.stencil = GradientStencil::FourthOrder;
    else throw ConfigError("gradient.stencil must be 'second_order' or 'fourth_order', got '" + st + "'");
    const std::string rec = cfg.get<std::string>("gradient.in_primitive_recovery");
    if (rec == "lagged") g.recovery = GradientRecovery::Lagged;
    else if (rec == "force_only") g.recovery = GradientRecovery::ForceOnly;
    else throw ConfigError("gradient.in_primitive_recovery must be 'lagged' or 'force_only', got '" + rec + "'");
    const std::string hs = cfg.get<std::string>("gradient.half_step_refresh");
    if (hs == "lagged") g.half_step = GradientHalfStep::Lagged;
    else if (hs == "midpoint") g.half_step = GradientHalfStep::Midpoint;
    else throw ConfigError("gradient.half_step_refresh must be 'lagged' or 'midpoint', got '" + hs + "'");
    g.rho_s = cfg.get<Real>("constants.rho_s_fm3");
    g.m_N = cfg.get<Real>("constants.m_N_GeV");
    return g;
  }
};

namespace detail {

// p_grad = -coef rho Lap(rho) on the interior; rho must have its ghost layers filled (all three directions).
struct PGradK {
  Field3 rho, pg;
  Real coef, inv_dx2;
  int fourth;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
    const Real r0 = rho(i, j, k);
    Real lap;
    if (fourth == 0) {
      lap = (rho(i + 1, j, k) + rho(i - 1, j, k) + rho(i, j + 1, k) + rho(i, j - 1, k) + rho(i, j, k + 1) + rho(i, j, k - 1) -
             6.0 * r0) * inv_dx2;
    } else {
      const Real sx = 16.0 * (rho(i + 1, j, k) + rho(i - 1, j, k)) - (rho(i + 2, j, k) + rho(i - 2, j, k));
      const Real sy = 16.0 * (rho(i, j + 1, k) + rho(i, j - 1, k)) - (rho(i, j + 2, k) + rho(i, j - 2, k));
      const Real sz = 16.0 * (rho(i, j, k + 1) + rho(i, j, k - 1)) - (rho(i, j, k + 2) + rho(i, j, k - 2));
      lap = (sx + sy + sz - 90.0 * r0) * (inv_dx2 / 12.0);
    }
    pg(i, j, k) = -coef * r0 * lap;
  }
};

// Min over the interior of p_grad / p0 (p0 > 0 cells only); p is the total pressure p0 + p_grad of the primitive recovery.
struct MinPgradRatioK {
  Field3 p, pex;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k, Real& m) const {
    const Real pe = pex(i, j, k), p0 = p(i, j, k) - pe;
    if (p0 > 0.0) {
      const Real r = pe / p0;
      if (r < m) m = r;
    }
  }
};

}  // namespace detail

// Compute p_grad on the interior of `pg` from the ghost-filled rest-frame density `rho`.
inline void compute_p_grad(const Grid& gr, const GradientParams& gp, const Field3& rho, const Field3& pg) {
  detail::PGradK k{rho, pg, gp.coef(), 1.0 / (gr.dx * gr.dx), gp.stencil == GradientStencil::FourthOrder ? 1 : 0};
  Kokkos::parallel_for("spf_p_grad", interior_policy(gr), k);
}

}  // namespace spf
