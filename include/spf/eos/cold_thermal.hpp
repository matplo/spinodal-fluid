#pragma once
// Development equation of state "cold_thermal":
//
//   p0(eps, rho) = p_c(rho) + (Gamma_th - 1) (eps - eps_c(rho)),   Gamma_th = 4/3 by default,
//
// eps_c(rho) = T = 0 energy density (rest mass included) of the two-phase model, p_c = rho eps_c' - eps_c.
// For eps < eps_c(rho) the thermal term is clamped to zero (pressure evaluated at eps = eps_c).
//
// eps_c and mu_c = eps_c' are read from data/eos/cold_curve.dat (tools/eos/make_cold_curve.py) on a uniform grid and
// interpolated with a C1 cubic Hermite polynomial per cell, so eps_c, mu_c and p_c are continuous.  p_c' = rho eps_c''
// is piecewise linear in rho and jumps by O(h^2) at the grid nodes.
// Variants: two_phase (the table everywhere) and maxwell (inside [rho_1, rho_2]: eps_c linear with slope mu_coex,
// p_c = p_coex, p_c' = 0 exactly; outside identical to two_phase).
// Units: see concept.hpp.
#include "spf/eos/concept.hpp"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace spf {

struct ColdCurveRange {
  Real rho_min;  // fm^-3
  Real rho_max;  // fm^-3
};

// Host-side owner of the cold-curve data.  Must outlive every EosColdThermal built from it.
class ColdCurveTable {
 public:
  Real rho_s = 0.0, rho_1 = 0.0, rho_2 = 0.0, mu_coex = 0.0, p_coex = 0.0, d_rho = 0.0;
  int n_nodes = 0;
  Kokkos::View<Real*> eps_c, mu_c;  // default memory space

  // Parse the table; throws std::runtime_error on a missing file, missing header key or non-uniform grid.
  static ColdCurveTable load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open cold curve table '" + path + "'");
    ColdCurveTable t;
    std::vector<Real> rho, e, mu;
    int have = 0;
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      if (line[0] == '#') {
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(1, eq - 1), val = line.substr(eq + 1);
        const auto b = key.find_first_not_of(" \t"), c = key.find_last_not_of(" \t");
        key = (b == std::string::npos) ? "" : key.substr(b, c - b + 1);
        Real* target = nullptr;
        if (key == "rho_s_fm3") target = &t.rho_s;
        else if (key == "rho_1_fm3") target = &t.rho_1;
        else if (key == "rho_2_fm3") target = &t.rho_2;
        else if (key == "mu_coex_GeV") target = &t.mu_coex;
        else if (key == "p_coex_GeV_fm3") target = &t.p_coex;
        else if (key == "d_rho_fm3") target = &t.d_rho;
        if (target) {
          *target = std::stod(val);
          ++have;
        }
        continue;
      }
      std::istringstream ls(line);
      Real a, b, c;
      if (ls >> a >> b >> c) {
        rho.push_back(a);
        e.push_back(b);
        mu.push_back(c);
      }
    }
    if (have != 6) throw std::runtime_error("cold curve table '" + path + "': missing header keys");
    if (rho.size() < 4) throw std::runtime_error("cold curve table '" + path + "': too few rows");
    for (size_t i = 0; i < rho.size(); ++i) {
      if (std::abs(rho[i] - static_cast<Real>(i) * t.d_rho) > 1e-9 * t.d_rho * static_cast<Real>(i + 1))
        throw std::runtime_error("cold curve table '" + path + "': grid is not uniform");
    }
    t.n_nodes = static_cast<int>(rho.size());
    t.eps_c = Kokkos::View<Real*>("cold_curve_eps_c", t.n_nodes);
    t.mu_c = Kokkos::View<Real*>("cold_curve_mu_c", t.n_nodes);
    auto he = Kokkos::create_mirror_view(t.eps_c);
    auto hm = Kokkos::create_mirror_view(t.mu_c);
    for (int i = 0; i < t.n_nodes; ++i) {
      he(i) = e[i];
      hm(i) = mu[i];
    }
    Kokkos::deep_copy(t.eps_c, he);
    Kokkos::deep_copy(t.mu_c, hm);
    return t;
  }

  // Densities for which the interpolant is defined; outside, EosColdThermal clamps rho to this range.
  ColdCurveRange valid_range() const { return {0.0, static_cast<Real>(n_nodes - 1) * d_rho}; }
};

enum class ColdVariant { two_phase = 0, maxwell = 1 };

struct EosColdThermal {
  const Real* eps_c_tab = nullptr;
  const Real* mu_c_tab = nullptr;
  int n_nodes = 0;
  Real d_rho = 1.0;
  Real gamma_th = 4.0 / 3.0;
  int maxwell = 0;  // 1: Maxwell construction inside [rho_1, rho_2]
  Real rho_1 = 0.0, rho_2 = 0.0, mu_coex = 0.0, p_coex = 0.0;

  EosColdThermal() = default;
  EosColdThermal(const ColdCurveTable& t, ColdVariant variant, Real gamma_th_ = 4.0 / 3.0)
      : eps_c_tab(t.eps_c.data()),
        mu_c_tab(t.mu_c.data()),
        n_nodes(t.n_nodes),
        d_rho(t.d_rho),
        gamma_th(gamma_th_),
        maxwell(variant == ColdVariant::maxwell ? 1 : 0),
        rho_1(t.rho_1),
        rho_2(t.rho_2),
        mu_coex(t.mu_coex),
        p_coex(t.p_coex) {}

  ColdCurveRange valid_range() const { return {0.0, static_cast<Real>(n_nodes - 1) * d_rho}; }
  static const char* name() { return "cold_thermal"; }

  // Cold curve: eps_c, mu_c = eps_c', and p_c' = rho eps_c'' at rho (clamped to the table range).
  KOKKOS_INLINE_FUNCTION void cold(Real rho, Real& ec, Real& mu, Real& dpc) const {
    const Real rmax = static_cast<Real>(n_nodes - 1) * d_rho;
    if (rho < 0.0) rho = 0.0;
    if (rho > rmax) rho = rmax;
    if (maxwell != 0 && rho > rho_1 && rho < rho_2) {
      ec = rho_1 * mu_coex - p_coex + (rho - rho_1) * mu_coex;
      mu = mu_coex;
      dpc = 0.0;
      return;
    }
    int i = static_cast<int>(rho / d_rho);
    if (i > n_nodes - 2) i = n_nodes - 2;
    if (i < 0) i = 0;
    const Real t = rho / d_rho - static_cast<Real>(i);
    const Real e0 = eps_c_tab[i], e1 = eps_c_tab[i + 1];
    const Real m0 = mu_c_tab[i] * d_rho, m1 = mu_c_tab[i + 1] * d_rho;
    const Real c2 = 3.0 * (e1 - e0) - 2.0 * m0 - m1;
    const Real c3 = -2.0 * (e1 - e0) + m0 + m1;
    ec = e0 + t * (m0 + t * (c2 + t * c3));
    mu = (m0 + t * (2.0 * c2 + 3.0 * c3 * t)) / d_rho;
    dpc = rho * (2.0 * c2 + 6.0 * c3 * t) / (d_rho * d_rho);
  }

  KOKKOS_INLINE_FUNCTION Real eps_cold(Real rho) const {
    Real ec, mu, dpc;
    cold(rho, ec, mu, dpc);
    return ec;
  }
  KOKKOS_INLINE_FUNCTION Real mu_cold(Real rho) const {
    Real ec, mu, dpc;
    cold(rho, ec, mu, dpc);
    return mu;
  }
  // p_c = rho mu_c - eps_c
  KOKKOS_INLINE_FUNCTION Real p_cold(Real rho) const {
    Real ec, mu, dpc;
    cold(rho, ec, mu, dpc);
    return pc_of(rho, mu, ec);
  }
  KOKKOS_INLINE_FUNCTION Real dp_cold(Real rho) const {
    Real ec, mu, dpc;
    cold(rho, ec, mu, dpc);
    return dpc;
  }

  KOKKOS_INLINE_FUNCTION Real p0(Real eps, Real rho) const {
    Real ec, mu, dpc;
    cold(rho, ec, mu, dpc);
    const Real pc = pc_of(rho, mu, ec);
    const Real th = eps - ec;
    return th > 0.0 ? pc + (gamma_th - 1.0) * th : pc;
  }
  KOKKOS_INLINE_FUNCTION Real dp_deps(Real eps, Real rho) const { return eps > eps_cold(rho) ? gamma_th - 1.0 : 0.0; }
  // In the clamped region the pressure is p_c(rho), so the derivative is p_c' there.
  KOKKOS_INLINE_FUNCTION Real dp_drho(Real eps, Real rho) const {
    Real ec, mu, dpc;
    cold(rho, ec, mu, dpc);
    return eps > ec ? dpc - (gamma_th - 1.0) * mu : dpc;
  }
  KOKKOS_INLINE_FUNCTION Real vs2(Real eps, Real rho) const {
    Real ec, mu, dpc;
    cold(rho, ec, mu, dpc);
    const Real pc = pc_of(rho, mu, ec);
    const bool above = eps > ec;
    const Real p = above ? pc + (gamma_th - 1.0) * (eps - ec) : pc;
    const Real a = above ? gamma_th - 1.0 : 0.0;
    const Real b = above ? dpc - (gamma_th - 1.0) * mu : dpc;
    const Real h = eps + p;
    return h > 0.0 ? a + rho / h * b : a;
  }

 private:
  KOKKOS_INLINE_FUNCTION Real pc_of(Real rho, Real mu, Real ec) const {
    return (maxwell != 0 && rho > rho_1 && rho < rho_2) ? p_coex : rho_clamped(rho) * mu - ec;
  }
  KOKKOS_INLINE_FUNCTION Real rho_clamped(Real rho) const {
    const Real rmax = static_cast<Real>(n_nodes - 1) * d_rho;
    return rho < 0.0 ? 0.0 : (rho > rmax ? rmax : rho);
  }
};

}  // namespace spf
