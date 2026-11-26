#pragma once
// Thermal integrals and thermal momentum sampling of the Cooper-Frye sampler (host only, no device code).
//
// One species of degeneracy g, mass m, temperature T and chemical potential mu (all GeV), statistics Boltzmann / Fermi-Dirac / Bose-Einstein:
//   n = g/(2 pi^2) Int k^2 f(E) dk / hbarc^3 [fm^-3],   e = g/(2 pi^2) Int k^2 E f dk / hbarc^3 [GeV fm^-3],
//   P = g/(6 pi^2) Int k^4 / E f dk / hbarc^3 [GeV fm^-3],   f = 1 / (exp((E - mu)/T) + s), s = +1, 0, -1.
// The momentum integral is a composite Gauss-Legendre rule in k. The panel edges follow the structure of the integrand:
// E = m + T t (t = 0 ... 50, thermal tail) and E = mu + T j (j = -30 ... 45, Fermi edge); the upper limit is max(m, mu) + 50 T
// (the integrand there is below 1e-21 of its maximum). Nodes per panel: sampler.quad_nodes (16*).
// The same panels give the cumulative weights of the momentum sampler, so that sampled and integrated distribution are one function.
#include "spf/particlisation/rng.hpp"

#include <cstddef>
#include <vector>

namespace spf::particlisation {

constexpr double kHbarcGeVfm = 0.1973269804;  // = constants.hbarc_GeV_fm in config/defaults.json

enum class Stat { boltzmann, fermi, bose };

// Occupation f at x = (E - mu)/T. Fermi: overflow-free; Bose: requires x > 0.
double occupation(Stat s, double x);

struct Thermal {
  double n = 0, e = 0, P = 0;  // fm^-3, GeV/fm^3, GeV/fm^3
};

// Momentum panel edges k_0 = 0 < k_1 < ... (GeV) for the integrand of the given state.
void thermal_edges(double m, double T, double mu, Stat st, std::vector<double>& k_edges);

// n, e, P of one species (degeneracy g included). Requires m > 0, T > 0.
Thermal thermal_integrals(double m, double g, double T, double mu, Stat st, int nodes = 16);

// Density only (cheaper by two thirds).
double thermal_density(double m, double g, double T, double mu, Stat st, int nodes = 16);

// Density n and mean acceptance alpha = <max(p*.dsig*, 0)/(E* dSigma_max)> of one species in an element with LRF dsig*_0 = d0 and |dsig*_vec| = s
// (dSigma_max = |d0| + s), averaged over the thermal distribution and the isotropic direction: the expected number of accepted particles is n dSigma_max alpha.
// Closed form for the angle: with v = k/E, the average of max(d0 - v s c, 0)/dSigma_max over c = cos(theta) in [-1, 1] is max(d0, 0)/dSigma_max for v s <= |d0|
// (this covers every time-like element: alpha = max(d0, 0)/dSigma_max, n dSigma_max alpha = n d0 for d0 > 0) and (v s + d0)^2/(4 v s dSigma_max) otherwise;
// the kink at v = |d0|/s is a panel edge of the momentum quadrature.
struct DensityAcceptance {
  double n = 0, alpha = 0;
};
DensityAcceptance thermal_density_acceptance(double m, double g, double T, double mu, Stat st, int nodes, double d0, double s);

// One momentum modulus k (GeV) drawn from k^2 f(E) dk: panel by its weight, then rejection inside the panel
// with the bound k_R^2 f(E(k_L)) (k^2 increases, f decreases with k: the bound is rigorous).
double sample_momentum(double m, double T, double mu, Stat st, Rng& rng, int nodes = 16);

// Decision S24 (sampler.mu_rule = rho_matched): the chemical potential mu of a nucleon gas (g, m, statistics st) at temperature T whose net density
// n(T, mu) - n(T, -mu) equals rho [fm^-3] (rho may be zero or negative). Brent's method on the monotone function mu -> n(mu) - n(-mu), bracketed from
// mu_guess; converged when |net - rho| <= rel_tol |rho| (or the bracket is at machine resolution). status: 0 converged, 1 rho == 0 (mu = 0),
// 2 not bracketed / not finite (the returned mu is mu_guess). T must be > 0.
struct MuSolve {
  double mu = 0;
  double rel_residual = 0;  // (net(mu) - |rho|) / |rho|, 0 for rho == 0
  int iterations = 0;
  int status = 0;
};
MuSolve solve_mu_net_density(double m, double g, double T, double rho, Stat st, int nodes, double mu_guess, double rel_tol = 1e-12);

}  // namespace spf::particlisation
