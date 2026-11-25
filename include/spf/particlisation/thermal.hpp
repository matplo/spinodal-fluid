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

// One momentum modulus k (GeV) drawn from k^2 f(E) dk: panel by its weight, then rejection inside the panel
// with the bound k_R^2 f(E(k_L)) (k^2 increases, f decreases with k: the bound is rigorous).
double sample_momentum(double m, double T, double mu, Stat st, Rng& rng, int nodes = 16);

}  // namespace spf::particlisation
