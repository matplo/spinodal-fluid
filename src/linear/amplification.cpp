#include "spf/linear/amplification.hpp"

#include <cmath>
#include <stdexcept>

namespace spf::linear {

std::vector<double> gamma_history(const ThermoProvider& thermo, const std::vector<TrajectoryPoint>& traj, double C, const Transport& tr, double k) {
  std::vector<double> g(traj.size());
  for (std::size_t i = 0; i < traj.size(); ++i) g[i] = growth_rate(make_dispersion(thermo(traj[i].rho, traj[i].T), C, tr), k);
  return g;
}

std::vector<double> cumulative_gamma(const std::vector<double>& t, const std::vector<double>& gamma) {
  std::vector<double> G(t.size(), 0.0);
  for (std::size_t i = 1; i < t.size(); ++i) G[i] = G[i - 1] + 0.5 * (gamma[i] + gamma[i - 1]) * (t[i] - t[i - 1]);
  return G;
}

namespace {
// int of max(g, 0) and the length where g > 0, for g linear from g0 to g1 over dt.
void positive_part(double g0, double g1, double dt, double& integral, double& length) {
  if (g0 >= 0 && g1 >= 0) { integral = 0.5 * (g0 + g1) * dt; length = dt; return; }
  if (g0 <= 0 && g1 <= 0) { integral = 0; length = 0; return; }
  const double gp = g0 > 0 ? g0 : g1, gm = g0 > 0 ? g1 : g0;  // positive end, negative end
  const double frac = gp / (gp - gm);                         // fraction of the step with g > 0
  length = frac * dt;
  integral = 0.5 * gp * length;
}
}  // namespace

double log_amplification(const std::vector<double>& t, const std::vector<double>& gamma) {
  double s = 0;
  for (std::size_t i = 1; i < t.size(); ++i) { double I, L; positive_part(gamma[i - 1], gamma[i], t[i] - t[i - 1], I, L); s += I; }
  return s;
}
double unstable_time(const std::vector<double>& t, const std::vector<double>& gamma) {
  double s = 0;
  for (std::size_t i = 1; i < t.size(); ++i) { double I, L; positive_part(gamma[i - 1], gamma[i], t[i] - t[i - 1], I, L); s += L; }
  return s;
}

DiffusionFn einstein_hold_diffusion(double sigma_eq2) {
  auto last = std::make_shared<double>(0.0);
  return [sigma_eq2, last](std::size_t i, double, double gamma) {
    if (i == 0) *last = 0.0;
    if (gamma < 0.0) *last = -gamma * sigma_eq2;
    return *last;
  };
}

double lalime_stationary(double gamma, double D, double s0, double dt) {
  const double e = std::exp(2.0 * gamma * dt);
  if (gamma == 0.0) return s0 + 2.0 * D * dt;
  return D / gamma * (e - 1.0) + s0 * e;
}

std::vector<double> lalime_solve(const std::vector<double>& t, const std::vector<double>& gamma, const DiffusionFn& D, double s0) {
  const std::size_t n = t.size();
  if (gamma.size() != n) throw std::invalid_argument("lalime_solve: size mismatch");
  std::vector<double> Dv(n), s(n);
  for (std::size_t i = 0; i < n; ++i) Dv[i] = D(i, t[i], gamma[i]);  // evaluated once, in order (the default model keeps state)
  s[0] = s0;
  for (std::size_t i = 1; i < n; ++i) {
    const double g = 0.5 * (gamma[i] + gamma[i - 1]), d = 0.5 * (Dv[i] + Dv[i - 1]), dt = t[i] - t[i - 1];
    // exact: s' = 2 g s + 2 d  ->  s(dt) = s e^{2 g dt} + d (e^{2 g dt} - 1)/g, with the g -> 0 limit via expm1
    const double x = 2.0 * g * dt;
    const double fac = std::abs(x) < 1e-8 ? 2.0 * dt * (1.0 + 0.5 * x) : 2.0 * std::expm1(x) / (2.0 * g);
    s[i] = s[i - 1] * std::exp(x) + d * fac;
  }
  return s;
}

}  // namespace spf::linear
