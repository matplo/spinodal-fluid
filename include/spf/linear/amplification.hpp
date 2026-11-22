#pragma once
// Amplification along a phase trajectory (rho(t), T(t)): Gamma_k(t) (Eq. 47), G_k (Eq. 49), Lalime equation (Eqs. 45-48).
#include "spf/linear/dispersion.hpp"

#include <functional>
#include <memory>
#include <vector>

namespace spf::linear {

struct TrajectoryPoint { double t = 0, rho = 0, T = 0; };  // fm/c, fm^-3, GeV
using ThermoProvider = std::function<ThermoPoint(double rho, double T)>;

// gamma_k(t_i) along the trajectory.
std::vector<double> gamma_history(const ThermoProvider& thermo, const std::vector<TrajectoryPoint>& traj, double C, const Transport& tr, double k);

// Cumulative Gamma(t_i) = int_{t_0}^{t_i} gamma dt (trapezoid rule; piecewise-linear gamma).
std::vector<double> cumulative_gamma(const std::vector<double>& t, const std::vector<double>& gamma);
// Eq. (49): ln G = int of max(gamma, 0) dt, exact for piecewise-linear gamma (zero crossings inside a step handled analytically).
double log_amplification(const std::vector<double>& t, const std::vector<double>& gamma);
// Time during which gamma > 0 (same piecewise-linear treatment).
double unstable_time(const std::vector<double>& t, const std::vector<double>& gamma);

// Diffusion coefficient D_k(t) of the Lalime equation. Default (choice documented in docs/equations.md):
//   gamma < 0 (stable):  D = -gamma sigma_eq2   (Einstein relation, paper below Eq. 48: sigma^2 relaxes to -D/gamma = sigma_eq2);
//   gamma >= 0:          D = value at the last stable time (noise keeps being produced and amplified, paper Sect. V B);
//                        zero if the trajectory starts unstable.
// A user model replaces it: D(i, t, gamma) is evaluated at every trajectory point.
using DiffusionFn = std::function<double(std::size_t i, double t, double gamma)>;
DiffusionFn einstein_hold_diffusion(double sigma_eq2);

// Solve d sigma2/dt = 2 gamma sigma2 + 2 D on the trajectory grid. Over each step gamma and D are replaced by their
// step averages and the linear equation is integrated exactly, so constant gamma and D reproduce Eq. (48) to rounding.
std::vector<double> lalime_solve(const std::vector<double>& t, const std::vector<double>& gamma, const DiffusionFn& D, double sigma2_initial);
// Eq. (48): closed form for constant gamma, D.
double lalime_stationary(double gamma, double D, double sigma2_initial, double dt);

}  // namespace spf::linear
