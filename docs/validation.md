# Validation log

Every number in this file is produced by a test or script in this repository. Entries are appended per work package.

## WP0 - repository bootstrap (2026-11-20)

Test set: `ctest` (2 tests: `spf_tests` with 9 doctest cases / 45 assertions, and `spf_info`, which runs a
`parallel_reduce` of 2e8 terms and checks it to 1e-7).

| Platform | Preset | Toolchain | Kokkos backend | Result |
|---|---|---|---|---|
| GitHub Actions `ubuntu-24.04` | `cpu-serial` | runner default | Serial | 2/2 tests passed (commit a3386f7) |
| GitHub Actions `ubuntu-24.04` | `cpu-openmp` | runner default | OpenMP | 2/2 tests passed (commit a3386f7) |
| GitHub Actions `macos-latest` | `cpu-serial` | runner default | Serial | 2/2 tests passed (commit a3386f7) |
| GitHub Actions `macos-latest` | `cpu-openmp` | runner default + Homebrew libomp | OpenMP | 2/2 tests passed (commit a3386f7) |
| macOS arm64 workstation, 16 cores | `cpu-serial`, `cpu-openmp` | conda Clang, CMake 4.5.0 | Serial, OpenMP (16 threads) | 2/2 tests passed each; 9/9 cases, 45/45 assertions |
| Linux x86_64 node, 20 cores, NVIDIA TITAN RTX | `cpu-openmp` | conda GCC 13.4.0, CMake 4.5.0, Ninja | OpenMP (20 threads) | 2/2 tests passed; 9/9 cases, 45/45 assertions (commit 2788999) |
| same node | `cuda-turing75` | as above + CUDA toolkit 12.9 (nvcc 12.9.86), driver 595.99 | Cuda | 2/2 tests passed; 9/9 cases, 45/45 assertions (commit 2788999) |

Notes

* The system CUDA toolkit of the GPU node (11.5) is older than the minimum of Kokkos 5.3.0 (nvcc 12.2); the build uses a
  conda environment (conda-forge: `cuda-version=12.9 cuda-nvcc cuda-cudart-dev cuda-driver-dev gxx_linux-64=13 cmake ninja git`).
* The `macos-14` runner image is no longer offered by GitHub Actions; CI uses `macos-latest`.
* The `spf_info` reduction is a smoke test of the backend, not a benchmark. Benchmarks are part of WP3g.

## WP2a - equation-of-state interface and analytic equations of state (2026-11-20)

Test source: `tests/test_eos.cpp` (doctest, inside `spf_tests`; 18 cases / 148 assertions with the existing ones). Every compared value is
evaluated in a Kokkos kernel. Measured with `cpu-serial` and `cpu-openmp` (16 threads) on the macOS arm64 workstation (identical printed values).
Cold-curve table: `data/eos/cold_curve.dat` (3201 rows, 0 to 16 rho_s, step 0.005 rho_s), made by `tools/eos/make_cold_curve.py`.

| Test | Quantity | Measured | Tolerance | Result |
|---|---|---|---|---|
| T2.0 | concept satisfied by `EosIdeal`, `EosGammaLaw`, `EosColdThermal`; trivially copyable | yes (static_assert and runtime) | - | pass |
| T2.1 | spinodals rhoA, rhoB (rho_s) | 5.49056, 8.61301 (JSON 5.48835, 8.610) | 0.01 rho_s | pass |
| T2.1 | coexistence rho_1, rho_2 (rho_s) | 4.49988, 10.45761 (JSON 4.49988, 10.45761) | 0.01 rho_s | pass |
| T2.1 | largest density difference of the four | 3.01e-3 rho_s | 0.01 rho_s | pass |
| T2.1 | mu_coex, p_coex relative to JSON | 2.0e-14, 6.4e-14 (1.700075 GeV, 0.379983 GeV/fm^3) | 1e-3 | pass |
| T2.2 | max abs(dp_c/drho) inside (rho_1, rho_2), 20 densities | 0 (exact) | 0 | pass |
| T2.2 | max relative p_c - p_coex inside | 0 | 1e-12 | pass |
| T2.2 | p0, dp_deps, dp_drho vs two_phase outside, 20 densities | max relative difference 0 | 1e-13 | pass |
| T2.2 | relative jump of p0 (fixed eps = 3 GeV/fm^3, rho +- 1e-9 fm^-3) at rho_1, rho_2 | 2.9e-11, 9.5e-10 | 1e-6 | pass |
| T2.3 | vs2 vs finite-difference isentrope, ideal / gamma_law (10 points each) | 8.5e-12 / 1.6e-11 | 1e-6 | pass |
| T2.3 | same, cold_thermal two_phase / maxwell (10 points each) | 5.5e-11 / 1.7e-10 | 1e-6 | pass |
| T2.4 | 21 reference points: max relative error of p0 / h | 1.9e-15 / 4.6e-16 | 1e-5 | pass |
| T2.4 | max abs error of vs2 | 1.07e-6 | 2e-3 | pass |
| T2.4 | (5.8 rho_s, 2.0 eps_s) | vs2 = -0.05428, h = 1.98840 GeV/fm^3 | - | pass |
| T2.5 | p_c(rho_s) | -7.0e-11 GeV/fm^3 | 1e-4 | pass |
| T2.5 | eps_c(rho_s)/rho_s - 0.940 GeV | -0.011872 GeV | 1e-4 around -0.01187 | pass |
| T2.6 | clamped region: max abs(p0 - p_c), max abs(dp_deps), max abs(dp_drho - p_c') | 0, 0, 0 (both variants) | exact | pass |

Notes

* T2.3 method: the isentrope through (eps, rho) is integrated with one RK4 step of d eps/d rho = (eps + p0)/rho; dp/drho along it is a central
  difference at +-delta and +-delta/2 with Richardson extrapolation, delta = 2e-5 fm^-3; vs2 = (rho/h) dp/drho. Points sit at cell centres of
  the table grid (cell width 7.65e-4 fm^-3) so the stencil stays inside one cubic piece.
* T2.1: coexistence by two-dimensional Newton iteration on equal mu_c and p_c, spinodals as the first and last sign changes of p_c' on 4.2 to 10.5 rho_s
  (bisection); all run in a single-thread kernel. The spinodals differ from the quintic-spline values by 2e-3 rho_s because p_c' of a cubic
  Hermite interpolant on the grid is only piecewise linear.
* T2.4 evaluates the EoS at eps = (eps/eps_s) * 0.940 GeV * 0.153 fm^-3 taken from the JSON entries.
