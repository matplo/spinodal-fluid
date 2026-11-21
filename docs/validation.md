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

## WP1 - SHASTA and the 1D relativistic solver (2026-11-20)

Code: `include/spf/shasta.hpp`, `include/spf/hydro1d.hpp`; tests `tests/test_shasta.cpp`, `tests/test_hydro1d.cpp`
(each test prints its numbers with the prefix `[WP1]`; run `spf_tests` to see them). Scheme: `docs/equations.md`.
Numbers below are from the macOS arm64 build, Serial backend (the OpenMP build gives identical values, see T1.5).
Setup of the shock tube: box 100 fm, discontinuity at 50 fm, t = 40 fm, outflow boundaries, `p = eps/3`,
`eps` = 10 | 1 GeV/fm^3, `rho` = 1 | 0.5 fm^-3. Exact solution: `eps* = 3.1398315582`, `v* = 0.4633814654`,
`V_shock = 0.7521153854`, fan tail at `zeta = -0.1555959083`. L1 = sum |eps_num - eps_exact(cell centre)| dx in GeV/fm^2.

| Test | Quantity | Measured | Tolerance | Result |
|---|---|---|---|---|
| T1.1 square, 100/200/400 cells | relative change of grid sum | 0 / 5.3e-16 / 9.8e-15 | 1e-12 | pass |
| T1.1 square | max, min after one crossing | 1 -> 1, 0 -> 0 | no new extrema, 1e-12 | pass |
| T1.1 Gaussian, 100/200/400 cells | relative change of grid sum | 4.3e-16 / 4.3e-16 / 7.1e-16 | 1e-12 | pass |
| T1.1 Gaussian | max / min never exceeded (all three resolutions) | max 0.99501 / 0.99875 / 0.99969 unchanged; min 5.2e-22 / 3.2e-22 / 2.5e-22 unchanged | 1e-12 | pass |
| T1.1 Gaussian (sigma = 0.05 L) | L1 error at 100 / 200 / 400 cells | 7.156e-3 / 1.929e-3 / 5.333e-4 | monotonic decrease | pass |
| T1.1 Gaussian | measured order 100->200, 200->400 | 1.89, 1.86 | reported | - |
| T1.1 square | L1 at 100 / 200 / 400 cells (reported) | 3.03e-2 / 1.86e-2 / 1.15e-2 | reported | - |
| T1.2 shock tube | L1 error at 200 / 400 / 800 cells | 3.1445 / 1.5925 / 0.81574 | monotonic decrease | pass |
| T1.2 | measured order 200->400, 400->800 | 0.981, 0.965 | reported | - |
| T1.2 | relative change of total E (200/400/800) | 8.3e-16 / 0 / 1.2e-15 | 1e-11 | pass |
| T1.2 | relative change of total R (200/400/800) | 9.5e-16 / 9.5e-16 / 1.9e-16 | 1e-11 | pass |
| T1.2 | recovery failures; cells with E < abs(M); clamp events | 0; 0; 0 | none | pass |
| exact solution check | Rankine-Hugoniot residuals | 4.9e-16, 2.5e-16 | 1e-12 | pass |
| primitive recovery round trip | 96 states (v up to 0.999, eps 1e-6 .. 50) | 0 failures, worst combined error 4.4e-11 | 1e-9 | pass |
| T1.3 phoenical | L1 of T^00 over -2<=zeta<=3 (relative to int T^00) | 2.075e-2 (1.04%) | reported | - |
| T1.3 phoenical | L1 of local eps (relative) | 1.850e-2 (1.04%) | reported | - |
| T1.3 explicit | L1 of T^00 (relative) | 2.226e-2 (1.11%) | reported | - |
| T1.3 explicit | L1 of local eps (relative) | 1.907e-2 (1.07%) | reported | - |
| T1.3 phoenical / explicit | int T^00 dzeta at zeta>1 (fraction of int over -1..1); max E there | 9.52e-4 (9.5e-4); 2.7e-2 / 1.05e-3 (1.1e-3); 2.9e-2 (eps0 = 1) | reported | - |
| T1.3 both | recovery failures; cells with E < abs(M) at the end | 0; 0 | none | pass |
| T1.3 | times the E >= abs(M) restriction acted (100 steps, 2 sub-steps each) | 200 (phoenical), 199 (explicit) | reported | - |
| T1.4 shock tube L1 at 400 cells, phoenical, eta 0.125, lambda 0.4 / 0.2 | | 1.5925 / 1.9194 | runs, selectable | pass |
| T1.4 phoenical, eta 0.1, lambda 0.4 / 0.2 | | 3.7501 / 5.8571 | | pass |
| T1.4 explicit, eta 0.125, lambda 0.4 / 0.2 | | 1.8014 / 2.1596 | | pass |
| T1.4 explicit, eta 0.1, lambda 0.4 / 0.2 | | 3.8306 / 5.9734 | | pass |
| T1.5 Serial vs OpenMP (16 threads), 400 cells | max relative deviation of eps, rho, v (vs max abs v) | 0, 0, 0 (profile files byte-identical) | 1e-12 | pass |

Notes

* The T1.2 order is close to 1 because the error is dominated by the fan (about 76% of the L1 error at 400 cells lies between the
  fan head and tail: 1.207 of 1.5925) and by the shock layer (0.25); the plateau region contributes 0.13. The profile follows the
  exact solution qualitatively (checked by plot).
* Reducing `eta_ad` from 0.125 to 0.1 or `lambda` from 0.4 to 0.2 increases the T1.2 error (more residual numerical diffusion).
* T1.3 has no acceptance bound in the work-package brief; the test only carries a loose regression guard (relative L1 < 10%, acausal
  fraction < 5%). The explicit and phoenical results are close; in this test the phoenical variant is slightly better.
* The `E >= abs(M)` restriction acts in the vacuum front of T1.3 (about 2 cells per step); it never acts in T1.1 or T1.2.
* The reference file `tests/data/shock400_reference.csv` was written by the Serial build of this branch
  (`SPF_WRITE_REFERENCE=1 spf_tests -tc="T1.5*"`); it checks that other builds reproduce it.

### WP1 regression: Release CUDA build crashed in the 1D solver tests (found in review, fixed)

* Symptom: with the preset `cuda-turing75` (Release, CUDA 12.9, GCC 13.4, TITAN RTX) every test case that uses `Hydro1D`
  (T1.2, T1.3, T1.4, T1.5) ended with SIGSEGV; T1.1 and the equation-of-state tests passed. The `RelWithDebInfo` build
  (-O2 -g) with and without Kokkos debug checks passed, as did the CPU builds and a CPU build with AddressSanitizer and
  UBSan (28 cases, 264 assertions).
* Diagnosis: a Release build with `-g` (same -O3 as Release) under gdb showed a call through a null function pointer
  (`pc = 0x0`) from nvcc's host-side wrapper of an extended lambda (`__nv_hdl_wrapper_t`) in the first `fill_ghost` launch
  made from `test_hydro1d.cpp`. `fill_ghost`, `shasta_step` and the field reductions were non-template inline functions in
  `shasta.hpp` containing `KOKKOS_LAMBDA`s and were compiled in two translation units (`test_shasta.cpp`,
  `test_hydro1d.cpp`); in the optimised build the copy selected at link time had an unset wrapper pointer.
* Fix: all kernels of `shasta.hpp` and `hydro1d.hpp` are now named functor structs with `KOKKOS_INLINE_FUNCTION` call
  operators (no extended lambdas in the library headers). Algorithm and arithmetic are unchanged; the CPU results are
  identical (Serial and OpenMP reproduce the T1.5 reference with deviation 0).
* GPU result after the fix (the GPU node, Release preset `cuda-turing75`, Kokkos Cuda backend):

| Tree | Result |
|---|---|
| WP1 branch alone | `ctest` 2/2 passed; `spf_tests` 19 cases, 161 assertions passed; also with `CUDA_LAUNCH_BLOCKING=1` (T1.* cases: 7 cases, 103 assertions) |
| WP1 headers on top of the EoS package (28 test cases) | `ctest` 2/2 passed; `spf_tests` 28 cases, 264 assertions passed |
| T1.2 L1(eps), 200 / 400 / 800 cells | 3.144507 / 1.592549 / 0.8157396 GeV/fm^2 (identical to the CPU values), order 0.981, 0.965 |
| T1.3 relative L1 of T^00 (phoenical / explicit) | 1.04% / 1.11% (CPU: 1.04% / 1.11%); restriction count 200 / 199 |
| T1.4 L1 at 400 cells, 8 variants | identical to the CPU values in the table above |
| T1.5 Cuda vs stored reference, 400 cells | max relative deviation eps 7.1e-14, rho 1.6e-13, v 4.1e-14 (tolerance 1e-12) |
| T1.1 | conservation <= 1.0e-14, extrema unchanged, Gaussian L1 7.156e-3 / 1.929e-3 / 5.333e-4 |

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

## WP2c - tabulated equation of state and scaled-EoS wrapper (2026-11-20)

Test source: `tests/test_eos_table.cpp` (doctest, inside `spf_tests`; 36 cases / 399 assertions with all existing ones). Every compared evaluation runs
inside a Kokkos kernel. CPU: `cpu-serial` and `cpu-openmp` (16 threads) on the macOS arm64 workstation (identical printed values). GPU: Release preset
`cuda-turing75` (CUDA 12.9, GCC 13.4) on the GPU node (NVIDIA TITAN RTX), `ctest` 2/2 passed, `spf_tests` 36 cases / 399 assertions passed.
Committed table: `tests/data/synthetic_eos.{json,bin}`, 201 x 401 lattice (x in [0, 40], dx = 0.2; y in [0, 16], dy = 0.04), 2.58 MB, synthetic, reader tests only.
"Default lattice" is the plan lattice (x in [0, 40], dx = 0.05; y in [0, 16], dy = 0.02: 801 x 801), generated in the test from `data/eos/cold_curve.dat`.

| Test | Quantity | Measured (CPU) | Measured (GPU) | Tolerance | Result |
|---|---|---|---|---|---|
| T2c.0 | `EosTable`, `EosScaled<EosTable>`, `EosScaled<EosColdThermal>` satisfy `EquationOfState`, trivially copyable | yes | yes | - | pass |
| T2c.1 | sizes, lattice, checksum verified; truncated, long, bad checksum (header and data), 10 missing-key / bad-version files rejected | all rejected with messages | same | - | pass |
| T2c.2 | nodes: max relative deviation of p0, T, mu, s, bilinear / bicubic (3894 nodes each) | 0 / 0 | 0 / 0 | 1e-14 | pass |
| T2c.3 | default lattice, bilinear: p0 rel. error max / median | 1.7282e-04 / 8.1699e-07 | same | max < 1e-3 | pass |
| T2c.3 | default lattice, bicubic: p0 rel. error max / median | 2.9136e-06 / 2.0766e-09 | same | max < 1e-3 | pass |
| T2c.3 | default lattice, bilinear: vs2 abs. error max / median / max in spinodal region | 6.8834e-03 / 8.5701e-05 / 3.8578e-03 | same | reported | - |
| T2c.3 | default lattice, bicubic: vs2 abs. error max / median / max in spinodal region | 6.6579e-04 / 3.3835e-07 / 1.9660e-05 | same | reported | - |
| T2c.3 | committed lattice, bilinear: p0 rel. error max / median | 5.3377e-04 / 3.2917e-06 | same | reported | - |
| T2c.3 | committed lattice, bicubic: p0 rel. error max / median | 2.9336e-05 / 1.7550e-08 | same | reported | - |
| T2c.3 | committed lattice, bilinear: vs2 abs. error max / median / spinodal max | 1.2075e-02 / 1.7425e-04 / 8.1090e-03 | same | reported | - |
| T2c.3 | committed lattice, bicubic: vs2 abs. error max / median / spinodal max | 2.9332e-03 / 1.5748e-06 / 7.8229e-05 | same | reported | - |
| T2c.4 | clamp: max abs(p0(eps<eps_min) - p0(eps_min)), max abs(dp_deps) (500 points), both interpolations | 0, 0 | 0, 0 | exact | pass |
| T2c.4 | clamped dp_drho vs centred difference, bilinear / bicubic (relative to max(abs(fd), 1e-3)) | 1.64e-08 / 1.63e-08 | same | 1e-5 | pass |
| T2c.5 | 200 interior points: dp_deps vs centred difference (step 1e-6 relative), bilinear / bicubic | 2.223e-10 / 2.283e-10 | same | 1e-6 | pass |
| T2c.5 | dp_drho, bilinear / bicubic (156 points with abs(dp_drho) >= 1e-3 GeV) | 1.318e-08 / 8.124e-08 | same | 1e-6 | pass |
| T2c.5 | dp_drho absolute difference at the 44 points with abs(dp_drho) < 1e-3 GeV, bilinear / bicubic | 6.05e-11 / 6.57e-11 GeV | same | 1e-9 GeV | pass |
| T2c.6 | four identities of the wrapper (p0, dp_deps, dp_drho, vs2), b = c = 1/1.3 and b = 0.8, c = 1.25, EosColdThermal and EosTable (both interpolations); 100 points | 0 (all) | at most 3.5e-16 (EosColdThermal p0) | 1e-13 | pass |
| T2c.6 | T/b, (c/b) mu, s identities for EosTable | 0 | 0 | 1e-13 | pass |
| T2c.6 | vs2 vs finite-difference isentropic derivative of wrapped p0 (20 points), max over all 6 cases | 3.31e-11 | 2.47e-11 (EosColdThermal, b = c = 1/1.3; table cases same as CPU to 3 digits) | 1e-6 | pass |
| T2c.6 | b = c = 1 minus base, max absolute difference | 0 | 0 | exact | pass |
| T2c.7 | extrapolate = false: max abs(p0(outside) - p0(clamped argument)); check_range reports all four violations | 0; reported | 0; reported | exact | pass |
| T2c.7 | extrapolate = true: linear table reproduced (36 points inside and outside), p0, T, mu, s; bilinear / bicubic | 1.20e-14 / 1.20e-14 | 1.11e-14 / 1.11e-14 | 1e-12 | pass |
| T2c.7 | extrapolate = true: dp_deps, dp_drho; bilinear / bicubic | 2.70e-14 / 1.32e-14 | 8.49e-16 / 5.82e-15 | 1e-12 | pass |
| GPU | `ctest` and `spf_tests` on `cuda-turing75` at commit 3949d11 | - | ctest 2/2; 36 cases, 399 assertions, all passed | all pass | pass |
| GPU | T2c.3 error numbers equal CPU to three significant digits | - | all printed values (four lines) identical | 3 digits | pass |

Notes

* Random points of T2c.3: fixed seed, 2000 points, rho uniform in [0.2, 16] rho_s, eps uniform between the analytic cold curve and 40 eps_s; the spinodal-region subset (5.49 < rho/rho_s < 8.61) has 425 points.
* The bilinear vs2 error is dominated by the discontinuity of the bilinear derivatives at cell edges; bicubic reduces the spinodal-region vs2 error by about 200 times on the default lattice.
* T2c.5 compares the derivative with a centred difference of p0; for dp_drho on the conformal plasma branch (p = eps/3) the exact value is zero, so points with abs(dp_drho) < 1e-3 GeV are compared in absolute terms.
* T2c.6 and T2c.7 differences at the 1e-16 level between CPU and GPU are rounding-level differences between the host and device floating-point code paths.
* Reading of Steinheimer and Randrup, Phys. Rev. C 87, 054903, Sect. V C: Eqs. (17)-(19) agree with the brief (T_bc = T_0/b, mu_bc = (c/b) mu_0, p_bc = p_0/b, v_s,bc = v_s,0; entropy density s_bc = s_0(b eps, c rho), scaled critical point (eps_crit/b, rho_crit/c)). The derivative identities dp_deps_bc and dp_drho_bc follow from the chain rule and are not in the paper. The identity p = T s - eps + mu rho (hence the paper's p_bc) requires a thermodynamically consistent base; the synthetic table is not, and the wrapper applies the four rules of the brief literally.
