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

## WP3 and WP3g - 3+1D solver, GPU validation and benchmark (2026-11-21)

Code: `include/spf/hydro3d.hpp` (`Hydro3D`), `src/hydro3d_io.cpp` (snapshots), `src/ic_random.cpp` (smooth random state), `apps/spf_bench.cpp`,
`apps/spf_compare.cpp`; tests `tests/test_hydro3d.cpp` (each test prints its numbers with the prefix `[WP3]`). Scheme and the points the Rischke papers
leave open: `docs/equations.md`, sections 7-9. Config keys added: `grid.nx`, `grid.ny`, `grid.nz` (64 each), `output.snapshot_every` (0).

### CPU acceptance tests

Numbers from the macOS arm64 workstation (Clang, Release), identical on `cpu-serial` and `cpu-openmp` (16 threads) unless stated.
Default parameters (`phoenical`, `eta_ad` 0.125, `lambda` 0.4) except where stated.

| Test | Quantity | Measured | Tolerance | Result |
|---|---|---|---|---|
| T3.1 | 1D shock tube (400 cells) and vacuum expansion (300 cells) embedded along x, y, z of a 3D grid (4 transverse cells, periodic), `fixed`; max relative deviation from `Hydro1D` | 0 for all six cases | 1e-12 | pass |
| T3.1 | same, `permuted` (reported) | 0 for all six cases | - | reported |
| T3.2 | periodic 48^3, smooth random state, 500 steps, `permuted`: rel. change of R, E; M_i/E_total | R 7.4e-15, E 2.5e-14; M_i/E <= 6.9e-17 (OpenMP: R 4.4e-16, E 6.9e-16) | 1e-11 | pass |
| T3.2 | recovery failures, E < \|M\| cells, clamps | 0, 0, 0 | 0 | pass |
| T3.3 | spherical blast, 64^3, dx = 0.2 fm, 40 steps, outflow: largest octant asymmetry, max \|a - mirror(b)\| / max\|field\| over the 7 reflections and fields R, M_i, E; `fixed` | 2.326e-14 | 1e-10 | pass |
| T3.3 | same, `permuted` (reported) | 2.871e-14 | - | reported |
| T3.3 | fixed vs permuted, max-norm max\|a-b\| / max\|field\|: R, Mx, My, Mz, E | 6.578e-02, 6.839e-02, 6.324e-02, 6.675e-02, 6.948e-02 | - | reported |
| T3.3 | fixed vs permuted, L1 measure sum\|a-b\| / sum\|a\| (fixed as reference): R, Mx, My, Mz, E | 1.686e-03, 2.776e-02, 1.967e-02, 2.777e-02, 2.337e-03 | - | reported |
| T3.4 | `EosColdThermal` two_phase, rho = 2 rho_s = 0.306 fm^-3, eps = eps_c + 0.5 eps_s = 0.361229301 GeV/fm^3, p = 0.045498074, vs2 = 0.293538362 (> 0), travelling wave, relative amplitude 1e-4, 64 x 4 x 4 cells, dx = 0.2 fm; phase velocity measured / analytic for m = 1..4 (k = 0.4909, 0.9817, 1.4726, 1.9635 fm^-1) | 0.999778, 0.999316, 0.998854, 0.996979 | within 2 % | pass |
| T3.4 | amplitude after one period / initial, m = 1..4 | 0.9998, 0.9973, 0.9871, 0.9639 | - | reported |
| T3.5 | smooth sound wave, p = eps/3, one wavelength, one period, 32 / 64 / 128 cells: relative L1 error of the eps fluctuation | 3.501e-02, 8.193e-03, 1.928e-03 | monotone decrease | pass |
| T3.5 | measured order 32->64, 64->128 | 2.095, 2.087 | report if < 1.5 | pass (about 2) |
| T3.6 | zero `p_extra` against the hook bypassed | bitwise identical | identical | pass |
| T3.6 | nonzero `p_extra` (0.05 GeV/fm^3, sinusoidal) changes the result (sanity check) | 2.554e-01 of max\|field\| | differs | pass |
| set_state | `EosColdThermal` (two_phase, maxwell), `EosTable` (synthetic, bilinear and bicubic), `EosIdeal`: conserved fields from `set_state_fn` (3D) and `Hydro1D::set_state` against values computed in a kernel; 3 steps without recovery failure | 3D 4.0e-16 to 6.0e-16, 1D 2.1e-16 to 4.3e-16 (relative to max\|field\|); 0 failures | 1e-13 | pass |
| T3.7 | 96^3, 100 steps, ideal: Serial | 3.196e6 cell-updates/s | - | recorded |
| T3.7 | 96^3, 100 steps, OpenMP 16 threads: ideal, cold_thermal_two_phase | 1.584e7, 7.243e6 cell-updates/s | - | recorded |

Test totals after the review fixes (this branch merged with `main`): 46 cases, 533 assertions on `cpu-serial` and on `cpu-openmp` (16 threads); `spf_compare` Serial vs OpenMP on the workstation: 0 in every field, totals <= 1.8e-14.

Notes

* T3.1 is bit-identical because a sweep along a direction with a uniform state and zero velocity is the identity (`Q+ + Q- = 1` exactly, zero differences), and the
  per-cell expressions of the 3D and the 1D solver are written identically (`|M|` as `sqrt(M_x^2 + M_y^2 + M_z^2)`, `v = c (M_i / |M|)`, `M` restored as `(M_i / |M|) E`).
* T3.3: the fixed and permuted sweep orders differ by up to 7 % of the maximum in the shock region of the blast. The octant symmetry of each order is at the rounding level.
* T3.4: the antidiffusion damps the wave by 3.6 % per period at k dx = 0.39; the phase velocity is within 0.3 % of the analytic value at every k.
* T3.5: the amplitude of the wave is 1e-6 of eps (linear regime); the time step of each run is `lambda <= 0.4` adjusted so that an integer number of steps is one period.
* The workstation and the GPU node differ: on the workstation Serial gives 3.2e6 cell-updates/s, about four times the Serial rate of the GPU node (7.4e5).

### Defects found in review, and their fixes

* **Host-side equation-of-state call (crashed on CUDA).** `Hydro3D::set_state_fn` and `Hydro1D::set_state` called `eos.p0(e, r)` on the host. `EosColdThermal`
  and `EosTable` read their tables through device pointers, so the host dereferenced device memory on the CUDA backend; the CPU builds hide this because host and
  device memory coincide. On the GPU node (before the fix) the CUDA test run had 34 cases run, 33 passed, 1 crashed (T3.4 with SIGSEGV), and `spf_bench` with
  `cold_thermal_two_phase` failed at 160^3 and 320^3. Fix: the host fills mirrors of the primitive fields, copies them to the device, and a named functor
  (`ConsFromPrim1K`, `ConsFromPrim3K`) computes `R, M_i, E`; `docs/device_rules.md` has a new rule 9. Every host-side call of an EoS member in the code base (include, src, apps, tests):
  1. `include/spf/hydro3d.hpp`, `Hydro3D::set_state_fn` (fixed);
  2. `include/spf/hydro1d.hpp`, `Hydro1D::set_state` (fixed);
  3. `tests/test_hydro1d.cpp`, round-trip test of `recover_primitives` (host call with the ideal EoS; moved into a kernel);
  4. `tests/test_hydro3d.cpp`, round-trip test of `recover_core` with `p_extra` (host call with the ideal EoS; moved into a kernel).
  All other EoS member calls in `tests/test_eos.cpp`, `tests/test_eos_table.cpp`, `tests/test_hydro3d.cpp` (T3.4 probe), `include/spf/eos/scaled.hpp` and the solver
  kernels are inside functors or kernels. `EosTable::check_range` and `ColdCurveTable::valid_range` read scalar members only and are host-callable.
  Regression test: "set_state / set_state_fn with EosColdThermal and EosTable" (above), which fails on CUDA without the fix; it was run on the CPU backends only.
* **Metric of the backend comparison.** `spf_compare diff` first used the per-cell relative deviation `|a-b| / max(|a|,|b|)`, which reaches 1e-9 where a momentum or velocity
  component passes through zero although the absolute difference is at the rounding level. The pass criterion of G2 is now `max |a-b| / max|field| <= 1e-11` per field and
  totals (R, E; M_i relative to the total E) `<= 1e-11`; the per-cell relative numbers are still printed for information.

### GPU node (WP3g)

Linux x86_64 node, 20 cores, 31 GB RAM, NVIDIA TITAN RTX (24 GiB), CUDA 12.9, GCC 13.4, Release presets, tree = `main` (b55ac1c) + commit 6d627da of this branch,
i.e. before the fix above. The GPU numbers below were measured by the reviewer.

| Test | Quantity | Measured | Tolerance | Result |
|---|---|---|---|---|
| G1 | `cuda-turing75`, `spf_tests` | before the fix: 34 cases run, 33 passed, 1 crashed (T3.4: host call of `eos.p0`, see above); fixed head 602437b: 46 cases / 533 assertions passed | all pass | pass after the fix |
| G1 | T3.2 on CUDA: rel. change of R, E; M_i/E | R 2.2e-16, E 0, M_i/E <= 2.0e-18 | 1e-11 | pass |
| G1 | T3.3 on CUDA: octant asymmetry fixed, permuted | 1.113e-13, 2.850e-14 | 1e-10 (fixed) | pass |
| G1 | `cpu-openmp`, 20 threads, same node | 45 cases, 513 assertions, all passed | all pass | pass |
| G2 | T3.2 state, 100 steps, Serial vs CUDA: max\|a-b\|/max\|field\| for R, Mx, My, Mz, E, eps, rho, vx, vy, vz, p | 1.232e-14, 1.559e-14, 1.238e-14, 1.366e-14, 4.245e-15, 4.287e-15, 1.221e-14, 1.568e-14, 1.321e-14, 1.441e-14, 4.344e-15 | 1e-11 | pass |
| G2 | totals R, E (M_i relative to E_total <= 1.7e-14) | 6.1e-15, 1.7e-14 | 1e-11 | pass |
| G2 | Serial vs OpenMP (20 threads) | 0 in every field; totals <= 1.8e-14 | 1e-11 | pass |
| G2 | per-cell relative deviation, Serial vs CUDA (information only) | up to 1.027e-09 (Mz), 2.581e-10 (Mx), 1.879e-10 (My, vy) where a component is near zero | not a criterion | reported |

Benchmark (`spf_bench`, periodic box, smooth random state, `permuted`, one untimed warm-up step). Time per step and cell-updates per second.
No other process used the GPU during the timed runs (checked with `nvidia-smi`).

| EoS | Grid | CUDA | OpenMP, 20 threads | Serial, 1 thread |
|---|---|---|---|---|
| ideal | 160^3 | 0.06518 s / 6.285e7 | 1.356 s / 3.021e6 | 5.529 s / 7.408e5 |
| ideal | 320^3 | 0.5011 s / 6.539e7 | 10.79 s / 3.037e6 | 43.62 s / 7.513e5 |
| cold_thermal_two_phase | 160^3 | 0.1077 s / 3.803e7 (fixed head 602437b) | 1.776 s / 2.306e6 | 12.43 s / 3.294e5 |
| cold_thermal_two_phase | 320^3 | 0.8266 s / 3.964e7 (fixed head 602437b) | 13.97 s / 2.346e6 | 97.77 s / 3.351e5 |

* 320^3 used 20 steps for CUDA and OpenMP and 3 steps for Serial; wall time for 150 steps is 150 times the time per step.
* Peak GPU memory 1018 MiB (160^3) and 6424 MiB (320^3); GPU utilisation 100 % during the ideal-EoS runs.
* Peak resident memory of one Serial process: 1.08 GiB (160^3), 8.27 GiB (320^3).

G4 throughput criterion (160^3 and 320^3, ideal EoS, events of 150 steps):

| Grid | (a) one GPU | (b) 20 cores | Ratio (a)/(b) |
|---|---|---|---|
| 160^3 | 368.2 events/h (3600 / (150 x 0.06518 s) taken from the table above) | 20 independent single-thread processes: 28.57 s per step each, 5.17 times slower than one process alone (memory-bandwidth bound), 16.80 events/h; OpenMP-20: 17.70 events/h | 21.9 |
| 320^3 | 47.9 events/h (measured) | estimate from one Serial process, ignoring contention: at most 1.10 events/h, because only 2 processes of 8.27 GiB fit in 31 GB; OpenMP-20: 2.22 events/h | >= 21.6 against OpenMP-20 |

Conclusion: on the GPU node the GPU is the production backend for both grid sizes; the criterion of the plan (events per hour of one GPU >= events per hour of the 20 cores)
is met by a factor of about 22 for the ideal EoS and about 16.5 for `cold_thermal_two_phase` (160^3: 222.8 events/h on the GPU vs 13.5 events/h with OpenMP-20). No default (preset, `run.backend`) was changed.

Rerun on the fixed head 602437b (GPU node, Release `cuda-turing75`): `spf_tests` 46 cases / 533 assertions passed, `ctest` 2/2 (including the set_state regression test
with `EosColdThermal` and `EosTable`). G2 Serial vs CUDA: worst max|a-b|/max|field| 1.647e-14, totals <= 1.736e-14 -> PASS (per-cell relative, information only: 5.8e-10).
Ideal EoS CUDA timings reproduced: 0.06566 s (160^3), 0.5056 s (320^3) per step. Conservation in the benchmark runs: relative change <= 1.9e-16.
