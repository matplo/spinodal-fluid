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
