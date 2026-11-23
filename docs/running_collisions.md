# Running collision simulations (WP7)

Step-by-step instructions for producing the results of Stage A: central Pb+Pb events from the SMASH cascade, deposited on a lattice, evolved with the
3+1D fluid code with the finite-range term, observables recorded every step, ensembles compared between the two-phase equation of state and its
Maxwell partner. Every number in the section "Cost" was measured (Stage A and the dx demonstration of `docs/validation.md`, section WP7 Stage A).

## 1. Building

Requirements: C++20 compiler (GCC >= 10.4 or Clang >= 14), CMake >= 3.22, git, Python 3 with numpy and matplotlib (plots only). For the GPU build the
CUDA toolkit >= 12.2 (tested: CUDA 12.9, GCC 13.4, NVIDIA TITAN RTX = `Kokkos_ARCH_TURING75`).

```
cmake --preset cpu-openmp       && cmake --build --preset cpu-openmp --parallel   # CPU, threads: OMP_NUM_THREADS=16
cmake --preset cpu-serial       && cmake --build --preset cpu-serial --parallel   # CPU reference
cmake --preset cuda-turing75 -G Ninja && cmake --build --preset cuda-turing75     # GPU (other cards: preset cuda + Kokkos_ARCH_*), about 6 min
ctest --preset <preset>                                                           # all tests (CPU openmp 520 s, serial 300 s, GPU 60 s)
```

The executables are in `build/<preset>/`; the driver is `spf_collide`. Kokkos flags (`--kokkos-num-threads=...`) are accepted.

## 2. Equation-of-state tables (finite temperature)

The cold-thermal EoS (`eos.model = cold_thermal_two_phase | cold_thermal_maxwell`) needs no table. The finite-temperature tables of WP2b
(`eos.model = table_two_phase | table_maxwell`) are generated once (16 processes; run times in `docs/eos_finite_T.md`) or unpacked from an archive:

```
python3 tools/eos/finite_T/make_finite_T_table.py all        # see docs/eos_finite_T.md; writes data/eos/generated/eos_{two_phase,maxwell}.{bin,json}
sha256sum data/eos/generated/*.bin
```

Expected SHA-256 of the `.bin` files (the reader verifies them against the `.json` header at every load, and refuses the table otherwise):

| table | sha256 |
|---|---|
| `eos_two_phase.bin` | `3702ebde72cd7071552549af6bbe69a84afd10841335552d1cf0cc32284a78bc` |
| `eos_maxwell.bin` | `ed86c63c2b3ca061730de4516f23a66e7dd30eb1ecccb770e54adb44a38aa52a` |

(Both values were re-measured on the GPU node and on the development machine from the tables used for Stage A.) The driver takes `eos.table_path`
if it is set, otherwise `eos.table_path_two_phase` / `eos.table_path_maxwell` according to `eos.model`; relative paths are relative to the source tree.

## 3. SMASH and the events

SMASH 3.4.2 and Pythia 8.316 are external programs (GPL; only their output files are read). Build recipe: `tools/smash/BUILD.md`.

```
python3 tools/collisions/gen_events.py --ekin 3 --nevents 20 --seed 1 --smash <smash build>/smash --out runs/events_e3
# -> runs/events_e3/smash_data/particle_lists.oscar : OSCAR2013, one block per event at the fluid start time t_init, events 0 ... 19
```

`--ekin` is the kinetic energy per nucleon [GeV] in the fixed-target frame (central, b = 0, cascade mode, frozen Fermi motion); the same `--seed` gives the same
events, so every equation of state is run on the same list. The output time is `t_init = 2 R / sqrt(gamma_cm^2 - 1)` with `R = 1.2 A^(1/3) fm`
(`ic.t_init_r0_fm`); the driver reads the block nearest to that time.

## 4. Running events

`spf_collide` (`apps/spf_collide.cpp`; a separate program, `spf_ic` stays the single-state tool of WP5) runs a list of events of one OSCAR file in one process:
deposit (WP5) -> `Hydro3D` -> `ObservableRecorder` (WP6) -> summary. Configuration: defaults `config/defaults.json`, then `--config FILE` (repeatable), then `key=value`
arguments. `config/collisions/` holds the files for collision runs:

| file | content |
|---|---|
| `common.json` | outflow boundaries, dx = 0.2 fm, bicubic table interpolation, `paper_pressure` gradient with a = 0.033 fm, sigma_init = 1 fm, `ic.max_outside_fraction = 1e-5`, vacuum threshold `shasta.e_vac_GeV_fm3 = 1e-8`, record every step, clump and phase histograms every 0.5 fm/c |
| `e1.json` ... `e6.json` | `ic.e_lab_gev` and the box in fm (section 6) |
| `stage_a.json` | `run.stop_at_fluid_end = false`: events run the full 6 fm/c (the fluid end is recorded, not enforced), as needed for moment curves of equal length |

Single event (CPU), then a batch over the 20 events of the file, resumable, on the GPU node:

```
build/cpu-openmp/spf_collide --out runs/one --config config/collisions/common.json --config config/collisions/e3.json \
    ic.oscar_path=runs/events_e3/smash_data/particle_lists.oscar eos.model=table_two_phase run.event_first=0 run.n_events=1

python3 tools/collisions/run_batch.py --exe build/cuda-turing75/spf_collide --oscar runs/events_e3/smash_data/particle_lists.oscar \
    --out runs/e3_table_two_phase --chunk 5 --config config/collisions/common.json --config config/collisions/stage_a.json \
    --config config/collisions/e3.json eos.model=table_two_phase
```

**Resume.** An event whose `event_NNN/summary.json` exists is skipped (`run.resume = true`); calling the same command again continues after an
interruption (an event that was interrupted has no summary and is run again from the start). An event that ended in failure is not run again unless
`run.retry_failed = true`. `--chunk N` of `run_batch.py` limits what an interruption can cost to N events.

**Ending an event.** (a) the fluid end: all cells have `eps < obs.fluid_end_eps_eps_s * eps_s` (default 5), counted only after a record had a cell at or above that value, and only if
`run.stop_at_fluid_end = true` (default); (b) `run.t_max_after_init_fm` (default 6) after the start of the fluid stage; (c) the number of primitive-recovery failures exceeds
`run.max_recovery_failures` (default 0): the event stops at once with status `recovery_failure`. Exit code of the program: 0 if every event ended with status `fluid_end` or `t_max`, 3 if some event failed.

## 5. Configuration keys used in collision runs

| key | default | meaning |
|---|---|---|
| `grid.dx_fm` | 0.2 | lattice spacing [fm]. **The only key to change for another resolution when the box is given in fm** (section 7) |
| `grid.Lx_fm`, `Ly_fm`, `Lz_fm` | 0 | box length [fm] per direction; if positive, the cell count is `round(L / dx)` (an error if L is not a multiple of dx); if 0, `grid.nx/ny/nz` are used |
| `grid.nx`, `ny`, `nz` | 64 | cell counts when no box length is given |
| `grid.boundary` | periodic | `outflow` for collisions (`common.json`) |
| `eos.model` | table_two_phase | `table_two_phase`, `table_maxwell` (finite-T tables), `cold_thermal_two_phase`, `cold_thermal_maxwell`: **one key selects the equation of state** (table path from `eos.table_path_*`) |
| `eos.table_interpolation` | bilinear | `bicubic` in `common.json`: the first derivatives of a bilinear interpolant of p0(eps, rho) jump across cell boundaries, so v_s^2 is discontinuous from cell to cell and its sign in the narrow spinodal region is resolved only to the table spacing; the bicubic Hermite patches have continuous first derivatives (WP2c; the sign tests of v_s^2 in WP2b use bicubic) |
| `gradient.form`, `gradient.a_fm` | paper_pressure, 0.033 | finite-range term, range a [fm] (0 / 0.01 / 0.033 / 0.05 in the paper) |
| `gradient.pex_floor_kappa` | 0.5 | recovery floor of the gradient pressure (decision S17); a negative value switches it off |
| `ic.sigma_init_fm` | 1.0 | Gaussian width of each hadron [fm] (0.7, 0.5 in the paper's variants) |
| `ic.gaussian_contraction` | particle_gamma_z | Lorentz contraction of the Gaussians (`none`, `particle_gamma_z`, `cm_gamma`) |
| `ic.e_lab_gev` | 3.0 | kinetic energy per nucleon (decides `t_init`; must match the SMASH file) |
| `ic.t_init_r0_fm` | 1.2 | radius parameter of `t_init` (R = r0 A^(1/3)) |
| `ic.oscar_path`, `ic.oscar_event` | "", 0 | OSCAR file; the driver sets the event number itself |
| `ic.max_outside_fraction` | 1e-6 (library), 1e-5 (collision files) | the event is rejected (status `error`) if more baryon number than this falls outside the box at deposition |
| `run.event_first`, `run.n_events` | 0, 1 | event slice of the file (`n_events = -1`: all events) |
| `run.t_max_after_init_fm` | 6 | maximum fluid time after `t_init` |
| `run.stop_at_fluid_end` | true | stop at the fluid end |
| `run.max_recovery_failures` | 0 | stop the event when more primitive-recovery failures (cells, summed over all sweeps) occur |
| `run.resume`, `run.retry_failed` | true, false | skip finished events / run failed ones again |
| `run.marginals` | true | write `marginals.json` (baryon number per slab at start and end; used by the box study) |
| `shasta.e_vac_GeV_fm3` | 1e-12 (library), 1e-8 (collision files) | cells with energy density below this are treated as vacuum; see section 9 |
| `obs.every_steps` | 1 | recording interval in steps (dt = 0.4 dx = 0.08 fm/c at dx = 0.2) |
| `obs.hist_times_fm` | [] (library), every 0.5 fm to 6 (collision files) | times (after `t_init`) at which clump and phase-plane histograms are written |
| `obs.clump_rho_min_rho_s`, `obs.fluid_end_eps_eps_s` | 7, 5 | clump threshold, fluid-end criterion |
| `obs.phase_*` | 0.1 rho_s, 0.25 eps_s bins | phase-plane histogram of the baryon number |

Every program writes `config_used.json` (effective configuration and the git description of the code) into its output directory, and each event directory as well.

## 6. Box for 1 to 6 A GeV

Dimensions in fm, chosen from 24 events (4 per energy, SMASH, cold-thermal two-phase, 6 fm/c, generous box 50 x 50 x 64 fm, dx = 0.2 fm; `tools/collisions/box_study.py` on
the `marginals.json` files): the smallest half-widths X (x and y) and Z for which the baryon fraction outside |x|, |y| < X, |z| < Z (sum of the three one-direction
fractions at the end of the run, an upper bound) stays below 1e-4 for every one of the four events and for the initial state, plus 2 fm on each half-width:

| E_lab [A GeV] | t_init [fm/c] | minimum box (< 1e-4) | box in `config/collisions/eN.json` [fm] | cells at dx = 0.2 | outside fraction at that box (max of 4 events) |
|---|---|---|---|---|---|
| 1 | 19.50 | 40 x 40 x 34 | 44 x 44 x 38 | 220 x 220 x 190 | 8.9e-10 |
| 2 | 13.79 | 36 x 36 x 30 | 40 x 40 x 34 | 200 x 200 x 170 | 8.0e-11 |
| 3 | 11.26 | 34 x 34 x 30 | 38 x 38 x 34 | 190 x 190 x 170 | 1.1e-07 |
| 4 | 9.75 | 34 x 34 x 28 | 38 x 38 x 32 | 190 x 190 x 160 | 2.3e-07 |
| 5 | 8.72 | 34 x 34 x 26 | 38 x 38 x 30 | 190 x 190 x 150 | 1.0e-08 |
| 6 | 7.96 | 34 x 34 x 26 | 38 x 38 x 30 | 190 x 190 x 150 | 1.6e-09 |

The baryon number that actually left the box in the production runs is a result of every run (`baryon_leaving_fraction` in `summary.json`; Stage A: section 10 of `docs/validation.md`).
Only 4 events per energy entered the study: check `baryon_leaving_fraction` of a larger sample before using a box for Stage B.
(The 1 A GeV box of WP5, 200 x 200 x 240 cells, was set by an initial-state criterion of 1e-6; here `ic.max_outside_fraction = 1e-5` and the budget is 1e-4 until the end time.)

## 7. Changing dx

Because the box is given in fm, changing `grid.dx_fm` changes the cell counts and not the physical box; nothing else needs to be changed (the time step is `dt = 0.4 dx`, the number of steps for the
same end time doubles when dx is halved):

```
build/cuda-turing75/spf_collide ... grid.dx_fm=0.1          # 3 A GeV: 380 x 380 x 340 cells instead of 190 x 190 x 170
```

The box lengths must be multiples of dx (else an error). Tests: `T7.1`, `T7.3` in `tests/test_collide.cpp`. Memory grows with the cell count (about 330 bytes per cell on the GPU, section 10); 0.1 fm at 3 A GeV needs 14.2 GB of the 24 GB card.

## 8. Output files

Per run directory: `config_used.json`, `run_index.json` (status per event of the last call), and per event `event_NNN/`:

| file | content |
|---|---|
| `summary.json` | event id, `e_lab_gev`, `t_init_fm`, `status` (`fluid_end`, `t_max`, `recovery_failure`, `error`), `steps`, `time_after_init_fm`, `fluid_end_time_fm`, `recovery_failures` (initial state + run), `floor_activations`, `ic_limiter_cells`, `limiter_cells_run`, `baryon_leaving_fraction`, `energy_leaving_fraction` (1 - final / initial integral on the grid), `max_enhancement` (N = 2 ... 7 with time), `unstable_fraction_peak`, wall times (`wall_ic_s`, `wall_evolve_s`, `wall_observe_s`, `wall_total_s`, `wall_per_step_s`), `peak_rss_MB`, `backend`, grid, code version |
| `observables.csv` | one line per step: `t_fm, A, mom2 ... mom7, U_perp, unstable_fraction, eps_max_eps_s, step` (`docs/observables.md`); `unstable_fraction` is the baryon fraction in cells with v_s^2 < 0 (the isentropic spinodal of the EoS in use) |
| `observables_summary.json` | recorder summary (fluid end, maximum enhancement) |
| `clumps_step<S>.json`, `phase_step<S>.json` | clump histogram and phase-plane histogram at the first record at or after each `obs.hist_times_fm` |
| `marginals.json` | baryon number per slab along x, y, z at start and end |
| `config_used.json` | configuration echo of the event |

`recovery_failures` counts cells whose primitive-variable recovery did not converge, summed over all sweeps; `floor_activations` counts cells in which the recovery floor on the gradient pressure acted (decision S17); `limiter_cells_run` counts cells in which the E >= |M| limiter acted.

## 9. Aggregation and plots

```
python3 tools/collisions/aggregate.py --out agg --run two_phase=runs/e3_table_two_phase --run maxwell=runs/e3_table_maxwell
python3 tools/collisions/plot_stage.py --agg agg --tp two_phase --mx maxwell --out plots --tag table_3gev --paper tools/data
python3 tools/collisions/plot_scan.py --out plots --paper tools/data --point 3.0:two_phase=agg/two_phase --point 3.0:maxwell=agg/maxwell   # Stage B: one --point per energy
```

`aggregate.py` calls `tools/obs_ensemble.py` (mean and standard error of the enhancement of N = 2 ... 7 versus `t - t_init`, per-event maxima) and writes `events_summary.csv`, `run_summary.json`
(totals of failures, floor activations, mean losses, wall times), `provenance.json` (configuration echo, git description). `plot_stage.py` draws the moment curves
(two-phase solid, Maxwell dashed, s.e. bands, with the digitised Fig. 5 of arXiv:1302.2956), the distribution of the per-event maxima, the baryon fraction in the isentropic spinodal versus time, and the clump-size
distribution (two-phase minus Maxwell at the time of the maximum of the mean N = 7 enhancement, with the digitised Fig. 7 of arXiv:1302.2956); every figure carries the git description of the code.
The digitised curves are in `tools/data/p1209_fig3_*, p1209_fig4_*, p1302_fig5_*, p1302_fig6_*, p1302_fig7_*` (`tools/digitise_wp7.py`, headers give source, page, path colour and the axis-fit residual).

## 10. Vacuum threshold and recovery failures

With the finite-temperature tables and the library default `shasta.e_vac_GeV_fm3 = 1e-12`, the primitive recovery fails in cells at the very dilute edge of the matter distribution
(E from 1e-11 to 5e-5 eps_s, R from 1e-12 to 1e-5 rho_s, |M|/E = 1; together 7.7e-7 baryons and 2.3e-6 GeV in 1642 cells after the first step), which the cold-thermal EoS does not show.
Measured on event 0 of the 3 A GeV file (table two-phase, CPU, 13 steps): 1241 failures at 1e-12, 294 at 1e-11, 64 at 1e-10, 0 at 1e-9 (and 2 failures over the full 75 steps at 1e-9, 0 at 1e-8).
The collision files therefore set `shasta.e_vac_GeV_fm3 = 1e-8` (7e-8 eps_s): cells below it are vacuum. The effect on the baryon number and energy is below 1e-6 of the totals
(0.2 fm, 13 steps: baryon loss 2.2e-9 and energy loss 1.4e-9 at 1e-9; 1.7e-7 and 2.5e-7 at 1e-7). The same value is used for the cold-thermal runs, so that all four equations of state are treated alike.
A run is never allowed to continue past a failure silently: with `run.max_recovery_failures = 0` the event stops with status `recovery_failure`.

## 11. Cost per event (3 A GeV, dx = 0.2 fm, 190 x 190 x 170 cells, 75 steps, observables every step, 13 clump and phase histograms)

GPU node (NVIDIA TITAN RTX, Release `cuda-turing75`), 20 events per equation of state, 6.137e6 cells, 75 steps of 0.08 fm/c, observables every step, 13 histogram records:

| equation of state | wall per event (mean, min - max) | stepping | per step | observables + histograms | GPU memory (peak, nvidia-smi) |
|---|---|---|---|---|---|
| cold-thermal two-phase | 12.04 s (11.77 - 12.29 s) | 8.64 s | 0.1152 s | 2.96 s | 2042 MiB (during the whole Stage A job) |
| cold-thermal Maxwell | 12.29 s | 8.85 s | 0.1181 s | | |
| finite-T table two-phase (bicubic) | 25.34 s (24.28 - 26.27 s) | 19.78 s | 0.2637 s | 5.11 s | |
| finite-T table Maxwell (bicubic) | 25.30 s | 19.78 s | 0.2637 s | | |

CPU, same event (event 0, table two-phase, bicubic, 75 steps; macOS arm64, `cpu-openmp`, 16 threads, `OMP_PROC_BIND=spread`): 68.5 s per event (stepping 54.3 s = 0.724 s per step, observables 13.9 s, host memory 2081 MB); the GPU needs 24.8 s for this event (stepping 19.4 s, observables 5.0 s), a factor 2.8 in wall time and 2.8 in stepping.
Resolution 0.1 fm (same event, table two-phase): 380 x 380 x 340 = 4.91e7 cells, 150 steps, 329.3 s (stepping 270.8 s = 1.805 s per step, observables 55.5 s), 14184 MiB of GPU memory (2042 MiB at 0.2 fm).
Stage B estimate (not measured at the other energies): 2400 events with the table at about 25 s is about 17 hours of GPU time at dx = 0.2 fm.

## 12. Troubleshooting

* **`status: recovery_failure`**: the event stopped at the first cell failure. Look at `recovery_failures` and `recovery_failures_initial_state` in `summary.json`. Failures in the first step at the edge of the matter: check `shasta.e_vac_GeV_fm3` (section 10);
  failures inside the matter: report them (the configuration echo is in the event directory); do not raise `run.max_recovery_failures` without reading `floor_activations` and the failing-cell diagnostics of `spf_ic --evolve` (it prints energy, density, |M|/E and radius of the failing cells).
* **`error: baryon number outside the box`** at the start: the box is too small for the energy or the event; enlarge `grid.L*_fm` or raise `ic.max_outside_fraction` consciously.
* **`baryon_leaving_fraction` above 1e-4**: enlarge the box (section 6), check `marginals.json` (`tools/collisions/box_study.py` gives the smallest box for a limit).
* **`grid.Lx_fm = ... is not a multiple of grid.dx_fm`**: choose a box length that is a multiple of dx.
* **`eos.table_path is empty` / table checksum error**: generate the tables (section 2) or set `eos.table_path`; the reader refuses a table whose SHA-256 differs from its header.
* **CUDA out of memory**: about 330 bytes per cell; use a smaller box or a larger dx.
* **Different results on CPU and GPU in the 6th digit**: expected (summation order); compare ensembles, not single cells.
