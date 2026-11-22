# Decision register

Every modelling choice is a named key in `config/defaults.json` (or a build preset) with its alternatives.
`*` marks the default. A superseded choice is appended as a new row; rows are never deleted.
"planned" = key reserved, alternatives not implemented yet; "implemented" = code and at least one test exist.

| ID | Key | Alternatives | Status |
|---|---|---|---|
| S01 | `ic.generator` | `smash_cascade`* / `oscar_file` / `glauber_gaussians` / `analytic` | planned (WP5) |
| S02 | `ic.t_init_radius_mode`, `ic.t_init_r0_fm`, `ic.t_init_radius_fm` | `r0_A13`* with r0 = 1.2 fm (R = r0 A^(1/3)) / `fixed` with R = 6.62 fm / explicit time | planned (WP5) |
| S03 | `ic.sigma_init_fm` | 1.0* / 0.7 / 0.5 | planned (WP5) |
| S04 | `ic.gaussian_contraction` | `undecided`* (to be fixed in WP5 from arXiv:0710.0332 and arXiv:0806.1695) / `none` | planned (WP5) |
| S05 | `eos.model`, `eos.scale_b`, `eos.scale_c` | `table_two_phase`* / `table_maxwell` / `cold_thermal_two_phase` / `cold_thermal_maxwell` / `ideal` / `gamma_law`; b = c = 1* | analytic and cold-thermal variants done (WP2a); table reader (`table_two_phase`, `table_maxwell`) and scaled wrapper `EosScaled` implemented and tested (WP2c); physical tables planned (WP2b) |
| S06 | `eos.table_interpolation` | `bilinear`* / `bicubic` (Hermite, centred-difference node derivatives) | implemented and tested (WP2c); `eos.extrapolate` (`false`*: clamp + `check_range`; `true`: linear extrapolation) implemented (WP2c) |
| S07 | `gradient.form`, `gradient.a_fm`, `gradient.stencil` | `paper_pressure`* / `korteweg_full` / `off`; a = 0.033 fm* / 0 / 0.01 / 0.05; `second_order`* / `fourth_order` | `paper_pressure` and `off` implemented and tested (WP4; `off` is bit-identical to WP3); `korteweg_full` stops with "not implemented, see plan WP9"; 4th-order stencil implemented (ghost width 2 of 3) and tested (kernel accuracy, T4.0; growth rates, `docs/validation.md`) |
| S08 | `gradient.in_primitive_recovery` | `lagged`* / `force_only` | both implemented and tested (WP4); `lagged` is consistent with the conserved-field definitions (total pressure in `E`, `M`, recovery and fluxes); growth rates differ by < 1e-6 between them (T4.1) |
| S09 | `shasta.antidiffusion`, `shasta.eta_ad`, `shasta.lambda`, `shasta.sweep_order`, `grid.dx_fm` | `phoenical`* / `explicit`; 0.125*; 0.4*; `permuted`* / `fixed`; 0.2 fm* / 0.1 fm | antidiffusion, eta_ad, lambda implemented and tested (WP1); `grid.boundary` (`periodic`* / `outflow`) added; `shasta.sweep_order` (`permuted`*, `fixed`), `grid.dx_fm`, `grid.nx/ny/nz` and `output.snapshot_every` implemented and tested (WP3; `docs/validation.md`) |
| S10 | `obs.moment_density`, `obs.clump_rho_min_rho_s`, `obs.clump_connectivity`, `obs.fluid_end_eps_eps_s`, `obs.every_steps`, `obs.clump_hist_max`, `obs.phase_*`, `obs.hist_times_fm` | `computational_frame`* / `rest_frame`; 7 rho_s*; 6* / 26; 5*; 1*; 25*; phase-plane bins 0..12 rho_s step 0.1, 0..30 eps_s step 0.25*; [] * | implemented and tested (WP6; `docs/validation.md`); clumps wrap iff `grid.boundary = periodic` |
| S11 | `eos.finite_T_construction` | `undecided`* (WP2b); each rejected candidate is kept as an alternative | planned (WP2b) |
| S12 | `linear.particle_density_in_spline_region` | `linear_in_rho`* / `hadron_side` / `plasma_side` | implemented and tested (WP8: `particle_density_spline`, T8.0); no nuclear provider yet, it is used once the finite-temperature table supplies the hadron-side and plasma-side densities. The other `linear.*` keys (eta0, kappa0, g_gluon, g_quark, k grid, sigma_eq2, input files, toy EoS parameters) are listed in `config/defaults.json` and `apps/spf_linear.cpp`; the choice of `D_k` is in `docs/equations.md`, Sect. 12 |
| S13 | `eos.hadron_constants` | `relativistic`* (A = 52.127 MeV, B = 19.105 MeV) / `non_relativistic` (A = 52.385 MeV, B = 19.048 MeV) | relativistic constants produced by the prototype (WP2a); the non-relativistic variant is not produced; planned (WP2b) |
| S14 | build preset; `run.backend` | `cpu-openmp`* / `cpu-serial` (reference) / `cuda`, `cuda-turing75`; `run.backend = auto`* records whatever was compiled | presets implemented (WP0); production backend fixed after WP3g: on the GPU node the CUDA backend is 22 times faster than the 20 cores for 160^3 and 320^3 (ideal EoS; `docs/validation.md`, WP3g, G4); the default preset is not changed here |
| S15 | `run.precision` | `double`* / `single` | planned (later; needs its own validation) |
| S16 | `gradient.half_step_refresh` | `lagged`* / `midpoint` | both implemented (WP4): `lagged` refreshes `p_grad` from the density of the previous recovery before each recovery (brief); `midpoint` adds a refresh from the half-step density and a second recovery of the half-step state; no material difference in the noise tests (T4.5c) |
| S01 | `ic.generator`, `ic.oscar_path`, `ic.oscar_event`, `ic.oscar_time_fm`, `ic.smash_time_offset_fm` (WP5) | `smash_cascade`* / `oscar_file` / `glauber_gaussians` (explicit fallback) / `analytic` | implemented and tested (WP5); SMASH 3.4.2 (Pythia 8.316) in cascade mode, `Fermi_Motion: frozen`, centre-of-velocity frame; `smash_cascade` and `oscar_file` share the reader |
| S02 | `ic.t_init_radius_mode`, `ic.t_init_r0_fm`, `ic.t_init_radius_fm`, `ic.smash_time_offset_fm` (WP5) | `r0_A13`* (7.110 fm, 11.257 fm/c at 3 A GeV) / `fixed` (6.62 fm, 10.481 fm/c); offset between SMASH time and `t_init`: 0* | implemented (WP5); SMASH time zero = contact of the contracted spheres of radius R + d, measured first NN collision 0.53 +- 0.45 fm/c, offset 0; no explicit-time key (use `ic.oscar_time_fm`) |
| S03 | `ic.sigma_init_fm`, `ic.cutoff_sigma` | 1.0* / 0.7 / 0.5; cutoff 4 sigma* | implemented and tested for the three widths (WP5, T5.1); central state at 3 A GeV: 5.04 / 5.70 / 6.07 `rho_s` |
| S04 | `ic.gaussian_contraction` (WP5) | `particle_gamma_z`* (gamma_z of the hadron; the papers contract along z, arXiv:0710.0332 Eq. 6, arXiv:0806.1695 Eq. 4, without defining gamma_z) / `cm_gamma` / `none`; supersedes `undecided` | implemented and tested (WP5, T5.1); central 3 A GeV state: 5.04 (default), 5.67 (`cm_gamma`), 4.97 (`none`) |

## Changes with respect to the plan

* S02: the single key `ic.t_init_radius_fm` of the plan became three keys so that every key has one JSON type
  (the configuration loader rejects an override whose type differs from the default).
* S10: `obs.clump_rho_min` is named `obs.clump_rho_min_rho_s` to carry its unit.
* S04 (WP5): the default of `ic.gaussian_contraction` changed from `undecided` to `particle_gamma_z`; the four WP5 rows S01 to S04 above (appended, the earlier "planned" rows are kept) carry the implemented status.
* Dependency: Kokkos 5.3.0 (release of 2026-11-11) is pinned, not 5.1.0.
