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
| S07 | `gradient.form`, `gradient.a_fm` | `paper_pressure`* / `korteweg_full` / `off`; a = 0.033 fm* / 0 / 0.01 / 0.05 | planned (WP4) |
| S08 | `gradient.in_primitive_recovery` | `lagged`* / `force_only` | planned (WP4) |
| S09 | `shasta.antidiffusion`, `shasta.eta_ad`, `shasta.lambda`, `shasta.sweep_order`, `grid.dx_fm` | `phoenical`* / `explicit`; 0.125*; 0.4*; `permuted`* / `fixed`; 0.2 fm* / 0.1 fm | antidiffusion, eta_ad, lambda implemented and tested (WP1); `grid.boundary` (`periodic`* / `outflow`) added; sweep_order and grid.dx_fm planned (WP3) |
| S10 | `obs.moment_density`, `obs.clump_rho_min_rho_s` | `computational_frame`* / `rest_frame`; 7 rho_s* | planned (WP6) |
| S11 | `eos.finite_T_construction` | `undecided`* (WP2b); each rejected candidate is kept as an alternative | planned (WP2b) |
| S12 | `linear.particle_density_in_spline_region` | `linear_in_rho`* / `hadron_side` / `plasma_side` | planned (WP8) |
| S13 | `eos.hadron_constants` | `relativistic`* (A = 52.127 MeV, B = 19.105 MeV) / `non_relativistic` (A = 52.385 MeV, B = 19.048 MeV) | relativistic constants produced by the prototype (WP2a); the non-relativistic variant is not produced; planned (WP2b) |
| S14 | build preset; `run.backend` | `cpu-openmp`* / `cpu-serial` (reference) / `cuda`, `cuda-turing75`; `run.backend = auto`* records whatever was compiled | presets implemented (WP0); production backend fixed after WP3g |
| S15 | `run.precision` | `double`* / `single` | planned (later; needs its own validation) |

## Changes with respect to the plan

* S02: the single key `ic.t_init_radius_fm` of the plan became three keys so that every key has one JSON type
  (the configuration loader rejects an override whose type differs from the default).
* S10: `obs.clump_rho_min` is named `obs.clump_rho_min_rho_s` to carry its unit.
* Dependency: Kokkos 5.3.0 (release of 2026-11-11) is pinned, not 5.1.0.
