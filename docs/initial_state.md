# Initial state for the fluid stage (WP5)

Code: `include/spf/initial_state.hpp`, `src/initial_state.cpp`, `apps/spf_ic.cpp`, `tools/smash/`. Tests: `tests/test_initial_state.cpp`
(measured numbers: `docs/validation.md`, section WP5). Decisions: `docs/decisions_spinodal.md`, S01 to S04.

## Pipeline

```
SMASH (external program, cascade mode)  ->  OSCAR2013 particle list at t_init  ->  reader (host)  ->  deposition (device kernel)
  -> R = N^0, M_j = T^0j, E = T^00 on the grid  ->  E >= |M| limiter  ->  Hydro3D::set_conserved  ->  primitive recovery
```

`spf_ic [--out DIR] [--evolve T_fm] [--no-snapshot] [key=value ...]` runs the whole chain and writes `<DIR>/ic_summary.json` (and, unless
`--no-snapshot`, the Hydro3D snapshot `ic_state.{bin,json}`). With `--evolve T` it also runs the fluid for `T` fm/c and reports recovery failures,
limiter activations and the location of the cells that fail. Equation of state of the recovery: `eos.model` (for the central values of a nearly
static centre the choice matters little: 4.9809 vs 4.9829 `rho_s` for `ideal` vs `cold_thermal_two_phase`, same SMASH event, GPU runs).

### Generators (`ic.generator`, decision S01)

| value | source |
|---|---|
| `smash_cascade` | OSCAR2013 file `ic.oscar_path`, event `ic.oscar_event`; block with the time nearest to `t_init + ic.smash_time_offset_fm` (or `ic.oscar_time_fm` if >= 0) |
| `oscar_file` | same reader, any OSCAR2013 file (for instance converted UrQMD output); block of `ic.oscar_time_fm`, or the last `out` block if it is negative |
| `glauber_gaussians` | explicit fallback, never a substitute for missing SMASH events: `2 ic.nucleus_A` nucleons from Woods-Saxon (`ic.ws_radius_fm`, `ic.ws_a_fm`, seed `ic.seed`), full overlap, positions contracted along z by `gamma_cm`, `p_z = +- gamma beta m_N` |
| `analytic` | two counter-streaming Lorentz-contracted Woods-Saxon nuclei at full overlap, `R = 2 gamma rho_WS(sqrt(x^2 + y^2 + gamma^2 z^2))`, `E = 2 gamma^2 m_N rho_WS`, `M = 0` (smooth test state; the centre reproduces `2 gamma_cm rho_0`) |

### OSCAR2013 reader

Host side, ASCII only. Column names are taken from the `#!OSCAR2013` header line (default `t x y z mass p0 px py pz pdg ID charge`). Block headers
`# event N [ensemble E] in|out M` (the `ensemble` word is what SMASH 3.4.2 writes) and `# event N [ensemble E] end 0 impact b ...` are recognised;
the declared particle count is checked against the lines found. Energy is the `p0` column.

### Baryon number from the PDG id (`baryon_number_from_pdg`)

* `|id| >= 1000000000` (nuclei, `10LZZZAAAI`): `B = sign(id) * AAA`, `AAA = (|id| / 10) mod 1000`.
* otherwise, with `n_q1 = (|id| / 1000) mod 10`, `n_q2 = (|id| / 100) mod 10`, `n_q3 = (|id| / 10) mod 10`: three non-zero digits give a baryon,
  `B = sign(id)`; everything else (mesons, diquarks such as 2101, leptons, photons) gives 0.

### Deposition (`deposit_particles`)

Each hadron deposits `N^0 = B`, `T^00 = E`, `T^0j = p^j` with a normalised Gaussian of width `ic.sigma_init_fm` (the three quantities share
one weight), truncated at `ic.cutoff_sigma` (4) sigma. Weights are separable, `w = w_x(i) w_y(j) w_z(k) / (S_x S_y S_z)`, where `S` are the sums of the 1D
weights over the full stencil (grid points within the cutoff, inside the box or not). Consequences: a particle whose stencil lies in the box
contributes exactly `B, E, p` (grid sums equal particle sums to rounding, about 1e-16 measured); the part of a stencil outside the box is lost and
reported (`outside_fraction_B`, `outside_fraction_E`, `n_particles_clipped`). `spf_ic` stops if the lost fraction of the baryon number exceeds
`ic.max_outside_fraction` (1e-6). If the window holds no grid point on some axis (a Gaussian narrower than a cell, only possible with a strong
contraction and a small sigma), the particle goes to the nearest grid point (`n_particles_empty`).

Kernel: `TeamPolicy`, one team per particle, threads over the stencil cells, `Kokkos::atomic_add` into five Views (no ScatterView). The summation order is
not deterministic, so backends agree to rounding only. The box is centred on the origin, cell centres at `(i - (n - 1)/2) dx`.

Densities are the deposited amounts divided by `dx^3`. `R = N^0`, `E = T^00`, `M_j = T^0j` are the conserved fields of Hydro3D (no equation of state is
needed); `limit_momentum` rescales `M` to `|M| = E` where `|M| > E` (count reported; it is 0 for deposited states because every `|p_i| <= E_i`).
`Hydro3D::set_conserved(R, Mx, My, Mz, E)` copies the fields; primitives follow from the solver's recovery (`update_primitives`), failures are counted.

### Gaussian contraction (`ic.gaussian_contraction`, decision S04)

arXiv:0710.0332 Eq. (6) and arXiv:0806.1695 Eq. (4) both write the energy density of a particle in the computational frame as
`eps_cf = N exp(-[(x - x_p)^2 + (y - y_p)^2 + (gamma_z (z - z_p))^2] / (2 sigma^2))`, `N = (2 pi)^(-3/2) gamma_z sigma^-3 E_cf`, "to account for the
Lorentz contraction of the nuclei in the longitudinal direction". So the papers contract the Gaussian along z. They do not say which `gamma_z`; the
brief asks for the particle's own. Values:

* `particle_gamma_z` (default): `gamma_z = 1 / sqrt(1 - (p_z/E)^2)` of the hadron, width `sigma / gamma_z` along z;
* `cm_gamma`: `gamma_z = gamma_cm` for every particle (the reading in which the contraction is that of the nuclei);
* `none`: spherical Gaussians.

The choice matters for the central state at 3 A GeV (20 events, `data/ic/t52_variants_3AGeV.json`): central `rho/rho_s` = 4.97 (none), 5.04
(`particle_gamma_z`), 5.67 (`cm_gamma`).

## Collision kinematics and the start time

`gamma_lab = 1 + E_lab/m_N`, `gamma_cm = sqrt((gamma_lab + 1)/2)`, `sqrt(s_NN) = 2 gamma_cm m_N` (`m_N = 0.940 GeV`; SMASH uses the mean nucleon mass
0.938 GeV, so its `sqrt(s_NN)` is 3.0245 instead of 3.0289 GeV at 3 A GeV). `t_init = 2 R / sqrt(gamma_cm^2 - 1)`, `R` from `ic.t_init_radius_mode`:
`r0_A13` gives `R = 1.2 * 208^(1/3) = 7.110` fm (11.257 fm/c at 3 A GeV), `fixed` gives 6.62 fm (10.481 fm/c). This is the time at which two nuclei of
contracted radius `R/gamma` that touched at `t = 0` have passed through each other.

### SMASH's time zero (verified by measurement)

SMASH's collider modus starts at negative time and places the nuclei so that the contracted spheres of radius `R_nuc + d` (Pb: 6.67 + 0.54 fm) touch exactly at
`t = 0` (`ColliderModus::initial_conditions`, comment "Contracted spheres with nuclear radii should touch exactly at t=0"; `Initial_Distance` only sets how
early the run starts). So SMASH `t = 0` is the moment of first contact in the sense of the papers, and the driver sets

`t_SMASH = t_init + offset`, with `--time-offset` (config key `ic.smash_time_offset_fm`) = 0.

Measurement, 3 A GeV, 8 events, `tools/smash/run_smash.py --measure` + `tools/smash/time_convention.py` (`data/ic/time_convention_3AGeV.json`):
first nucleon-nucleon collision at **0.53 +- 0.45 fm/c** (event mean +- standard deviation, range -0.18 to 1.06; single earlier run: -0.013);
median nucleon-nucleon collision time 7.2 fm/c; central baryon density (cylinder `r_perp < 3` fm, `|z| < 1.5` fm, computational frame) maximal at
**8.2 +- 1.4 fm/c** (per event 6.0 to 10.0 fm/c; value of the maximum 5.9 +- 0.6 `rho_s`). Interpretation: the first collisions occur at about `t = 0`,
so the offset of SMASH time zero from first contact is zero within the 0.5 fm/c event-to-event scatter (it is not exactly 0 because the nuclear
surfaces are diffuse and the contact is defined with `R + d`). The time of maximal density is not a measure of the offset: without stopping it would be
the time of full overlap, `t_init/2 = 5.6` fm/c; the measured 8.2 fm/c is later because the matter is compressed and stopped, and `t_init = 11.3`
fm/c is later still. Not measured: the dependence of these times on the energy (only 3 A GeV was measured); the SMASH touching radius 7.21 fm
differs from the `R` of `ic.t_init_radius_mode` (7.11 or 6.62 fm), which shifts the start time by up to 1.5 fm/c at 3 A GeV in the `fixed` mode.

## SMASH configuration

`tools/smash/pbpb_cascade.yaml.in` (filled by `tools/smash/run_smash.py`): Pb+Pb (`{2212: 82, 2112: 126}`), `Impact: Value: 0`, `E_Kin` per nucleon in the fixed-target frame,
`Calculation_Frame: "center of velocity"` (equal to the nucleon-nucleon c.m. frame here), no potentials section (cascade), `Fermi_Motion: "frozen"`
(brief did not fix it; `--fermi-motion off|on|frozen`), `Force_Decays_At_End: false`, OSCAR2013 particle output at `Output_Times` (`Only_Final: "No"`).
SMASH writes one block at the output time and one at `End_Time` (`t_out + 0.2` fm/c, ignored by the reader), plus the initial list. Build: `tools/smash/BUILD.md`.

## Limits

* Box size: the baryon number outside the box is a hard check. 160^3 cells of 0.2 fm (32 fm) is enough at 2 to 6 A GeV (at most 9.1e-15 of the baryon number lost in all 40 events at 2 to 6 A GeV
  in T5.2); at 1 A GeV the box has to be 200 x 200 x 240 cells (the 160^3 box lost 5e-6 to 2e-4 of the baryon number).
* With the default `gradient.form = paper_pressure` the evolution of a deposited state fails the primitive recovery in the dilute outer fringe
  (see `docs/validation.md`, T5.4). With `gradient.form = off` there are no failures.
