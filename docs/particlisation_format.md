# Particlisation: switching surface, Cooper-Frye sampler, particle-list output (WP10a, WP10b, G5)

The fluid stage of a collision event ends on a switching surface; the sampler (`spf_sample`, WP10b, Sect. 4) converts each surface element into hadrons with the
Cooper-Frye formula, and a SMASH afterburner follows. This document fixes the file format shared by the surface writer (`spf_collide`,
`include/spf/surface.hpp`) and the sampler, and describes how the surface is constructed. Validation: `docs/validation.md`, section WP10a.
How to switch it on in collision runs: `docs/running_collisions.md`, Sect. 15.

## 1. Files

One surface per fluid event: `<base>.surface.bin` and `<base>.surface.json` (sidecar). `spf_collide` writes them as
`<out>/event_NNN/event_NNN.surface.{bin,json}` (or into `particlisation.surface.output_dir` if that key is set). With the isochronous method
(cross-check) one file pair is written per listed time, `<base>.iso_<t>fm.surface.{bin,json}`.

### 1.1 `.surface.bin`

Little-endian float64 records, 16 values per element, in this order:

| index | name | unit | meaning |
|---|---|---|---|
| 0 - 3 | `t, x, y, z` | fm | centroid of the element (computational frame = nucleus-nucleus centre-of-mass frame; `t` is the absolute time, `t_init` + fluid time; the origin of `x, y, z` is the centre of the box) |
| 4 - 7 | `dsig_0 .. dsig_3` | fm^3 | covariant normal vector `dSigma_mu` (lower index, metric +,-,-,-), oriented so that `dSigma_mu u^mu > 0` for matter leaving the fluid region (time-like part positive for an isochronous slice) |
| 8 - 11 | `u^0, u^1, u^2, u^3` | 1 | fluid four-velocity at the centroid, `u^mu u_mu = 1` |
| 12 | `eps` | GeV/fm^3 | rest-frame energy density |
| 13 | `rho` | fm^-3 | rest-frame net baryon density |
| 14 | `T` | GeV | temperature from the EoS table at (`eps`, `rho`) |
| 15 | `mu_B` | GeV | baryon chemical potential from the EoS table at (`eps`, `rho`) |

The record size is 128 bytes; the number of records is `n_elements` of the sidecar. The first `n_elements_initial_slice` records are the initial
slice (Sect. 2.3); the remaining ones are the surface proper. Within each time step the records are sorted lexicographically by their 16 values
(the order does not depend on the backend's order of atomic appends). For equations of state without temperature (`ideal`, `gamma_law`,
`cold_thermal_*`), `T` and `mu_B` are written as 0 and the sidecar has `T_mu_from_eos = false`.

### 1.2 `.surface.json`

| key | meaning |
|---|---|
| `format_version` | 1 |
| `byte_order`, `dtype`, `values_per_element`, `columns`, `units` | `little`, `float64`, 16, the column names of the table above, units |
| `n_elements`, `n_elements_initial_slice`, `n_elements_surface` | counts; `methods_in_file` repeats them by method tag (`initial_slice`, `isoenergy` or `isochronous`) |
| `eps_sw_GeV_fm3`, `eps_sw_eps_s` | switching energy density |
| `method` | `isoenergy` or `isochronous` (the initial slice is always in the same file, tagged by its count) |
| `dx_fm`, `dt_fm`, `grid_shape`, `origin_fm`, `t_init_fm` | lattice spacing, time step, cells per direction, position of cell (0, 0, 0), start time of the fluid stage |
| `eos_model`, `eos_table_sha256`, `eos_table_interpolation`, `eos_scale_b`, `eos_scale_c`, `gradient_form` | equation of state (table checksum: SHA-256 of the `.bin` table, empty for analytic models) |
| `event_id`, `E_lab_GeV`, `generator`, `oscar_path`, `oscar_time_fm` | event |
| `initial_slice`, `interpolation`, `first_level`, `isochronous_*` | construction options (Sect. 2) |
| `totals_initial_on_grid` | `E_GeV`, `Px_GeV`, `Py_GeV`, `Pz_GeV`, `B` of the fluid at `t_init` before the initial slice is removed (sums of the conserved fields times `dx^3`) |
| `totals_initial_slice`, `totals_surface`, `totals_all_elements` | surface integrals `int T^{0 mu} dSigma_mu`, `int T^{i mu} dSigma_mu` (i = 1..3), `int N^mu dSigma_mu`, computed by the writer from the stored `eps`, `rho`, `u` with `T^{mu nu} = (eps + p0) u^mu u^nu - p0 g^{mu nu}`, `N^mu = rho u^mu` and the table's `p0(eps, rho)` |
| `inflow_part_of_totals` | the part of `E` and `B` carried by elements with `u^mu dSigma_mu < 0` (matter entering the fluid region) |
| `totals_inside_fluid_at_end` | conserved totals of the cells with `eps >= eps_sw` when the run ended (0 when the surface is complete) |
| `closure` | `(all elements + inside at end - initial on grid) / initial on grid` for `E` and `B`; for the momenta relative to the initial energy |
| `closed` | true if no cell had `eps >= eps_sw` at the end (isoenergy) or all listed times were reached (isochronous) |
| `diagnostics` | hypercubes with inconsistent topology (always 0), pieces dropped (more than 16 per hypercube; always 0 so far), element recovery failures, the largest number of cells at or above `eps_sw` on the outermost layer of the box (the surface would be cut by the box if > 0), final buffer capacity, wall time of the surface finder |
| `status`, `steps`, `time_after_init_fm`, `t_max_after_init_fm`, `hit_t_max` | end of the run (`spf_collide`) |

Python reader: `tools/particlisation/surface_io.py` (`read_surface(base)` returns the records as an `(n, 16)` array and the sidecar).

## 2. Construction

### 2.1 Isoenergy surface (`particlisation.surface.method = isoenergy`, default)

After every full time step `n -> n+1` the iso-surface `eps = eps_sw` is found in the 4D lattice cells spanned by 2 x 2 x 2 neighbouring cell
centres and the two time levels (one GPU thread per hypercube, named functor `detail::SurfaceIsoK`, compact append buffer with an atomic counter;
the buffer is enlarged and the step repeated if it overflows). The geometry is a re-implementation, from the description in P. Huovinen and
H. Petersen, Eur. Phys. J. A 48 (2012) 171, arXiv:1206.3371, of the Cornelius construction (`include/spf/surface_geometry.hpp`):

1. Corners `c = bt | bx << 1 | by << 2 | bz << 3`; `f = eps - eps_sw`; a corner is *above* if `f >= 0`.
2. Crossing points on the 32 edges whose ends differ in side, at `s = f(c0) / (f(c0) - f(c1))` from the lower corner (the same arithmetic in every
   hypercube that shares the edge, so neighbours see bitwise identical points).
3. Segments in the 24 squares: two crossings give one segment; four crossings (ambiguous square) are resolved by the mean of the four corner
   values: mean >= 0 connects the above corners (the below corners are cut off), mean < 0 cuts off the above corners. The decision depends only on
   the square, so the two cubes and all hypercubes that share a square agree.
4. Polygons in the 8 boundary cubes of the hypercube: in a cube every crossing point belongs to exactly one segment of each of its two squares,
   so the segments close into polygons; each polygon is triangulated from its centre (mean of its points).
5. Pieces: polygons that share a crossing point bound the same piece; every piece is one surface element (a hypercube can hold several, e.g. at
   saddle points or where two blobs meet; up to 4 were seen in random fields). The element normal is the sum of the normals of the tetrahedra
   (piece centre, polygon centre, segment); the centroid is the 3-volume-weighted mean of the tetrahedron centroids.

**Orientation** is decided combinatorially, never from the positions of the crossing points, so degenerate positions (a crossing point on a
corner) cannot flip it: in a square the side of the segment towards the *below* corners is known from the corner signs (direction `m2`); in a
cube with fixed axis `k` at side `sigma`, a segment of the square with outward normal `o_s` is oriented along `m2 x o_s` (free axes in ascending
order), which is the boundary orientation of the polygon with its normal towards the below side; in 4D the boundary orientation of the cube's
triangles relative to the hypercube is `-(2 sigma - 1) (-1)^k`. The resulting element normal is the Euclidean outward normal of the region
`f >= 0`, whose components are the covariant components `dSigma_mu`. Because the normal of a piece depends only on its boundary polygons, which
neighbouring hypercubes share, the sum of `dSigma_mu` over a closed surface vanishes to rounding (test T10a.0: 2e-17 relative on random fields).

**Values at the centroid** (`particlisation.surface.interpolation`):

* `conserved` (default): the conserved fields `R = N^0`, `M_i = T^{0i}`, `E = T^{00}` of the solver are interpolated multilinearly (16 corners) to
  the centroid, and the rest-frame state is obtained from them (Sect. 2.4). Dilute cells moving at nearly the speed of light, which the vacuum
  fronts of the solver produce, then enter with their (small) weight in `E`, `M`.
* `primitive`: `eps`, `rho` and `u^i` are interpolated multilinearly and `u^0 = sqrt(1 + u^i u^i)`. This is the rule of the work-package brief.
  It is kept as an alternative but is not usable in collision runs: next to the surface there are cells with `|M| -> E` (u^0 up to 1e7), and a
  linear interpolation of `u^i` carries their four-velocity into the elements (closure errors of order 1e10, `docs/validation.md`, WP10a).

`eps` of the iso criterion is always the interpolated rest-frame energy density; `T` and `mu_B` come from the table at the element's (`eps`, `rho`)
(device functions `EosTable::T`, `EosTable::mu`).

**First time level** (`particlisation.surface.first_level`): `before_removal` (default) uses the state at `t_init` before the initial slice is set to
vacuum as the lower level of the first time step, so that the surface starts where the cells of the initial slice end; `after_removal` uses the
state after the removal.

### 2.2 Isochronous surface (`particlisation.surface.method = isochronous`, cross-check)

At the first full step at or after each time of `particlisation.surface.isochronous_times_fm` (fluid time after `t_init`), each selected cell gives
one element with `dSigma_mu = (dx^3, 0, 0, 0)` at the cell centre. `particlisation.surface.isochronous_cells = all` (default) selects every cell with
`eps > 0`, as the isochronous Cooper-Frye integral over all of space in arXiv:1311.0999 (composite yields, Cooper-Frye prescription on an isochronous hypersurface); `above_eps_sw` selects only the cells
with `eps >= eps_sw` (the cells between 0 and `eps_sw` at that time are then not emitted; their totals are written as `totals_not_emitted`).

### 2.3 Initial slice

At `t_init` every cell with `0 < eps < eps_sw` gives an element with `dSigma_mu = (dx^3, 0, 0, 0)` at `t = t_init`, and
(`particlisation.initial_slice = remove`, default) is set to vacuum, so that every bit of energy, momentum and baryon number leaves the fluid
exactly once (decision S23). The alternative `keep` leaves those cells in the fluid; the surface integral then still counts every bit once,
because matter that later flows back into the region `eps >= eps_sw` crosses the surface inwards (negative contributions) before it leaves again.
Passing the original SMASH hadrons of those cells through to the afterburner instead is not implemented.

### 2.4 Rest-frame state of an element

Elements of the initial slice and of the isochronous surface are made from the conserved values of their cell, surface elements from the
interpolated conserved values (default). The recovery equation of the solver, `v (E + p0(eps, rho)) = |M|`, is solved for `w = 1 - v` by a
bracketed regula falsi (`detail::state_from_conserved`):

```
h(w) = (E - |M|) + p0 - w (E + p0) = 0,   eps = (E - |M|) + |M| w,   rho = R sqrt(w (2 - w)),   p0 = p0(eps, rho),
gamma^2 = 1 / (w (2 - w)),   u^i = gamma (1 - w) M^i / |M|,
```

so that `E - |M|` enters directly and `1 - v` keeps its digits for dilute matter moving at nearly the speed of light (the velocity of the
solver's recovery alone loses them, and `T^{00}` rebuilt from such a record can be off by orders of magnitude). The stored `(eps, rho)` give the
pressure used, so `T^{00}` and `N^0` built from the record reproduce the cell: on a whole isochronous slice (T10a.5) the element totals equal the
grid content to 1e-15 for both. Light-like cells (`gamma > 1e4`, in particular `|M| = E` exactly, which the `E >= |M|` limiter of the solver
produces at vacuum fronts) cannot be represented by a finite four-velocity: they get `gamma = 1e4` along `M` and `eps = (E + p) / gamma^2 - p`, so
that `T^{00} = E` still holds and `|T^{0i}|` differs from `|M|` by a relative `O(1e-8)`. The finite-range pressure `p_grad` is not part of the
stored state (the format fixes `T^{mu nu}` with `p0`); the conserved fields of the solver include it.

### 2.5 Configuration keys (`config/defaults.json`, block `particlisation`)

| key | default | alternatives | meaning |
|---|---|---|---|
| `particlisation.enabled` | false (`config/collisions/common.json`: true) | true | write the surface in `spf_collide`; the event then runs until no cell has `eps >= eps_sw` (cap `run.t_max_after_init_fm`, 30 fm/c in the collision files) |
| `particlisation.eps_sw_eps_s` | 4.0 | 5.0 (`config/collisions/eps_sw_5.json`), any positive value | switching energy density in units of `eps_s = m_N rho_s` |
| `particlisation.initial_slice` | `remove` | `keep` | Sect. 2.3 |
| `particlisation.surface.method` | `isoenergy` | `isochronous` | Sect. 2.1, 2.2 |
| `particlisation.surface.isochronous_times_fm` | [] | list of fluid times after `t_init` | isochronous method |
| `particlisation.surface.isochronous_cells` | `all` | `above_eps_sw` | Sect. 2.2 |
| `particlisation.surface.first_level` | `before_removal` | `after_removal` | Sect. 2.1 |
| `particlisation.surface.interpolation` | `conserved` | `primitive` | Sect. 2.1 |
| `particlisation.surface.output_dir` | "" (event directory) | path | where the surface files are written |

## 3. Surface file as read by the sampler (`spf::particlisation::read_surface`)

### 3.1 Keys

The reader requires `format_version` = 1 and `n_elements` (the byte count of the `.bin` must equal `n_elements * 128`; every value must be finite).
The sampler then uses only the following sidecar keys, all written by `spf_collide`:

| sampler needs | key in the file written by `spf_collide` | note |
|---|---|---|
| cell size for the position smearing | `dx_fm` (the reader also accepts `dx`) | required |
| seed component | `event_id` (the reader also accepts `event`; 0 if absent) | optional |
| surface integrals for the per-sample mismatch report | `totals_all_elements` = {`E_GeV`, `Px_GeV`, `Py_GeV`, `Pz_GeV`, `B`} | optional; renamed in the integration: the stand-alone sampler read `integrals` with the keys `E`, `Px`, `Py`, `Pz`, `B` (the old name is still accepted as a fallback) |

All other sidecar keys are not interpreted by the sampler. Production runs must sample in place and not keep surfaces: a surface file of a 3 A GeV event at dx = 0.2 fm
is about 150 MB (Sect. 7 of this document gives the measured size); run `spf_sample` directly after `spf_collide` and delete the `.surface.bin` (keep the small `.surface.json`).

## 4. The sampler (`spf_sample`, `spf::particlisation::Sampler`)

### 4.1 Hadron content (`sampler.species`)

* `eos_consistent`\* - exactly the degrees of freedom of the hadron phase of the equation of state (`tools/eos/finite_T/eos2b.py`, `docs/eos_finite_T.md`):
  nucleons N (p, n; g = 4 in total, spin 2 each, m = 0.940 GeV, Fermi-Dirac) and antinucleons (pbar, nbar), with the **kinetic** chemical potential mu_0 = mu_B - dw/drho(rho)
  of the mean-field hadron phase (antinucleons -mu_0); pions (pi+, pi0, pi-; g = 3, m = 0.140 GeV, Bose-Einstein, mu = 0).
  The mean field is w(rho) = rho (-A c + B c^2), c = rho/rho_s, dw/drho = -2 A c + 3 B c^2, with the constants of `eos2b.py` computed in the same way (T = 0 Fermi-gas binding energy E/A = e_kin0(rho)/rho - m_N,
  central differences with step 1e-4 fm^-3 at rho_s, K = 300 MeV: B = (K - 9 rho_s^2 d2)/18, A = rho_s d1 + 2B). Measured: A = 52.127066 MeV, B = 19.104872 MeV (relativistic constants, decision S13).
  The mean field is not a particle: the sampled kinetic energy density is eps - w(rho); see Section 4.7.
  **Chemical potential (decision S24, `sampler.mu_rule`).** `rho_matched`\* solves, for every element, n_N(T, mu_kin) - n_Nbar(T, mu_kin) = rho with the species above (Brent, relative tolerance 1e-12; rho <= 0 solved with mu_kin <= 0, rho = 0 gives 0; T below `sampler.mu_T_floor_GeV` = 0.005 is solved and sampled at the floor and counted in the summary as `mu_rule_n_at_T_floor`; solver failure falls back to the table value, `mu_rule_n_solver_failed`).
  `table` uses mu_kin = mu_B - dw/drho of the stored element. Reason: 97 - 100 % of the surface energy flux lies in the spline region (rho = 2 - 3.5 rho_s, T = 100 - 142 MeV) where the table rule does not reproduce rho (mismatch up to 67 %). Energy is not matched by either rule (Section 4.7).
* `hrg` - every hadron of SMASH's `particles.txt` (`sampler.hrg_particles_file`; pole masses, widths ignored, resonances are left to SMASH to decay), quantum statistics (baryons Fermi-Dirac, mesons Bose-Einstein), mu = B mu_B (mu_Q = mu_S = 0).
  Degeneracy 2J+1 = last PDG digit, baryon number and charge from the PDG code (`pdg_baryon`, `pdg_charge`); antiparticles are added for all baryons and for all mesons with different flavour digits that are not listed; nuclei (|PDG| > 10^9) are skipped;
  `sampler.hrg_max_mass_GeV` > 0 drops heavier species.

### 4.2 Cooper-Frye sampling

For an element with dSigma_mu, u^mu, T, mu_B: in the local rest frame (LRF) dSigma*_0 = u^mu dSigma_mu, and the spatial part dSigma*_vec follows from the boost (upper-index components; `element_kinematics`);
|dSigma*_vec|^2 = dSigma*_0^2 - dSigma_mu dSigma^mu. With dSigma_max = |dSigma*_0| + |dSigma*_vec|, the expected number of candidates of species s is lambda_{e,s} = n_s(T, mu_s) dSigma_max.
Per sample and species: the number of candidates is Poisson(Lambda_s), Lambda_s = sum_e lambda_{e,s}; each candidate is placed in an element with probability lambda_{e,s}/Lambda_s
(superposition of the independent Poisson processes of the elements; implemented with prefix sums and a binary search, so one sample costs O(candidates log N_elements) and the table is built once per surface),
draws an LRF momentum from k^2 f(E) dk (Section 4.3) and an isotropic direction, and is accepted with probability max(p*.dSigma*, 0) / (E* dSigma_max), p*.dSigma* = E* dSigma*_0 - p*_vec . dSigma*_vec.
Accepted momenta are boosted to the computational frame (`p = Lambda(u) p*`), so p^mu is on shell to rounding.
This is the standard method (Huovinen and Petersen; iSS) and gives dN = g f p.dSigma/E d^3p/(2 pi)^3 for p.dSigma > 0.

**Negative contributions.** For every candidate the weight w = p*.dSigma*/(E* dSigma_max) is in [-1, 1]. The sample records sum w^+ and sum w^- (rejected negative contributions) over the candidates of the accepted trial, as number, lab-energy
and |baryon number| weighted sums, and reports `negative_fraction` = sum w^- / (sum w^+ + sum w^-) for each (so that a purely space-like element, p.dSigma = p^x A dt, gives 0.5). Particles with p.dSigma < 0 are never produced.
The fraction is a Monte Carlo estimate over the candidates of one sample, so it fluctuates with the number of candidates (in `exact_counts` the candidates are drawn until the required number is accepted, a stopping rule that biases the estimate slightly); the exact expectation for the surface is `expected_negative_fraction_number` in the preparation report.

**Closed-form angular average.** With v = k/E and c = cos(theta) between the momentum and dSigma*_vec, w = (d0 - v s c)/dSigma_max with s = |dSigma*_vec|; the average over c of max(w, 0) is max(d0, 0)/dSigma_max for v s <= |d0| and (v s + d0)^2/(4 v s dSigma_max) otherwise, so alpha_e,s = <max(w,0)> over the thermal distribution and the expected accepted number is n dSigma_max alpha (the negative part is alpha - d0/dSigma_max).

### 4.3 Thermal integrals and momentum sampling (`thermal.hpp`)

n, e, P of one species: n = g/(2 pi^2 hbarc^3) Int k^2 f dk, e = ... Int k^2 E f dk, P = g/(6 pi^2 hbarc^3) Int k^4/E f dk, f = 1/(exp((E-mu)/T) + s). The momentum integral is a composite Gauss-Legendre rule (`sampler.quad_nodes`, 16\* per panel);
panel edges at E = m + T t (t = 0 ... 50) and E = mu + T j (j = -30 ... 45, the Fermi edge), upper limit max(m, mu) + 50 T. A momentum is drawn by choosing a panel with its weight and rejection inside the panel with the rigorous bound k_R^2 f(E(k_L))
(k^2 increases and f decreases with k). Tail beyond the upper limit (integrand < 1e-21 of its maximum) is dropped.

### 4.4 Conservation (`sampler.conservation`)

* `global_B_Q`\* - the sampled event has exactly B_target and Q_target. B_target = round(Int N^mu dSigma_mu over the **sampled** elements) (elements skipped by `spline_policy = skip` or cold are not part of the target), Q_target = round(B_target Z/A) with `sampler.charge_Z/charge_A` = 82/208.
  Two methods (`sampler.global_method`):
  * `exact_counts`\* (`eos_consistent` only; **no rejection**). The model is a Poisson process, so the accepted numbers of the charge states are independent Poisson variables: p, n with mean Abar_N/2 each, pbar, nbar with Abar_Nbar/2, pi+, pi0, pi- with Abar_pi/3,
    where Abar_s = Sum_e n_s dSigma_max,e alpha_e,s is the expected number of accepted particles of species s, computed at preparation by quadrature with the closed-form angular average alpha of Section 4.2 (`thermal_density_acceptance`; for every time-like element
    alpha = max(u.dSigma, 0)/dSigma_max and n dSigma_max alpha = n u.dSigma, no integral). The sampled event is this process conditioned on B = Np + Nn - Npb - Nnb = B_target and Q = Np - Npb + Npi+ - Npi- = Q_target. Because the numbers are Poisson, the conditioning is done on the numbers:
    (1) the number of antinucleons nNb is drawn from P(nNb) ~ Pois(B_target + nNb; Abar_N) Pois(nNb; Abar_Nbar) G(nNb), G = P(Q | nN, nNb) (table built once per surface; window mean + 14 sigma + 30, probability left outside below 1e-40);
    (2) a - b = u and s = pi+ - pi-: P(u) ~ pU(u) Skellam(Q - u), a ~ Bin(nN, 1/2) given u, then pi+ given s from Pois x Pois, pi0 ~ Pois(Abar_pi/3) (`draw_charge_counts`);
    (3) exactly that many accepted particles of each group are drawn (candidates until the required number is accepted; the accepted particles of a Poisson process are iid from its accepted intensity), and the charge labels are assigned by uniform random permutations.
    The whole procedure is exact (no approximation beyond the truncation windows), costs one pass over about 1.3 candidates per accepted particle, and has acceptance 1. Fixing the numbers first and assigning charges afterwards is **not** exact: the conditioning on Q also changes the distribution of the pion and antinucleon numbers
    (tested: that variant failed the comparison with the brute-force conditioned Poisson mode, pi+ multiplicity p = 1e-4, pi- 4e-3, see `docs/validation.md`).
  * `rejection` (reference, `eos_consistent`): whole-event rejection of the baryon sector until the net baryon number is B_target, pions once, then the label vector redrawn until Q = Q_target. Exact but its acceptance falls with B (baryon stage 4.1 % at B = 100, 3.0 % at 200, 2.5 % at 400; label stage 1.6 %, 0.36 %, 0.025 %).
  * `hrg`: the charge is carried by the species and the species have different masses, so the joint (B, Q) rejection is kept: a trial is the baryon sector (checked for B first) followed by the meson sector, and the whole trial is rejected unless Q = Q_target (`sampler.max_trials`). Exact, low efficiency (Section 7).
  The per-sample summary reports `trials_baryon_or_joint` and `trials_charge_labels` (1 and 1 for `exact_counts`); the aggregated acceptance is samples/trials. `max_label_trials` applies to `rejection` only.
* `poisson` - no constraint; N, Nbar, pions fluctuate, labels are uniform.

### 4.5 Equation-of-state region (`sampler.spline_policy`)

The surface file carries no phase flag. An element is **flagged as spline (two-phase interpolation) region** when the hadron gas of the stored (T, mu_B) does not reproduce the stored density,
|n_N(T, mu_0) - n_Nbar(T, mu_0) - rho| > `spline_tol_rel` rho + `spline_tol_abs_fm3` (0.02, 1e-6 fm^-3), with mu_0 = mu_B - dw/drho(rho); on the lattice nodes of the hadron branch of the test table the left side is below 1.1e-7 rho.
`use_table`\* uses the stored T and mu_B (every element is sampled; the count and baryon flux of the flagged elements are reported), `skip` leaves flagged elements out and reports their count, N^mu dSigma_mu and eps u^0 (u.dSigma)
(the energy flux without the pressure term: the surface file has no p). Elements with T < `T_min_GeV` or without positive dSigma_max are skipped as cold and reported in the same fields. The default `use_table` was kept after the census; the census result is the reason for the S24 rule (Section 4.1), which makes the density of the sampled gas equal to the element's rho in the spline region as well, so that the flag only reports where the table's (T, mu_B) is not the gas value.

### 4.6 Positions, seeds, threads

Position = element centroid plus a uniform offset in a cube of side dx (`sampler.position = cell_uniform`\*; `none` keeps the centroid); time = element time. `sampler.sort_by_time` (true\*) orders the particles of a sample by time (stable), the ID column counts after the sort.
`sampler.n_samples` samples per surface (1\*). The generator of sample k is xoshiro256** seeded from `splitmix64`-mixes of (`sampler.seed`, event id, k) (`derive_seed`). Samples do not share state, the element table is a pure function of the surface with a fixed summation order,
so the output is bit-identical for any number of threads and for repeated runs (OpenMP parallel loop over samples and over elements; `sampler.threads` 0 = runtime default). Different seed, event id or sample id give different samples.

### 4.7 What the closure test shows (eos_consistent)

The table's (eps, p) contain the mean field; the sampled particles carry only the kinetic part: eps_kin = eps - w(rho), p_kin = p - (rho dw/drho - w(rho)). On a surface element the sampled minus tabulated energy and momentum fluxes are therefore
for an isochronous box with u = gamma(1, v, 0, 0) and dSigma = (V, 0, 0, 0), Delta E = -V (w + rho w' gamma^2 v^2) and Delta P_x = -V rho w' gamma^2 v (sampled minus tabulated),
and the baryon number is reproduced exactly. In the rest frame Delta E per baryon is -w/rho (about +32 MeV per baryon at rho_s). The pressure term is Delta p = -(rho w' - w) = w - rho w'. This is a property of the model, not of the sampler (`docs/validation.md`, B5).

## 5. Output

`<out>/<sampler.output_prefix>.oscar`: OSCAR2013 particle lists, the format read by SMASH's List modus (`#!OSCAR2013 particle_lists t x y z mass p0 px py pz pdg ID charge`, `# Units: fm fm fm fm GeV GeV GeV GeV GeV none none e`),
one event block per sample (`# event <k> ensemble 0 out <n>` ... `# event <k> ensemble 0 end 0 impact 0.000 scattering_projectile_target yes`; the block layout is that of the files read by `spf::read_oscar2013`, and the round trip is tested).
The mass column carries the mass of the sampler (0.940 / 0.140 GeV for `eos_consistent`, the pole mass of `particles.txt` for `hrg`), p0 is on shell with that mass. The particles of a block have different times (the element times), sorted ascending;
whether SMASH's List modus of the version in use accepts a block with several times has to be checked by the afterburner package (not tested here: SMASH is not run by this package).
Failed samples are omitted from the OSCAR file and flagged in the JSON.

`<out>/<sampler.output_prefix>.json`: `preparation` (element counts, spline/cold elements and their fluxes, B_target, Q_target, expected number of candidates, `expected_accepted_per_species`, `expected_net_B_accepted`, `expected_negative_fraction_number`: the exact expected accepted numbers and negative fraction), timings, and `samples`: one summary per sample with
`mu_rule_n_at_T_floor`, `mu_rule_n_solver_failed`, `mu_rule_max_rel_residual` (in `preparation`), `counts` (per species), `sum` (E, Px, Py, Pz, B, Q), `targets`, `surface_integrals`, `mismatch_sampled_minus_surface`, `negative_fraction`, `skipped`, `spline_region_flagged_elements`, `candidates_in_accepted_trial`, `efficiency`, `seed`, `ok`.
`<out>/config_used.json`: effective configuration.

## 6. Configuration keys (block `sampler`, `config/defaults.json`; default marked \* in the decision register, `docs/decisions_spinodal.md` S24 - S30)

`surface` (base path), `output_prefix`, `species`, `quantum_statistics` (true\*; false = Boltzmann), `nucleon_mass_GeV` 0.940, `pion_mass_GeV` 0.140, `mean_field_K_MeV` 300, `hrg_particles_file`, `hrg_max_mass_GeV`,
`conservation`, `global_method`, `charge_Z`, `charge_A`, `mu_rule` (rho_matched\*, table), `mu_T_floor_GeV` (0.005), `spline_policy`, `spline_tol_rel`, `spline_tol_abs_fm3`, `T_min_GeV`, `position`, `sort_by_time`, `n_samples`, `seed`, `quad_nodes`, `max_trials`, `max_label_trials`, `threads`.

    spf_sample --surface <base> --out <dir> [--config file] [--override file] [sampler.key=value ...]

## 7. Known limitations

1. `hrg` computes the density of every species for every element (the table of prefix sums has one array of N_elements doubles per species): memory 8 N_elements N_species bytes and about 1 microsecond per species and element; a 10^6-element surface with a full `particles.txt` (several hundred species) needs several GB.
   The joint (B, Q) rejection of `hrg` has a low efficiency (about 1/(sqrt(2 pi Var B) sqrt(2 pi Var Q)) per trial); a unit test measures it on a small system only.
2. The SMASH input file `particles.txt` was not available to this package; the `hrg` set is tested with a reduced file in the same layout.
3. The energy flux of skipped elements is reported without the pressure term (the surface file has no p).
4. Lattice spacing, time stepping and the sidecar keys other than those listed in Section 3.1 are not interpreted by the sampler.

## 8. Run the whole chain (fluid, surface, particles)

`tools/particlisation/g5_endtoend.sh BUILD_DIR OSCAR_FILE [FIRST N K THREADS]` runs `spf_collide` (surface on, `config/collisions/common.json`) for two EoS tables and `spf_sample` with both mu rules on the same surfaces; `tools/particlisation/g5_analyse.py ROOT --json OUT --fig OUT.png`
reads the result (multiplicities, B and Q exactness, E and P balance, initial-slice fraction, timings, OSCAR2013 parse check). First measurement: `docs/validation.md`, G5-INT. A surface of a 3 A GeV event at dx = 0.2 fm has 1.2 million elements and is 148 - 156 MB.
**Production runs must sample in place and not keep surfaces**: run `spf_sample` directly on the file written by `spf_collide` and delete `<base>.surface.bin` afterwards (keep the small `.surface.json` and the sampler output); the driver script deletes the `.bin` files after both mu rules have been sampled.
