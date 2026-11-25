# Particlisation: switching-surface file, Cooper-Frye sampler, particle-list output (WP10b)

Units: GeV and fm, c = 1, metric (+,-,-,-); `eps`, `p` in GeV/fm^3, `rho` in fm^-3. This note fixes (1) the switching-surface file that a fluid run writes (package WP10a)
and the sampler reads (package WP10b), (2) what `spf_sample` does with it, (3) the output files read by the afterburner.
Code: `include/spf/particlisation/`, `src/particlisation/`, `apps/spf_sample.cpp`; tests: `tests/test_sampler.cpp`; measured numbers: `docs/validation.md` (WP10b).

## 1. Switching-surface file (one surface per fluid event)

`<base>.surface.bin` + `<base>.surface.json`.

**`.bin`**: little-endian float64, 16 values per element, in this order (no header; the element count is in the sidecar, the file size must equal `n_elements * 128` bytes):

| index | field | unit | meaning |
|---|---|---|---|
| 0-3 | `t, x, y, z` | fm | centroid of the element |
| 4-7 | `dsig_0 ... dsig_3` | fm^3 | covariant normal vector dSigma_mu (lower index), oriented so that dSigma_mu u^mu > 0 for matter leaving the fluid region (the time-like part is positive for an isochronous slice) |
| 8-11 | `u^0 ... u^3` | 1 | fluid four-velocity at the centroid, u^mu u_mu = 1 |
| 12-15 | `eps, rho, T, mu_B` | GeV/fm^3, fm^-3, GeV, GeV | rest-frame energy density, rest-frame net baryon density, temperature and baryon chemical potential from the EoS table at (eps, rho) |

The elements of the initial slice (cells that are already below `eps_sw` at `t_init`) are in the same file, with `dsig = (dV, 0, 0, 0)` at `t = t_init`; the sidecar counts them separately.

**`.json`** (sidecar), keys written by the surface writer: `format_version` (1), `n_elements`, `n_initial_slice` (count of initial-slice elements), `eps_sw` [GeV/fm^3],
`method` (`isoenergy` | `isochronous` | `initial_slice`: the method of the main part of the surface), `dx` [fm], `dt` [fm], `grid_shape`, `eos_model`, `eos_table_sha256`, `event_id`, `E_lab`,
`integrals` = {`E`, `Px`, `Py`, `Pz` (the surface integrals of T^{0 mu} dSigma_mu and T^{i mu} dSigma_mu, i = 1..3, in GeV; computed by the writer from the stored fields and the table's p0), `B` (the integral of N^mu dSigma_mu)},
`left_in_fluid` = the same five totals remaining inside the fluid at the end of the run.
The reader (`spf::particlisation::read_surface`) requires `n_elements` (and checks the byte count of the `.bin` against it); the sampler needs `dx` (alias `dx_fm`) for the position smearing and uses `event_id` for the seeds (0 when absent).
`integrals` is optional: when present, the per-sample summary reports the mismatch (sampled minus surface integral) of E and P. The baryon-number target is always computed from the `.bin` (Section 2.5), not from the sidecar.
The reader rejects non-finite values.

## 2. The sampler (`spf_sample`, `spf::particlisation::Sampler`)

### 2.1 Hadron content (`sampler.species`)

* `eos_consistent`\* - exactly the degrees of freedom of the hadron phase of the equation of state (`tools/eos/finite_T/eos2b.py`, `docs/eos_finite_T.md`):
  nucleons N (p, n; g = 4 in total, spin 2 each, m = 0.940 GeV, Fermi-Dirac) and antinucleons (pbar, nbar), with the **kinetic** chemical potential mu_0 = mu_B - dw/drho(rho)
  of the mean-field hadron phase (antinucleons -mu_0); pions (pi+, pi0, pi-; g = 3, m = 0.140 GeV, Bose-Einstein, mu = 0).
  The mean field is w(rho) = rho (-A c + B c^2), c = rho/rho_s, dw/drho = -2 A c + 3 B c^2, with the constants of `eos2b.py` computed in the same way (T = 0 Fermi-gas binding energy E/A = e_kin0(rho)/rho - m_N,
  central differences with step 1e-4 fm^-3 at rho_s, K = 300 MeV: B = (K - 9 rho_s^2 d2)/18, A = rho_s d1 + 2B). Measured: A = 52.127066 MeV, B = 19.104872 MeV (relativistic constants, decision S13).
  The mean field is not a particle: the sampled kinetic energy density is eps - w(rho); see Section 2.7.
* `hrg` - every hadron of SMASH's `particles.txt` (`sampler.hrg_particles_file`; pole masses, widths ignored, resonances are left to SMASH to decay), quantum statistics (baryons Fermi-Dirac, mesons Bose-Einstein), mu = B mu_B (mu_Q = mu_S = 0).
  Degeneracy 2J+1 = last PDG digit, baryon number and charge from the PDG code (`pdg_baryon`, `pdg_charge`); antiparticles are added for all baryons and for all mesons with different flavour digits that are not listed; nuclei (|PDG| > 10^9) are skipped;
  `sampler.hrg_max_mass_GeV` > 0 drops heavier species.

### 2.2 Cooper-Frye sampling

For an element with dSigma_mu, u^mu, T, mu_B: in the local rest frame (LRF) dSigma*_0 = u^mu dSigma_mu, and the spatial part dSigma*_vec follows from the boost (upper-index components; `element_kinematics`);
|dSigma*_vec|^2 = dSigma*_0^2 - dSigma_mu dSigma^mu. With dSigma_max = |dSigma*_0| + |dSigma*_vec|, the expected number of candidates of species s is lambda_{e,s} = n_s(T, mu_s) dSigma_max.
Per sample and species: the number of candidates is Poisson(Lambda_s), Lambda_s = sum_e lambda_{e,s}; each candidate is placed in an element with probability lambda_{e,s}/Lambda_s
(superposition of the independent Poisson processes of the elements; implemented with prefix sums and a binary search, so one sample costs O(candidates log N_elements) and the table is built once per surface),
draws an LRF momentum from k^2 f(E) dk (Section 2.3) and an isotropic direction, and is accepted with probability max(p*.dSigma*, 0) / (E* dSigma_max), p*.dSigma* = E* dSigma*_0 - p*_vec . dSigma*_vec.
Accepted momenta are boosted to the computational frame (`p = Lambda(u) p*`), so p^mu is on shell to rounding.
This is the standard method (Huovinen and Petersen; iSS) and gives dN = g f p.dSigma/E d^3p/(2 pi)^3 for p.dSigma > 0.

**Negative contributions.** For every candidate the weight w = p*.dSigma*/(E* dSigma_max) is in [-1, 1]. The sample records sum w^+ and sum w^- (rejected negative contributions) over the candidates of the accepted trial, as number, lab-energy
and |baryon number| weighted sums, and reports `negative_fraction` = sum w^- / (sum w^+ + sum w^-) for each (so that a purely space-like element, p.dSigma = p^x A dt, gives 0.5). Particles with p.dSigma < 0 are never produced.
The fraction is a Monte Carlo estimate over the candidates of one sample, so it fluctuates with the number of candidates (in `exact_counts` the candidates are drawn until the required number is accepted, a stopping rule that biases the estimate slightly); the exact expectation for the surface is `expected_negative_fraction_number` in the preparation report.

**Closed-form angular average.** With v = k/E and c = cos(theta) between the momentum and dSigma*_vec, w = (d0 - v s c)/dSigma_max with s = |dSigma*_vec|; the average over c of max(w, 0) is max(d0, 0)/dSigma_max for v s <= |d0| and (v s + d0)^2/(4 v s dSigma_max) otherwise, so alpha_e,s = <max(w,0)> over the thermal distribution and the expected accepted number is n dSigma_max alpha (the negative part is alpha - d0/dSigma_max).

### 2.3 Thermal integrals and momentum sampling (`thermal.hpp`)

n, e, P of one species: n = g/(2 pi^2 hbarc^3) Int k^2 f dk, e = ... Int k^2 E f dk, P = g/(6 pi^2 hbarc^3) Int k^4/E f dk, f = 1/(exp((E-mu)/T) + s). The momentum integral is a composite Gauss-Legendre rule (`sampler.quad_nodes`, 16\* per panel);
panel edges at E = m + T t (t = 0 ... 50) and E = mu + T j (j = -30 ... 45, the Fermi edge), upper limit max(m, mu) + 50 T. A momentum is drawn by choosing a panel with its weight and rejection inside the panel with the rigorous bound k_R^2 f(E(k_L))
(k^2 increases and f decreases with k). Tail beyond the upper limit (integrand < 1e-21 of its maximum) is dropped.

### 2.4 Conservation (`sampler.conservation`)

* `global_B_Q`\* - the sampled event has exactly B_target and Q_target. B_target = round(Int N^mu dSigma_mu over the **sampled** elements) (elements skipped by `spline_policy = skip` or cold are not part of the target), Q_target = round(B_target Z/A) with `sampler.charge_Z/charge_A` = 82/208.
  Two methods (`sampler.global_method`):
  * `exact_counts`\* (`eos_consistent` only; **no rejection**). The model is a Poisson process, so the accepted numbers of the charge states are independent Poisson variables: p, n with mean Abar_N/2 each, pbar, nbar with Abar_Nbar/2, pi+, pi0, pi- with Abar_pi/3,
    where Abar_s = Sum_e n_s dSigma_max,e alpha_e,s is the expected number of accepted particles of species s, computed at preparation by quadrature with the closed-form angular average alpha of Section 2.2 (`thermal_density_acceptance`; for every time-like element
    alpha = max(u.dSigma, 0)/dSigma_max and n dSigma_max alpha = n u.dSigma, no integral). The sampled event is this process conditioned on B = Np + Nn - Npb - Nnb = B_target and Q = Np - Npb + Npi+ - Npi- = Q_target. Because the numbers are Poisson, the conditioning is done on the numbers:
    (1) the number of antinucleons nNb is drawn from P(nNb) ~ Pois(B_target + nNb; Abar_N) Pois(nNb; Abar_Nbar) G(nNb), G = P(Q | nN, nNb) (table built once per surface; window mean + 14 sigma + 30, probability left outside below 1e-40);
    (2) a - b = u and s = pi+ - pi-: P(u) ~ pU(u) Skellam(Q - u), a ~ Bin(nN, 1/2) given u, then pi+ given s from Pois x Pois, pi0 ~ Pois(Abar_pi/3) (`draw_charge_counts`);
    (3) exactly that many accepted particles of each group are drawn (candidates until the required number is accepted; the accepted particles of a Poisson process are iid from its accepted intensity), and the charge labels are assigned by uniform random permutations.
    The whole procedure is exact (no approximation beyond the truncation windows), costs one pass over about 1.3 candidates per accepted particle, and has acceptance 1. Fixing the numbers first and assigning charges afterwards is **not** exact: the conditioning on Q also changes the distribution of the pion and antinucleon numbers
    (tested: that variant failed the comparison with the brute-force conditioned Poisson mode, pi+ multiplicity p = 1e-4, pi- 4e-3, see `docs/validation.md`).
  * `rejection` (reference, `eos_consistent`): whole-event rejection of the baryon sector until the net baryon number is B_target, pions once, then the label vector redrawn until Q = Q_target. Exact but its acceptance falls with B (baryon stage 4.1 % at B = 100, 3.0 % at 200, 2.5 % at 400; label stage 1.6 %, 0.36 %, 0.025 %).
  * `hrg`: the charge is carried by the species and the species have different masses, so the joint (B, Q) rejection is kept: a trial is the baryon sector (checked for B first) followed by the meson sector, and the whole trial is rejected unless Q = Q_target (`sampler.max_trials`). Exact, low efficiency (Section 5).
  The per-sample summary reports `trials_baryon_or_joint` and `trials_charge_labels` (1 and 1 for `exact_counts`); the aggregated acceptance is samples/trials. `max_label_trials` applies to `rejection` only.
* `poisson` - no constraint; N, Nbar, pions fluctuate, labels are uniform.

### 2.5 Equation-of-state region (`sampler.spline_policy`)

The surface file carries no phase flag. An element is **flagged as spline (two-phase interpolation) region** when the hadron gas of the stored (T, mu_B) does not reproduce the stored density,
|n_N(T, mu_0) - n_Nbar(T, mu_0) - rho| > `spline_tol_rel` rho + `spline_tol_abs_fm3` (0.02, 1e-6 fm^-3), with mu_0 = mu_B - dw/drho(rho); on the lattice nodes of the hadron branch of the test table the left side is below 1.1e-7 rho.
`use_table`\* uses the stored T and mu_B (every element is sampled; the count and baryon flux of the flagged elements are reported), `skip` leaves flagged elements out and reports their count, N^mu dSigma_mu and eps u^0 (u.dSigma)
(the energy flux without the pressure term: the surface file has no p). Elements with T < `T_min_GeV` or without positive dSigma_max are skipped as cold and reported in the same fields. The reviewer decides the default after a surface census (decision S27).

### 2.6 Positions, seeds, threads

Position = element centroid plus a uniform offset in a cube of side dx (`sampler.position = cell_uniform`\*; `none` keeps the centroid); time = element time. `sampler.sort_by_time` (true\*) orders the particles of a sample by time (stable), the ID column counts after the sort.
`sampler.n_samples` samples per surface (1\*). The generator of sample k is xoshiro256** seeded from `splitmix64`-mixes of (`sampler.seed`, event id, k) (`derive_seed`). Samples do not share state, the element table is a pure function of the surface with a fixed summation order,
so the output is bit-identical for any number of threads and for repeated runs (OpenMP parallel loop over samples and over elements; `sampler.threads` 0 = runtime default). Different seed, event id or sample id give different samples.

### 2.7 What the closure test shows (eos_consistent)

The table's (eps, p) contain the mean field; the sampled particles carry only the kinetic part: eps_kin = eps - w(rho), p_kin = p - (rho dw/drho - w(rho)). On a surface element the sampled minus tabulated energy and momentum fluxes are therefore
for an isochronous box with u = gamma(1, v, 0, 0) and dSigma = (V, 0, 0, 0), Delta E = -V (w + rho w' gamma^2 v^2) and Delta P_x = -V rho w' gamma^2 v (sampled minus tabulated),
and the baryon number is reproduced exactly. In the rest frame Delta E per baryon is -w/rho (about +32 MeV per baryon at rho_s). The pressure term is Delta p = -(rho w' - w) = w - rho w'. This is a property of the model, not of the sampler (`docs/validation.md`, B5).

## 3. Output

`<out>/<sampler.output_prefix>.oscar`: OSCAR2013 particle lists, the format read by SMASH's List modus (`#!OSCAR2013 particle_lists t x y z mass p0 px py pz pdg ID charge`, `# Units: fm fm fm fm GeV GeV GeV GeV GeV none none e`),
one event block per sample (`# event <k> ensemble 0 out <n>` ... `# event <k> ensemble 0 end 0 impact 0.000 scattering_projectile_target yes`; the block layout is that of the files read by `spf::read_oscar2013`, and the round trip is tested).
The mass column carries the mass of the sampler (0.940 / 0.140 GeV for `eos_consistent`, the pole mass of `particles.txt` for `hrg`), p0 is on shell with that mass. The particles of a block have different times (the element times), sorted ascending;
whether SMASH's List modus of the version in use accepts a block with several times has to be checked by the afterburner package (not tested here: SMASH is not run by this package).
Failed samples are omitted from the OSCAR file and flagged in the JSON.

`<out>/<sampler.output_prefix>.json`: `preparation` (element counts, spline/cold elements and their fluxes, B_target, Q_target, expected number of candidates, `expected_accepted_per_species`, `expected_net_B_accepted`, `expected_negative_fraction_number`: the exact expected accepted numbers and negative fraction), timings, and `samples`: one summary per sample with
`counts` (per species), `sum` (E, Px, Py, Pz, B, Q), `targets`, `surface_integrals`, `mismatch_sampled_minus_surface`, `negative_fraction`, `skipped`, `spline_region_flagged_elements`, `candidates_in_accepted_trial`, `efficiency`, `seed`, `ok`.
`<out>/config_used.json`: effective configuration.

## 4. Configuration keys (block `sampler`, `config/defaults.json`; default marked \* in the decision register, `docs/decisions_spinodal.md` S25 - S29)

`surface` (base path), `output_prefix`, `species`, `quantum_statistics` (true\*; false = Boltzmann), `nucleon_mass_GeV` 0.940, `pion_mass_GeV` 0.140, `mean_field_K_MeV` 300, `hrg_particles_file`, `hrg_max_mass_GeV`,
`conservation`, `global_method`, `charge_Z`, `charge_A`, `spline_policy`, `spline_tol_rel`, `spline_tol_abs_fm3`, `T_min_GeV`, `position`, `sort_by_time`, `n_samples`, `seed`, `quad_nodes`, `max_trials`, `max_label_trials`, `threads`.

    spf_sample --surface <base> --out <dir> [--config file] [--override file] [sampler.key=value ...]

## 5. Known limitations

1. `hrg` computes the density of every species for every element (the table of prefix sums has one array of N_elements doubles per species): memory 8 N_elements N_species bytes and about 1 microsecond per species and element; a 10^6-element surface with a full `particles.txt` (several hundred species) needs several GB.
   The joint (B, Q) rejection of `hrg` has a low efficiency (about 1/(sqrt(2 pi Var B) sqrt(2 pi Var Q)) per trial); a unit test measures it on a small system only.
2. The SMASH input file `particles.txt` was not available to this package; the `hrg` set is tested with a reduced file in the same layout.
3. The energy flux of skipped elements is reported without the pressure term (the surface file has no p).
4. Lattice spacing, time stepping and the sidecar keys other than those listed in Section 1 are not interpreted by the sampler.
