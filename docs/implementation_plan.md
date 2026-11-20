# Spinodal amplification of density fluctuations — implementation plan

Target: re-implement the fluid-dynamical model of Steinheimer & Randrup, PRL 109, 212301 (arXiv:1209.2462) and the linear-response analysis of
Randrup, PRC 82, 034902 (arXiv:1007.1448). Work packages marked **[E]** are self-contained implementation tasks with numeric acceptance tests;
packages marked **[S]** are design or review tasks for the project lead.

Companion files: `tools/eos/eos_reference_prototype.py` (Python reference for the equation of state) and `tools/eos/eos_reference_values.json`
(numbers produced by it; every "prototype" value quoted below is read from that file).

Status tags used below: **[paper]** stated in the cited paper; **[proto]** computed in the Python
prototype on 2026-11-20; **[derived]** derived while writing this plan, not yet machine-checked;
**[open]** not specified by the papers, needs a decision or calibration.

---

## 1. Scope

A new 3+1D Cartesian code written from the published equations. No solver source from other heavy-ion codes is used.
External programs: SMASH (cascade initial state, run separately) and, optionally, other hydro codes for cross-checks of the stable evolution.

---

## 2. What has to be reproduced

### 2.1 Fluid model (1209.2462; details in the long paper arXiv:1302.2956)  [paper]

* Ideal fluid: `d_mu T^{mu nu} = 0`, `T^{mu nu} = (eps + p) u^mu u^nu - p g^{mu nu}`; `d_mu N^mu = 0`, `N^mu = rho u^mu`.
* Solver: SHASTA (Rischke, Bernard, Maruhn), three spatial directions propagated consecutively. Standard grid `dx = 0.2 fm`.
* Finite-range term in the local pressure
  `p(r) = p0(eps, rho) - a^2 (eps_s / rho_s^2) rho(r) Laplacian rho(r)`,
  `rho_s = 0.153 fm^-3`, `eps_s = m_N rho_s`; standard `a = 0.033 fm` (variants 0, 0.01, 0.05 fm).
* Linear growth rate inside the isentropic spinodal (`v_s^2 < 0`):
  `gamma_k^2 = |v_s|^2 k^2 - a^2 (eps_s / h) (rho/rho_s)^2 k^4`, `h = eps + p`,
  `v_s^2 = (rho/h) (dp/drho)_{s/rho} = p_eps + (rho/h) p_rho` (partials of `p0(eps, rho)`).
* Interface tension `sigma_T = a * Integral_{c1}^{c2} [2 eps_s Delta f_T(c)]^{1/2} dc`, `c = rho/rho_s`, `Delta f = f_T - f_T^Maxwell`.
* Growth-rate measurement: periodic box, `rho = rho_bar + d_rho sin(kx)`, `eps = eps_bar + d_eps sin(kx)`, no flow,
  `(rho_bar, eps_bar) = (6 rho_s, 10 eps_s)`, `(d_rho, d_eps) = (0.1 rho_s, 0.2 eps_s)`; amplitude follows `cosh(gamma_k t)`; k = 0.8 … 4.0 fm^-1.
* Collisions: head-on Pb+Pb on fixed target, `E_lab` = 1 … 6 A GeV, 200 events per energy (several hundred in 1302.2956).
  Initial state from UrQMD in cascade mode at `t_init = 2R / sqrt(gamma_cm^2 - 1)`; each hadron is a Gaussian of width
  `sigma_init = 1 fm` (variants 0.7, 0.5 fm) from which `rho, eps, u` are tabulated on the grid.
* Two EoS: two-phase ("std") and its Maxwell one-phase partner; later a scaled family `p_bc(eps, rho) = p0(b eps, c rho)/b` (b = c = 1/1.3).
* Observables: normalised density moments `<rho^N>/<rho>^N` with `<rho^N> = (1/A) Integral rho^N rho d^3r`, N = 2 … 7, versus `t - t_init`;
  mean maximum enhancement versus `E_lab` (N = 5, 7); clump-size distribution above `rho_min = 7 rho_s`;
  transverse flow `U_perp(t)`; end of fluid stage when all cells have `eps < 5 eps_s`.

### 2.2 Linear response (1007.1448)  [paper]

* Dispersion equation with viscosity `xi = 4/3 eta + zeta`, heat conductivity `kappa`, gradient strength `C = a^2 eps_s / rho_s^2`:
  `omega^2 = v_T^2 k^2 + C (rho0^2/h0) k^4 - i xi (omega/h0) k^2 + (v_s^2 - v_T^2) k^2 / (1 + i kappa k^2 / (omega c_v))` — a cubic in `omega`.
* Transport coefficients: `eta = eta0 (c0/c) d h`, `kappa = kappa0 c0 c d c_v`, `d = n^{-1/3}` (n = particle density),
  `c0 = (1/4pi) [(g_g + 3/2 g_q) zeta(3)/pi^2]^{1/3}`; so `xi k^2/h = (4/3) eta0 c0 d k^2` and `kappa k^2/c_v = kappa0 c0 d k^2`.
* Amplification along a phase trajectory `(rho(t), T(t))`: `G_k = exp( Integral_{gamma_k>0} gamma_k(t) dt )`; fluctuation evolution
  `d sigma_k^2/dt = 2 gamma_k sigma_k^2 + 2 D_k` ("Lalime equation").
* EoS (App. A): hadron gas (pi + N + Nbar, interaction `w(rho) = [-A c + B c^2] rho`, `K = 300 MeV`), bag-model QGP (`B = 300 MeV/fm^3`),
  fifth-order polynomial in rho joining the two free-energy densities with matched value, first and second derivative.
  Critical temperature `T_c ≈ 142.5 MeV`; pure-phase crossing ends at `T_max ≈ 156.4 MeV`.

---

## 3. Pre-work already done and what it showed

Run: `tools/eos/eos_reference_prototype.py` → `tools/eos/eos_reference_values.json`.

| Check | Result | Source value | Verdict |
|---|---|---|---|
| Interaction constants from saturation at rho_s and K = 300 MeV (relativistic Fermi gas) | A = 52.127 MeV, B = 19.105 MeV, E/A = −11.87 MeV, p(rho_s, T=0) = 0 | "binding roughly reproduced" | consistent [proto] |
| `c0` | 0.127788 | 0.12779 | reproduced [proto] |
| Pure-phase coexistence at T = 0 | rho_H = 4.442 rho_s, rho_Q = 10.451 rho_s, mu = 1699.8 MeV, p = 379.5 MeV/fm^3 | Fig. 10 of 1007.1448: ≈ 4.4 and ≈ 10.4 (read by eye) | reproduced [proto] |
| T = 0 quintic with matching points 4.2 and 10.5 rho_s (read by eye from Fig. 11) | coexistence 4.500 / 10.458 rho_s; spinodal 5.488 / 8.610 rho_s; max Delta f = 28.8 MeV/fm^3 | Fig. 11: ≈ 4.55 / 10.5 and ≈ 5.5 / 8.7 (by eye) | reproduced [proto] |
| Interface tension at T = 0 | sigma_0 = 3.49 / 11.52 / 17.45 MeV/fm^2 for a = 0.01 / 0.033 / 0.05 fm | "≈ 10 MeV/fm^2" for a = 0.033 fm | consistent [proto] |
| Counter-streaming points of Fig. 1 (1209.2462): `rho = 2 gamma_cm rho_s`, `eps = 2 gamma_cm^2 eps_s` | E_lab 1, 2, 5, 10 A GeV → (2.48, 3.06), (2.87, 4.13), (3.83, 7.32), (5.03, 12.64) | circles in Fig. 1 at the same places (by eye) | reproduced [proto] |
| `T_max` (p_H = p_Q at mu = 0) | 157.32 MeV | 156.4 MeV | **0.9 MeV off, unresolved.** Dropping nucleons, m_pi = 138 MeV or B^{1/4} = 220 MeV does not explain it (157.06 / 157.33 / 157.97 MeV; `T_max.variants` in the JSON). |
| Temperature dependence of the spline points | see below | T_c ≈ 142.5 MeV | **not specified in the papers [open]** |

Finding on the temperature dependence (the one real gap in the published specification):

1. With matching densities kept at fixed fractions of the pure-phase coexistence densities, the quintic stays concave up to
   `T_max`: there is no critical point. The papers adjust the matching densities "for each T" without giving the rule.
2. Widening both matching points by a common factor (`rho~_H = 0.9455 (1-u) rho_H(T)`, `rho~_Q = 1.0046 (1+u) rho_Q(T)`) removes the
   concavity at 142.5 MeV for `u = 0.102`, with the last unstable density at 2.57 rho_s (Fig. 10 shows the tip near 2.6 rho_s;
   Fig. 1 of 1209.2462 puts the critical point near 2.75 rho_s, 7.4 eps_s — both by eye).
3. With this multiplicative widening a convex quintic still exists at 146 MeV (u between about 0.2 and 0.7) but **not** at 150, 153 or 156 MeV for any u up to 0.9
   (table `T_dependence_experiments.min_f2_by_T` in the JSON). Temperature-independent, wide matching intervals behave differently: 0.5–8 rho_s stays convex up to
   157 MeV (minimum of f'' = 223, 120, 36 MeV fm^3 at 150, 154, 157 MeV) and turns concave at 165 MeV and above; narrower fixed intervals fail earlier
   (table `fixed_matching_experiments`). So the matching points must move from "tied to the coexistence densities" below T_c to "wide and nearly fixed" towards T_max,
   and the plasma branch alone must be used above T_max. Choosing that interpolation smoothly is a design task (WP2b), not a transcription task.
4. In 1007.1448 the value of `C` (or `a`) is not printed. The quoted ideal-fluid numbers (fastest growth time 1.38 fm/c at
   k0 ≈ 2.0 fm^-1, rho = 6.5 rho_s, T = 70 MeV) imply `|v_s| = 0.512` and `C rho0^2/h0 = 0.0328 fm^2` [derived from the ideal dispersion relation].
   The earlier paper arXiv:0903.4736 used a = 0.2 fm with a different EoS; 1209.2462 uses 0.033 fm.

Recommendation: **ask the authors for the tabulated `p0(eps, rho)` of PRL 109, 212301 and its Maxwell partner.**
If it is available, WP2b reduces to a format conversion and the quantitative comparison becomes meaningful.

One further point found while preparing the interface test [derived]: the paper's pressure `p0 - C rho Laplacian rho` is the
small-amplitude form. The stress that follows from the free energy `f + (C/2)|grad rho|^2` is the Korteweg tensor
`P_ij = (p0 - C rho Laplacian rho - (C/2)|grad rho|^2) delta_ij + C d_i rho d_j rho`. With the paper's form a planar equilibrium
interface is not exactly stationary (`p_xx = p_coex - (C/2) rho'^2` instead of `p_coex`). Replication uses the paper's form;
the full tensor is a switchable alternative (decision S07).

---

## 4. Repository and architecture

One public repository, containing both the fluid code and the linear-response library.
The SHASTA code is written from the published equations; no source from UrQMD or its hybrid version may be copied
(restrictive user agreement).

```
spinodal-fluid/
  CMakeLists.txt            C++20; Kokkos (pinned release) is the only mandatory dependency
  cmake/                    FetchContent: Kokkos, nlohmann_json (config), doctest (tests); presets cpu-serial, cpu-openmp, cuda
  include/spf/ , src/
    grid.hpp                uniform Cartesian grid, ghost layers, boundary conditions (periodic | outflow)
    shasta.hpp/.cpp         1D sweeps (explicit | phoenical antidiffusion), flux limiter
    eos.hpp                 device-callable EoS structs, no virtual functions (Sect. 4.1, WP2a)
    eos_analytic.cpp        ideal (p = eps/3), gamma-law, cold+thermal
    eos_table.cpp           tabulated p0(eps, rho), T, mu, s; scaled wrapper (b, c)
    primitives.cpp          conserved -> (eps, rho, v)
    gradient.cpp            finite-range pressure (paper form | Korteweg)
    hydro3d.cpp             dimensional splitting, half-step, vacuum handling (Kokkos kernels)
    init_particles.cpp      OSCAR2013 reader, Gaussian mapping
    init_analytic.cpp       uniform + harmonic perturbation, slabs, spheres, interface profile
    observables.cpp         moments, clumps, U_perp, phase-plane occupancy, conservation monitor
    linear/                 dispersion solver, transport coefficients, trajectory integration (1007.1448)
    io.cpp                  raw binary + JSON sidecar snapshots; CSV time series
  apps/  spf_run, spf_growth, spf_linear, spf_ic
  tests/                    doctest unit tests + regression tests (ctest)
  tools/                    Python: EoS prototype and table builder, SMASH drivers, plotting, digitised figure data
  config/defaults.json      every modelling choice as a named key (Sect. 7)
  docs/                     equations.md, decisions_spinodal.md, validation.md
  .github/workflows/ci.yml  build + ctest on Linux and macOS, Serial and OpenMP backends (CUDA is tested manually, WP3g)
```

Conventions (fixed): natural units with `hbar c = 0.1973269804 GeV fm`; energies in GeV, lengths and times in fm, densities in fm^-3,
`eps`, `p` in GeV/fm^3 (the Python prototype uses MeV — convert at the table boundary only); metric (+,−,−,−);
computational frame = nucleus–nucleus centre-of-mass frame; double precision; one `Kokkos::View<double***>` per field, indexed `(ix, iy, iz)`, ghost layers included.
Conserved fields per cell: `R = gamma rho`, `M_i = (eps + p) gamma^2 v_i`, `E = (eps + p) gamma^2 - p`.

### 4.1 GPU strategy

Reference hardware: a 20-core Linux node with one NVIDIA TITAN RTX (24 GiB). Development also happens on macOS without a usable GPU,
so the same source must run on CPU there.

Where a GPU pays off — all grid work, which is stencil or per-cell with no dependence between cells inside one stage:
SHASTA transport, antidiffusion and limiter stages (5 fields × 3 directions × half and full step), primitive recovery (per-cell root find with table look-ups),
the Laplacian of the gradient term, Gaussian deposition of the initial hadrons, and the reductions (conservation sums, density moments, phase-plane histograms).

Where it does not: SMASH (external, CPU), the connected-component clump finder (run on the host from a copied density field), the linear-response library (WP8; small),
file output, and the Python EoS tools.

Caveat on the expected gain: the standard ensemble is about 2400 independent events on a 160^3 grid, which a 16–20-core CPU already runs concurrently.
The GPU matters most for dx = 0.1 fm (320^3 = 3.3e7 cells; 4.9–7.3 GiB at 20–30 doubles per cell, so it fits the 24 GiB card; 160^3 needs 0.6–0.9 GiB),
for parameter scans and for ML training sets. The gain is not assumed: WP3g measures it and applies the criterion given there.

Technology: **Kokkos**, one source for the Serial and OpenMP backends (CPU, including macOS) and the CUDA backend (TITAN RTX is a Turing card: `Kokkos_ARCH_TURING75`).
Kokkos 5.3.0 (released 2026-11-11) is pinned by tag and commit. Its requirements, read from `cmake/kokkos_compiler_id.cmake` of that tag: C++20, CMake ≥ 3.22,
GCC ≥ 10.4 or Clang ≥ 14, and nvcc ≥ 12.2 for the CUDA backend. (Revision 2 of this plan named 5.1.0 as the current release; that was out of date.)
Alternatives considered and not chosen: native CUDA (no development path on a machine without an NVIDIA GPU, two code paths to keep in agreement), OpenMP target offload and `std::par` (GPU support tied to specific compilers).

Coding rules that keep every kernel device-compatible (apply from WP1 on, also when only the CPU backend is being run):

1. Fields live in `Kokkos::View`; no STL containers, raw `new`, I/O, exceptions, virtual calls, `std::function` or allocation inside a kernel.
2. Kernels are `KOKKOS_LAMBDA` / `KOKKOS_INLINE_FUNCTION`, launched with `parallel_for` over an `MDRangePolicy`.
3. Each SHASTA stage (transport + diffusion, antidiffusive flux, limiter, update) is its own `parallel_for` that reads one set of Views and writes another, so the result does not depend on execution order.
4. An EoS is a trivially copyable struct with `KOKKOS_INLINE_FUNCTION` members (`p0`, `dp_deps`, `dp_drho`, …); table data sit in a `Kokkos::View<double**>` member; the solver is templated on the EoS type and the run-time choice is a switch in the driver, outside the kernels.
5. Primitive recovery uses a safeguarded iteration with a fixed maximum count (≤ 60) and a per-cell failure flag in an integer View, summed with `parallel_reduce`.
6. Reductions use `parallel_reduce`; summation order differs between backends, so backends are compared with a relative tolerance, never bitwise.
7. Host–device copies (`create_mirror_view`, `deep_copy`) happen only at snapshots and are counted and timed.
8. Double precision throughout; single precision is a separate, later option (S15) with its own validation.

Equations in the computational frame:

```
d_t R   + div(R v)   = 0
d_t M_i + div(M_i v) = - d_i p
d_t E   + div(E v)   = - div(p v)
p = p0(eps, rho) + p_grad ,   p_grad = - a^2 (eps_s/rho_s^2) rho Laplacian(rho)
```

---

## 5. Work packages

Rules for every package: (1) code + tests + a short entry in `docs/validation.md` with the measured numbers;
(2) every alternative listed in Sect. 7 for that package is implemented and has at least one test;
(3) a failing acceptance test is reported, never worked around by loosening the tolerance or changing the reference;
(4) no synthetic stand-in for a missing input (EoS table, SMASH events, digitised curves) — stop and report instead.

### WP0 — Repository bootstrap  [E]
* Create the repository, CMake skeleton with Kokkos and the three presets, CI, `config/defaults.json` loader with
  override file and command-line `key=value`, config echo into every output directory, `docs/decisions_spinodal.md` seeded from Sect. 7.
* Record the toolchain on the GPU node (`nvcc --version`, host compiler, CMake, available modules) and build a Kokkos "hello reduction" with the `cuda` preset there.
* Done when: the `cpu-serial` and `cpu-openmp` presets build and pass `ctest` on Linux (CI) and macOS arm64, and the `cuda` preset builds and runs the reduction on the GPU node.

### WP1 — SHASTA in one dimension  [E]
* Written as Kokkos kernels acting along a direction index of the 3D grid (a 1D problem is a grid with ny = nz = 1), following the rules of Sect. 4.1.
* Implement Eqs. (10)–(19) of nucl-th/9504018, Sect. 3.1, for the generic law `d_t U + d_x(U v + f) = 0`:
  `eps_j = lambda v_j^{n+1/2}`, `Q± = (1/2 ∓ eps_j) / (1 ± (eps_{j±1} - eps_j))`,
  `U~_j = 1/2 Q+^2 D_j - 1/2 Q-^2 D_{j-1} + (Q+ + Q-) U_j + lambda Df_j`, `D_j = U_{j+1} - U_j`,
  source `Df_j = -1/2 (f_{j+1} - f_{j-1})` (their Eq. 14), antidiffusion flux explicit `A_j = eta_ad D~_j` or phoenical
  `A_j = eta_ad (D~_j - 1/8 [D_{j+1} - 2 D_j + D_{j-1}])`, limiter `A~_j = s max{0, min[s D~_{j+1}, |A_j|, s D~_{j-1}]}`, `s = sgn A_j`,
  `U_j^{n+1} = U~_j - A~_j + A~_{j-1}`. Defaults: phoenical, `eta_ad = 1/8`, `lambda = dt/dx = 0.4`.
* Half-step method: advance by `dt/2` using `v^n, f^n`, recover primitives, then advance `U^n` by the full `dt` using `v^{n+1/2}, f^{n+1/2}`.
  Read Sect. 3 and the Appendix of the paper before coding; where the text and this summary differ, the paper wins and the difference is reported.
* Relativistic 1D system with `U = (R, M, E)`, `f = (0, p, p v)`; after each step enforce `E >= |M|` as in the paper; vacuum cells (`E < E_vac`, default 1e-12 GeV/fm^3) are zeroed with `v = 0`.
* Tests: (T1.1) advection of a square and a Gaussian with constant v: total conserved to 1e-12 relative, no new extrema;
  (T1.2) relativistic shock tube, `p = eps/3`, against the exact Riemann solution: L1 error decreases with resolution (report the measured order);
  (T1.3) expansion of semi-infinite matter into vacuum, `p = eps/3`: similarity profile of nucl-th/9504018 Sect. 4 reproduced after 100 steps (L1 error reported; compare with their Fig. 6 qualitatively);
  (T1.4) explicit vs phoenical vs reduced `eta_ad` all run and are selectable.

### WP2a — EoS interface and analytic EoS  [E]
* Interface (a concept, not a base class — Sect. 4.1 rule 4): `p0(eps, rho)`, `dp_deps`, `dp_drho`, `vs2(eps, rho) = p_eps + (rho/h) p_rho`, optional `T`, `mu`, `s`; `name()`, `valid_range()`.
* `ideal`: `p = eps/3`. `gamma_law`: `p = (Gamma - 1)(eps - m rho)`.
* `cold_thermal` (development EoS, fully specified): `p0 = p_c(rho) + (Gamma_th - 1)(eps - eps_c(rho))`, `Gamma_th = 4/3` default, where
  `eps_c(rho) = f_{T=0}(rho)` and `p_c = rho eps_c' - eps_c` come from the T = 0 isotherm of Sect. 3 (variants: `two_phase`, `maxwell`).
  Note: this EoS is for code development. Its spinodal is that of the T = 0 isotherm and is stronger than at the temperatures reached in collisions, so Stage-A collision
  results are not to be compared with the paper quantitatively.
  `tools/make_cold_curve.py` (wrapping the prototype) writes the 1D table (rho from 0 to 16 rho_s, step 0.005 rho_s); C++ uses a cubic spline of `eps_c` so that `p_c` and `p_c'` are continuous.
  Cells with `eps < eps_c(rho)` are clamped to `eps_c` for the pressure evaluation (the "forbidden region" of Fig. 1).
* Tests: (T2.1) coexistence and spinodal densities of the cold curve equal the JSON values (4.500/10.458 and 5.488/8.610 rho_s) to 0.01 rho_s;
  (T2.2) Maxwell variant has `dp_c/drho = 0` between the coexistence densities and is identical outside;
  (T2.3) `vs2` from the interface equals a finite-difference isentropic derivative to 1e-6.

### WP2b — Finite-temperature two-phase table and Maxwell partner  [S]  (Python, `tools/eos/`)
* Preferred route: obtain the original table from the authors; then only convert and run the acceptance checks.
* Otherwise design the temperature dependence of the matching densities. Constraints: `f(T, rho)` twice continuously differentiable in T
  across T_c (so that `eps = f - T d_T f` and `c_v` are continuous); concave anomaly for T < T_c only; convex for T > T_c; pure plasma for T ≥ T_max;
  `c_v > 0` everywhere. Starting point: findings 1–3 of Sect. 3. Candidate constructions for T_c < T < T_max: matching densities that move smoothly from the widened coexistence densities to a wide,
  nearly fixed interval (finding 3); a lower-order Hermite join; or a smooth switch in the grand-canonical pressure `p(T, mu)`. Each candidate is a named alternative.
* Outputs: `eos_two_phase.bin/.json`, `eos_maxwell.bin/.json` on a uniform `(eps/eps_s, rho/rho_s)` lattice
  (default `rho` 0…16 step 0.02, `eps` 0…40 step 0.05; fields `p0, T, mu, s`), plus the canonical tables `f, p, eps, s, mu` on `(T, rho)`.
* Acceptance: T_c within 142.5 ± 0.5 MeV; T = 0 numbers of Sect. 3; coexistence, isothermal and isentropic spinodal lines within 0.15 rho_s / 0.3 eps_s
  of curves digitised from Fig. 1 of 1209.2462 and Figs. 1, 10 of 1007.1448 (vector graphics; digitise from the PDF paths, not by eye);
  `sigma_T(T)` for a = 0.01/0.033/0.05 fm against Fig. 2 (upper) of 1209.2462; ideal growth time at (6.5 rho_s, 70 MeV) against 1.38 fm/c at k0 ≈ 2 fm^-1
  once `C` is fixed; thermodynamic consistency `p = T s - eps + mu rho` to 1e-8 on the canonical grid.
* The `T_max` discrepancy (157.3 vs 156.4 MeV) is investigated here and its outcome recorded.

### WP2c — Table reader and scaled EoS  [E]
* Reader for the WP2b format (raw little-endian doubles + JSON header); bilinear interpolation (default) and bicubic (alternative); derivative fields
  by centred differences of the table; clamp below the cold curve; error (not extrapolation) outside the lattice unless `eos.extrapolate = true`.
* Wrapper `p_bc(eps, rho) = p0(b eps, c rho)/b`, `T_bc = T0/b`, `mu_bc = (c/b) mu0` (1302.2956, Eqs. 17–18).
* Tests on a synthetic table written from `cold_thermal`: interpolation error below 1e-3 relative; scaled wrapper reproduces `vs2_bc(eps, rho) = vs2_0(b eps, c rho)`.
  (A synthetic table is allowed here because the test checks the reader, not physics.)

### WP3 — 3+1D solver  [E]
* Dimensional splitting: one SHASTA sweep per direction per time step; order `xyz` cyclically permuted each step (alternative: fixed).
  In the sweep along `i`: `U ∈ {R, M_x, M_y, M_z, E}`, `f = p` for `M_i`, `f = p v_i` for `E`, `f = 0` otherwise.
* Primitive recovery per cell: solve `v = |M| / (E + p(eps, rho))` with `eps = E - |M| v`, `rho = R sqrt(1 - v^2)` for `v ∈ [0, 1)` by a bracketing
  method (Brent, tolerance 1e-12, bisection fallback); direction of v along M. `p` includes `p_grad` taken from the last evaluated density field (lagged; alternative: force-only, Sect. 7 S08).
  Count and report cells where no root or more than one sign change is found on a 64-point scan (diagnostic mode only).
* Vacuum and limiter as in WP1; boundary conditions periodic or outflow; snapshots and a conservation monitor
  (total `R`, `E`, `M_i` every step).
* Tests: (T3.1) 1D problems of WP1 embedded along x, y, z give identical profiles to 1e-12; (T3.2) periodic box, random smooth initial state, `p = eps/3`:
  totals conserved to 1e-11 relative over 500 steps; (T3.3) spherical blast: octant symmetry to 1e-10, comparison of fixed vs permuted sweep order reported;
  (T3.4) stable sound wave in `cold_thermal` at rho = 2 rho_s: measured phase velocity within 2 % of `v_s` for k ≤ 2 fm^-1 at dx = 0.2 fm;
  (T3.5) performance: cell-updates per second with the Serial backend and with OpenMP on 16 threads recorded (needed for the WP7 budget).

### WP3g — GPU validation and benchmark  [E, review gate G1]
* Build with the `cuda` preset on the GPU node; run the complete WP1–WP3 test set there (and the WP4 set once it exists).
* Acceptance: after 100 steps of T3.2 every field agrees with the Serial backend to 1e-11 relative (cells above the vacuum threshold); conservation totals to 1e-11;
  growth rates of T4.1 agree to 1e-6 relative. Any larger difference is a bug to be found, not a tolerance to be widened.
* Benchmark on 160^3 and 320^3 with the tabulated-EoS reader in the loop: cell-updates per second for Serial, OpenMP on 20 threads, CUDA; time per event (150 steps);
  events per hour for (a) one GPU, (b) the 20 cores running independent events concurrently. Report peak device memory.
* Criterion: the GPU is the production backend for a given grid size if (a) ≥ (b); otherwise production runs on CPU and the GPU is kept for the 320^3 checks. Both numbers go into `docs/validation.md`.
* If several events fit in device memory, add an event-batch mode only after the single-event numbers are reviewed.

### WP4 — Finite-range term and spinodal growth  [E, review gate G2]
* `p_grad = -a^2 (eps_s/rho_s^2) rho Lap(rho)` with the 7-point second-order Laplacian of the rest-frame density, evaluated at step n and at the half step.
  Alternative stencil: 4th order. Alternative stress: Korteweg (extension, WP9).
* `spf_growth`: periodic box, harmonic perturbation, Fourier amplitude of rho versus time, fit `cosh(gamma_k t)` over the window where the amplitude
  is between 1.05 and 3 times its initial value; scan k = 0.4 … 6 fm^-1.
* Tests with `cold_thermal/two_phase` at the spinodal point `rho = 5.8 rho_s`, thermal energy `2.0 eps_s` above the cold curve (`eps = 9.97 eps_s`), for which the prototype gives
  `vs2 = -0.0543` (|v_s| = 0.233), `h = 1.988 GeV/fm^3`, and growth-rate maxima `gamma = 0.527 c/fm` at `k = 3.20 fm^-1` (a = 0.033 fm, cut-off 4.53 fm^-1) and
  `0.348 c/fm` at `2.11 fm^-1` (a = 0.05 fm) [proto, `cold_thermal_points` in the JSON]. This point is chosen because its fastest mode is resolved at dx = 0.2 fm;
  deeper in the cold spinodal the instability is much stronger (`vs2 = -0.41` at 7 rho_s on the cold curve, fastest mode near 7 fm^-1) and under-resolved. Tests:
  (T4.1) for a = 0.033 and 0.05 fm, measured `gamma_k` within 10 % of `sqrt(|v_s|^2 k^2 - a^2 (eps_s/h)(rho/rho_s)^2 k^4)` for k up to the maximum of the curve, and the
  position of the maximum within 10 %; (T4.2) stable side: modes with k above the cut-off oscillate, measured `omega^2 = v_s^2 k^2 + a^2 (eps_s/h)(rho/rho_s)^2 k^4` within 10 %;
  (T4.3) a = 0: numerical `gamma_k` falls below `|v_s| k` at large k, and `gamma_{2k}[dx/2] = 2 gamma_k[dx]` holds to 5 % (statement in 1302.2956, Sect. III B);
  (T4.4) a = 0.033 fm: `gamma_k` at dx = 0.2 and 0.1 fm agree within 10 %; (T4.5) stability scan in `lambda ∈ {0.2, 0.4}` and a up to 0.05 fm: no growth of grid-scale
  (k = pi/dx) noise in a stable uniform state over 10 fm/c. Estimate to be checked [derived]: the gradient term has a grid-scale frequency
  `a sqrt(eps_s/h) (rho/rho_s) (pi/dx)^2` ≈ 15 fm^-1 at 6 rho_s, i.e. `omega dt` ≈ 1.2 at dt = 0.08 fm — close to an explicit-stability limit.
* With the WP2b table: repeat at `(6 rho_s, 10 eps_s)` with `(0.1 rho_s, 0.2 eps_s)` and compare with Fig. 2 (lower) of 1209.2462 / Fig. 4 of 1302.2956 (digitised).
* Gate G2: the project lead reviews growth-rate plots before any collision run.

### WP5 — Initial conditions  [E]
* `tools/smash/`: SMASH 3.4.2 (existing local build) config for Pb+Pb, impact parameter 0, fixed-target kinetic energy per nucleon 1…6 GeV, cascade
  (no potentials), calculation frame = centre of velocity, particle output (OSCAR2013) at `t_init`. Kinematics from the JSON
  (`sqrt(s_NN)` = 2.327, 2.701, 3.029, 3.325, 3.596, 3.849 GeV for 1…6 A GeV). `t_init = 2R/sqrt(gamma_cm^2 - 1)` counted from first contact of the
  nuclear surfaces; `R` is a config key (`1.2 A^{1/3}` = 7.11 fm default → 11.26 fm/c at 3 A GeV; alternative 6.62 fm → 10.48 fm/c).
  SMASH's own time-zero convention must be read from its documentation and verified by measuring the time of maximum central density; report the offset.
* `init_particles`: each hadron deposits baryon number, energy and momentum with a normalised Gaussian of width `sigma_init`
  (truncated at 4 sigma, renormalised on the grid so that the sums are exact): `N^0 = sum B_i G`, `T^{00} = sum E_i G`, `T^{0j} = sum p_i^j G`.
  Variant with the Gaussian Lorentz-contracted along z by `gamma_z` of the particle: read arXiv:0710.0332 and arXiv:0806.1695 and make the variant they describe the default; record which.
  Then set `R = N^0`, `E = T^{00}`, `M = T^{0j}` and recover primitives with the chosen EoS (cells failing `E >= |M|` are limited as in WP1 and counted).
* `init_analytic`: (i) uniform + harmonic perturbation (WP4), (ii) two counter-streaming Lorentz-contracted Woods–Saxon nuclei at full overlap (smooth test state),
  (iii) Glauber-sampled nucleon positions with the same Gaussian mapping (fallback when no cascade events are available; never a silent substitute).
* Tests: (T5.1) total baryon number, energy and momentum on the grid equal the particle sums to 1e-10; (T5.2) for 3 A GeV the event-averaged central
  `(rho, eps)` at `t_init` lies between the counter-streaming point (3.22 rho_s, 5.19 eps_s) and the cold curve — value reported, compared with the
  dashed contours of Fig. 1 after evolution; (T5.3) reader round trip on a hand-written 3-particle file.
* If the user supplies UrQMD cascade output, the same reader path is used after conversion to OSCAR2013 (no UrQMD source in the repository).

### WP6 — Observables  [E]
* Density moments for N = 1…7 from the definition in Sect. 2.1; the density entering is a config key (S10): computational-frame `N^0`
  (default; then `A` is exactly the baryon number) or rest-frame `rho`. Enhancement = moment(t)/moment(t_init); maximum over time; ensemble mean and
  standard error (1209.2462 reports event-to-event spread up to a factor ≈ 5).
* Clump finder: 6-connected regions with `rho > rho_min` (default 7 rho_s), baryon number per region, histogram.
* `U_perp(t)` (1302.2956, Eqs. 14–16); phase-plane histograms of baryon number in `(rho, eps)`; baryon fraction inside the coexistence, isothermal-spinodal and
  isentropic-spinodal regions versus time (needs the region masks from WP2b; with `cold_thermal` use `vs2 < 0`); time when all cells have `eps < 5 eps_s`.
* Tests on analytic fields: uniform density gives all normalised moments = 1 to 1e-12; a sum of N identical well-separated Gaussians gives the closed-form moment
  `<rho^N>/<rho>^N` (derive and check numerically to 1e-3); clump finder on planted blobs returns their baryon numbers to 1e-3.

### WP7 — Collision runs and comparison with the papers  [E to run, S to review: gate G3]
* Stage A (development EoS `cold_thermal`, two-phase vs Maxwell): 20 events at 3 A GeV, dx = 0.2 fm, 6 fm/c after `t_init`; grid extent chosen so that less than
  1e-4 of the baryon number leaves the box (reported). Purpose: stability, cost, qualitative two-phase/Maxwell difference.
* Stage B (WP2b tables): 200 events at each of 1, 2, 3, 4, 5, 6 A GeV, both EoS; then the variations a = 0.01/0.05 fm, sigma_init = 0.7/0.5 fm, scaled EoS b = c = 1/1.3, at 3 A GeV first.
* Resolution check at 3 A GeV with dx = 0.1 fm on 5 events (moments may change; the change is reported, not hidden).
* Comparison targets (digitised from the vector figures; tolerances are for the qualitative statement, since initial states and EoS details differ):
  two-phase N = 7 mean enhancement exceeds the Maxwell one by more than a factor 3 at 3 A GeV (paper: "well over an order of magnitude" for the two-phase moment, Maxwell ≈ none);
  the mean maximum enhancement versus E_lab peaks between 2 and 4 A GeV for the two-phase EoS and is flat for Maxwell;
  smaller `a` gives larger enhancement; clump-size distribution: two-phase minus Maxwell peaks at 5–8 baryons (1302.2956 Fig. 7);
  `U_perp` differs by only a small amount between the EoS. Any target not met is reported with the measured value.
* Budget estimate (not measured; to be replaced by T3.5 and the WP3g benchmark, which also decides CPU vs GPU for production): a 160^3 grid has 4.1e6 cells, 75–150 steps per event; of order minutes per event on 16 threads;
  the full Stage B is about 2400 events plus variations.

### WP8 — Linear-response library (1007.1448)  [E; final numbers need WP2b]
* Cubic dispersion solver (roots of the polynomial obtained by clearing the denominator of the dispersion equation), `gamma_k = max Im omega`;
  transport-coefficient ansatz with `(eta0, kappa0)`; particle density `n` for `d = n^{-1/3}`: pions + nucleons + antinucleons in the hadron gas, gluons + quarks + antiquarks in the plasma,
  interpolated linearly in rho across the spline region [open; config key]. `G_k` and the Lalime equation along trajectories `(rho(t), T(t))`.
* Trajectories: digitise Figs. 4 and 5 of 1007.1448 (3-fluid and UrQMD, 5 A GeV) from the PDF vector paths into `tools/data/`.
* Tests independent of the nuclear EoS: (T8.1) kappa = 0, xi = 0 reduces to `gamma_k^2 = |v_s|^2 k^2 - C (rho^2/h) k^4`; (T8.2) small xi: `gamma_k` shifts by `-(1/2)(xi/h) k^2` to first order;
  (T8.3) with kappa > 0 the instability boundary moves from `v_s = 0` to `v_T = 0` (checked on a toy EoS with analytic `v_s`, `v_T`, `c_v`);
  (T8.4) stationary Lalime solution equals Eq. (48). With WP2b: Fig. 2 (four dissipation combinations at 6.5 rho_s, 70 MeV), Fig. 3, Fig. 6 and Fig. 7
  (peak of `G_k` at k0 ≈ 1.3–1.6 fm^-1) within 20 % after `C` is fixed from the 1.38 fm/c / 2.0 fm^-1 pair.

### WP9 — Extensions (planned after G3)
See Sect. 9.

### Order and gates
```
WP0 -> WP1 -> WP3 -> WP3g -> WP4 -> WP5 -> WP6 -> WP7(A) --G3--> WP7(B)
        WP2a ----^      ^                       ^
        WP2b [S] -------+--- WP2c --------------+        WP8 in parallel after WP2a
```
G1 after WP1 + WP3 + WP3g (solver tests and CPU–GPU agreement reviewed); G2 after WP4 (growth rates); G3 after WP7 Stage A; G4 after Stage B (comparison with the papers).

---

## 6. Validation matrix against the papers

| Paper item | Package | Kind |
|---|---|---|
| 1007.1448 Figs. 8–12 (pure phases, T = 0 spline) | WP2a/2b | quantitative; T = 0 part already reproduced in the prototype |
| 1007.1448 Fig. 1; 1209.2462 Fig. 1; 1302.2956 Fig. 2 (phase diagram) | WP2b | quantitative after calibration |
| 1209.2462 Fig. 2 upper (sigma_T) | WP2b | quantitative; T = 0 end reproduced (11.5 vs ≈ 10 MeV/fm^2) |
| 1209.2462 Fig. 2 lower / 1302.2956 Figs. 3–4 (growth rates) | WP4 | quantitative (analytic formula), numerical dissipation trend |
| 1209.2462 Fig. 3 / 1302.2956 Fig. 5 (moments vs time) | WP7 | qualitative + factor-level |
| 1209.2462 Fig. 4 / 1302.2956 Fig. 6 (enhancement vs E_lab) | WP7 | position of the maximum; magnitude factor-level |
| 1302.2956 Figs. 7–14 (clumps, U_perp, parameter studies) | WP7 | qualitative trends |
| 1007.1448 Figs. 2, 3, 6, 7 (dissipative growth, G_k) | WP8 | quantitative after WP2b |

---

## 7. Decision register for the new project (all switchable; * = default)

| ID | Config key | Alternatives |
|---|---|---|
| S01 | `ic.generator` | `smash_cascade`* / `oscar_file` (user-supplied, e.g. converted UrQMD) / `glauber_gaussians` / `analytic` |
| S02 | `ic.t_init_radius_fm` | `1.2*A^(1/3)`* / `6.62` / explicit time |
| S03 | `ic.sigma_init_fm` | `1.0`* / `0.7` / `0.5` |
| S04 | `ic.gaussian_contraction` | as in arXiv:0710.0332 (to be read)* / `none` |
| S05 | `eos.model` | `table_two_phase`* / `table_maxwell` / `cold_thermal_two_phase` / `cold_thermal_maxwell` / `ideal` / `gamma_law`; `eos.scale_b`, `eos.scale_c` = 1* |
| S06 | `eos.table_interpolation` | `bilinear`* / `bicubic` |
| S07 | `gradient.form` | `paper_pressure`* / `korteweg_full` / `off`; `gradient.a_fm` = 0.033* / 0 / 0.01 / 0.05 |
| S08 | `gradient.in_primitive_recovery` | `lagged`* / `force_only` |
| S09 | `shasta.antidiffusion` | `phoenical`* / `explicit`; `shasta.eta_ad` = 0.125*; `shasta.lambda` = 0.4*; `shasta.sweep_order` = `permuted`* / `fixed`; `grid.dx_fm` = 0.2* / 0.1 |
| S10 | `obs.moment_density` | `computational_frame`* / `rest_frame`; `obs.clump_rho_min` = 7 rho_s* |
| S11 | `eos.finite_T_construction` (WP2b) | chosen construction* / each rejected candidate kept as an alternative |
| S12 | `linear.particle_density_in_spline_region` | `linear_in_rho`* / `hadron_side` / `plasma_side` |
| S13 | `eos.hadron_constants` | relativistic Fermi gas (A = 52.127, B = 19.105 MeV)* / non-relativistic as written in the paper (A = 52.385, B = 19.048 MeV) |
| S14 | `run.backend` (build preset) | `cpu-openmp`* / `cpu-serial` (reference) / `cuda`; production backend fixed after WP3g |
| S15 | `run.precision` | `double`* / `single` (later, needs its own validation) |

Superseded choices are appended, not deleted. 

---

## 8. Project decisions (2026-11-20)

1. One public repository, BSD-3-Clause; the linear-response library lives inside it.
2. GPU use where it gives a measured gain (Sect. 4.1, WP3g).
3. SMASH in cascade mode is the default initial-state generator. UrQMD cascade events, as used in the papers, can be supplied as OSCAR files through the same reader;
   until then comparisons with the published figures carry the caveat "different cascade model for the initial state".
4. The original equation-of-state table of the papers is not available to the project at present. All packages except WP2b and Stage B of WP7 use the development
   equation of state; the table reader accepts any table in the WP2b format, so an author-supplied table can be used later.

---

## 9. Extensions after the replication

1. **Consistent capillary stress** (S07 `korteweg_full`) and the matching gradient energy `(1/2) a^2 eps_s |grad(rho/rho_s)|^2`; test: planar interface stays stationary and the numerically integrated tension equals the Eq. (8) value (11.52 MeV/fm^2 at T = 0, a = 0.033 fm).
2. **Dissipation in the 3D code**: shear + bulk viscosity and heat conduction, validated against the WP8 dispersion relation (heat conduction moves the instability boundary to the isothermal spinodal).
3. **Thermal noise** (fluctuating hydrodynamics, Kapusta–Müller–Stephanov, arXiv:1112.6405) as a source of seeds besides the initial state.
4. **Particlisation and afterburner**: Cornelius-type surface at `eps = 5 eps_s` → Cooper–Frye sampling with baryon number → SMASH afterburner; light-nuclei yield ratios (Sun, Ko, Li, arXiv:2006.08929) and baryon-number cumulants as observables; EoS dependence as in Steinheimer, Randrup, Koch (arXiv:1311.0999).
5. **Transport counterpart**: Vlasov dynamics with a density-functional mean field that has a first-order transition (Sorensen & Koch, arXiv:2011.06635; availability of these potentials in SMASH 3.4.2 to be verified) — compare spinodal growth rates and clumping between transport and this fluid code at the same EoS.
6. **Machine learning**: (a) per-event classification "two-phase vs Maxwell" from coordinate-space and momentum-space baryon distributions, following Steinheimer, Pang, Zhou, Koch, Randrup, Stöcker, JHEP 12 (2019) 122 (arXiv:1906.06562), whose abstract reports clear coordinate-space signals and weak momentum-space ones; (b) a Gaussian-process emulator of the maximum enhancement as a function of (E_lab, a, sigma_init, b, c) for sensitivity analysis and Bayesian constraints.

---

## 10. Instructions to implementers

* Work one package at a time in the order of Sect. 5; do not start a package whose inputs are missing.
* Before coding a package, read the cited sections of the papers (arXiv identifiers in Sect. 11); equations in this plan are summaries.
* Keep every number in `docs/validation.md` traceable to a test or script in the repository.
* Stop and report (do not improvise) when: an acceptance test fails after two honest attempts; a paper and this plan disagree; a required input is unavailable;
  a choice not listed in Sect. 7 has to be made.
* Never loosen a tolerance, replace a reference value, shorten an ensemble or reduce the resolution silently. If any of these is done on request, it is stated at the top of the report.
* Follow the device rules of Sect. 4.1 in every kernel, even while only the CPU backend is being built; a kernel that cannot run on the device is not done.
* Packages marked [S] (WP2b, gates G1–G4, WP9 planning) are started only by the project lead.

---

## 11. References (identifiers verified against arXiv on 2026-11-20)

* J. Steinheimer, J. Randrup, PRL 109, 212301 (2012), arXiv:1209.2462 — target paper 1.
* J. Randrup, PRC 82, 034902 (2010), arXiv:1007.1448 — target paper 2.
* J. Steinheimer, J. Randrup, PRC 87, 054903 (2013), arXiv:1302.2956 — detailed account (grid, t_init, growth-rate extraction, clumps, scaled EoS).
* J. Randrup, PRC 79, 054911 (2009), arXiv:0903.4736 — gradient term, interface tension.
* D.H. Rischke, S. Bernard, J.A. Maruhn, NPA 595, 346 (1995), arXiv:nucl-th/9504018 — SHASTA, one dimension.
* D.H. Rischke, Y. Pürsün, J.A. Maruhn, NPA 595, 383 (1995), arXiv:nucl-th/9504021 — part II (not yet read for this plan; check for multi-dimensional details).
* J. Steinheimer et al., PRC 77, 034901 (2008), arXiv:0710.0332; H. Petersen et al., PRC 78, 044901 (2008), arXiv:0806.1695 — cascade → fluid mapping.
* I.C. Arsene et al., PRC 75, 034902 (2007), arXiv:nucl-th/0609042 — phase trajectories used in 1007.1448.
* J. Steinheimer, J. Randrup, V. Koch, PRC 89, 034901 (2014), arXiv:1311.0999; J. Steinheimer et al., JHEP 12 (2019) 122, arXiv:1906.06562;
  K.-J. Sun, C.M. Ko, F. Li, arXiv:2006.08929; A. Sorensen, V. Koch, PRC 104, 034904 (2021), arXiv:2011.06635;
  J.I. Kapusta, B. Müller, M. Stephanov, PRC 85, 054906 (2012), arXiv:1112.6405; J. Weil et al. (SMASH), PRC 94, 054905 (2016), arXiv:1606.06642.
* Kokkos: https://github.com/kokkos/kokkos and https://kokkos.org/kokkos-core-wiki/get-started/requirements.html (release and compiler requirements, read 2026-11-20).
