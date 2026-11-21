# Equations as implemented

Units: GeV, fm, c = 1; energy density and pressure in GeV/fm^3, baryon density in fm^-3.
Source of the algorithm: D.H. Rischke, S. Bernard, J.A. Maruhn, Nucl. Phys. A 595, 346 (1995), arXiv:nucl-th/9504018
(Sect. 3.1, 3.3, 4). Equation numbers below refer to that paper.

## 1. Conservation laws in one direction

For the conserved fields

    R = gamma rho,     M = (eps + p) gamma^2 v,     E = (eps + p) gamma^2 - p

ideal relativistic hydrodynamics in direction x reads  d_t U + d_x (U v + f) = 0  with

| U | f |
|---|---|
| R | 0 |
| M | p |
| E | p v |

The pressure is `p = eos.p0(eps, rho)`; no other property of the equation of state enters the transport.

## 2. SHASTA step (`include/spf/shasta.hpp`, `shasta_step`)

Cell index j along the sweep direction, `lam = dt/dx`, `D_j = U_{j+1} - U_j`:

    e_j  = lam v_j
    Q+   = (1/2 - e_j) / (1 + (e_{j+1} - e_j)),     Q- = (1/2 + e_j) / (1 - (e_{j-1} - e_j))          (Eq. 12)
    Df_j = -1/2 (f_{j+1} - f_{j-1})                                                                     (Eq. 14)
    U~_j = 1/2 Q+^2 D_j - 1/2 Q-^2 D_{j-1} + (Q+ + Q-) U_j + lam Df_j                                  (Eq. 11)
    D~_j = U~_{j+1} - U~_j                                                                              (Eq. 15)
    A_j  = eta_ad D~_j                                         (explicit, Eq. 16)
    A_j  = eta_ad ( D~_j - 1/8 [D_{j+1} - 2 D_j + D_{j-1}] )   (phoenical, Eq. 17)
    A~_j = s max{0, min[s D~_{j+1}, |A_j|, s D~_{j-1}]},  s = sgn A_j                                  (Eq. 18)
    U_j^{n+1} = U~_j - A~_j + A~_{j-1}                                                                  (Eq. 19)

The code follows the paper: the summary in the work-package brief and Eqs. (11)-(19) agree term by term. The
constant 1/8 inside the phoenical bracket is fixed; `eta_ad` (default 1/8) multiplies the whole flux. `s = +1` for
`A_j = 0` (then `A~_j = 0` anyway).

Every stage is a separate `parallel_for` that reads one set of Views and writes another:
(1) `U~` from `U, v, f`; (2) `A` from `U, U~`; (3) `A~` from `A, U~`; (4) `U^{n+1}` from `U~, A~`.
The output View must not alias the input View.

### Ghost width

Counting the dependencies backwards from the update of interior cell j:

* `U_j^{n+1}` needs `A~_j` and `A~_{j-1}`;
* `A~_j` needs `A_j`, `D~_{j+1}`, `D~_{j-1}`, i.e. `U~_{j-1} ... U~_{j+2}`; `A~_{j-1}` needs `U~_{j-2} ... U~_{j+1}`;
* `A_j` needs `D~_j` and `D_{j-1}, D_j, D_{j+1}`, i.e. `U_{j-1} ... U_{j+2}`;
* `U~_m` needs `U_{m-1}, U_m, U_{m+1}`, `v_{m-1..m+1}`, `f_{m-1}, f_{m+1}`.

So `U~` is needed on `[j-2, j+2]`, which needs `U, v, f` on `[j-3, j+3]`: **the ghost width is 3 cells on each side
along the sweep direction** (`kShastaGhost`). The stages run on the ranges `U~: [g-2, g+n+2)`, `A, A~: [g-1, g+n)`,
`update: [g, g+n)` (g = ghost width, n = interior cells). A 1D problem has `ny = nz = 1` and ghost layers along x only.
Transverse ghost layers are not used by a sweep.

Boundary conditions are applied by `fill_ghost` before each sweep: periodic (ghost cell a on the left <- interior cell
`a + n`, on the right <- cell `g + a`) or outflow (zero gradient: copy of the outermost interior cell).
Config key `grid.boundary` = `periodic` (default) | `outflow`.

## 3. Half-step sequence (`Hydro1D::step`)

With `lam = dt/dx` (`shasta.lambda`):

1. fill ghosts of `U^n`, recover primitives `(eps, rho, v, p)` and `p v` on all cells including ghosts;
2. `U^{n+1/2} = SHASTA(U^n; v^n, f^n; lam/2)` for R, M, E (the half step uses the full scheme, with limited antidiffusion);
3. enforce `E >= |M|` and the vacuum rule on `U^{n+1/2}`; fill ghosts; recover primitives at `n+1/2`;
4. `U^{n+1} = SHASTA(U^n; v^{n+1/2}, f^{n+1/2}; lam)`;
5. enforce `E >= |M|` and the vacuum rule on `U^{n+1}`.

The paper states that velocities and sources entering the transport are evaluated at the half step; it does not spell
out whether the half step itself includes the antidiffusion stage. Here it does (the brief's wording "advance U^n by dt/2").

## 4. Primitive recovery (`recover_primitives`)

Given `(R, M, E)`, `M -> |M|`: the velocity is the root of

    g(v) = v (E + p0(E - |M| v, R sqrt(1 - v^2))) - |M| = 0,   v in [0, 1)                       (Eq. 37)

with `eps = E - |M| v`, `rho = R sqrt(1 - v^2)` (Eq. 36). `g(0) = -|M| <= 0` and `g(1) = E + p0(E - |M|, 0) - |M| >= 0`
when `E >= |M|` and `p0 >= 0`, so the root is bracketed. The iteration is regula falsi with the Illinois modification on
the bracket `[0, 1]`, falling back to bisection if the interpolated point leaves the bracket. Stopping: `|g| <= 1e-15 E`,
or `|v_k - v_{k-1}| < 1e-12`, or bracket width `< 1e-12`; at most 60 iterations. Failure (no convergence, or no root in
the bracket) sets an integer flag per cell; flags are summed with `parallel_reduce`. Cells with `E < e_vac` (default 1e-12
GeV/fm^3) are vacuum: `eps = rho = v = p = 0`. Negative `R` is treated as `rho = 0` in the recovery only.

## 5. Constraint E >= |M| and vacuum (`Hydro1D::enforce`)

After each update, per cell: if `E < e_vac` then `R = M = E = 0` (vacuum, `v = 0` in the recovery); else if `|M| > E` then
`|M|` is restricted to `E` (paper, end of Sect. 3.2). The number of restricted cells is counted and reported.

## 6. Exact solutions used by the tests (`tests/exact_solutions.hpp`)

Equation of state `p = c_s^2 eps`, `c_s^2 = 1/3`. The baryon density does not enter the pressure, so `eps(x, t)` is
independent of `rho`.

### Simple rarefaction wave (also expansion into vacuum, Sect. 4.2)

For a right-moving fluid velocity (matter at the left, rarefaction head moving left) the Riemann invariant
`R+ = y + int_{eps0}^{eps} c_s d eps' / (eps' + p(eps'))` is constant (`y = artanh v`), hence

    v(eps) = tanh( -c_s / (1 + c_s^2) ln(eps/eps0) )                                              (Eq. 51)
    zeta = x/t = (v - c_s) / (1 - v c_s)                                                           (Eq. 52)

The inverse is `eps(zeta) = eps0 [ (1 - c_s)/(1 + c_s) (1 - zeta)/(1 + zeta) ]^{(1 + c_s^2)/(2 c_s)}` for
`-c_s <= zeta` (Eq. 53), `eps = eps0` for `zeta < -c_s`. For expansion into vacuum the fan extends to `zeta = 1`, vacuum beyond.
`T^00 = (eps + p) gamma^2 - p` with `v = (zeta + c_s)/(1 + zeta c_s)`.

### Shock tube: left state `(eps_L, v = 0)`, right state `(eps_R, v = 0)`, `eps_L > eps_R`

Wave pattern from left to right: rarefaction fan (head at `zeta = -c_s`), plateau `(eps*, v*)`, contact at `zeta = v*`
(pressure and velocity continuous, hence `eps` continuous, only `rho` jumps), the same plateau, shock at `zeta = V`,
right state.

* Fan: `v*` follows from the rarefaction wave, `v_r(eps*) = tanh(-c_s/(1 + c_s^2) ln(eps*/eps_L))`; the tail is at
  `zeta_t = (v* - c_s)/(1 - v* c_s)`.
* Shock into matter at rest: conservation of `T^{0x}` and `T^{xx}` across the front moving with speed V gives (Taub adiabat)

      v_s(eps*) = sqrt( (p* - p_R)(eps* - eps_R) / ((eps_R + p*)(eps* + p_R)) ),
      V(eps*)   = sqrt( (p* - p_R)(eps* + p_R) / ((eps* - eps_R)(eps_R + p*)) ).

  They satisfy the jump conditions `V (T^00* - T^00_R) = T^{0x}*` and `V T^{0x}* = T^{xx}* - p_R`; the test
  `exact shock tube solution satisfies the Rankine-Hugoniot conditions` checks the residuals of both equations.
* Matching `v_r(eps*) = v_s(eps*)` fixes `eps*` (bisection: `v_r - v_s` is positive at `eps* = eps_R` and negative at
  `eps* = eps_L`).

For `eps_L = 10`, `eps_R = 1` GeV/fm^3 the values are printed by the test (see `docs/validation.md`).

## Scaled equation of state and tabulated p0 (WP2c)

Scaled family (Steinheimer and Randrup, PRC 87, 054903, Sect. V C, Eqs. 17-19): `s_bc(eps, rho) = s_0(b eps, c rho)` gives
`T_bc = T_0(b eps, c rho)/b`, `mu_bc = (c/b) mu_0(b eps, c rho)`, `p_bc = p_0(b eps, c rho)/b`, `vs2_bc = vs2_0(b eps, c rho)`.
By the chain rule `dp_deps_bc = dp_deps_0(b eps, c rho)` and `dp_drho_bc = (c/b) dp_drho_0(b eps, c rho)`; with
`vs2 = dp_deps + rho/(eps + p) dp_drho` these reproduce `vs2_bc = vs2_0(b eps, c rho)` because `eps + p_bc = (b eps + p_0)/b`.
Implemented in `include/spf/eos/scaled.hpp` (`EosScaled<Base>`).

Tabulated `p0(eps, rho)` (`include/spf/eos/table.hpp`, format in `docs/eos_table_format.md`): with lattice coordinates `x = eps/eps_s`,
`y = rho/rho_s`, `dp_deps = (dp/dx)/eps_s`, `dp_drho = (dp/dy)/rho_s`. Below the cold curve the pressure is `P(rho) = p0(eps_min(rho), rho)`, so
`dp_deps = 0` and `dp_drho = p_y/rho_s + (p_x/eps_s) d eps_min/d rho`; `vs2` uses these with the unclamped `eps + P` in the enthalpy, as `EosColdThermal` does.

## 7. Three dimensions (`include/spf/hydro3d.hpp`, `Hydro3D`)

Source: Rischke, Bernard, Maruhn, Sect. 3 (arXiv:nucl-th/9504018): the divergence is treated by time-step splitting, "sequentially
solving d_t U + d_i (U v_i) = ... for i = x, y, z", and "it is advantageous to use permutations of the sequence (x, y, z) in each new
time step". The companion paper (arXiv:nucl-th/9504021) treats one-dimensional problems and adds nothing on the multi-dimensional sequence.

Conserved fields `U = (R, M_x, M_y, M_z, E)`, pressure `p = p0(eps, rho) + p_extra`. A sweep along direction `i` solves
`d_t U + d_i (U v_i + f) = 0` with `f = 0` for `R` and for the two momentum components `M_j`, `j != i`; `f = p` for `M_i`; `f = p v_i` for `E`.

One time step `dt = lam dx` (`lam = shasta.lambda`) is three sweeps in the order `o(n)`:

* `shasta.sweep_order = fixed`: `x, y, z` at every step;
* `shasta.sweep_order = permuted`: for step number `n` the cyclic permutation starting at `n mod 3`: `xyz`, `yzx`, `zxy`, `xyz`, ...

Each sweep along `i` (every sweep advances by the full `dt`) is the one-dimensional half-step sequence of section 3 applied to the five fields,
to the state left by the previous sweep:

1. fill the ghost layers of `U` along `i` (periodic or zero gradient, per direction, on the transverse interior);
2. recover `p`, `v_i`, `p v_i` of `U` on all cells along `i` (ghosts included) from `(R, M, E)` and `p_extra` (section 8);
3. `U^{1/2} = SHASTA(U; v_i, f; lam/2)` for the five fields (full scheme including the limited antidiffusion);
4. enforce `E >= |M|` (with `|M| = sqrt(M_x^2 + M_y^2 + M_z^2)`; the vector `M` is rescaled to length `E`) and the vacuum rule on `U^{1/2}`;
   fill ghosts of `U^{1/2}` along `i`; recover `p`, `v_i`, `p v_i` from `U^{1/2}`;
5. `U' = SHASTA(U; v_i^{1/2}, f^{1/2}; lam)` for the five fields, with `U` (not `U^{1/2}`) as the field being advanced;
6. enforce `E >= |M|` and the vacuum rule on `U'`; `U'` becomes the state `U` of the next sweep.

`p_extra` is a `Field3` supplied by the caller (all cells including ghosts); it is zero in this package. `use_p_extra = false` bypasses it.
A zero field gives bitwise identical results (adding `+0.0` does not change a double).

Initial state: `Hydro3D::set_state_fn` / `set_state` take local-rest-frame values `(eps, rho, v)` on the host, copy them to the device and compute
`R = gamma rho`, `M_i = (eps + p) gamma^2 v_i`, `E = (eps + p) gamma^2 - p` with `p = p0(eps, rho)` in a kernel (`eps <= 0` is a vacuum cell).
The conversion is a kernel because `p0` is a device function (`docs/device_rules.md`, rule 9). `Hydro1D::set_state` works the same way.

Points the papers leave open and the choices made here:

* The papers do not say how the half-step velocity and sources are obtained in several dimensions. Here each sweep is self-contained:
  its own half step along `i` uses primitives recovered from the state at the start of that sweep, so the half step of the sweep along `j` uses
  primitives of the state already updated by the sweep along `i`. This reduces to the 1D sequence of section 3 when only one direction carries gradients
  and velocity, which is tested bit for bit (T3.1).
* The papers do not say whether a split step advances by `dt` in each direction or by `dt/3`. Each sweep advances by `dt`; with `lam <= 0.4`
  each sweep is within the 1D stability limit.
* The ghost layers of the transverse directions are not filled before a sweep: a sweep reads only cells on its own line.
* For `E < e_vac` all of `R, M, E` are set to zero (vacuum cell, `v = 0` in the recovery). Rescaling `M` to length `E` preserves its direction.
* The vacuum and clamp steps act on whole cells after all five fields of a sweep are updated, so no ordering among the fields is needed.

## 8. Primitive recovery in 3D with `p_extra` (`recover_core`)

With `|M| = sqrt(M_x^2 + M_y^2 + M_z^2)`, the speed `c = |v|` is the root in `[0, 1)` of

    g(c) = c (E + p0(E - |M| c, R sqrt(1 - c^2)) + p_extra) - |M| = 0

by the regula-falsi/Illinois iteration of section 4 (at most 60 iterations, failure flag per cell, summed with `parallel_reduce`). Then
`eps = E - |M| c`, `rho = R sqrt(1 - c^2)`, `v_i = c M_i / |M|`, `p = p0(eps, rho) + p_extra`. `p_extra` is held fixed during the iteration
(it is a field evaluated from earlier primitives, "lagged"). The bracket needs `g(1) >= 0`, which holds for `p_extra >= 0` and for small negative values;
a cell where it fails is flagged.

## 9. Snapshots

`Hydro3D::write_snapshot(base, fields, config)` writes `<base>.bin` (raw little-endian float64, the requested fields concatenated, each in C order
`[ix][iy][iz]`, interior cells only) and `<base>.json` (shape, `dx_fm`, `time_fm`, `step`, field names with byte offsets, configuration echo).
Fields: `R Mx My Mz E eps rho vx vy vz p`. Host mirror copies are made only there.


## 10. Observables (`include/spf/observables.hpp`, WP6)

Sources: Steinheimer and Randrup, PRL 109, 212301 (arXiv:1209.2462), Eq. 3 and Figs. 3-4; PRC 87, 054903 (arXiv:1302.2956), Sect. IV A-C, Eqs. 13-16, Figs. 5-8.
Cell centres are at `x_i = (i + 1/2 - n/2) dx` measured from the centre of the box; `dV = dx^3`. `D` is the density selected by `obs.moment_density`:
the computational-frame baryon density `R = gamma rho` (default; then `A` is exactly the conserved baryon number) or the rest-frame `rho`.

* Moments: `A = Integral D d^3r`, `<D^N> = (1/A) Integral D^N D d^3r`, normalised moment `<D^N> / <D>^N` (N = 1 gives 1). Enhancement(t) = normalised moment(t) / normalised
  moment(t_init); the maximum over time is taken by `tools/obs_ensemble.py` and in `observables_summary.json`.
  **Prefactor note.** arXiv:1302.2956 Eq. 13 prints the prefactor ambiguously ("A = <rho^0>"). The reading `1/A` is the one for which the normalised moments
  are dimensionless and N = 1 gives exactly 1, as the paper states; it is used here. (With a prefactor `1/V` instead, the ratio `<D^N>/<D>^N` would carry a factor `(V/A)^(N-1)`... i.e. it would not be
  a pure density ratio.) The sums are evaluated on `D / mean(D)` (a pre-pass computes the mean) so that the rounding error of the N-th power stays near N machine epsilon.
* Transverse flow, Eqs. 14-16 with the weight `D` (the same density as the moments), laboratory-frame velocities, and `x`, `y` measured from the beam axis (the z axis through the centre of the box):
  `M = Integral D d^3r`, `M Q^2 = Integral (x^2 + y^2) D d^3r`, `U_perp = (1/(M Q)) Integral (v_x x + v_y y) D d^3r`. For `v = H (x, y)` this is `U_perp = H Q`, `Q^2 = <x^2 + y^2>`;
  for a Gaussian of width sigma `U_perp = sqrt(2) H sigma`.
* Phase plane: 2D histogram of baryon number (weights `R dV`) in `(rho/rho_s, eps/eps_s)`, rest-frame `rho`, `eps`; bins `[i d, (i+1) d)`, defaults 0..12 rho_s step 0.1 and 0..30 eps_s step 0.25;
  baryon number outside the range is kept in `outside` (histogram + outside = baryon number). Baryon fraction in a region: `sum(R dV over cells where mask(eps, rho)) / sum(R dV)`; the mask implemented is
  `vs2(eps, rho) < 0` of the equation of state in use (device function, rule 9). Hook: any struct with `KOKKOS_INLINE_FUNCTION bool operator()(Real eps, Real rho) const` is a mask (`compute_observables`,
  `compute_phase_plane`); coexistence and isothermal-spinodal masks of a finite-temperature table (WP2b) are not implemented.
* End of the fluid stage: first recorded time at which all cells have `eps < obs.fluid_end_eps_eps_s * eps_s` (arXiv:1302.2956, caption of Fig. 8). Resolution: `obs.every_steps * dt`.
* Clumps: cells with `D > obs.clump_rho_min_rho_s * rho_s`, connected with 6 faces (`obs.clump_connectivity` = 6) or 26 neighbours (edges and corners too); union-find on the host. The box wraps
  (a clump crossing a face is one clump) iff `grid.boundary = periodic`. Baryon number of a clump = `sum R dV`. Histogram bins: bin k holds the clumps with `round(B) = k`, k = 0..`obs.clump_hist_max` (25), plus an overflow count.

### Closed form for Gaussians (test T6.2)
For `rho_i = B_i (2 pi)^(-3/2) sigma^(-3) exp(-|r - c_i|^2 / (2 sigma^2))`, well separated, same sigma: `Integral rho_i^(N+1) d^3r = B_i^(N+1) (2 pi)^(-3N/2) sigma^(-3N) (N+1)^(-3/2)`, hence
`<rho^N> = K_N sum_i B_i^(N+1) / A`, `K_N = (2 pi)^(-3N/2) sigma^(-3N) (N+1)^(-3/2)`, and
`<rho^N>/<rho>^N = 2^(3N/2) (N+1)^(-3/2) [sum B_i^(N+1) / A] / [sum B_i^2 / A]^N`.
For identical Gaussians the bracket is 1 and the result is `2^(3N/2) / (N+1)^(3/2)`, independent of M, B and sigma (the reviewer's value; confirmed).
