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

Floor (`gradient.pex_floor_kappa = kappa`, default 0.5; decision S17). `g(1) = E - |M| + p0(E - |M|, 0) + p_extra`. If `g(1) < 0` and `kappa >= 0`,
the recovery of that cell uses

    p_extra -> -kappa (E - |M| + p0(E - |M|, 0)),

so that `g(1) = (1 - kappa)(E - |M| + p0(E - |M|, 0)) >= 0` and a root exists for `0 <= kappa < 1`; the returned pressure contains the floored value.
Such cells lie at the `E = |M|` limiter surface of dilute fringes, where the Gaussian tail of the density makes `p_grad = -a^2 (eps_s/rho_s^2) rho Lap(rho)`
negative. Conservation is unaffected (only the pressure of those cells changes); the count is reported as `n_pex_floor_total`.

## 9. Snapshots

`Hydro3D::write_snapshot(base, fields, config)` writes `<base>.bin` (raw little-endian float64, the requested fields concatenated, each in C order
`[ix][iy][iz]`, interior cells only) and `<base>.json` (shape, `dx_fm`, `time_fm`, `step`, field names with byte offsets, configuration echo).
Fields: `R Mx My Mz E eps rho vx vy vz p`. Host mirror copies are made only there.

## 10. Finite-range (gradient) pressure and linear analysis (`include/spf/gradient.hpp`, WP4)

Source: Steinheimer and Randrup, PRL 109, 212301 (2012), arXiv:1209.2462, Eq. 1; PRC 87, 054903 (2013), arXiv:1302.2956, Sect. III, Eqs. 4, 9-11.
The pressure of the fluid is

    p = p0(eps, rho) + p_grad,    p_grad = - a^2 (eps_s / rho_s^2) rho Lap(rho),

`rho` the net baryon density in the local rest frame, `eps_s = m_N rho_s`, so that the prefactor is `C = a^2 eps_s / rho_s^2 = a^2 m_N / rho_s`.
`p` replaces the pressure in the conserved-field definitions and in the fluxes (`gradient.in_primitive_recovery = lagged`, below). The Laplacian
is the 7-point second-order stencil `(sum of the six neighbours - 6 rho)/dx^2`, or with `gradient.stencil = fourth_order` the 13-point stencil
`sum over axes of (-rho_{+2} + 16 rho_{+1} - 30 rho + 16 rho_{-1} - rho_{-2}) / (12 dx^2)`. Ghost width: 1 cell (second order) or 2 cells (fourth order);
the grid carries `kShastaGhost = 3` (`static_assert` in `gradient.hpp`). Truncation errors on a harmonic density: `(k dx)^2/12` and `(k dx)^4/90`
(test T4.0: 8.2e-3 and 1.07e-4 at `k dx = 0.314`).

### Where p_grad is refreshed (`Hydro3D::sweep`)

The recovery of section 8 holds `p_extra` fixed. The solver writes `p_grad` into `p_extra` (interior and the ghost layers along the sweep direction) from
the field `rho_rf`, the rest-frame density written by the most recent recovery inside a sweep (`Recover3K` stores the recovered `rho`; `update_primitives()`
does not, so diagnostics do not change the solution). Sequence of a sweep along `i`, with the gradient term on:

1. fill the ghost layers of `U` along `i`;
2. **refresh `p_grad`** from `rho_rf` of the previous recovery (the half-step state of the previous sweep; at the very first sweep the initial density given to `set_state_fn`,
   or a recovery of `U` made by `init_rho_rf()`); this involves ghost-filling `rho_rf` in all three directions (the Laplacian needs transverse neighbours) and
   filling the ghost layers of `p_extra` along `i`;
3. recover `p, v_i, p v_i` of `U` (this writes `rho_rf` from `U`);
4. half step, clamp, ghost fill of `U^{1/2}`;
5. **refresh `p_grad`** from `rho_rf` of step 3 (density of `U`);
6. recover from `U^{1/2}` (this writes `rho_rf` from `U^{1/2}`); full step.

So `p_grad` is lagged by one recovery at both refresh points. The alternative `gradient.half_step_refresh = midpoint` adds, after step 6, a second refresh from the
density of `U^{1/2}` and a second recovery of `U^{1/2}`, so that the full-step force is evaluated at the half-step density (one extra recovery per sweep;
the recovery-failure counter then counts both recoveries). Default `lagged`.

### Two couplings (`gradient.in_primitive_recovery`)

* `lagged` (default): `p0 + p_grad` is the pressure in the recovery (`E = (eps + p) gamma^2 - p`, `M = (eps + p) gamma^2 v` with the total `p`) and in the fluxes
  (`f = p` for `M_i`, `f = p v_i` for `E`). This is consistent with the conserved-field definitions: `T^{mu nu}` has the ideal-fluid form with the total pressure, so
  `(R, M, E)` and `(eps, rho, v)` are related by one set of formulae in the initial state, the recovery and the flux. (At rest `E = eps`, so `set_state` is unaffected by
  the missing initial `p_grad`; for a moving initial state the conserved fields are built with `p0` only, as the hook starts at zero.)
* `force_only`: the recovery uses `p0` alone, `p_grad` enters only the source terms `f = p0 + p_grad` (`M_i`) and `f = (p0 + p_grad) v_i` (`E`). The conserved fields are
  then those of the pressure `p0`, and `p_grad` acts as an external force; the totals `R, M, E` are conserved in either variant because both are in flux form.

The two variants differ by 6e-8 (relative) in the growth rates of T4.1 (the background is uniform: `p_grad` is a first-order quantity).

### Linear analysis around uniform matter (rest frame, ideal fluid, one dimension)

`d_t rho = - rho d_x v`, `d_t eps = - h d_x v`, `h d_t v = - d_x p`, `h = eps + p0`, with `p = p0(eps, rho) - C rho d_x^2 rho` (the correction to `h` from `p_grad` is of second order).
Linearising: `d_t delta_rho = - rho d_x v`, `h d_t v = - d_x delta_p`, `delta_p = (dp0/deps) delta_eps + (dp0/drho) delta_rho - C rho0 d_x^2 delta_rho`.
For an isentropic perturbation `delta_eps = (h/rho) delta_rho` (the combination `delta_eps - (h/rho) delta_rho` is the entropy-per-baryon mode, which does not propagate):
`delta_p0 = (h/rho) (dp0/deps + (rho/h) dp0/drho) delta_rho = (h/rho) v_s^2 delta_rho`. Hence

    d_t^2 delta_rho = v_s^2 d_x^2 delta_rho - (C rho0^2 / h) d_x^4 delta_rho,
    omega^2 = v_s^2 k^2 + (C rho0^2 / h) k^4 = v_s^2 k^2 + a^2 (eps_s / h) (rho0 / rho_s)^2 k^4.

This is Eq. 9 of arXiv:1302.2956 (read in the published version, PRC 87, 054903: `gamma_k = [|v_s|^2 k^2 - a^2 (eps_s/h)(rho^2/rho_s^2) k^4]^{1/2}`); the pressure is its Eq. 4,
the tension its Eq. 8, and the harmonic initial state and the `cosh` form are its Eqs. 10 and 11. The matching gradient energy is not part of this package (plan WP9, `korteweg_full`).
For `v_s^2 < 0` the modes with `omega^2 < 0` grow as `exp(gamma t)`, `gamma = sqrt(|v_s|^2 k^2 - B k^4)`,
`B = a^2 (eps_s/h)(rho0/rho_s)^2`, with the maximum at `k_max^2 = |v_s|^2 / (2B)`, `gamma_max = |v_s|^2 / (2 sqrt(B))`, and the cut-off `k_c^2 = |v_s|^2 / B`. With initial `v = 0`
the solution is `delta_rho(t) = delta_rho(0) cosh(gamma t)` (unstable) or `cos(omega t)` (stable); these are the fit functions of `spf_growth`.
For `v_s^2 = -0.05428`, `h = 1.9884 GeV/fm^3`, `rho0 = 5.8 rho_s`: `B = 0.002650 fm^2` at `a = 0.033 fm`, which gives `gamma_max = 0.527` at `k = 3.20 fm^-1` and `k_c = 4.53 fm^-1` for `a = 0.033 fm`.

On the grid the second-order Laplacian replaces `k^2` by `(2/dx)^2 sin^2(k dx/2)`, so that the discrete cut-off along an axis is `(2/dx) asin(k_c dx / 2)`, 3.6 % above `k_c`
at `dx = 0.2 fm` and `a = 0.033 fm`.

The paper states that the numerical `gamma_k` for `a = 0` falls below `|v_s| k` because of the SHASTA scheme, that halving `dx` doubles `gamma_k` (`gamma_{2k}[dx/2] = 2 gamma_k[dx]`),
and that at `dx = 0.2 fm` the gradient-free rates are reliable only up to about 4 fm^-1; tests T4.3 and T4.4 check the same statements.

### Interface tension (arXiv:1302.2956, Eq. 8)

For the cold curve at `T = 0` (`f = eps_c`), `sigma_0 = a Integral sqrt(2 eps_s Delta f(c)) dc` with `c = rho / rho_s` between the coexistence densities `rho_1, rho_2` and
`Delta f = eps_c(rho) - [eps_c(rho_1) + mu_coex (rho - rho_1)]` (the Maxwell line). Test T4.7 evaluates it with the Hermite interpolant of `data/eos/cold_curve.dat`
(Simpson rule, 20001 points).

## 11. Observables (`include/spf/observables.hpp`, WP6)

Sources: Steinheimer and Randrup, PRL 109, 212301 (arXiv:1209.2462), Eq. 3 and Figs. 3-4; PRC 87, 054903 (arXiv:1302.2956), Sect. IV A-C, Eqs. 13-16, Figs. 5-8.
Cell centres are at `x_i = (i + 1/2 - n/2) dx` measured from the centre of the box; `dV = dx^3`. `D` is the density selected by `obs.moment_density`:
the computational-frame baryon density `R = gamma rho` (default; then `A` is exactly the conserved baryon number) or the rest-frame `rho`.

* Moments: `A = Integral D d^3r`, `<D^N> = (1/A) Integral D^N D d^3r`, normalised moment `<D^N> / <D>^N` (N = 1 gives 1). Enhancement(t) = normalised moment(t) / normalised
  moment(t_init); the maximum over time is taken by `tools/obs_ensemble.py` and in `observables_summary.json`.
  **Prefactor note.** arXiv:1302.2956 Eq. 13 prints the prefactor ambiguously ("A = <rho^0>"). The reading `1/A` is the one for which the normalised moments
  are dimensionless and N = 1 gives exactly 1, as the paper states; it is used here. A prefactor `1/V` would make the ratio carry a factor `(V/A)^(N-1)`, so it would no longer be a pure density ratio. The sums are evaluated on `D / mean(D)` (a pre-pass computes the mean) so that the rounding error of the N-th power stays near N machine epsilon.
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

## 12. Linear response (`include/spf/linear/`, `src/linear/`, `apps/spf_linear.cpp`, WP8)

Source: Randrup, PRC 82, 034902 (arXiv:1007.1448), Sects. III-V, Eqs. 33-38, 42-49. Host-side code (no Kokkos kernels). Units GeV, fm, c = 1.
Time dependence `exp(-i w t)`: `Im w > 0` is growth, `gamma_k = max Im w` over the roots.

### Dispersion relation and its cubic

Harmonic disturbance of a uniform background `(rho0, T0)` with the finite-range pressure `p = p0 - C rho0 Lap rho` (Eq. 30), `C = a^2 eps_s / rho_s^2 = a^2 m_N / rho_s`
(the same strength as `GradientParams::coef()`), effective viscosity `xi = 4/3 eta + zeta` with `zeta` neglected, heat current `q = -kappa (d_x T + T0 d_t v)`:

    w^2 = vT2 k^2 + B k^4 - i nu w + (vs2 - vT2) k^2 / (1 + i q / w)                     (Eq. 33)
    B = C rho0^2 / h0 [fm^2],   nu = lam_visc k^2 = (xi/h0) k^2,   q = lam_heat k^2 = (kappa/c_v) k^2.

Using `1/(1 + i q/w) = w/(w + i q)` and multiplying by `(w + i q)`:

    (w + i q)(w^2 + i nu w - A) - (vs2 - vT2) k^2 w = 0,    A = vT2 k^2 + B k^4,
    w^3 + i (nu + q) w^2 - (S + q nu) w - i q A = 0,        S = vs2 k^2 + B k^4,       (since A + (vs2 - vT2) k^2 = S)

a cubic with the three eigenmodes of the paper (one purely imaginary heat mode and a pair). For `q = 0` the cubic contains the spurious root `w = 0`
(the multiplication by `w + i q`); the code then solves the quadratic `w^2 + i nu w - S = 0` directly, with the growing root written as `-2 S/(nu + s)`,
`s = sqrt(nu^2 - 4 S)` (no cancellation), so that the ideal limit `gamma^2 = -S` holds to rounding. For `q > 0` the cubic is solved in closed form (Cardano in complex arithmetic,
the larger of the two cube-root radicands), each root polished with Newton steps on the original polynomial. Checks: residual, Vieta relations and the uncleared Eq. 33 (T8.0).
Limits used in the tests: `nu = q = 0`: `w^2 = S`; small `nu`: `w = w_ideal - i nu/2`; `k -> 0` with `q > 0`: the heat mode `w -> -i q vT2/vs2 = -i lam_heat k^2 vT2/vs2`,
so `gamma > 0` iff `vT2 < 0` (the instability boundary moves from `vs2 = 0` to `vT2 = 0`).

### Transport coefficients (Eqs. 34-38)

`c0 = (1/4 pi) [(g_g + 3/2 g_q) zeta(3)/pi^2]^(1/3)` with `g_g = 16`, `g_q = 12`: 0.127788 (paper 0.12779). `eta = eta0 c0 d h`, `kappa = kappa0 c0 d c_v`, `d = n^(-1/3)`;
`lam_visc = (4/3) eta0 c0 d`, `lam_heat = kappa0 c0 d` (both in fm; `c_v` drops out of `lam_heat`). `n` is the particle density: pions + nucleons + antinucleons in the hadron gas,
gluons + quarks + antiquarks in the plasma. Across the spline region `linear.particle_density_in_spline_region` (decision S12) selects `linear_in_rho` (weight `(rho - rho_lo)/(rho_hi - rho_lo)`
between the hadron-side and plasma-side densities at the same `T`), `hadron_side` or `plasma_side` (`particle_density_spline`).

### Input interface

`ThermoPoint {rho, T, h, vT2, vs2, cv, n}`: any equation of state can fill it (`vT2 = (rho/h)(dp/drho)_T`, `vs2 = vT2 + (T/(h c_v)) ((dp/dT)_rho)^2`, Eqs. 5-6;
a provider from the finite-temperature table is not available yet and belongs to the package that builds the table). The analytic provider used by the tests is `ToyVdw`:
a van der Waals fluid with constant specific heat per particle, `p = rho T/(1 - b rho) - a rho^2`, `eps = m rho + cv0 rho T - a rho^2`, `n = rho`, `c_v = cv0 rho`,
`(dp/dT)_rho = rho/(1 - b rho)`, critical point `rho_c = 1/(3b)`, `T_c = 8a/(27 b)`. It has `vT2 < 0` inside the isothermal spinodal and `vs2 < 0` in a narrower core below `T_c`; it is not the nuclear EoS.

### Amplification along a trajectory (Eqs. 45-49)

`Gamma_k(t) = Integral gamma_k dt` (trapezoid rule on the trajectory grid, Eq. 47); `G_k = exp(Integral over gamma_k > 0 of gamma_k dt)` (Eq. 49), exact for piecewise-linear `gamma_k(t)`
(a zero crossing inside a step is handled analytically). The Lalime equation `d sigma_k^2/dt = 2 gamma_k sigma_k^2 + 2 D_k` (Eq. 45 with `k = k'`, Eq. 46) is integrated step by step with `gamma_k` and `D_k`
replaced by their step averages and the linear equation solved exactly, so constant `gamma_k, D_k` reproduce Eq. 48 to rounding.
**Choice for `D_k` (default `einstein_hold_diffusion`):** for `gamma_k < 0` the paper's Einstein relation (text below Eq. 48: `sigma_k^2 -> -D_k/gamma_k = sigma_eq^2`) gives `D_k = -gamma_k sigma_eq^2`
(`sigma_eq^2` = `linear.sigma_eq2`, a constant per mode; the paper does not give its value, it would follow from the thermal density fluctuation at the local `T`);
for `gamma_k >= 0` the paper only says that the coupling "continually produces additional fluctuations that are also amplified" (Sect. V B), so `D_k` is held at the value of the last stable time
(zero if the trajectory starts unstable). A user-supplied `D_k(i, t, gamma)` replaces this.

### Fig. 2 numbers
Ideal case (`nu = q = 0`, `vT2 = vs2 < 0`): `gamma^2 = |vs2| k^2 - B k^4` is maximal at `k0^2 = |vs2|/(2B)` with `gamma_max = |vs2|/(2 sqrt(B))`, and vanishes at `k_cut = sqrt(2) k0`.
From `t0 = 1.38 fm/c`, `k0 = 2.0 fm^-1`: `B = (1/t0)^2/k0^4 = 0.03282 fm^2`, `|vs2| = 2 B k0^2 = 0.2626`, `|vs| = 0.5124`.
