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
