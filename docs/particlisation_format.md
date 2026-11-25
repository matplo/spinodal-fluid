# Switching (particlisation) surface: file format and construction (WP10a)

The fluid stage of a collision event ends on a switching surface; a sampler (WP10b) converts each surface element into hadrons with the
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

## 3. Configuration keys (`config/defaults.json`, block `particlisation`)

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
