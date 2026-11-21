# Equation-of-state reference prototype

`eos_reference_prototype.py` implements the pure phases of the two-phase equation of state of arXiv:1007.1448, App. A
(pion + nucleon gas with a density-dependent interaction; bag-model plasma), their phase crossing, the T = 0 quintic
join, the interface tension and the collision kinematics. Units: MeV, fm. Requires numpy and scipy.

`python eos_reference_prototype.py` rewrites `eos_reference_values.json`, the source of the reference numbers used
in the plan and in the tests. The temperature dependence of the spline matching densities is not part of the
prototype (it is not specified in the papers; work package WP2b).

## Cold curve for the `cold_thermal` equation of state

`make_cold_curve.py` imports the prototype, builds the T = 0 quintic spline with matching densities 4.2 rho_s and 10.5 rho_s
and writes `data/eos/cold_curve.dat` (GeV, fm):

```
python tools/eos/make_cold_curve.py [output_path]      # default: data/eos/cold_curve.dat; needs numpy, scipy
```

The header records rho_s, the matching densities, the coexistence densities rho_1 and rho_2, mu_coex, p_coex, the grid step,
the number of rows and the SHA-256 of `eos_reference_values.json`. Rows are `rho [fm^-3]`, `eps_c [GeV/fm^3]`,
`mu_c = d eps_c/d rho [GeV]` on a uniform grid from 0 to 16 rho_s in steps of 0.005 rho_s. At rho = 0 the row is
eps_c = 0, mu_c = 0.940 GeV (the analytic limit of the hadronic branch). The script asserts that the spline's
coexistence and spinodal numbers agree with the JSON before writing. The unit conversion MeV -> GeV happens only in this tool.
The committed table is the generated file; regenerate it after any change of the prototype or of the reference values.

## Writing a tabulated equation of state

`write_table.py` writes the table format described in `docs/eos_table_format.md` (header `<stem>.json` + little-endian binary `<stem>.bin`
with the SHA-256 of the binary in the header). Requires numpy.

```python
import sys; sys.path.insert(0, "tools/eos")
from write_table import write_table
# p0, T, mu, s: float arrays of shape (nx, ny), element [ix, iy] at x = x0 + ix*dx (x = eps/eps_s), y = y0 + iy*dy (y = rho/rho_s)
# eps_min: array of shape (ny,) with the cold curve eps_c(rho_j) in GeV/fm^3
write_table("out/my_eos", p0, T, mu, s, eps_min, x0=0.0, dx=0.05, y0=0.0, dy=0.02,
            rho_s=0.153, m_N=0.940, description="what this table is")
```

Units: p0 and eps_min in GeV/fm^3, T and mu in GeV, s in fm^-3. `read_table(stem)` reads a table back and verifies size and checksum.

The synthetic table used by the tests (analytic `cold_thermal` pressure; T, mu, s are smooth test functions without physical meaning) is
written with

```
python tools/eos/write_table.py synthetic --out tests/data/synthetic_eos --dx 0.2 --dy 0.04       # committed coarse table
python tools/eos/write_table.py synthetic --out /tmp/synthetic_default                              # plan lattice 801 x 801 (20 MB)
```
