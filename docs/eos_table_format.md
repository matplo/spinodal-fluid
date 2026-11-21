# Tabulated equation-of-state format (version 1)

A table is a pair of files with a common stem `<name>`:

* `<name>.json` - header (UTF-8 JSON),
* `<name>.bin` - raw little-endian IEEE-754 binary64 values, no padding, no header.

Reader: `include/spf/eos/table.hpp` (`spf::EosTableData::load`). Writer: `tools/eos/write_table.py` (`write_table`).
`eos.table_path` is the stem or the path of the `.json` file; `eos.model = table_two_phase | table_maxwell` selects the reader (both
variants use the same file format; they differ only in the table that is given).

## Lattice

Units are GeV and fm with c = 1. The table is defined on a uniform lattice of dimensionless variables

    x = eps / eps_s,   y = rho / rho_s,   eps_s = m_N * rho_s

with `rho_s` and `m_N` stored in the header (project defaults rho_s = 0.153 fm^-3, m_N = 0.940 GeV, hence eps_s = 0.14382 GeV/fm^3).
Node (ix, iy) sits at `x = x0 + ix*dx` (0 <= ix < nx) and `y = y0 + iy*dy` (0 <= iy < ny). The plan's default lattice is
x in [0, 40] with dx = 0.05 (nx = 801) and y in [0, 16] with dy = 0.02 (ny = 801).

## Data layout of the `.bin` file

Five arrays of float64, stored consecutively in this order (header key `fields`):

| # | name | shape | unit | content |
|---|---|---|---|---|
| 1 | `p0` | (nx, ny) | GeV/fm^3 | pressure of uniform matter |
| 2 | `T` | (nx, ny) | GeV | temperature |
| 3 | `mu` | (nx, ny) | GeV | baryon chemical potential |
| 4 | `s` | (nx, ny) | fm^-3 | entropy density |
| 5 | `eps_min` | (ny,) | GeV/fm^3 | lowest allowed energy density at each rho_j: the cold (T = 0) curve eps_c(rho_j) |

**Index order of arrays 1-4:** C order (row-major) with shape (nx, ny): element (ix, iy) is at flat index `ix*ny + iy`, i.e. `iy` runs
fastest. In numpy, `a.reshape(nx, ny)[ix, iy]`; `a.astype('<f8').tobytes()` writes it. `eps_min` is indexed by `iy`.
File size: `8 * (4*nx*ny + ny)` bytes.

Values at nodes with `eps < eps_min(rho)` are never returned by the reader (it clamps to the cold curve) but they take part in the
interpolation of the cells that straddle the curve, so the producer should continue the fields smoothly below the cold curve (any finite values
are legal). All values must be finite.

## Header (`<name>.json`)

Required keys (a missing or wrongly typed key is an error naming the key):

| key | content |
|---|---|
| `format_version` | integer, `1` |
| `description` | free text: what the table is, how it was made |
| `byte_order` | `"little"` |
| `dtype` | `"float64"` |
| `lattice` | object with `rho_s_fm3`, `m_N_GeV`, `x0`, `dx`, `nx`, `y0`, `dy`, `ny` (nx, ny >= 2, dx, dy > 0) |
| `fields` | `["p0", "T", "mu", "s", "eps_min"]`, exactly in this order |
| `units` | object `{"p0": "GeV/fm^3", "T": "GeV", "mu": "GeV", "s": "fm^-3", "eps_min": "GeV/fm^3"}` |
| `sha256` | 64 hexadecimal characters: SHA-256 of the whole `.bin` file |

Additional keys are ignored by the reader (the writer adds `format`, `index_order`, `bin_size_bytes` and an optional `generator` object).

## Verification on load (host side)

The loader checks, in this order: the header parses and has every required key with the required value or type; the lattice is sane; the size
of `<name>.bin` equals `8*(4*nx*ny + ny)` bytes; the SHA-256 of the file equals `sha256`; all values are finite. Any failure throws
`spf::EosTableError` with a message that names the file and the problem. The data are then copied into `Kokkos::View`s in the default memory
space (device memory for the CUDA backend).

## Semantics of the reader

See the header comment of `table.hpp`; in short: bilinear (default) or bicubic Hermite (centred-difference node derivatives, equivalent to
tensor-product Catmull-Rom) interpolation, selected by `eos.table_interpolation`; for `eps < eps_min(rho)` the fields are evaluated at
`eps_min(rho)` and `dp_deps = 0`; outside the lattice the arguments are clamped (`eos.extrapolate = false`, with `EosTable::check_range`
available to the driver) or linearly extrapolated from the edge cells (`eos.extrapolate = true`).

## Synthetic test table

`tests/data/synthetic_eos.{json,bin}` (lattice x in [0, 40], dx = 0.2, nx = 201; y in [0, 16], dy = 0.04, ny = 401; 2.58 MB) is made by

    python tools/eos/write_table.py synthetic --out tests/data/synthetic_eos --dx 0.2 --dy 0.04

It is for reader tests only and is not a physical equation of state (stated in its `description`).
