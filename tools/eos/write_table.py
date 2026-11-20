#!/usr/bin/env python3
"""Writer for the tabulated equation-of-state format of spinodal-fluid (docs/eos_table_format.md).

(a) ``write_table`` writes <stem>.json and <stem>.bin from arrays; reusable by any producer of tables.
(b) ``synthetic_cold_thermal`` samples the analytic development equation of state
        p0 = p_c(rho) + (Gamma_th - 1) (eps - eps_c(rho))
    from data/eos/cold_curve.dat on the project lattice; the command line writes the small synthetic table used by the tests:

        python tools/eos/write_table.py synthetic --out tests/data/synthetic_eos --dx 0.2 --dy 0.04

Requires numpy.  Units: GeV, fm; hbar c = 0.1973269804 GeV fm.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

FORMAT_VERSION = 1
FIELDS = ["p0", "T", "mu", "s", "eps_min"]
UNITS = {"p0": "GeV/fm^3", "T": "GeV", "mu": "GeV", "s": "fm^-3", "eps_min": "GeV/fm^3"}
INDEX_ORDER = ("p0, T, mu, s: C order, shape (nx, ny), element (ix, iy) at flat index ix*ny + iy (iy fastest); "
               "eps_min: shape (ny,) indexed by iy; all float64 little-endian, fields stored consecutively in the order of 'fields'")


def write_table(stem, p0, T, mu, s, eps_min, *, x0, dx, y0, dy, rho_s=0.153, m_N=0.940, description="", extra=None):
    """Write <stem>.json and <stem>.bin.

    p0, T, mu, s: arrays of shape (nx, ny) with element [ix, iy] at x = x0 + ix*dx (x = eps/eps_s), y = y0 + iy*dy (y = rho/rho_s),
    eps_s = m_N*rho_s.  Units: p0 GeV/fm^3, T GeV, mu GeV, s fm^-3.  eps_min: shape (ny,), GeV/fm^3, the lowest allowed energy density
    at each rho.  `extra` (dict) is stored under the header key 'generator'.  Returns the header dict.
    """
    p0, T, mu, s = (np.ascontiguousarray(a, dtype="<f8") for a in (p0, T, mu, s))
    eps_min = np.ascontiguousarray(eps_min, dtype="<f8")
    nx, ny = p0.shape
    for name, a in (("T", T), ("mu", mu), ("s", s)):
        if a.shape != (nx, ny):
            raise ValueError(f"{name} has shape {a.shape}, expected {(nx, ny)}")
    if eps_min.shape != (ny,):
        raise ValueError(f"eps_min has shape {eps_min.shape}, expected {(ny,)}")
    if nx < 2 or ny < 2 or dx <= 0 or dy <= 0:
        raise ValueError("need nx, ny >= 2 and dx, dy > 0")
    for a in (p0, T, mu, s, eps_min):
        if not np.all(np.isfinite(a)):
            raise ValueError("non-finite value in table")
    payload = b"".join(a.tobytes(order="C") for a in (p0, T, mu, s, eps_min))
    header = {
        "format": "spf_eos_table",
        "format_version": FORMAT_VERSION,
        "description": description,
        "byte_order": "little",
        "dtype": "float64",
        "lattice": {"rho_s_fm3": rho_s, "m_N_GeV": m_N, "x0": x0, "dx": dx, "nx": int(nx), "y0": y0, "dy": dy, "ny": int(ny)},
        "index_order": INDEX_ORDER,
        "fields": FIELDS,
        "units": UNITS,
        "bin_size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if extra:
        header["generator"] = extra
    d = os.path.dirname(stem)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(stem + ".bin", "wb") as f:
        f.write(payload)
    with open(stem + ".json", "w") as f:
        json.dump(header, f, indent=2)
        f.write("\n")
    return header


def read_table(stem):
    """Read a table back (checks size and checksum); returns (header, dict of arrays)."""
    with open(stem + ".json") as f:
        h = json.load(f)
    raw = open(stem + ".bin", "rb").read()
    L = h["lattice"]
    nx, ny = L["nx"], L["ny"]
    if len(raw) != 8 * (4 * nx * ny + ny):
        raise ValueError("size mismatch")
    if hashlib.sha256(raw).hexdigest() != h["sha256"]:
        raise ValueError("checksum mismatch")
    a = np.frombuffer(raw, dtype="<f8")
    out = {k: a[i * nx * ny:(i + 1) * nx * ny].reshape(nx, ny) for i, k in enumerate(FIELDS[:4])}
    out["eps_min"] = a[4 * nx * ny:]
    return h, out


# ---------------------------------------------------------------- synthetic development table

def read_cold_curve(path):
    """Parse data/eos/cold_curve.dat: returns (d_rho, eps_c nodes, mu_c nodes)."""
    keys, rows = {}, []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            if "=" in line:
                k, v = line[1:].split("=", 1)
                keys[k.strip()] = v.strip()
            continue
        rows.append([float(t) for t in line.split()[:3]])
    r = np.array(rows)
    return float(keys["d_rho_fm3"]), r[:, 1], r[:, 2]


def cold_hermite(rho, d_rho, eps_c, mu_c):
    """eps_c(rho), mu_c(rho), p_c(rho) with the same C1 cubic Hermite interpolation as EosColdThermal (include/spf/eos/cold_thermal.hpp)."""
    n = len(eps_c)
    rho = np.clip(np.asarray(rho, dtype=float), 0.0, (n - 1) * d_rho)
    i = np.clip((rho / d_rho).astype(int), 0, n - 2)
    t = rho / d_rho - i
    e0, e1 = eps_c[i], eps_c[i + 1]
    m0, m1 = mu_c[i] * d_rho, mu_c[i + 1] * d_rho
    c2 = 3 * (e1 - e0) - 2 * m0 - m1
    c3 = -2 * (e1 - e0) + m0 + m1
    ec = e0 + t * (m0 + t * (c2 + t * c3))
    mu = (m0 + t * (2 * c2 + 3 * c3 * t)) / d_rho
    return ec, mu, rho * mu - ec


def synthetic_functions(x, y):
    """Smooth analytic test functions for T, mu, s (x = eps/eps_s, y = rho/rho_s); NOT physical, chosen to be smooth on the lattice."""
    T = 0.1 * x / (1.0 + y)                                   # GeV
    mu = 0.94 + 0.3 * y / (1.0 + 0.1 * y) + 0.005 * x         # GeV
    s = 0.5 * x * (1.0 + 0.1 * y) / (1.0 + 0.01 * x) + 0.2 * y  # fm^-3
    return T, mu, s


SYNTH_DESCRIPTION = (
    "SYNTHETIC TABLE FOR READER TESTS ONLY - not a physical equation of state. p0 is sampled from the analytic development EoS "
    "p0 = p_c(rho) + (Gamma_th - 1)(eps - eps_c(rho)) with Gamma_th = {gamma:.12g}, p_c = rho mu_c - eps_c, from data/eos/cold_curve.dat "
    "(C1 cubic Hermite interpolation of the cold curve); the formula is continued smoothly below the cold curve (no clamp). "
    "T, mu, s are smooth analytic test functions with no thermodynamic meaning: T = 0.1 GeV x/(1+y); mu = 0.94 GeV + 0.3 GeV y/(1+0.1 y) + "
    "0.005 GeV x; s = 0.5 x (1+0.1 y)/(1+0.01 x) + 0.2 y fm^-3, with x = eps/eps_s, y = rho/rho_s. eps_min(rho) = eps_c(rho)."
)


def synthetic_cold_thermal(cold_curve_path, *, x0=0.0, dx=0.05, nx=801, y0=0.0, dy=0.02, ny=801, rho_s=0.153, m_N=0.940, gamma_th=4.0 / 3.0):
    """Arrays (p0, T, mu, s, eps_min) of the synthetic table on the given lattice (defaults: the plan's lattice x in [0,40], y in [0,16])."""
    d_rho, ec_tab, mc_tab = read_cold_curve(cold_curve_path)
    eps_s = m_N * rho_s
    x = x0 + dx * np.arange(nx)[:, None]
    y = y0 + dy * np.arange(ny)[None, :]
    ec, muc, pc = cold_hermite(y * rho_s, d_rho, ec_tab, mc_tab)
    p0 = pc + (gamma_th - 1.0) * (x * eps_s - ec)
    T, mu, s = synthetic_functions(x + 0 * y, y + 0 * x)
    return p0, T, mu, s, ec[0, :].copy()


def write_synthetic(stem, cold_curve_path, *, x_max=40.0, y_max=16.0, dx=0.05, dy=0.02, gamma_th=4.0 / 3.0, rho_s=0.153, m_N=0.940):
    nx, ny = int(round(x_max / dx)) + 1, int(round(y_max / dy)) + 1
    p0, T, mu, s, emin = synthetic_cold_thermal(cold_curve_path, dx=dx, nx=nx, dy=dy, ny=ny, rho_s=rho_s, m_N=m_N, gamma_th=gamma_th)
    return write_table(stem, p0, T, mu, s, emin, x0=0.0, dx=dx, y0=0.0, dy=dy, rho_s=rho_s, m_N=m_N,
                       description=SYNTH_DESCRIPTION.format(gamma=gamma_th),
                       extra={"tool": "tools/eos/write_table.py synthetic", "gamma_th": gamma_th, "cold_curve": "data/eos/cold_curve.dat"})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic", help="write the synthetic cold_thermal test table")
    s.add_argument("--out", required=True, help="output stem (writes STEM.json and STEM.bin)")
    s.add_argument("--cold-curve", default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "eos", "cold_curve.dat"))
    s.add_argument("--x-max", type=float, default=40.0)
    s.add_argument("--y-max", type=float, default=16.0)
    s.add_argument("--dx", type=float, default=0.05)
    s.add_argument("--dy", type=float, default=0.02)
    s.add_argument("--gamma-th", type=float, default=4.0 / 3.0)
    a = ap.parse_args(argv)
    h = write_synthetic(a.out, a.cold_curve, x_max=a.x_max, y_max=a.y_max, dx=a.dx, dy=a.dy, gamma_th=a.gamma_th)
    L = h["lattice"]
    print(f"wrote {a.out}.json/.bin: nx={L['nx']} ny={L['ny']} bytes={h['bin_size_bytes']} sha256={h['sha256']}")


if __name__ == "__main__":
    sys.exit(main())
