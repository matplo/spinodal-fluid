"""Reader of the switching-surface files of spf_collide (docs/particlisation_format.md) and helpers for the census.  numpy only.

  recs, side = read_surface("<dir>/event_000/event_000")      # base path without .surface.bin / .surface.json
  recs[:, COL["eps"]] ...                                      # (n, 16) float64
"""
import json
import os

import numpy as np

COLUMNS = ["t", "x", "y", "z", "dsig_0", "dsig_1", "dsig_2", "dsig_3", "u0", "u1", "u2", "u3", "eps", "rho", "T", "mu_B"]
COL = {c: i for i, c in enumerate(COLUMNS)}


def read_surface(base, mmap=False):
    """Records (n, 16) and the sidecar dict.  The number of records must equal n_elements of the sidecar."""
    with open(base + ".surface.json") as f:
        side = json.load(f)
    if side.get("format_version") != 1:
        raise ValueError("unsupported format_version %r" % side.get("format_version"))
    path = base + ".surface.bin"
    if mmap:
        recs = np.memmap(path, dtype="<f8", mode="r").reshape(-1, 16)
    else:
        recs = np.fromfile(path, dtype="<f8").reshape(-1, 16)
    if recs.shape[0] != side["n_elements"]:
        raise ValueError("%s: %d records, sidecar says %d" % (path, recs.shape[0], side["n_elements"]))
    return recs, side


class TableP0:
    """Bilinear interpolation of p0(eps, rho) of an EoS table (docs/eos_table_format.md), clamped to the lattice and to the cold curve as in
    the C++ reader with eos.table_interpolation = bilinear.  Used only to weight the census histograms by the energy flux."""

    def __init__(self, stem):
        with open(stem + ".json") as f:
            h = json.load(f)
        L = h["lattice"]
        self.nx, self.ny = int(L["nx"]), int(L["ny"])
        self.x0, self.dx, self.y0, self.dy = L["x0"], L["dx"], L["y0"], L["dy"]
        self.rho_s = L["rho_s_fm3"]
        self.eps_s = L["m_N_GeV"] * L["rho_s_fm3"]
        v = np.fromfile(stem + ".bin", dtype="<f8")
        n = self.nx * self.ny
        self.p = v[:n].reshape(self.nx, self.ny)
        self.emin = v[4 * n:4 * n + self.ny]
        self.sha256 = h.get("sha256", "")

    def p0(self, eps, rho):
        y = np.clip((rho / self.rho_s - self.y0) / self.dy, 0.0, self.ny - 1.0)
        j = np.minimum(np.floor(y).astype(int), self.ny - 2)
        fy = y - j
        emin = (1 - fy) * self.emin[j] + fy * self.emin[j + 1]
        e = np.maximum(eps, emin)
        x = np.clip((e / self.eps_s - self.x0) / self.dx, 0.0, self.nx - 1.0)
        i = np.minimum(np.floor(x).astype(int), self.nx - 2)
        fx = x - i
        P = self.p
        return ((1 - fx) * (1 - fy) * P[i, j] + fx * (1 - fy) * P[i + 1, j] + (1 - fx) * fy * P[i, j + 1] + fx * fy * P[i + 1, j + 1])


def fluxes(recs, p0):
    """Energy flux T^{0 mu} dSigma_mu [GeV] and baryon flux N^mu dSigma_mu per element, given p0 per element."""
    u = recs[:, 8:12]
    n = recs[:, 4:8]
    ud = np.einsum("ij,ij->i", u, n)
    e = recs[:, COL["eps"]]
    rho = recs[:, COL["rho"]]
    dE = (e + p0) * u[:, 0] * ud - p0 * n[:, 0]
    dB = rho * ud
    return dE, dB


def file_size_mb(base):
    return os.path.getsize(base + ".surface.bin") / 1e6
