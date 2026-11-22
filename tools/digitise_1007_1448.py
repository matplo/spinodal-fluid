#!/usr/bin/env python3
"""Digitise curves of arXiv:1007.1448 (Randrup, PRC 82, 034902) from the vector drawing operators of the PDF.

Usage:  python tools/digitise_1007_1448.py --pdf <1007.1448.pdf> [--out tools/data] [--figures docs/figures] [--catalogue]

For every panel the script
  1. finds the axis frame (the black stroked rectangle) in the page drawings,
  2. finds the tick marks (short black strokes attached to the bottom and left frame edges) and the numeric tick labels
     (text spans) and fits  value = a + b * (page coordinate)  separately for each axis (least squares over all labelled
     ticks, residuals reported),
  3. converts the vertices of the stroked curve paths and the centres of the marker paths to data units,
  4. writes tools/data/<name>.csv with a header (figure, curve, mapping, calibration residuals),
  5. draws the digitised points over a rasterised crop of the page (docs/figures/digitise_<panel>.png) for a visual check.
Nothing is read from a raster image; the raster is only the background of the check plot.
Figures covered: 2, 3 (both panels), 4, 5, 6, 7. Also derives trajectory CSVs for spf_linear from the Fig. 5 markers.  Curve names are assigned from the legends (colour, dash pattern, marker).
"""
import argparse
import csv
import math
import os
import re
import sys

import numpy as np
import pymupdf

BLACK = (0.0, 0.0, 0.0)


def is_black(c):
    return c is not None and max(c) < 0.05


# ---------------------------------------------------------------------------------------------------------------- geometry
def find_frames(page, min_w=100, min_h=60):
    """Black stroked 'quad' items (the box around a panel)."""
    frames = []
    for dr in page.get_drawings():
        for it in dr["items"]:
            if it[0] == "qu" and is_black(dr["color"]):
                r = it[1].rect
                if r.width > min_w and r.height > min_h:
                    frames.append(r)
    return frames


def segments(page):
    out = []
    for dr in page.get_drawings():
        for it in dr["items"]:
            if it[0] == "l":
                out.append((it[1].x, it[1].y, it[2].x, it[2].y, dr["color"], dr["width"]))
    return out


def parse_num(s):
    s = s.strip().replace("\u2212", "-").replace("\u2013", "-")
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
        return float(s)
    return None


def calibrate(page, frame, label_tol=3.2):
    """Linear maps for x (bottom axis) and y (left axis) from labelled ticks. Returns dict."""
    x0, y0, x1, y1 = frame.x0, frame.y0, frame.x1, frame.y1
    segs = [s for s in segments(page) if is_black(s[4])]
    xt, yt = [], []
    for ax, ay, bx, by, _, _ in segs:
        L = math.hypot(bx - ax, by - ay)
        if not (0.8 < L < 9):
            continue
        # bottom ticks: vertical, one end on the bottom edge (y1), inside the frame
        if abs(ax - bx) < 0.05 and min(abs(ay - y1), abs(by - y1)) < 0.3 and x0 - 0.3 <= ax <= x1 + 0.3:
            xt.append((ax, L))
        # left ticks: horizontal, one end on the left edge
        if abs(ay - by) < 0.05 and min(abs(ax - x0), abs(bx - x0)) < 0.3 and y0 - 0.3 <= ay <= y1 + 0.3:
            yt.append((ay, L))
    spans = []
    for b in page.get_text("dict", clip=pymupdf.Rect(x0 - 40, y0 - 10, x1 + 10, y1 + 24))["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                v = parse_num(s["text"])
                if v is not None:
                    spans.append((v, pymupdf.Rect(s["bbox"])))
    # major ticks = the longer class
    def major(t):
        if not t:
            return []
        Lmax = max(L for _, L in t)
        return sorted({round(p, 3) for p, L in t if L > 0.75 * Lmax})
    xm, ym = major(xt), major(yt)
    xs, xv, ys, yv = [], [], [], []
    for p in xm:
        c = [(abs((r.x0 + r.x1) / 2 - p), v) for v, r in spans if r.y0 > y1 - 3 and r.y0 < y1 + 8]
        if c and min(c)[0] < label_tol:
            xs.append(p); xv.append(min(c)[1])
    for p in ym:
        c = [(abs((r.y0 + r.y1) / 2 - 0.3 - p), v) for v, r in spans if r.x1 < x0 + 2 and r.x1 > x0 - 6]
        if c and min(c)[0] < label_tol:
            ys.append(p); yv.append(min(c)[1])
    if len(xs) < 2 or len(ys) < 2:
        raise RuntimeError(f"calibration failed: {len(xs)} x ticks, {len(ys)} y ticks")
    bx, ax = np.polyfit(xs, xv, 1)
    by, ay = np.polyfit(ys, yv, 1)
    rx = np.array(xv) - (ax + bx * np.array(xs))
    ry = np.array(yv) - (ay + by * np.array(ys))
    return dict(ax=ax, bx=bx, ay=ay, by=by, xticks=list(zip(xs, xv)), yticks=list(zip(ys, yv)),
                rms_x=float(np.sqrt(np.mean(rx**2))), max_x=float(np.abs(rx).max()),
                rms_y=float(np.sqrt(np.mean(ry**2))), max_y=float(np.abs(ry).max()))


def to_data(cal, px, py):
    return cal["ax"] + cal["bx"] * px, cal["ay"] + cal["by"] * py


def to_page(cal, x, y):
    return (x - cal["ax"]) / cal["bx"], (y - cal["ay"]) / cal["by"]


# ------------------------------------------------------------------------------------------------------------------ curves
def close(c, ref, tol=0.03):
    return c is not None and len(c) == 3 and all(abs(a - b) < tol for a, b in zip(c, ref))


def curve_paths(page, frame):
    """Stroked paths made of line items (>= 10 vertices) inside the frame: list of dicts with colour, dashes, points."""
    out = []
    fr = pymupdf.Rect(frame.x0 - 1, frame.y0 - 1, frame.x1 + 1, frame.y1 + 1)
    for dr in page.get_drawings():
        its = dr["items"]
        if len(its) < 8 or not all(i[0] == "l" for i in its) or dr["color"] is None:
            continue
        if not fr.contains(dr["rect"]) and not dr["rect"].intersects(fr):
            continue
        pts = [(its[0][1].x, its[0][1].y)]
        for i in its:
            if abs(i[1].x - pts[-1][0]) > 1e-6 or abs(i[1].y - pts[-1][1]) > 1e-6:
                pts.append((i[1].x, i[1].y))
            pts.append((i[2].x, i[2].y))
        out.append(dict(color=tuple(round(c, 2) for c in dr["color"]), dashes=dr.get("dashes"), width=dr["width"], pts=pts, rect=dr["rect"]))
    return out


def markers(page, frame, color, kind=None, minsize=1.5, maxsize=5.0):
    """Centres of small closed marker paths of a given stroke colour inside the frame (circles / squares)."""
    out = []
    fr = pymupdf.Rect(frame.x0 - 1, frame.y0 - 1, frame.x1 + 1, frame.y1 + 1)
    for dr in page.get_drawings():
        r = dr["rect"]
        if dr["color"] is None and dr["fill"] is None:
            continue
        col = dr["color"] if dr["color"] is not None else dr["fill"]
        if not close(col, color):
            continue
        if not (minsize < r.width < maxsize and minsize < r.height < maxsize) or not fr.contains(r):
            continue
        if len(dr["items"]) > 6 and False:
            continue
        out.append(((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2))
    return out


def dedupe(points, tol=0.15):
    res = []
    for p in points:
        if not any(abs(p[0] - q[0]) < tol and abs(p[1] - q[1]) < tol for q in res):
            res.append(p)
    return res


# ------------------------------------------------------------------------------------------------------------- panel specs
# Colour/dash/marker -> curve names were assigned from the legends of the figures (checked in the overlay plots).
RED, BLUE, GREEN, MAGENTA, ORANGE, BLK = (1.0, 0, 0), (0, 0, 1.0), (0, 1.0, 0), (1.0, 0, 1.0), (1.0, 0.65, 0), BLACK

PANELS = {
    # name: page index, frame index (sorted by position), axis names, curves [(csv name, description, selector)]
    "fig2": dict(page=5, frame_pt=(450, 200), xlab="k_fm-1", ylab="gamma_k_c_per_fm", fig="Fig. 2 (gamma_k(k) at rho = 6.5 rho_s, T = 70 MeV)",
                 curves=[("fig2_gamma_k_eta0_0_kappa0_0", "(eta0,kappa0) = (0,0), ideal", BLK),
                         ("fig2_gamma_k_eta0_1_kappa0_0", "(eta0,kappa0) = (1,0)", MAGENTA),
                         ("fig2_gamma_k_eta0_0_kappa0_1", "(eta0,kappa0) = (0,1)", GREEN),
                         ("fig2_gamma_k_eta0_1_kappa0_1", "(eta0,kappa0) = (1,1)", BLUE)]),
    "fig3a": dict(page=6, frame_pt=(180, 170), xlab="rho_over_rho_s", ylab="gamma0_c_per_fm", fig="Fig. 3 upper panel (fastest growth rate vs compression, T = 70 MeV)",
                  curves=[("fig3a_gamma0_vs_rho_0_0", "(eta0,kappa0) = (0,0)", BLK),
                          ("fig3a_gamma0_vs_rho_1_1", "(eta0,kappa0) = (1,1)", BLUE),
                          ("fig3a_gamma0_vs_rho_5_5", "(eta0,kappa0) = (5,5)", RED)]),
    "fig3b": dict(page=6, frame_pt=(180, 290), xlab="T_MeV", ylab="gamma0_c_per_fm", fig="Fig. 3 lower panel (fastest growth rate vs temperature, rho = 6.5 rho_s)",
                  curves=[("fig3b_gamma0_vs_T_0_0", "(eta0,kappa0) = (0,0)", BLK),
                          ("fig3b_gamma0_vs_T_1_1", "(eta0,kappa0) = (1,1)", BLUE),
                          ("fig3b_gamma0_vs_T_5_5", "(eta0,kappa0) = (5,5)", RED)]),
    "fig4": dict(page=7, frame_pt=(180, 190), xlab="t_fm_c", ylab="rho_fm-3", fig="Fig. 4 (net baryon density at the centre, from Ref. [17])",
                 curves=[("fig4_rho_3fluid_5", "3-fluid, 5 A GeV (solid magenta)", (MAGENTA, "solid")),
                         ("fig4_rho_3fluid_10", "3-fluid, 10 A GeV (dashed magenta)", (MAGENTA, "dashed")),
                         ("fig4_rho_urqmd_5", "UrQMD, 5 A GeV (solid orange with circles)", (ORANGE, "solid")),
                         ("fig4_rho_urqmd_10", "UrQMD, 10 A GeV (dashed orange with circles)", (ORANGE, "dashed"))]),
    "fig5": dict(page=7, frame_pt=(180, 420), xlab="rho_over_rho_s", ylab="T_MeV", fig="Fig. 5 (phase trajectories at 5 A GeV and phase boundaries)",
                 curves=[("fig5_boundary_coexistence", "coexistence boundary (red solid)", (RED, "solid")),
                         ("fig5_boundary_isothermal_spinodal", "isothermal spinodal vT = 0 (blue dashed)", (BLUE, "dashed")),
                         ("fig5_boundary_isentropic_spinodal", "isentropic spinodal vs = 0 (green solid)", (GREEN, "solid")),
                         ("fig5_traj_3fluid_5", "3-fluid 5 A GeV trajectory (magenta line)", (MAGENTA, "solid")),
                         ("fig5_traj_urqmd_5", "UrQMD 5 A GeV trajectory (orange line)", (ORANGE, "solid"))]),
    "fig6": dict(page=7, frame_pt=(450, 190), xlab="t_fm_c", ylab="gamma_k_c_per_fm", fig="Fig. 6 (gamma_k(t) for k = 2 fm^-1, eta0 = kappa0 = 1)",
                 curves=[("fig6_gamma_k2_3fluid", "3-fluid 5 A GeV (magenta)", MAGENTA),
                         ("fig6_gamma_k2_urqmd", "UrQMD 5 A GeV (orange)", ORANGE)]),
    "fig7": dict(page=9, frame_pt=(180, 190), xlab="k_fm-1", ylab="G_k", fig="Fig. 7 (amplification factor G_k, 3-fluid 5 A GeV trajectory unless noted)",
                 curves=[("fig7_Gk_0_0", "(eta0,kappa0) = (0,0)", BLK),
                         ("fig7_Gk_1_1", "(eta0,kappa0) = (1,1)", BLUE),
                         ("fig7_Gk_2_2", "(eta0,kappa0) = (2,2)", GREEN),
                         ("fig7_Gk_5_5", "(eta0,kappa0) = (5,5)", RED),
                         ("fig7_Gk_urqmd_1_1", "UrQMD trajectory, (eta0,kappa0) = (1,1) (orange)", ORANGE)]),
}


def dash_kind(d):
    if not d or d.strip() in ("[] 0", "[] 0.0", ""):
        return "solid"
    return "dashed"


def write_csv(path, header_lines, cols, rows):
    with open(path, "w", newline="") as f:
        for h in header_lines:
            f.write("# " + h + "\n")
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([f"{v:.6g}" for v in r])


def run_panel(doc, name, spec, outdir, figdir, summary):
    page = doc[spec["page"]]
    frames = find_frames(page)
    fx, fy = spec["frame_pt"]
    frame = next((r for r in frames if r.x0 <= fx <= r.x1 and r.y0 <= fy <= r.y1), None)
    if frame is None:
        raise RuntimeError(f"{name}: frame not found")
    cal = calibrate(page, frame)
    paths = curve_paths(page, frame)
    plotted = []
    for cname, desc, sel in spec["curves"]:
        col, kind = (sel if isinstance(sel[0], tuple) else (sel, None))
        cand = [p for p in paths if close(p["color"], col)]
        if kind:
            cand = [p for p in cand if dash_kind(p["dashes"]) == kind]
        if not cand:
            summary.append(f"{name}: curve '{cname}' NOT FOUND")
            continue
        # several matching paths: take the longest (the legend line is a 1-item path and is excluded by the >= 8 items rule)
        if name == "fig5" and cname.startswith("fig5_traj"):
            cand = [p for p in cand if len(p["pts"]) < 120]
        p = max(cand, key=lambda q: len(q["pts"]))
        px = np.array([q[0] for q in p["pts"]]); py = np.array([q[1] for q in p["pts"]])
        xd, yd = to_data(cal, px, py)
        # remove consecutive duplicates (both ends of adjacent line items)
        keep = np.r_[True, (np.abs(np.diff(xd)) > 1e-9) | (np.abs(np.diff(yd)) > 1e-9)]
        xd, yd = xd[keep], yd[keep]
        # keep only vertices inside the frame (curves that leave the frame are clipped by the viewer)
        hdr = [spec["fig"], f"curve: {desc}",
               f"source: arXiv:1007.1448 PDF, page {spec['page'] + 1}, vector path with {len(p['pts'])} vertices, stroke colour {p['color']}, dash {p['dashes']!r}",
               f"mapping: x = {cal['ax']:.6g} + {cal['bx']:.8g} * X_pt ; y = {cal['ay']:.6g} + {cal['by']:.8g} * Y_pt (PDF user-space points, y downwards)",
               f"calibration: {len(cal['xticks'])} x ticks, {len(cal['yticks'])} y ticks; residuals rms/max x = {cal['rms_x']:.3g}/{cal['max_x']:.3g}, y = {cal['rms_y']:.3g}/{cal['max_y']:.3g} (data units)",
               "columns in data units"]
        write_csv(os.path.join(outdir, cname + ".csv"), hdr, [spec["xlab"], spec["ylab"]], np.c_[xd, yd])
        plotted.append((cname, xd, yd))
        summary.append(f"{name}: {cname}: {len(xd)} points, x in [{xd.min():.4g}, {xd.max():.4g}], y in [{yd.min():.4g}, {yd.max():.4g}]")
    # marker centres (trajectory symbols, density-evolution circles)
    extra = []
    if name == "fig4":
        # independent accuracy check: the UrQMD points are plotted at integer times, so the vertex x values must be integers
        for cname, xd, yd in plotted:
            if "urqmd" in cname:
                dev = xd - np.round(xd)
                summary.append(f"fig4: {cname}: {len(xd)} vertices; deviation of the vertex time from the nearest integer fm/c: rms {np.sqrt(np.mean(dev**2)):.2e}, max {np.abs(dev).max():.2e}")
    if name == "fig5":
        for cname, col in (("fig5_markers_3fluid_5", MAGENTA), ("fig5_markers_urqmd_5", ORANGE)):
            m = dedupe(markers(page, frame, col, minsize=1.5, maxsize=6))
            xd, yd = to_data(cal, np.array([q[0] for q in m]), np.array([q[1] for q in m]))
            # legend symbols lie in the lower-left corner (T < 45 MeV): drop them
            sel = yd > 60
            xd, yd = xd[sel], yd[sel]
            o = np.argsort(xd) if False else np.arange(len(xd))
            hdr = [spec["fig"], f"{cname}: marker centres of the trajectory (spaced by dt = 1 fm/c), in drawing order",
                   f"mapping as fig5 curves", "columns in data units"]
            write_csv(os.path.join(outdir, cname + ".csv"), hdr, [spec["xlab"], spec["ylab"]], np.c_[xd, yd])
            extra.append((cname, xd, yd))
            summary.append(f"fig5: {cname}: {len(xd)} markers")
    # check image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dpi = 220
    clip = pymupdf.Rect(frame.x0 - 48, frame.y0 - 14, frame.x1 + 14, frame.y1 + 34)
    pix = page.get_pixmap(clip=clip, dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    fig, ax = plt.subplots(figsize=(pix.width / 100, pix.height / 100), dpi=100)
    ax.imshow(img, extent=(clip.x0, clip.x1, clip.y1, clip.y0))
    cols = plt.cm.tab10.colors
    for i, (cname, xd, yd) in enumerate(plotted):
        px, py = to_page(cal, xd, yd)
        ax.plot(px, py, "-", color="k", lw=0.6, alpha=0.9)
        ax.plot(px, py, ".", ms=2.2, color=cols[i % 10], label=cname.replace(name + "_", ""))
    for cname, xd, yd in extra:
        px, py = to_page(cal, xd, yd)
        ax.plot(px, py, "x", ms=5, color="k", mew=0.8)
    ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
    ax.set_title(f"{name}: digitised vertices over the rasterised page ({spec['fig'][:60]})", fontsize=7)
    ax.axis("off")
    fig.savefig(os.path.join(figdir, f"digitise_{name}.png"), dpi=100)
    plt.close(fig)
    return cal


def derive_trajectories(outdir, summary, rho_s=0.153):
    """Trajectory CSVs (t, rho/rho_s, T) for spf_linear from the Fig. 5 markers (spaced by dt = 1 fm/c).
    Time assignment: marker j <-> t = j fm/c. Tested against the Fig. 4 density curves (summary lines)."""
    def load(n):
        a = np.genfromtxt(os.path.join(outdir, n + ".csv"), delimiter=",", comments="#")
        return a[~np.isnan(a).any(axis=1)]
    for model, f4, mk, jmin in (("3fluid", "fig4_rho_3fluid_5", "fig5_markers_3fluid_5", 1), ("urqmd", "fig4_rho_urqmd_5", "fig5_markers_urqmd_5", 3)):
        c, m = load(f4), load(mk)
        t = np.arange(len(m), dtype=float)
        r4 = np.interp(t, c[:, 0], c[:, 1]) / rho_s
        dd = (r4 - m[:, 0])[jmin:]
        summary.append(f"trajectory {model}: {len(m)} markers; rho(Fig.5 marker j) vs rho(Fig.4, t = j) for j >= {jmin}: rms {np.sqrt(np.mean(dd**2)):.3f} rho_s, max {np.abs(dd).max():.3f} rho_s;"
                       f" first markers j < {jmin} lie at rho/rho_s = {', '.join(f'{v:.2f}' for v in m[:jmin, 0])} vs Fig. 4: {', '.join(f'{v:.2f}' for v in r4[:jmin])}")
        hdr = [f"trajectory {model} 5 A GeV from the markers of Fig. 5 (derived)", "t = marker index in fm/c (markers are spaced by dt = 1 fm/c, paper Fig. 5 caption; start time inferred from Fig. 4, see tools/digitise_1007_1448.py)",
               "rho/rho_s and T (MeV) are the digitised marker centres", "columns: t_fm_c, rho_over_rho_s, T_MeV"]
        write_csv(os.path.join(outdir, f"trajectory_{model}_5_markers.csv"), hdr, ["t_fm_c", "rho_over_rho_s", "T_MeV"], np.c_[t, m[:, 0], m[:, 1]])
        if model == "3fluid":
            tt = np.arange(1.0, len(m) - 1 + 1e-9, 0.1)
            rho = np.interp(tt, c[:, 0], c[:, 1]) / rho_s
            T = np.interp(tt, t, m[:, 1])
            hdr[0] = "trajectory 3fluid 5 A GeV, dense (derived): rho(t) from the Fig. 4 curve, T(t) linearly interpolated between the Fig. 5 markers"
            write_csv(os.path.join(outdir, "trajectory_3fluid_5_dense.csv"), hdr, ["t_fm_c", "rho_over_rho_s", "T_MeV"], np.c_[tt, rho, T])


def catalogue(doc):
    for name, spec in PANELS.items():
        page = doc[spec["page"]]
        frames = find_frames(page)
        fx, fy = spec["frame_pt"]
        frame = next((r for r in frames if r.x0 <= fx <= r.x1 and r.y0 <= fy <= r.y1), None)
        print("==", name, frame)
        cal = calibrate(page, frame)
        print("  ticks x", cal["xticks"], "\n  ticks y", cal["yticks"], "\n  rms", cal["rms_x"], cal["rms_y"], "max", cal["max_x"], cal["max_y"])
        for p in curve_paths(page, frame):
            print("  path", p["color"], p["dashes"], len(p["pts"]), [round(v, 1) for v in p["rect"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", default="tools/data")
    ap.add_argument("--figures", default="docs/figures")
    ap.add_argument("--catalogue", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    doc = pymupdf.open(a.pdf)
    if a.catalogue:
        catalogue(doc)
        return 0
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.figures, exist_ok=True)
    summary = []
    cals = {}
    for name, spec in PANELS.items():
        if a.only and name not in a.only.split(","):
            continue
        cals[name] = run_panel(doc, name, spec, a.out, a.figures, summary)
    if not a.only:
        derive_trajectories(a.out, summary)
    print("\n".join(summary))
    print("\ncalibration residuals (data units; rms / max):")
    for n, c in cals.items():
        print(f"  {n}: x {c['rms_x']:.3g} / {c['max_x']:.3g}  y {c['rms_y']:.3g} / {c['max_y']:.3g}   scale: {abs(c['bx']):.4g} x-units/pt, {abs(c['by']):.4g} y-units/pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
