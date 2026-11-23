#!/usr/bin/env python3
"""Digitise the collision figures of arXiv:1209.2462 (Fig. 3, Fig. 4) and arXiv:1302.2956 (Fig. 5, Fig. 6, Fig. 7) from the vector drawing
operators of the PDFs (the same principle as tools/digitise_1007_1448.py; nothing is read from a raster image).

usage: python tools/digitise_wp7.py --pdf-1209 1209.2462.pdf --pdf-1302 1302.2956.pdf [--out tools/data] [--check docs/figures]

Per panel: the axis frame is found among the black stroked paths, the major tick marks on the left and bottom edges are fitted to the
tick values given below (linear, or log10 for the log axes; residuals are written into the CSV header), and the vertices of the curve paths /
the bounding-box centres of the marker paths are converted to data units. Dotted curves (drawn as many one-item paths) give their dot centres.
Curve identities follow the legends (colour, dash pattern, filled / open marker). Markers are located by the centre of the bounding box of
their path (for triangles this is the middle of the bounding box, not the centroid).
Output: <out>/p1209_fig3_*.csv, p1209_fig4_*.csv, p1302_fig5_*.csv, p1302_fig6_*.csv, p1302_fig7_*.csv with the header
'# source: <paper>, Fig. N, page P (1-based); path colour/dash; axis fit rms' and check plots <check>/digitise_wp7_*.png.
"""
import argparse
import math
import os

import numpy as np
import pymupdf


def is_black(c):
    return c is not None and max(c) < 0.05


def seg_list(page, max_width=10.0):
    out = []
    for dr in page.get_drawings():
        if (dr["width"] or 0) > max_width:
            continue
        for it in dr["items"]:
            if it[0] == "l" and is_black(dr["color"]):
                out.append((it[1].x, it[1].y, it[2].x, it[2].y))
    return out


def find_frame(page, x0, y0, x1, y1, tol=3.0):
    """frame (rect) of the black stroked rectangle whose corners are within tol of the given approximate corners"""
    cands = []
    for dr in page.get_drawings():
        for it in dr["items"]:
            if it[0] == "qu" and is_black(dr["color"]):
                cands.append(it[1].rect)
    # frames built from four separate lines
    segs = [s for s in seg_list(page) if math.hypot(s[2] - s[0], s[3] - s[1]) > 40]
    xs = sorted({round(min(s[0], s[2]), 1) for s in segs if abs(s[0] - s[2]) < 0.05} | {round(s[0], 1) for s in segs if abs(s[0] - s[2]) < 0.05})
    for r in cands:
        if abs(r.x0 - x0) < tol and abs(r.y0 - y0) < tol and abs(r.x1 - x1) < tol and abs(r.y1 - y1) < tol:
            return r
    left = [s for s in segs if abs(s[0] - s[2]) < 0.05 and abs(s[0] - x0) < tol]
    right = [s for s in segs if abs(s[0] - s[2]) < 0.05 and abs(s[0] - x1) < tol]
    top = [s for s in segs if abs(s[1] - s[3]) < 0.05 and abs(s[1] - y0) < tol]
    bot = [s for s in segs if abs(s[1] - s[3]) < 0.05 and abs(s[1] - y1) < tol]
    if left and right and top and bot:
        return pymupdf.Rect(left[0][0], top[0][1], right[0][0], bot[0][1])
    raise RuntimeError("frame not found near %s" % ((x0, y0, x1, y1),))


def major_ticks(page, fr, max_width=10.0):
    """positions of the long tick class on the bottom edge (x) and left edge (y)"""
    xt, yt = [], []
    for ax, ay, bx, by in seg_list(page, max_width):
        L = math.hypot(bx - ax, by - ay)
        if not (0.8 < L < 8):
            continue
        # ticks point into the frame: one end on the edge, the other inside
        if abs(ax - bx) < 0.05 and fr.x0 - 0.4 <= ax <= fr.x1 + 0.4:
            far = ay if abs(by - fr.y1) < 0.4 else by if abs(ay - fr.y1) < 0.4 else None
            if far is not None and fr.y0 <= far <= fr.y1:
                xt.append((ax, L))
        if abs(ay - by) < 0.05 and fr.y0 - 0.4 <= ay <= fr.y1 + 0.4:
            far = bx if abs(ax - fr.x0) < 0.4 else ax if abs(bx - fr.x0) < 0.4 else None
            if far is not None and fr.x0 <= far <= fr.x1:
                yt.append((ay, L))
    def major(t):
        if not t:
            return []
        Lm = max(L for _, L in t)
        pos = sorted({round(p, 2) for p, L in t if L > 0.75 * Lm})
        merged = []
        for p in pos:
            if not merged or p - merged[-1] > 1.0:
                merged.append(p)
        return merged
    return major(xt), major(yt)


class Axes:
    def __init__(self, page, fr, xvals, yvals, ylog=False, xlog=False, xfrom=None, max_width=10.0):
        xm, ym = major_ticks(page, fr, max_width)
        if xfrom is not None:  # the x axis is that of another panel (shared axis)
            xm = xfrom[0]
        if len(xm) < len(xvals) or len(ym) < len(yvals):
            raise RuntimeError("expected %d x and %d y major ticks, found %d and %d" % (len(xvals), len(yvals), len(xm), len(ym)))
        # tick values are given from the left / the bottom; keep the matching ticks (a frame may have more major ticks than labelled ones)
        xm = xm[:len(xvals)] if len(xm) != len(xvals) else xm
        ym = sorted(ym, reverse=True)[:len(yvals)] if len(ym) != len(yvals) else sorted(ym, reverse=True)
        self.xlog, self.ylog = xlog, ylog
        xv = np.log10(xvals) if xlog else np.array(xvals, float)
        yv = np.log10(yvals) if ylog else np.array(yvals, float)
        self.bx, self.ax = np.polyfit(xm, xv, 1)
        self.by, self.ay = np.polyfit(ym, yv, 1)
        self.rms_x = float(np.sqrt(np.mean((xv - (self.ax + self.bx * np.array(xm))) ** 2)))
        self.rms_y = float(np.sqrt(np.mean((yv - (self.ay + self.by * np.array(ym))) ** 2)))
        self.fr = fr

    def x(self, px):
        v = self.ax + self.bx * px
        return 10 ** v if self.xlog else v

    def y(self, py):
        v = self.ay + self.by * py
        return 10 ** v if self.ylog else v

    def label(self):
        return "axis fit rms (x, y; log10 units on log axes) = %.3g, %.3g" % (self.rms_x, self.rms_y)


def inside(r, fr, pad=1.0):
    return fr.x0 - pad <= r.x0 and r.x1 <= fr.x1 + pad and fr.y0 - pad <= r.y0 and r.y1 <= fr.y1 + pad


def vertices(dr):
    pts = []
    for it in dr["items"]:
        if it[0] == "l":
            pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
        elif it[0] == "c":
            pts += [(it[1].x, it[1].y), (it[4].x, it[4].y)]
    out = []
    for p in pts:
        if not out or abs(p[0] - out[-1][0]) > 1e-6 or abs(p[1] - out[-1][1]) > 1e-6:
            out.append(p)
    return out


def write_csv(path, header, cols, rows):
    with open(path, "w") as f:
        for h in header:
            f.write("# " + h + "\n")
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("%.8g" % v for v in r) + "\n")


def cname(col):
    return {(0, 0, 0): "black", (0, 0.5, 0): "green", (1, 0, 0): "red", (0, 0, 1): "blue", (0.58, 0, 0.83): "violet", (1, 0, 1): "magenta",
            (0.5, 0.5, 0.5): "grey"}.get(tuple(round(c, 2) if abs(c - 0.54) > 0.01 else 0.5 for c in col), str(tuple(round(c, 2) for c in col)))


# --------------------------------------------------------------------------------------------------------------------------- figures
def fig3_1209(doc, out, checks):
    page = doc[2]
    fr = find_frame(page, 352.4, 57.1, 551.4, 196.4)
    ax = Axes(page, fr, [0, 1, 2, 3, 4, 5, 6], [1, 10, 100, 1000], ylog=True)
    order = {"black": 2, "green": 3, "blue": 4, "violet": 5, "magenta": 6, "red": 7}
    for dr in page.get_drawings():
        if dr["type"] != "s" or not dr["color"] or len(dr["items"]) < 40 or not inside(dr["rect"], fr):
            continue
        name = cname(dr["color"])
        if name not in order or abs(dr["width"] - 0.74) > 0.05:
            continue
        kind = "maxwell" if "3.72" in str(dr.get("dashes")) else "two_phase"
        pts = vertices(dr)
        rows = [(ax.x(x), ax.y(y)) for x, y in pts]
        hdr = ["source: arXiv:1209.2462 Fig. 3 (page 3), E_lab = 3 A GeV, a = 0.033 fm; N = %d; %s EoS (%s, %s); normalised density moment <rho^N>/<rho>^N versus t - t0 [fm/c]"
               % (order[name], kind, "solid" if kind == "two_phase" else "dashed", name), ax.label()]
        write_csv(os.path.join(out, "p1209_fig3_N%d_%s.csv" % (order[name], kind)), hdr, ["t_minus_t0_fm", "moment"], rows)
        checks.append(("1209 Fig.3 N=%d %s" % (order[name], kind), rows))


def fig4_1209(doc, out, checks):
    page = doc[3]
    # two panels: N = 7 above, N = 5 below
    fr7 = fr5 = None
    for dr in page.get_drawings():
        for it in dr["items"]:
            if it[0] == "qu" and is_black(dr["color"]):
                r = it[1].rect
                if r.width > 150:
                    (fr7 := r) if (fr7 is None and r.y0 < 150) else None
    quads = sorted({(round(it[1].rect.y0, 1), round(it[1].rect.y1, 1), it[1].rect) for dr in page.get_drawings() for it in dr["items"]
                    if it[0] == "qu" and is_black(dr["color"]) and it[1].rect.width > 150}, key=lambda t: t[0])
    fr7, fr5 = quads[0][2], quads[1][2]
    # tick values: top panel 0..15 (labels 0, 5, 10, 15), bottom panel 0..3 (labels 0 ... 3)
    ax5 = Axes(page, fr5, [0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3])
    ax7 = Axes(page, fr7, [0, 1, 2, 3, 4, 5, 6], [0, 5, 10, 15], xfrom=major_ticks(page, fr5))
    for N, fr, ax, colname in [(7, fr7, ax7, "red"), (5, fr5, ax5, "violet")]:
        std, mx = [], []
        for dr in page.get_drawings():
            r = dr["rect"]
            if not inside(r, fr, 2.0) or not (3.0 < r.width < 4.2 and 3.0 < r.height < 4.2) or not dr["color"] or cname(dr["color"]) != colname:
                continue
            c = ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
            e2 = 2.0 * ax.x(c[0])
            if abs(e2 - round(e2)) > 0.02:  # legend markers do not sit on the 0.5 A GeV grid of the data
                continue
            (std if dr["fill"] else mx).append((ax.x(c[0]), ax.y(c[1])))
        for kind, rows in (("two_phase", std), ("maxwell", mx)):
            rows = sorted(rows)
            hdr = ["source: arXiv:1209.2462 Fig. 4 (page 4), N = %d panel; %s EoS (%s markers, %s); mean maximum enhancement versus E_lab [A GeV]" %
                   (N, kind, "filled" if kind == "two_phase" else "open", colname), ax.label()]
            write_csv(os.path.join(out, "p1209_fig4_N%d_%s.csv" % (N, kind)), hdr, ["E_lab_GeV", "max_enhancement"], rows)
            checks.append(("1209 Fig.4 N=%d %s" % (N, kind), rows))


def fig5_1302(doc, out, checks):
    page = doc[5]
    fr = find_frame(page, 84.5, 61.3, 295.2, 203.9)
    ax = Axes(page, fr, [0, 1, 2, 3, 4, 5, 6], [0.1, 1, 10], ylog=True)
    order = {"black": 7, "green": 5, "red": 2}
    dots = {}
    for dr in page.get_drawings():
        if dr["type"] != "s" or not dr["color"] or not inside(dr["rect"], fr, 4.0):
            continue
        name = cname(dr["color"])
        if name not in order:
            continue
        if len(dr["items"]) > 40 and abs(dr["width"] - 0.67) < 0.05 and True:
            rows = [(ax.x(x), ax.y(y)) for x, y in vertices(dr)]
            hdr = ["source: arXiv:1302.2956 Fig. 5 (page 6), E_lab = 3 A GeV, a = 0.033 fm; N = %d; two-phase EoS (solid, %s); enhancement <rho^N>/<rho>^N versus t - t_init [fm]" % (order[name], name), ax.label()]
            write_csv(os.path.join(out, "p1302_fig5_N%d_two_phase.csv" % order[name]), hdr, ["t_minus_tinit_fm", "enhancement"], rows)
            checks.append(("1302 Fig.5 N=%d two_phase" % order[name], rows))
        elif len(dr["items"]) == 1 and dr["rect"].width < 1.5 and abs(dr["width"] - 0.67) < 0.05:
            r = dr["rect"]
            dots.setdefault(name, []).append((ax.x((r.x0 + r.x1) / 2), ax.y((r.y0 + r.y1) / 2)))
    for name, pts in dots.items():
        rows = sorted(pts)
        hdr = ["source: arXiv:1302.2956 Fig. 5 (page 6), E_lab = 3 A GeV, a = 0.033 fm; N = %d; Maxwell EoS (dotted, %s; dot centres of the one-item paths); enhancement versus t - t_init [fm]" % (order[name], name), ax.label()]
        write_csv(os.path.join(out, "p1302_fig5_N%d_maxwell.csv" % order[name]), hdr, ["t_minus_tinit_fm", "enhancement"], rows)
        checks.append(("1302 Fig.5 N=%d maxwell" % order[name], rows))


def fig6_1302(doc, out, checks):
    page = doc[5]
    fr = find_frame(page, 342.4, 61.3, 558.2, 203.9)
    ax = Axes(page, fr, [1, 2, 3, 4, 5, 6], [1, 10], ylog=True)
    legend = pymupdf.Rect(462, 63, 556, 91)
    order = {"red": 2, "green": 5, "black": 7}
    mk = {}
    for dr in page.get_drawings():
        r = dr["rect"]
        # every marker is drawn twice: a fill (clipped by the frame at the edge) and a stroked outline; filled markers have the thin outline
        # (0.3), open markers the thick one (0.82): use the outlines only
        if dr["type"] != "s" or not dr["color"] or dr["width"] is None or not (abs(dr["width"] - 0.3) < 0.05 or abs(dr["width"] - 0.82) < 0.05):
            continue
        name = cname(dr["color"])
        if name not in order or not inside(r, fr, 3.0) or legend.intersects(r) or not (2.5 < r.width < 6 and 2.5 < r.height < 6):
            continue
        filled = abs(dr["width"] - 0.3) < 0.05
        c = ((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2)
        mk.setdefault((name, filled), {})[round(c[0], 0)] = (ax.x(c[0]), ax.y(c[1]))
    for (name, filled), d in mk.items():
        kind = "two_phase" if filled else "maxwell"
        rows = sorted(d.values())
        hdr = ["source: arXiv:1302.2956 Fig. 6 (page 6), N = %d (%s markers: %s squares/circles/triangles for N = 2/5/7); %s EoS; mean maximum enhancement versus E_lab [A GeV]" %
               (order[name], "filled" if filled else "open", name, kind), ax.label()]
        write_csv(os.path.join(out, "p1302_fig6_N%d_%s.csv" % (order[name], kind)), hdr, ["E_lab_GeV", "max_enhancement"], rows)
        checks.append(("1302 Fig.6 N=%d %s" % (order[name], kind), rows))


def fig7_1302(doc, out, checks):
    page = doc[6]
    fr = find_frame(page, 129.0, 91.5, 340.0, 314.0, tol=12.0) if False else None
    # frame: the black 0.51 lines; locate by the long lines of the left panel
    segs = [s for s in seg_list(page) if math.hypot(s[2] - s[0], s[3] - s[1]) > 150 and s[0] < 300 and max(s[1], s[3]) < 330]
    xs = [min(s[0], s[2]) for s in segs if abs(s[1] - s[3]) < 0.05] + [max(s[0], s[2]) for s in segs if abs(s[1] - s[3]) < 0.05]
    ys = [s[1] for s in segs if abs(s[1] - s[3]) < 0.05] + [min(s[1], s[3]) for s in segs if abs(s[0] - s[2]) < 0.05] + [max(s[1], s[3]) for s in segs if abs(s[0] - s[2]) < 0.05]
    fr = pymupdf.Rect(min(xs), min(ys), max(xs), max(ys))
    ax = Axes(page, fr, [0, 5, 10, 15, 20, 25], [1e-4, 1e-3, 1e-2, 1e-1, 1.0], ylog=True, max_width=0.6)
    legend = None
    for dr in page.get_drawings():
        if dr["type"] == "s" and is_black(dr["color"]) and abs((dr["width"] or 0) - 0.65) < 0.02 and len(dr["items"]) > 40:
            # the path of the histogram outline also holds the legend sample lines: keep the segments that lie inside the frame
            segs = [(it[1].x, it[1].y, it[2].x, it[2].y) for it in dr["items"] if it[0] == "l" and
                    fr.x0 - 0.5 <= min(it[1].x, it[2].x) and max(it[1].x, it[2].x) <= fr.x1 + 0.5 and fr.y0 - 0.5 <= min(it[1].y, it[2].y) and max(it[1].y, it[2].y) <= fr.y1 + 0.5]
            rows = [(ax.x(x0), ax.y(y0)) for x0, y0, x1, y1 in segs] + [(ax.x(segs[-1][2]), ax.y(segs[-1][3]))]
            hdr = ["source: arXiv:1302.2956 Fig. 7 (page 7), E_lab = 3 A GeV, rho_min = 7 rho_s; two-phase EoS size distribution (solid histogram outline: start points of the step segments); probability versus baryon number in clump", ax.label()]
            write_csv(os.path.join(out, "p1302_fig7_two_phase_outline.csv"), hdr, ["baryon_number", "probability"], rows)
            checks.append(("1302 Fig.7 two_phase outline", rows))
            bins = []
            for x0, y0, x1, y1 in segs:
                if abs(y0 - y1) < 1e-3 and abs(x1 - x0) > 1.0:
                    xa, xb = sorted((x0, x1))
                    bins.append((ax.x(xa), ax.x(xb), ax.y(y0)))
            write_csv(os.path.join(out, "p1302_fig7_two_phase_bins.csv"),
                      ["source: arXiv:1302.2956 Fig. 7 (page 7); two-phase histogram: horizontal runs of the step outline", ax.label()],
                      ["x_lo", "x_hi", "probability"], bins)
    # legend box: the labels of the legend and the sample lines to their left
    lab = [pymupdf.Rect(sp["bbox"]) for blk in page.get_text("dict")["blocks"] for ln in blk.get("lines", []) for sp in ln["spans"]
           if any(k in sp["text"] for k in ("Unstable", "Maxwell", "Exponential", "Fit")) and sp["bbox"][0] < 300 and sp["bbox"][1] < 160]
    legend = []
    if lab:
        u = pymupdf.Rect(lab[0])
        for r in lab[1:]:
            u |= r
        legend = [pymupdf.Rect(u.x0 - 40, u.y0 - 3, u.x1 + 3, u.y1 + 3)]
    dots = {}
    for dr in page.get_drawings():
        if dr["type"] != "s" or not dr["color"] or len(dr["items"]) != 1 or dr["rect"].width > 1.5 or not inside(dr["rect"], fr) or abs(dr["width"] - 0.65) > 0.02:
            continue
        if any(l.contains(pymupdf.Point((dr["rect"].x0 + dr["rect"].x1) / 2, (dr["rect"].y0 + dr["rect"].y1) / 2)) for l in legend):  # the legend box
            continue
        name = cname(dr["color"])
        if name in ("grey", "red"):
            r = dr["rect"]
            dots.setdefault(name, []).append((ax.x((r.x0 + r.x1) / 2), ax.y((r.y0 + r.y1) / 2)))
    for name, kind in (("grey", "maxwell"), ("red", "difference")):
        rows = sorted(dots.get(name, []))
        hdr = ["source: arXiv:1302.2956 Fig. 7 (page 7); %s (dotted, %s; dot centres of the one-item paths); probability versus baryon number" %
               ("Maxwell EoS distribution" if kind == "maxwell" else "unstable EoS minus Maxwell EoS", name), ax.label()]
        write_csv(os.path.join(out, "p1302_fig7_%s_dots.csv" % kind), hdr, ["baryon_number", "probability"], rows)
        checks.append(("1302 Fig.7 %s dots" % kind, rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-1209", required=True)
    ap.add_argument("--pdf-1302", required=True)
    ap.add_argument("--out", default="tools/data")
    ap.add_argument("--check", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    checks = []
    d1, d2 = pymupdf.open(a.pdf_1209), pymupdf.open(a.pdf_1302)
    fig3_1209(d1, a.out, checks)
    fig4_1209(d1, a.out, checks)
    fig5_1302(d2, a.out, checks)
    fig6_1302(d2, a.out, checks)
    fig7_1302(d2, a.out, checks)
    for n, rows in checks:
        print("%-34s %4d points" % (n, len(rows)))
    if a.check:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(a.check, exist_ok=True)
        groups = {}
        for n, rows in checks:
            groups.setdefault(n.split(" N=")[0].split(" two_phase")[0].split(" maxwell")[0].split(" difference")[0], []).append((n, rows))
        for g, items in groups.items():
            fig, ax = plt.subplots(figsize=(6, 4))
            for n, rows in items:
                r = np.array(rows)
                ax.plot(r[:, 0], r[:, 1], ".-", ms=3, lw=0.8, label=n)
            if "Fig.3" in g or "Fig.5" in g or "Fig.7" in g or "Fig.6" in g:
                ax.set_yscale("log")
            ax.legend(fontsize=5)
            ax.set_title("digitised " + g)
            fig.savefig(os.path.join(a.check, "digitise_wp7_%s.png" % g.replace(" ", "_").replace(".", "")), dpi=130)
            plt.close(fig)


if __name__ == "__main__":
    main()
