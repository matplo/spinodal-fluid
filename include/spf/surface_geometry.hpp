#pragma once
// WP10a: geometry of the iso-surface f = 0 inside one 4D lattice cell (a hypercube spanned by 2 x 2 x 2 neighbouring cell centres and
// two consecutive time levels).  Re-implementation, from the description in P. Huovinen and H. Petersen, Eur. Phys. J. A 48 (2012) 171,
// arXiv:1206.3371, of the "Cornelius" construction; no code from other projects is used.  Algorithm and orientation rules:
// docs/particlisation_format.md, Sect. 3.
//
//   corners  c = bt | bx << 1 | by << 2 | bz << 3  (bit a = coordinate along axis a; axis 0 = t, 1..3 = x, y, z), local coordinates in [0, 1]^4
//   above    f(c) >= 0 (fluid side), below f(c) < 0
//   edges    32 edges, id = 8 a + (index of the lower corner with bit a removed); a crossing point lies on every edge whose end corners
//            differ in side, at s = f(c0) / (f(c0) - f(c1)) from the lower corner c0 (the same arithmetic in every hypercube sharing the edge)
//   squares  24 squares; a square with two crossing points gives one segment; with four (ambiguous) the sign of the mean of its four
//            corner values decides which diagonal pair is connected (mean >= 0: the above corners are connected, the below corners are
//            cut off; mean < 0: the above corners are cut off)
//   cubes    8 boundary cubes; their segments close into polygons (every crossing point has exactly one incoming and one outgoing segment)
//   pieces   polygons that share a crossing point belong to the same piece (a closed polyhedral 2-surface on the boundary of the hypercube);
//            each piece is one surface element: the cone from the mean of its crossing points over the triangles (polygon centre,
//            segment) gives tetrahedra whose normals add up to the element normal.
// The normal of a piece is exact for the polyhedral boundary it spans (it depends only on that boundary), and neighbouring hypercubes see
// identical boundaries on their common cube, so the sum of the element normals over a closed surface vanishes to rounding.
// Orientation: every normal points from the above side to the below side (the Euclidean outward normal of the region f >= 0); its components
// are the covariant components dSigma_mu (metric +,-,-,-), so that dSigma_mu u^mu > 0 for matter leaving the region f >= 0.
// Orientation is decided combinatorially (from which corners are above), never from the position of the crossing points, so degenerate
// positions (crossing points on a corner) cannot flip it.  All functions are device functions (docs/device_rules.md).
#include "spf/types.hpp"

#include <cstdint>

namespace spf {
namespace surf {

inline constexpr int kMaxPieces = 16;  // surface elements per hypercube (a hypercube has at most 32 polygons)

struct Piece {
  Real xc[4];  // centroid, local coordinates in [0, 1]^4
  Real n[4];   // dSigma_mu in physical units (L0 L1 L2 L3 / L_mu times the lattice-unit normal)
};

KOKKOS_INLINE_FUNCTION int edge_c0(int e) {
  const int a = e >> 3, j = e & 7;
  return (j & ((1 << a) - 1)) | ((j >> a) << (a + 1));
}
KOKKOS_INLINE_FUNCTION int edge_of(int c0, int a) { return a * 8 + ((c0 & ((1 << a) - 1)) | ((c0 >> (a + 1)) << a)); }
// coordinate ax of the midpoint of edge e (used for the combinatorial orientation only)
KOKKOS_INLINE_FUNCTION Real edge_mid(int e, int ax) {
  const int a = e >> 3;
  return ax == a ? 0.5 : static_cast<Real>((edge_c0(e) >> ax) & 1);
}

KOKKOS_INLINE_FUNCTION Real det3(Real a0, Real a1, Real a2, Real b0, Real b1, Real b2, Real c0, Real c1, Real c2) {
  return a0 * (b1 * c2 - b2 * c1) - a1 * (b0 * c2 - b2 * c0) + a2 * (b0 * c1 - b1 * c0);
}

// n_mu = d/dw^mu det[w; a; b; c] / 6 (rows), i.e. the covariant normal of the tetrahedron spanned by a, b, c.
KOKKOS_INLINE_FUNCTION void tet_normal(const Real a[4], const Real b[4], const Real c[4], Real n[4]) {
  for (int mu = 0; mu < 4; ++mu) {
    int k[3], q = 0;
    for (int x = 0; x < 4; ++x)
      if (x != mu) k[q++] = x;
    const Real d = det3(a[k[0]], a[k[1]], a[k[2]], b[k[0]], b[k[1]], b[k[2]], c[k[0]], c[k[1]], c[k[2]]);
    n[mu] = ((mu & 1) ? -d : d) / 6.0;
  }
}

// Multilinear interpolation of 16 corner values at local coordinates x.
KOKKOS_INLINE_FUNCTION Real interp16(const Real v[16], const Real x[4]) {
  Real s = 0.0;
  for (int c = 0; c < 16; ++c) {
    Real w = 1.0;
    for (int a = 0; a < 4; ++a) w *= ((c >> a) & 1) ? x[a] : 1.0 - x[a];
    s += w * v[c];
  }
  return s;
}

// Surface pieces of f = 0 in one hypercube with edge lengths L[4] (physical units).  Returns the number of pieces written to out
// (at most max_out); -1: inconsistent topology (a crossing point without exactly one incoming and one outgoing segment in some cube; cannot
// happen for finite f); -2: more pieces than max_out (the excess is dropped).
KOKKOS_INLINE_FUNCTION int find_pieces(const Real f[16], const Real L[4], Piece* out, int max_out) {
  unsigned above = 0;
  for (int c = 0; c < 16; ++c)
    if (f[c] >= 0.0) above |= 1u << c;
  if (above == 0u || above == 0xFFFFu) return 0;
  auto is_above = [&](int c) { return ((above >> c) & 1u) != 0u; };

  // crossing points
  Real P[32][4];
  unsigned cross = 0;
  for (int e = 0; e < 32; ++e) {
    const int a = e >> 3, c0 = edge_c0(e), c1 = c0 | (1 << a);
    if (is_above(c0) == is_above(c1)) continue;
    cross |= 1u << e;
    const Real s = f[c0] / (f[c0] - f[c1]);
    for (int x = 0; x < 4; ++x) P[e][x] = x == a ? s : static_cast<Real>((c0 >> x) & 1);
  }

  // segments in the 24 squares: end points (edge ids), in-square direction m2 towards the below side (4 components), fixed axes and values
  int sA[48], sB[48], sFix0[48], sFix1[48], sV0[48], sV1[48];
  Real sM[48][4];
  int nseg = 0;
  for (int a = 0; a < 4; ++a)
    for (int b = a + 1; b < 4; ++b) {
      int cd[2], q = 0;
      for (int x = 0; x < 4; ++x)
        if (x != a && x != b) cd[q++] = x;
      for (int vc = 0; vc < 2; ++vc)
        for (int vd = 0; vd < 2; ++vd) {
          const int cb = (vc << cd[0]) | (vd << cd[1]);
          const int qc[4] = {cb, cb | (1 << a), cb | (1 << a) | (1 << b), cb | (1 << b)};  // q00, q10, q11, q01
          const int E[4] = {edge_of(qc[0], a), edge_of(qc[1], b), edge_of(qc[3], a), edge_of(qc[0], b)};
          // corner k lies between edges E[adj0[k]] and E[adj1[k]]
          const int adj0[4] = {3, 0, 1, 2}, adj1[4] = {0, 1, 2, 3};
          int cr[4], n = 0;
          for (int i = 0; i < 4; ++i) {
            cr[i] = (cross >> E[i]) & 1u;
            n += cr[i];
          }
          if (n == 0) continue;
          auto add = [&](int ea, int eb, const Real m[4]) {
            sA[nseg] = ea; sB[nseg] = eb;
            for (int x = 0; x < 4; ++x) sM[nseg][x] = m[x];
            sFix0[nseg] = cd[0]; sV0[nseg] = vc; sFix1[nseg] = cd[1]; sV1[nseg] = vd;
            ++nseg;
          };
          auto cut = [&](int k, bool toward) {  // segment around corner k; m2 towards the corner if toward
            const int ea = E[adj0[k]], eb = E[adj1[k]];
            Real m[4];
            for (int x = 0; x < 4; ++x) {
              const Real kc = static_cast<Real>((qc[k] >> x) & 1);
              const Real d = kc - 0.5 * (edge_mid(ea, x) + edge_mid(eb, x));
              m[x] = toward ? d : -d;
            }
            add(ea, eb, m);
          };
          if (n == 2) {
            if (cr[0] && cr[2]) {
              Real m[4] = {0, 0, 0, 0};
              m[a] = is_above(qc[0]) ? 1.0 : -1.0;  // the below side is a = 1 if q00 is above
              add(E[0], E[2], m);
            } else if (cr[1] && cr[3]) {
              Real m[4] = {0, 0, 0, 0};
              m[b] = is_above(qc[0]) ? 1.0 : -1.0;
              add(E[1], E[3], m);
            } else {
              for (int k = 0; k < 4; ++k)
                if (cr[adj0[k]] && cr[adj1[k]]) cut(k, !is_above(qc[k]));
            }
          } else if (n == 4) {
            const Real mean = 0.25 * (f[qc[0]] + f[qc[1]] + f[qc[2]] + f[qc[3]]);
            for (int k = 0; k < 4; ++k) {
              if (mean >= 0.0 && !is_above(qc[k])) cut(k, true);
              if (mean < 0.0 && is_above(qc[k])) cut(k, false);
            }
          } else {
            return -1;  // an odd number of crossings cannot occur
          }
        }
    }

  // polygons in the 8 boundary cubes
  int pStart[33], pEdge[96];
  unsigned pMask[32];
  Real pC[32][4];
  int npoly = 0, npe = 0;
  pStart[0] = 0;
  for (int k = 0; k < 4; ++k)
    for (int side = 0; side < 2; ++side) {
      int fa[3], q = 0;
      for (int x = 0; x < 4; ++x)
        if (x != k) fa[q++] = x;
      // boundary orientation factor of this cube (see the header comment of this file and docs/particlisation_format.md)
      const int F = -(side == 1 ? 1 : -1) * ((k & 1) ? -1 : 1);
      int nxt[32];
      for (int e = 0; e < 32; ++e) nxt[e] = -1;
      int nin = 0;
      unsigned cubeMask = 0;
      for (int s = 0; s < nseg; ++s) {
        int l, vl;
        if (sFix0[s] == k && sV0[s] == side) { l = sFix1[s]; vl = sV1[s]; }
        else if (sFix1[s] == k && sV1[s] == side) { l = sFix0[s]; vl = sV0[s]; }
        else continue;
        // tau = m2 x o_s in the free coordinates of the cube (ascending axes), o_s = outward normal of the square in the cube
        Real m[3], o[3], d[3];
        for (int i = 0; i < 3; ++i) {
          m[i] = sM[s][fa[i]];
          o[i] = fa[i] == l ? (vl == 1 ? 1.0 : -1.0) : 0.0;
          d[i] = edge_mid(sB[s], fa[i]) - edge_mid(sA[s], fa[i]);
        }
        const Real t0 = m[1] * o[2] - m[2] * o[1], t1 = m[2] * o[0] - m[0] * o[2], t2 = m[0] * o[1] - m[1] * o[0];
        int from = sA[s], to = sB[s];
        if (d[0] * t0 + d[1] * t1 + d[2] * t2 < 0.0) { from = sB[s]; to = sA[s]; }
        if (F < 0) { const int tt = from; from = to; to = tt; }
        if (nxt[from] >= 0) return -1;
        nxt[from] = to;
        ++nin;
        cubeMask |= 1u << from;
      }
      if (nin == 0) continue;
      // every point must also be the end of exactly one segment
      unsigned toMask = 0;
      for (int e = 0; e < 32; ++e)
        if (nxt[e] >= 0) {
          if ((toMask >> nxt[e]) & 1u) return -1;
          toMask |= 1u << nxt[e];
        }
      if (toMask != cubeMask) return -1;
      unsigned seen = 0;
      for (int e0 = 0; e0 < 32; ++e0) {
        if (nxt[e0] < 0 || ((seen >> e0) & 1u)) continue;
        if (npoly >= 32) return -1;
        Real cx[4] = {0, 0, 0, 0};
        int cnt = 0;
        unsigned mask = 0;
        int e = e0;
        while (!((seen >> e) & 1u)) {
          seen |= 1u << e;
          mask |= 1u << e;
          pEdge[npe++] = e;
          for (int x = 0; x < 4; ++x) cx[x] += P[e][x];
          ++cnt;
          e = nxt[e];
        }
        if (e != e0) return -1;
        for (int x = 0; x < 4; ++x) pC[npoly][x] = cx[x] / cnt;
        pMask[npoly] = mask;
        ++npoly;
        pStart[npoly] = npe;
      }
    }

  // pieces: polygons sharing a crossing point
  int parent[32];
  for (int i = 0; i < npoly; ++i) parent[i] = i;
  auto find = [&](int i) {
    while (parent[i] != i) i = parent[i];
    return i;
  };
  for (int i = 0; i < npoly; ++i)
    for (int j = i + 1; j < npoly; ++j)
      if (pMask[i] & pMask[j]) {
        const int ri = find(i), rj = find(j);
        if (ri != rj) parent[rj < ri ? ri : rj] = rj < ri ? rj : ri;
      }
  const Real Lp = L[0] * L[1] * L[2] * L[3];
  int npieces = 0;
  bool overflow = false;
  for (int r = 0; r < npoly; ++r) {
    if (find(r) != r) continue;
    unsigned mask = 0;
    for (int i = 0; i < npoly; ++i)
      if (find(i) == r) mask |= pMask[i];
    Real C[4] = {0, 0, 0, 0};
    int cnt = 0;
    for (int e = 0; e < 32; ++e)
      if ((mask >> e) & 1u) {
        for (int x = 0; x < 4; ++x) C[x] += P[e][x];
        ++cnt;
      }
    for (int x = 0; x < 4; ++x) C[x] /= cnt;
    Real n[4] = {0, 0, 0, 0}, xw[4] = {0, 0, 0, 0}, wsum = 0.0;
    for (int i = 0; i < npoly; ++i) {
      if (find(i) != r) continue;
      const int b0 = pStart[i], b1 = pStart[i + 1];
      Real va[4];
      for (int x = 0; x < 4; ++x) va[x] = pC[i][x] - C[x];
      for (int q = b0; q < b1; ++q) {
        const int e1 = pEdge[q], e2 = pEdge[q + 1 < b1 ? q + 1 : b0];
        Real vb[4], vc[4], nt[4];
        for (int x = 0; x < 4; ++x) { vb[x] = P[e1][x] - C[x]; vc[x] = P[e2][x] - C[x]; }
        tet_normal(va, vb, vc, nt);
        Real w2 = 0.0;
        for (int x = 0; x < 4; ++x) {
          nt[x] *= Lp / L[x];
          n[x] += nt[x];
          w2 += nt[x] * nt[x];
        }
        const Real w = Kokkos::sqrt(w2);
        for (int x = 0; x < 4; ++x) xw[x] += w * 0.25 * (C[x] + pC[i][x] + P[e1][x] + P[e2][x]);
        wsum += w;
      }
    }
    if (npieces >= max_out) { overflow = true; continue; }
    for (int x = 0; x < 4; ++x) {
      out[npieces].n[x] = n[x];
      out[npieces].xc[x] = wsum > 0.0 ? xw[x] / wsum : C[x];
    }
    ++npieces;
  }
  return overflow ? -2 : npieces;
}

}  // namespace surf
}  // namespace spf
