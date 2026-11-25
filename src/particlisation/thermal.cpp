#include "spf/particlisation/thermal.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

namespace spf::particlisation {

namespace {

struct GL {
  std::vector<double> x, w;
};

GL make_gl(int n) {
  GL g;
  g.x.resize(n);
  g.w.resize(n);
  const double pi = 3.14159265358979323846;
  for (int i = 0; i < (n + 1) / 2; ++i) {
    double z = std::cos(pi * (i + 0.75) / (n + 0.5)), pp = 1.0;
    for (int it = 0; it < 100; ++it) {
      double p1 = 1.0, p2 = 0.0;
      for (int j = 0; j < n; ++j) {
        const double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
      }
      pp = n * (z * p1 - p2) / (z * z - 1.0);
      const double z1 = z;
      z = z1 - p1 / pp;
      if (std::fabs(z - z1) < 1e-16) break;
    }
    // recompute derivative at the converged root
    double p1 = 1.0, p2 = 0.0;
    for (int j = 0; j < n; ++j) {
      const double p3 = p2;
      p2 = p1;
      p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
    }
    pp = n * (z * p1 - p2) / (z * z - 1.0);
    g.x[i] = -z;
    g.x[n - 1 - i] = z;
    g.w[i] = g.w[n - 1 - i] = 2.0 / ((1.0 - z * z) * pp * pp);
  }
  return g;
}

const GL& gl(int n) {
  static const std::map<int, GL> tab = [] {
    std::map<int, GL> t;
    for (int k : {4, 8, 12, 16, 20, 24, 32, 48, 64}) t[k] = make_gl(k);
    return t;
  }();
  auto it = tab.find(n);
  if (it == tab.end()) throw std::invalid_argument("thermal: quad_nodes must be one of 4, 8, 12, 16, 20, 24, 32, 48, 64");
  return it->second;
}

const double kTj[] = {0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 26, 30, 36, 42, 50};
const double kFj[] = {-30, -20, -12, -6, -3, -1.5, 0, 1.5, 3, 6, 12, 20, 30, 45};

}  // namespace

double occupation(Stat s, double x) {
  switch (s) {
    case Stat::boltzmann: return x > 745.0 ? 0.0 : std::exp(-x);
    case Stat::fermi: {
      if (x > 0.0) {
        const double e = std::exp(-x);
        return e / (1.0 + e);
      }
      return 1.0 / (1.0 + std::exp(x));
    }
    case Stat::bose: return x > 745.0 ? 0.0 : 1.0 / std::expm1(x);
  }
  return 0.0;
}

void thermal_edges(double m, double T, double mu, Stat st, std::vector<double>& k_edges) {
  (void)st;
  thread_local std::vector<double> E;
  E.clear();
  const double Eref = std::max(m, mu), Emax = Eref + 50.0 * T;
  for (double t : kTj) {
    const double e = m + T * t;
    if (e <= Emax) E.push_back(e);
  }
  for (double j : kFj) {
    const double e = mu + T * j;
    if (e > m && e < Emax) E.push_back(e);
  }
  E.push_back(Emax);
  std::sort(E.begin(), E.end());
  k_edges.clear();
  k_edges.push_back(0.0);
  double last = m;
  for (double e : E) {
    if (e - last <= 1e-13 * e) continue;
    last = e;
    k_edges.push_back(std::sqrt(std::max(e * e - m * m, 0.0)));
  }
}

Thermal thermal_integrals(double m, double g, double T, double mu, Stat st, int nodes) {
  if (!(m > 0.0) || !(T > 0.0)) throw std::invalid_argument("thermal_integrals: m > 0 and T > 0 required");
  thread_local std::vector<double> ed;
  thermal_edges(m, T, mu, st, ed);
  const GL& q = gl(nodes);
  double sn = 0, se = 0, sp = 0;
  for (size_t p = 0; p + 1 < ed.size(); ++p) {
    const double a = ed[p], b = ed[p + 1], h = 0.5 * (b - a), c = 0.5 * (b + a);
    double pn = 0, pe = 0, pP = 0;
    for (int i = 0; i < nodes; ++i) {
      const double k = c + h * q.x[i], E = std::sqrt(k * k + m * m);
      const double f = occupation(st, (E - mu) / T) * q.w[i];
      const double k2f = k * k * f;
      pn += k2f;
      pe += k2f * E;
      pP += k2f * k * k / E;
    }
    sn += h * pn;
    se += h * pe;
    sp += h * pP;
  }
  const double pre = g / (2.0 * 3.14159265358979323846 * 3.14159265358979323846) / (kHbarcGeVfm * kHbarcGeVfm * kHbarcGeVfm);
  return {pre * sn, pre * se, pre * sp / 3.0};
}

double thermal_density(double m, double g, double T, double mu, Stat st, int nodes) {
  if (!(m > 0.0) || !(T > 0.0)) throw std::invalid_argument("thermal_density: m > 0 and T > 0 required");
  thread_local std::vector<double> ed;
  thermal_edges(m, T, mu, st, ed);
  const GL& q = gl(nodes);
  double sn = 0;
  for (size_t p = 0; p + 1 < ed.size(); ++p) {
    const double a = ed[p], b = ed[p + 1], h = 0.5 * (b - a), c = 0.5 * (b + a);
    double pn = 0;
    for (int i = 0; i < nodes; ++i) {
      const double k = c + h * q.x[i], E = std::sqrt(k * k + m * m);
      pn += k * k * occupation(st, (E - mu) / T) * q.w[i];
    }
    sn += h * pn;
  }
  const double pi = 3.14159265358979323846;
  return g / (2.0 * pi * pi) / (kHbarcGeVfm * kHbarcGeVfm * kHbarcGeVfm) * sn;
}

DensityAcceptance thermal_density_acceptance(double m, double g, double T, double mu, Stat st, int nodes, double d0, double s) {
  DensityAcceptance out;
  const double dmax = std::fabs(d0) + s;
  if (!(dmax > 0.0)) return out;
  if (s <= std::fabs(d0)) {
    out.n = thermal_density(m, g, T, mu, st, nodes);
    out.alpha = std::max(d0, 0.0) / dmax;
    return out;
  }
  thread_local std::vector<double> ed;
  thermal_edges(m, T, mu, st, ed);
  const double r = std::fabs(d0) / s;  // < 1: kink at v = r
  if (r > 0.0) {
    const double ks = m * r / std::sqrt(1.0 - r * r);
    if (ks < ed.back()) ed.insert(std::upper_bound(ed.begin(), ed.end(), ks), ks);
  }
  const GL& q = gl(nodes);
  double sn = 0, sa = 0;
  for (size_t p = 0; p + 1 < ed.size(); ++p) {
    const double a = ed[p], b = ed[p + 1], h = 0.5 * (b - a), c = 0.5 * (b + a);
    double pn = 0, pa = 0;
    for (int i = 0; i < nodes; ++i) {
      const double k = c + h * q.x[i], E = std::sqrt(k * k + m * m);
      const double w = k * k * occupation(st, (E - mu) / T) * q.w[i];
      const double vs = k / E * s;
      const double A = vs <= std::fabs(d0) ? std::max(d0, 0.0) / dmax : (vs + d0) * (vs + d0) / (4.0 * vs * dmax);
      pn += w;
      pa += w * A;
    }
    sn += h * pn;
    sa += h * pa;
  }
  const double pi = 3.14159265358979323846;
  out.n = g / (2.0 * pi * pi) / (kHbarcGeVfm * kHbarcGeVfm * kHbarcGeVfm) * sn;
  out.alpha = sn > 0 ? sa / sn : 0.0;
  return out;
}

double sample_momentum(double m, double T, double mu, Stat st, Rng& rng, int nodes) {
  thread_local std::vector<double> ed, W;
  thermal_edges(m, T, mu, st, ed);
  const GL& q = gl(nodes);
  const size_t np = ed.size() - 1;
  W.assign(np, 0.0);
  double tot = 0;
  for (size_t p = 0; p < np; ++p) {
    const double a = ed[p], b = ed[p + 1], h = 0.5 * (b - a), c = 0.5 * (b + a);
    double s = 0;
    for (int i = 0; i < nodes; ++i) {
      const double k = c + h * q.x[i], E = std::sqrt(k * k + m * m);
      s += k * k * occupation(st, (E - mu) / T) * q.w[i];
    }
    W[p] = h * s;
    tot += W[p];
  }
  if (!(tot > 0.0)) return 0.5 * (ed[0] + ed.back());
  const double r = rng.uniform() * tot;
  double acc = 0;
  size_t pick = np - 1;
  for (size_t p = 0; p < np; ++p) {
    acc += W[p];
    if (r < acc) {
      pick = p;
      break;
    }
  }
  const double kL = ed[pick], kR = ed[pick + 1];
  const double EL = std::sqrt(kL * kL + m * m);
  const double bound = kR * kR * occupation(st, (EL - mu) / T);
  for (int it = 0; it < 100000; ++it) {
    const double k = kL + (kR - kL) * rng.uniform();
    const double E = std::sqrt(k * k + m * m);
    if (rng.uniform() * bound <= k * k * occupation(st, (E - mu) / T)) return k;
  }
  return 0.5 * (kL + kR);
}

MuSolve solve_mu_net_density(double m, double g, double T, double rho, Stat st, int nodes, double mu_guess, double rel_tol) {
  MuSolve r;
  if (rho == 0.0) { r.status = 1; return r; }
  r.mu = mu_guess;
  const double sgn = rho > 0 ? 1.0 : -1.0, target = std::fabs(rho);
  auto net = [&](double mu) { return thermal_density(m, g, T, mu, st, nodes) - thermal_density(m, g, T, -mu, st, nodes) - target; };
  // net(0) = -target < 0 and net increases with mu: bracket the root in [lo, hi], lo >= 0.
  double hi = std::max(std::fabs(mu_guess), 1e-3), step = std::max(T, 1e-3);
  double lo = 0.0, flo = -target, fhi = net(hi);
  int it = 0;
  while (fhi < 0.0 && it < 200) {
    lo = hi; flo = fhi;
    hi += step; step *= 2.0;
    fhi = net(hi);
    ++it;
  }
  if (!std::isfinite(fhi) || fhi < 0.0) { r.status = 2; return r; }
  double a = lo, b = hi, fa = flo, fb = fhi, c = a, fc = fa, d = b - a, e = d;
  for (int k = 0; k < 200; ++k) {
    ++it;
    if (fb == 0.0 || std::fabs(fb) <= rel_tol * target) break;
    if ((fb > 0) == (fc > 0)) { c = a; fc = fa; d = b - a; e = d; }
    if (std::fabs(fc) < std::fabs(fb)) { a = b; b = c; c = a; fa = fb; fb = fc; fc = fa; }
    const double tol1 = 2.0 * 2.2e-16 * std::fabs(b) + 1e-16, xm = 0.5 * (c - b);
    if (std::fabs(xm) <= tol1) break;
    if (std::fabs(e) >= tol1 && std::fabs(fa) > std::fabs(fb)) {
      const double sdiv = fb / fa;
      double pp, q;
      if (a == c) { pp = 2.0 * xm * sdiv; q = 1.0 - sdiv; }
      else {
        const double qq = fa / fc, rr = fb / fc;
        pp = sdiv * (2.0 * xm * qq * (qq - rr) - (b - a) * (rr - 1.0));
        q = (qq - 1.0) * (rr - 1.0) * (sdiv - 1.0);
      }
      if (pp > 0) q = -q;
      pp = std::fabs(pp);
      if (2.0 * pp < std::min(3.0 * xm * q - std::fabs(tol1 * q), std::fabs(e * q))) { e = d; d = pp / q; }
      else { d = xm; e = d; }
    } else { d = xm; e = d; }
    a = b; fa = fb;
    b += std::fabs(d) > tol1 ? d : (xm > 0 ? tol1 : -tol1);
    fb = net(b);
  }
  r.mu = sgn * b;
  r.rel_residual = fb / target;
  r.iterations = it;
  r.status = std::isfinite(fb) ? 0 : 2;
  return r;
}

}  // namespace spf::particlisation
