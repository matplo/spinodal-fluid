#include "spf/initial_state.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>

namespace spf {

// ------------------------------------------------------------------ PDG rule
int baryon_number_from_pdg(long pdg) {
  const long a = pdg < 0 ? -pdg : pdg;
  const int sign = pdg < 0 ? -1 : 1;
  if (a >= 1000000000L) return sign * static_cast<int>((a / 10) % 1000);
  const long nq1 = (a / 1000) % 10, nq2 = (a / 100) % 10, nq3 = (a / 10) % 10;
  if (nq1 > 0 && nq2 > 0 && nq3 > 0) return sign;
  return 0;
}

// ------------------------------------------------------------------ OSCAR2013
namespace {

std::vector<std::string> split_ws(const std::string& s) {
  std::istringstream is(s);
  std::vector<std::string> out;
  std::string w;
  while (is >> w) out.push_back(w);
  return out;
}

}  // namespace

OscarFile read_oscar2013(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open OSCAR file '" + path + "'");
  OscarFile f;
  std::map<std::string, int> col;
  const char* dflt[] = {"t", "x", "y", "z", "mass", "p0", "px", "py", "pz", "pdg", "ID", "charge"};
  for (int i = 0; i < 12; ++i) col[dflt[i]] = i;
  std::string line;
  long lineno = 0;
  OscarBlock* cur = nullptr;
  while (std::getline(in, line)) {
    ++lineno;
    if (line.empty()) continue;
    if (line.rfind("#!OSCAR2013", 0) == 0) {
      f.format_line = line;
      const auto w = split_ws(line);
      // "#!OSCAR2013 particle_lists [<code> <version>] t x y z ..." : the column names start at the first known column name
      size_t k0 = 0;
      for (size_t i = 2; i < w.size(); ++i)
        if (w[i] == "t" || w[i] == "x" || w[i] == "pdg" || w[i] == "p0" || w[i] == "mass" || w[i] == "ID") { k0 = i; break; }
      if (k0 > 0) {
        col.clear();
        for (size_t i = k0; i < w.size(); ++i) col[w[i]] = static_cast<int>(i - k0);
      }
      continue;
    }
    if (line[0] == '#') {
      const auto w = split_ws(line.substr(1));
      // "event N [ensemble E] in|out M" opens a block, "event N [ensemble E] end 0 impact b ..." closes the event
      size_t k = 2;
      if (w.size() > 4 && w[0] == "event" && w[2] == "ensemble") k = 4;
      if (w.size() >= k + 2 && w[0] == "event" && (w[k] == "out" || w[k] == "in")) {
        f.blocks.emplace_back();
        cur = &f.blocks.back();
        cur->event = std::stoi(w[1]);
        cur->kind = w[k];
        cur->n_declared = std::stol(w[k + 1]);
      } else if (w.size() >= k + 1 && w[0] == "event" && w[k] == "end") {
        cur = nullptr;
        for (size_t i = k + 1; i + 1 < w.size(); ++i)
          if (w[i] == "impact") { f.impact_parameter.push_back(std::stod(w[i + 1])); break; }
      }
      continue;
    }
    if (!cur) throw std::runtime_error(path + ":" + std::to_string(lineno) + ": particle line outside an event block");
    const auto w = split_ws(line);
    for (const char* need : {"t", "x", "y", "z", "p0", "px", "py", "pz", "pdg"})
      if (!col.count(need)) throw std::runtime_error(path + ": column '" + need + "' missing in the OSCAR2013 header");
    const int nmax = std::max({col["t"], col["x"], col["y"], col["z"], col["p0"], col["px"], col["py"], col["pz"], col["pdg"]});
    if (static_cast<int>(w.size()) <= nmax) throw std::runtime_error(path + ":" + std::to_string(lineno) + ": too few columns");
    Particle p;
    try {
      p.t = std::stod(w[col["t"]]); p.x = std::stod(w[col["x"]]); p.y = std::stod(w[col["y"]]); p.z = std::stod(w[col["z"]]);
      p.E = std::stod(w[col["p0"]]); p.px = std::stod(w[col["px"]]); p.py = std::stod(w[col["py"]]); p.pz = std::stod(w[col["pz"]]);
      p.pdg = std::stol(w[col["pdg"]]);
    } catch (const std::exception&) {
      throw std::runtime_error(path + ":" + std::to_string(lineno) + ": malformed particle line");
    }
    p.B = baryon_number_from_pdg(p.pdg);
    if (cur->particles.empty()) cur->time = p.t;
    cur->particles.push_back(p);
  }
  for (const auto& b : f.blocks)
    if (b.n_declared >= 0 && b.n_declared != static_cast<long>(b.particles.size()))
      throw std::runtime_error(path + ": block of event " + std::to_string(b.event) + " declares " + std::to_string(b.n_declared) +
                               " particles, found " + std::to_string(b.particles.size()));
  return f;
}

const OscarBlock& find_oscar_block(const OscarFile& f, int event, Real t_target) {
  const OscarBlock* best = nullptr;
  for (const auto& b : f.blocks) {
    if (b.event != event || b.kind != "out") continue;
    if (t_target < 0.0) best = &b;
    else if (!best || std::fabs(b.time - t_target) < std::fabs(best->time - t_target)) best = &b;
  }
  if (!best) throw std::runtime_error("no 'out' particle block for event " + std::to_string(event) + " in the OSCAR file");
  return *best;
}

// ------------------------------------------------------------------ kinematics and Woods-Saxon
Kinematics kinematics_from_elab(Real e_lab, Real m_n) {
  Kinematics k;
  k.e_lab = e_lab;
  k.gamma_lab = 1.0 + e_lab / m_n;
  k.gamma_cm = std::sqrt(0.5 * (k.gamma_lab + 1.0));
  k.sqrt_snn = 2.0 * k.gamma_cm * m_n;
  k.beta_cm = std::sqrt(1.0 - 1.0 / (k.gamma_cm * k.gamma_cm));
  return k;
}

Real t_init_fm(const Kinematics& k, Real radius_fm) { return 2.0 * radius_fm / std::sqrt(k.gamma_cm * k.gamma_cm - 1.0); }

Real t_init_radius_from_config(const Config& cfg, int mass_number) {
  const std::string m = cfg.get<std::string>("ic.t_init_radius_mode");
  if (m == "r0_A13") return cfg.get<Real>("ic.t_init_r0_fm") * std::cbrt(static_cast<Real>(mass_number));
  if (m == "fixed") return cfg.get<Real>("ic.t_init_radius_fm");
  throw ConfigError("ic.t_init_radius_mode must be 'r0_A13' or 'fixed', got '" + m + "'");
}

namespace {
inline Real ws_f(Real r, Real R, Real a) { return 1.0 / (1.0 + std::exp((r - R) / a)); }
inline Real unit_uniform(std::mt19937_64& g) { return static_cast<Real>(g() >> 11) * (1.0 / 9007199254740992.0); }
}  // namespace

Real woods_saxon_rho0(int A, Real R, Real a) {
  const Real rmax = R + 20.0 * a;
  const int n = 20000;  // Simpson, even
  const Real h = rmax / n;
  Real s = 0.0;
  for (int i = 0; i <= n; ++i) {
    const Real r = i * h;
    const Real w = (i == 0 || i == n) ? 1.0 : (i % 2 ? 4.0 : 2.0);
    s += w * r * r * ws_f(r, R, a);
  }
  s *= h / 3.0 * 4.0 * M_PI;
  return static_cast<Real>(A) / s;
}

std::vector<std::array<Real, 3>> sample_woods_saxon(int A, Real R, Real a, std::uint64_t seed) {
  std::mt19937_64 g(seed);
  const Real rmax = R + 10.0 * a;
  Real fmax = 0.0;
  for (int i = 1; i <= 4000; ++i) {
    const Real r = rmax * i / 4000.0;
    fmax = std::max(fmax, r * r * ws_f(r, R, a));
  }
  fmax *= 1.02;
  std::vector<std::array<Real, 3>> out;
  while (static_cast<int>(out.size()) < A) {
    const Real r = rmax * unit_uniform(g);
    if (unit_uniform(g) * fmax > r * r * ws_f(r, R, a)) continue;
    const Real ct = 2.0 * unit_uniform(g) - 1.0, ph = 2.0 * M_PI * unit_uniform(g);
    const Real st = std::sqrt(1.0 - ct * ct);
    out.push_back({r * st * std::cos(ph), r * st * std::sin(ph), r * ct});
  }
  return out;
}

std::vector<Particle> glauber_particles(const Kinematics& k, int A, Real R, Real a, std::uint64_t seed, Real m_n) {
  std::vector<Particle> out;
  for (int nuc = 0; nuc < 2; ++nuc) {
    const auto pos = sample_woods_saxon(A, R, a, seed * 2 + static_cast<std::uint64_t>(nuc));
    const Real sgn = nuc == 0 ? 1.0 : -1.0;
    for (const auto& r : pos) {
      Particle p;
      p.x = r[0]; p.y = r[1]; p.z = r[2] / k.gamma_cm;
      p.E = k.gamma_cm * m_n;
      p.pz = sgn * k.gamma_cm * k.beta_cm * m_n;
      p.pdg = 2212;
      p.B = 1;
      out.push_back(p);
    }
  }
  return out;
}

// ------------------------------------------------------------------ deposition kernels
Contraction parse_contraction(const std::string& s) {
  if (s == "none") return Contraction::None;
  if (s == "particle_gamma_z") return Contraction::ParticleGammaZ;
  if (s == "cm_gamma") return Contraction::CmGamma;
  throw ConfigError("ic.gaussian_contraction must be 'none', 'particle_gamma_z' or 'cm_gamma', got '" + s + "'");
}

namespace detail {

struct ParticleViews {
  Kokkos::View<Real*> x, y, z, E, px, py, pz, B;
};

struct DepositK {
  using Team = Kokkos::TeamPolicy<ExecSpace>::member_type;
  ParticleViews P;
  Field3 R, Mx, My, Mz, E;
  Kokkos::View<int*> status;  // 0 inside, 1 clipped, 2 empty stencil
  Real sigma, cutoff, dx, gamma_cm, ox, oy, oz;
  int contraction, nx, ny, nz, gx, gy, gz;

  KOKKOS_INLINE_FUNCTION void operator()(const Team& t) const {
    const int ip = t.league_rank();
    const Real xp = P.x(ip), yp = P.y(ip), zp = P.z(ip);
    Real gz_ = 1.0;
    if (contraction == 1) {
      const Real e = P.E(ip), pz = P.pz(ip);
      const Real bz = e > 0.0 ? pz / e : 0.0;
      const Real b2 = bz * bz < 1.0 ? bz * bz : 1.0 - 1e-16;
      gz_ = 1.0 / Kokkos::sqrt(1.0 - b2);
    } else if (contraction == 2) {
      gz_ = gamma_cm;
    }
    const Real sx = sigma, sz = sigma / gz_;
    const Real rx = cutoff * sx, rz = cutoff * sz;
    // stencil: grid points (cell centres, possibly outside the box) within the cutoff, per axis
    int ia = static_cast<int>(Kokkos::ceil((xp - rx - ox) / dx)), ib = static_cast<int>(Kokkos::floor((xp + rx - ox) / dx));
    int ja = static_cast<int>(Kokkos::ceil((yp - rx - oy) / dx)), jb = static_cast<int>(Kokkos::floor((yp + rx - oy) / dx));
    int ka = static_cast<int>(Kokkos::ceil((zp - rz - oz) / dx)), kb = static_cast<int>(Kokkos::floor((zp + rz - oz) / dx));
    // A Gaussian narrower than the cutoff window holds no grid point: use the nearest grid point alone (all of the particle's
    // B, E, p go there; counted in n_particles_empty as "single-cell fallback" - the particle is not resolved by the grid).
    int fallback = 0;
    if (ib < ia) { ia = ib = static_cast<int>(Kokkos::floor((xp - ox) / dx + 0.5)); fallback = 1; }
    if (jb < ja) { ja = jb = static_cast<int>(Kokkos::floor((yp - oy) / dx + 0.5)); fallback = 1; }
    if (kb < ka) { ka = kb = static_cast<int>(Kokkos::floor((zp - oz) / dx + 0.5)); fallback = 1; }
    const int mx = ib - ia + 1, my = jb - ja + 1, mz = kb - ka + 1;
    Real Sx = 0.0, Sy = 0.0, Sz = 0.0;
    for (int i = ia; i <= ib; ++i) { const Real d = ox + i * dx - xp; Sx += Kokkos::exp(-0.5 * d * d / (sx * sx)); }
    for (int j = ja; j <= jb; ++j) { const Real d = oy + j * dx - yp; Sy += Kokkos::exp(-0.5 * d * d / (sx * sx)); }
    for (int k = ka; k <= kb; ++k) { const Real d = oz + k * dx - zp; Sz += Kokkos::exp(-0.5 * d * d / (sz * sz)); }
    const Real norm = 1.0 / (Sx * Sy * Sz * dx * dx * dx);
    const Real b = P.B(ip) * norm, e = P.E(ip) * norm, qx = P.px(ip) * norm, qy = P.py(ip) * norm, qz = P.pz(ip) * norm;
    if (t.team_rank() == 0) {
      const bool clipped = ia < 0 || ib >= nx || ja < 0 || jb >= ny || ka < 0 || kb >= nz;
      status(ip) = clipped ? 1 : (fallback ? 2 : 0);
    }
    const long ncell = static_cast<long>(mx) * my * mz;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(t, ncell), [&](const long c) {
      const int k = static_cast<int>(c % mz);
      const int j = static_cast<int>((c / mz) % my);
      const int i = static_cast<int>(c / (static_cast<long>(mz) * my));
      const int ci = ia + i, cj = ja + j, ck = ka + k;
      if (ci < 0 || ci >= nx || cj < 0 || cj >= ny || ck < 0 || ck >= nz) return;
      const Real ddx = ox + ci * dx - xp, ddy = oy + cj * dx - yp, ddz = oz + ck * dx - zp;
      const Real w = Kokkos::exp(-0.5 * (ddx * ddx + ddy * ddy) / (sx * sx) - 0.5 * ddz * ddz / (sz * sz));
      const int ii = gx + ci, jj = gy + cj, kk = gz + ck;
      if (b != 0.0) Kokkos::atomic_add(&R(ii, jj, kk), b * w);
      Kokkos::atomic_add(&E(ii, jj, kk), e * w);
      Kokkos::atomic_add(&Mx(ii, jj, kk), qx * w);
      Kokkos::atomic_add(&My(ii, jj, kk), qy * w);
      Kokkos::atomic_add(&Mz(ii, jj, kk), qz * w);
    });
  }
};

struct StatusCountK {
  Kokkos::View<int*> s;
  int which;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, long& acc) const { acc += (s(i) == which); }
};

struct LimitK {
  Field3 Mx, My, Mz, E;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k, long& acc) const {
    const Real mx = Mx(i, j, k), my = My(i, j, k), mz = Mz(i, j, k), e = E(i, j, k);
    const Real m = Kokkos::sqrt(mx * mx + my * my + mz * mz);
    if (m > e) {
      const Real f = e / m;
      Mx(i, j, k) = mx * f; My(i, j, k) = my * f; Mz(i, j, k) = mz * f;
      acc += 1;
    }
  }
};

struct WsFieldK {
  Field3 R, E;
  Real ox, oy, oz, dx, gamma, rho0, rad, a, mass;
  int gx, gy, gz;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, const int j, const int k) const {
    const Real x = ox + (i - gx) * dx, y = oy + (j - gy) * dx, z = oz + (k - gz) * dx;
    const Real r = Kokkos::sqrt(x * x + y * y + gamma * gamma * z * z);
    const Real f = rho0 / (1.0 + Kokkos::exp((r - rad) / a));
    R(i, j, k) = 2.0 * gamma * f;
    E(i, j, k) = 2.0 * gamma * gamma * mass * f;
  }
};

}  // namespace detail

Totals5 grid_totals(const Grid& grid, const DepositResult& d) {
  const Real v = grid.dx * grid.dx * grid.dx;
  Totals5 t;
  t.B = field_sum(grid, d.R) * v;
  t.E = field_sum(grid, d.E) * v;
  t.Px = field_sum(grid, d.Mx) * v;
  t.Py = field_sum(grid, d.My) * v;
  t.Pz = field_sum(grid, d.Mz) * v;
  return t;
}

DepositResult deposit_particles(const Grid& grid, const DepositParams& prm, const std::vector<Particle>& particles) {
  DepositResult res;
  res.R = grid.make_field("ic_R"); res.Mx = grid.make_field("ic_Mx"); res.My = grid.make_field("ic_My");
  res.Mz = grid.make_field("ic_Mz"); res.E = grid.make_field("ic_E");
  const int n = static_cast<int>(particles.size());
  res.n_particles = n;
  Totals5& ps = res.particles;
  for (const auto& p : particles) {
    ps.B += p.B; ps.E += p.E; ps.Px += p.px; ps.Py += p.py; ps.Pz += p.pz;
    ps.absPx += std::fabs(p.px); ps.absPy += std::fabs(p.py); ps.absPz += std::fabs(p.pz);
  }
  if (n > 0) {
    detail::ParticleViews P;
    P.x = Kokkos::View<Real*>("px_x", n); P.y = Kokkos::View<Real*>("px_y", n); P.z = Kokkos::View<Real*>("px_z", n);
    P.E = Kokkos::View<Real*>("px_E", n); P.px = Kokkos::View<Real*>("px_px", n); P.py = Kokkos::View<Real*>("px_py", n);
    P.pz = Kokkos::View<Real*>("px_pz", n); P.B = Kokkos::View<Real*>("px_B", n);
    {
      auto hx = Kokkos::create_mirror_view(P.x), hy = Kokkos::create_mirror_view(P.y), hz = Kokkos::create_mirror_view(P.z);
      auto hE = Kokkos::create_mirror_view(P.E), hpx = Kokkos::create_mirror_view(P.px), hpy = Kokkos::create_mirror_view(P.py);
      auto hpz = Kokkos::create_mirror_view(P.pz), hB = Kokkos::create_mirror_view(P.B);
      for (int i = 0; i < n; ++i) {
        const auto& p = particles[i];
        hx(i) = p.x; hy(i) = p.y; hz(i) = p.z; hE(i) = p.E; hpx(i) = p.px; hpy(i) = p.py; hpz(i) = p.pz; hB(i) = p.B;
      }
      Kokkos::deep_copy(P.x, hx); Kokkos::deep_copy(P.y, hy); Kokkos::deep_copy(P.z, hz); Kokkos::deep_copy(P.E, hE);
      Kokkos::deep_copy(P.px, hpx); Kokkos::deep_copy(P.py, hpy); Kokkos::deep_copy(P.pz, hpz); Kokkos::deep_copy(P.B, hB);
    }
    Kokkos::View<int*> status("dep_status", n);
    detail::DepositK k{P, res.R, res.Mx, res.My, res.Mz, res.E, status, prm.sigma, prm.cutoff, grid.dx, prm.gamma_cm,
                       prm.origin[0], prm.origin[1], prm.origin[2], static_cast<int>(prm.contraction), grid.n[0], grid.n[1], grid.n[2],
                       grid.g[0], grid.g[1], grid.g[2]};
    Kokkos::parallel_for("spf_deposit", Kokkos::TeamPolicy<ExecSpace>(n, Kokkos::AUTO), k);
    Kokkos::fence();
    Kokkos::parallel_reduce("spf_dep_clipped", Kokkos::RangePolicy<ExecSpace>(0, n), detail::StatusCountK{status, 1}, res.n_particles_clipped);
    Kokkos::parallel_reduce("spf_dep_empty", Kokkos::RangePolicy<ExecSpace>(0, n), detail::StatusCountK{status, 2}, res.n_particles_empty);
  }
  res.grid = grid_totals(grid, res);
  res.outside_fraction_B = ps.B != 0.0 ? 1.0 - res.grid.B / ps.B : 0.0;
  res.outside_fraction_E = ps.E != 0.0 ? 1.0 - res.grid.E / ps.E : 0.0;
  return res;
}

long limit_momentum(const Grid& grid, DepositResult& d) {
  long n = 0;
  Kokkos::parallel_reduce("spf_ic_limit", interior_policy(grid), detail::LimitK{d.Mx, d.My, d.Mz, d.E}, n);
  d.grid = grid_totals(grid, d);
  return n;
}

DepositResult analytic_counterstreaming(const Grid& grid, const Kinematics& k, int A, Real radius_fm, Real a_fm, Real m_n) {
  DepositResult res;
  res.R = grid.make_field("ic_R"); res.Mx = grid.make_field("ic_Mx"); res.My = grid.make_field("ic_My");
  res.Mz = grid.make_field("ic_Mz"); res.E = grid.make_field("ic_E");
  const Real rho0 = woods_saxon_rho0(A, radius_fm, a_fm);
  const Real ox = -0.5 * (grid.n[0] - 1) * grid.dx, oy = -0.5 * (grid.n[1] - 1) * grid.dx, oz = -0.5 * (grid.n[2] - 1) * grid.dx;
  Kokkos::parallel_for("spf_ic_ws", full_policy(grid),
                       detail::WsFieldK{res.R, res.E, ox, oy, oz, grid.dx, k.gamma_cm, rho0, radius_fm, a_fm, m_n, grid.g[0], grid.g[1], grid.g[2]});
  Kokkos::fence();
  // ghost cells were filled by the full policy: clear them so that only the interior carries the state
  {
    auto hR = Kokkos::create_mirror_view(res.R), hE = Kokkos::create_mirror_view(res.E);
    Kokkos::deep_copy(hR, res.R); Kokkos::deep_copy(hE, res.E);
    for (int i = 0; i < grid.size(0); ++i)
      for (int j = 0; j < grid.size(1); ++j)
        for (int l = 0; l < grid.size(2); ++l) {
          const bool in = i >= grid.g[0] && i < grid.g[0] + grid.n[0] && j >= grid.g[1] && j < grid.g[1] + grid.n[1] && l >= grid.g[2] &&
                          l < grid.g[2] + grid.n[2];
          if (!in) { hR(i, j, l) = 0.0; hE(i, j, l) = 0.0; }
        }
    Kokkos::deep_copy(res.R, hR); Kokkos::deep_copy(res.E, hE);
  }
  res.grid = grid_totals(grid, res);
  res.particles = res.grid;
  return res;
}

}  // namespace spf

namespace spf {

InitialState build_initial_state(const Config& cfg, const Grid& grid) {
  InitialState st;
  st.generator = cfg.get<std::string>("ic.generator");
  const int A = cfg.get<int>("ic.nucleus_A");
  const Real mN = cfg.get<Real>("constants.m_N_GeV");
  st.kin = kinematics_from_elab(cfg.get<Real>("ic.e_lab_gev"), mN);
  st.radius_fm = t_init_radius_from_config(cfg, A);
  st.t_init_fm = t_init_fm(st.kin, st.radius_fm);
  DepositParams dp;
  dp.sigma = cfg.get<Real>("ic.sigma_init_fm");
  dp.cutoff = cfg.get<Real>("ic.cutoff_sigma");
  dp.contraction = parse_contraction(cfg.get<std::string>("ic.gaussian_contraction"));
  dp.gamma_cm = st.kin.gamma_cm;
  const auto o = centred_origin(grid);
  dp.origin[0] = o[0]; dp.origin[1] = o[1]; dp.origin[2] = o[2];
  std::vector<Particle> parts;
  if (st.generator == "smash_cascade" || st.generator == "oscar_file") {
    const std::string path = cfg.get<std::string>("ic.oscar_path");
    if (path.empty()) throw std::runtime_error("ic.generator = '" + st.generator + "' needs ic.oscar_path");
    const OscarFile f = read_oscar2013(path);
    // time of the block: ic.oscar_time_fm if >= 0; otherwise, for smash_cascade, the block nearest to t_init + ic.smash_time_offset_fm
    // (SMASH writes the particle list at the output time and once more at the end time); for oscar_file the last "out" block.
    Real t_sel = cfg.get<Real>("ic.oscar_time_fm");
    if (t_sel < 0.0 && st.generator == "smash_cascade") t_sel = st.t_init_fm + cfg.get<Real>("ic.smash_time_offset_fm");
    const OscarBlock& b = find_oscar_block(f, cfg.get<int>("ic.oscar_event"), t_sel);
    parts = b.particles;
    st.oscar_time_fm = b.time;
  } else if (st.generator == "glauber_gaussians") {
    parts = glauber_particles(st.kin, A, cfg.get<Real>("ic.ws_radius_fm"), cfg.get<Real>("ic.ws_a_fm"),
                              cfg.get<std::uint64_t>("ic.seed"), mN);
  } else if (st.generator != "analytic") {
    throw ConfigError("unknown ic.generator '" + st.generator + "'");
  }
  if (st.generator == "analytic") {
    st.dep = analytic_counterstreaming(grid, st.kin, A, cfg.get<Real>("ic.ws_radius_fm"), cfg.get<Real>("ic.ws_a_fm"), mN);
  } else {
    st.n_hadrons = static_cast<long>(parts.size());
    for (const auto& p : parts) {
      if (p.B == 0) continue;
      st.baryon_total += p.B;
      const Real d = p.E - p.pz;
      if (d > 0.0 && p.E + p.pz > 0.0 && std::fabs(0.5 * std::log((p.E + p.pz) / d)) < 0.5) st.baryon_y_window += p.B;
    }
    st.dep = deposit_particles(grid, dp, parts);
    const Real lim = cfg.get<Real>("ic.max_outside_fraction");
    if (std::fabs(st.dep.outside_fraction_B) > lim)
      throw std::runtime_error("baryon number outside the box: fraction " + std::to_string(st.dep.outside_fraction_B) +
                               " exceeds ic.max_outside_fraction; enlarge the box (grid.nx/ny/nz, grid.dx_fm)");
  }
  st.limiter_cells = limit_momentum(grid, st.dep);
  return st;
}

}  // namespace spf
