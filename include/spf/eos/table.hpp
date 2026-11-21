#pragma once
// Tabulated equation of state p0(eps, rho) with T, mu, s (docs/eos_table_format.md).
//
// Lattice: x = eps / eps_s, y = rho / rho_s, eps_s = m_N rho_s, uniform in both: x_i = x0 + i dx (i < nx), y_j = y0 + j dy (j < ny).
// Fields are stored row-major, index = ix * ny + iy (iy fastest), plus eps_min(rho_j) [GeV/fm^3], the cold curve.
//
// Interpolation (config key eos.table_interpolation):
//   bilinear  value bilinear in the cell; derivatives are those of the bilinear polynomial of the cell that contains the point.
//             p0 is continuous, dp_deps and dp_drho jump at cell edges (a point exactly on an edge uses the cell to its upper side,
//             except on the upper lattice edge, which uses the last cell).
//   bicubic   bicubic Hermite patch per cell (equivalent to tensor-product Catmull-Rom) with node values f, f_x, f_y, f_xy,
//             where f_x, f_y are centred differences of the nodes (one-sided at the lattice boundary) and f_xy is the centred
//             difference of f_y along x. p0, dp_deps and dp_drho are continuous (C1); the second derivatives jump at cell edges.
// Below the cold curve (eps < eps_min(rho)) every field is evaluated at eps = eps_min(rho) (same clamp as EosColdThermal), dp_deps = 0 and
// dp_drho is the derivative of that clamped function, p_y + p_x * eps_min'(rho).  eps_min(rho) is interpolated in rho with the same rule
// (linear or 1D Hermite).  Outside the lattice (eos.extrapolate):
//   false  arguments are clamped to the lattice, the derivative in a clamped direction is zero; EosTable::check_range reports the
//          violation on the host so that a driver can stop with an error;
//   true   linear extrapolation from the edge cell: value at the clamped point plus gradient times the distance.
// A position within 1e-10 cell units of a lattice node is moved onto the node (nodes are reproduced to rounding).
// Everything is host- and device-callable; the table data live in Views owned by EosTableData, which must outlive the kernels.
#include "spf/config.hpp"
#include "spf/eos/concept.hpp"
#include "spf/eos/model.hpp"
#include "spf/sha256.hpp"

#include <nlohmann/json.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace spf {

class EosTableError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

enum class TableInterp { bilinear, bicubic };

inline TableInterp parse_table_interp(const std::string& s) {
  if (s == "bilinear") return TableInterp::bilinear;
  if (s == "bicubic") return TableInterp::bicubic;
  throw ConfigError("unknown eos.table_interpolation '" + s + "'");
}

inline bool eos_model_uses_table(EosModel m) { return m == EosModel::table_two_phase || m == EosModel::table_maxwell; }

// Result of EosTable::check_range.
struct RangeCheck {
  bool ok = true;
  bool eps_low = false, eps_high = false, rho_low = false, rho_high = false;
  Real eps_lo_lattice = 0, eps_hi_lattice = 0, rho_lo_lattice = 0, rho_hi_lattice = 0;  // lattice limits in physical units
  std::string message() const {
    if (ok) return "inside the lattice";
    std::ostringstream o;
    o << "EoS table: requested range outside the lattice (eps in [" << eps_lo_lattice << ", " << eps_hi_lattice << "] GeV/fm^3, rho in ["
      << rho_lo_lattice << ", " << rho_hi_lattice << "] fm^-3):";
    if (eps_low) o << " eps below the lattice;";
    if (eps_high) o << " eps above the lattice;";
    if (rho_low) o << " rho below the lattice;";
    if (rho_high) o << " rho above the lattice;";
    return o.str();
  }
};

struct EosTable {
  const Real* p_tab = nullptr;
  const Real* T_tab = nullptr;
  const Real* mu_tab = nullptr;
  const Real* s_tab = nullptr;
  const Real* emin_tab = nullptr;
  int nx = 0, ny = 0;
  Real x0 = 0.0, dx = 1.0, y0 = 0.0, dy = 1.0;
  Real rho_s = 0.153, eps_s = 0.94 * 0.153;
  int bicubic = 0;      // 1: bicubic Hermite, 0: bilinear
  int extrapolate = 0;  // 1: linear extrapolation outside the lattice

  static const char* name() { return "table"; }

  // Lattice limits in physical units.
  Real eps_lattice_min() const { return x0 * eps_s; }
  Real eps_lattice_max() const { return (x0 + (nx - 1) * dx) * eps_s; }
  Real rho_lattice_min() const { return y0 * rho_s; }
  Real rho_lattice_max() const { return (y0 + (ny - 1) * dy) * rho_s; }

  // Host side: does the box [eps_min, eps_max] x [rho_min, rho_max] lie inside the lattice?  (Energy densities below the cold curve
  // are legal: they are clamped.)  A driver stops with an error when extrapolate is false and !ok.
  RangeCheck check_range(Real eps_min, Real eps_max, Real rho_min, Real rho_max) const {
    RangeCheck r;
    r.eps_lo_lattice = eps_lattice_min();
    r.eps_hi_lattice = eps_lattice_max();
    r.rho_lo_lattice = rho_lattice_min();
    r.rho_hi_lattice = rho_lattice_max();
    r.eps_low = eps_min < r.eps_lo_lattice;
    r.eps_high = eps_max > r.eps_hi_lattice;
    r.rho_low = rho_min < r.rho_lo_lattice;
    r.rho_high = rho_max > r.rho_hi_lattice;
    r.ok = !(r.eps_low || r.eps_high || r.rho_low || r.rho_high);
    return r;
  }

  KOKKOS_INLINE_FUNCTION Real eps_min_of_rho(Real rho) const {
    Real em, dem;
    const Loc L = locate(0.0, rho);  // eps is irrelevant for the cold curve
    cold_curve(rho, L, em, dem);
    return em;
  }

  KOKKOS_INLINE_FUNCTION Real p0(Real eps, Real rho) const {
    const Loc L = locate(eps, rho);
    Real v, vx, vy;
    eval(p_tab, L, v, vx, vy);
    return v;
  }
  KOKKOS_INLINE_FUNCTION Real dp_deps(Real eps, Real rho) const {
    const Loc L = locate(eps, rho);
    if (L.clamped) return 0.0;
    Real v, vx, vy;
    eval(p_tab, L, v, vx, vy);
    return vx / (dx * eps_s);
  }
  KOKKOS_INLINE_FUNCTION Real dp_drho(Real eps, Real rho) const {
    const Loc L = locate(eps, rho);
    Real v, vx, vy;
    eval(p_tab, L, v, vx, vy);
    Real r = vy / (dy * rho_s);
    if (L.clamped) r += vx / (dx * eps_s) * L.emin_slope;
    return r;
  }
  KOKKOS_INLINE_FUNCTION Real vs2(Real eps, Real rho) const {
    const Loc L = locate(eps, rho);
    Real v, vx, vy;
    eval(p_tab, L, v, vx, vy);
    const Real px = vx / (dx * eps_s);
    const Real a = L.clamped ? 0.0 : px;
    const Real b = vy / (dy * rho_s) + (L.clamped ? px * L.emin_slope : 0.0);
    const Real h = eps + v;
    return h > 0.0 ? a + rho / h * b : a;
  }
  // Temperature [GeV], chemical potential [GeV] and entropy density [fm^-3] by the same interpolation and clamp.
  KOKKOS_INLINE_FUNCTION Real T(Real eps, Real rho) const { return field(T_tab, eps, rho); }
  KOKKOS_INLINE_FUNCTION Real mu(Real eps, Real rho) const { return field(mu_tab, eps, rho); }
  KOKKOS_INLINE_FUNCTION Real s(Real eps, Real rho) const { return field(s_tab, eps, rho); }

 private:
  struct Loc {
    int i = 0, j = 0;
    Real t = 0.0, u = 0.0;        // position inside the cell, in [0, 1] (cell units)
    Real exoff = 0.0, eyoff = 0.0;  // extrapolation distance beyond the lattice, in cell units (zero unless extrapolate)
    bool clamped = false;         // eps < eps_min(rho)
    bool zx = false, zy = false;  // argument clamped to the lattice with extrapolate = false: derivative is zero
    Real emin_slope = 0.0;        // d eps_min / d rho [GeV]
  };

  KOKKOS_INLINE_FUNCTION Real field(const Real* F, Real eps, Real rho) const {
    const Loc L = locate(eps, rho);
    Real v, vx, vy;
    eval(F, L, v, vx, vy);
    return v;
  }

  // 1D interpolation along rho of a length-ny array (cell j, position u): value and derivative per cell.
  KOKKOS_INLINE_FUNCTION void interp1(const Real* E, int j, Real u, Real& v, Real& dv) const {
    if (!bicubic) {
      v = E[j] + u * (E[j + 1] - E[j]);
      dv = E[j + 1] - E[j];
      return;
    }
    const Real m0 = d1(E, j), m1 = d1(E, j + 1);
    const Real u2 = u * u, u3 = u2 * u;
    v = (2 * u3 - 3 * u2 + 1) * E[j] + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * E[j + 1] + (u3 - u2) * m1;
    dv = (6 * u2 - 6 * u) * E[j] + (3 * u2 - 4 * u + 1) * m0 + (-6 * u2 + 6 * u) * E[j + 1] + (3 * u2 - 2 * u) * m1;
  }
  KOKKOS_INLINE_FUNCTION Real d1(const Real* E, int j) const {
    if (j == 0) return E[1] - E[0];
    if (j == ny - 1) return E[ny - 1] - E[ny - 2];
    return 0.5 * (E[j + 1] - E[j - 1]);
  }

  KOKKOS_INLINE_FUNCTION void cold_curve(Real, const Loc& L, Real& em, Real& dem) const {
    interp1(emin_tab, L.j, L.u, em, dem);
    if (extrapolate) em += dem * L.eyoff;
  }

  // The conversion eps -> (eps/eps_s - x0)/dx carries a rounding error of up to ~1e-13 cell units at the largest lattice indices; a position
  // within 1e-10 cell units of a node is moved onto the node so that nodes are reproduced to machine precision.  The effect on any other
  // point is below 1e-10 of the change of the field across one cell.
  KOKKOS_INLINE_FUNCTION static Real snap(Real f) {
    const Real r = Kokkos::floor(f + 0.5);
    return Kokkos::fabs(f - r) < 1e-10 ? r : f;
  }

  KOKKOS_INLINE_FUNCTION Loc locate(Real eps, Real rho) const {
    Loc L;
    const Real fy = snap((rho / rho_s - y0) / dy);
    const Real fyc = fy < 0.0 ? 0.0 : (fy > ny - 1 ? Real(ny - 1) : fy);
    L.zy = (extrapolate == 0) && (fy != fyc);
    L.eyoff = extrapolate ? fy - fyc : 0.0;
    int j = static_cast<int>(fyc);
    if (j > ny - 2) j = ny - 2;
    L.j = j;
    L.u = fyc - j;
    Real em, dem;
    interp1(emin_tab, j, L.u, em, dem);
    if (extrapolate) em += dem * L.eyoff;
    L.emin_slope = L.zy ? 0.0 : dem / (dy * rho_s);
    Real x = eps / eps_s;
    const Real xm = em / eps_s;
    if (x < xm) {
      L.clamped = true;
      x = xm;
    }
    const Real fx = snap((x - x0) / dx);
    const Real fxc = fx < 0.0 ? 0.0 : (fx > nx - 1 ? Real(nx - 1) : fx);
    L.zx = (extrapolate == 0) && (fx != fxc);
    L.exoff = extrapolate ? fx - fxc : 0.0;
    int i = static_cast<int>(fxc);
    if (i > nx - 2) i = nx - 2;
    L.i = i;
    L.t = fxc - i;
    return L;
  }

  KOKKOS_INLINE_FUNCTION Real at(const Real* F, int i, int j) const { return F[i * ny + j]; }
  KOKKOS_INLINE_FUNCTION Real dxn(const Real* F, int i, int j) const {
    if (i == 0) return at(F, 1, j) - at(F, 0, j);
    if (i == nx - 1) return at(F, nx - 1, j) - at(F, nx - 2, j);
    return 0.5 * (at(F, i + 1, j) - at(F, i - 1, j));
  }
  KOKKOS_INLINE_FUNCTION Real dyn(const Real* F, int i, int j) const {
    if (j == 0) return at(F, i, 1) - at(F, i, 0);
    if (j == ny - 1) return at(F, i, ny - 1) - at(F, i, ny - 2);
    return 0.5 * (at(F, i, j + 1) - at(F, i, j - 1));
  }
  KOKKOS_INLINE_FUNCTION Real dxyn(const Real* F, int i, int j) const {
    if (i == 0) return dyn(F, 1, j) - dyn(F, 0, j);
    if (i == nx - 1) return dyn(F, nx - 1, j) - dyn(F, nx - 2, j);
    return 0.5 * (dyn(F, i + 1, j) - dyn(F, i - 1, j));
  }

  // Value and derivatives per cell (index units) of the interpolant in the cell L, with linear extrapolation if requested.
  KOKKOS_INLINE_FUNCTION void eval(const Real* F, const Loc& L, Real& v, Real& vx, Real& vy) const {
    const Real t = L.t, u = L.u;
    if (!bicubic) {
      const int i = L.i, j = L.j;
      const Real f00 = at(F, i, j), f10 = at(F, i + 1, j), f01 = at(F, i, j + 1), f11 = at(F, i + 1, j + 1);
      v = (1 - t) * (1 - u) * f00 + t * (1 - u) * f10 + (1 - t) * u * f01 + t * u * f11;
      vx = (1 - u) * (f10 - f00) + u * (f11 - f01);
      vy = (1 - t) * (f01 - f00) + t * (f11 - f10);
    } else {
      const Real t2 = t * t, t3 = t2 * t, u2 = u * u, u3 = u2 * u;
      const Real ht0[2] = {2 * t3 - 3 * t2 + 1, -2 * t3 + 3 * t2}, ht1[2] = {t3 - 2 * t2 + t, t3 - t2};
      const Real dt0[2] = {6 * t2 - 6 * t, -6 * t2 + 6 * t}, dt1[2] = {3 * t2 - 4 * t + 1, 3 * t2 - 2 * t};
      const Real hu0[2] = {2 * u3 - 3 * u2 + 1, -2 * u3 + 3 * u2}, hu1[2] = {u3 - 2 * u2 + u, u3 - u2};
      const Real du0[2] = {6 * u2 - 6 * u, -6 * u2 + 6 * u}, du1[2] = {3 * u2 - 4 * u + 1, 3 * u2 - 2 * u};
      v = 0.0;
      vx = 0.0;
      vy = 0.0;
      for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
          const int i = L.i + a, j = L.j + b;
          const Real f = at(F, i, j), fx = dxn(F, i, j), fy = dyn(F, i, j), fxy = dxyn(F, i, j);
          v += f * ht0[a] * hu0[b] + fx * ht1[a] * hu0[b] + fy * ht0[a] * hu1[b] + fxy * ht1[a] * hu1[b];
          vx += f * dt0[a] * hu0[b] + fx * dt1[a] * hu0[b] + fy * dt0[a] * hu1[b] + fxy * dt1[a] * hu1[b];
          vy += f * ht0[a] * du0[b] + fx * ht1[a] * du0[b] + fy * ht0[a] * du1[b] + fxy * ht1[a] * du1[b];
        }
      }
    }
    if (extrapolate) v += vx * L.exoff + vy * L.eyoff;
    if (L.zx) vx = 0.0;
    if (L.zy) vy = 0.0;
  }
};

// Lattice description shared by the file header and the host-side owner.
struct EosTableLattice {
  Real rho_s = 0.153, m_N = 0.940;
  Real x0 = 0.0, dx = 0.05, y0 = 0.0, dy = 0.02;
  int nx = 0, ny = 0;
};

// Host-side owner of the table data (Views in the default memory space).  Must outlive every EosTable made from it.
class EosTableData {
 public:
  EosTableLattice lat;
  std::string description;
  std::string sha256;  // of the .bin file (empty when built in memory)
  Kokkos::View<Real*> p0, T, mu, s, eps_min;

  // Allocate the Views for the given lattice (contents undefined); used by loaders and by tests that fill the arrays in a kernel.
  static EosTableData allocate(const EosTableLattice& lat, const std::string& description = "") {
    EosTableData d;
    d.lat = lat;
    d.description = description;
    const size_t n = static_cast<size_t>(lat.nx) * static_cast<size_t>(lat.ny);
    d.p0 = Kokkos::View<Real*>("eos_table_p0", n);
    d.T = Kokkos::View<Real*>("eos_table_T", n);
    d.mu = Kokkos::View<Real*>("eos_table_mu", n);
    d.s = Kokkos::View<Real*>("eos_table_s", n);
    d.eps_min = Kokkos::View<Real*>("eos_table_eps_min", static_cast<size_t>(lat.ny));
    return d;
  }

  // Load <stem>.json and <stem>.bin; `path` may be the stem or the .json path.  Throws EosTableError with a message naming the problem.
  static EosTableData load(const std::string& path) {
    static_assert(std::endian::native == std::endian::little, "EoS table files are little-endian; byte swapping is not implemented");
    std::string stem = path;
    if (stem.size() > 5 && stem.compare(stem.size() - 5, 5, ".json") == 0) stem.resize(stem.size() - 5);
    const std::string jpath = stem + ".json", bpath = stem + ".bin";

    std::ifstream jin(jpath);
    if (!jin) throw EosTableError("EoS table: cannot open header '" + jpath + "'");
    nlohmann::json h;
    try {
      h = nlohmann::json::parse(jin);
    } catch (const nlohmann::json::exception& e) {
      throw EosTableError("EoS table: header '" + jpath + "' is not valid JSON: " + e.what());
    }
    auto need = [&](const nlohmann::json& obj, const char* key, const std::string& where) -> const nlohmann::json& {
      if (!obj.is_object() || !obj.contains(key)) throw EosTableError("EoS table: header '" + jpath + "' lacks the key '" + where + key + "'");
      return obj.at(key);
    };
    auto num = [&](const nlohmann::json& obj, const char* key, const std::string& where) -> Real {
      const auto& v = need(obj, key, where);
      if (!v.is_number()) throw EosTableError("EoS table: header key '" + where + key + "' must be a number");
      return v.get<Real>();
    };
    auto str = [&](const nlohmann::json& obj, const char* key, const std::string& where) -> std::string {
      const auto& v = need(obj, key, where);
      if (!v.is_string()) throw EosTableError("EoS table: header key '" + where + key + "' must be a string");
      return v.get<std::string>();
    };

    const int version = static_cast<int>(num(h, "format_version", ""));
    if (version != 1) throw EosTableError("EoS table: unsupported format_version " + std::to_string(version) + " (this reader handles 1)");
    if (str(h, "byte_order", "") != "little") throw EosTableError("EoS table: byte_order must be 'little'");
    if (str(h, "dtype", "") != "float64") throw EosTableError("EoS table: dtype must be 'float64'");
    EosTableData d;
    d.description = str(h, "description", "");
    d.sha256 = str(h, "sha256", "");
    const auto& L = need(h, "lattice", "");
    d.lat.rho_s = num(L, "rho_s_fm3", "lattice.");
    d.lat.m_N = num(L, "m_N_GeV", "lattice.");
    d.lat.x0 = num(L, "x0", "lattice.");
    d.lat.dx = num(L, "dx", "lattice.");
    d.lat.y0 = num(L, "y0", "lattice.");
    d.lat.dy = num(L, "dy", "lattice.");
    d.lat.nx = static_cast<int>(num(L, "nx", "lattice."));
    d.lat.ny = static_cast<int>(num(L, "ny", "lattice."));
    const auto& fl = need(h, "fields", "");
    const std::vector<std::string> expect_fields = {"p0", "T", "mu", "s", "eps_min"};
    if (!fl.is_array() || fl.get<std::vector<std::string>>() != expect_fields)
      throw EosTableError("EoS table: 'fields' must be [p0, T, mu, s, eps_min] in this order");
    const auto& un = need(h, "units", "");
    const char* ukeys[5] = {"p0", "T", "mu", "s", "eps_min"};
    const char* uvals[5] = {"GeV/fm^3", "GeV", "GeV", "fm^-3", "GeV/fm^3"};
    for (int k = 0; k < 5; ++k)
      if (str(un, ukeys[k], "units.") != uvals[k]) throw EosTableError(std::string("EoS table: units.") + ukeys[k] + " must be '" + uvals[k] + "'");
    if (d.sha256.size() != 64) throw EosTableError("EoS table: 'sha256' must be a 64-character hex string");

    if (d.lat.nx < 2 || d.lat.ny < 2) throw EosTableError("EoS table: nx and ny must be at least 2");
    if (!(d.lat.dx > 0.0) || !(d.lat.dy > 0.0)) throw EosTableError("EoS table: dx and dy must be positive");
    if (!(d.lat.rho_s > 0.0) || !(d.lat.m_N > 0.0)) throw EosTableError("EoS table: rho_s_fm3 and m_N_GeV must be positive");

    const size_t nxy = static_cast<size_t>(d.lat.nx) * static_cast<size_t>(d.lat.ny);
    const size_t n_values = 4 * nxy + static_cast<size_t>(d.lat.ny);
    std::ifstream bin(bpath, std::ios::binary);
    if (!bin) throw EosTableError("EoS table: cannot open data file '" + bpath + "'");
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(bin)), std::istreambuf_iterator<char>());
    if (bytes.size() != n_values * sizeof(double))
      throw EosTableError("EoS table: '" + bpath + "' has " + std::to_string(bytes.size()) + " bytes, the header lattice requires " +
                          std::to_string(n_values * sizeof(double)));
    const std::string digest = sha256_hex(bytes.data(), bytes.size());
    if (digest != d.sha256)
      throw EosTableError("EoS table: checksum mismatch for '" + bpath + "' (header " + d.sha256 + ", file " + digest + ")");

    const double* v = reinterpret_cast<const double*>(bytes.data());
    for (size_t k = 0; k < n_values; ++k)
      if (!std::isfinite(v[k])) throw EosTableError("EoS table: non-finite value in '" + bpath + "' at element " + std::to_string(k));

    EosTableData t = allocate(d.lat, d.description);
    t.sha256 = d.sha256;
    auto copy_in = [&](Kokkos::View<Real*>& dst, size_t offset, size_t n) {
      auto hm = Kokkos::create_mirror_view(dst);
      for (size_t k = 0; k < n; ++k) hm(k) = v[offset + k];
      Kokkos::deep_copy(dst, hm);
    };
    copy_in(t.p0, 0, nxy);
    copy_in(t.T, nxy, nxy);
    copy_in(t.mu, 2 * nxy, nxy);
    copy_in(t.s, 3 * nxy, nxy);
    copy_in(t.eps_min, 4 * nxy, static_cast<size_t>(d.lat.ny));
    return t;
  }

  // Load the table named by eos.table_path (stem or .json path); requires eos.model to be table_two_phase or table_maxwell.
  static EosTableData from_config(const Config& cfg) {
    const EosModel m = eos_model_from_config(cfg);
    if (!eos_model_uses_table(m)) throw ConfigError(std::string("eos.model '") + eos_model_name(m) + "' does not use a table");
    const std::string p = cfg.get<std::string>("eos.table_path");
    if (p.empty()) throw ConfigError("eos.table_path is empty but eos.model selects a table");
    return load(p);
  }

  // Device-callable view of the data.
  EosTable eos(TableInterp interp = TableInterp::bilinear, bool extrapolate = false) const {
    EosTable e;
    e.p_tab = p0.data();
    e.T_tab = T.data();
    e.mu_tab = mu.data();
    e.s_tab = s.data();
    e.emin_tab = eps_min.data();
    e.nx = lat.nx;
    e.ny = lat.ny;
    e.x0 = lat.x0;
    e.dx = lat.dx;
    e.y0 = lat.y0;
    e.dy = lat.dy;
    e.rho_s = lat.rho_s;
    e.eps_s = lat.m_N * lat.rho_s;
    e.bicubic = interp == TableInterp::bicubic ? 1 : 0;
    e.extrapolate = extrapolate ? 1 : 0;
    return e;
  }
  EosTable eos(const Config& cfg) const {
    return eos(parse_table_interp(cfg.get<std::string>("eos.table_interpolation")), cfg.get<bool>("eos.extrapolate"));
  }

  RangeCheck check_range(Real eps_min_, Real eps_max, Real rho_min, Real rho_max) const {
    return eos().check_range(eps_min_, eps_max, rho_min, rho_max);
  }
};

}  // namespace spf
