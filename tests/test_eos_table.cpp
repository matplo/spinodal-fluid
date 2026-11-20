// Acceptance tests of the tabulated equation of state and the scaled wrapper (WP2c).  Every compared value is evaluated in a Kokkos
// kernel (functors, so that the same source compiles with nvcc); measured numbers are printed as "[T2c.x] ...".
#include <doctest/doctest.h>

#include "spf/config.hpp"
#include "spf/eos/eos.hpp"
#include "spf/types.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using spf::EosScaled;
using spf::EosTable;
using spf::EosTableData;
using spf::EosTableError;
using spf::EosTableLattice;
using spf::ExecSpace;
using spf::Real;
using spf::TableInterp;
using VecD = std::vector<Real>;

static_assert(spf::EquationOfState<spf::EosTable>);
static_assert(spf::EquationOfState<spf::EosScaled<spf::EosTable>>);
static_assert(spf::EquationOfState<spf::EosScaled<spf::EosColdThermal>>);
static_assert(std::is_trivially_copyable_v<spf::EosTable>);
static_assert(std::is_trivially_copyable_v<spf::EosScaled<spf::EosTable>>);
static_assert(std::is_trivially_copyable_v<spf::EosScaled<spf::EosColdThermal>>);

namespace {

namespace fs = std::filesystem;
const std::string kSrc = SPF_SOURCE_DIR;
const std::string kData = SPF_TEST_DATA_DIR;
constexpr Real kRhoS = 0.153, kMN = 0.940, kEpsS = kMN * kRhoS, kGamma = 4.0 / 3.0;

using DView = Kokkos::View<Real*>;
DView to_device(const VecD& v) {
  DView d("in", v.size());
  auto h = Kokkos::create_mirror_view(d);
  for (size_t i = 0; i < v.size(); ++i) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}
VecD to_host(const DView& d) {
  auto h = Kokkos::create_mirror_view(d);
  Kokkos::deep_copy(h, d);
  return VecD(h.data(), h.data() + h.extent(0));
}

struct Rng {  // splitmix64: identical sequence on every platform
  std::uint64_t s;
  explicit Rng(std::uint64_t seed) : s(seed) {}
  std::uint64_t next() {
    std::uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  Real u() { return static_cast<Real>(next() >> 11) * (1.0 / 9007199254740992.0); }
};

Real rel(Real a, Real b) { return a == b ? 0.0 : std::abs(a - b) / std::abs(b); }
Real median(VecD v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
}
Real vmax(const VecD& v) { return v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()); }

// ---- synthetic test functions (must match tools/eos/write_table.py) ----
KOKKOS_INLINE_FUNCTION Real synth_T(Real x, Real y) { return 0.1 * x / (1.0 + y); }
KOKKOS_INLINE_FUNCTION Real synth_mu(Real x, Real y) { return 0.94 + 0.3 * y / (1.0 + 0.1 * y) + 0.005 * x; }
KOKKOS_INLINE_FUNCTION Real synth_s(Real x, Real y) { return 0.5 * x * (1.0 + 0.1 * y) / (1.0 + 0.01 * x) + 0.2 * y; }

// Fill a table with the analytic cold-thermal EoS (smooth continuation below the cold curve) on the device.
struct FillSynth {
  spf::EosColdThermal eos;
  DView p, T, mu, s, emin;
  int nx, ny;
  Real dx, dy;
  KOKKOS_INLINE_FUNCTION void operator()(const int k) const {
    const int ix = k / ny, iy = k % ny;
    const Real x = ix * dx, y = iy * dy, rho = y * kRhoS, eps = x * kEpsS;
    p(k) = eos.p_cold(rho) + (kGamma - 1.0) * (eps - eos.eps_cold(rho));
    T(k) = synth_T(x, y);
    mu(k) = synth_mu(x, y);
    s(k) = synth_s(x, y);
    if (k < ny) emin(k) = eos.eps_cold(k * dy * kRhoS);
  }
};
EosTableData make_synthetic(const spf::ColdCurveTable& cold, int nx, int ny, Real dx, Real dy) {
  EosTableLattice lat;
  lat.rho_s = kRhoS;
  lat.m_N = kMN;
  lat.x0 = 0.0;
  lat.dx = dx;
  lat.y0 = 0.0;
  lat.dy = dy;
  lat.nx = nx;
  lat.ny = ny;
  EosTableData d = EosTableData::allocate(lat, "synthetic, built in the test");
  FillSynth f{spf::EosColdThermal(cold, spf::ColdVariant::two_phase), d.p0, d.T, d.mu, d.s, d.eps_min, nx, ny, dx, dy};
  Kokkos::parallel_for("fill_synthetic", Kokkos::RangePolicy<ExecSpace>(0, nx * ny), f);
  Kokkos::fence();
  return d;
}

// ---- generic evaluators ----
template <class E>
struct EvalK {
  E eos;
  DView eps, rho, p, dpe, dpr, vs;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    p(i) = eos.p0(eps(i), rho(i));
    dpe(i) = eos.dp_deps(eps(i), rho(i));
    dpr(i) = eos.dp_drho(eps(i), rho(i));
    vs(i) = eos.vs2(eps(i), rho(i));
  }
};
struct Out {
  VecD p, dpe, dpr, vs;
};
template <class E>
Out eval(const E& eos, const VecD& eps, const VecD& rho) {
  const int n = static_cast<int>(eps.size());
  EvalK<E> k{eos, to_device(eps), to_device(rho), DView("p", n), DView("a", n), DView("b", n), DView("v", n)};
  Kokkos::parallel_for("eval_eos", Kokkos::RangePolicy<ExecSpace>(0, n), k);
  return {to_host(k.p), to_host(k.dpe), to_host(k.dpr), to_host(k.vs)};
}

struct TmuSK {
  EosTable eos;
  DView eps, rho, T, mu, s;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    T(i) = eos.T(eps(i), rho(i));
    mu(i) = eos.mu(eps(i), rho(i));
    s(i) = eos.s(eps(i), rho(i));
  }
};
struct OutTmuS {
  VecD T, mu, s;
};
OutTmuS eval_tmus(const EosTable& eos, const VecD& eps, const VecD& rho) {
  const int n = static_cast<int>(eps.size());
  TmuSK k{eos, to_device(eps), to_device(rho), DView("T", n), DView("mu", n), DView("s", n)};
  Kokkos::parallel_for("eval_tmus", Kokkos::RangePolicy<ExecSpace>(0, n), k);
  return {to_host(k.T), to_host(k.mu), to_host(k.s)};
}

struct EminK {
  EosTable eos;
  DView rho, em;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const { em(i) = eos.eps_min_of_rho(rho(i)); }
};
VecD eps_min_of(const EosTable& eos, const VecD& rho) {
  EminK k{eos, to_device(rho), DView("em", rho.size())};
  Kokkos::parallel_for("eval_emin", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(rho.size())), k);
  return to_host(k.em);
}
struct ColdEpsK {
  spf::EosColdThermal eos;
  DView rho, ec;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const { ec(i) = eos.eps_cold(rho(i)); }
};
VecD eps_cold_of(const spf::EosColdThermal& eos, const VecD& rho) {
  ColdEpsK k{eos, to_device(rho), DView("ec", rho.size())};
  Kokkos::parallel_for("eval_ec", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(rho.size())), k);
  return to_host(k.ec);
}

// Centred finite differences of p0 (relative step h) in eps and in rho.
template <class E>
struct FdK {
  E eos;
  DView eps, rho, fx, fy;
  Real h;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    const Real he = h * eps(i), hr = h * rho(i);
    fx(i) = (eos.p0(eps(i) + he, rho(i)) - eos.p0(eps(i) - he, rho(i))) / (2.0 * he);
    fy(i) = (eos.p0(eps(i), rho(i) + hr) - eos.p0(eps(i), rho(i) - hr)) / (2.0 * hr);
  }
};
template <class E>
std::pair<VecD, VecD> fd(const E& eos, const VecD& eps, const VecD& rho, Real h) {
  const int n = static_cast<int>(eps.size());
  FdK<E> k{eos, to_device(eps), to_device(rho), DView("fx", n), DView("fy", n), h};
  Kokkos::parallel_for("fd_p0", Kokkos::RangePolicy<ExecSpace>(0, n), k);
  return {to_host(k.fx), to_host(k.fy)};
}

// Isentropic sound speed from p0 alone: RK4 on d eps / d rho = (eps + p0) / rho through (eps, rho), central differences at +-delta and
// +-delta/2 with Richardson extrapolation, vs2 = (rho / h) dp/drho.
template <class E>
struct IsenK {
  E eos;
  DView eps, rho, out;
  Real delta;
  KOKKOS_INLINE_FUNCTION Real f(Real e, Real r) const { return (e + eos.p0(e, r)) / r; }
  KOKKOS_INLINE_FUNCTION Real p_at(Real e0, Real r0, Real s) const {
    const Real k1 = f(e0, r0), k2 = f(e0 + 0.5 * s * k1, r0 + 0.5 * s), k3 = f(e0 + 0.5 * s * k2, r0 + 0.5 * s), k4 = f(e0 + s * k3, r0 + s);
    return eos.p0(e0 + s / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4), r0 + s);
  }
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    const Real e0 = eps(i), r0 = rho(i);
    const Real d1 = (p_at(e0, r0, delta) - p_at(e0, r0, -delta)) / (2.0 * delta);
    const Real d2 = (p_at(e0, r0, 0.5 * delta) - p_at(e0, r0, -0.5 * delta)) / delta;
    const Real dp = (4.0 * d2 - d1) / 3.0;
    out(i) = r0 / (e0 + eos.p0(e0, r0)) * dp;
  }
};
template <class E>
VecD isentropic_vs2(const E& eos, const VecD& eps, const VecD& rho, Real delta) {
  IsenK<E> k{eos, to_device(eps), to_device(rho), DView("o", eps.size()), delta};
  Kokkos::parallel_for("isentropic_vs2", Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(eps.size())), k);
  return to_host(k.out);
}

// ---- scaled wrapper vs the base at (b eps, c rho): all identities in one kernel ----
template <class Base>
struct ScaleCheckK {
  Base base;
  EosScaled<Base> w;
  Real b, c;
  DView eps, rho, d;  // d: 4 relative differences per point
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    const Real e0 = b * eps(i), r0 = c * rho(i);
    auto r = [](Real a, Real bb) { return a == bb ? 0.0 : Kokkos::fabs(a - bb) / Kokkos::fabs(bb); };
    d(4 * i + 0) = r(w.p0(eps(i), rho(i)), base.p0(e0, r0) / b);
    d(4 * i + 1) = r(w.dp_deps(eps(i), rho(i)), base.dp_deps(e0, r0));
    d(4 * i + 2) = r(w.dp_drho(eps(i), rho(i)), (c / b) * base.dp_drho(e0, r0));
    d(4 * i + 3) = r(w.vs2(eps(i), rho(i)), base.vs2(e0, r0));
  }
};
struct ScaleCheckTmuSK {
  EosTable base;
  EosScaled<EosTable> w;
  Real b, c;
  DView eps, rho, d;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    const Real e0 = b * eps(i), r0 = c * rho(i);
    auto r = [](Real a, Real bb) { return a == bb ? 0.0 : Kokkos::fabs(a - bb) / Kokkos::fabs(bb); };
    d(3 * i + 0) = r(w.T(eps(i), rho(i)), base.T(e0, r0) / b);
    d(3 * i + 1) = r(w.mu(eps(i), rho(i)), (c / b) * base.mu(e0, r0));
    d(3 * i + 2) = r(w.s(eps(i), rho(i)), base.s(e0, r0));
  }
};

std::string tmp_dir(const char* tag) {
  const fs::path d = fs::temp_directory_path() / (std::string("spf_eos_table_test_") + tag);
  fs::remove_all(d);
  fs::create_directories(d);
  return d.string();
}
void copy_table(const std::string& stem, const std::string& dst_stem) {
  fs::copy_file(stem + ".json", dst_stem + ".json", fs::copy_options::overwrite_existing);
  fs::copy_file(stem + ".bin", dst_stem + ".bin", fs::copy_options::overwrite_existing);
}
nlohmann::json read_json(const std::string& p) {
  std::ifstream in(p);
  return nlohmann::json::parse(in);
}
void write_json(const std::string& p, const nlohmann::json& j) {
  std::ofstream o(p);
  o << j.dump(2);
}

const std::string kCommitted = kData + "/synthetic_eos";

}  // namespace

TEST_CASE("T2c.0 concept and trivially-copyable checks") {
  CHECK(spf::EquationOfState<EosTable>);
  CHECK(spf::EquationOfState<EosScaled<EosTable>>);
  CHECK(spf::EquationOfState<EosScaled<spf::EosColdThermal>>);
  CHECK(std::is_trivially_copyable_v<EosTable>);
  CHECK(std::is_trivially_copyable_v<EosScaled<EosTable>>);
  CHECK(std::is_trivially_copyable_v<EosScaled<spf::EosColdThermal>>);
  std::printf("[T2c.0] EosTable, EosScaled<EosTable>, EosScaled<EosColdThermal>: concept satisfied, trivially copyable\n");
}

TEST_CASE("T2c.1 loader: sizes, lattice, checksum, rejection of damaged files") {
  // SHA-256 known-answer tests
  CHECK(spf::sha256_hex(reinterpret_cast<const unsigned char*>("abc"), 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(spf::sha256_hex(nullptr, 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  {
    const std::string m = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";  // 56 bytes: two-block padding
    CHECK(spf::sha256_hex(reinterpret_cast<const unsigned char*>(m.data()), m.size()) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  }

  const EosTableData t = EosTableData::load(kCommitted);
  CHECK(t.lat.nx == 201);
  CHECK(t.lat.ny == 401);
  CHECK(t.lat.x0 == 0.0);
  CHECK(t.lat.dx == doctest::Approx(0.2));
  CHECK(t.lat.dy == doctest::Approx(0.04));
  CHECK(t.lat.rho_s == doctest::Approx(0.153));
  CHECK(t.lat.m_N == doctest::Approx(0.940));
  CHECK(t.p0.extent(0) == 201u * 401u);
  CHECK(t.eps_min.extent(0) == 401u);
  CHECK(t.sha256.size() == 64);
  CHECK(t.description.find("SYNTHETIC") == 0);
  CHECK(EosTableData::load(kCommitted + ".json").lat.nx == 201);  // header path accepted as well

  const std::string dir = tmp_dir("loader");
  auto fresh = [&](const char* name) {
    const std::string s = dir + "/" + name;
    copy_table(kCommitted, s);
    return s;
  };
  // truncated .bin
  {
    const std::string s = fresh("trunc");
    const auto sz = fs::file_size(s + ".bin");
    fs::resize_file(s + ".bin", sz - 8);
    CHECK_THROWS_WITH_AS(EosTableData::load(s), doctest::Contains("bytes, the header lattice requires"), EosTableError);
    fs::resize_file(s + ".bin", sz + 8);  // too long (zero padded)
    CHECK_THROWS_WITH_AS(EosTableData::load(s), doctest::Contains("bytes, the header lattice requires"), EosTableError);
  }
  // wrong checksum: header digest altered / one data bit flipped
  {
    const std::string s = fresh("sha");
    auto j = read_json(s + ".json");
    std::string d = j["sha256"].get<std::string>();
    d[0] = (d[0] == '0') ? '1' : '0';
    j["sha256"] = d;
    write_json(s + ".json", j);
    CHECK_THROWS_WITH_AS(EosTableData::load(s), doctest::Contains("checksum mismatch"), EosTableError);
  }
  {
    const std::string s = fresh("flip");
    std::fstream f(s + ".bin", std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(1000);
    char ch;
    f.seekg(1000);
    f.get(ch);
    ch = static_cast<char>(ch ^ 0x01);
    f.seekp(1000);
    f.put(ch);
    f.close();
    CHECK_THROWS_WITH_AS(EosTableData::load(s), doctest::Contains("checksum mismatch"), EosTableError);
  }
  // missing header keys
  for (const char* key : {"format_version", "description", "sha256", "lattice", "fields", "units", "byte_order"}) {
    const std::string s = fresh((std::string("miss_") + key).c_str());
    auto j = read_json(s + ".json");
    j.erase(key);
    write_json(s + ".json", j);
    CHECK_THROWS_WITH_AS(EosTableData::load(s), doctest::Contains(std::string("lacks the key '") + key + "'"), EosTableError);
  }
  for (const char* key : {"dx", "ny", "rho_s_fm3"}) {
    const std::string s = fresh((std::string("miss_lat_") + key).c_str());
    auto j = read_json(s + ".json");
    j["lattice"].erase(key);
    write_json(s + ".json", j);
    CHECK_THROWS_WITH_AS(EosTableData::load(s), doctest::Contains(std::string("lacks the key 'lattice.") + key + "'"), EosTableError);
  }
  // lattice inconsistent with the .bin size
  {
    const std::string s = fresh("lat");
    auto j = read_json(s + ".json");
    j["lattice"]["nx"] = 200;
    write_json(s + ".json", j);
    CHECK_THROWS_WITH_AS(EosTableData::load(s), doctest::Contains("bytes, the header lattice requires"), EosTableError);
  }
  // missing files, wrong version
  CHECK_THROWS_WITH_AS(EosTableData::load(dir + "/does_not_exist"), doctest::Contains("cannot open header"), EosTableError);
  {
    const std::string s = fresh("ver");
    auto j = read_json(s + ".json");
    j["format_version"] = 2;
    write_json(s + ".json", j);
    CHECK_THROWS_WITH_AS(EosTableData::load(s), doctest::Contains("unsupported format_version"), EosTableError);
  }
  fs::remove_all(dir);

  // configuration: eos.model table_* selects the reader
  spf::Config cfg = spf::Config::from_file(SPF_DEFAULT_CONFIG);
  cfg.set_from_string("eos.table_path=" + kCommitted);
  cfg.set_from_string("eos.table_interpolation=bicubic");
  CHECK(spf::eos_model_uses_table(spf::eos_model_from_config(cfg)));
  const EosTableData tc = EosTableData::from_config(cfg);
  const EosTable e = tc.eos(cfg);
  CHECK(e.bicubic == 1);
  CHECK(e.extrapolate == 0);
  cfg.set_from_string("eos.model=table_maxwell");
  CHECK_NOTHROW(EosTableData::from_config(cfg));
  cfg.set_from_string("eos.model=cold_thermal_two_phase");
  CHECK_THROWS_AS(EosTableData::from_config(cfg), spf::ConfigError);
  std::printf("[T2c.1] committed table loaded: nx=%d ny=%d sha256=%s; truncated, long, bad-checksum (header and data), 10 missing-key and bad-version files rejected\n",
              t.lat.nx, t.lat.ny, t.sha256.c_str());
}

TEST_CASE("T2c.2 lattice nodes return the stored values") {
  const EosTableData t = EosTableData::load(kCommitted);
  auto hp = Kokkos::create_mirror_view(t.p0);
  auto hT = Kokkos::create_mirror_view(t.T);
  auto hmu = Kokkos::create_mirror_view(t.mu);
  auto hs = Kokkos::create_mirror_view(t.s);
  auto he = Kokkos::create_mirror_view(t.eps_min);
  Kokkos::deep_copy(hp, t.p0);
  Kokkos::deep_copy(hT, t.T);
  Kokkos::deep_copy(hmu, t.mu);
  Kokkos::deep_copy(hs, t.s);
  Kokkos::deep_copy(he, t.eps_min);
  const int nx = t.lat.nx, ny = t.lat.ny;
  VecD eps, rho, ep, eT, emu, es;
  for (int ix = 0; ix < nx; ix += (ix + 3 < nx ? 3 : 1)) {
    for (int iy = 0; iy < ny; iy += (iy + 5 < ny ? 5 : 1)) {
      const Real x = t.lat.x0 + ix * t.lat.dx, y = t.lat.y0 + iy * t.lat.dy;
      if (x * kEpsS < he(iy) * (1.0 + 1e-9)) continue;  // nodes on or below the cold curve are clamped by design
      eps.push_back(x * kEpsS);
      rho.push_back(y * kRhoS);
      const size_t k = static_cast<size_t>(ix) * ny + iy;
      ep.push_back(hp(k));
      eT.push_back(hT(k));
      emu.push_back(hmu(k));
      es.push_back(hs(k));
    }
  }
  REQUIRE(eps.size() > 1000);
  Real worst = 0.0;
  for (TableInterp ti : {TableInterp::bilinear, TableInterp::bicubic}) {
    const EosTable e = t.eos(ti);
    const Out o = eval(e, eps, rho);
    const OutTmuS m = eval_tmus(e, eps, rho);
    Real w = 0.0;
    for (size_t i = 0; i < eps.size(); ++i) {
      w = std::max({w, rel(o.p[i], ep[i]), rel(m.T[i], eT[i]), rel(m.mu[i], emu[i]), rel(m.s[i], es[i])});
    }
    std::printf("[T2c.2] %s: %zu nodes, max relative deviation of p0, T, mu, s from the stored values: %.3e\n",
                ti == TableInterp::bilinear ? "bilinear" : "bicubic", eps.size(), w);
    CHECK(w < 1e-14);
    worst = std::max(worst, w);
  }
  CHECK(worst < 1e-14);
}

namespace {
struct Acc {
  Real max_relp, med_relp, max_dvs, med_dvs, max_dvs_spin, max_relT, max_relmu, max_rels;
  int n_spin;
};
struct AccK {
  EosTable tab;
  spf::EosColdThermal ref;
  DView eps, rho, relp, dvs, rT, rmu, rs;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    const Real pr = ref.p0(eps(i), rho(i)), pt = tab.p0(eps(i), rho(i));
    relp(i) = Kokkos::fabs(pt - pr) / Kokkos::fabs(pr);
    dvs(i) = Kokkos::fabs(tab.vs2(eps(i), rho(i)) - ref.vs2(eps(i), rho(i)));
    const Real x = eps(i) / kEpsS, y = rho(i) / kRhoS;
    rT(i) = Kokkos::fabs(tab.T(eps(i), rho(i)) / synth_T(x, y) - 1.0);
    rmu(i) = Kokkos::fabs(tab.mu(eps(i), rho(i)) / synth_mu(x, y) - 1.0);
    rs(i) = Kokkos::fabs(tab.s(eps(i), rho(i)) / synth_s(x, y) - 1.0);
  }
};
Acc accuracy(const EosTableData& t, const spf::ColdCurveTable& cold, TableInterp ti, const VecD& eps, const VecD& rho) {
  const int n = static_cast<int>(eps.size());
  AccK k{t.eos(ti), spf::EosColdThermal(cold, spf::ColdVariant::two_phase), to_device(eps), to_device(rho), DView("a", n), DView("b", n),
         DView("c", n), DView("d", n), DView("e", n)};
  Kokkos::parallel_for("accuracy", Kokkos::RangePolicy<ExecSpace>(0, n), k);
  const VecD relp = to_host(k.relp), dvs = to_host(k.dvs), rT = to_host(k.rT), rmu = to_host(k.rmu), rs = to_host(k.rs);
  Acc a{};
  a.max_relp = vmax(relp);
  a.med_relp = median(relp);
  a.max_dvs = vmax(dvs);
  a.med_dvs = median(dvs);
  VecD spin;
  for (int i = 0; i < n; ++i) {
    const Real y = rho[i] / kRhoS;
    if (y > 5.49 && y < 8.61) spin.push_back(dvs[i]);
  }
  a.max_dvs_spin = vmax(spin);
  a.n_spin = static_cast<int>(spin.size());
  a.max_relT = vmax(rT);
  a.max_relmu = vmax(rmu);
  a.max_rels = vmax(rs);
  return a;
}
}  // namespace

TEST_CASE("T2c.3 interpolation accuracy against the analytic EosColdThermal") {
  const spf::ColdCurveTable cold = spf::ColdCurveTable::load(kSrc + "/data/eos/cold_curve.dat");
  const spf::EosColdThermal ref(cold, spf::ColdVariant::two_phase);
  // 2000 points, fixed seed: rho uniform in [0.2, 16] rho_s, eps uniform in (eps_c(rho), 40 eps_s)
  Rng rng(20261120);
  VecD rho(2000), u(2000);
  for (int i = 0; i < 2000; ++i) {
    rho[i] = kRhoS * (0.2 + 15.8 * rng.u());
    u[i] = rng.u();
  }
  const VecD ec = eps_cold_of(ref, rho);
  VecD eps(2000);
  for (int i = 0; i < 2000; ++i) eps[i] = ec[i] + (40.0 * kEpsS - ec[i]) * u[i];

  const EosTableData committed = EosTableData::load(kCommitted);
  const EosTableData def = make_synthetic(cold, 801, 801, 0.05, 0.02);
  struct Case {
    const char* name;
    const EosTableData* t;
  } cases[2] = {{"committed 201x401 (dx 0.2, dy 0.04)", &committed}, {"default 801x801 (dx 0.05, dy 0.02)", &def}};
  for (const Case& c : cases) {
    for (TableInterp ti : {TableInterp::bilinear, TableInterp::bicubic}) {
      const Acc a = accuracy(*c.t, cold, ti, eps, rho);
      std::printf(
          "[T2c.3] %s, %s: p0 rel err max %.4e median %.4e | vs2 abs err max %.4e median %.4e | vs2 abs err max in spinodal region "
          "(5.49<rho/rho_s<8.61, %d points) %.4e | T, mu, s rel err max %.2e %.2e %.2e\n",
          c.name, ti == TableInterp::bilinear ? "bilinear" : "bicubic", a.max_relp, a.med_relp, a.max_dvs, a.med_dvs, a.n_spin, a.max_dvs_spin,
          a.max_relT, a.max_relmu, a.max_rels);
      if (c.t == &def) CHECK(a.max_relp < 1e-3);
      CHECK(a.n_spin > 100);
    }
  }
}

TEST_CASE("T2c.4 clamp below the cold curve") {
  const spf::ColdCurveTable cold = spf::ColdCurveTable::load(kSrc + "/data/eos/cold_curve.dat");
  const spf::EosColdThermal ref(cold, spf::ColdVariant::two_phase);
  const EosTableData t = EosTableData::load(kCommitted);
  Rng rng(7);
  VecD rho, f;
  for (int i = 0; i < 500; ++i) {
    rho.push_back(kRhoS * (0.05 + 15.9 * rng.u()));
    f.push_back(rng.u());
  }
  for (TableInterp ti : {TableInterp::bilinear, TableInterp::bicubic}) {
    const EosTable e = t.eos(ti);
    const VecD em = eps_min_of(e, rho);
    VecD below(rho.size());
    for (size_t i = 0; i < rho.size(); ++i) below[i] = em[i] * f[i];  // 0 <= eps < eps_min
    const Out ob = eval(e, below, rho), oc = eval(e, em, rho);
    Real dp = 0.0, dpe = 0.0;
    for (size_t i = 0; i < rho.size(); ++i) {
      if (below[i] < em[i]) {
        dp = std::max(dp, std::abs(ob.p[i] - oc.p[i]));
        dpe = std::max(dpe, std::abs(ob.dpe[i]));
      }
    }
    // dp_drho in the clamped region is the derivative of the clamped function: compare with centred differences in rho
    const auto fdv = fd(e, below, rho, 1e-6);
    Real dpr = 0.0;
    for (size_t i = 0; i < rho.size(); ++i) {
      if (below[i] < em[i] && below[i] > 0.0) dpr = std::max(dpr, std::abs(ob.dpr[i] - fdv.second[i]) / std::max(std::abs(fdv.second[i]), 1e-3));
    }
    // the table cold curve agrees with the analytic one
    const VecD ec = eps_cold_of(ref, rho);
    Real dcold = 0.0;
    for (size_t i = 0; i < rho.size(); ++i) dcold = std::max(dcold, std::abs(em[i] - ec[i]) / std::max(ec[i], 1e-3));
    std::printf("[T2c.4] %s: max |p0(eps<eps_min) - p0(eps_min)| = %.1e, max |dp_deps| = %.1e, dp_drho vs centred difference (rel to max(|fd|,1e-3)) %.2e, eps_min(rho) vs analytic eps_c rel %.2e\n",
                ti == TableInterp::bilinear ? "bilinear" : "bicubic", dp, dpe, dpr, dcold);
    CHECK(dp == 0.0);
    CHECK(dpe == 0.0);
    CHECK(dpr < 1e-5);
    CHECK(dcold < 1e-4);
  }
}

TEST_CASE("T2c.5 derivative consistency with centred finite differences") {
  const EosTableData t = EosTableData::load(kCommitted);
  const int nx = t.lat.nx, ny = t.lat.ny;
  for (TableInterp ti : {TableInterp::bilinear, TableInterp::bicubic}) {
    const EosTable e = t.eos(ti);
    Rng rng(11);
    VecD eps, rho;
    int guard = 0;
    while (eps.size() < 200 && guard++ < 100000) {
      const int ix = 1 + static_cast<int>(rng.u() * (nx - 3)), iy = 1 + static_cast<int>(rng.u() * (ny - 3));
      const Real tt = 0.15 + 0.7 * rng.u(), uu = 0.15 + 0.7 * rng.u();
      const Real x = t.lat.x0 + (ix + tt) * t.lat.dx, y = t.lat.y0 + (iy + uu) * t.lat.dy;
      const VecD em = eps_min_of(e, VecD{y * kRhoS});
      if (x * kEpsS < em[0] + 3.0 * t.lat.dx * kEpsS) continue;  // stay above the cold curve (no clamp inside the stencil)
      eps.push_back(x * kEpsS);
      rho.push_back(y * kRhoS);
    }
    REQUIRE(eps.size() == 200);
    const Out o = eval(e, eps, rho);
    const auto fdv = fd(e, eps, rho, 1e-6);
    // dp_drho vanishes identically on the conformal plasma branch (p = eps/3), where a relative comparison is undefined: points with
    // |dp_drho| < 1e-3 GeV (the natural scale is 0.3-1 GeV) are compared in absolute terms instead.
    Real wx = 0.0, wy = 0.0, wy_abs = 0.0;
    int n_small = 0;
    for (size_t i = 0; i < eps.size(); ++i) {
      wx = std::max(wx, rel(o.dpe[i], fdv.first[i]));
      if (std::abs(fdv.second[i]) >= 1e-3) wy = std::max(wy, rel(o.dpr[i], fdv.second[i]));
      else {
        ++n_small;
        wy_abs = std::max(wy_abs, std::abs(o.dpr[i] - fdv.second[i]));
      }
    }
    std::printf("[T2c.5] %s: 200 interior points, max relative difference dp_deps %.3e; dp_drho %.3e (%d points with |dp_drho| >= 1e-3 GeV), and max absolute difference %.2e GeV at the %d points with |dp_drho| < 1e-3 GeV\n",
                ti == TableInterp::bilinear ? "bilinear" : "bicubic", wx, wy, 200 - n_small, wy_abs, n_small);
    CHECK(wx < 1e-6);
    CHECK(wy < 1e-6);
    CHECK(wy_abs < 1e-9);
  }
}

TEST_CASE("T2c.6 scaled wrapper") {
  const spf::ColdCurveTable cold = spf::ColdCurveTable::load(kSrc + "/data/eos/cold_curve.dat");
  const spf::EosColdThermal ct(cold, spf::ColdVariant::two_phase);
  const EosTableData t = EosTableData::load(kCommitted);
  struct BC {
    Real b, c;
    const char* name;
  } sets[2] = {{1.0 / 1.3, 1.0 / 1.3, "b=c=1/1.3"}, {0.8, 1.25, "b=0.8, c=1.25"}};

  // random points (some below the cold curve of the base)
  Rng rng(3);
  VecD eps0(100), rho0(100);
  for (int i = 0; i < 100; ++i) {
    rho0[i] = kRhoS * (0.1 + 15.0 * rng.u());
    eps0[i] = kEpsS * (0.2 + 38.0 * rng.u());
  }
  auto run_identities = [&](const char* label, auto base, const BC& s) {
    using B = decltype(base);
    VecD eps(100), rho(100);
    for (int i = 0; i < 100; ++i) {
      eps[i] = eps0[i] / s.b;
      rho[i] = rho0[i] / s.c;
    }
    ScaleCheckK<B> k{base, EosScaled<B>(base, s.b, s.c), s.b, s.c, to_device(eps), to_device(rho), DView("d", 400)};
    Kokkos::parallel_for("scale_check", Kokkos::RangePolicy<ExecSpace>(0, 100), k);
    const VecD d = to_host(k.d);
    Real w[4] = {0, 0, 0, 0};
    for (int i = 0; i < 100; ++i)
      for (int q = 0; q < 4; ++q) w[q] = std::max(w[q], d[4 * i + q]);
    std::printf("[T2c.6] %s, %s: identities (max relative difference) p0 %.1e, dp_deps %.1e, dp_drho %.1e, vs2 %.1e\n", label, s.name, w[0], w[1], w[2], w[3]);
    for (int q = 0; q < 4; ++q) CHECK(w[q] < 1e-13);
  };
  auto run_vs2 = [&](const char* label, auto base, const BC& s, const VecD& e_base, const VecD& r_base) {
    using B = decltype(base);
    const EosScaled<B> w(base, s.b, s.c);
    VecD eps(e_base.size()), rho(e_base.size());
    for (size_t i = 0; i < e_base.size(); ++i) {
      eps[i] = e_base[i] / s.b;
      rho[i] = r_base[i] / s.c;
    }
    const Out o = eval(w, eps, rho);
    const VecD iso = isentropic_vs2(w, eps, rho, 2e-5);
    Real wd = 0.0;
    for (size_t i = 0; i < eps.size(); ++i) wd = std::max(wd, std::abs(o.vs[i] - iso[i]));
    std::printf("[T2c.6] %s, %s: max |vs2_wrapper - finite-difference isentropic derivative of wrapped p0| = %.2e (%zu points)\n", label, s.name, wd, eps.size());
    CHECK(wd < 1e-6);
  };

  // cell-centre points for the finite-difference test: cold-curve cells for EosColdThermal, lattice cells for the table
  VecD ec_pts_r, ec_pts_e, tb_r, tb_e;
  {
    Rng r2(5);
    for (int i = 0; i < 20; ++i) ec_pts_r.push_back((10.0 + static_cast<int>(r2.u() * 3100)) * cold.d_rho + 0.5 * cold.d_rho);
    const VecD ec = eps_cold_of(ct, ec_pts_r);
    for (int i = 0; i < 20; ++i) ec_pts_e.push_back(ec[i] + 0.2 + 3.0 * r2.u());
    VecD ix_r;
    for (int i = 0; i < 20; ++i) {
      const int iy = 6 + static_cast<int>(r2.u() * 390);
      tb_r.push_back((iy + 0.5) * t.lat.dy * kRhoS);
      ix_r.push_back(5 + r2.u() * 190);
    }
    const VecD em = eps_min_of(t.eos(), tb_r);
    for (int i = 0; i < 20; ++i) {
      const Real cell = t.lat.dx * kEpsS;
      int ix = static_cast<int>(ix_r[i]);
      const int ix_min = static_cast<int>(std::ceil(em[i] / cell)) + 4;  // cell centre well above the cold curve
      if (ix < ix_min) ix = ix_min;
      tb_e.push_back((ix + 0.5) * cell);
    }
  }

  for (const BC& s : sets) {
    run_identities("EosColdThermal", ct, s);
    run_vs2("EosColdThermal", ct, s, ec_pts_e, ec_pts_r);
    for (TableInterp ti : {TableInterp::bilinear, TableInterp::bicubic}) {
      const EosTable e = t.eos(ti);
      const char* nm = ti == TableInterp::bilinear ? "EosTable bilinear" : "EosTable bicubic";
      run_identities(nm, e, s);
      run_vs2(nm, e, s, tb_e, tb_r);
      // T, mu, s
      VecD eps(100), rho(100);
      for (int i = 0; i < 100; ++i) {
        eps[i] = eps0[i] / s.b;
        rho[i] = rho0[i] / s.c;
      }
      ScaleCheckTmuSK k{e, EosScaled<EosTable>(e, s.b, s.c), s.b, s.c, to_device(eps), to_device(rho), DView("d", 300)};
      Kokkos::parallel_for("scale_check_tmus", Kokkos::RangePolicy<ExecSpace>(0, 100), k);
      const VecD d = to_host(k.d);
      Real w[3] = {0, 0, 0};
      for (int i = 0; i < 100; ++i)
        for (int q = 0; q < 3; ++q) w[q] = std::max(w[q], d[3 * i + q]);
      std::printf("[T2c.6] %s, %s: T/b, (c/b) mu, s identities (max relative difference) %.1e, %.1e, %.1e\n", nm, s.name, w[0], w[1], w[2]);
      for (int q = 0; q < 3; ++q) CHECK(w[q] < 1e-13);
    }
  }

  // b = c = 1 reproduces the base exactly
  {
    VecD eps(100), rho(100);
    for (int i = 0; i < 100; ++i) {
      eps[i] = eps0[i];
      rho[i] = rho0[i];
    }
    auto exact = [&](const char* label, auto base) {
      using B = decltype(base);
      const Out a = eval(base, eps, rho), b = eval(EosScaled<B>(base, 1.0, 1.0), eps, rho);
      Real w = 0.0;
      for (int i = 0; i < 100; ++i)
        w = std::max({w, std::abs(a.p[i] - b.p[i]), std::abs(a.dpe[i] - b.dpe[i]), std::abs(a.dpr[i] - b.dpr[i]), std::abs(a.vs[i] - b.vs[i])});
      std::printf("[T2c.6] %s, b=c=1: max absolute difference to the base %.1e\n", label, w);
      CHECK(w == 0.0);
    };
    exact("EosColdThermal", ct);
    exact("EosTable bilinear", t.eos(TableInterp::bilinear));
    exact("EosTable bicubic", t.eos(TableInterp::bicubic));
  }
}

TEST_CASE("T2c.7 range handling") {
  const EosTableData t = EosTableData::load(kCommitted);
  const EosTable e0 = t.eos(TableInterp::bilinear, false);
  const Real elo = e0.eps_lattice_min(), ehi = e0.eps_lattice_max(), rlo = e0.rho_lattice_min(), rhi = e0.rho_lattice_max();
  CHECK(ehi == doctest::Approx(40.0 * kEpsS));
  CHECK(rhi == doctest::Approx(16.0 * kRhoS));

  // check_range
  CHECK(e0.check_range(elo, ehi, rlo, rhi).ok);
  CHECK(e0.check_range(0.5, 3.0, 0.2, 1.0).ok);
  CHECK(t.check_range(0.5, 3.0, 0.2, 1.0).ok);
  auto r1 = e0.check_range(0.5, ehi * 1.01, 0.2, 1.0);
  CHECK((!r1.ok && r1.eps_high && !r1.eps_low && !r1.rho_low && !r1.rho_high));
  auto r2 = e0.check_range(-0.1, 3.0, 0.2, 1.0);
  CHECK((!r2.ok && r2.eps_low && !r2.eps_high));
  auto r3 = e0.check_range(0.5, 3.0, -0.01, 1.0);
  CHECK((!r3.ok && r3.rho_low && !r3.rho_high));
  auto r4 = e0.check_range(0.5, 3.0, 0.2, rhi * 1.01);
  CHECK((!r4.ok && r4.rho_high && !r4.rho_low));
  CHECK(r4.message().find("rho above the lattice") != std::string::npos);
  CHECK(!t.check_range(0.5, ehi * 2, 0.2, rhi * 2).ok);

  // extrapolate = false: arguments are clamped to the lattice
  for (TableInterp ti : {TableInterp::bilinear, TableInterp::bicubic}) {
    const EosTable e = t.eos(ti, false);
    const VecD eps = {ehi * 1.5, ehi * 3.0, 2.0, 2.0, ehi * 1.5, 5.0, -3.0};
    const VecD rho = {0.8, 0.8, rhi * 1.5, rhi * 4.0, rhi * 2.0, -0.2, 0.8};
    const VecD ec = {ehi, ehi, 2.0, 2.0, ehi, 5.0, 0.0};  // clamped arguments
    const VecD rc = {0.8, 0.8, rhi, rhi, rhi, 0.0, 0.8};
    const Out a = eval(e, eps, rho), b = eval(e, ec, rc);
    Real w = 0.0;
    for (size_t i = 0; i < eps.size(); ++i) w = std::max(w, std::abs(a.p[i] - b.p[i]));
    std::printf("[T2c.7] %s, extrapolate=false: max |p0(outside) - p0(clamped argument)| = %.1e; check_range reports eps_high, rho_high, rho_low, eps_low\n",
                ti == TableInterp::bilinear ? "bilinear" : "bicubic", w);
    CHECK(w == 0.0);
    CHECK(a.dpe[0] == 0.0);
    CHECK(a.dpr[2] == 0.0);
  }

  // extrapolate = true: a linear function stored in a table is reproduced outside the lattice
  EosTableLattice lat;
  lat.x0 = 0.5;
  lat.dx = 0.25;
  lat.nx = 40;
  lat.y0 = 0.2;
  lat.dy = 0.1;
  lat.ny = 30;
  EosTableData lin = EosTableData::allocate(lat, "linear test table");
  const Real pa = 0.3, pb = 0.17, pc = -0.08;  // p = pa + pb x + pc y
  auto f_p = [&](Real x, Real y) { return pa + pb * x + pc * y; };
  auto f_T = [&](Real x, Real y) { return 0.01 + 0.02 * x + 0.005 * y; };
  auto f_mu = [&](Real x, Real y) { return 0.94 + 0.01 * x - 0.02 * y; };
  auto f_s = [&](Real x, Real y) { return 0.1 * x + 0.3 * y; };
  {
    auto hp = Kokkos::create_mirror_view(lin.p0);
    auto hT = Kokkos::create_mirror_view(lin.T);
    auto hmu = Kokkos::create_mirror_view(lin.mu);
    auto hs = Kokkos::create_mirror_view(lin.s);
    auto he = Kokkos::create_mirror_view(lin.eps_min);
    for (int ix = 0; ix < lat.nx; ++ix)
      for (int iy = 0; iy < lat.ny; ++iy) {
        const Real x = lat.x0 + ix * lat.dx, y = lat.y0 + iy * lat.dy;
        const size_t k = static_cast<size_t>(ix) * lat.ny + iy;
        hp(k) = f_p(x, y);
        hT(k) = f_T(x, y);
        hmu(k) = f_mu(x, y);
        hs(k) = f_s(x, y);
      }
    for (int iy = 0; iy < lat.ny; ++iy) he(iy) = 0.0;  // below the lattice: the clamp never acts
    Kokkos::deep_copy(lin.p0, hp);
    Kokkos::deep_copy(lin.T, hT);
    Kokkos::deep_copy(lin.mu, hmu);
    Kokkos::deep_copy(lin.s, hs);
    Kokkos::deep_copy(lin.eps_min, he);
  }
  const Real xs[] = {0.1, 0.5, 3.0, 10.0, 14.0, 25.0}, ys[] = {-1.1, 0.2, 1.7, 3.1, 3.5, 6.0};  // inside and outside both ranges
  VecD eps, rho;
  for (Real x : xs)
    for (Real y : ys) {
      eps.push_back(x * kEpsS);
      rho.push_back(y * kRhoS);
    }
  for (TableInterp ti : {TableInterp::bilinear, TableInterp::bicubic}) {
    const EosTable e = lin.eos(ti, true);
    const Out o = eval(e, eps, rho);
    const OutTmuS m = eval_tmus(e, eps, rho);
    Real w = 0.0, wd = 0.0;
    for (size_t i = 0; i < eps.size(); ++i) {
      const Real x = eps[i] / kEpsS, y = rho[i] / kRhoS;
      w = std::max({w, rel(o.p[i], f_p(x, y)), rel(m.T[i], f_T(x, y)), rel(m.mu[i], f_mu(x, y)), rel(m.s[i], f_s(x, y))});
      wd = std::max({wd, rel(o.dpe[i], pb / kEpsS), rel(o.dpr[i], pc / kRhoS)});
    }
    std::printf("[T2c.7] %s, extrapolate=true: linear table reproduced at %zu points inside and outside the lattice, max relative error of p0, T, mu, s %.2e, of dp_deps, dp_drho %.2e\n",
                ti == TableInterp::bilinear ? "bilinear" : "bicubic", eps.size(), w, wd);
    CHECK(w < 1e-12);
    CHECK(wd < 1e-12);
  }
}
