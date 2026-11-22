// Tests of the finite-temperature two-phase table and its Maxwell partner (WP2b), reduced test pair data/eos/test/*_test.{json,bin}
// (lattice step 0.2 in eps/eps_s, 0.08 in rho/rho_s) written by tools/eos/finite_T/make_finite_T_table.py together with
// data/eos/test/finite_T_reference.json.  Every value is evaluated in a Kokkos kernel (functors, so that the same source compiles with nvcc).
// Measured numbers are printed as "[T2b.x] ...".
#include <doctest/doctest.h>

#include "spf/eos/eos.hpp"
#include "spf/types.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using spf::EosTable;
using spf::EosTableData;
using spf::Real;
using spf::TableInterp;
using VecD = std::vector<Real>;

namespace {

const std::string kRefDir = std::string(SPF_SOURCE_DIR) + "/data/eos/test/";
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

struct PointK {
  EosTable eos;
  DView eps, rho, p, T, mu, s, vs;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    p(i) = eos.p0(eps(i), rho(i));
    T(i) = eos.T(eps(i), rho(i));
    mu(i) = eos.mu(eps(i), rho(i));
    s(i) = eos.s(eps(i), rho(i));
    vs(i) = eos.vs2(eps(i), rho(i));
  }
};
struct PointOut {
  VecD p, T, mu, s, vs;
};
PointOut eval_points(const EosTable& eos, const VecD& eps, const VecD& rho) {
  const int n = static_cast<int>(eps.size());
  PointK k{eos, to_device(eps), to_device(rho), DView("p", n), DView("T", n), DView("mu", n), DView("s", n), DView("v", n)};
  Kokkos::parallel_for("eval_points", Kokkos::RangePolicy<spf::ExecSpace>(0, n), k);
  return {to_host(k.p), to_host(k.T), to_host(k.mu), to_host(k.s), to_host(k.vs)};
}
Real rel(Real a, Real b) { return a == b ? 0.0 : std::abs(a - b) / std::abs(b); }

}  // namespace

TEST_CASE("T2b.1 reference lattice values of the finite-T test tables (p0, T, mu, s) to 1e-10") {
  std::ifstream in(kRefDir + "finite_T_reference.json");
  REQUIRE(in.good());
  const nlohmann::json ref = nlohmann::json::parse(in);
  const std::vector<std::string> names = {"eos_two_phase_test", "eos_maxwell_test"};
  Real worst = 0.0;
  int n_checked = 0;
  for (const auto& name : names) {
    const EosTableData d = EosTableData::load(kRefDir + name);
    const EosTable eos = d.eos(TableInterp::bilinear, false);
    VecD eps, rho;
    std::vector<nlohmann::json> pts;
    for (const auto& pt : ref.at("lattice_points")) {
      if (pt.at("table").get<std::string>() != name) continue;
      eps.push_back(pt.at("eps_GeV_fm3").get<Real>());
      rho.push_back(pt.at("rho_fm3").get<Real>());
      pts.push_back(pt);
    }
    REQUIRE(!pts.empty());
    const PointOut o = eval_points(eos, eps, rho);
    for (size_t i = 0; i < pts.size(); ++i) {
      worst = std::max({worst, rel(o.p[i], pts[i].at("p0").get<Real>()), rel(o.T[i], pts[i].at("T").get<Real>()),
                        rel(o.mu[i], pts[i].at("mu").get<Real>()), rel(o.s[i], pts[i].at("s").get<Real>())});
      ++n_checked;
    }
  }
  std::printf("[T2b.1] %d lattice points of both test tables: max relative deviation of p0, T, mu, s from the Python values %.3e (tolerance 1e-10)\n", n_checked, worst);
  CHECK(n_checked == 10);
  CHECK(worst <= 1e-10);
}

TEST_CASE("T2b.2 sign of vs2: spinodal (two-phase), outside (two-phase), Maxwell table") {
  std::ifstream in(kRefDir + "finite_T_reference.json");
  REQUIRE(in.good());
  const nlohmann::json ref = nlohmann::json::parse(in);
  int n_checked = 0;
  for (const auto& pt : ref.at("vs2_points")) {
    const std::string name = pt.at("table").get<std::string>();
    const EosTableData d = EosTableData::load(kRefDir + name);
    const EosTable eos = d.eos(TableInterp::bicubic, false);
    const PointOut o = eval_points(eos, {pt.at("eps_GeV_fm3").get<Real>()}, {pt.at("rho_fm3").get<Real>()});
    const int sign = pt.at("sign").get<int>();
    const Real py = pt.at("vs2_python").get<Real>();
    std::printf("[T2b.2] %s, %s: rho = %.4f fm^-3, eps = %.5f GeV/fm^3, vs2 = %.6f (Python %.6f), expected sign %+d\n", pt.at("label").get<std::string>().c_str(),
                name.c_str(), pt.at("rho_fm3").get<Real>(), pt.at("eps_GeV_fm3").get<Real>(), o.vs[0], py, sign);
    CHECK((sign < 0 ? o.vs[0] < 0.0 : o.vs[0] > 0.0));
    CHECK(rel(o.vs[0], py) <= 1e-8);
    ++n_checked;
  }
  CHECK(n_checked == 3);
}
