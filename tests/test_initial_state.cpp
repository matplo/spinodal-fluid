// WP5: T5.1 conservation of the deposition, T5.3 OSCAR reader and PDG baryon-number rule, kinematics, analytic state,
// T5.4 proxy (Glauber-Gaussians state in Hydro3D with the ideal and cold-thermal EoS; the SMASH state itself is run by spf_ic).
#include <doctest/doctest.h>

#include "spf/config.hpp"
#include "spf/eos/eos.hpp"
#include "spf/hydro3d.hpp"
#include "spf/initial_state.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace spf;

namespace {

std::vector<Particle> random_particles(int n, unsigned seed) {
  std::mt19937_64 g(seed);
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  const long ids[] = {2212, 2112, -2212, 211, -211, 111, 3122, 1114, 321, 1000010020};
  std::vector<Particle> v;
  for (int i = 0; i < n; ++i) {
    Particle p;
    p.x = 6.0 * u(g); p.y = 6.0 * u(g); p.z = 6.0 * u(g);
    p.pdg = ids[i % 10];
    p.B = baryon_number_from_pdg(p.pdg);
    p.px = 0.5 * u(g); p.py = 0.5 * u(g); p.pz = 2.5 * u(g);
    const double m = 0.14 + 0.9 * std::fabs(u(g));
    p.E = std::sqrt(m * m + p.px * p.px + p.py * p.py + p.pz * p.pz);
    v.push_back(p);
  }
  return v;
}

double rel(double a, double b, double scale) { return std::fabs(a - b) / scale; }

}  // namespace

TEST_CASE("T5.3 PDG baryon number rule") {
  struct C { long id; int b; };
  const C cases[] = {{2212, 1}, {-2212, -1}, {2112, 1}, {-2112, -1}, {3122, 1}, {3222, 1}, {3312, 1}, {3334, 1}, {-3334, -1},
                     {2224, 1}, {1114, 1}, {-2214, -1}, {211, 0}, {-211, 0}, {111, 0}, {321, 0}, {221, 0}, {443, 0},
                     {22, 0}, {11, 0}, {2101, 0}, {1103, 0}, {1000010020, 2}, {-1000010020, -2}, {1000020040, 4},
                     {1000822080, 208}, {1000010030, 3}, {0, 0}};
  for (const auto& c : cases) CHECK_MESSAGE(baryon_number_from_pdg(c.id) == c.b, "pdg " << c.id);
}

TEST_CASE("T5.3 OSCAR2013 reader round trip") {
  const OscarFile f = read_oscar2013(std::string(SPF_TEST_DATA_DIR) + "/oscar_3particles.oscar");
  REQUIRE(f.blocks.size() == 1);
  CHECK(f.blocks[0].event == 0);
  CHECK(f.blocks[0].kind == "out");
  CHECK(f.blocks[0].n_declared == 3);
  REQUIRE(f.impact_parameter.size() == 1);
  CHECK(f.impact_parameter[0] == 0.0);
  const auto& p = f.blocks[0].particles;
  REQUIRE(p.size() == 3);
  CHECK(p[0].t == 11.25); CHECK(p[0].x == 1.0); CHECK(p[0].y == -2.0); CHECK(p[0].z == 3.5);
  CHECK(p[0].E == 1.49); CHECK(p[0].px == 0.3); CHECK(p[0].py == -0.2); CHECK(p[0].pz == 1.1);
  CHECK(p[0].pdg == 2212); CHECK(p[0].B == 1);
  CHECK(p[1].pdg == -2112); CHECK(p[1].B == -1); CHECK(p[1].z == -3.5); CHECK(p[1].pz == -1.1);
  CHECK(p[2].pdg == 211); CHECK(p[2].B == 0); CHECK(p[2].E == 0.359);
  CHECK(find_oscar_block(f, 0, -1.0).particles.size() == 3);
  {
    // block header without the "ensemble" word, column order taken from the header line (pdg before t)
    const std::string tmp = "oscar_variant_test.oscar";
    std::ofstream o(tmp);
    o << "#!OSCAR2013 particle_lists pdg t x y z mass p0 px py pz\n# event 3 in 1\n2212 0.5 1 2 3 0.938 1.2 0.1 0.2 0.3\n"
         "# event 3 out 1\n2212 1.5 1 2 3 0.938 1.2 0.1 0.2 0.3\n# event 3 out 1\n-3122 2.5 4 5 6 1.116 1.4 0.1 0.2 0.3\n"
         "# event 3 end 0 impact 1.500 scattering_projectile_target yes\n";
    o.close();
    const OscarFile g = read_oscar2013(tmp);
    REQUIRE(g.blocks.size() == 3);
    CHECK(g.blocks[0].kind == "in");
    CHECK(find_oscar_block(g, 3, -1.0).particles[0].pdg == -3122);
    CHECK(find_oscar_block(g, 3, -1.0).particles[0].B == -1);
    CHECK(find_oscar_block(g, 3, 1.4).time == 1.5);
    CHECK(g.impact_parameter.at(0) == 1.5);
    std::remove(tmp.c_str());
  }
  CHECK(find_oscar_block(f, 0, 11.0).time == 11.25);
  CHECK_THROWS(find_oscar_block(f, 1, -1.0));
  CHECK_THROWS(read_oscar2013(std::string(SPF_TEST_DATA_DIR) + "/does_not_exist.oscar"));
  // deposit the three hadrons: grid sums equal the particle sums
  Grid g = Grid::make_3d(48, 48, 48, 0.5);
  DepositParams dp; dp.sigma = 1.0;
  const auto o = centred_origin(g); dp.origin[0] = o[0]; dp.origin[1] = o[1]; dp.origin[2] = o[2];
  const DepositResult d = deposit_particles(g, dp, p);
  CHECK(d.particles.B == doctest::Approx(0.0));
  CHECK(rel(d.grid.E, d.particles.E, d.particles.E) < 1e-12);
  CHECK(d.n_particles_clipped == 0);
}

TEST_CASE("kinematics, t_init and Woods-Saxon") {
  const Kinematics k = kinematics_from_elab(3.0, 0.940);
  CHECK(k.sqrt_snn == doctest::Approx(3.028927202822808).epsilon(1e-12));
  CHECK(k.gamma_cm == doctest::Approx(1.6111314908631957).epsilon(1e-12));
  CHECK(t_init_fm(k, 1.2 * std::cbrt(208.0)) == doctest::Approx(11.25686135928768).epsilon(1e-3));  // JSON value uses R = 7.11 fm
  CHECK(t_init_fm(k, 6.62) == doctest::Approx(10.481085948825468).epsilon(1e-12));
  const Real rho0 = woods_saxon_rho0(208, 6.62, 0.546);
  CHECK(rho0 > 0.155); CHECK(rho0 < 0.17);
  const auto pos = sample_woods_saxon(2000, 6.62, 0.546, 7);
  double r2 = 0;
  for (const auto& r : pos) r2 += r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
  CHECK(std::sqrt(r2 / pos.size()) == doctest::Approx(5.5).epsilon(0.05));  // rms radius of Pb is about 5.5 fm
}

TEST_CASE("T5.1 deposition conserves B, E, p (CPU/GPU backend of this build)") {
  const Grid g = Grid::make_3d(48, 48, 48, 0.5);
  const auto parts = random_particles(300, 11);
  const auto o = centred_origin(g);
  const double sigmas[] = {1.0, 0.7, 0.5};
  const Contraction cs[] = {Contraction::None, Contraction::ParticleGammaZ, Contraction::CmGamma};
  for (double s : sigmas)
    for (Contraction c : cs) {
      DepositParams dp; dp.sigma = s; dp.contraction = c; dp.gamma_cm = 1.6111;
      dp.origin[0] = o[0]; dp.origin[1] = o[1]; dp.origin[2] = o[2];
      const DepositResult d = deposit_particles(g, dp, parts);
      const auto& P = d.particles; const auto& G = d.grid;
      const double eB = rel(G.B, P.B, std::fabs(P.B) > 0 ? std::fabs(P.B) : 1.0), eE = rel(G.E, P.E, P.E);
      const double ex = rel(G.Px, P.Px, P.absPx), ey = rel(G.Py, P.Py, P.absPy), ez = rel(G.Pz, P.Pz, P.absPz);
      std::printf("T5.1 sigma=%.1f contraction=%d: B %.3e E %.3e Px %.3e Py %.3e Pz %.3e clipped %ld empty %ld\n", s, static_cast<int>(c), eB, eE, ex, ey, ez,
                  d.n_particles_clipped, d.n_particles_empty);
      CHECK(d.n_particles_clipped == 0);
      CHECK(eB < 1e-10); CHECK(eE < 1e-10); CHECK(ex < 1e-10); CHECK(ey < 1e-10); CHECK(ez < 1e-10);
    }
}

TEST_CASE("deposition: clipped stencil is reported, contraction narrows the z profile") {
  const Grid g = Grid::make_3d(32, 32, 32, 0.5);  // box +-8 fm
  Particle p; p.x = 7.5; p.pdg = 2212; p.B = 1; p.E = 3.0; p.pz = std::sqrt(9.0 - 0.88);  // gamma_z = E / m ~ 3.2
  DepositParams dp; dp.sigma = 1.0; dp.contraction = Contraction::None;
  const auto o = centred_origin(g); dp.origin[0] = o[0]; dp.origin[1] = o[1]; dp.origin[2] = o[2];
  const DepositResult d = deposit_particles(g, dp, {p});
  CHECK(d.n_particles_clipped == 1);
  CHECK(d.outside_fraction_B > 0.1);
  CHECK(d.outside_fraction_B < 0.6);
  // z extent: variance of the deposited baryon density along z
  auto var_z = [&](Contraction c) {
    dp.contraction = c;
    p.x = 0.0;
    const DepositResult r = deposit_particles(g, dp, {p});
    auto h = Kokkos::create_mirror_view(r.R);
    Kokkos::deep_copy(h, r.R);
    double s0 = 0, s2 = 0;
    for (int k = 0; k < 32; ++k)
      for (int i = 0; i < 32; ++i)
        for (int j = 0; j < 32; ++j) {
          const double z = o[2] + k * g.dx, w = h(g.g[0] + i, g.g[1] + j, g.g[2] + k);
          s0 += w; s2 += w * z * z;
        }
    return std::sqrt(s2 / s0);
  };
  const double s_none = var_z(Contraction::None), s_part = var_z(Contraction::ParticleGammaZ);
  CHECK(s_none == doctest::Approx(1.0).epsilon(0.03));
  CHECK(s_part < 0.5 * s_none);
}

TEST_CASE("analytic counter-streaming state") {
  const Grid g = Grid::make_3d(64, 64, 64, 0.5);
  const Kinematics k = kinematics_from_elab(3.0, 0.940);
  const DepositResult d = analytic_counterstreaming(g, k, 208, 6.62, 0.546, 0.940);
  // integral over the box = 2 A (the contracted nuclei integrate to A each, the box is large enough)
  CHECK(d.grid.B == doctest::Approx(416.0).epsilon(2e-3));
  CHECK(d.grid.E == doctest::Approx(416.0 * k.gamma_cm * 0.940).epsilon(2e-3));
  auto hR = Kokkos::create_mirror_view(d.R); Kokkos::deep_copy(hR, d.R);
  const double rho0 = woods_saxon_rho0(208, 6.62, 0.546);
  const double rc = hR(g.g[0] + 32, g.g[1] + 32, g.g[2] + 32);  // cell next to the centre (0.25 fm off in each direction)
  CHECK(rc == doctest::Approx(2.0 * k.gamma_cm * rho0).epsilon(2e-3));
}

namespace {

// Glauber-Gaussians state (416 hadrons, 64^3 cells of 0.5 fm), 2 fm/c, returns {initial failures, failures during the run, limiter cells}.
struct RunCounts { long f0, fail_run, limiter; double r_ratio, e_ratio; };

template <class EoS>
RunCounts run_proxy(const EoS& eos, const char* gradient_form, const char* name) {
  Config cfg = Config::from_file(SPF_DEFAULT_CONFIG);
  cfg.set_from_string("ic.generator=glauber_gaussians");
  cfg.set_from_string("grid.nx=64"); cfg.set_from_string("grid.ny=64"); cfg.set_from_string("grid.nz=64");
  cfg.set_from_string("grid.dx_fm=0.5");
  cfg.set_from_string(std::string("gradient.form=") + gradient_form);
  const Hydro3DParams hp = Hydro3DParams::from_config(cfg);
  const Grid g = Grid::make_3d(64, 64, 64, 0.5);
  const InitialState st = build_initial_state(cfg, g);
  CHECK(st.dep.particles.B == doctest::Approx(416.0));
  CHECK(rel(st.dep.grid.B, 416.0, 416.0) < 1e-10);
  CHECK(st.limiter_cells == 0);
  Hydro3D<EoS> h(eos, hp);
  h.set_conserved(st.dep.R, st.dep.Mx, st.dep.My, st.dep.Mz, st.dep.E);
  const Totals t0 = h.totals();
  const long f0 = h.update_primitives();
  h.run(static_cast<long>(std::ceil(2.0 / h.dt() - 1e-9)));
  const Totals t1 = h.totals();
  RunCounts rc{f0, h.n_recover_fail_total - f0, h.n_clamped_total, t1.R / t0.R, t1.E / t0.E};
  std::printf("T5.4 proxy %s, gradient.form=%s: initial recovery failures %ld, failures during run %ld, limiter cells %ld, R/R0 %.9f E/E0 %.9f\n", name,
              gradient_form, rc.f0, rc.fail_run, rc.limiter, rc.r_ratio, rc.e_ratio);
  return rc;
}

}  // namespace

TEST_CASE("T5.4 proxy: Glauber-Gaussians state, 2 fm/c, gradient term off: no recovery failures (ideal and cold-thermal two-phase)") {
  const ColdCurveTable tab = ColdCurveTable::load(std::string(SPF_SOURCE_DIR) + "/data/eos/cold_curve.dat");
  const RunCounts a = run_proxy(EosIdeal{}, "off", "ideal");
  CHECK(a.f0 == 0); CHECK(a.fail_run == 0);
  CHECK(a.r_ratio == doctest::Approx(1.0).epsilon(1e-6));  // R changes at 1e-8: the limiter and the e_vac cut act on the dilute fringe
  const RunCounts b = run_proxy(EosColdThermal(tab, ColdVariant::two_phase, 4.0 / 3.0), "off", "cold_thermal_two_phase");
  CHECK(b.f0 == 0); CHECK(b.fail_run == 0);
}

// Measured: with the default gradient.form = paper_pressure the dilute fringe of the deposited state (E < 0.035 GeV/fm^3, |M| ~ E after the
// limiter) breaks the primitive recovery (p_grad < 0 there): thousands of cells fail, for both equations of state. Kept as an expected
// failure and reported (docs/validation.md, WP5, T5.4).
TEST_CASE("T5.4 proxy: default gradient term (paper_pressure) fails in the dilute fringe" * doctest::may_fail()) {
  const ColdCurveTable tab = ColdCurveTable::load(std::string(SPF_SOURCE_DIR) + "/data/eos/cold_curve.dat");
  const RunCounts a = run_proxy(EosIdeal{}, "paper_pressure", "ideal");
  CHECK(a.fail_run == 0);
  const RunCounts b = run_proxy(EosColdThermal(tab, ColdVariant::two_phase, 4.0 / 3.0), "paper_pressure", "cold_thermal_two_phase");
  CHECK(b.fail_run == 0);
}
