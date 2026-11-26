#pragma once
// WP10e: treatment of the matter below the switching density at the start of the fluid stage ("corona"), decision S31.
//   particlisation.corona.mode = core_corona | thermal | keep
//   core_corona  every hadron of the initial state is classified before the deposition: eps (recovered with the REFERENCE equation of state,
//                particlisation.corona.reference_eos, trilinear to the hadron position) < eps_sw -> corona.  Only core hadrons are deposited;
//                the corona hadrons are written unchanged to <base>.corona.oscar and appended to every sample by the sampler.
//   thermal      all hadrons are deposited; the cells with eps < eps_sw at t_init are emitted thermally at once (decision S23)
//   keep         all hadrons are deposited; the cells stay in the fluid; the initial slice is emitted at the end of the run (surface.hpp)
// Host code plus one device-resident recovery through Hydro3D (the equation of state is only called inside Hydro3D kernels, rule 9).
#include "spf/config.hpp"
#include "spf/hydro3d.hpp"
#include "spf/initial_state.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <string>
#include <vector>

namespace spf {

enum class CoronaMode { CoreCorona, Thermal, Keep };

struct CoronaParams {
  CoronaMode mode = CoronaMode::CoreCorona;
  std::string reference_eos = "table_two_phase";
  static CoronaParams from_config(const Config& cfg);
  const char* mode_name() const;
};

// Sums over a set of hadrons (integer B and Q).
struct CoronaTotals {
  long n = 0;
  double E = 0, Px = 0, Py = 0, Pz = 0;
  long long B = 0, Q = 0;
};

// Charge in units of e of a hadron of the initial state: the OSCAR column if it is non-zero, otherwise the PDG-code rule (nuclei: 0).
int corona_charge(const Particle& p);
// Mass of a hadron: the OSCAR column, or sqrt(E^2 - p^2) when the column is absent.
double corona_mass(const Particle& p);

CoronaTotals corona_totals(const std::vector<Particle>& all, const std::vector<char>& mask, bool select);
nlohmann::json corona_totals_json(const CoronaTotals& t);

// Trilinear interpolation of a cell-centred interior field (host copy, index (i * ny + j) * nz + k as Hydro3D::interior_copy) at the physical
// position (x, y, z); cell (0, 0, 0) is centred at `origin`.  Positions more than half a cell outside the box give 0; inside that margin the
// nearest edge value is used.
double interp_trilinear(const std::vector<Real>& f, const int n[3], Real dx, const Real origin[3], Real x, Real y, Real z);

struct CoronaSplit {
  std::vector<char> is_core;   // per hadron of InitialState::particles
  std::vector<double> eps;     // reference-EoS energy density at each hadron [GeV/fm^3]
  CoronaTotals core, corona;
};

// Classification with the reference equation of state `eref`: all hadrons of st are deposited (st.dep as built by build_initial_state), eps is
// recovered in every cell with Hydro3D<RefEoS> (the finite-range gradient term does not act at t_init), interpolated to the hadron positions;
// corona = eps < eps_sw.
template <class RefEoS>
CoronaSplit classify_corona(const Config& cfg, const Grid& grid, const InitialState& st, const RefEoS& eref, Real eps_sw) {
  CoronaSplit s;
  Hydro3D<RefEoS> hr(eref, Hydro3DParams::from_config(cfg));
  hr.set_conserved(st.dep.R, st.dep.Mx, st.dep.My, st.dep.Mz, st.dep.E);
  hr.update_primitives();
  Kokkos::fence();
  const std::vector<Real> e = hr.interior_copy(hr.eps);
  const auto o = centred_origin(grid);
  const Real org[3] = {o[0], o[1], o[2]};
  const int n[3] = {grid.n[0], grid.n[1], grid.n[2]};
  s.is_core.resize(st.particles.size());
  s.eps.resize(st.particles.size());
  for (size_t i = 0; i < st.particles.size(); ++i) {
    const Particle& p = st.particles[i];
    s.eps[i] = interp_trilinear(e, n, grid.dx, org, p.x, p.y, p.z);
    s.is_core[i] = s.eps[i] >= eps_sw ? 1 : 0;
  }
  s.core = corona_totals(st.particles, s.is_core, true);
  s.corona = corona_totals(st.particles, s.is_core, false);
  return s;
}

// OSCAR2013 file of the corona hadrons (mask == 0), SMASH species, positions, times and momenta unchanged (same column set as the sampler
// output; one event block "event <event_id> ensemble 0 out N"); returns the totals written.
CoronaTotals write_corona_oscar(const std::string& path, const std::vector<Particle>& all, const std::vector<char>& is_core, int event_id,
                                const std::string& comment);

}  // namespace spf
