#pragma once
// WP5: initial states for the fluid stage.
//   * host side: OSCAR2013 particle-list reader, PDG baryon-number rule, collision kinematics, Woods-Saxon sampling;
//   * device side: Gaussian deposition of hadrons on the grid (named functors, docs/device_rules.md rules 1, 2, 6);
//   * conserved fields R = N^0, M_j = T^0j, E = T^00 handed to Hydro3D::set_conserved.
// Docs: docs/initial_state.md.
#include "spf/config.hpp"
#include "spf/shasta.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace spf {

// ------------------------------------------------------------------ particles and OSCAR2013
struct Particle {
  Real t = 0, x = 0, y = 0, z = 0;      // fm
  Real E = 0, px = 0, py = 0, pz = 0;   // GeV
  long pdg = 0;
  int B = 0;                            // baryon number (baryon_number_from_pdg)
  Real mass = 0;                        // GeV, OSCAR column "mass" (0 if the column is absent)
  long id = 0;                          // OSCAR column "ID" (0 if absent)
  int charge = 0;                       // OSCAR column "charge" in units of e (0 if absent)
};

// Baryon number from a PDG Monte Carlo id.
//   |id| >= 1000000000 (nuclei, 10LZZZAAAI): B = sign(id) * AAA, AAA = (|id| / 10) % 1000.
//   otherwise, with n_q1 = (|id| / 1000) % 10, n_q2 = (|id| / 100) % 10, n_q3 = (|id| / 10) % 10:
//   three non-zero quark digits -> baryon, B = sign(id); anything else (mesons, diquarks, leptons, photons) -> 0.
int baryon_number_from_pdg(long pdg);

struct OscarBlock {
  int event = -1;
  std::string kind;           // "in" or "out" (the word after "event N")
  long n_declared = -1;       // particle count of the block header
  Real time = 0;              // t of the first particle (SMASH writes all particles of a block at one time)
  std::vector<Particle> particles;
};

struct OscarFile {
  std::string format_line;                // the "#!OSCAR2013 ..." line
  std::vector<OscarBlock> blocks;         // in file order
  std::vector<Real> impact_parameter;     // per "event N end" line, if present
};

// Reads ASCII OSCAR2013 particle lists. Column order is taken from the "#!OSCAR2013" header line (names t x y z mass p0 px py pz pdg ...),
// default t x y z mass p0 px py pz pdg ID charge. Throws std::runtime_error on an unreadable file, a missing column or a malformed line.
OscarFile read_oscar2013(const std::string& path);

// Block of event `event`: kind "out" whose time is closest to t_target (t_target < 0: the last "out" block of that event).
// Throws std::runtime_error if there is none.
const OscarBlock& find_oscar_block(const OscarFile& f, int event, Real t_target);

// ------------------------------------------------------------------ kinematics
struct Kinematics {
  Real e_lab = 0, gamma_lab = 0, gamma_cm = 0, sqrt_snn = 0, beta_cm = 0;
};
// Fixed target, kinetic energy per nucleon e_lab [GeV], nucleon mass m_n [GeV].
Kinematics kinematics_from_elab(Real e_lab, Real m_n = 0.940);
// t_init = 2 R / sqrt(gamma_cm^2 - 1)  [fm/c] (arXiv:1302.2956, Sect. IV).
Real t_init_fm(const Kinematics& k, Real radius_fm);
// Radius from config: ic.t_init_radius_mode = "r0_A13" (ic.t_init_r0_fm * A^(1/3)) or "fixed" (ic.t_init_radius_fm).
Real t_init_radius_from_config(const Config& cfg, int mass_number);

// Woods-Saxon sampling of nucleon positions (rejection, radial density ~ 1/(1 + exp((r - R)/a))); fixed seed -> reproducible.
std::vector<std::array<Real, 3>> sample_woods_saxon(int A, Real radius_fm, Real a_fm, std::uint64_t seed);
// Central normalised Woods-Saxon density rho0 [fm^-3] with 4 pi int rho r^2 dr = A.
Real woods_saxon_rho0(int A, Real radius_fm, Real a_fm);

// Glauber-like fallback (ic.generator = "glauber_gaussians"): A nucleons per nucleus sampled from Woods-Saxon, both nuclei with
// coinciding centres (full overlap) at the moment of the state, positions contracted along z by gamma_cm, momenta
// +-gamma beta m_N along z, E = gamma m_N.
std::vector<Particle> glauber_particles(const Kinematics& k, int A, Real radius_fm, Real a_fm, std::uint64_t seed, Real m_n = 0.940);

// ------------------------------------------------------------------ deposition
enum class Contraction { None, ParticleGammaZ, CmGamma };
Contraction parse_contraction(const std::string& s);   // "none", "particle_gamma_z", "cm_gamma"; "undecided" throws

struct DepositParams {
  Real sigma = 1.0;          // fm
  Real cutoff = 4.0;         // truncation radius in units of sigma (along z: of sigma / gamma_z)
  Contraction contraction = Contraction::ParticleGammaZ;
  Real gamma_cm = 1.0;       // used by Contraction::CmGamma
  // physical position of the centre of interior cell (0, 0, 0)
  Real origin[3] = {0, 0, 0};
};

// Origin (physical position of the centre of interior cell (0,0,0)) for a box centred on the coordinate origin.
inline std::array<Real, 3> centred_origin(const Grid& g) {
  return {-0.5 * (g.n[0] - 1) * g.dx, -0.5 * (g.n[1] - 1) * g.dx, -0.5 * (g.n[2] - 1) * g.dx};
}

struct Totals5 {
  Real B = 0, E = 0, Px = 0, Py = 0, Pz = 0;
  Real absPx = 0, absPy = 0, absPz = 0;   // sums of |p_j| (scale for relative comparison of momentum sums that cancel)
};

struct DepositResult {
  Field3 R, Mx, My, Mz, E;      // densities (per fm^3) on the grid of the call, ghost layers zero
  Totals5 particles;            // sums over the input particles
  Totals5 grid;                 // sum of density * dx^3 over the interior
  Real outside_fraction_B = 0;  // 1 - grid.B / particles.B (weight of the particle Gaussians that fell outside the box; 0 if no baryons)
  Real outside_fraction_E = 0;
  long n_particles = 0;
  long n_particles_clipped = 0; // particles with part of their stencil outside the box
  long n_particles_empty = 0;   // particles whose cutoff window holds no grid point on some axis (deposited in the nearest point; unresolved)
};

// Deposit with a normalised, truncated Gaussian per hadron. Normalisation is over the full stencil (the grid points within the cutoff,
// whether inside the box or not), so that every particle whose stencil lies inside the box contributes exactly its B, E, p; the part of a
// stencil outside the box is lost and reported in outside_fraction_*. Atomic adds into the output Views (team per particle, threads over
// the stencil cells); the summation order differs between backends, so sums agree to rounding only.
DepositResult deposit_particles(const Grid& grid, const DepositParams& prm, const std::vector<Particle>& particles);

// E >= |M| limiter on the interior: M is rescaled to |M| = E where it is exceeded. Returns the number of cells changed.
long limit_momentum(const Grid& grid, DepositResult& d);

// Two counter-streaming Lorentz-contracted Woods-Saxon nuclei at full overlap: R = 2 gamma rho_WS(sqrt(x^2 + y^2 + gamma^2 z^2)),
// E = 2 gamma^2 m_N rho_WS(...), M = 0 (the momenta of the two streams cancel). Returns fields like deposit_particles (particles/grid
// sums are the integral over the box).
DepositResult analytic_counterstreaming(const Grid& grid, const Kinematics& k, int A, Real radius_fm, Real a_fm, Real m_n = 0.940);

// Totals of the interior of a result (density * dx^3).
Totals5 grid_totals(const Grid& grid, const DepositResult& d);

// ------------------------------------------------------------------ pipeline from the configuration
struct InitialState {
  std::string generator;
  Kinematics kin;
  Real radius_fm = 0, t_init_fm = 0;
  Real oscar_time_fm = -1;          // time column of the chosen OSCAR block (smash_cascade / oscar_file)
  long n_hadrons = 0;
  Real baryon_total = 0, baryon_y_window = 0;   // baryon number of the input, and with |y| < 0.5 (y from E, pz of each baryon)
  DepositResult dep;
  long limiter_cells = 0;
  std::vector<Particle> particles;  // the hadrons of the input list (empty for the analytic generator)
  DepositParams dep_params;         // parameters used for the deposition (reused by redeposit_subset)
};
// ic.generator: "smash_cascade" / "oscar_file" (read ic.oscar_path, event ic.oscar_event, block time ic.oscar_time_fm),
// "glauber_gaussians" (explicit fallback, never chosen silently), "analytic" (counter-streaming Woods-Saxon). Deposition with
// ic.sigma_init_fm, ic.gaussian_contraction, ic.cutoff_sigma; the E >= |M| limiter is applied. Throws std::runtime_error when the
// fraction of baryon number outside the box exceeds ic.max_outside_fraction.
InitialState build_initial_state(const Config& cfg, const Grid& grid);

// WP10e (core/corona split): replaces st.dep by the deposition of the particles with keep[i] != 0 only (same deposition parameters, momentum
// limiter applied again); st.baryon_total, st.baryon_y_window and st.n_hadrons are those of the kept particles; st.particles is unchanged.
// Throws std::runtime_error when keep has the wrong size or the outside fraction of baryon number exceeds ic.max_outside_fraction.
void redeposit_subset(const Config& cfg, const Grid& grid, InitialState& st, const std::vector<char>& keep);

}  // namespace spf
