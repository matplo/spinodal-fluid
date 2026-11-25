#pragma once
// Switching-surface file of one fluid event (docs/particlisation_format.md): <base>.surface.bin + <base>.surface.json.
// 16 little-endian float64 per element: t x y z | dsig_0..3 (lower index) | u^0..3 | eps rho T mu_B.
#include <nlohmann/json.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace spf::particlisation {

class SurfaceError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

constexpr int kSurfaceRecordDoubles = 16;
// record offsets
enum SurfaceField { kT = 0, kX = 1, kY = 2, kZ = 3, kDs = 4, kU = 8, kEps = 12, kRho = 13, kTemp = 14, kMuB = 15 };

struct Surface {
  std::vector<double> rec;  // n * 16
  nlohmann::json meta;      // the sidecar
  std::size_t size() const { return rec.size() / kSurfaceRecordDoubles; }
  const double* at(std::size_t i) const { return rec.data() + i * kSurfaceRecordDoubles; }
  double* at(std::size_t i) { return rec.data() + i * kSurfaceRecordDoubles; }

  // Accessors with the documented key names; fall back to the stated default when the key is absent.
  double dx() const;            // cell size [fm] used to smear the position (key "dx")
  long long event_id() const;   // key "event_id"
  void add_element(const double r[kSurfaceRecordDoubles]) { rec.insert(rec.end(), r, r + kSurfaceRecordDoubles); }
};

// Reads <base>.surface.bin and <base>.surface.json (checks the byte count against "n_elements" and rejects non-finite values).
Surface read_surface(const std::string& base);
// Writes both files (n_elements is set from the records). The caller fills the other sidecar keys.
void write_surface(const std::string& base, const Surface& s);

// Minimal valid sidecar for synthetic surfaces (tests): version 1, method, dx, dt, event id.
nlohmann::json make_surface_meta(const std::string& method, double dx, double dt, long long event_id);

}  // namespace spf::particlisation
