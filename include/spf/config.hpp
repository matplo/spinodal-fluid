#pragma once
// Run configuration: every modelling choice is a named key (docs/decisions_spinodal.md).
//
//   defaults file  ->  optional override file(s)  ->  optional "a.b=value" strings
//
// An override may only set a key that exists in the defaults, with the same JSON type.
// Host-side only: values are read once in the driver and passed to kernels as plain numbers.
#include <nlohmann/json.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

namespace spf {

class ConfigError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class Config {
 public:
  // Load the defaults file (throws ConfigError if unreadable or not a JSON object).
  static Config from_file(const std::string& defaults_path);
  // Build from an in-memory JSON object (used by tests).
  static Config from_json(const nlohmann::json& defaults);

  // Apply an override file; throws ConfigError on an unknown key or a type mismatch.
  void merge_file(const std::string& path);
  void merge_json(const nlohmann::json& overrides);
  // Apply one "section.key=value" string. The value is parsed as JSON when possible
  // (0.05, true, "text"), otherwise taken as a bare string (phoenical).
  void set_from_string(const std::string& key_eq_value);

  bool has(const std::string& dotted_key) const;

  template <class T>
  T get(const std::string& dotted_key) const {
    const nlohmann::json& v = at(dotted_key);
    try {
      return v.get<T>();
    } catch (const nlohmann::json::exception& e) {
      throw ConfigError("config key '" + dotted_key + "': " + e.what());
    }
  }

  const nlohmann::json& json() const { return data_; }

  // Write <dir>/config_used.json holding the effective configuration and runtime_info().
  // Creates <dir> if needed. Every program that produces output calls this first.
  void echo(const std::string& dir) const;

 private:
  const nlohmann::json& at(const std::string& dotted_key) const;
  void assign(const std::string& dotted_key, const nlohmann::json& value);
  nlohmann::json data_;
};

// Code version, git description, Kokkos execution space and concurrency.
// Requires Kokkos to be initialised.
nlohmann::json runtime_info();

// Grid size from the configuration. grid.nx/ny/nz are the cell counts unless the box lengths grid.Lx_fm/Ly_fm/Lz_fm are positive: then
// n = round(L / dx_fm) in that direction, so that changing grid.dx_fm alone changes the cell counts and not the physical box.
// length_fm is the box length actually used (n * dx). Throws ConfigError when n < 4 or when round(L / dx) * dx differs from L by more than
// 1e-6 of dx + 1e-9 L (a box length that is not a multiple of dx).
struct GridCells {
  int n[3] = {0, 0, 0};
  double length_fm[3] = {0, 0, 0};
  double dx = 0.2;
};

inline GridCells grid_cells(const Config& cfg) {
  GridCells g;
  g.dx = cfg.get<double>("grid.dx_fm");
  if (!(g.dx > 0.0)) throw ConfigError("grid.dx_fm must be positive");
  const char* nk[3] = {"grid.nx", "grid.ny", "grid.nz"};
  const char* lk[3] = {"grid.Lx_fm", "grid.Ly_fm", "grid.Lz_fm"};
  for (int d = 0; d < 3; ++d) {
    const double L = cfg.get<double>(lk[d]);
    if (L > 0.0) {
      const double q = L / g.dx;
      g.n[d] = static_cast<int>(q + 0.5);
      if (std::fabs(g.n[d] * g.dx - L) > 1e-6 * g.dx + 1e-9 * L)
        throw ConfigError(std::string(lk[d]) + " = " + std::to_string(L) + " is not a multiple of grid.dx_fm = " + std::to_string(g.dx));
    } else {
      g.n[d] = cfg.get<int>(nk[d]);
    }
    if (g.n[d] < 4) throw ConfigError(std::string("fewer than 4 cells in direction ") + std::to_string(d));
    g.length_fm[d] = g.n[d] * g.dx;
  }
  return g;
}

}  // namespace spf
