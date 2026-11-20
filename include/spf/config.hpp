#pragma once
// Run configuration: every modelling choice is a named key (docs/decisions_spinodal.md).
//
//   defaults file  ->  optional override file(s)  ->  optional "a.b=value" strings
//
// An override may only set a key that exists in the defaults, with the same JSON type.
// Host-side only: values are read once in the driver and passed to kernels as plain numbers.
#include <nlohmann/json.hpp>

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

}  // namespace spf
