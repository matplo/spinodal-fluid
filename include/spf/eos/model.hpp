#pragma once
// Host-side selection of the equation of state from the configuration (key eos.model).
// The run-time switch over the EoS type lives in the driver, outside the kernels.
#include "spf/config.hpp"

#include <string>

namespace spf {

enum class EosModel { ideal, gamma_law, cold_thermal_two_phase, cold_thermal_maxwell, table_two_phase, table_maxwell };

inline EosModel parse_eos_model(const std::string& s) {
  if (s == "ideal") return EosModel::ideal;
  if (s == "gamma_law") return EosModel::gamma_law;
  if (s == "cold_thermal_two_phase") return EosModel::cold_thermal_two_phase;
  if (s == "cold_thermal_maxwell") return EosModel::cold_thermal_maxwell;
  if (s == "table_two_phase") return EosModel::table_two_phase;
  if (s == "table_maxwell") return EosModel::table_maxwell;
  throw ConfigError("unknown eos.model '" + s + "'");
}

inline const char* eos_model_name(EosModel m) {
  switch (m) {
    case EosModel::ideal: return "ideal";
    case EosModel::gamma_law: return "gamma_law";
    case EosModel::cold_thermal_two_phase: return "cold_thermal_two_phase";
    case EosModel::cold_thermal_maxwell: return "cold_thermal_maxwell";
    case EosModel::table_two_phase: return "table_two_phase";
    case EosModel::table_maxwell: return "table_maxwell";
  }
  return "";
}

inline EosModel eos_model_from_config(const Config& cfg) { return parse_eos_model(cfg.get<std::string>("eos.model")); }

}  // namespace spf
