#pragma once
// Run-time choice of the equation of state outside the kernels: with_eos(cfg, source_dir, f) calls f(eos) with the EoS selected by
// eos.model (generic lambda, returns void). Relative table paths are resolved against source_dir.
#include "spf/eos/eos.hpp"

#include <string>

namespace spf {

template <class F>
void with_eos(const Config& cfg, const std::string& source_dir, F&& f) {
  const EosModel m = eos_model_from_config(cfg);
  if (m == EosModel::ideal) {
    f(EosIdeal{});
  } else if (m == EosModel::gamma_law) {
    f(EosGammaLaw(cfg.get<Real>("eos.gamma_law_gamma"), cfg.get<Real>("eos.gamma_law_mass_GeV")));
  } else if (m == EosModel::cold_thermal_two_phase || m == EosModel::cold_thermal_maxwell) {
    std::string path = cfg.get<std::string>("eos.cold_curve_path");
    if (!path.empty() && path[0] != '/') path = source_dir + "/" + path;
    const ColdCurveTable table = ColdCurveTable::load(path);
    const ColdVariant v = m == EosModel::cold_thermal_two_phase ? ColdVariant::two_phase : ColdVariant::maxwell;
    const EosColdThermal e(table, v, cfg.get<Real>("eos.gamma_th"));
    const Real b = cfg.get<Real>("eos.scale_b"), c = cfg.get<Real>("eos.scale_c");
    if (b == 1.0 && c == 1.0) f(e);
    else f(EosScaled<EosColdThermal>(e, b, c));
  } else {
    const EosTableData data = EosTableData::from_config(cfg);
    const EosTable e = data.eos(cfg);
    const Real b = cfg.get<Real>("eos.scale_b"), c = cfg.get<Real>("eos.scale_c");
    if (b == 1.0 && c == 1.0) f(e);
    else f(EosScaled<EosTable>(e, b, c));
  }
}

}  // namespace spf
