#pragma once
// Template members of ObservableRecorder (included by observables.hpp).
#include <algorithm>
#include <cmath>
#include <filesystem>

namespace spf {

template <class EoS>
ObservableRecorder<EoS>::ObservableRecorder(const Config& cfg, const EoS& eos, const std::string& dir)
    : mask_{eos}, prm_(ObsParams::from_config(cfg)), dir_(dir) {
  cfg.echo(dir);
  csv_.open(dir + "/observables.csv");
  if (!csv_) throw std::runtime_error("cannot open " + dir + "/observables.csv");
  csv_ << obs_csv_header() << "\n";
  csv_.flush();
  pending_ = prm_.hist_times;
  std::sort(pending_.begin(), pending_.end());
}

template <class EoS>
void ObservableRecorder<EoS>::record(const ObsFields& f, Real time, long step) {
  const ObsResult r = compute_observables(f, prm_, mask_, time, step);
  hist_.push_back(r);
  csv_ << obs_csv_line(r) << "\n";
  csv_.flush();
  if (std::isnan(fluid_end_time_) && r.eps_max_eps_s < prm_.fluid_end_eps_eps_s) {
    fluid_end_time_ = time;
    fluid_end_step_ = step;
  }
  if (!pending_.empty() && time >= pending_.front() - 1e-9 * (1.0 + std::fabs(pending_.front()))) {
    write_hists(f, r);
    while (!pending_.empty() && time >= pending_.front() - 1e-9 * (1.0 + std::fabs(pending_.front()))) pending_.erase(pending_.begin());
  }
}

template <class EoS>
void ObservableRecorder<EoS>::write_hists(const ObsFields& f, const ObsResult& r) {
  const ClumpResult c = find_clumps(f, prm_);
  const PhaseHist p = compute_phase_plane(f, prm_, mask_);
  const std::string s = std::to_string(r.step);
  std::ofstream(dir_ + "/clumps_step" + s + ".json") << clump_json(c, prm_, r.time, r.step).dump(1) << "\n";
  std::ofstream(dir_ + "/phase_step" + s + ".json") << phase_json(p, prm_, r.time, r.step, r.unstable_fraction).dump(1) << "\n";
}

template <class EoS>
void ObservableRecorder<EoS>::finalize() {
  nlohmann::json j;
  j["n_records"] = hist_.size();
  if (std::isnan(fluid_end_time_)) { j["fluid_end_time_fm"] = nullptr; j["fluid_end_step"] = nullptr; }
  else { j["fluid_end_time_fm"] = fluid_end_time_; j["fluid_end_step"] = fluid_end_step_; }
  j["fluid_end_eps_eps_s"] = prm_.fluid_end_eps_eps_s;
  if (!hist_.empty()) {
    for (int n = 2; n <= kObsMaxMoment; ++n) {
      const Real ref = hist_.front().moment[n];
      Real best = -1.0, tbest = hist_.front().time;
      for (const auto& r : hist_) {
        const Real e = ref != 0.0 ? r.moment[n] / ref : 0.0;
        if (e > best) { best = e; tbest = r.time; }
      }
      j["max_enhancement"]["N" + std::to_string(n)] = {{"value", best}, {"time_fm", tbest}};
    }
    j["reference_time_fm"] = hist_.front().time;
  }
  std::ofstream(dir_ + "/observables_summary.json") << j.dump(2) << "\n";
}

template <class EoS>
void observe_snapshots(const Config& cfg, const EoS& eos, const std::vector<std::string>& bases, const std::string& outdir) {
  ObservableRecorder<EoS> rec(cfg, eos, outdir);
  for (const auto& b : bases) {
    LoadedSnapshot s = load_snapshot_fields(b);
    rec.record(s.fields, s.time, s.step);
  }
  rec.finalize();
}

}  // namespace spf
