// WP10b: Cooper-Frye particle sampler.
//   spf_sample --surface BASE [--out DIR] [--config FILE] [--override FILE]... [key=value ...]
// Reads <BASE>.surface.bin/.json (docs/particlisation_format.md) and writes <DIR>/<sampler.output_prefix>.oscar (OSCAR2013, one event block per
// sample) and <DIR>/<sampler.output_prefix>.json (per-sample summaries, preparation report, effective configuration is in config_used.json).
#include "spf/config.hpp"
#include "spf/particlisation/sampler.hpp"
#include "spf/types.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  std::string defaults = SPF_DEFAULT_CONFIG, out = "out_sample", surface;
  std::vector<std::string> override_files, assignments;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--config" && i + 1 < argc) defaults = argv[++i];
    else if (a == "--override" && i + 1 < argc) override_files.push_back(argv[++i]);
    else if (a == "--out" && i + 1 < argc) out = argv[++i];
    else if (a == "--surface" && i + 1 < argc) surface = argv[++i];
    else if (a.find('=') != std::string::npos && a.rfind("--", 0) != 0) assignments.push_back(a);
    else {
      std::cerr << "usage: spf_sample --surface BASE [--out DIR] [--config FILE] [--override FILE]... [key=value ...]\nunrecognised argument: " << a << "\n";
      return 2;
    }
  }
  try {
    namespace sp = spf::particlisation;
    spf::Config cfg = spf::Config::from_file(defaults);
    for (const auto& f : override_files) cfg.merge_file(f);
    for (const auto& s : assignments) cfg.set_from_string(s);
    if (surface.empty()) surface = cfg.get<std::string>("sampler.surface");
    if (surface.empty()) throw spf::ConfigError("no surface: pass --surface BASE or set sampler.surface");
    cfg.echo(out);
    if (const int th = cfg.get<int>("sampler.threads"); th > 0) sp::set_threads(th);
    const auto t0 = std::chrono::steady_clock::now();
    sp::Sampler sampler(sp::SamplerOptions::from_config(cfg), sp::read_surface(surface), sp::make_species_set(cfg));
    const auto t1 = std::chrono::steady_clock::now();
    const int n = cfg.get<int>("sampler.n_samples");
    auto res = sampler.sample_all(n);
    // corona hadrons (WP10e): appended to every sample when sampler.corona = append and the surface sidecar reports a corona
    const std::string corona_mode = cfg.get<std::string>("sampler.corona");
    if (corona_mode != "append" && corona_mode != "omit") throw spf::ConfigError("sampler.corona must be 'append' or 'omit', got '" + corona_mode + "'");
    long n_corona_sidecar = 0;
    {
      const auto& m = sampler.surface().meta;
      if (m.contains("corona") && m["corona"].is_object() && m["corona"].contains("n_corona")) n_corona_sidecar = m["corona"]["n_corona"].get<long>();
    }
    std::vector<sp::Particle> corona;
    if (corona_mode == "append" && n_corona_sidecar > 0) {
      corona = sp::read_corona_oscar(sp::corona_path(surface));
      if (static_cast<long>(corona.size()) != n_corona_sidecar)
        throw std::runtime_error("corona file holds " + std::to_string(corona.size()) + " hadrons, the surface sidecar reports " + std::to_string(n_corona_sidecar));
      for (auto& r : res) sp::append_corona(r, corona);
    }
    const auto t2 = std::chrono::steady_clock::now();
    const std::string prefix = out + "/" + cfg.get<std::string>("sampler.output_prefix");
    {
      std::ofstream os(prefix + ".oscar");
      sp::write_oscar2013(os, res, "spf_sample (spinodal-fluid WP10b)");
    }
    nlohmann::json j;
    j["surface"] = surface;
    j["preparation"] = sampler.report().to_json();
    j["threads"] = sp::max_threads();
    j["seconds_preparation"] = std::chrono::duration<double>(t1 - t0).count();
    j["seconds_sampling"] = std::chrono::duration<double>(t2 - t1).count();
    j["corona"] = {{"mode", corona_mode}, {"n_corona_sidecar", n_corona_sidecar}, {"n_appended", corona.size()}};
    j["samples"] = nlohmann::json::array();
    int nfail = 0;
    for (auto& r : res) {
      j["samples"].push_back(r.summary);
      if (!r.ok) ++nfail;
    }
    std::ofstream(prefix + ".json") << j.dump(2) << "\n";
    std::printf("spf_sample: %d samples (%d failed), prep %.2f s, sampling %.2f s, B_target %lld, Q_target %lld -> %s.oscar\n", n, nfail, j["seconds_preparation"].get<double>(),
                j["seconds_sampling"].get<double>(), sampler.report().B_target, sampler.report().Q_target, prefix.c_str());
    return nfail ? 3 : 0;
  } catch (const std::exception& e) {
    std::cerr << "spf_sample: " << e.what() << "\n";
    return 1;
  }
}
