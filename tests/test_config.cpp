#include <doctest/doctest.h>

#include "spf/config.hpp"

#include <filesystem>
#include <fstream>

using spf::Config;
using spf::ConfigError;

namespace {
nlohmann::json small_defaults() {
  return nlohmann::json::parse(R"({
    "shasta": {"antidiffusion": "phoenical", "eta_ad": 0.125, "lambda": 0.4},
    "grid": {"dx_fm": 0.2, "nx": 64},
    "eos": {"extrapolate": false}
  })");
}
}  // namespace

TEST_CASE("shipped defaults load and hold the documented default values") {
  const Config cfg = Config::from_file(SPF_DEFAULT_CONFIG);
  CHECK(cfg.get<std::string>("shasta.antidiffusion") == "phoenical");
  CHECK(cfg.get<double>("shasta.eta_ad") == doctest::Approx(0.125));
  CHECK(cfg.get<double>("shasta.lambda") == doctest::Approx(0.4));
  CHECK(cfg.get<double>("grid.dx_fm") == doctest::Approx(0.2));
  CHECK(cfg.get<double>("gradient.a_fm") == doctest::Approx(0.033));
  CHECK(cfg.get<std::string>("gradient.form") == "paper_pressure");
  CHECK(cfg.get<double>("ic.sigma_init_fm") == doctest::Approx(1.0));
  CHECK(cfg.get<double>("constants.rho_s_fm3") == doctest::Approx(0.153));
  CHECK(cfg.get<double>("constants.hbarc_GeV_fm") == doctest::Approx(0.1973269804));
  CHECK(cfg.get<std::string>("run.precision") == "double");
}

TEST_CASE("dotted access, has() and typed get()") {
  const Config cfg = Config::from_json(small_defaults());
  CHECK(cfg.has("grid.dx_fm"));
  CHECK_FALSE(cfg.has("grid.dy_fm"));
  CHECK_FALSE(cfg.has(""));
  CHECK(cfg.get<int>("grid.nx") == 64);
  CHECK(cfg.get<bool>("eos.extrapolate") == false);
  CHECK_THROWS_AS(cfg.get<double>("grid.dy_fm"), ConfigError);
  CHECK_THROWS_AS(cfg.get<double>("shasta.antidiffusion"), ConfigError);
}

TEST_CASE("override object: known keys are applied, others untouched") {
  Config cfg = Config::from_json(small_defaults());
  cfg.merge_json(nlohmann::json::parse(R"({"shasta": {"lambda": 0.2}, "grid": {"nx": 128}})"));
  CHECK(cfg.get<double>("shasta.lambda") == doctest::Approx(0.2));
  CHECK(cfg.get<int>("grid.nx") == 128);
  CHECK(cfg.get<double>("shasta.eta_ad") == doctest::Approx(0.125));
}

TEST_CASE("override object: an unknown key or a type mismatch is rejected and nothing is changed") {
  Config cfg = Config::from_json(small_defaults());
  CHECK_THROWS_AS(cfg.merge_json(nlohmann::json::parse(R"({"shasta": {"lambda": 0.2, "lamda": 0.3}})")), ConfigError);
  CHECK(cfg.get<double>("shasta.lambda") == doctest::Approx(0.4));  // first entry must not have been applied
  CHECK_THROWS_AS(cfg.merge_json(nlohmann::json::parse(R"({"shasta": {"lambda": "fast"}})")), ConfigError);
  CHECK_THROWS_AS(cfg.merge_json(nlohmann::json::parse(R"({"shasta": 1})")), ConfigError);
  CHECK(cfg.get<double>("shasta.lambda") == doctest::Approx(0.4));
}

TEST_CASE("key=value strings: numbers, booleans, bare and quoted strings") {
  Config cfg = Config::from_json(small_defaults());
  cfg.set_from_string("shasta.lambda=0.25");
  cfg.set_from_string("eos.extrapolate=true");
  cfg.set_from_string("shasta.antidiffusion=explicit");
  CHECK(cfg.get<double>("shasta.lambda") == doctest::Approx(0.25));
  CHECK(cfg.get<bool>("eos.extrapolate") == true);
  CHECK(cfg.get<std::string>("shasta.antidiffusion") == "explicit");
  cfg.set_from_string("shasta.antidiffusion=\"phoenical\"");
  CHECK(cfg.get<std::string>("shasta.antidiffusion") == "phoenical");
  cfg.set_from_string("grid.nx=32");
  CHECK(cfg.get<int>("grid.nx") == 32);
  CHECK_THROWS_AS(cfg.set_from_string("shasta.lambda"), ConfigError);
  CHECK_THROWS_AS(cfg.set_from_string("=3"), ConfigError);
  CHECK_THROWS_AS(cfg.set_from_string("shasta.nope=3"), ConfigError);
  CHECK_THROWS_AS(cfg.set_from_string("shasta.lambda=fast"), ConfigError);
  CHECK_THROWS_AS(cfg.set_from_string("shasta=3"), ConfigError);
}

TEST_CASE("override file and echo round trip") {
  const auto dir = std::filesystem::temp_directory_path() / "spf_test_config_echo";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const auto override_path = dir / "override.json";
  { std::ofstream(override_path) << R"({"grid": {"dx_fm": 0.1}})"; }

  Config cfg = Config::from_json(small_defaults());
  cfg.merge_file(override_path.string());
  CHECK(cfg.get<double>("grid.dx_fm") == doctest::Approx(0.1));
  CHECK_THROWS_AS(cfg.merge_file((dir / "missing.json").string()), ConfigError);

  const auto out = dir / "run" / "nested";
  cfg.echo(out.string());
  std::ifstream in(out / "config_used.json");
  REQUIRE(in.good());
  const nlohmann::json j = nlohmann::json::parse(in);
  CHECK(j.at("config") == cfg.json());
  CHECK(j.at("runtime").at("execution_space").get<std::string>().size() > 0);
  CHECK(j.at("runtime").at("real_bytes").get<int>() == 8);
  std::filesystem::remove_all(dir);
}
