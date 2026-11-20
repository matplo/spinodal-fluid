#include "spf/config.hpp"

#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>

namespace spf {

namespace {

nlohmann::json read_json_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw ConfigError("cannot open config file '" + path + "'");
  nlohmann::json j = nlohmann::json::parse(in, nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded()) throw ConfigError("config file '" + path + "' is not valid JSON");
  if (!j.is_object()) throw ConfigError("config file '" + path + "' must hold a JSON object");
  return j;
}

nlohmann::json::json_pointer to_pointer(const std::string& dotted_key) {
  std::string p = "/";
  for (char ch : dotted_key) p += (ch == '.') ? '/' : ch;
  return nlohmann::json::json_pointer(p);
}

// number (integer or floating), string, boolean, other
int type_class(const nlohmann::json& v) {
  if (v.is_number()) return 0;
  if (v.is_string()) return 1;
  if (v.is_boolean()) return 2;
  return 3;
}

void flatten_into(const nlohmann::json& node, const std::string& prefix,
                  std::vector<std::pair<std::string, nlohmann::json>>& out) {
  for (auto it = node.begin(); it != node.end(); ++it) {
    const std::string key = prefix.empty() ? it.key() : prefix + "." + it.key();
    if (it->is_object()) {
      flatten_into(*it, key, out);
    } else {
      out.emplace_back(key, *it);
    }
  }
}

}  // namespace

Config Config::from_file(const std::string& defaults_path) { return from_json(read_json_file(defaults_path)); }

Config Config::from_json(const nlohmann::json& defaults) {
  if (!defaults.is_object()) throw ConfigError("defaults must be a JSON object");
  Config c;
  c.data_ = defaults;
  return c;
}

void Config::merge_file(const std::string& path) { merge_json(read_json_file(path)); }

void Config::merge_json(const nlohmann::json& overrides) {
  if (!overrides.is_object()) throw ConfigError("overrides must be a JSON object");
  std::vector<std::pair<std::string, nlohmann::json>> flat;
  flatten_into(overrides, "", flat);
  // validate everything before changing anything
  Config trial = *this;
  for (const auto& [key, value] : flat) trial.assign(key, value);
  data_ = std::move(trial.data_);
}

void Config::set_from_string(const std::string& key_eq_value) {
  const auto eq = key_eq_value.find('=');
  if (eq == std::string::npos || eq == 0) {
    throw ConfigError("expected 'section.key=value', got '" + key_eq_value + "'");
  }
  const std::string key = key_eq_value.substr(0, eq);
  const std::string text = key_eq_value.substr(eq + 1);
  nlohmann::json value = nlohmann::json::parse(text, nullptr, /*allow_exceptions=*/false);
  if (value.is_discarded()) value = text;
  assign(key, value);
}

bool Config::has(const std::string& dotted_key) const {
  if (dotted_key.empty()) return false;
  return data_.contains(to_pointer(dotted_key));
}

const nlohmann::json& Config::at(const std::string& dotted_key) const {
  if (!has(dotted_key)) throw ConfigError("unknown config key '" + dotted_key + "'");
  return data_.at(to_pointer(dotted_key));
}

void Config::assign(const std::string& dotted_key, const nlohmann::json& value) {
  if (!has(dotted_key)) throw ConfigError("unknown config key '" + dotted_key + "'");
  nlohmann::json& slot = data_.at(to_pointer(dotted_key));
  if (slot.is_object()) throw ConfigError("config key '" + dotted_key + "' is a section, not a value");
  if (type_class(slot) != type_class(value)) {
    throw ConfigError("config key '" + dotted_key + "': type mismatch (default is " + std::string(slot.type_name()) +
                      ", override is " + std::string(value.type_name()) + ")");
  }
  slot = value;
}

void Config::echo(const std::string& dir) const {
  std::filesystem::create_directories(dir);
  nlohmann::json out;
  out["config"] = data_;
  out["runtime"] = runtime_info();
  const std::string path = (std::filesystem::path(dir) / "config_used.json").string();
  std::ofstream f(path);
  if (!f) throw ConfigError("cannot write '" + path + "'");
  f << out.dump(2) << "\n";
}

}  // namespace spf
