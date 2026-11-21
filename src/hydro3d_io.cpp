// Snapshot output of the 3D solver: raw little-endian doubles plus a JSON sidecar.
#include "spf/hydro3d.hpp"

#include <bit>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace spf {

void write_snapshot_files(const std::string& base, const int shape[3], Real dx, Real time, long step,
                          const std::vector<std::string>& names, const std::vector<std::vector<Real>>& data,
                          const nlohmann::json& config_echo) {
  static_assert(std::endian::native == std::endian::little, "snapshot format is little-endian; add a byte swap for big-endian hosts");
  if (names.size() != data.size()) throw std::invalid_argument("write_snapshot_files: names and data differ in size");
  const std::filesystem::path p(base);
  if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
  const size_t n = static_cast<size_t>(shape[0]) * shape[1] * shape[2];
  std::ofstream bin(base + ".bin", std::ios::binary);
  if (!bin) throw std::runtime_error("cannot open " + base + ".bin");
  nlohmann::json fields = nlohmann::json::array();
  size_t offset = 0;
  for (size_t f = 0; f < names.size(); ++f) {
    if (data[f].size() != n) throw std::invalid_argument("write_snapshot_files: field '" + names[f] + "' has the wrong size");
    bin.write(reinterpret_cast<const char*>(data[f].data()), static_cast<std::streamsize>(n * sizeof(Real)));
    fields.push_back({{"name", names[f]}, {"byte_offset", offset}});
    offset += n * sizeof(Real);
  }
  bin.close();
  nlohmann::json j;
  j["format"] = "raw little-endian float64, fields concatenated, each in C order [ix][iy][iz], interior cells only";
  j["file"] = std::filesystem::path(base + ".bin").filename().string();
  j["shape"] = {shape[0], shape[1], shape[2]};
  j["dx_fm"] = dx;
  j["time_fm"] = time;
  j["step"] = step;
  j["fields"] = fields;
  j["config"] = config_echo;
  std::ofstream js(base + ".json");
  if (!js) throw std::runtime_error("cannot open " + base + ".json");
  js << j.dump(2) << "\n";
}

}  // namespace spf
