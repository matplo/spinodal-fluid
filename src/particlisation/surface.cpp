#include "spf/particlisation/surface.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>

namespace spf::particlisation {

static_assert(std::endian::native == std::endian::little, "the surface format is little-endian; a byte swap is not implemented");

double Surface::dx() const {
  for (const char* k : {"dx", "dx_fm"})
    if (meta.contains(k) && meta[k].is_number()) return meta[k].get<double>();
  throw SurfaceError("surface sidecar has no 'dx'");
}

long long Surface::event_id() const {
  for (const char* k : {"event_id", "event"})
    if (meta.contains(k) && meta[k].is_number_integer()) return meta[k].get<long long>();
  return 0;
}

Surface read_surface(const std::string& base) {
  Surface s;
  std::ifstream js(base + ".surface.json");
  if (!js) throw SurfaceError("cannot open " + base + ".surface.json");
  try {
    js >> s.meta;
  } catch (const std::exception& e) {
    throw SurfaceError("cannot parse " + base + ".surface.json: " + e.what());
  }
  if (!s.meta.contains("n_elements")) throw SurfaceError("sidecar lacks 'n_elements'");
  const long long n = s.meta["n_elements"].get<long long>();
  if (s.meta.contains("format_version") && s.meta["format_version"].get<int>() != 1)
    throw SurfaceError("unsupported surface format_version (this reader implements 1)");
  std::ifstream bs(base + ".surface.bin", std::ios::binary | std::ios::ate);
  if (!bs) throw SurfaceError("cannot open " + base + ".surface.bin");
  const long long bytes = static_cast<long long>(bs.tellg());
  const long long need = n * kSurfaceRecordDoubles * 8;
  if (bytes != need) {
    std::ostringstream o;
    o << base << ".surface.bin has " << bytes << " bytes, " << need << " expected for n_elements = " << n;
    throw SurfaceError(o.str());
  }
  bs.seekg(0);
  s.rec.resize(static_cast<size_t>(n) * kSurfaceRecordDoubles);
  bs.read(reinterpret_cast<char*>(s.rec.data()), static_cast<std::streamsize>(need));
  if (!bs) throw SurfaceError("short read of " + base + ".surface.bin");
  for (size_t i = 0; i < s.rec.size(); ++i)
    if (!std::isfinite(s.rec[i])) {
      std::ostringstream o;
      o << "non-finite value in " << base << ".surface.bin at element " << i / kSurfaceRecordDoubles << ", field " << i % kSurfaceRecordDoubles;
      throw SurfaceError(o.str());
    }
  return s;
}

void write_surface(const std::string& base, const Surface& s) {
  nlohmann::json m = s.meta;
  m["n_elements"] = s.size();
  if (!m.contains("format_version")) m["format_version"] = 1;
  std::ofstream bs(base + ".surface.bin", std::ios::binary);
  if (!bs) throw SurfaceError("cannot write " + base + ".surface.bin");
  bs.write(reinterpret_cast<const char*>(s.rec.data()), static_cast<std::streamsize>(s.rec.size() * sizeof(double)));
  bs.close();
  std::ofstream js(base + ".surface.json");
  if (!js) throw SurfaceError("cannot write " + base + ".surface.json");
  js << m.dump(2) << "\n";
}

nlohmann::json make_surface_meta(const std::string& method, double dx, double dt, long long event_id) {
  nlohmann::json m;
  m["format_version"] = 1;
  m["method"] = method;
  m["dx"] = dx;
  m["dt"] = dt;
  m["event_id"] = event_id;
  return m;
}

}  // namespace spf::particlisation
