#include "spf/config.hpp"
#include "spf/types.hpp"
#include "spf/version.hpp"

namespace spf {

nlohmann::json runtime_info() {
  nlohmann::json j;
  j["version"] = SPF_VERSION;
  j["git"] = SPF_GIT_DESCRIBE;
  j["kokkos_version"] = std::to_string(KOKKOS_VERSION_MAJOR) + "." + std::to_string(KOKKOS_VERSION_MINOR) + "." +
                        std::to_string(KOKKOS_VERSION_PATCH);
  j["execution_space"] = ExecSpace::name();
  j["concurrency"] = ExecSpace().concurrency();
  j["real_bytes"] = sizeof(Real);
  return j;
}

}  // namespace spf
