cmake_minimum_required(VERSION 3.22)
project(spinodal_fluid VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SPF_BUILD_TESTS "Build the unit and regression tests" ON)
option(SPF_USE_SYSTEM_KOKKOS "Use an installed Kokkos (find_package) instead of fetching the pinned release" OFF)

include(cmake/Dependencies.cmake)
include(cmake/Version.cmake)

add_library(spf
  src/config.cpp
  src/runtime_info.cpp
  src/hydro3d_io.cpp
  src/ic_random.cpp
)
target_include_directories(spf PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_link_libraries(spf PUBLIC Kokkos::kokkos nlohmann_json::nlohmann_json)
target_compile_definitions(spf PUBLIC SPF_DEFAULT_CONFIG="${CMAKE_CURRENT_SOURCE_DIR}/config/defaults.json")

add_executable(spf_info apps/spf_info.cpp)
target_link_libraries(spf_info PRIVATE spf)

add_executable(spf_bench apps/spf_bench.cpp)
target_link_libraries(spf_bench PRIVATE spf)
target_compile_definitions(spf_bench PRIVATE SPF_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(spf_compare apps/spf_compare.cpp)
target_link_libraries(spf_compare PRIVATE spf)

if(SPF_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
