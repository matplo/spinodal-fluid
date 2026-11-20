# Third-party dependencies, pinned to release tags and verified against the release commit.
#   Kokkos        Apache-2.0 WITH LLVM-exception
#   nlohmann_json MIT
#   doctest       MIT
include(FetchContent)

# fetch <name> at <tag>, then fail unless the checked-out commit is <sha>
function(spf_fetch name repo tag sha)
  FetchContent_Declare(${name} GIT_REPOSITORY ${repo} GIT_TAG ${tag} GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(${name})
  FetchContent_GetProperties(${name} SOURCE_DIR _src)
  find_package(Git QUIET)
  if(Git_FOUND AND EXISTS "${_src}/.git")
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD WORKING_DIRECTORY ${_src}
                    OUTPUT_VARIABLE _head OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _head STREQUAL "${sha}")
      message(FATAL_ERROR "${name}: tag ${tag} resolved to ${_head}, expected ${sha}")
    endif()
  endif()
endfunction()

if(SPF_USE_SYSTEM_KOKKOS)
  find_package(Kokkos 5.0 REQUIRED)
else()
  spf_fetch(kokkos https://github.com/kokkos/kokkos.git 5.3.0 dc826b0fad122b57fee60223e86154aecb8e60f4)
endif()

set(JSON_BuildTests OFF CACHE INTERNAL "")
spf_fetch(nlohmann_json https://github.com/nlohmann/json.git v3.12.0 55f93686c01528224f448c19128836e7df245f72)

if(SPF_BUILD_TESTS)
  spf_fetch(doctest https://github.com/doctest/doctest.git v2.5.3 2d0a9359a60c51affe2a9bebb1be1dca47868151)
endif()
