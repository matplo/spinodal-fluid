# Generates spf/version.hpp with the project version and `git describe` of the source tree.
set(SPF_GIT_DESCRIBE "unknown")
find_package(Git QUIET)
if(Git_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  execute_process(COMMAND ${GIT_EXECUTABLE} describe --always --dirty
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  OUTPUT_VARIABLE SPF_GIT_DESCRIBE OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(NOT SPF_GIT_DESCRIBE)
    set(SPF_GIT_DESCRIBE "unknown")
  endif()
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hpp.in
               ${CMAKE_CURRENT_BINARY_DIR}/generated/spf/version.hpp @ONLY)
