add_executable(spf_tests
  test_main.cpp
  test_config.cpp
  test_kokkos.cpp
  test_eos.cpp
  test_eos_table.cpp
  test_shasta.cpp
  test_hydro1d.cpp
  test_hydro3d.cpp
)
target_link_libraries(spf_tests PRIVATE spf doctest::doctest)
target_compile_definitions(spf_tests PRIVATE SPF_SOURCE_DIR="${PROJECT_SOURCE_DIR}" SPF_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_include_directories(spf_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME spf_tests COMMAND spf_tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME spf_info COMMAND spf_info --echo ${CMAKE_CURRENT_BINARY_DIR}/spf_info_out shasta.lambda=0.2)
