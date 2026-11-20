add_executable(spf_tests
  test_main.cpp
  test_config.cpp
  test_kokkos.cpp
)
target_link_libraries(spf_tests PRIVATE spf doctest::doctest)
add_test(NAME spf_tests COMMAND spf_tests)
add_test(NAME spf_info COMMAND spf_info --echo ${CMAKE_CURRENT_BINARY_DIR}/spf_info_out shasta.lambda=0.2)
