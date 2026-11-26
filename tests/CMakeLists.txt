add_executable(spf_tests
  test_main.cpp
  test_config.cpp
  test_kokkos.cpp
  test_eos.cpp
  test_eos_table.cpp
  test_eos_finite_T.cpp
  test_shasta.cpp
  test_hydro1d.cpp
  test_hydro3d.cpp
  test_observables.cpp
  test_gradient.cpp
  test_linear.cpp
  test_initial_state.cpp
  test_collide.cpp
  test_surface.cpp
  test_sampler.cpp
)
target_link_libraries(spf_tests PRIVATE spf doctest::doctest)
target_compile_definitions(spf_tests PRIVATE SPF_SOURCE_DIR="${PROJECT_SOURCE_DIR}" SPF_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_include_directories(spf_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_dependencies(spf_tests spf_observe spf_collide)
target_compile_definitions(spf_tests PRIVATE SPF_COLLIDE_EXE="$<TARGET_FILE:spf_collide>")
target_compile_definitions(spf_tests PRIVATE SPF_OBSERVE_EXE="$<TARGET_FILE:spf_observe>" SPF_ENSEMBLE_SCRIPT="${PROJECT_SOURCE_DIR}/tools/obs_ensemble.py")
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  target_compile_definitions(spf_tests PRIVATE SPF_PYTHON="${Python3_EXECUTABLE}")
endif()
add_test(NAME spf_tests COMMAND spf_tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_test(NAME spf_info COMMAND spf_info --echo ${CMAKE_CURRENT_BINARY_DIR}/spf_info_out shasta.lambda=0.2)
add_dependencies(spf_tests spf_sample)
target_compile_definitions(spf_tests PRIVATE SPF_SAMPLE_EXE="$<TARGET_FILE:spf_sample>")
