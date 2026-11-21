# Rules for device-compatible kernels

One source runs on the Kokkos Serial, OpenMP and CUDA backends. These rules apply to every kernel,
also while only a CPU backend is being built. A kernel that cannot run on the device is not finished.

1. Fields live in `Kokkos::View` (`spf::Field3`). No STL containers, raw `new`, I/O, exceptions, virtual calls,
   `std::function` or allocation inside a kernel.
2. Kernels in headers are named functor structs with a `KOKKOS_INLINE_FUNCTION operator()`, launched with `Kokkos::parallel_for` over an `MDRangePolicy`.
   `KOKKOS_LAMBDA` is allowed only inside a single `.cpp` file (tests, apps). Reason: with nvcc, a lambda kernel inside a non-template inline header
   function that is compiled in more than one translation unit crashed the optimised CUDA build (see `docs/validation.md`, WP1).
3. Each stage of an algorithm (for SHASTA: transport + diffusion, antidiffusive flux, limiter, update) is its own
   `parallel_for` that reads one set of Views and writes another, so that the result does not depend on execution order.
4. An equation of state is a trivially copyable struct with `KOKKOS_INLINE_FUNCTION` members; table data sit in a
   `Kokkos::View` member. Solvers are templated on the EoS type; the run-time choice is a switch in the driver, outside the kernels.
5. Iterative per-cell solves use a fixed maximum iteration count and report failure through an integer View that is
   summed with `parallel_reduce`.
6. Sums and extrema use `parallel_reduce`. Summation order differs between backends: compare backends with a relative
   tolerance, never bitwise.
7. Host-device copies (`create_mirror_view`, `deep_copy`) happen only at snapshots and at the end of a test.
8. Double precision (`spf::Real`) throughout.
9. Equation-of-state member functions are device functions; host code never calls them (tables are reached through device pointers).
   Host-side conversions go through a kernel: the host fills mirrors of the primitive fields, copies them to the device, and a named
   functor computes the derived fields (`Hydro1D::set_state`, `Hydro3D::set_state_fn`). On the CPU backends host and device memory coincide, so a
   violation of this rule passes every CPU test and crashes only on CUDA. Scalar members that do not read table data (for instance
   `EosTable::check_range`, `ColdCurveTable::valid_range`) are host-callable.

`tests/test_kokkos.cpp` shows the pattern.
