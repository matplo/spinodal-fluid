# spinodal-fluid

Relativistic ideal fluid dynamics with a first-order phase transition and a finite-range (gradient) term,
for studying the spinodal amplification of baryon-density fluctuations in nuclear collisions at
beam energies of a few GeV per nucleon.

The project re-implements, from the published equations,

* J. Steinheimer and J. Randrup, *Spinodal amplification of density fluctuations in fluid-dynamical simulations of
  relativistic nuclear collisions*, Phys. Rev. Lett. 109, 212301 (2012), arXiv:1209.2462
  (detailed account: Phys. Rev. C 87, 054903 (2013), arXiv:1302.2956), and
* J. Randrup, *Spinodal phase separation in relativistic nuclear collisions*, Phys. Rev. C 82, 034902 (2010),
  arXiv:1007.1448 (dissipative dispersion relation and amplification along phase trajectories).

The transport algorithm is SHASTA as described by Rischke, Bernard and Maruhn, Nucl. Phys. A 595, 346 (1995),
arXiv:nucl-th/9504018. No source code from other heavy-ion codes is included.

## Status

Under construction. See `docs/implementation_plan.md` for the work packages and `docs/validation.md` for what has
been measured so far.

| Package | Content | State |
|---|---|---|
| WP0 | repository, build presets, configuration, CI | done |
| WP1 | SHASTA in one dimension | implemented, see `docs/validation.md` |
| WP2 | equations of state | analytic EoS done; table reader and scaled-EoS wrapper (WP2c) implemented, see `docs/validation.md`; physical tables pending (WP2b) |
| WP3 | 3+1D solver, GPU validation | implemented, see `docs/validation.md` (WP3 and WP3g) |
| WP4 | finite-range term, spinodal growth rates | implemented, see `docs/validation.md` (WP4); several comparisons with the continuum formulae at 8 to 10 cells per wavelength are reported as expected failures |
| WP5, WP7 | initial conditions, collision runs | not started |
| WP6 | observables (moments, clumps, U_perp, phase plane), `spf_observe`, ensemble script | implemented, see `docs/validation.md` and `docs/observables.md` |
| WP8 | linear-response library (dispersion relation, transport coefficients, G_k, Lalime), `spf_linear`, digitised Figs. 2-7 of arXiv:1007.1448 | implemented, see `docs/validation.md` (WP8); nuclear-EoS provider pending (WP2b) |

## Build

Requirements: a C++20 compiler (GCC >= 10.4 or Clang >= 14), CMake >= 3.22, git. For the GPU build additionally the
CUDA toolkit >= 12.2. Kokkos 5.3.0, nlohmann_json and doctest are fetched at configure time at pinned commits.

```
cmake --preset cpu-openmp          # or cpu-serial, cuda, cuda-turing75
cmake --build --preset cpu-openmp --parallel
ctest --preset cpu-openmp
./build/cpu-openmp/spf_info        # prints version, backend, and runs a reduction
```

## Configuration

Every modelling choice is a named key in `config/defaults.json`; `docs/decisions_spinodal.md` lists the alternatives.
Programs accept `--override file.json` and `section.key=value` arguments and write the effective configuration to
`config_used.json` in their output directory.

## Layout

```
include/spf, src   library (Kokkos kernels; rules in docs/device_rules.md)
apps               executables
tests              doctest unit and regression tests
tools/digitise_1007_1448.py, tools/data   digitiser for the figures of arXiv:1007.1448 (vector paths of the PDF) and its CSV output
tools/eos          Python reference for the two-phase equation of state, its reference numbers, and the EoS table writer
config             default configuration
docs               plan, decisions, validation log
```

## Licence

BSD 3-Clause, see `LICENSE`. Third-party components keep their own licences:
Kokkos (Apache-2.0 WITH LLVM-exception), nlohmann_json (MIT), doctest (MIT).
