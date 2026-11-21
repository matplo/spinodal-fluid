# Observables: output files (WP6)

`ObservableRecorder` (in a run) and `spf_observe` (on snapshots) write the same files into one output directory. Definitions: `docs/equations.md`, Sect. 10.

| File | Content |
|---|---|
| `config_used.json` | configuration echo (`Config::echo`) |
| `observables.csv` | one line per recorded time: `t_fm, A, mom2 ... mom7, U_perp, unstable_fraction, eps_max_eps_s, step`; `momN` = normalised moment `<D^N>/<D>^N`; `A` = integral of `D` (baryon number in the computational frame); numbers with 17 digits |
| `clumps_step<S>.json` | clump histogram (`histogram`, bins `round(B) = 0..obs.clump_hist_max`, `overflow`), `baryon_per_clump`, `cells_per_clump`, threshold, connectivity, wrapping |
| `phase_step<S>.json` | phase-plane histogram: `baryon[ir * n_eps + ie]` (baryon number), `outside`, `total`, bin widths in rho_s and eps_s, `unstable_fraction` |
| `observables_summary.json` | number of records, `fluid_end_time_fm` (`null` if never reached), maximum enhancement of N = 2..7 with its time |

The histograms are written at the first record at or after each time in `obs.hist_times_fm` (default: none). The recorder records when `step % obs.every_steps == 0`:

```cpp
spf::ObservableRecorder<spf::EosColdThermal> rec(cfg, eos, "out/event_000");
for (long s = 0; s <= nsteps; ++s) { if (s > 0) h.step(); rec.on_step(h); }
rec.finalize();
```

Post-processing: `spf_observe [--out DIR] [key=value ...] snapshot ...`; the snapshots need the fields `R rho eps vx vy` (`Hydro3D::write_snapshot`); the configuration is the echo in the first snapshot.
Ensemble: `python3 tools/obs_ensemble.py --out DIR event_dir ...` writes `ensemble_enhancement.csv` (mean and standard error of the enhancement of N = 2..7 versus `t - t_init`) and `ensemble_summary.json`
(mean, standard error, sample std, spread max/mean and max/min of the per-event maximum enhancement, per-event maxima).
