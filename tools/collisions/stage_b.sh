#!/usr/bin/env bash
# Stage B driver: SMASH events and spf_collide runs for several energies, finite-T table two-phase and Maxwell, the same events for both equations
# of state, resumable. This is the sequence of commands of the Stage B pilot (docs/running_collisions.md, Sect. 13).
#
#   tools/collisions/stage_b.sh --bin build/cuda-turing75 --smash <smash build>/smash --out stage_b --energies "1 2 3 4 5 6" --nevents 50
#
# Extension to more events (the same output directories; finished events are skipped): call it again with a larger --nevents. SMASH events are
# regenerated with the same seed (the first N events of a longer file are identical to the shorter file: checked for 1 and 6 A GeV, 50 of 200 events);
# a file that already exists is kept only when it holds at least --nevents events, otherwise it is regenerated.
# Options: --seed (SMASH, default 1), --chunk (events per process, default 10), --tinit-config (default config/collisions/t_init_fixed.json),
#          --stage-config (default config/collisions/stage_a.json: run the full 6 fm/c, fluid end recorded), --diag-event (event with run.cell_diagnostics, default 0),
#          --wrapper (command prefix for SMASH, e.g. 'micromamba run -n spf-smash'), extra key=value overrides after "--".
set -euo pipefail
BIN=build/cuda-turing75; SMASH=; OUT=stage_b; ENERGIES="1 2 3 4 5 6"; NEV=50; SEED=1; CHUNK=10; DIAG=0
SRC=$(cd "$(dirname "$0")/../.." && pwd)
TINIT=$SRC/config/collisions/t_init_fixed.json; STAGE=$SRC/config/collisions/stage_a.json; WRAP=
EXTRA=()
while [ $# -gt 0 ]; do
  case $1 in
    --bin) BIN=$2; shift 2;; --smash) SMASH=$2; shift 2;; --out) OUT=$2; shift 2;; --energies) ENERGIES=$2; shift 2;;
    --nevents) NEV=$2; shift 2;; --seed) SEED=$2; shift 2;; --chunk) CHUNK=$2; shift 2;; --diag-event) DIAG=$2; shift 2;;
    --tinit-config) TINIT=$2; shift 2;; --stage-config) STAGE=$2; shift 2;; --wrapper) WRAP=$2; shift 2;;
    --) shift; EXTRA=("$@"); break;;
    *) echo "unknown option $1" >&2; exit 2;;
  esac
done
[ -n "$SMASH" ] || { echo "--smash is required" >&2; exit 2; }
mkdir -p "$OUT"
for E in $ENERGIES; do
  OSC=$OUT/smash/e$E/smash_data/particle_lists.oscar
  have=0
  if [ -f "$OSC" ]; then have=$(grep -c "^# event .* ensemble .* in " "$OSC" || true); fi
  if [ "$have" -lt "$NEV" ]; then
    rm -rf "$OUT/smash/e$E"
    python3 "$SRC/tools/collisions/gen_events.py" --ekin "$E" --nevents "$NEV" --seed "$SEED" --smash "$SMASH" ${WRAP:+--smash-wrapper "$WRAP"} \
        --out "$OUT/smash/e$E" --extra --radius-mode fixed
  fi
  OSCABS=$(cd "$(dirname "$OSC")" && pwd)/$(basename "$OSC")
  CF=(--config "$SRC/config/collisions/common.json" --config "$STAGE" --config "$SRC/config/collisions/e$E.json" --config "$TINIT")
  # two-phase: the diagnostic event first (cell diagnostics on), then all events
  python3 "$SRC/tools/collisions/run_batch.py" --exe "$BIN/spf_collide" --oscar "$OSCABS" --out "$OUT/runs/e${E}_table_two_phase" --first "$DIAG" --n 1 \
      "${CF[@]}" eos.model=table_two_phase run.cell_diagnostics=true ${EXTRA[@]+"${EXTRA[@]}"} || true
  python3 "$SRC/tools/collisions/run_batch.py" --exe "$BIN/spf_collide" --oscar "$OSCABS" --out "$OUT/runs/e${E}_table_two_phase" --first 0 --n "$NEV" --chunk "$CHUNK" \
      "${CF[@]}" eos.model=table_two_phase ${EXTRA[@]+"${EXTRA[@]}"} || true
  python3 "$SRC/tools/collisions/run_batch.py" --exe "$BIN/spf_collide" --oscar "$OSCABS" --out "$OUT/runs/e${E}_table_maxwell" --first 0 --n "$NEV" --chunk "$CHUNK" \
      "${CF[@]}" eos.model=table_maxwell ${EXTRA[@]+"${EXTRA[@]}"} || true
  python3 "$SRC/tools/collisions/aggregate.py" --out "$OUT/agg_e$E" --run two_phase="$OUT/runs/e${E}_table_two_phase" --run maxwell="$OUT/runs/e${E}_table_maxwell"
done
