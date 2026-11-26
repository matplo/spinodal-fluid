#!/usr/bin/env bash
# G5-INT end-to-end test: fluid (spf_collide with the switching surface) -> surface file -> particles (spf_sample), events 0-4 of the
# 3 A GeV Stage B pilot file, two-phase and Maxwell tables, mu_rule = rho_matched and table.
#
#   g5_endtoend.sh BUILD_DIR OSCAR_FILE [FIRST_EVENT N_EVENTS N_SAMPLES THREADS]
#
# Run from the repository root with the tables in data/eos/generated. Writes ./g5_out/ (the large surface .bin files are deleted
# after both mu rules have been sampled; production runs sample in place, see docs/particlisation_format.md).
set -eo pipefail
BUILD=${1:?build dir}
OSC=$(realpath "${2:?oscar file}")
FIRST=${3:-0}
NEV=${4:-5}
NS=${5:-10}
THR=${6:-16}
OUT=${G5_OUT:-g5_out}
export OMP_PROC_BIND=false
CF="--config config/collisions/common.json --config config/collisions/e3.json --config config/collisions/t_init_fixed.json"
now() { date +%s.%N; }
mkdir -p "$OUT/timing"
echo "eos,rule,event,wall_s,surface_bytes" > "$OUT/timing/sampling_wall.csv"
for EOS in two_phase maxwell; do
  t0=$(now)
  "$BUILD/spf_collide" --out "$OUT/runs/$EOS" $CF ic.oscar_path="$OSC" eos.model=table_$EOS run.event_first=$FIRST run.n_events=$NEV \
      > "$OUT/timing/collide_$EOS.log" 2>&1
  t1=$(now)
  echo "collide $EOS: $(awk "BEGIN{print $t1-$t0}") s for $NEV events"
done
for EOS in two_phase maxwell; do
  for RULE in rho_matched table; do
    for ((e = FIRST; e < FIRST + NEV; e++)); do
      EV=$(printf "event_%03d" $e)
      BASE="$OUT/runs/$EOS/$EV/$EV"
      SZ=$(wc -c < "$BASE.surface.bin" | tr -d " ")
      t0=$(now)
      "$BUILD/spf_sample" --surface "$BASE" --out "$OUT/samples/$EOS/$RULE/$EV" sampler.mu_rule=$RULE sampler.n_samples=$NS sampler.threads=$THR \
          sampler.species=eos_consistent sampler.conservation=global_B_Q sampler.global_method=exact_counts \
          > "$OUT/timing/sample_${EOS}_${RULE}_$EV.log" 2>&1
      t1=$(now)
      echo "$EOS,$RULE,$e,$(awk "BEGIN{print $t1-$t0}"),$SZ" >> "$OUT/timing/sampling_wall.csv"
    done
  done
done
rm -f "$OUT"/runs/*/event_*/*.surface.bin
echo "surface .bin files deleted"
ls -lh "$OUT/timing" | head -30
