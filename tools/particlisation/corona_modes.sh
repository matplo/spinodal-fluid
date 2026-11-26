#!/usr/bin/env bash
# WP10e end-to-end comparison of the corona modes (decision S31): fluid (spf_collide) -> surface -> sampler (spf_sample), events 0-4 of the
# 3 A GeV pilot file, two-phase and Maxwell tables, K samples per event, mu_rule rho_matched.
#
#   corona_modes.sh BUILD_DIR OSCAR_FILE PYTHON [FIRST_EVENT N_EVENTS N_SAMPLES THREADS]
#
# Run from the repository root with the tables in data/eos/generated.  Configurations (label: corona mode, switching density):
#   core_corona: core_corona, 4 eps_s      thermal: thermal, 4 eps_s      keep: keep, 4 eps_s
#   thermal_sw3: thermal, 3 eps_s          thermal_sw2: thermal, 2 eps_s
# Output ./corona_out/{runs,samples,census,timing}; the large .surface.bin files are deleted after the census.
set -eo pipefail
BUILD=${1:?build dir}
OSC=$(realpath "${2:?oscar file}")
PY=${3:?python with numpy}
FIRST=${4:-0}
NEV=${5:-5}
NS=${6:-10}
THR=${7:-16}
OUT=${CORONA_OUT:-corona_out}
export OMP_PROC_BIND=false
CF="--config config/collisions/common.json --config config/collisions/e3.json --config config/collisions/t_init_fixed.json"
mkdir -p "$OUT/timing" "$OUT/census"
declare -A MODE EXTRA
MODE[core_corona]=core_corona; EXTRA[core_corona]=""
MODE[thermal]=thermal;         EXTRA[thermal]=""
MODE[keep]=keep;               EXTRA[keep]=""
MODE[thermal_sw3]=thermal;     EXTRA[thermal_sw3]="--config config/collisions/eps_sw_3.json"
MODE[thermal_sw2]=thermal;     EXTRA[thermal_sw2]="--config config/collisions/eps_sw_2.json"
for CFG in core_corona thermal keep thermal_sw3 thermal_sw2; do
  for EOS in two_phase maxwell; do
    R="$OUT/runs/$CFG/$EOS"
    s0=$(date +%s)
    "$BUILD/spf_collide" --out "$R" $CF ${EXTRA[$CFG]} ic.oscar_path="$OSC" eos.model=table_$EOS run.event_first=$FIRST run.n_events=$NEV \
        particlisation.corona.mode=${MODE[$CFG]} > "$OUT/timing/collide_${CFG}_$EOS.log" 2>&1
    echo "collide $CFG $EOS: $(( $(date +%s) - s0 )) s for $NEV events"
    for ((e = FIRST; e < FIRST + NEV; e++)); do
      EV=$(printf "event_%03d" $e)
      "$BUILD/spf_sample" --surface "$R/$EV/$EV" --out "$OUT/samples/$CFG/$EOS/$EV" sampler.n_samples=$NS sampler.threads=$THR \
          sampler.mu_rule=rho_matched > "$OUT/timing/sample_${CFG}_${EOS}_$EV.log" 2>&1
    done
    "$PY" tools/particlisation/census.py --run "${CFG}_$EOS=$R" --table table_two_phase=data/eos/generated/eos_two_phase \
        --table table_maxwell=data/eos/generated/eos_maxwell --ab tools/particlisation/data/ab_two_phase.json --out "$OUT/census/${CFG}_$EOS" --delete-bin \
        > "$OUT/timing/census_${CFG}_$EOS.log" 2>&1
  done
done
"$PY" tools/particlisation/corona_analyse.py "$OUT" --json "$OUT/corona_modes.json"
echo "done"
