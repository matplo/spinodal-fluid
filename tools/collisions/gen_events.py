#!/usr/bin/env python3
"""Generate N SMASH (cascade, central Pb+Pb) events at one fixed-target energy into an OSCAR2013 file: a wrapper of tools/smash/run_smash.py.

  gen_events.py --ekin 3 --nevents 20 --seed 1 --smash <build>/smash --out runs/events_e3 [--smash-wrapper 'micromamba run -n spf-smash']

Writes <out>/smash_data/particle_lists.oscar (one block per event at the fluid start time), <out>/run_info.json and prints the path of the
OSCAR file and the number of events found in it. Events are numbered 0 ... N-1 in file order; the same --seed gives the same events, so
every equation of state can be run on the same list. Standard library only.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def count_events(path):
    ev = set()
    with open(path) as f:
        for line in f:
            if line.startswith("# event") and " out " in line and "start" in line:
                ev.add(line.split()[2])
    return len(ev)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ekin", type=float, required=True, help="kinetic energy per nucleon in the fixed-target frame [GeV]")
    ap.add_argument("--nevents", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--smash", required=True, help="path of the smash executable (tools/smash/BUILD.md)")
    ap.add_argument("--smash-wrapper", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="further arguments for run_smash.py (e.g. --radius-mode fixed)")
    a = ap.parse_args()
    cmd = [sys.executable, os.path.join(HERE, "..", "smash", "run_smash.py"), "--ekin", str(a.ekin), "--nevents", str(a.nevents),
           "--seed", str(a.seed), "--smash", a.smash, "--out", a.out]
    if a.smash_wrapper:
        cmd += ["--smash-wrapper", a.smash_wrapper]
    cmd += a.extra
    rc = subprocess.call(cmd)
    if rc != 0:
        return rc
    oscar = os.path.join(os.path.abspath(a.out), "smash_data", "particle_lists.oscar")
    if not os.path.exists(oscar):
        print("no OSCAR file written: see", os.path.join(a.out, "smash.log"), file=sys.stderr)
        return 1
    print("oscar:", oscar)
    print("events (blocks '# event N out'): %d" % count_events(oscar))
    return 0


if __name__ == "__main__":
    sys.exit(main())
