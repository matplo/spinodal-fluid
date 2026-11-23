#!/usr/bin/env python3
"""Run spf_collide over the events of one OSCAR file; resumable (finished events are skipped by the program itself).

  run_batch.py --exe build/cuda-turing75/spf_collide --oscar runs/events_e3/smash_data/particle_lists.oscar --out runs/e3_two_phase \
      --config config/collisions/common.json --config config/collisions/e3.json eos.model=table_two_phase [--first 0 --n 20] [--chunk 5]

Everything after the options that has the form key=value is passed on as a configuration override. --chunk N runs the event list in
processes of N events (so that an interrupted batch loses at most one chunk; every finished event stays on disk). Calling the same command
again continues where it stopped: an event with a summary.json is not run again (a failed event is run again only with run.retry_failed=true).
Writes <out>/batch_info.json (command lines, git commit of the source tree, return codes). Exit code: 0 if every event ended with status
fluid_end or t_max, 3 if some event failed, 1 on a program error.
"""
import argparse
import json
import os
import subprocess
import sys
import time


def git_commit(path):
    try:
        return subprocess.check_output(["git", "-C", path, "describe", "--always", "--dirty"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--oscar", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--n", type=int, default=-1, help="number of events (-1: all events of the file)")
    ap.add_argument("--chunk", type=int, default=0, help="events per process (0: one process for all)")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    ev = set()
    with open(a.oscar) as f:
        for line in f:
            if line.startswith("# event") and " out " in line and "start" in line:
                ev.add(line.split()[2])
    nfile = len(ev)
    last = nfile if a.n < 0 else min(nfile, a.first + a.n)
    os.makedirs(a.out, exist_ok=True)
    chunk = a.chunk if a.chunk > 0 else max(1, last - a.first)
    info = {"exe": os.path.abspath(a.exe), "oscar": os.path.abspath(a.oscar), "git": git_commit(os.path.dirname(os.path.abspath(a.exe))),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"), "calls": []}
    worst = 0
    for s in range(a.first, last, chunk):
        n = min(chunk, last - s)
        cmd = [a.exe, "--out", a.out]
        for c in a.config:
            cmd += ["--config", c]
        cmd += ["ic.oscar_path=" + json.dumps(os.path.abspath(a.oscar)), "run.event_first=%d" % s, "run.n_events=%d" % n] + a.overrides
        t0 = time.time()
        rc = subprocess.call(cmd)
        info["calls"].append({"cmd": cmd, "return_code": rc, "wall_s": time.time() - t0})
        worst = max(worst, rc)
        json.dump(info, open(os.path.join(a.out, "batch_info.json"), "w"), indent=1)
        if rc not in (0, 3):
            print("spf_collide returned %d; stopping" % rc, file=sys.stderr)
            break
    return worst


if __name__ == "__main__":
    sys.exit(main())
