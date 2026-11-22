#!/usr/bin/env python3
"""Run SMASH (external program, output files only) for Pb+Pb, b = 0, cascade mode, and write an OSCAR2013 particle list at the
fluid start time. Standard library only.

  run_smash.py --ekin 3 --nevents 20 --seed 1 --smash /path/to/smash/build/smash --out out_dir
  run_smash.py --ekin 3 --nevents 1 --measure --smash ... --out out_dir     # time-convention measurement (see docs/initial_state.md)

Time convention: SMASH starts at negative time and places the nuclei so that the Lorentz-contracted spheres of radius R_nuc + d
touch exactly at t = 0 (collidermodus.cc, ColliderModus::initial_conditions). The fluid start time is counted from this moment:
t_SMASH = t_init + offset, t_init = 2 R / sqrt(gamma_cm^2 - 1), offset = --time-offset (default 0.0; the measured value is in
docs/initial_state.md).
"""
import argparse, json, math, os, string, subprocess, sys

M_N = 0.940  # GeV, as in the project (config constants.m_N_GeV)


def kinematics(ekin, m=M_N):
    g_lab = 1.0 + ekin / m
    g_cm = math.sqrt(0.5 * (g_lab + 1.0))
    return {"gamma_lab": g_lab, "gamma_cm": g_cm, "sqrt_snn": 2.0 * g_cm * m}


def t_init(ekin, radius, m=M_N):
    g = kinematics(ekin, m)["gamma_cm"]
    return 2.0 * radius / math.sqrt(g * g - 1.0)


def radius_from_mode(mode, r0=1.2, a=208, fixed=6.62):
    if mode == "r0_A13":
        return r0 * a ** (1.0 / 3.0)
    if mode == "fixed":
        return fixed
    raise SystemExit("radius mode must be r0_A13 or fixed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ekin", type=float, required=True, help="kinetic energy per nucleon, fixed target [GeV]")
    ap.add_argument("--nevents", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--smash", required=True, help="path of the smash executable")
    ap.add_argument("--smash-wrapper", default="", help="command prefix, e.g. 'micromamba run -n spf-smash' (libraries of the build env)")
    ap.add_argument("--ld-library-path", default="", help="prepended to the library search path of SMASH: the lib directory of the conda "
                    "environment it was built in (only needed if the binary was not patched with patchelf, see BUILD.md; "
                    "an RPATH entry in the binary takes precedence over it)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius-mode", default="r0_A13", choices=["r0_A13", "fixed"])
    ap.add_argument("--radius", type=float, default=0.0, help="explicit R [fm] overrides --radius-mode")
    ap.add_argument("--time-offset", type=float, default=0.0, help="SMASH time of the fluid start = t_init + offset [fm/c]")
    ap.add_argument("--fermi-motion", default="frozen", choices=["off", "on", "frozen"])
    ap.add_argument("--initial-distance", type=float, default=4.0)
    ap.add_argument("--measure", action="store_true", help="output every 0.5 fm/c up to 2 t_init and the collision list")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    R = a.radius if a.radius > 0 else radius_from_mode(a.radius_mode)
    ti = t_init(a.ekin, R)
    t_out = ti + a.time_offset
    if a.measure:
        n = int(math.ceil(2.0 * ti / 0.5))
        times = [round(0.5 * i, 3) for i in range(0, n + 1)]
        end_time = times[-1]
        coll = "    Collisions:\n        Format: [\"Oscar2013\"]\n        Print_Start_End: False"
    else:
        times = [round(t_out, 4)]
        end_time = round(t_out + 0.2, 4)   # a little after the output time; no forced decays, so nothing changes in between
        coll = ""
    tpl = string.Template(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pbpb_cascade.yaml.in")).read())
    cfg = tpl.substitute(end_time=end_time, seed=a.seed, nevents=a.nevents, ekin=a.ekin, fermi_motion=a.fermi_motion,
                         initial_distance=a.initial_distance, output_times="[" + ", ".join(str(t) for t in times) + "]",
                         collisions_block=coll)
    a.out = os.path.abspath(a.out)
    a.smash = os.path.abspath(a.smash)
    os.makedirs(a.out, exist_ok=True)
    cpath = os.path.join(a.out, "smash_config.yaml")
    open(cpath, "w").write(cfg)
    info = {"ekin_GeV": a.ekin, "radius_fm": R, "t_init_fm": ti, "time_offset_fm": a.time_offset, "smash_output_time_fm": t_out,
            "nevents": a.nevents, "seed": a.seed, **kinematics(a.ekin)}
    json.dump(info, open(os.path.join(a.out, "run_info.json"), "w"), indent=1)
    cmd = (a.smash_wrapper.split() if a.smash_wrapper else []) + [a.smash, "-i", cpath, "-o", os.path.join(a.out, "smash_data"), "-f"]
    print(" ".join(cmd))
    print(json.dumps(info))
    if a.dry_run:
        return 0
    with open(os.path.join(a.out, "smash.log"), "w") as log:
        env = dict(os.environ)
        if a.ld_library_path:
            old = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = a.ld_library_path + (":" + old if old else "")
        rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=os.path.dirname(a.smash), env=env)
    if rc != 0:
        print("SMASH failed, see", os.path.join(a.out, "smash.log"), file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
