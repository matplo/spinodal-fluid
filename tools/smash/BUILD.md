# Building SMASH 3.4.2 (Linux x86_64)

SMASH (GPL-3) is used as an external program: only its output files are read; no SMASH source is part of this repository.
SMASH 3.4.2 requires exactly Pythia 8.316. The recipe below was verified on Ubuntu 22.04 (system GCC 11, too old for C++20 SMASH),
so the compilers come from a conda-forge environment.

## Environment

```bash
export MAMBA_ROOT_PREFIX=<micromamba root>
micromamba create -y -n spf-smash -c conda-forge --override-channels gcc=13 gxx=13 cmake make gsl eigen git curl binutils patchelf
```

Do not install the conda `pythia8` package: its headers shadow the ones of the pinned source build.

## Pythia 8.316 and SMASH

```bash
micromamba run -n spf-smash bash <<'EOS'
set -e
cd <build tree directory>
curl -sSL -o pythia8316.tgz https://pythia.org/releases/pythia83/pythia8316.tgz      # pythia.org/download/... is a 404
tar xf pythia8316.tgz && rm pythia8316.tgz && cd pythia8316
./configure --cxx-common='-std=c++17 -march=native -O3 -fPIC -pthread'
make -j12                                                  # about 1.5 minutes
cd ..
git clone --depth 1 --branch SMASH-3.4.2 https://github.com/smash-transport/smash.git
cd smash && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPythia_CONFIG_EXECUTABLE=$PWD/../../pythia8316/bin/pythia8-config ..
make -j12 smash                                            # about 2 minutes
# CMake puts the system library directory (found for GSL) first in the RPATH, so the old system libstdc++ wins; put the conda one first:
ENV=<micromamba root>/envs/spf-smash
patchelf --set-rpath "$ENV/lib:$PWD/src:$PWD/../../pythia8316/lib" smash
EOS
```

## Running

After the `patchelf` step the binary finds the conda `libstdc++` (GLIBCXX_3.4.32) by itself; without it, the build fails at start with
"version `GLIBCXX_3.4.32' not found" (an `LD_LIBRARY_PATH` does not help, because the RPATH entry comes first). SMASH must be started with
its build tree in place (Pythia is linked statically, but `share/Pythia8/xmldoc` is read at run time).

```bash
python3 tools/smash/run_smash.py --ekin 3 --nevents 20 --seed 1 \
    --smash <build tree>/smash/build/smash --out runs/pbpb_3
# -> runs/pbpb_3/smash_data/particle_lists.oscar   (OSCAR2013, one block per event at the fluid start time)
```

Smoke test: `smash -v` prints the version (3.4.2); the driver run above finishes in seconds per event.
