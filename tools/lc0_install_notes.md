# Lc0 install + benchmark notes

Steps to reproduce the Lc0 measurements in `BENCHMARKS.md` on a similar Linux + Blackwell setup.

## Why we built from source

Lc0 ships prebuilt binaries only for Windows and Mac. Linux users build via `meson + ninja`.
On this machine the build hit two snags worth recording:

1. **CUDA 12.0 doesn't support `compute_120` (Blackwell sm_120).** Solution: build for `sm_89` and embed PTX so the driver JIT-compiles to sm_120 at load time.
2. **nvcc 12.0 + gcc 13's `<amxtileintrin.h>` conflict.** gcc 13's AMX intrinsics use builtins that nvcc's host preprocessor doesn't recognize. Solution: pre-define the AMX header guards so the headers self-skip when `<immintrin.h>` pulls them in.

Both are patches to `meson.build` — included below.

## Prereqs

- CUDA toolkit (12.0+; we used 12.0.140)
- gcc/g++ 12 or 13
- ninja (`sudo apt install ninja-build` or via miniconda)
- meson (`pip install --user meson`)
- About 2 GB disk for source + build artifacts

cuDNN is **not** required if building only the `plain_cuda` backend (which is what we used).

## Steps

```bash
# 1. Get source
git clone --depth 1 https://github.com/LeelaChessZero/lc0.git /tmp/lc0
cd /tmp/lc0

# 2. Apply two meson.build patches (see "Patches" section below)

# 3. Configure
PATH="$HOME/.local/bin:$PATH" meson setup build/release --buildtype release \
  -Dcudnn=false -Dplain_cuda=true \
  -Dcc_cuda=89 -Dnative_cuda=false \
  -Dnvcc_ccbin=/usr/bin/g++-12 \
  -Dopenblas=false -Dopencl=false -Ddx=false -Dtensorflow=false -Dblas=false

# 4. Build (~5 minutes)
PATH="$HOME/.local/bin:$PATH" meson compile -C build/release

# 5. Verify
build/release/lc0 --help | head -5

# 6. Get a small Lc0-format network
wget https://github.com/CSSLab/maia-chess/raw/master/maia_weights/maia-1100.pb.gz
```

## Patches to `meson.build`

Locate the CUDA argument-construction block (around line 487 in v0.33.0-dev master).

**Patch 1: predefine AMX header guards**

```diff
       nvcc_arguments += ['-Xcompiler', '-fPIC']
+      # Workaround: nvcc 12.0's host preprocessor doesn't recognize AMX
+      # builtins added in gcc 12+. Pre-define the header guards so the AMX
+      # intrinsics headers self-skip when pulled in via <immintrin.h>.
+      nvcc_arguments += ['-D_AMXTILEINTRIN_H_INCLUDED',
+                         '-D_AMXINT8INTRIN_H_INCLUDED',
+                         '-D_AMXBF16INTRIN_H_INCLUDED']
       if get_option('debug')
         nvcc_arguments += ['-g']
       endif
```

**Patch 2: embed PTX so sm_120 driver can JIT-compile**

```diff
     if cuda_cc != ''
-      nvcc_extra_args = ['-arch=compute_' + cuda_cc, '-code=sm_' + cuda_cc]
+      # Also embed PTX so the driver can JIT to newer architectures (sm_120).
+      nvcc_extra_args = ['-gencode=arch=compute_' + cuda_cc + ',code=sm_' + cuda_cc,
+                         '-gencode=arch=compute_' + cuda_cc + ',code=compute_' + cuda_cc]
```

## Reproduction commands

The numbers in `BENCHMARKS.md` came from these invocations:

```bash
cd /tmp

# Pure NN inference scan (cuda-fp16 backend)
/tmp/lc0/build/release/lc0 backendbench \
  --weights=maia-1100.pb.gz --backend=cuda-fp16 \
  --batches=20 --start-batch-size=1 --max-batch-size=128 --batch-step=8

# End-to-end self-play (matches Caissawary's 200 sims/move config)
time /tmp/lc0/build/release/lc0 selfplay \
  --weights=maia-1100.pb.gz --backend=cuda-fp16 \
  --games=10 --parallelism=8 --visits=200
```

`selfplay` reports `npm` (nodes per move) and totals. Divide total nodes by wall time for sims/sec; divide total moves by wall time for samples/sec.

## Caveats

- Maia-1100 is a 12×128 SE-ResNet, twice as deep as Caissawary's 6×128. We do not have a publicly-distributed Lc0 network at exactly our size. Lc0 on a 6×128 would likely be ~2× faster than the measured Maia-1100 numbers.
- The `cuda-fp16` backend was used (Lc0 itself recommends it over `cuda` fp32 on this hardware).
- `parallelism=8` was chosen by hand; raising it further might help for very small nets but didn't move the needle in spot checks.
