#!/usr/bin/env bash
# Run on the GPU server via scripts/remote.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $# == 0 || ( $# == 1 && $1 == --ck ) ]] || { echo 'usage: build.sh [--ck]' >&2; exit 2; }
export BUILD_CK=${1:-}
export CK_ROOT=${CK_ROOT:-reference/ck-bshd}
if [[ $BUILD_CK == --ck ]]; then
    test -f "$CK_ROOT/example/ck_tile/01_fmha/fmha_fwd.hpp" || {
        echo 'Set CK_ROOT to the composablekernel directory (see README).' >&2; exit 2;
    }
fi
mkdir -p build results
flags=(-O3 -std=c++17 --offload-arch=gfx1250 -fPIC -Iinclude
    -fgpu-flush-denormals-to-zero -mllvm -enable-post-misched=0
    -Rpass-analysis=kernel-resource-usage)
hipcc "${flags[@]}" -shared src/bshd.hip -o build/libfav4_bshd.so >results/build-hip.log 2>&1
hipcc "${flags[@]}" tests/bshd_tdm.hip -o build/bshd_tdm >results/build-tdm.log 2>&1
build/bshd_tdm >results/tdm-validation.log 2>&1
if [[ $BUILD_CK == --ck ]]; then
    hipcc "${flags[@]}" -shared -I"$CK_ROOT/include" -I"$CK_ROOT/example/ck_tile/01_fmha" \
        -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -DUSE_NEW_UNIFIED_FRAMEWORK=0 \
        -DCK_CMAKE_GPU_TARGET_IDS=0x1250 -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=5 \
        -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false \
        benchmarks/ck_bshd.hip -o build/libck_bshd.so >results/build-ck.log 2>&1
fi
python3 - <<'PY'
from pathlib import Path
import hashlib, json, os, subprocess
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for d in ('src','include','tests') for p in sorted(Path(d).glob('*')) if p.is_file()]
files += [Path('scripts/build.sh'),Path('benchmarks/ck_bshd.hip')]
binaries=[Path('build/libfav4_bshd.so')]
m=dict(compiler=subprocess.check_output(['hipcc','--version'],text=True),
       source_sha256={str(p):sha(p) for p in files})
if os.environ['BUILD_CK']=='--ck':
    ck=Path(os.environ['CK_ROOT'])
    binaries.append(Path('build/libck_bshd.so'))
    m['ck_root']=str(ck.resolve())
    m['ck_source_sha256']={str(p.relative_to(ck)):sha(p)
        for d in ('include','example/ck_tile/01_fmha')
        for p in sorted((ck/d).rglob('*')) if p.is_file()}
m['binary_sha256']={p.name:sha(p) for p in binaries}
data=json.dumps(m,indent=2)+'\n'
Path('build/build-manifest.json').write_text(data)
Path('results/build-manifest.json').write_text(data)
PY
