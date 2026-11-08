#!/usr/bin/env bash
# Run on the GPU server via scripts/remote.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
[[ $# == 0 || ( $# == 1 && $1 == --ck ) ]] || { echo 'usage: build.sh [--ck]' >&2; exit 2; }
export BUILD_CK=${1:-}
export HIPCC=${HIPCC:-hipcc}
export FAV4_ARCH=${FAV4_ARCH:-gfx1250}
export CK_ROOT=${CK_ROOT:-reference/ck-bshd}
if [[ $BUILD_CK == --ck ]]; then
    test -f "$CK_ROOT/example/ck_tile/01_fmha/fmha_fwd.hpp" || {
        echo 'Set CK_ROOT to the composablekernel directory (see README).' >&2; exit 2;
    }
fi
export FAV4_RESULTS_DIR=${FAV4_RESULTS_DIR:-results}
out=$FAV4_RESULTS_DIR
mkdir -p build "$out"
flags=(-O3 -std=c++17 --offload-arch="$FAV4_ARCH" -fPIC -Iinclude
    -fgpu-flush-denormals-to-zero -mllvm -enable-post-misched=0
    -Rpass-analysis=kernel-resource-usage)
read -r -a candidate_flags <<< "${FAV4_CXXFLAGS:-}"
# Some A0 runtimes report gfx1250 while accepting gfx1250-strict code objects.
# Keep the strict ELF unchanged; only the HIP bundle selector uses gfx1250.
compile_hip() {
    if [[ $FAV4_ARCH != gfx1250-strict ]]; then "$HIPCC" "$@"; return; fi
    python3 - "$@" <<'PY'
from pathlib import Path
import os, subprocess, sys
cc=os.environ['HIPCC']
args=sys.argv[1:]
idx=args.index('-o'); output=Path(args[idx+1])
common=[x for i,x in enumerate(args) if i not in (idx,idx+1) and not x.startswith('--offload-arch=')]
bundler=os.environ.get('CLANG_OFFLOAD_BUNDLER') or subprocess.check_output(
    [cc,'--print-prog-name=clang-offload-bundler'],text=True).strip()
device=Path(str(output)+'.device.bundle')
obj=Path(str(output)+'.strict.hsaco'); fat=Path(str(output)+'.strict.fat.bin')
subprocess.run([cc,'--offload-arch=gfx1250-strict','--offload-device-only',
    *[x for x in common if x!='-shared'],'-o',str(device)],check=True)
subprocess.run([bundler,'-unbundle','-type=o',
    '-targets=hipv4-amdgcn-amd-amdhsa--gfx1250-strict',
    '-input='+str(device),'-output='+str(obj)],check=True)
subprocess.run([bundler,'-type=o','-bundle-align=4096',
    '-targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx1250',
    '-input=/dev/null','-input='+str(obj),'-output='+str(fat)],check=True)
subprocess.run([cc,'--offload-arch=gfx1250','--offload-host-only',
    '-Xclang','-fcuda-include-gpubinary','-Xclang',str(fat),*common,'-o',str(output)],check=True)
PY
}
schedule_flags=(-mllvm -amdgpu-expert-scheduling-mode
    -mllvm -amdgpu-sched-strategy=max-ilp -mllvm -amdgpu-barrier-signal-wait-latency=32)
compile_hip "${flags[@]}" "${schedule_flags[@]}" "${candidate_flags[@]}" -shared src/bshd.hip -o build/libfav4_bshd.so >"$out/build-hip.log" 2>&1
compile_hip "${flags[@]}" tests/bshd_tdm.hip -o build/bshd_tdm >"$out/build-tdm.log" 2>&1
build/bshd_tdm >"$out/tdm-validation.log" 2>&1
if [[ $BUILD_CK == --ck ]]; then
    compile_hip "${flags[@]}" -shared -I"$CK_ROOT/include" -I"$CK_ROOT/example/ck_tile/01_fmha" \
        -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -DUSE_NEW_UNIFIED_FRAMEWORK=0 \
        -DCK_CMAKE_GPU_TARGET_IDS=0x1250 -DCK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=5 \
        -mllvm -amdgpu-expert-scheduling-mode \
        -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false \
        benchmarks/ck_bshd.hip -o build/libck_bshd.so >"$out/build-ck.log" 2>&1
fi
python3 - <<'PY'
from pathlib import Path
import hashlib, json, os, subprocess
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[p for d in ('src','include','tests') for p in sorted(Path(d).glob('*')) if p.is_file()]
files += [Path('scripts/build.sh'),Path('benchmarks/ck_bshd.hip')]
binaries=[Path('build/libfav4_bshd.so')]
m=dict(compiler=subprocess.check_output([os.environ['HIPCC'],'--version'],text=True),
       compiler_command=os.environ['HIPCC'],offload_arch=os.environ['FAV4_ARCH'],
       hip_bundle_selector='gfx1250' if os.environ['FAV4_ARCH']=='gfx1250-strict' else os.environ['FAV4_ARCH'],
       candidate_scheduling_flags='-mllvm -amdgpu-expert-scheduling-mode '
           '-mllvm -amdgpu-sched-strategy=max-ilp -mllvm -amdgpu-barrier-signal-wait-latency=32',
       candidate_extra_flags=os.environ.get('FAV4_CXXFLAGS',''),
       source_sha256={str(p):sha(p) for p in files})
if os.environ['BUILD_CK']=='--ck':
    ck=Path(os.environ['CK_ROOT'])
    binaries.append(Path('build/libck_bshd.so'))
    m['ck_root']=str(ck.resolve())
    m['ck_compiler']=m['compiler']
    m['ck_offload_arch']=m['offload_arch']
    m['ck_source_sha256']={str(p.relative_to(ck)):sha(p)
        for d in ('include','example/ck_tile/01_fmha')
        for p in sorted((ck/d).rglob('*')) if p.is_file()}
elif Path('build/libck_bshd.so').exists():
    previous=json.loads(Path('build/build-manifest.json').read_text())
    ck_binary=Path('build/libck_bshd.so')
    assert sha(ck_binary)==previous['binary_sha256'][ck_binary.name], 'Rebuild CK with --ck'
    assert sha(Path('benchmarks/ck_bshd.hip'))==previous['source_sha256']['benchmarks/ck_bshd.hip'], 'Rebuild changed CK comparator with --ck'
    assert previous.get('ck_offload_arch','gfx1250')==m['offload_arch'], 'Rebuild CK for the selected architecture with --ck'
    m['ck_compiler']=previous.get('ck_compiler',previous['compiler'])
    m['ck_offload_arch']=previous.get('ck_offload_arch','gfx1250')
    for key in ('ck_root','ck_source_sha256'):m[key]=previous[key]
    binaries.append(ck_binary)
m['binary_sha256']={p.name:sha(p) for p in binaries}
data=json.dumps(m,indent=2)+'\n'
Path('build/build-manifest.json').write_text(data)
(Path(os.environ['FAV4_RESULTS_DIR'])/'build-manifest.json').write_text(data)
PY
