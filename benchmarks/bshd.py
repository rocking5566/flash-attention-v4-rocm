#!/usr/bin/env python3
"""Native BSHD BF16 D128 validation and same-device qr_tdm_sched comparison."""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time

import torch
ROOT = Path(__file__).resolve().parents[1]

class Args(C.Structure):
    _fields_ = [(n, C.c_void_p) for n in ("q", "k", "v", "o", "lse")] + [
        (n, C.c_int) for n in ("batch", "heads", "sq", "sk")] + [("scale", C.c_float)]

def errors(o, lse, ref, ref_lse):
    diff = (o.double()-ref).abs()
    atol, rtol, rms_limit, lse_limit = 0.01, 0.01, 0.005, 2e-4
    norm = (diff/(atol+rtol*ref.abs())).max().item()
    rms_rel = (diff.square().mean().sqrt()/ref.square().mean().sqrt().clamp_min(1e-30)).item()
    lse_err = (lse.double()-ref_lse).abs().max().item() if lse is not None else None
    result = dict(max_abs=diff.max().item(), rms_rel=rms_rel, max_tolerance_ratio=norm, lse_max_abs=lse_err)
    assert torch.isfinite(o).all() and norm <= 1 and rms_rel <= rms_limit, result
    assert lse_err is None or lse_err <= lse_limit, result
    return result

def graph_for(fn, count):
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(count): fn()
    return g

def graph_us(g, count, repeats, warmup):
    # Each measurement: exactly 5 unmeasured kernel nodes, then 20 measured
    # nodes. Capturing a graph does not execute its nodes. Graph replay avoids
    # Python/ctypes launch gaps. Independent shuffled rounds assess drift.
    warmup.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats): g.replay()
    end.record(); end.synchronize()
    return start.elapsed_time(end)*1000/(count*repeats)

class Runner:
    def __init__(self):
        self.hip = C.CDLL(str(ROOT/'build/libfav4_bshd.so'))
        self.ck = C.CDLL(str(ROOT/'build/libck_bshd.so'))
        self.hip.fav4_bshd_forward.argtypes = [C.POINTER(Args),C.c_int,C.c_int,C.c_void_p]
        self.hip.fav4_bshd_select_config.argtypes = [C.POINTER(Args)]
        self.hip.fav4_bshd_select_config.restype = C.c_int
        self.ck.ck_bshd_forward.argtypes = [C.POINTER(Args),C.c_int,C.c_void_p]
        self.hip.fav4_bshd_forward.restype = self.ck.ck_bshd_forward.restype = C.c_int

    def call(self,q,k,v,o,lse,causal,config=0):
        assert all(x.dtype==torch.bfloat16 and x.is_cuda and x.is_contiguous() for x in (q,k,v,o))
        b,sq,h,d=q.shape
        assert d==128 and k.shape==v.shape and k.shape[0]==b and k.shape[2:]==(h,d) and o.shape==q.shape
        assert lse is None or (lse.shape==(b,h,sq) and lse.is_contiguous() and lse.dtype==torch.float32)
        a=Args(q.data_ptr(),k.data_ptr(),v.data_ptr(),o.data_ptr(),lse.data_ptr() if lse is not None else None,
               b,h,sq,k.shape[1],1/math.sqrt(d))
        stream=torch.cuda.current_stream().cuda_stream
        status=self.ck.ck_bshd_forward(C.byref(a),int(causal),stream) if config==-1 else \
            self.hip.fav4_bshd_forward(C.byref(a),int(causal),config,stream)
        if status: raise RuntimeError(f'config {config}: HIP status {status}')

def inputs(b,h,sq,sk,pattern='normal'):
    q=torch.randn(b,sq,h,128,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(b,sk,h,128,device='cuda',dtype=torch.bfloat16)
    v=torch.randn_like(k)
    if pattern=='zero':q.zero_();k.zero_()
    elif pattern=='large':q.mul_(4);k.mul_(4)
    elif pattern=='negative':q.fill_(8);k.fill_(-8)
    elif pattern=='late_max':
        q.fill_(1)
        k.copy_((torch.arange(sk,device='cuda')//64).view(1,sk,1,1).expand_as(k))
    elif pattern in ('sum_guard','below_guard','above_guard','mixed_guard'):
        # Exercise the partial-sum guard both without and with a max update.
        # The mixed case also makes different rows overflow or stay unchanged.
        shift={'sum_guard':0.125,'below_guard':0.484375,'above_guard':0.5,'mixed_guard':16}[pattern]
        q.fill_(1);k.zero_();k[:,128:].fill_(shift)
        if pattern=='mixed_guard':
            q[:,0::4].zero_();q[:,1::4].fill_(0.5);q[:,2::4].fill_(-1)
    elif pattern in ('first_low','first_high','mixed_first','late_after_first'):
        # The initial base uses column zero. Exercise first-tile recovery and
        # rows that recover differently, including causal row zero by itself.
        q.fill_(1);k.zero_();k[:,0].fill_(16 if pattern=='first_high' else -16)
        if pattern=='mixed_first':
            q[:,0::3].fill_(-1);q[:,1::3].zero_()
        if pattern=='late_after_first':k[:,128:].fill_(16)
    return q,k,v

def reference(q,k,v,causal,rows):
    # Gather only reference rows. All timed tensors retain their native BSHD storage.
    qs=q[:,rows].permute(0,2,1,3).double()
    ks=k.permute(0,2,1,3).double()
    scores=(qs@ks.transpose(-1,-2))/math.sqrt(128)
    del qs,ks
    if causal:scores.masked_fill_(torch.arange(k.shape[1],device='cuda')[None,:]>rows[:,None],-float('inf'))
    lse=torch.logsumexp(scores,-1)
    probabilities=torch.softmax(scores,-1)
    del scores
    return probabilities@v.permute(0,2,1,3).double(),lse

def check(runner,q,k,v,o,lse,causal,config,rows,ref,ref_lse):
    o.fill_(float('nan'))
    if lse is not None:lse.fill_(float('nan'))
    runner.call(q,k,v,o,lse,causal,config)
    return errors(o[:,rows].permute(0,2,1,3),lse[:,:,rows] if lse is not None else None,
                  ref,ref_lse)

def validate(runner,output,configs):
    cases=[(1,1,1,1,'normal'),(2,3,17,31,'normal'),(2,3,63,65,'normal'),
           (1,8,127,129,'normal'),(3,8,193,257,'normal'),(1,8,256,512,'zero'),
           (1,8,256,512,'large'),(1,8,256,512,'late_max'),(1,8,129,65,'negative'),
           (2,8,512,512,'normal'),(1,8,257,257,'sum_guard'),
           (1,8,257,257,'below_guard'),(1,8,257,257,'above_guard'),
           (32,8,257,257,'mixed_guard')]
    # Exercise the actual fixed-shape Auto paths, as well as generic tiles above.
    cases += [(32,8,n,n,pattern) for n in (512,2048)
              for pattern in ('first_low','first_high','mixed_first','late_after_first')]
    result=[]
    for b,h,sq,sk,pattern in cases:
        q,k,v=inputs(b,h,sq,sk,pattern)
        rows=(torch.arange(sq,device='cuda') if sq<=512 else
              torch.tensor(sorted({0,1,15,16,31,63,64,127,128,129,255,sq//2,sq-1}),device='cuda'))
        o=torch.empty_like(q);lse=torch.empty((b,h,sq),device='cuda',dtype=torch.float32)
        for causal in (False,True):
            ref,ref_lse=reference(q,k,v,causal,rows)
            for config in configs:
                for with_lse in (False,True):
                    e=check(runner,q,k,v,o,lse if with_lse else None,causal,config,rows,ref,ref_lse)
                    result.append(dict(b=b,h=h,sq=sq,sk=sk,pattern=pattern,causal=causal,
                        config=config,lse=with_lse,reference_rows=rows.tolist(),**e))
        print(f'PASS BSHD b{b}h{h} {sq}x{sk} {pattern}',flush=True)
        output.write_text(json.dumps(result,indent=2)+'\n')
    print(f'PASS {len(result)} BF16 correctness comparisons (high-precision reference)',flush=True)

def telemetry(enabled=True):
    if not enabled: return dict(available=False,reason="disabled by --no-telemetry")
    # Read-only query, outside the timed interval; device index is inherited from .remote.env.
    cmd=['amd-smi','metric','--gpu',os.environ['HIP_VISIBLE_DEVICES'],
         '--clock','--temperature','--power','--usage','--json']
    try:
        p=subprocess.run(cmd,text=True,capture_output=True,timeout=30)
        return dict(timestamp=time.time(),command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
    except (OSError,subprocess.TimeoutExpired) as e:
        return dict(timestamp=time.time(),error=str(e))

def benchmark(runner,output,seqs,config,order_phase,collect_telemetry=True):
    records=[]
    for n in seqs:
        q,k,v=inputs(32,8,n,n)
        rows=torch.tensor(sorted(set([0,1,63,64,127,128,n-1]+
            list(range(0,n,max(1,n//24))))),device='cuda')
        for causal in (False,True):
            ref,ref_lse=reference(q,k,v,causal,rows)
            graphs,warmups,outs,metrics={},{},{},{}
            for vi in (config,-1):
                o=torch.empty_like(q);outs[vi]=o
                lse=torch.empty((32,8,n),device='cuda',dtype=torch.float32)
                metrics[vi]=dict(without_lse=check(runner,q,k,v,o,None,causal,vi,rows,ref,ref_lse),
                    with_lse=check(runner,q,k,v,o,lse,causal,vi,rows,ref,ref_lse))
                fn=lambda vi=vi,o=o:runner.call(q,k,v,o,None,causal,vi)
                graphs[vi]=graph_for(fn,20);warmups[vi]=graph_for(fn,5)
            del ref,ref_lse,lse
            torch.cuda.synchronize()
            before=telemetry(collect_telemetry)
            samples={config:[],-1:[]}
            order=[config,-1,-1,config, -1,config,config,-1, config,-1,-1,config]
            if order_phase:order=[-1 if vi==config else config for vi in order]
            for vi in order:
                samples[vi].append(graph_us(graphs[vi],20,1,warmups[vi]))
            after=telemetry(collect_telemetry)
            # Same useful QK+PV FLOP convention as CK: no softmax FLOPs or padded tiles.
            area=n*(n+1)//2 if causal else n*n
            flop=4*32*8*128*area
            for vi in (config,-1):
                us=statistics.mean(samples[vi]);tf=[flop/t/1e6 for t in samples[vi]]
                row=dict(b=32,h=8,sq=n,sk=n,d=128,dtype='bf16',layout='BSHD',causal=causal,lse=False,
                    kernel='qr_tdm_sched' if vi==-1 else 'fav4_bshd',config=vi,
                    selected_config=128 if vi==-1 else (runner.hip.fav4_bshd_select_config(C.byref(
                        Args(batch=32,heads=8,sq=n,sk=n))) if vi==0 else vi),
                    strides_q=list(q.stride()),strides_k=list(k.stride()),strides_o=list(outs[vi].stride()),
                    samples_us=samples[vi],mean_us=us,median_us=statistics.median(samples[vi]),
                    effective_flop=flop,samples_tflops=tf,mean_tflops=statistics.mean(tf),
                    tflops_from_mean_us=flop/us/1e6,speedup_vs_ck=statistics.mean(samples[-1])/us,
                    correctness=metrics[vi],reference_rows=rows.tolist(),
                    telemetry_before=before,telemetry_after=after)
                records.append(row)
                print(f'B32 H8 S{n} causal={causal} {row["kernel"]}: {us:.3f} us, '
                      f'{row["tflops_from_mean_us"]:.2f} TFLOPS, {row["speedup_vs_ck"]:.4f}x CK',flush=True)
            output.write_text(json.dumps(records,indent=2)+'\n')
            del graphs,warmups,outs,metrics,o,fn
        del q,k,v
        torch.cuda.empty_cache()
    return records

def capture(runner,output,sequence,config,causal):
    q,k,v=inputs(32,8,sequence,sequence)
    o=torch.empty_like(q)
    for _ in range(5):runner.call(q,k,v,o,None,causal,config)
    torch.cuda.synchronize()
    runner.call(q,k,v,o,None,causal,config)
    torch.cuda.synchronize()
    rows=torch.tensor(sorted({0,sequence//2,sequence-1}),device='cuda')
    ref,lse=reference(q,k,v,causal,rows)
    metrics=errors(o[:,rows].permute(0,2,1,3),None,ref,lse)
    output.write_text(json.dumps(dict(b=32,h=8,sq=sequence,sk=sequence,d=128,
        layout='BSHD',dtype='bf16',causal=causal,config=config,warmup=5,
        captured_iteration=6,correctness=metrics),indent=2)+'\n')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['validate','bench','capture'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seqs',default='512,1024,2048,4096,8192,16384,32768')
    p.add_argument('--config',type=int,default=0)
    p.add_argument('--causal',type=int,choices=[0,1],default=0,help='Mask for ATT capture')
    p.add_argument('--min-speedup',type=float,default=None)
    p.add_argument('--validate-configs',default='-1,0,1,2,3,25,132,133,145,150,155,156')
    p.add_argument('--no-telemetry',action='store_true',help='Skip amd-smi clock/temperature/power/usage queries')
    p.add_argument('--order-phase',type=int,choices=[0,1],default=0)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    prop=torch.cuda.get_device_properties(0)
    assert prop.gcnArchName.split(':')[0]=='gfx1250'
    torch.manual_seed(11939)
    hip_runtime=C.CDLL('libamdhip64.so')
    runtime_version=C.c_int()
    assert hip_runtime.hipRuntimeGetVersion(C.byref(runtime_version))==0
    runtime_paths=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
        if any(name in line for name in ('libamdhip64.so','libhsa-runtime64.so','libamd_comgr.so'))})
    m=dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),hostname=socket.gethostname(),
        device=str(prop),hip_visible_devices=os.environ.get('HIP_VISIBLE_DEVICES'),torch=torch.__version__,
        hip_runtime_version=runtime_version.value,
        runtime_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in runtime_paths},
        dtype='bf16',layout='BSHD',seed=11939,input_distribution='torch.randn BF16',
        order_phase=a.order_phase,telemetry_enabled=not a.no_telemetry,
        timing='Each observation: warmup=5, repeat=20 HIP graph nodes with GPU events; '
               'ABBA, BAAB, ABBA = 6 observations/kernel/shape/process; LSE off; warm cache; kernel only',
        build_manifest=json.loads((ROOT/'build/build-manifest.json').read_text()),
        harness_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                        for f in (Path(__file__).resolve(),)})
    a.output.with_suffix('.metadata.json').write_text(json.dumps(m,indent=2)+'\n')
    runner=Runner()
    if a.mode=='validate':validate(runner,a.output,list(map(int,a.validate_configs.split(','))))
    elif a.mode=='capture':capture(runner,a.output,int(a.seqs),a.config,bool(a.causal))
    else:
        rows=benchmark(runner,a.output,list(map(int,a.seqs.split(','))),a.config,a.order_phase,not a.no_telemetry)
        if a.min_speedup is not None:
            for hip,ck in zip(rows[::2],rows[1::2]):
                ratio=hip['speedup_vs_ck']
                assert ratio>=a.min_speedup, f"PERF_GAP S{hip['sq']} causal={hip['causal']}: {ratio:.4f}x < {a.min_speedup}x"

if __name__=='__main__':main()
