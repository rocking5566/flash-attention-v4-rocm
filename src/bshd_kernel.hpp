// Included inside a private kernel namespace.
using F8 = float __attribute__((ext_vector_type(8)));
using F2 = float __attribute__((ext_vector_type(2)));
template<class T> using V8 = T __attribute__((ext_vector_type(8)));
template<class T> using V16 = T __attribute__((ext_vector_type(16)));
constexpr int D = 128;

template<class T>
__device__ __forceinline__ F8 mma(V16<T> a, V16<T> b, F8 c) {
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    static_assert(std::is_same_v<T, __bf16>);
    return __builtin_amdgcn_wmma_f32_16x16x32_bf16(0,a,0,b,0,c,0,0);
#else
    return c;
#endif
}

template<class T>
__device__ __forceinline__ V16<T> read_k32(const T* p, int pitch, int row, int k) {
    const int half = (threadIdx.x % 32) / 16;
    V16<T> r;
    using ReadVec=const V8<T>;
    const V8<T> a = *reinterpret_cast<ReadVec*>(p + row*pitch + k + half*8);
    const V8<T> b = *reinterpret_cast<ReadVec*>(p + row*pitch + k + 16 + half*8);
#pragma unroll
    for(int i=0;i<8;++i) {r[i]=a[i]; r[8+i]=b[i];}
    return r;
}

// Optional cooperative TDM: each issuing wave owns disjoint rows of the tile.
// All issuing waves drain their own TENSORcnt before the workgroup barrier.
template<int Rows,int Waves,bool Distributed,int PadBytes=16,class T>
__device__ __forceinline__ void load_tile(T* lds,const T* global,int remaining,int stride) {
    if constexpr(Distributed) {
        static_assert(Rows%Waves==0);
        constexpr int Part=Rows/Waves;
        const int offset=(threadIdx.x/32)*Part;
        gfx1250::load_strided<Part,0,PadBytes>(lds+offset*(128+PadBytes/2),
            global+int64_t(offset)*stride,max(0,remaining-offset),stride);
    } else {
        gfx1250::load_strided<Rows,0,PadBytes>(lds,global,remaining,stride);
    }
}

// Swap the two 16-lane halves through VALU, without entering the LDS queue.
__device__ __forceinline__ float swap16(float x) {
    return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(
        0, __builtin_bit_cast(int, x), 0x76543210, 0xfedcba98, false, true));
}

template<class T,int QM,bool Causal,bool Pad,int Mode,int Waves=4,int NB=64,int SumStyle=0,bool FullQ=false,bool StoreLSE=true,int EpiMode=0,int StartupMode=0,int GridGroup=0,int RegPad=0>
__global__ __launch_bounds__(32*Waves) void bshd_kernel(Fav4Args a) {
    constexpr int BM=16*Waves*QM, BN=NB, Pitch=136, VP=144;

    __shared__ __align__(16) T ks[2][BN*Pitch],vs[2][BN*VP];
    const int heads=a.heads;
    auto tile_div=[&](unsigned x) {
        return x/gridDim.x;
    };
    const int lane=threadIdx.x%32,wave=threadIdx.x/32,tiles=gridDim.x;

    int linear=blockIdx.x+tiles*blockIdx.y,bh,tile;
    { if constexpr(Causal) {
        const int per_batch=tiles*heads,within=linear%per_batch;
        bh=linear/per_batch*heads+within%heads;
        tile=tiles-1-within/heads;
    } else {
        const int total=tiles*gridDim.y,xcd=linear%8;
        linear=xcd*(total/8)+min(xcd,total%8)+linear/8;
        {bh=linear/tiles; tile=linear%tiles;}
    } }
    if constexpr(GridGroup>0) {
        // Whole batch/head rows per XCD, Q tiles traversed in bounded row groups.
        // This is a bijection when each XCD owns a multiple of GridGroup rows.
        if(gridDim.y%(8*GridGroup)==0) {
            const unsigned physical=blockIdx.x+tiles*blockIdx.y;
            const unsigned slot=physical/8, depth=slot/GridGroup;
            const unsigned chunk=tile_div(depth);
            bh=(physical%8)*(gridDim.y/8)+chunk*GridGroup+slot%GridGroup;
            tile=Causal ? tiles-1-(depth-chunk*tiles) : depth-chunk*tiles;
        }
    }
    bh=__builtin_amdgcn_readfirstlane(bh); tile=__builtin_amdgcn_readfirstlane(tile);
    const int row0=tile*BM+wave*16+lane%16;
    const int batch=bh/heads,head=bh%heads,row_stride=heads*D;
    const int64_t qbase=int64_t(batch)*a.sq*row_stride+head*D;
    const int64_t kbase=int64_t(batch)*a.sk*row_stride+head*D;
    const auto* q=static_cast<const T*>(a.q)+qbase;
    const auto* k=static_cast<const T*>(a.k)+kbase;
    const auto* v=static_cast<const T*>(a.v)+kbase;
    auto* o=static_cast<T*>(a.o)+qbase;
    const int end=Causal ? min(a.sk,(tile+1)*BM) : a.sk;
     {
        if(StartupMode==1 || (wave==0)) {
            if constexpr(BM<=BN) {
                load_tile<BN,Waves,StartupMode==1>(ks[0],k,a.sk,row_stride);
                load_tile<BN,Waves,StartupMode==1,(VP-128)*2>(vs[0],v,a.sk,row_stride);
                load_tile<BM,Waves,StartupMode==1>(ks[1],q+int64_t(tile*BM)*row_stride,a.sq-tile*BM,row_stride);
            } else {
                static_assert(BM<=2*BN);
                load_tile<BM,Waves,StartupMode==1>(ks[0],q+int64_t(tile*BM)*row_stride,a.sq-tile*BM,row_stride);
            }
            gfx1250::wait();
        }
        __syncthreads();
    }
    V16<T> qr[QM][4];
    F8 out[QM][8]={};
    float maxima[QM],sums[QM]={};
#pragma unroll
    for(int qm=0;qm<QM;++qm) {
        maxima[qm]=-INFINITY;
#pragma unroll
        for(int t=0;t<4;++t)
            { qr[qm][t]=read_k32(ks[BM<=BN ? 1 : 0],Pitch,wave*16+lane%16+qm*16*Waves,t*32); }

    }
    if constexpr(BM>BN) {
        __syncthreads();
        if((wave==0)) { load_tile<BN,Waves,false>(ks[0],k,a.sk,row_stride); load_tile<BN,Waves,false,(VP-128)*2>(vs[0],v,a.sk,row_stride); }
    }
    const float scale2=a.scale*1.4426950408889634f;
    V16<T> kc[8];
    int stage=0;
    // Give matrix work priority while another resident wave can run softmax.
    if constexpr(Mode&(1<<26)) __builtin_amdgcn_s_setprio(1);
    for(int start=0;start<end;start+=BN,stage^=1) {
        if(!(Mode&4096) || start==0) {
            if((wave==0)) gfx1250::wait();
            __syncthreads();
        }
        if((wave==0) && start+BN<end)
            {
                { load_tile<BN,Waves,false>(ks[stage^1],k+int64_t(start+BN)*row_stride,a.sk-start-BN,row_stride); }
                { load_tile<BN,Waves,false,(VP-128)*2>(vs[stage^1],v+int64_t(start+BN)*row_stride,a.sk-start-BN,row_stride); }
            }


        V16<T> vr0[8];
        // Keep a small register range live across QK to tune clang's allocation.
        // Empty asm emits no arithmetic; the reserved values never enter attention.
        // This is compiler-sensitive and must be checked with same-device timing.
        using Padding = float __attribute__((ext_vector_type(RegPad ? RegPad : 1)));
        Padding reg_padding;
        if constexpr(RegPad) asm volatile("" : "=v"(reg_padding));
        F8 s[QM][BN/16]={};
        if constexpr(Mode&512) {
            // Keep the next N32 slice live while the current slice executes.
#pragma unroll
            for(int i=0;i<8;++i) {
                if(!(Mode&4096) || start==0) kc[i]=read_k32(ks[stage],Pitch,(i%2)*16+lane%16,(i/2)*32);
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for(int ns=0;ns<BN/32;++ns) {

                V16<T> kn[8];
                if(ns+1<BN/32) {
#pragma unroll
                    for(int i=0;i<8;++i) kn[i]=read_k32(ks[stage],Pitch,(ns+1)*32+(i%2)*16+lane%16,(i/2)*32);
                } else if constexpr((Mode&2048)) {
#pragma unroll
                    for(int d=0;d<8;++d) vr0[d]=gfx1250::read_v32<VP>(vs[stage],d*16,0);
                }

                {
#pragma unroll
                for(int i=0;i<8;++i) {
#pragma unroll
                    for(int qm=0;qm<QM;++qm) s[qm][ns*2+i%2]=mma(kc[i],qr[qm][i/2],s[qm][ns*2+i%2]);
                }
                }
                if(ns+1<BN/32 || ((Mode&2048)))
                    { if constexpr(Mode&16384) {
                        constexpr int chunk=(Mode&65536) ? 8 : 16;
#pragma unroll
                        for(int j=0;j<16/chunk;++j) {
                            __builtin_amdgcn_sched_group_barrier(0x100,chunk,0);
                            __builtin_amdgcn_sched_group_barrier(0x008,chunk/2*QM,0);
                        }
                    } else {
#pragma unroll
                    for(int i=0;i<8;++i) {
                        __builtin_amdgcn_sched_group_barrier(0x100,2,0);
                        __builtin_amdgcn_sched_group_barrier(0x008,QM,0);
                    }
                    } }

                __builtin_amdgcn_sched_barrier(0);

                if(ns+1<BN/32) {
#pragma unroll
                    for(int i=0;i<8;++i) kc[i]=kn[i];
                }
            }
        } else {
#pragma unroll
        for(int t=0;t<4;++t) {
#pragma unroll
            for(int n=0;n<BN/16;++n) {
                const auto kr=read_k32(ks[stage],Pitch,n*16+lane%16,t*32);
#pragma unroll
                for(int qm=0;qm<QM;++qm) s[qm][n]=mma(kr,qr[qm][t],s[qm][n]);
            }
        }
        }

        if constexpr((Mode&1) && !(Mode&2048)) {
#pragma unroll
            for(int d=0;d<8;++d) vr0[d]=gfx1250::read_v32<VP>(vs[stage],d*16,0);
            __builtin_amdgcn_sched_barrier(0);
        }

        if constexpr(RegPad) asm volatile("" : : "v"(reg_padding));
        if constexpr(Mode&(1<<26)) __builtin_amdgcn_s_setprio(0);
        V16<T> p[QM][BN/32];
#pragma unroll
        for(int qm=0;qm<QM;++qm) {
            const int row=row0+qm*16*Waves;
            if constexpr(Causal || Pad) {
                if((Pad && start+BN>a.sk) || (Causal && start+BN>tile*BM)) {
#pragma unroll
                    for(int n=0;n<BN/16;++n) {
#pragma unroll
                        for(int i=0;i<8;++i) {
                            const int col=start+n*16+lane/16*8+i;
                            if((Pad && col>=a.sk) || (Causal && col>row)) s[qm][n][i]=-INFINITY;
                        }
                    }
                }
            }
            if constexpr(Mode&536870912) {
                auto local_max = [&](const auto& logits) {
                float local[4]={-INFINITY,-INFINITY,-INFINITY,-INFINITY};
#pragma unroll
                for(int n=0;n<BN/16;++n) {
#pragma unroll
                    for(int i=0;i<8;++i) local[n%4]=fmaxf(local[n%4],logits[n][i]);
                }
                const float lm=fmaxf(fmaxf(local[0],local[1]),fmaxf(local[2],local[3]));
                    return fmaxf(lm,swap16(lm))*scale2;
                };
                auto exponentiate = [&](const auto& logits,float base) {
                    constexpr int Parts=2;
                    F2 partial[Parts]={};
#pragma unroll
                    for(int n=0;n<BN/16;++n) {
#pragma unroll
                        for(int i=0;i<8;i+=2) {
                            const F2 x=__builtin_elementwise_fma(
                                F2{logits[n][i],logits[n][i+1]},F2{scale2,scale2},F2{-base,-base});
                            const float e0=__builtin_amdgcn_exp2f(x[0]);
                            const F2 e={e0,__builtin_amdgcn_exp2f(x[1])};
                            p[qm][n/2][(n%2)*8+i]=T(e[0]);
                            p[qm][n/2][(n%2)*8+i+1]=T(e[1]);
                            if(SumStyle && n<Parts && i==0) partial[n%Parts]=e;
                            else partial[n%Parts]+=e;
                        }
                    }
                    F2 total=partial[0];
                    total+=partial[1];
                    return total[0]+total[1];
                };
                float base=maxima[qm];
                if(start==0) {base=local_max(s[qm]);maxima[qm]=base;}
                float sum=exponentiate(s[qm],base);
                float row_sum=(Mode&8) ? sum : sum+swap16(sum);

                // A nonnegative partial sum <= 2^8 proves each of its exp values <= 2^8.
                // Only the slow path needs the logit maximum. Overflow also enters it.
                if(start!=0 && __any(!(row_sum<=256.f))) {
                    auto correct = [&](const auto& logits) {
                        const float candidate=fmaxf(base,local_max(logits));
                        const bool update=candidate>base+8.f;
                        if(__any(update)) {
                            const float next=update ? candidate : base;
                            const float alpha=__builtin_amdgcn_exp2f(base-next);
#pragma unroll
                            for(int d=0;d<8;++d) out[qm][d]*=alpha;
                            sums[qm]*=alpha;
                            maxima[qm]=next;
                            sum=exponentiate(logits,next);
                            row_sum=(Mode&8) ? sum : sum+swap16(sum);
                        }
                    };
                    {
                        correct(s[qm]);
                    }
                }
                sums[qm]+=row_sum;
            } else {
            float lm;
            {
                float local[4]={-INFINITY,-INFINITY,-INFINITY,-INFINITY};
#pragma unroll
                for(int n=0;n<BN/16;++n) {
#pragma unroll
                    for(int i=0;i<8;++i) local[n%4]=fmaxf(local[n%4],s[qm][n][i]);
                }
                lm=fmaxf(fmaxf(local[0],local[1]),fmaxf(local[2],local[3]));
            }
            const float m=maxima[qm];
            const float candidate=fmaxf(m,fmaxf(lm,swap16(lm))*scale2);
            const bool update=start==0 || candidate>m+8.f;
            const float next=update ? candidate : m;
            float alpha=1.f;
            if constexpr(Mode&131072) {
                if(__any(update)) {
                    alpha=start==0 ? 0.f : __builtin_amdgcn_exp2f(m-next);
#pragma unroll
                    for(int d=0;d<8;++d) out[qm][d]*=alpha;
                    sums[qm]*=alpha;
                }
            } else {
                alpha=start==0 ? 0.f : __builtin_amdgcn_exp2f(m-next);
                if(__any(update)) {
#pragma unroll
                    for(int d=0;d<8;++d) out[qm][d]*=alpha;
                }
            }
            maxima[qm]=next;
            float sum;
            if constexpr(Mode&256) {
                F2 partial[4]={};
#pragma unroll
                for(int n=0;n<BN/16;++n) {
#pragma unroll
                    for(int i=0;i<8;i+=2) {
                        const F2 x=__builtin_elementwise_fma(
                            F2{s[qm][n][i],s[qm][n][i+1]},F2{scale2,scale2},F2{-next,-next});
                        float e0;
                        e0=__builtin_amdgcn_exp2f(x[0]);
                        const F2 e={e0,__builtin_amdgcn_exp2f(x[1])};
                        p[qm][n/2][(n%2)*8+i]=T(e[0]);
                        p[qm][n/2][(n%2)*8+i+1]=T(e[1]);
                        partial[n%4]+=e;
                    }
                }

                const F2 total=(partial[0]+partial[1])+(partial[2]+partial[3]);
                sum=total[0]+total[1];
            } else {
            float partial[4]={};
#pragma unroll
            for(int n=0;n<BN/16;++n) {
#pragma unroll
                for(int i=0;i<8;++i) {
                    const float x=fmaf(s[qm][n][i],scale2,-next);
                    float e;
                    e=__builtin_amdgcn_exp2f(x);
                    p[qm][n/2][(n%2)*8+i]=T(e);
                    partial[n%4]+=e;
                }
            }
            sum=(partial[0]+partial[1])+(partial[2]+partial[3]);
            }
            if constexpr(Mode&131072) sums[qm]+=sum+((Mode&8) ? 0.f : swap16(sum));
            else if constexpr(Mode&8) sums[qm]=sums[qm]*alpha+sum;
            else sums[qm]=sums[qm]*alpha+sum+swap16(sum);
            }

        }

        if constexpr(Mode&(1<<26)) __builtin_amdgcn_s_setprio(1);
        if constexpr(Mode&1024) {
            V16<T> vc[8];
#pragma unroll
            for(int d=0;d<8;++d)
                { vc[d]=vr0[d]; }

            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for(int t=0;t<BN/32;++t) {
                V16<T> vn[8];
                if(t+1<BN/32) {
#pragma unroll
                    for(int d=0;d<8;++d) vn[d]=gfx1250::read_v32<VP>(vs[stage],d*16,(t+1)*32);
                } else if constexpr(Mode&4096)
                    {
                    // All reads of the old V tile finish before its buffer is reused.
                    // Next K is ready here; its first slice overlaps this final PV stage.
                    {if((wave==0)) gfx1250::wait();}
                    __syncthreads();
#pragma unroll
                    for(int i=0;i<8;++i) kc[i]=read_k32(ks[stage^1],Pitch,(i%2)*16+lane%16,(i/2)*32);
                    }

                {
                if constexpr(Mode&(1<<25)) {
                    // Keep each query fragment's PV instructions in one scheduling
                    // region. The same P operand feeds all eight output fragments.
                    // This is a compiler scheduling boundary, not a workgroup sync.
#pragma unroll
                    for(int qm=0;qm<QM;++qm) {
#pragma unroll
                        for(int d=0;d<8;++d) out[qm][d]=mma(vc[d],p[qm][t],out[qm][d]);
                        __builtin_amdgcn_sched_barrier(0);
                    }
                } else {
#pragma unroll
                for(int d=0;d<8;++d) {
#pragma unroll
                    for(int qm=0;qm<QM;++qm) out[qm][d]=mma(vc[d],p[qm][t],out[qm][d]);
                }
                }
                }
                if(t+1<BN/32 || (Mode&4096)) {
                    if constexpr(Mode&16384) {
                        constexpr int chunk=(Mode&65536) ? 8 : 16;
#pragma unroll
                        for(int j=0;j<16/chunk;++j) {
                            __builtin_amdgcn_sched_group_barrier(0x100,chunk,1);
                            __builtin_amdgcn_sched_group_barrier(0x008,chunk/2*QM,1);
                        }
                    } else {
#pragma unroll
                    for(int d=0;d<8;++d) {
                        __builtin_amdgcn_sched_group_barrier(0x100,2,1);
                        __builtin_amdgcn_sched_group_barrier(0x008,QM,1);
                    }
                    }
                }
                __builtin_amdgcn_sched_barrier(0);

                if(t+1<BN/32) {
#pragma unroll
                    for(int d=0;d<8;++d) vc[d]=vn[d];
                }
            }
        } else {
#pragma unroll
        for(int t=0;t<BN/32;++t) {
#pragma unroll
            for(int d=0;d<8;++d) {
                V16<T> vr;
                { vr=t==0 ? vr0[d] : gfx1250::read_v32<VP>(vs[stage],d*16,t*32); }
#pragma unroll
                for(int qm=0;qm<QM;++qm) out[qm][d]=mma(vr,p[qm][t],out[qm][d]);
            }
        }
        }

    }

    if constexpr(Mode&(1<<26)) __builtin_amdgcn_s_setprio(0);
    if constexpr(EpiMode) {
        static_assert(BM<=BN);
        constexpr int OPitch=Pitch;
        // Reuse the now-dead K0 storage after all waves finish their LDS accesses.
        gfx1250::wait();
        __syncthreads();
#pragma unroll
        for(int qm=0;qm<QM;++qm) {
            if constexpr(Mode&8) sums[qm]+=swap16(sums[qm]);
            const float inv=1.f/sums[qm];
            const int local_row=wave*16+lane%16+qm*16*Waves;
#pragma unroll
            for(int d=0;d<8;++d) {
                V8<T> result;
#pragma unroll
                for(int i=0;i<8;++i) result[i]=T(out[qm][d][i]*inv);
                *reinterpret_cast<V8<T>*>(ks[0]+local_row*OPitch+d*16+lane/16*8)=result;
            }
        }
        {
            // The row redistribution stays within each wave's own 16-row fragments.
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("s_wait_dscnt 0" ::: "memory");
        }
          { if constexpr(FullQ) {
#pragma unroll
            for(int qm=0;qm<QM;++qm) {
                uint32_t offset[8],local[8];
#pragma unroll
                for(int t=0;t<8;++t) {
                    const int local_row=wave*16+qm*16*Waves+2*t+lane/16;
                    offset[t]=uint32_t(local_row*row_stride+(lane%16)*8)*2;
                    local[t]=uint32_t(reinterpret_cast<uintptr_t>(ks[0]+local_row*OPitch+(lane%16)*8));
                }
                gfx1250::store_async_x8(o+int64_t(tile*BM)*row_stride,offset,local);
            }
            __builtin_amdgcn_s_wait_asynccnt(0);
        } else {
            // Redistribute each row to 16 lanes, making each store cover two full rows.
            V8<T> result[QM][8];
#pragma unroll
            for(int qm=0;qm<QM;++qm) {
#pragma unroll
                for(int t=0;t<8;++t) {
                    const int local_row=wave*16+qm*16*Waves+2*t+lane/16;
                    result[qm][t]=*reinterpret_cast<const V8<T>*>(ks[0]+local_row*OPitch+(lane%16)*8);
                }
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for(int qm=0;qm<QM;++qm) {
#pragma unroll
                for(int t=0;t<8;++t) {
                    const int row=tile*BM+wave*16+qm*16*Waves+2*t+lane/16;
                    if(FullQ || row<a.sq)
                        *reinterpret_cast<V8<T>*>(o+int64_t(row)*row_stride+(lane%16)*8)=result[qm][t];
                }
            }
        } }
        if constexpr(StoreLSE) {
#pragma unroll
            for(int qm=0;qm<QM;++qm) {
                const int row=row0+qm*16*Waves;
                if((FullQ || row<a.sq) && a.lse && lane<16)
                    a.lse[int64_t(bh)*a.sq+row]=maxima[qm]*0.6931471805599453f+logf(sums[qm]);
            }
        }
    } else {
#pragma unroll
    for(int qm=0;qm<QM;++qm) {
        if constexpr(Mode&8) sums[qm]+=swap16(sums[qm]);
        const int row=row0+qm*16*Waves;
        if(FullQ || row<a.sq) {
            const float inv=1.f/sums[qm];
            {
#pragma unroll
                for(int d=0;d<8;++d) {
                    V8<T> result;
#pragma unroll
                    for(int i=0;i<8;++i) result[i]=T(out[qm][d][i]*inv);
                    *reinterpret_cast<V8<T>*>(o+int64_t(row)*row_stride+d*16+lane/16*8)=result;
                }
            }
            if constexpr(StoreLSE) {
                if(a.lse && lane<16) a.lse[int64_t(bh)*a.sq+row]=maxima[qm]*0.6931471805599453f+logf(sums[qm]);
            }
        }
    }
    }
}

template<class T,bool Causal,int QM,int Mode,int Waves=4,int NB=64,int SumStyle=0,int EpiMode=0,int StartupMode=0,int GridGroup=0,int RegPad=0>
void launch_bshd(const Fav4Args& a,hipStream_t stream) {
    dim3 grid((a.sq+QM*16*Waves-1)/(QM*16*Waves),a.batch*a.heads);
    auto dispatch=[&](auto full_q,auto store_lse) {
        if(a.sk%NB==0) bshd_kernel<T,QM,Causal,false,Mode,Waves,NB,SumStyle,decltype(full_q)::value,decltype(store_lse)::value,EpiMode,StartupMode,GridGroup,RegPad><<<grid,32*Waves,0,stream>>>(a);
        else bshd_kernel<T,QM,Causal,true,Mode,Waves,NB,SumStyle,decltype(full_q)::value,decltype(store_lse)::value,EpiMode,StartupMode,GridGroup,RegPad><<<grid,32*Waves,0,stream>>>(a);
    };
    if constexpr(EpiMode) {
        if(a.sq%(QM*16*Waves)==0) {
            if(a.lse) dispatch(std::true_type{},std::true_type{});
            else dispatch(std::true_type{},std::false_type{});
        } else {
            if(a.lse) dispatch(std::false_type{},std::true_type{});
            else dispatch(std::false_type{},std::false_type{});
        }
    } else dispatch(std::false_type{},std::true_type{});
}
