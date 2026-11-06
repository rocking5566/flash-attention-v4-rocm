// Included inside a private kernel namespace.
using F8 = float __attribute__((ext_vector_type(8)));
template<class T> using V8 = T __attribute__((ext_vector_type(8)));
template<class T> using V16 = T __attribute__((ext_vector_type(16)));
constexpr int D = 128;

template<class T>
__device__ __forceinline__ F8 mma(V16<T> a, V16<T> b, F8 c) {
#if defined(__gfx1250__)
    if constexpr(std::is_same_v<T, __bf16>)
        return __builtin_amdgcn_wmma_f32_16x16x32_bf16(0,a,0,b,0,c,0,0);
    else
        return __builtin_amdgcn_wmma_f32_16x16x32_f16(0,a,0,b,0,c,0,0);
#else
    return c;
#endif
}

// FA4 degree-3 minimax coefficients (Tri Dao, BSD-3-Clause; see NOTICE).
// Explicitly map masked -infinity and values below the normal range to zero.
__device__ __forceinline__ float poly_exp2(float x) {
    const float z = fmaxf(x, -126.f);
    const float whole = floorf(z);
    const float f = z - whole;
    float p = fmaf(0.07711908966302872f, f, 0.22756439447402954f);
    p = fmaf(p, f, 0.6951461434364319f);
    p = fmaf(p, f, 1.f);
    const uint32_t bits = __float_as_uint(p) + (uint32_t(int(whole)) << 23);
    return x < -126.f ? 0.f : __uint_as_float(bits);
}

template<class T>
__device__ __forceinline__ V16<T> read_k32(const T* p, int pitch, int row, int k) {
    const int half = (threadIdx.x % 32) / 16;
    V16<T> r;
    const V8<T> a = *reinterpret_cast<const V8<T>*>(p + row*pitch + k + half*8);
    const V8<T> b = *reinterpret_cast<const V8<T>*>(p + row*pitch + k + 16 + half*8);
#pragma unroll
    for(int i=0;i<8;++i) {r[i]=a[i]; r[8+i]=b[i];}
    return r;
}

// Swap the two 16-lane halves through VALU, without entering the LDS queue.
__device__ __forceinline__ float swap16(float x) {
    return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(
        0, __builtin_bit_cast(int, x), 0x76543210, 0xfedcba98, false, true));
}


template<class T,int QM,bool Causal,bool Pad,int Mode,bool Lazy=true,int Waves=4,int NB=64>
__global__ __launch_bounds__(32*Waves) void bshd_kernel(Fav4Args a) {
    constexpr int BM=16*Waves*QM, BN=NB, Pitch=136, VP=(Mode&2) ? 144 : 136;
    __shared__ __align__(16) T ks[2][BN*Pitch],vs[2][BN*VP];
    const int lane=threadIdx.x%32,wave=threadIdx.x/32,tiles=gridDim.x;
    int linear=blockIdx.x+tiles*blockIdx.y,bh,tile;
    if constexpr(Causal) {
        const int per_batch=tiles*a.heads,within=linear%per_batch;
        bh=linear/per_batch*a.heads+within%a.heads;
        tile=tiles-1-within/a.heads;
    } else {
        const int total=tiles*gridDim.y,xcd=linear%8;
        linear=xcd*(total/8)+min(xcd,total%8)+linear/8;
        bh=linear/tiles; tile=linear%tiles;
    }
    bh=__builtin_amdgcn_readfirstlane(bh); tile=__builtin_amdgcn_readfirstlane(tile);
    const int row0=tile*BM+wave*16+lane%16;
    const int batch=bh/a.heads,head=bh%a.heads,row_stride=a.heads*D;
    const int64_t qbase=int64_t(batch)*a.sq*row_stride+head*D;
    const int64_t kbase=int64_t(batch)*a.sk*row_stride+head*D;
    const auto* q=static_cast<const T*>(a.q)+qbase;
    const auto* k=static_cast<const T*>(a.k)+kbase;
    const auto* v=static_cast<const T*>(a.v)+kbase;
    auto* o=static_cast<T*>(a.o)+qbase;
    const int end=Causal ? min(a.sk,(tile+1)*BM) : a.sk;
    if constexpr(Mode&64) {
        if(wave==0) {
            if constexpr(BM<=BN) {
                gfx1250::load_strided<BN>(ks[0],k,a.sk,row_stride);
                gfx1250::load_strided<BN,0,(VP-128)*2>(vs[0],v,a.sk,row_stride);
                gfx1250::load_strided<BM>(ks[1],q+int64_t(tile*BM)*row_stride,a.sq-tile*BM,row_stride);
            } else {
                static_assert(BM<=2*BN);
                gfx1250::load_strided<BM>(ks[0],q+int64_t(tile*BM)*row_stride,a.sq-tile*BM,row_stride);
            }
            gfx1250::wait();
        }
        __syncthreads();
    } else {
        if(wave==0) { gfx1250::load_strided<BN>(ks[0],k,a.sk,row_stride); gfx1250::load_strided<BN,0,(VP-128)*2>(vs[0],v,a.sk,row_stride); }
    }
    V16<T> qr[QM][4];
    F8 out[QM][8]={};
    float maxima[QM],sums[QM]={};
#pragma unroll
    for(int qm=0;qm<QM;++qm) {
        maxima[qm]=-INFINITY;
#pragma unroll
        for(int t=0;t<4;++t) {
            if constexpr(Mode&64) qr[qm][t]=read_k32(ks[BM<=BN ? 1 : 0],Pitch,wave*16+lane%16+qm*16*Waves,t*32);
            else qr[qm][t]=row0+qm*16*Waves<a.sq ? read_k32(q,row_stride,row0+qm*16*Waves,t*32) : V16<T>{};
        }
    }
    if constexpr((Mode&64) && BM>BN) {
        __syncthreads();
        if(wave==0) { gfx1250::load_strided<BN>(ks[0],k,a.sk,row_stride); gfx1250::load_strided<BN,0,(VP-128)*2>(vs[0],v,a.sk,row_stride); }
    }
    const float scale2=a.scale*1.4426950408889634f;
    int stage=0;
    for(int start=0;start<end;start+=BN,stage^=1) {
        if(wave==0) gfx1250::wait();
        __syncthreads();
        if(wave==0 && start+BN<end) {
            gfx1250::load_strided<BN>(ks[stage^1],k+int64_t(start+BN)*row_stride,a.sk-start-BN,row_stride);
            gfx1250::load_strided<BN,0,(VP-128)*2>(vs[stage^1],v+int64_t(start+BN)*row_stride,a.sk-start-BN,row_stride);
        }
        F8 s[QM][BN/16]={};
#pragma unroll
        for(int t=0;t<4;++t) {
#pragma unroll
            for(int n=0;n<BN/16;++n) {
                const auto kr=read_k32(ks[stage],Pitch,n*16+lane%16,t*32);
#pragma unroll
                for(int qm=0;qm<QM;++qm) s[qm][n]=mma(kr,qr[qm][t],s[qm][n]);
            }
        }
        V16<T> vr0[8];
        if constexpr(Mode&1) {
#pragma unroll
            for(int d=0;d<8;++d) vr0[d]=gfx1250::read_v32<VP>(vs[stage],d*16,0);
            __builtin_amdgcn_sched_barrier(0);
        }
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
            float local[4]={-INFINITY,-INFINITY,-INFINITY,-INFINITY};
#pragma unroll
            for(int n=0;n<BN/16;++n) {
#pragma unroll
                for(int i=0;i<8;++i) local[n%4]=fmaxf(local[n%4],s[qm][n][i]);
            }
            const float lm=fmaxf(fmaxf(local[0],local[1]),fmaxf(local[2],local[3]));
            const float m=maxima[qm];
            const float candidate=fmaxf(m,fmaxf(lm,swap16(lm))*scale2);
            const bool update=start==0 || !Lazy || candidate>m+8.f;
            const float next=update ? candidate : m;
            const float alpha=start==0 ? 0.f : __builtin_amdgcn_exp2f(m-next);
            if(!Lazy || __any(update)) {
#pragma unroll
                for(int d=0;d<8;++d) out[qm][d]*=alpha;
            }
            maxima[qm]=next;
            float partial[4]={};
#pragma unroll
            for(int n=0;n<BN/16;++n) {
#pragma unroll
                for(int i=0;i<8;++i) {
                    const float x=fmaf(s[qm][n][i],scale2,-next);
                    float e;
                    if constexpr(Mode&16) e=i%4==0 ? poly_exp2(x) : __builtin_amdgcn_exp2f(x);
                    else if constexpr(Mode&32) e=i%2==0 ? poly_exp2(x) : __builtin_amdgcn_exp2f(x);
                    else e=__builtin_amdgcn_exp2f(x);
                    p[qm][n/2][(n%2)*8+i]=T(e);
                    partial[n%4]+=e;
                }
            }
            const float sum=(partial[0]+partial[1])+(partial[2]+partial[3]);
            if constexpr(Mode&8) sums[qm]=sums[qm]*alpha+sum;
            else sums[qm]=sums[qm]*alpha+sum+swap16(sum);
        }
#pragma unroll
        for(int t=0;t<BN/32;++t) {
#pragma unroll
            for(int d=0;d<8;++d) {
                V16<T> vr;
                if constexpr(Mode&1) vr=t==0 ? vr0[d] : gfx1250::read_v32<VP>(vs[stage],d*16,t*32);
                else vr=gfx1250::read_v32<VP>(vs[stage],d*16,t*32);
#pragma unroll
                for(int qm=0;qm<QM;++qm) out[qm][d]=mma(vr,p[qm][t],out[qm][d]);
            }
        }
    }
#pragma unroll
    for(int qm=0;qm<QM;++qm) {
        if constexpr(Mode&8) sums[qm]+=swap16(sums[qm]);
        const int row=row0+qm*16*Waves;
        if(row<a.sq) {
            const float inv=1.f/sums[qm];
#pragma unroll
            for(int d=0;d<8;++d) {
                V8<T> result;
#pragma unroll
                for(int i=0;i<8;++i) result[i]=T(out[qm][d][i]*inv);
                *reinterpret_cast<V8<T>*>(o+int64_t(row)*row_stride+d*16+lane/16*8)=result;
            }
            if(a.lse && lane<16) a.lse[int64_t(bh)*a.sq+row]=maxima[qm]*0.6931471805599453f+logf(sums[qm]);
        }
    }
}

template<class T,bool Causal,int QM,int Mode,bool Lazy=true,int Waves=4,int NB=64>
void launch_bshd(const Fav4Args& a,hipStream_t stream) {
    dim3 grid((a.sq+QM*16*Waves-1)/(QM*16*Waves),a.batch*a.heads);
    if(a.sk%NB==0) bshd_kernel<T,QM,Causal,false,Mode,Lazy,Waves,NB><<<grid,32*Waves,0,stream>>>(a);
    else bshd_kernel<T,QM,Causal,true,Mode,Lazy,Waves,NB><<<grid,32*Waves,0,stream>>>(a);
}
