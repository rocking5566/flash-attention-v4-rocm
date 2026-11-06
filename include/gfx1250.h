#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <type_traits>

namespace gfx1250 {
template<class T> using V8 = T __attribute__((ext_vector_type(8)));
template<class T> using V16 = T __attribute__((ext_vector_type(16)));
using U4 = uint32_t __attribute__((ext_vector_type(4)));
using I4 = int32_t __attribute__((ext_vector_type(4)));
using I8 = int32_t __attribute__((ext_vector_type(8)));

// TDM descriptor layout adapted from AMD Composable Kernel (MIT; see NOTICE).
// A [sequence, 128] slice of native BSHD memory. Source row_stride is in
// halfword elements; the destination remains densely packed with LDS padding.
template<int Rows, int CacheHint=0, int PadBytes=16, class T>
__device__ __forceinline__ void load_strided(T* lds, const T* global,
                                            int remaining, uint32_t row_stride) {
#if defined(__gfx1250__)
    static_assert(sizeof(T)==2);
    static_assert(PadBytes==16 || PadBytes==32);
    const uintptr_t addr=reinterpret_cast<uintptr_t>(global);
    const uint32_t local=uint32_t(reinterpret_cast<uintptr_t>(lds));
    U4 g0={1,local,uint32_t(addr),uint32_t(addr>>32)|(2u<<30)};
    I8 g1={int((1u<<16)|(1u<<20)|(5u<<22)|((PadBytes/4-1u)<<25)),
           128<<16, int(uint32_t(remaining)<<16), (128<<16)|(remaining>>16),
           Rows, int(row_stride), 0, 0};
#pragma unroll
    for(int i=0;i<4;++i) g0[i]=__builtin_amdgcn_readfirstlane(g0[i]);
#pragma unroll
    for(int i=0;i<8;++i) g1[i]=__builtin_amdgcn_readfirstlane(g1[i]);
    __builtin_amdgcn_tensor_load_to_lds(g0,g1,I4{},I4{},I8{},CacheHint);
#endif
}

__device__ __forceinline__ void wait() {
#if defined(__gfx1250__)
    __builtin_amdgcn_s_wait_tensorcnt(0);
#endif
}

// ds_load_tr16 consumes an 8x16 patch per half-wave. Input lane supplies a
// contiguous vector at (lane%8, lane/8*8); output lane receives a column of 8.
template<class T>
__device__ __forceinline__ V8<T> transpose8(const T* lds) {
#if defined(__gfx1250__)
    if constexpr(std::is_same_v<T,__bf16>) {
        using Native=__bf16 __attribute__((vector_size(16)));
        using Pointer=__attribute__((address_space(3))) Native*;
        auto p=(Pointer)(const_cast<T*>(lds));
        return __builtin_bit_cast(V8<T>,__builtin_amdgcn_ds_load_tr16_b128_v8bf16(p));
    } else {
        using Native=__fp16 __attribute__((vector_size(16)));
        using Pointer=__attribute__((address_space(3))) Native*;
        auto p=(Pointer)(const_cast<T*>(lds));
        return __builtin_bit_cast(V8<T>,__builtin_amdgcn_ds_load_tr16_b128_v8f16(p));
    }
#else
    return {};
#endif
}

template<int Pitch=136,class T>
__device__ __forceinline__ V16<T> read_v32(const T* lds,int d,int k) {
    const int lane=threadIdx.x%32;
    const int row=k+lane/16*8+lane%8;
    const int col=d+(lane%16)/8*8;
    V8<T> a=transpose8(lds+row*Pitch+col);
    V8<T> b=transpose8(lds+(row+16)*Pitch+col);
    V16<T> result;
#pragma unroll
    for(int i=0;i<8;++i) { result[i]=a[i]; result[8+i]=b[i]; }
    return result;
}
}
