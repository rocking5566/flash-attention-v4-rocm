#include "fav4_bshd.h"
#include "gfx1250.h"
#include <cmath>
#include <cstdint>
#include <climits>
#include <type_traits>

namespace fav4_bshd {
#include "bshd_kernel.hpp"

// These two measured shapes can fold grid division and BSHD strides at compile time.
// Keep scale as a runtime argument and reject shape mismatches before dispatch.
template<bool Causal,int Sequence>
void launch_short_static(const Fav4Args& a,hipStream_t stream) {
    auto dispatch=[&](auto store_lse) {
        bshd_kernel<__bf16,2,Causal,false,1721917259,4,128,1,true,
            decltype(store_lse)::value,9,1,16,0,Sequence>
            <<<dim3(Sequence/128,256),128,0,stream>>>(a);
    };
    if(a.lse) dispatch(std::true_type{});
    else dispatch(std::false_type{});
}

// Fixed B32/H8 square prefill. Dense unrolls four K tiles; causal keeps the
// diagonal separate and unrolls two complete tiles (S512 keeps its single-tile loop; S32768 unrolls four).
// Scale remains runtime. Short causal shapes retain their measured matrix priorities.
template<bool Causal,int Sequence>
void launch_fixed(const Fav4Args& a,hipStream_t stream) {
    auto dispatch=[&](auto store_lse) {
        constexpr int Mode=Sequence<=2048 ? 1721917259 : 1719820107;
        constexpr int Group=Sequence<=2048 ? 16 : 0;
        constexpr int Padding=Causal || Sequence<=2048 ? 0 : 4;
        constexpr int Unroll=Causal ? (Sequence==512 ? 1 : (Sequence==32768 ? 4 : 2)) : 4;
        constexpr int QKPriority=Causal && Sequence==512 ? 2 : 1;
        constexpr int PVPriority=Sequence==512 || (Causal && Sequence<=4096) ? 1 : 0;
        // S2048 dense benefits from omitting the redundant softmax-to-PV priority0 hint.
        constexpr bool SetPVPriority=Causal || Sequence!=2048;
        bshd_kernel<__bf16,2,Causal,false,Mode,4,128,1,true,
            decltype(store_lse)::value,9,1,Group,Padding,Sequence,
            Unroll,Causal && Sequence!=512,QKPriority,PVPriority,SetPVPriority>
            <<<dim3(Sequence/128,256),128,0,stream>>>(a);
    };
    if(a.lse) dispatch(std::true_type{});
    else dispatch(std::false_type{});
}

template<bool Causal>
int launch(const Fav4Args& a,int config,hipStream_t stream) {
    switch(config) {
    case 1: launch_bshd<__bf16,Causal,1,67,4,64>(a,stream); break;
    case 2: launch_bshd<__bf16,Causal,1,67,8,128>(a,stream); break;
    case 3: launch_bshd<__bf16,Causal,2,67,4,64>(a,stream); break;
    case 25: launch_bshd<__bf16,Causal,2,155459,4,128>(a,stream); break;
    case 132: launch_bshd<__bf16,Causal,2,1610768203,4,128,1,9,1,16>(a,stream); break;
    case 133: launch_bshd<__bf16,Causal,2,1610768203,4,128,1,9,1>(a,stream); break;
    case 145: launch_bshd<__bf16,Causal,2,1610768203,4,128,1,9,1,0,4>(a,stream); break;
    case 150: launch_bshd<__bf16,Causal,2,1719820107,4,128,1,9,1,0,4>(a,stream); break;
    case 155: launch_bshd<__bf16,Causal,2,1719820107,4,128,1,9,1,16,0>(a,stream); break;
    case 156: launch_bshd<__bf16,Causal,2,1721917259,4,128,1,9,1,16,0>(a,stream); break;
    case 157:
        if(a.batch!=32 || a.heads!=8 || a.sq!=a.sk) return hipErrorInvalidValue;
        if(a.sq==512) launch_short_static<Causal,512>(a,stream);
        else if(a.sq==2048) launch_short_static<Causal,2048>(a,stream);
        else return hipErrorInvalidValue;
        break;
    case 158:
        if(a.batch!=32 || a.heads!=8 || a.sq!=a.sk) return hipErrorInvalidValue;
        switch(a.sq) {
        case 512: launch_fixed<Causal,512>(a,stream); break;
        case 1024: launch_fixed<Causal,1024>(a,stream); break;
        case 2048: launch_fixed<Causal,2048>(a,stream); break;
        case 4096: launch_fixed<Causal,4096>(a,stream); break;
        case 8192: launch_fixed<Causal,8192>(a,stream); break;
        case 16384: launch_fixed<Causal,16384>(a,stream); break;
        case 32768: launch_fixed<Causal,32768>(a,stream); break;
        default: return hipErrorInvalidValue;
        }
        break;
    default: return hipErrorInvalidValue;
    }
    return hipGetLastError();
}
}

extern "C" int fav4_bshd_select_config(const Fav4Args* a) {
    if(!a || a->sq<=0) return 0;
    if(a->sq<128) return 1;
    if(a->batch==32 && a->heads==8 && a->sq==a->sk &&
       a->sq>=512 && a->sq<=32768 && (a->sq&(a->sq-1))==0)
        return 158;
    if(a->sq<=512) return 155;
    if(a->sq<=2048) return 156;
    return 150;
}
extern "C" int fav4_bshd_forward(const Fav4Args* a,int causal,int config,hipStream_t stream) {
    if(!a || !a->q || !a->k || !a->v || !a->o || a->batch<=0 || a->heads<=0 ||
       a->sq<=0 || a->sk<=0 || (causal!=0 && causal!=1) || config<0 || config>158 ||
       !std::isfinite(a->scale) || a->scale<=0) return hipErrorInvalidValue;
    if(a->batch>INT_MAX/a->heads || a->heads>INT_MAX/128 ||
       a->sq>INT_MAX/(a->heads*128) || a->sk>INT_MAX/(a->heads*128) ||
       ((reinterpret_cast<uintptr_t>(a->q)|reinterpret_cast<uintptr_t>(a->k)|
         reinterpret_cast<uintptr_t>(a->v)|reinterpret_cast<uintptr_t>(a->o))&15))
        return hipErrorInvalidValue;
    if(config==0) config=fav4_bshd_select_config(a);
    return causal ? fav4_bshd::launch<true>(*a,config,stream)
                  : fav4_bshd::launch<false>(*a,config,stream);
}
