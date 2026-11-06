#include "fav4_bshd.h"
#include "gfx1250.h"
#include <cmath>
#include <cstdint>
#include <climits>
#include <type_traits>

namespace fav4_bshd {
#include "bshd_kernel.hpp"

template<bool Causal>
int launch(const Fav4Args& a,int config,hipStream_t stream) {
    switch(config) {
    case 1: launch_bshd<__bf16,Causal,1,67,true,4,64>(a,stream); break;
    case 2: launch_bshd<__bf16,Causal,1,67,true,8,128>(a,stream); break;
    case 3: launch_bshd<__bf16,Causal,2,67,true,4,64>(a,stream); break;
    default: return hipErrorInvalidValue;
    }
    return hipGetLastError();
}
}

extern "C" int fav4_bshd_select_config(const Fav4Args* a) {
    return a->sq<128 ? 1 : 2;
}
extern "C" int fav4_bshd_forward(const Fav4Args* a,int causal,int config,hipStream_t stream) {
    if(!a || !a->q || !a->k || !a->v || !a->o || a->batch<=0 || a->heads<=0 ||
       a->sq<=0 || a->sk<=0 || (causal!=0 && causal!=1) || config<0 || config>3 ||
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
