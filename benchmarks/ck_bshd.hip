// The only translation unit that depends on CK. The candidate library never does.
#include "fav4_bshd.h"
#include "ck_tile/ops/fmha/block/variants.hpp"
#include "fmha_fwd.hpp"

namespace {
template<class Config, int BM, bool Causal, bool LSE, bool Pad>
int run(const Fav4Args& x, hipStream_t stream) {
    using namespace ck_tile;
    // qr_tdm's D128 long-prefill tile from the snapshotted code generator.
    using Shape=TileFmhaShape<sequence<BM,64,32,128,32,128>,
        sequence<4,1,1>,sequence<16,16,32>,sequence<4,1,1>,sequence<16,16,32>,
        true,true>;
    using Traits=TileFmhaTraits<Pad,Pad,false,false,false,
        BlockAttentionBiasEnum::NO_BIAS,false,LSE,false,
        BlockAttentionQuantScaleEnum::NO_SCALE,-1,false,false>;
    using Types=FmhaFwdTypeConfig<Config>;
    using Mask=std::conditional_t<Causal && !LSE,FmhaMasks::CausalMask,
        SimplifiedGenericAttentionMask<Causal>>;
    using Problem=BlockFmhaPipelineProblem<
        typename Types::QDataType,typename Types::KDataType,typename Types::VDataType,
        typename Types::SaccDataType,typename Types::SMPLComputeDataType,
        typename Types::BiasDataType,typename Types::RandValOutputDataType,
        typename Types::LSEDataType,typename Types::PDataType,
        typename Types::OaccDataType,typename Types::ODataType,
        Shape,false,ComposedAttention<0,CK_TILE_FMHA_FWD_FAST_EXP2>,Mask,false,Traits,true,true>;
    using Pipeline=BlockFmhaPipelineQRKSVSTdm<Problem>;
    using Epilogue=Default2DEpilogue<Default2DEpilogueProblem<
        typename Types::OaccDataType,typename Types::ODataType,Pad,false>>;
    using Kernel=FmhaFwdKernel<Pipeline,Epilogue>;
    fmha_fwd_args a{};
    a.q_ptr=x.q; a.k_ptr=x.k; a.v_ptr=x.v; a.o_ptr=x.o; a.lse_ptr=x.lse;
    a.batch=x.batch; a.nhead_q=x.heads; a.nhead_k=x.heads;
    a.seqlen_q=x.sq; a.seqlen_k=x.sk; a.max_seqlen_q=x.sq;
    a.hdim_q=128; a.hdim_v=128; a.scale_s=x.scale;
    a.stride_q=a.stride_k=a.stride_v=a.stride_o=x.heads*128;
    a.nhead_stride_q=a.nhead_stride_o=128;
    a.nhead_stride_k=a.nhead_stride_v=128;
    a.nhead_stride_lse=x.sq;
    a.batch_stride_q=a.batch_stride_o=x.heads*x.sq*128;
    a.batch_stride_k=a.batch_stride_v=x.heads*x.sk*128;
    a.batch_stride_lse=x.heads*x.sq;
    a.window_size_left=-1; a.window_size_right=Causal ? 0 : -1;
    a.mask_type=Causal ? 1 : 0;
    auto [args,grid]=fmha_fwd_create_kargs_and_grids<Kernel>(a);
    if(!Kernel::IsSupportedArgument(args)) return hipErrorInvalidValue;
    launch_kernel(stream_config{stream,false},
        make_kernel<Kernel::kBlockPerCu,gfx125_t>(Kernel{},grid,Kernel::BlockSize(),0,args));
    return hipGetLastError();
}
template<class T, int BM, bool Pad> int choose_mask(const Fav4Args& a,int causal,hipStream_t stream) {
    if(a.lse) return causal ? run<T,BM,true,true,Pad>(a,stream) : run<T,BM,false,true,Pad>(a,stream);
    return causal ? run<T,BM,true,false,Pad>(a,stream) : run<T,BM,false,false,Pad>(a,stream);
}
template<class T,int BM> int choose(const Fav4Args& a,int causal,hipStream_t stream) {
    return a.sq%BM==0 && a.sk%64==0 ? choose_mask<T,BM,false>(a,causal,stream)
                                    : choose_mask<T,BM,true>(a,causal,stream);
}
}
// BSHD D128 qr_tdm uses M64 below 128 Q rows, M128 otherwise. The BHSD-only
// square-causal override does not apply. Every PR benchmark row uses M128.
extern "C" int ck_bshd_forward(const Fav4Args* a,int causal,hipStream_t stream) {
    if(!a || (causal!=0 && causal!=1)) return hipErrorInvalidValue;
    return a->sq<128 ? choose<FmhaFwdBf16,64>(*a,causal,stream)
                     : choose<FmhaFwdBf16,128>(*a,causal,stream);
}
