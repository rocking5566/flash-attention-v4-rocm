#pragma once
#include <hip/hip_runtime.h>

struct Fav4Args {
    const void *q, *k, *v;
    void *o;
    float *lse;
    int batch, heads, sq, sk;
    float scale;
};

// Native contiguous BF16 [batch, sequence, heads, 128] Q/K/V/O.
// LSE remains optional FP32 [batch, heads, sq]. Top-left causal, equal Q/K/V
// heads. Positive finite scale, finite FP32 dot products and scaled logits.
// Q/K/V/O must be device-resident, 16-byte aligned and non-overlapping.
// The stream belongs to the calling device; no allocation or host synchronization.
// config 0: Auto: 1 for sq<128; 155 for 128..512; 156 for 513..2048;
// 150 for sq>2048. B32/H8 square power-of-two S512..32768 use config 158.
// config 1: M64/N64; 2: M128/N128; 3: M128/N64 sharing K/V across two Q tiles.
// config 25: M128/N128, paired Q tiles, staged operands and conditional rescaling.
// config 132/133: M128/N128, guarded max, half-row denominators, cooperative
// startup and async LDS output stores. 132 groups batch/head rows by 16; 133
// retains the long-sequence grid order. 145 adds QK register-allocation spacing.
// config 150/155: group PV by query fragment and prioritize matrix phases.
// 150 extends 145; 155 extends the short-sequence grid order of 132.
// config 156: 155 with an initial base from the first valid score; first-tile
// overflow recovery remains active. 157 specializes this for B32/H8/Sq=Sk at
// S512 or S2048, and rejects other shapes. Both preserve the runtime scale.
// config 158: fixed B32/H8 square S512,1024,2048,4096,8192,16384,32768;
// dense loop unrolling and a separate final causal tile. Other shapes rejected.
// Priorities favor QK on measured shapes; causal S512 uses an unpeeled loop.
// Runtime scale is preserved.
extern "C" int fav4_bshd_forward(const Fav4Args*,int causal,int config,hipStream_t);
extern "C" int fav4_bshd_select_config(const Fav4Args*);
