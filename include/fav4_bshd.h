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
// config 0: Auto (1 for sq<128, 132 for sq<=2048, 145 for longer Q).
// config 1: M64/N64; 2: M128/N128; 3: M128/N64 sharing K/V across two Q tiles.
// config 25: M128/N128, paired Q tiles, staged operands and conditional rescaling.
// config 132/133: M128/N128, guarded max, half-row denominators, cooperative
// startup and async LDS output stores. 132 groups batch/head rows by 16; 133
// retains the long-sequence grid order. 145 adds QK register-allocation spacing.
extern "C" int fav4_bshd_forward(const Fav4Args*,int causal,int config,hipStream_t);
extern "C" int fav4_bshd_select_config(const Fav4Args*);
