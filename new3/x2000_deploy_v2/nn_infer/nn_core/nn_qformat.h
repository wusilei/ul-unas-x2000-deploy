/**
 * nn_qformat.h — Unified Fixed-Point Q-Format System
 * ===================================================
 * Verified across GTCRN and UL-UNAS MATLAB→C conversion projects.
 * All constants can be overridden by defining them before #include.
 *
 * Q-format convention:
 *   s32f20 = int32_t  with 20 fractional bits (×1,048,576)
 *   s16f15 = int16_t  with 15 fractional bits (×32,768)
 *   u16f15 = uint16_t with 15 fractional bits (×32,768)
 *   s16f14 = int16_t  with 14 fractional bits (×16,384)
 *   s16f13 = int16_t  with 13 fractional bits (×8,192)
 *   s16f12 = int16_t  with 12 fractional bits (×4,096)
 *   s16f10 = int16_t  with 10 fractional bits (×1,024)
 *
 * Data layout (row-major, consistent with C):
 *   2D (C,W):  element[c][w] = data[c*W + w]
 *   3D (C,H,W): element[c][h][w] = data[c*H*W + h*W + w]
 *   4D weights (Cout,Cin,Kh,Kw):
 *     weight[cout][cin][kh][kw] = weight[cout*Cin*Kh*Kw + cin*Kh*Kw + kh*Kw + kw]
 *
 * Target: Ingenic X2000 MIPS32R2 (no FPU) + PC gcc/clang verification
 * License: MIT
 */

#ifndef NN_QFORMAT_H
#define NN_QFORMAT_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Default Q-format Constants (override via #define before #include)
 * ================================================================ */

/* Activation Q-formats */
#ifndef NN_Q_ACT
#define NN_Q_ACT         20    /* int32_t,  main activations (×1,048,576) */
#endif
#ifndef NN_Q_GRU_H
#define NN_Q_GRU_H       15    /* int16_t,  GRU hidden state / tanh output (×32,768) */
#endif
#ifndef NN_Q_SIG
#define NN_Q_SIG         15    /* uint16_t, sigmoid output (×32,768) */
#endif
#ifndef NN_Q_LOG
#define NN_Q_LOG         20    /* uint32_t, log10 output (×1,048,576) */
#endif
#ifndef NN_Q_LN_VAR
#define NN_Q_LN_VAR      11    /* uint16_t, LN 1/sqrt(var) (×2,048) */
#endif

/* Weight Q-formats (float→int quantization shifts) */
#ifndef NN_Q_W_CONV
#define NN_Q_W_CONV      14    /* s16f14: Conv/DeConv weights */
#endif
#ifndef NN_Q_W_PCONV
#define NN_Q_W_PCONV     13    /* s16f13: PConv/GCONV/TConv weights */
#endif
#ifndef NN_Q_W_GRU
#define NN_Q_W_GRU       12    /* s16f12: GRU ih/hh weights */
#endif
#ifndef NN_Q_W_GRU_B
#define NN_Q_W_GRU_B     10    /* s16f10: GRU bias */
#endif
#ifndef NN_Q_W_FC
#define NN_Q_W_FC        13    /* s16f13: FC weights */
#endif
#ifndef NN_Q_W_FC_B
#define NN_Q_W_FC_B      20    /* s32f20: FC bias */
#endif
#ifndef NN_Q_W_BN_W
#define NN_Q_W_BN_W      14    /* s16f14/u16f14: BN weight */
#endif
#ifndef NN_Q_W_BN_B
#define NN_Q_W_BN_B      20    /* s32f20: BN bias */
#endif
#ifndef NN_Q_W_BN_M
#define NN_Q_W_BN_M      20    /* s32f20: BN running_mean */
#endif
#ifndef NN_Q_W_BN_V
#define NN_Q_W_BN_V      14    /* u16f14: BN running_var */
#endif
#ifndef NN_Q_W_LN
#define NN_Q_W_LN        12    /* s16f12: LN weight/bias */
#endif
#ifndef NN_Q_W_AP
#define NN_Q_W_AP        14    /* s16f14: AffinePReLU weight */
#endif
#ifndef NN_Q_W_AP_S
#define NN_Q_W_AP_S      13    /* s16f13: AffinePReLU slope */
#endif
#ifndef NN_Q_W_ERB
#define NN_Q_W_ERB       15    /* u16f15: ERB merge/split weights */
#endif

/* ================================================================
 * Scale Constants
 * ================================================================ */
#define NN_Q20_SCALE  1048576     /* 2^20 */
#define NN_Q18_SCALE  262144      /* 2^18 */
#define NN_Q15_SCALE  32768       /* 2^15 */
#define NN_Q15_ONE    32768       /* 1.0 in u16f15 */
#define NN_Q14_SCALE  16384       /* 2^14 */
#define NN_Q13_SCALE  8192        /* 2^13 */
#define NN_Q12_SCALE  4096        /* 2^12 */
#define NN_Q11_SCALE  2048        /* 2^11 */
#define NN_Q10_SCALE  1024        /* 2^10 */

/* ================================================================
 * Rounding & Saturation Primitives
 * ================================================================ */

/** Round(x / 2^N) with half-up rounding (matching MATLAB round()). Uses division for portability. */
static inline int32_t nn_round_shr(int64_t x, int N) {
    int64_t denom = (int64_t)1 << N;
    int64_t half = denom >> 1;
    if (x >= 0) return (int32_t)((x + half) / denom);
    else        return (int32_t)((x - half) / denom);
}

/** Saturate int64_t to int32_t */
static inline int32_t nn_sat_i32(int64_t x) {
    if (x > INT32_MAX) return INT32_MAX;
    if (x < INT32_MIN) return INT32_MIN;
    return (int32_t)x;
}

/** Saturate int32_t to int16_t */
static inline int16_t nn_sat_i16(int32_t x) {
    if (x > INT16_MAX) return INT16_MAX;
    if (x < INT16_MIN) return INT16_MIN;
    return (int16_t)x;
}

/** Saturate int32_t to uint16_t */
static inline uint16_t nn_sat_u16(int32_t x) {
    if (x > UINT16_MAX) return UINT16_MAX;
    if (x < 0) return 0;
    return (uint16_t)x;
}

/** Clamp int32_t to int16_t range (preserves sign, saturates) */
static inline int16_t nn_clamp_i16(int32_t x) {
    if (x > 32767)  return 32767;
    if (x < -32768) return -32768;
    return (int16_t)x;
}

/** Clamp int32_t to uint16_t range */
static inline uint16_t nn_clamp_u16(int32_t x) {
    if (x > 65535) return 65535;
    if (x < 0)     return 0;
    return (uint16_t)x;
}

/** Round float→Q20 with saturation (matching MATLAB Fix_point.m) */
static inline int32_t nn_f2q20(float x) {
    float v = x * 1048576.0f;
    if (v > 2147483647.0f) return INT32_MAX;
    if (v < -2147483648.0f) return INT32_MIN;
    return (int32_t)(v >= 0 ? (v + 0.5f) : (v - 0.5f));
}

/** Round float→Q15 (int16_t, safely clamped) */
static inline int16_t nn_f2q15(float x) {
    float r = x * 32768.0f;
    if (r > 32767.0f) return 32767;
    if (r < -32768.0f) return -32768;
    return (int16_t)(r >= 0 ? (r + 0.5f) : (r - 0.5f));
}

/** Dequant: Q20 → float */
static inline float nn_q20_to_f(int32_t x) {
    return (float)x / 1048576.0f;
}

/** Dequant: Q15 → float */
static inline float nn_q15_to_f(int16_t x) {
    return (float)x / 32768.0f;
}

/* ================================================================
 * Tensor Indexing Macros (row-major)
 * ================================================================ */

/** 2D index: (C, W) → linear */
#define NN_IDX_2D(c, w, W)       ((c) * (W) + (w))

/** 3D index: (C, H, W) → linear */
#define NN_IDX_3D(c, h, w, H, W) ((c) * (H) * (W) + (h) * (W) + (w))

/** 4D weight index: (Cout, Cin, Kh, Kw) → linear */
#define NN_IDX_W4D(co, ci, kh, kw, Cin, Kh, Kw) \
    ((co) * (Cin) * (Kh) * (Kw) + (ci) * (Kh) * (Kw) + (kh) * (Kw) + (kw))

#ifdef __cplusplus
}
#endif

#endif /* NN_QFORMAT_H */
