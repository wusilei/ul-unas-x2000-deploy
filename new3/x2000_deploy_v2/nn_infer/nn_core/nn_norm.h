/**
 * nn_norm.h — Normalization Operators (Fixed-Point)
 * ==================================================
 * BatchNorm (signed + unsigned weight) — GTCRN + ULUNAS
 * LayerNorm — GTCRN + ULUNAS
 *
 * BN formula:
 *   x_norm = round((x - mean) * var * 2^qr1)
 *   y = round(x_norm * weight * 2^qr2) + bias
 *
 * Verified Qr combinations:
 *   Encoder Conv BN:     qr1=-14, qr2=13
 *   Encoder PConv BN:    qr1=-11, qr2=14
 *   Decoder DeConv BN:   qr1=-14, qr2=13
 *   Decoder PConv BN:    qr1=-13, qr2=14
 *   GDPRNN FC BN:        qr1=-13, qr2=20
 *
 * ⚠️ BN running_var Q-format varies per block (Q10-Q14).
 *    Must match the MATLAB Fix_point.m export.
 *
 * ⚠️ LN channel index: data[T][C] row-major, channel = i % C (NOT i / T!)
 *    Bug source: ULUNAS Bug #10.
 *
 * Verified: GTCRN + ULUNAS, X2000 MIPS32R2
 * License: MIT
 */

#ifndef NN_NORM_H
#define NN_NORM_H

#include "nn_qformat.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * nn_bn_sw — BatchNorm with signed weight (ULUNAS)
 * x[C*W] in Q20, weight[C] in s16f14, bias[C] in Q20
 * running_mean[C] in Q20, running_var[C] in u16 (Q varies)
 */
static inline void nn_bn_sw(const int32_t *x, const int16_t *weight,
                             const int32_t *bias,
                             const int32_t *running_mean,
                             const uint16_t *running_var,
                             int qr1, int qr2, int C, int N,
                             int32_t *y) {
    for (int c = 0; c < C; c++) {
        int32_t w = weight[c];
        int32_t b = bias ? bias[c] : 0;
        int32_t mean = running_mean[c];
        int32_t var = running_var[c];
        for (int i = 0; i < N; i++) {
            int32_t xv = x[c * N + i];
            int64_t xn = (int64_t)(xv - mean) * var;
            xn = nn_round_shr(xn, -qr1);
            int64_t v = xn * w;
            v = nn_round_shr(v, -qr2);
            y[c * N + i] = nn_sat_i32(v + b);
        }
    }
}

/**
 * nn_bn_uw — BatchNorm with unsigned weight (GTCRN + ULUNAS)
 * weight[C] in u16f14
 */
static inline void nn_bn_uw(const int32_t *x, const uint16_t *weight,
                             const int32_t *bias,
                             const int32_t *running_mean,
                             const uint16_t *running_var,
                             int qr1, int qr2, int C, int N,
                             int32_t *y) {
    for (int c = 0; c < C; c++) {
        uint16_t w = weight[c];
        int32_t b = bias ? bias[c] : 0;
        int32_t mean = running_mean[c];
        int32_t var = running_var[c];
        for (int i = 0; i < N; i++) {
            int32_t xv = x[c * N + i];
            int64_t xn = (int64_t)(xv - mean) * var;
            xn = nn_round_shr(xn, -qr1);
            int64_t v = xn * (int64_t)(uint64_t)w;
            v = nn_round_shr(v, -qr2);
            y[c * N + i] = nn_sat_i32(v + b);
        }
    }
}

/**
 * nn_ln — LayerNorm (GTCRN + ULUNAS)
 * Computes mean/var online per LN call.
 * x[T*C] row-major (T time steps, C channels)
 * weight[C] in s16f12, bias[C] in s32f20
 *
 * ⚠️ CRITICAL: channel index = i % C, NOT i / T
 *   Data layout: [T=33][C=16] row-major
 *   x[0]=t0ch0, x[1]=t0ch1, ..., x[15]=t0ch15, x[16]=t1ch0, ...
 *   LN normalizes across time per channel → group by (i % C)
 */
static inline void nn_ln(const int32_t *x, const int16_t *weight,
                          const int32_t *bias, int qr, int C, int N,
                          int32_t *y) {
    int T = N / C;
    /* Per-channel mean */
    for (int c = 0; c < C; c++) {
        int64_t sum = 0;
        for (int t = 0; t < T; t++)
            sum += x[t * C + c];               /* Correct: channel c across all time steps */
        int32_t mean = (int32_t)(sum / T);
        /* Per-channel variance */
        int64_t sq_sum = 0;
        for (int t = 0; t < T; t++) {
            int32_t d = x[t * C + c] - mean;
            sq_sum += (int64_t)d * d;
        }
        int32_t var = (int32_t)(sq_sum / T);
        /* 1/sqrt(var) approximation using integer sqrt + Q20 div */
        if (var <= 0) var = 1;
        uint32_t inv_std = nn_sqrt_q40_to_q20((uint64_t)(1ULL << 40) / (uint64_t)var);
        /* Normalize */
        int32_t w = weight[c];
        int32_t b = bias ? bias[c] : 0;
        for (int t = 0; t < T; t++) {
            int32_t xv = x[t * C + c];
            int64_t xn = (int64_t)(xv - mean) * inv_std;
            xn = nn_round_shr(xn, NN_Q_LN_VAR);  /* inv_std is Q20, shift by LN_VAR */
            int64_t v = xn * w;
            v = nn_round_shr(v, -qr);
            y[t * C + c] = nn_sat_i32(v + b);
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif /* NN_NORM_H */
