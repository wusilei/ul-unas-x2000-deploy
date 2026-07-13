/**
 * nn_act.h — Activation Functions (Fixed-Point)
 * =============================================
 * PReLU (GTCRN):  negative × slope, positive passthrough
 * AffinePReLU (ULUNAS): PReLU + affine transform + residual
 *
 * Both operate in Q20 in-place.
 *
 * Verified: GTCRN + ULUNAS, X2000 MIPS32R2
 * License: MIT
 */

#ifndef NN_ACT_H
#define NN_ACT_H

#include "nn_qformat.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * nn_prelu — Parametric ReLU (GTCRN)
 * y = x > 0 ? x : round(x * slope * 2^qr)
 * x[C*W] in Q20, modified in-place
 * slope[C] in s16f14
 */
static inline void nn_prelu(int32_t *x, int C, int W,
                             const int16_t *slope, int qr) {
    for (int c = 0; c < C; c++) {
        int16_t s = slope[c];
        for (int w = 0; w < W; w++) {
            int32_t v = x[c * W + w];
            if (v < 0) {
                int64_t neg = (int64_t)v * s;
                x[c * W + w] = nn_sat_i32(nn_round_shr(neg, -qr));
            }
        }
    }
}

/**
 * nn_affine_prelu — AffinePReLU (ULUNAS)
 * 1. PReLU: negative part × slope[channel] >> qr1
 * 2. Affine: x_copy × weight[channel] >> qr2 + bias[channel]
 * 3. Residual: y = affine_part + neg_processed_part
 *
 * x[C][W] in Q20 → y[C][W] in Q20
 * weight[C] in s16f14, bias[C] in s32f20, slope[C] in s16f13
 */
static inline void nn_affine_prelu(const int32_t *x,
                                    const int16_t *weight, const int32_t *bias,
                                    const int16_t *slope, int qr1, int qr2,
                                    int C, int W, int32_t *y) {
    for (int c = 0; c < C; c++) {
        int16_t s = slope[c];
        int32_t bw = weight[c];
        int32_t bb = bias ? bias[c] : 0;
        for (int w = 0; w < W; w++) {
            int32_t xv = x[c * W + w];
            /* PReLU */
            int32_t x_neg = 0;
            if (xv < 0) {
                int64_t v = (int64_t)xv * s;
                x_neg = nn_round_shr(v, -qr1);
            }
            /* Affine */
            int64_t affine = (int64_t)xv * bw;
            int32_t affine_v = nn_round_shr(affine, -qr2);
            /* Residual: affine + neg_processed + bias */
            y[c * W + w] = nn_sat_i32((int64_t)affine_v + x_neg + bb);
        }
    }
}

/**
 * nn_sigmoid_apply — Apply sigmoid element-wise to Q20 input → Q15 output
 */
static inline void nn_sigmoid_apply(const int32_t *x, int N, uint16_t *y) {
    for (int i = 0; i < N; i++)
        y[i] = nn_sigmoid_q15(x[i]);  /* from nn_lut.h */
}

/**
 * nn_tanh_apply — Apply tanh element-wise to Q20 input → Q15 output
 */
static inline void nn_tanh_apply(const int32_t *x, int N, int16_t *y) {
    for (int i = 0; i < N; i++)
        y[i] = nn_tanh_q15(x[i]);  /* from nn_lut.h */
}

#ifdef __cplusplus
}
#endif

#endif /* NN_ACT_H */
