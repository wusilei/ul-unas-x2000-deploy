/**
 * nn_erb.h — ERB Band Merge / Split (BM/BS)
 * ==========================================
 * Equivalent Rectangular Bandwidth filterbank for frequency-domain feature compression.
 *
 * BM (Band Merge): 257 bins → 129 bins
 *   Low freq (0-2000Hz, bins 0-64): pass-through
 *   High freq (2031-8000Hz, bins 65-256): matrix multiply → bins 65-128
 *
 * BS (Band Split): 129 bins → 257 bins (inverse)
 *   Weights in u16f15, inputs/outputs in Q20
 *
 * Verified: UL-UNAS (linux_api9), GTCRN (denoise_v19_q15)
 * License: MIT
 */

#ifndef NN_ERB_H
#define NN_ERB_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * nn_bm — ERB Band Merge
 * x[W_in] in Q20, weight[(W_out - W_low) × W_high] in u16f15
 * y[W_out] in Q20
 */
static inline void nn_bm(const int32_t *x, const uint16_t *weight,
                          int W_in, int W_out, int32_t *y) {
    /* Low frequencies: direct pass-through */
    int W_low = (W_out > W_in) ? W_in : W_out;
    for (int i = 0; i < W_low && i < W_out; i++)
        y[i] = x[i];

    /* High frequencies: matrix merge (if W_in > W_out) */
    if (W_in > W_out) {
        int W_high_in = W_in - W_low;
        int W_high_out = W_out - W_low;
        for (int wo = 0; wo < W_high_out; wo++) {
            int64_t sum = 0;
            for (int wi = 0; wi < W_high_in; wi++) {
                sum += (int64_t)x[W_low + wi] * weight[wo * W_high_in + wi];
            }
            y[W_low + wo] = (int32_t)((sum + (NN_Q15_SCALE >> 1)) >> 15);  /* Q20 = Q20×Q15 >> 15 */
        }
    }
}

/**
 * nn_bs — ERB Band Split (inverse of BM)
 * x[W_in] in u16f15, weight[(W_out - W_low) × W_high_in] in u16f15
 * y[W_out] in s16f15
 */
static inline void nn_bs(const uint16_t *x, const uint16_t *weight,
                          int W_in, int W_out, int16_t *y) {
    int W_low = (W_in < W_out) ? W_in : W_out;
    /* Low frequencies: direct pass-through */
    for (int i = 0; i < W_low && i < W_out; i++)
        y[i] = (int16_t)x[i];

    /* High frequencies: matrix split */
    if (W_out > W_in) {
        int W_high_in = W_in - W_low;
        int W_high_out = W_out - W_low;
        for (int wo = 0; wo < W_high_out; wo++) {
            int64_t sum = 0;
            for (int wi = 0; wi < W_high_in; wi++) {
                sum += (int64_t)(int32_t)x[W_low + wi] * weight[wo * W_high_in + wi];
            }
            y[W_low + wo] = (int16_t)((sum + (NN_Q15_SCALE >> 1)) >> 15);
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif /* NN_ERB_H */
