/**
 * nn_wola.h — WOLA (Weighted Overlap-Add) Synthesis
 * ===================================================
 * Pre-computes normalization inverse table for arbitrary WIN_INC.
 * Supports both 2-window and 3-window overlap modes.
 *
 * Usage:
 *   1. nn_wola_compute_inv_table(win_q15, win_len, win_inc, inv_table)
 *   2. For each frame output:
 *      nn_wola_output(ola_buf, &ola_pos, inv_table, win_inc, n, gain, out)
 *
 * Verified: UL-UNAS v6 (WIN_INC=200 with 3/2-window hybrid)
 * License: MIT
 */

#ifndef NN_WOLA_H
#define NN_WOLA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * nn_wola_compute_inv_table — Pre-compute WOLA normalization reciprocals
 *
 * win_q15[win_len]: analysis/synthesis window in Q15
 * wola_inv_q30[win_inc]: output table, each entry = round(2^30 / win²_sum[i])
 *
 * win²_sum[i] = Σ win²[i + k*win_inc] for all k where i + k*win_inc < win_len
 * For WIN_INC=200, WIN_LEN=512: positions 0-111 have 3 windows, 112-199 have 2.
 */
static inline void nn_wola_compute_inv_table(const int16_t *win_q15,
                                              int win_len, int win_inc,
                                              uint32_t *wola_inv_q30) {
    for (int i = 0; i < win_inc; i++) {
        int64_t sum_sq = 0;
        for (int k = 0; i + k * win_inc < win_len; k++) {
            int32_t w = win_q15[i + k * win_inc];
            sum_sq += (int64_t)w * w;  /* Q30 */
        }
        if (sum_sq > 0)
            wola_inv_q30[i] = (uint32_t)((1ULL << 60) / sum_sq);  /* 2^60 / (2^30) = 2^30 */
        else
            wola_inv_q30[i] = 1;
    }
}

/**
 * nn_wola_output — Normalize and output WOLA samples
 *
 * ola_buf[win_len + win_inc]: overlap-add accumulation buffer (int32)
 * ola_pos: current read position in ola_buf [0, win_inc)
 * wola_inv_q30[win_inc]: pre-computed normalization table
 * n: number of samples to output
 * gain_q15: output gain (32768 = 1.0)
 * out[n]: output samples in Q15 int16
 */
static inline void nn_wola_output(int32_t *ola_buf, int *ola_pos,
                                   const uint32_t *wola_inv_q30,
                                   int win_inc, int n,
                                   int32_t gain_q15, int16_t *out) {
    int buf_len = win_inc + 512;  /* default: WIN_LEN + WIN_INC */
    for (int i = 0; i < n; i++) {
        int32_t v = ola_buf[*ola_pos];
        ola_buf[*ola_pos] = 0;
        int idx = *ola_pos % win_inc;
        *ola_pos = (*ola_pos + 1) % buf_len;

        /* Normalize: (v * inv_q30 + 0.5) >> 30 → Q0, then * gain_q15 → Q15 */
        int64_t norm = ((int64_t)v * (int64_t)wola_inv_q30[idx] + (1LL << 29)) >> 30;
        int32_t scaled = (int32_t)(((int64_t)norm * gain_q15 + (NN_Q15_SCALE >> 1)) >> 15);

        if (scaled >  32767) scaled =  32767;
        if (scaled < -32768) scaled = -32768;
        out[i] = (int16_t)scaled;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* NN_WOLA_H */
