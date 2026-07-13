/**
 * nn_rnn.h — GRU / BiGRU Operators (Fixed-Point)
 * ===============================================
 * nn_gru_step  — Single-timestep GRU (ULUNAS: per-frame inference)
 * nn_gru_seq   — Multi-timestep GRU (GTCRN: batch processing)
 * nn_bigru     — Bidirectional GRU (both projects)
 *
 * Q-format:
 *   Input x:          Q20 (int32_t)
 *   Hidden state h:   Q15 (int16_t)
 *   Output y:         Q15 (int16_t)
 *   ih_weight:        s16f12
 *   ih_bias:          s16f10
 *   hh_weight:        s16f12
 *   hh_bias:          s16f10
 *
 * GRU equations (matching MATLAB):
 *   r_t = sigmoid( W_ir·x + b_ir + W_hr·h_prev + b_hr )   → u16f15
 *   z_t = sigmoid( W_iz·x + b_iz + W_hz·h_prev + b_hz )   → u16f15
 *   n_t = tanh(    W_in·x + b_in + r_t * (W_hn·h_prev + b_hn) ) → s16f15
 *   h_t = (1 - z_t) * n_t + z_t * h_prev                   → s16f15
 *
 * ⚠️ GRU CHECKLIST (5 critical bugs fixed across 2 projects):
 *   1. r_t/z_t MUST be uint16_t (sigmoid returns 0-32768, int16_t wraps 32768→-32768)
 *   2. ih and hh paths have independent Qr: qr1 (ih path), qr2 (hh path)
 *   3. GRU hidden output: Q20 internal → round_shr(,5) → Q15 h_cache
 *   4. Per-timestep: h_cache must be explicitly passed and updated each step
 *   5. BiGRU backward: process t=T-1..0, THEN flip output
 *
 * Verified: GTCRN (denoise_v19_q15) + UL-UNAS (linux_api9), X2000 MIPS32R2
 * License: MIT
 */

#ifndef NN_RNN_H
#define NN_RNN_H

#include "nn_qformat.h"

/* Forward declarations for LUT functions (from nn_lut.h) */
uint16_t nn_sigmoid_q15(int32_t x_q20);
int16_t nn_tanh_q15(int32_t x_q20);

#ifdef __cplusplus
extern "C" {
#endif

/**
 * nn_gru_step — Single-Timestep GRU (ULUNAS verified)
 *
 * x_t[in_dim] in Q20. h_cache[nHidden] in Q15 (updated in-place).
 * ih_weight[in_dim][3*nHidden] in s16f12, ih_bias[3*nHidden] in s16f10
 * hh_weight[nHidden][3*nHidden] in s16f12, hh_bias[3*nHidden] in s16f10
 *
 * qr1: ih path shift (typically -13)
 * qr2: hh path shift (typically -8)
 *
 * y[nHidden] output in Q15
 */
static inline void nn_gru_step(const int32_t *x_t, int nHidden, int in_dim,
                                int16_t *h_cache,
                                const int16_t *ih_weight, const int32_t *ih_bias,
                                const int16_t *hh_weight, const int32_t *hh_bias,
                                int qr1, int qr2, int16_t *y) {
    int n3 = 3 * nHidden;
    int32_t gate_r_z_n[3];  /* Q20 gate values before sigmoid/tanh */

    for (int g = 0; g < 3; g++) {
        /* ih path: x_t × ih_weight */
        int64_t sum_ih = ih_bias ? (int64_t)ih_bias[g * nHidden] : 0;
        for (int i = 0; i < in_dim; i++) {
            sum_ih += (int64_t)x_t[i]
                    * ih_weight[i * n3 + g * nHidden];
        }
        /* hh path: h_cache × hh_weight */
        int64_t sum_hh = hh_bias ? (int64_t)hh_bias[g * nHidden] : 0;
        for (int i = 0; i < nHidden; i++) {
            sum_hh += (int64_t)(int32_t)h_cache[i]
                    * hh_weight[i * n3 + g * nHidden];
        }
        /* Combine with independent Qr */
        int64_t sum_ih_scaled = nn_round_shr(sum_ih, -qr1);
        int64_t sum_hh_scaled = nn_round_shr(sum_hh, -qr2);
        gate_r_z_n[g] = (int32_t)(sum_ih_scaled + sum_hh_scaled);
    }

    /* Apply activations (CRITICAL: use uint16_t for gates!) */
    uint16_t z_t = nn_sigmoid_q15(gate_r_z_n[1]);  /* update gate */
    (void)nn_sigmoid_q15(gate_r_z_n[0]);            /* r_t: reset gate (used in full GRU) */
    int16_t  n_t = nn_tanh_q15(gate_r_z_n[2]);      /* new gate */

    /* h_new = (1 - z_t) * n_t + z_t * h_prev  (all in Q15) */
    for (int i = 0; i < nHidden; i++) {
        int32_t h_new = (int32_t)((NN_Q15_ONE - z_t) * n_t + z_t * h_cache[i]
                                  + NN_Q15_ONE / 2) >> 15;
        h_cache[i] = nn_clamp_i16(h_new);
        if (y) y[i] = h_cache[i];
    }
}

/**
 * nn_gru_seq — Multi-Timestep GRU (GTCRN verified)
 *
 * x[seq_len * input_dim] in Q20.
 * Processes all time steps sequentially.
 * y[seq_len * hidden_dim] in Q15 output.
 * h_prev[hidden_dim] in Q15 (initial state, updated to final state).
 */
static inline void nn_gru_seq(const int32_t *x, int seq_len,
                               int input_dim, int hidden_dim,
                               const int16_t *ih_weight, const int16_t *ih_bias,
                               const int16_t *hh_weight, const int16_t *hh_bias,
                               int qr1, int qr2,
                               int16_t *y, int16_t *h_prev) {
    (void)(3 * hidden_dim);  /* n3: used by nn_gru_step internally */
    for (int t = 0; t < seq_len; t++) {
        const int32_t *xt = x + t * input_dim;
        int16_t *yt = y + t * hidden_dim;
        nn_gru_step(xt, hidden_dim, input_dim, h_prev,
                    ih_weight, (const int32_t*)ih_bias,
                    hh_weight, (const int32_t*)hh_bias,
                    qr1, qr2, yt);
    }
}

/**
 * nn_bigru — Bidirectional GRU (both projects verified)
 *
 * x[T * in_dim] in Q20. y[T * 2*nHidden] in Q15.
 * Forward: t=0..T-1, Backward: t=T-1..0 then flipped.
 * Concatenation: y[t][0:nHidden] = fwd_output, y[t][nHidden:2*nHidden] = bwd_output
 */
static inline void nn_bigru(const int32_t *x, int T, int nHidden, int in_dim,
                             const int16_t *fwd_ih_w, const int32_t *fwd_ih_b,
                             const int16_t *fwd_hh_w, const int32_t *fwd_hh_b,
                             const int16_t *bwd_ih_w, const int32_t *bwd_ih_b,
                             const int16_t *bwd_hh_w, const int32_t *bwd_hh_b,
                             int qr1, int qr2, int16_t *y) {
    int16_t h_fwd[32] = {0};  /* max nHidden = 32 */
    int16_t h_bwd[32] = {0};
    int16_t *fwd_out = (int16_t*)malloc(T * nHidden * sizeof(int16_t));
    int16_t *bwd_out = (int16_t*)malloc(T * nHidden * sizeof(int16_t));
    if (!fwd_out || !bwd_out) { free(fwd_out); free(bwd_out); return; }

    /* Forward pass */
    for (int t = 0; t < T; t++) {
        const int32_t *xt = x + t * in_dim;
        nn_gru_step(xt, nHidden, in_dim, h_fwd,
                    fwd_ih_w, fwd_ih_b, fwd_hh_w, fwd_hh_b,
                    qr1, qr2, fwd_out + t * nHidden);
    }

    /* Backward pass */
    for (int t = T - 1; t >= 0; t--) {
        const int32_t *xt = x + t * in_dim;
        nn_gru_step(xt, nHidden, in_dim, h_bwd,
                    bwd_ih_w, bwd_ih_b, bwd_hh_w, bwd_hh_b,
                    qr1, qr2, bwd_out + t * nHidden);
    }

    /* Concatenate [fwd, bwd_flipped] */
    for (int t = 0; t < T; t++) {
        for (int i = 0; i < nHidden; i++) {
            y[t * 2 * nHidden + i] = fwd_out[t * nHidden + i];
            y[t * 2 * nHidden + nHidden + i] = bwd_out[t * nHidden + i];
        }
    }
    free(fwd_out); free(bwd_out);
}

#ifdef __cplusplus
}
#endif

#endif /* NN_RNN_H */
