/**
 * nn_shuffle.h — Shuffle Interleave / Deinterleave
 * =================================================
 * MATLAB-compatible channel shuffle operations used in UL-UNAS
 * encoder/decoder skip connections (XMB/XWS blocks).
 *
 * interleave:  y(1:2:end) = x(1:N/2);  y(2:2:end) = x(N/2+1:end)
 * deinterleave: inverse of interleave
 *
 * Verified: UL-UNAS (linux_api9)
 * License: MIT
 */

#ifndef NN_SHUFFLE_H
#define NN_SHUFFLE_H

#include "nn_qformat.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * nn_shuffle_interleave: MATLAB y(1:2:end)=x(1:N/2); y(2:2:end)=x(N/2+1:end)
 * x[C][W] in Q20 → y[C][W] in Q20
 * For each channel c: interleave the two halves along W dimension.
 */
static inline void nn_shuffle_interleave(const int32_t *src, int C, int W,
                                          int32_t *dst) {
    int half = W / 2;
    for (int c = 0; c < C; c++) {
        const int32_t *sc = src + c * W;
        int32_t *dc = dst + c * W;
        for (int i = 0; i < half; i++) {
            dc[2 * i]     = sc[i];
            dc[2 * i + 1] = sc[half + i];
        }
    }
}

/**
 * nn_shuffle_deinterleave: Reverse of interleave
 * x[C][W] in Q20 → y[C][W] in Q20
 * y(1:W/2) = x(1:2:W);  y(W/2+1:W) = x(2:2:W)
 */
static inline void nn_shuffle_deinterleave(const int32_t *src, int C, int W,
                                            int32_t *dst) {
    int half = W / 2;
    for (int c = 0; c < C; c++) {
        const int32_t *sc = src + c * W;
        int32_t *dc = dst + c * W;
        for (int i = 0; i < half; i++) {
            dc[i]       = sc[2 * i];
            dc[half + i] = sc[2 * i + 1];
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif /* NN_SHUFFLE_H */
