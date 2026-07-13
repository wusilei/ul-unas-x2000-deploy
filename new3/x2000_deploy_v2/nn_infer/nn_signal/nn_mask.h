/**
 * nn_mask.h — CRM Complex Ratio Mask Application
 * ===============================================
 * y_real = round(x_real * mask * 2^(-15))
 * y_imag = round(x_imag * mask * 2^(-15))
 *
 * Output format: y[W] = real_masked, y[W+W] = imag_masked (concatenated)
 *
 * Verified: UL-UNAS (linux_api9), GTCRN (denoise_v19_q15)
 * License: MIT
 */

#ifndef NN_MASK_H
#define NN_MASK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * nn_mask — Apply complex ratio mask
 * mask[W] in s16f15 (Q15), x_real[W]/x_imag[W] in Q20
 * y[2*W] output: [real_masked; imag_masked] in Q20
 */
static inline void nn_mask(const int16_t *mask,
                            const int32_t *x_real, const int32_t *x_imag,
                            int W, int32_t *y) {
    for (int i = 0; i < W; i++) {
        /* mask * real >> 15 : keeping Q20 output */
        int64_t re = (int64_t)x_real[i] * mask[i];
        y[i] = (int32_t)((re + (NN_Q15_SCALE >> 1)) >> 15);
        int64_t im = (int64_t)x_imag[i] * mask[i];
        y[W + i] = (int32_t)((im + (NN_Q15_SCALE >> 1)) >> 15);
    }
}

#ifdef __cplusplus
}
#endif

#endif /* NN_MASK_H */
