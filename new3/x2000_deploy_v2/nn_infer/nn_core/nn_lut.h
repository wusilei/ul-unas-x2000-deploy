/**
 * nn_lut.h — Fixed-Point LUT: Sigmoid / Tanh / Log10 / Sqrt
 * ===========================================================
 * 1024-point Q20→Q15 sigmoid/tanh LUT with linear interpolation.
 * 512-point Q20→Q20 log10 LUT.
 * Integer sqrt Q40→Q20 (binary search).
 *
 * All LUT data is in nn_lut.c (const tables, compiled once).
 * Lookup functions are also in nn_lut.c (non-inline to save I$ on X2000).
 *
 * Verified: GTCRN (denoise_v19_q15) + UL-UNAS (linux_api9), X2000 MIPS32R2
 * License: MIT
 */

#ifndef NN_LUT_H
#define NN_LUT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* LUT sizes */
#define NN_SIGMOID_LUT_SIZE  1024
#define NN_TANH_LUT_SIZE     1024
#define NN_LOG10_LUT_SIZE    512

/* LUT data tables (defined in nn_lut.c) */
extern const uint16_t nn_sigmoid_lut_q15[NN_SIGMOID_LUT_SIZE];
extern const int16_t  nn_tanh_lut_q15[NN_TANH_LUT_SIZE];
extern const int32_t  nn_log10_lut_q20[NN_LOG10_LUT_SIZE];

/**
 * nn_sigmoid_q15: Q20 input → uint16_t Q15 output
 * Input range:  [-8.0, 8.0) mapped to [-8388608, 8388608)
 * Output range: [0, 32768]
 * LUT: 1024 entries, step = 16/1024 = 0.015625
 * Index: (x + 8388608) >> 14
 * Max error: < 0.5 LSB Q15
 */
uint16_t nn_sigmoid_q15(int32_t x_q20);

/**
 * nn_tanh_q15: Q20 input → int16_t Q15 output
 * Input range:  [-4.0, 4.0) mapped to [-4194304, 4194304)
 * Output range: [-32768, 32767]
 * LUT: 1024 entries, step = 8/1024 = 0.0078125
 * Index: (x + 4194304) >> 13
 * Max error: < 0.5 LSB Q15
 */
int16_t nn_tanh_q15(int32_t x_q20);

/**
 * nn_log10_q20: Q20 magnitude → Q20 log10
 * Input:  magnitude > 0 (clamped to min_value)
 * Output: log10(magnitude) in Q20
 * Uses 512-point LUT + log2 approximation + change of base
 */
int32_t nn_log10_q20(int32_t x_q20);

/**
 * nn_sqrt_q40_to_q20: Q40 → Q20 integer sqrt
 * Input:  sum-of-squares in Q40 (from squaring Q20 values)
 * Output: sqrt in Q20
 * Uses binary search, ~32 iterations max
 */
uint32_t nn_sqrt_q40_to_q20(uint64_t x_q40);

/**
 * nn_log_gen_q20: Log-magnitude compression (replaces log_gen_fixed)
 * Input:  real[W] (Q20), imag[W] (Q20)
 * Output: log10(sqrt(real² + imag²)) in Q20, per frequency bin
 */
static inline void nn_log_gen_q20(const int32_t *real, const int32_t *imag,
                                   int W, int32_t *out) {
    for (int i = 0; i < W; i++) {
        int64_t re = real[i], im = imag[i];
        uint64_t sq_sum = (uint64_t)(re * re + im * im);  /* Q40 */
        uint32_t mag_q20 = nn_sqrt_q40_to_q20(sq_sum);     /* Q20 */
        out[i] = nn_log10_q20((int32_t)mag_q20);
    }
}

/**
 * nn_sigmoidf_fp: Float→Float sigmoid using LUT
 * Drop-in replacement for expf() on MIPS soft-float (~30 vs ~1000 cycles).
 */
static inline float nn_sigmoidf_fp(float x) {
    int32_t q10 = (int32_t)(x * 1024.0f + (x >= 0 ? 0.5f : -0.5f));
    /* Use Q20→Q15 LUT via scaling: Q10 << 10 = Q20 */
    uint16_t q15 = nn_sigmoid_q15(q10 << 10);
    return (float)q15 * (1.0f / 32768.0f);
}

/**
 * nn_tanhf_fp: Float→Float tanh using LUT
 */
static inline float nn_tanhf_fp(float x) {
    int32_t q10 = (int32_t)(x * 1024.0f + (x >= 0 ? 0.5f : -0.5f));
    int16_t q15 = nn_tanh_q15(q10 << 10);
    return (float)q15 * (1.0f / 32768.0f);
}

#ifdef __cplusplus
}
#endif

#endif /* NN_LUT_H */
