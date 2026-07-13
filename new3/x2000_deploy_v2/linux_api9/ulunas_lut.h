/**
 * ulunas_lut.h — Fixed-Point LUT Declarations
 * ============================================
 * Sigmoid/Tanh: 1024-point Q20→Q15 LUT with linear interpolation
 * Log10: 512-point Q20→Q20 LUT
 * Sqrt: integer binary search Q40→Q20
 */

#ifndef ULUNAS_LUT_H
#define ULUNAS_LUT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * LUT sizes
 * ================================================================ */
#define SIGMOID_LUT_SIZE        1024
#define TANH_LUT_SIZE           1024
#define LOG10_LUT_SIZE          512

/* ================================================================
 * LUT data tables (defined in ulunas_lut.c)
 * ================================================================ */
extern const uint16_t sigmoid_lut_q15[SIGMOID_LUT_SIZE];
extern const int16_t tanh_lut_q15[TANH_LUT_SIZE];
extern const int32_t log10_lut_q20[LOG10_LUT_SIZE];

/* ================================================================
 * Lookup Functions
 * ================================================================ */

/**
 * sigmoid_q20_to_q15: int32_t Q20 → uint16_t Q15
 * Input range:  [-8.0, 8.0]  mapped to [-8388608, 8388608]
 * Output range: [0, 32768]
 * LUT: 1024 entries covering [-8, 8), step = 16/1024 = 0.015625
 * Index: (x + 8388608) >> 14  → 10-bit index
 * Fraction: (x + 8388608) & 0x3FFF → 14-bit fraction for interpolation
 */
uint16_t sigmoid_q20_to_q15(int32_t x_q20);

/**
 * tanh_q20_to_q15: int32_t Q20 → int16_t Q15
 * Input range:  [-4.0, 4.0]  mapped to [-4194304, 4194304]
 * Output range: [-32768, 32767]
 * LUT: 1024 entries covering [-4, 4), step = 8/1024 = 0.0078125
 * Index: (x + 4194304) >> 13  → 10-bit index
 * Fraction: (x + 4194304) & 0x1FFF → 13-bit fraction
 */
int16_t tanh_q20_to_q15(int32_t x_q20);

/**
 * log10_q20: int32_t Q20 → int32_t Q20
 * Input:  magnitude > 0 (clamped to min_value)
 * Output: log10(magnitude) in Q20
 * Uses 512-point LUT + log2 approximation + change of base
 */
int32_t log10_q20(int32_t x_q20);

/**
 * sqrt_q40_to_q20: uint64_t Q40 → uint32_t Q20
 * Integer binary search sqrt
 * Input:  mag² in Q40 (from squaring Q20 real + Q20 imag)
 * Output: sqrt in Q20
 */
uint32_t sqrt_q40_to_q20(uint64_t x_q40);

#ifdef __cplusplus
}
#endif

#endif /* ULUNAS_LUT_H */
