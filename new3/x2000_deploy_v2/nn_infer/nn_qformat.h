/* nn_qformat.h — Unified Q-format type system for fixed-point inference
 * Part of nn_infer: reusable fixed-point inference framework.
 *
 * Design principles:
 *  - All Q-format constants are overridable via #ifndef before inclusion
 *  - Data layout: row-major (C, W) — matches C convention
 *    MATLAB column-major (W, C) is handled at weight-export time
 *  - All accumulators use int64 to prevent overflow
 */

#ifndef NN_QFORMAT_H
#define NN_QFORMAT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * 1. DEFAULT Q-FORMAT CONSTANTS (overridable per-project)
 * ======================================================================== */

/* ── Activation / Signal domains ── */
#ifndef NN_Q_ACT
#define NN_Q_ACT         20    /* int32_t, main activations (conv i/o, BN, residual) */
#endif
#ifndef NN_Q_LOG
#define NN_Q_LOG         20    /* uint32_t, log_gen / cTFA aggregation */
#endif
#ifndef NN_Q_GRU_H
#define NN_Q_GRU_H       15    /* int16_t, GRU hidden state / tanh output */
#endif
#ifndef NN_Q_SIG
#define NN_Q_SIG         15    /* uint16_t, sigmoid output / cTFA attention mask */
#endif

/* ── Weight Q-formats (float→int quantization shift) ── */
#ifndef NN_Q_W_CONV
#define NN_Q_W_CONV      14    /* s16f14: Conv/DeConv weight */
#endif
#ifndef NN_Q_W_PCONV
#define NN_Q_W_PCONV     13    /* s16f13: PConv weight */
#endif
#ifndef NN_Q_W_GRU
#define NN_Q_W_GRU       12    /* s16f12: GRU ih/hh weight */
#endif
#ifndef NN_Q_W_GRU_B
#define NN_Q_W_GRU_B     10    /* s16f10: GRU bias */
#endif
#ifndef NN_Q_W_FC
#define NN_Q_W_FC        13    /* s16f13: FC weight */
#endif
#ifndef NN_Q_W_FC_B
#define NN_Q_W_FC_B      20    /* s32f20: FC bias */
#endif
#ifndef NN_Q_W_BN_W
#define NN_Q_W_BN_W      14    /* s16f14 or u16f14: BN weight */
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

/* ========================================================================
 * 2. SCALE FACTORS
 * ======================================================================== */

#define NN_SCALE_Q20     ((int64_t)1 << 20)   /* 1048576 */
#define NN_SCALE_Q15     ((int32_t)1 << 15)   /* 32768 */
#define NN_SCALE_Q14     ((int32_t)1 << 14)   /* 16384 */
#define NN_SCALE_Q13     ((int32_t)1 << 13)   /* 8192 */
#define NN_SCALE_Q12     ((int32_t)1 << 12)   /* 4096 */
#define NN_SCALE_Q11     ((int32_t)1 << 11)   /* 2048 */
#define NN_SCALE_Q10     ((int32_t)1 << 10)   /* 1024 */

/* ── "1.0" constants in various Q-formats ── */
#define NN_ONE_Q20       ((int32_t)1048576)
#define NN_ONE_Q15       ((uint16_t)32768)
#define NN_ONE_Q14       ((int16_t)16384)

/* ========================================================================
 * 3. SATURATION HELPERS
 * ======================================================================== */

/* Saturate int64 → int32 */
static inline int32_t nn_sat_i32(int64_t x) {
    if (x >  2147483647LL) return  2147483647;
    if (x < -2147483648LL) return -2147483648;
    return (int32_t)x;
}

/* Saturate int32 → int16 */
static inline int16_t nn_sat_i16(int32_t x) {
    if (x >  32767) return  32767;
    if (x < -32768) return -32768;
    return (int16_t)x;
}

/* Saturate int32 → uint16 */
static inline uint16_t nn_sat_u16(int32_t x) {
    if (x > 65535) return 65535;
    if (x < 0)     return 0;
    return (uint16_t)x;
}

/* ========================================================================
 * 4. ROUNDING + SHIFT HELPERS
 * ======================================================================== */

/* Round(x / 2^N) for N < 0 (meaning right-shift)
 * Example: nn_round_shr(x, -13) = round(x / 2^13) */
static inline int32_t nn_round_shr(int64_t x, int shift) {
    /* shift is negative for right-shift: shift = -N means >> N */
    int N = -shift;
    int64_t half = ((int64_t)1 << (N - 1));
    if (x >= 0) return (int32_t)((x + half) >> N);
    else        return (int32_t)((x - half) >> N);
}

/* Convenience macro: round(x * weight) then shift by qr bits */
#define NN_RND_SHR(x, qr)   nn_round_shr((int64_t)(x), (qr))

/* Element-wise multiply + shift: y = round(x * w * 2^qr)
 * where qr is negative (right-shift). Used for applying weights. */
#define NN_MUL_Q(x, w, qr)  ((int32_t)(((int64_t)(x) * (int64_t)(w) + (((int64_t)1 << ((-(qr)) - 1)))) >> (-(qr))))
#define NN_MUL_UQ(x, w, qr) ((int32_t)(((int64_t)(x) * (int64_t)(uint32_t)(w) + (((int64_t)1 << ((-(qr)) - 1)))) >> (-(qr))))

/* ========================================================================
 * 5. CONVERSION HELPERS
 * ======================================================================== */

/* Q20 ↔ float (for debug/tool use only, not used in inference) */
#define NN_Q20_TO_FLOAT(x)   ((float)(x) / 1048576.0f)
#define NN_FLOAT_TO_Q20(x)   ((int32_t)((x) * 1048576.0f + 0.5f))

/* Q15 ↔ float */
#define NN_Q15_TO_FLOAT(x)   ((float)(x) / 32768.0f)
#define NN_FLOAT_TO_Q15(x)   ((int16_t)((x) * 32768.0f + 0.5f))

/* ========================================================================
 * 6. INTEGER MATH HELPERS
 * ======================================================================== */

/* Integer sqrt: Q40 input → Q20 output (Newton's method) */
static inline uint32_t nn_isqrt64(uint64_t x_q40) {
    if (x_q40 == 0) return 0;
    uint64_t y = x_q40;
    uint64_t y1 = (y + (x_q40 / y)) >> 1;
    while (y1 < y) {
        y = y1;
        y1 = (y + (x_q40 / y)) >> 1;
    }
    return (uint32_t)y;  /* Q20 output from Q40 input */
}

#ifdef __cplusplus
}
#endif

#endif /* NN_QFORMAT_H */
