/**
 * nn_agc.h — Voice AGC: LMS Adaptive Gain Control + 3-Level Energy Gate
 * =====================================================================
 * Merged from GTCRN (agc.c v20) + UL-UNAS (linux_api9).
 *
 * Pipeline topologies (all verified):
 *   Pre-AGC:   AGC → NR
 *   Post-AGC:  NR → AGC
 *   Hybrid:    Gain → NR → LMS AGC     (ULUNAS v7, GTCRN v14+)
 *   3-Gate:    Gain → NR → LMS + Gate  (ULUNAS v9) ← RECOMMENDED
 *
 * Energy gate tiers (UL-UNAS linux_api9):
 *   raw_energy > thr_voice  → GOAL_HIGH + HIGH_GAIN   (near-field voice)
 *   thr_far < energy ≤ thr  → GOAL_LOW  + LOW_GAIN    (far-field voice)
 *   energy ≤ thr_far        → GOAL_NOISE + NOISE_GAIN (noise/echo, attenuated)
 *
 * Verified: X2000 walkie-talkie real deployment
 * License: MIT
 */

#ifndef NN_AGC_H
#define NN_AGC_H

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Default Gate Configuration (UL-UNAS linux_api9)
 * ================================================================ */
#define NN_AGC_THR_VOICE     200    /* >200 → near-field voice */
#define NN_AGC_THR_FAR        30    /* 30-200 → far-field; ≤30 → noise */
#define NN_AGC_GOAL_HIGH  (1LL<<20) /* 1048576, RMS≈1024 */
#define NN_AGC_GOAL_LOW   (1LL<<18) /* 262144,  RMS≈512 */
#define NN_AGC_GOAL_NOISE (1LL<<10) /* 1024,    strong attenuation */
#define NN_AGC_GAIN_HIGH_NUM  5    /* 2.5x */
#define NN_AGC_GAIN_LOW_NUM   5    /* 2.5x */
#define NN_AGC_GAIN_NOISE_NUM 1    /* 0.5x */
#define NN_AGC_GAIN_DEN       2

/* LMS parameters (GTCRN verified) */
#define NN_AGC_A        800
#define NN_AGC_MU1      100
#define NN_AGC_MU2      800
#define NN_AGC_MU3      1600
#define NN_AGC_SHIFT1   10
#define NN_AGC_SHIFT2   45
#define NN_AGC_SHIFT3   15
#define NN_AGC_GAIN_MIN 31
#define NN_AGC_GAIN_MAX 32767
#define NN_AGC_HISTORY_FAC 20783   /* Q15, ~0.634 */

/* ================================================================
 * Types
 * ================================================================ */

typedef struct {
    int64_t goal_high, goal_low, goal_noise;
    int gain_high_num, gain_low_num, gain_noise_num, gain_den;
    uint16_t thr_voice, thr_far;
} nn_agc_gate_config_t;

typedef struct {
    int64_t px_pre, py_pre, g_pre;
    int state_inited, first_frame;
    uint16_t energy_smoothed;
    nn_agc_gate_config_t gate_cfg;
} nn_agc_state_t;

/* ================================================================
 * Default Config
 * ================================================================ */
static const nn_agc_gate_config_t NN_AGC_GATE_DEFAULT = {
    NN_AGC_GOAL_HIGH, NN_AGC_GOAL_LOW, NN_AGC_GOAL_NOISE,
    NN_AGC_GAIN_HIGH_NUM, NN_AGC_GAIN_LOW_NUM, NN_AGC_GAIN_NOISE_NUM, NN_AGC_GAIN_DEN,
    NN_AGC_THR_VOICE, NN_AGC_THR_FAR
};

/* ================================================================
 * Energy Calculation + Smoothing (LongXueKun)
 * ================================================================ */

/** Returns raw current energy (abs average). Updates smoothed energy. */
static inline uint16_t nn_agc_energy_smooth(const int16_t *data, uint16_t len,
                                             uint16_t history_fac,
                                             uint16_t *energy_out) {
    uint32_t sum = 0;
    for (uint16_t i = 0; i < len; i++) {
        int16_t v = data[i];
        sum += (v < 0) ? (uint16_t)(-v) : (uint16_t)v;
    }
    uint16_t cur = (uint16_t)(sum / len);
    *energy_out = (uint16_t)((((uint32_t)history_fac * (*energy_out))
                   + ((0x7fffU - history_fac) * cur) + 16384) >> 15);
    return cur;
}

/* ================================================================
 * LMS Adaptive Gain Control (GTCRN verified)
 * ================================================================ */

static inline int16_t nn_agc_s16_clamp(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

static inline void nn_agc_lms_core(const int16_t *in, int16_t *out,
                                    int n_samples, int64_t goal_val,
                                    int first_flag,
                                    int64_t *px_pre, int64_t *py_pre,
                                    int64_t *g_pre, int *state_inited) {
    int i;
    if (!(*state_inited)) {
        if (n_samples > 0) {
            int16_t x = in[0]; out[0] = x;
            *px_pre = *py_pre = (int64_t)x * x;
            *g_pre = 32768; *state_inited = 1;
        }
        i = 1;
    } else {
        i = first_flag ? 1 : 0;
    }

    for (; i < n_samples; i++) {
        int16_t x = in[i];
        int32_t mu;
        if (x > 1024 || x < -1024) mu = NN_AGC_MU1;
        else if (x > -16 && x < 16) mu = NN_AGC_MU3;
        else mu = NN_AGC_MU2;

        int64_t px_cur = (NN_AGC_A * (*px_pre) + ((1LL << NN_AGC_SHIFT1) - NN_AGC_A)
                         * (int64_t)((int)x * (int)x)) >> NN_AGC_SHIFT1;
        int64_t g_cur = (*g_pre) * ((1LL << (60 - NN_AGC_SHIFT2))
                       + ((mu * px_cur * (goal_val - (*py_pre))) >> NN_AGC_SHIFT2));
        int16_t y = (int16_t)((g_cur * x) >> (NN_AGC_SHIFT3 + (60 - NN_AGC_SHIFT2)));
        g_cur >>= (60 - NN_AGC_SHIFT2);
        if (g_cur < NN_AGC_GAIN_MIN) g_cur = NN_AGC_GAIN_MIN;
        if (g_cur > NN_AGC_GAIN_MAX) g_cur = NN_AGC_GAIN_MAX;
        int64_t py_cur = (g_cur * g_cur * px_cur) >> (NN_AGC_SHIFT3 * 2);
        *px_pre = px_cur; *py_pre = py_cur; *g_pre = g_cur;
        out[i] = y;
    }
}

/* ================================================================
 * 3-Level Energy Gate Decision
 * ================================================================ */

static inline void nn_agc_gate_decide(uint16_t raw_energy,
                                       const nn_agc_gate_config_t *cfg,
                                       int64_t *goal_val, int *gain_num, int *gain_den) {
    if (raw_energy > cfg->thr_voice) {
        *goal_val = cfg->goal_high;
        *gain_num = cfg->gain_high_num;
    } else if (raw_energy > cfg->thr_far) {
        *goal_val = cfg->goal_low;
        *gain_num = cfg->gain_low_num;
    } else {
        *goal_val = cfg->goal_noise;
        *gain_num = cfg->gain_noise_num;
    }
    *gain_den = cfg->gain_den;
}

/* ================================================================
 * High-Level AGC API
 * ================================================================ */

static inline void nn_agc_init(nn_agc_state_t *st, const nn_agc_gate_config_t *cfg) {
    memset(st, 0, sizeof(*st));
    st->gate_cfg = *cfg;
}

/**
 * nn_agc_process — Full AGC pipeline with 3-level gate
 *
 * st: AGC state (persistent across frames)
 * in[n]: input PCM Q15
 * out[n]: output PCM Q15
 * n: number of samples
 * enable_gate: 1 = use energy gate, 0 = use fixed goal (AGC_GOAL_HIGH)
 * fixed_gain_num/den: applied after LMS (e.g., 1.5x = {3, 2})
 */
static inline void nn_agc_process_gated(nn_agc_state_t *st,
                                         const int16_t *in, int16_t *out,
                                         int n_samples,
                                         int enable_gate,
                                         int fixed_gain_num, int fixed_gain_den) {
    /* Energy detection */
    uint16_t raw = nn_agc_energy_smooth(in, (uint16_t)n_samples,
                                         NN_AGC_HISTORY_FAC, &st->energy_smoothed);
    /* Gate decision */
    int64_t goal;
    int gn, gd;
    if (enable_gate) {
        nn_agc_gate_decide(raw, &st->gate_cfg, &goal, &gn, &gd);
    } else {
        goal = st->gate_cfg.goal_high;
        gn = fixed_gain_num; gd = fixed_gain_den;
    }

    /* LMS AGC */
    int vf = st->state_inited ? 1 : 0;
    nn_agc_lms_core(in, out, n_samples, goal, vf,
                    &st->px_pre, &st->py_pre, &st->g_pre, &st->state_inited);

    /* Fixed gain (skip for first frame) */
    if (st->state_inited && !st->first_frame) {
        for (int i = 0; i < n_samples; i++) {
            int32_t v = (int32_t)out[i] * gn / gd;
            out[i] = nn_agc_s16_clamp(v);
        }
    }
    if (st->first_frame) st->first_frame = 0;
}

/** Simplified: default gate mode */
static inline void nn_agc_process(nn_agc_state_t *st,
                                   const int16_t *in, int16_t *out, int n_samples) {
    nn_agc_process_gated(st, in, out, n_samples, 1, 1, 1);
}

#ifdef __cplusplus
}
#endif

#endif /* NN_AGC_H */
