/**
 * agc.h — 语音 AGC + 降噪联合模块 (v22 — linux_api17, 前置AGC)
 * ================================================================
 *
 * GTCRN linux_api19 方案: AGC先 → 固定增益 → NR
 *
 * 管线:
 *   state_flag=0: bypass
 *   state_flag=1: 只降噪 (DENOISE)
 *   state_flag=2: 只AGC (LMS + 固定增益)
 *   state_flag=3: 前置AGC (LMS→固定增益→NR)
 *
 * linux_api17 变更 (vs api14):
 *   - 前置AGC: LMS压动态 → 固定增益(2.5x/2.0x) → NR
 *   - 去掉五级pre-gain, 只保留两级能量门控
 *   - GTCRN参数: GOAL_HIGH=2^20, GOAL_LOW=2^10, MU1=100
 */
#ifndef AGC_H
#define AGC_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STATE_FLAG_NONE     0
#define STATE_FLAG_DENOISE  1
#define STATE_FLAG_AGC      2
#define STATE_FLAG_BOTH     3

#define ENERGY_ABS_THR_HIGH   250
#define ENERGY_ABS_THR_LOW    144
#define ENERGY_SMOOTH_THR     144
#define ENERGY_HISTORY_FAC    20783

/* ── GTCRN 两级能量门控 (v17b: 减轻近场压缩) ── */
#define AGC_GOAL_HIGH  (1LL << 21)   /* 2097152 → RMS≈1448  减轻压缩 */
#define AGC_GOAL_LOW   (1LL << 10)   /* 1024    → RMS≈32    底噪强衰减 */

/* ── 固定增益 (LMS后, v17b: 降低以补偿更高goal) ── */
#define AGC_GAIN_HIGH_NUM 4           /* 2.0x (原2.5x) */
#define AGC_GAIN_LOW_NUM  4           /* 2.0x */
#define AGC_GAIN_DEN      2

/* ── LMS 参数 (GTCRN 原版) ── */
#define AGC_V_A        800
#define AGC_V_MU1      100            /* 大信号慢速 (GTCRN原值) */
#define AGC_V_MU2      800
#define AGC_V_MU3      1600
#define AGC_V_SHIFT1   10
#define AGC_V_SHIFT2   45
#define AGC_V_SHIFT3   15
#define AGC_V_GAIN_MIN 31
#define AGC_V_GAIN_MAX 32767          /* 1.0x, LMS只衰减 (GTCRN原值) */

unsigned short energy_calculate_and_smooth_s16(short *data_in,
    unsigned short data_len, unsigned short history_fac,
    unsigned short *energy_out);

void voice_NoiseReductionAndAGC(short *in, short *out,
    long long goal, short flag, int n_samples);

void agc_init(void);

#ifdef __cplusplus
}
#endif
#endif
