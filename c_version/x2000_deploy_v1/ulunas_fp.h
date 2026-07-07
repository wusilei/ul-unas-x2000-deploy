/**
 * ulunas_fp.h — UL-UNAS MATLAB→C Fixed-Point Implementation
 * ==========================================================
 * Q-format matching MATLAB Fix_point.m exactly:
 *
 *   s32f20: activations (int32_t, 20 frac bits, ×1048576)
 *   s32f18: Conv weights encoder (int32_t, 18 frac bits, ×262144)
 *   s16f15: GRU hidden / sigmoid output (int16_t, 15 frac bits, ×32768)
 *   s16f14: BN/PReLU/DeConv weights (int16_t, 14 frac bits, ×16384)
 *   s16f13: FC/DD-Conv/PC weights (int16_t, 13 frac bits, ×8192)
 *   s16f12: GRU/LN weights (int16_t, 12 frac bits, ×4096)
 *   s16f11: LN running_var (int16_t, 11 frac bits, ×2048)
 *   s16f10: GRU bias (int16_t, 10 frac bits, ×1024)
 *   u16f15: sigmoid/tanh output / BM/BS (uint16_t, 15 frac bits, ×32768)
 *   u16f14: BN weight (uint16_t, 14 frac bits, ×16384)
 *   u16f13: BN running_var (uint16_t, 13 frac bits, ×8192)
 *   u16f12: BN running_var (uint16_t, 12 frac bits, ×4096)
 *   u16f11: LN running_var (uint16_t, 11 frac bits, ×2048)
 *
 * Target: Ingenic X2000 MIPS32R2 (no FPU) + PC verification
 *
 * Key differences from GTCRN:
 * - Single channel log-mag input (not 3ch mag+real+imag)
 * - cTFA (channel-wise T-F Attention) instead of TRA/DeTRA
 * - ERB filterbank with low-freq pass-through
 * - log10(mag) instead of sqrt(mag)
 * - Hann² + wsum COLA instead of sqrt-Hann
 * - XConv/XMB/XDWS encoder instead of GT-Conv
 *
 * Data layout: (C, W) for 2D tensors
 *   Indexing: element[c][w] = data[c*W + w]
 */

#ifndef ULUNAS_FP_H
#define ULUNAS_FP_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Q-format Constants (matching MATLAB Fix_point.m)
 * ================================================================ */

/* Activation Q-formats */
#define Q_ACT       20    /* s32f20: main activations */
#define Q_ACT_TR    18    /* s32f18: TRA/DeTRA agg quant (reserved) */

/* Weight Q-formats */
#define Q_WGT_CONV_E  18  /* s32f18: Conv weights (encoder Conv0, DeConv0) */
#define Q_WGT_FC      13  /* s16f13: FC weights, DD-Conv, PC weights */
#define Q_WGT_GRU     12  /* s16f12: GRU weights, LN weights */
#define Q_WGT_BN      14  /* s16f14 / u16f14: BN/PReLU weights */
#define Q_WGT_GRU_BIAS 10 /* s16f10: GRU bias */
#define Q_WGT_SIGMOID 15  /* u16f15: sigmoid/tanh output */
#define Q_WGT_BM_BS   15  /* u16f15: BM/BS weights */
#define Q_WGT_LN_VAR  11  /* u16f11: LN running_var */

/* BN running_var Q-formats (varies per block) */
#define Q_BN_VAR_14   14  /* u16f14 */
#define Q_BN_VAR_13   13  /* u16f13 */
#define Q_BN_VAR_12   12  /* u16f12 */
#define Q_BN_VAR_10   10  /* u16f10 */

/* ================================================================
 * Scale / Conversion Macros
 * ================================================================ */

/* Safe clamp to int16_t range */
static inline int16_t sat_s16f15(float x) {
    float r = roundf(x * 32768.0f);
    if (r > 32767.0f) return 32767;
    if (r < -32768.0f) return -32768;
    return (int16_t)r;
}

#define F2Q20(x)  ((int32_t)roundf((x) * 1048576.0f))   /* × 2^20 */
#define F2Q18(x)  ((int32_t)roundf((x) * 262144.0f))    /* × 2^18 */
#define F2Q15(x)  (sat_s16f15(x))                        /* × 2^15 */
#define F2Q14(x)  ((int16_t)roundf((x) * 16384.0f))     /* × 2^14 */
#define F2Q13(x)  ((int16_t)roundf((x) * 8192.0f))      /* × 2^13 */
#define F2Q12(x)  ((int16_t)roundf((x) * 4096.0f))      /* × 2^12 */
#define F2Q11(x)  ((int16_t)roundf((x) * 2048.0f))      /* × 2^11 */
#define F2Q10(x)  ((int16_t)roundf((x) * 1024.0f))      /* × 2^10 */

#define U2Q15(x)  ((uint16_t)roundf((x) * 32768.0f))
#define U2Q14(x)  ((uint16_t)roundf((x) * 16384.0f))
#define U2Q13(x)  ((uint16_t)roundf((x) * 8192.0f))
#define U2Q12(x)  ((uint16_t)roundf((x) * 4096.0f))
#define U2Q11(x)  ((uint16_t)roundf((x) * 2048.0f))
#define U2Q10(x)  ((uint16_t)roundf((x) * 1024.0f))

/* Dequant */
#define Q20_TO_F(x)  ((float)(x) / 1048576.0f)
#define Q18_TO_F(x)  ((float)(x) / 262144.0f)
#define Q15_TO_F(x)  ((float)(x) / 32768.0f)
#define Q10_TO_F(x)  ((float)(x) / 1024.0f)

/* ================================================================
 * Saturation Helpers
 * ================================================================ */

static inline int32_t sat32(int64_t x) {
    if (x > INT32_MAX) return INT32_MAX;
    if (x < INT32_MIN) return INT32_MIN;
    return (int32_t)x;
}

static inline int16_t sat16(int32_t x) {
    if (x > INT16_MAX) return INT16_MAX;
    if (x < INT16_MIN) return INT16_MIN;
    return (int16_t)x;
}

static inline uint16_t usat16(int32_t x) {
    if (x > UINT16_MAX) return UINT16_MAX;
    if (x < 0) return 0;
    return (uint16_t)x;
}

/* ROUND(x) in C: add half and truncate towards zero */
static inline int32_t round_div(int64_t num, int64_t den) {
    int64_t half = den >> 1;
    if (num >= 0) return (int32_t)((num + half) / den);
    else return (int32_t)((num - half) / den);
}

/* ================================================================
 * Sigmoid & Tanh — 512-entry LUT with linear interpolation
 * ================================================================
 * Copied from GTCRN (gtcrn_fp.h). ~30x faster than soft-float expf/tanhf
 * on MIPS. Max error < 1 LSB Q15.
 *
 * LUT: Q10 input (×1024) → Q15 output (×32768)
 *   sigmoid: [-8192, 8191] → [0, 32767]
 *   tanh:    [-4096, 4095] → [-32768, 32767]
 *
 * Direct Q20→Q15 path: sigmoid_q20_to_q15 / tanh_q20_to_q15
 *   avoid float round-trip entirely.
 */

static const int16_t sigmoid_lut_q15[512] = {
      11,    11,    12,    12,    12,    13,    13,    14,    14,    15,    15,    16,    16,    17,    17,    18,
      18,    19,    19,    20,    21,    21,    22,    23,    23,    24,    25,    26,    26,    27,    28,    29,
      30,    31,    32,    33,    34,    35,    36,    37,    38,    40,    41,    42,    44,    45,    46,    48,
      49,    51,    53,    54,    56,    58,    60,    61,    63,    65,    67,    70,    72,    74,    76,    79,
      81,    84,    87,    89,    92,    95,    98,   101,   104,   108,   111,   115,   118,   122,   126,   130,
     134,   138,   143,   147,   152,   157,   162,   167,   172,   177,   183,   189,   195,   201,   207,   214,
     221,   228,   235,   242,   250,   258,   266,   274,   283,   292,   301,   310,   320,   330,   341,   351,
     362,   374,   386,   398,   410,   423,   436,   450,   464,   479,   494,   509,   525,   542,   558,   576,
     594,   612,   632,   651,   672,   692,   714,   736,   759,   783,   807,   832,   858,   884,   912,   940,
     969,   999,  1029,  1061,  1094,  1127,  1162,  1197,  1234,  1272,  1311,  1351,  1392,  1434,  1478,  1522,
    1569,  1616,  1665,  1715,  1767,  1820,  1874,  1930,  1988,  2047,  2108,  2171,  2235,  2301,  2369,  2439,
    2511,  2584,  2660,  2737,  2817,  2898,  2982,  3068,  3156,  3247,  3340,  3435,  3532,  3632,  3734,  3839,
    3947,  4057,  4169,  4284,  4402,  4523,  4647,  4773,  4902,  5034,  5169,  5307,  5447,  5591,  5738,  5887,
    6040,  6196,  6355,  6517,  6682,  6850,  7021,  7195,  7373,  7553,  7737,  7923,  8113,  8305,  8501,  8700,
    8901,  9106,  9313,  9523,  9736,  9952, 10170, 10391, 10614, 10840, 11068, 11299, 11532, 11767, 12004, 12243,
   12484, 12727, 12972, 13218, 13466, 13715, 13965, 14217, 14469, 14722, 14977, 15232, 15487, 15743, 15999, 16256,
   16512, 16769, 17025, 17281, 17536, 17791, 18046, 18299, 18551, 18803, 19053, 19302, 19550, 19796, 20041, 20284,
   20525, 20764, 21001, 21236, 21469, 21700, 21928, 22154, 22377, 22598, 22816, 23032, 23245, 23455, 23662, 23867,
   24068, 24267, 24463, 24655, 24845, 25031, 25215, 25395, 25573, 25747, 25918, 26086, 26251, 26413, 26572, 26728,
   26881, 27030, 27177, 27321, 27461, 27599, 27734, 27866, 27995, 28121, 28245, 28366, 28484, 28599, 28711, 28821,
   28929, 29034, 29136, 29236, 29333, 29428, 29521, 29612, 29700, 29786, 29870, 29951, 30031, 30108, 30184, 30257,
   30329, 30399, 30467, 30533, 30597, 30660, 30721, 30780, 30838, 30894, 30948, 31001, 31053, 31103, 31152, 31199,
   31246, 31290, 31334, 31376, 31417, 31457, 31496, 31534, 31571, 31606, 31641, 31674, 31707, 31739, 31769, 31799,
   31828, 31856, 31884, 31910, 31936, 31961, 31985, 32009, 32032, 32054, 32076, 32096, 32117, 32136, 32156, 32174,
   32192, 32210, 32226, 32243, 32259, 32274, 32289, 32304, 32318, 32332, 32345, 32358, 32370, 32382, 32394, 32406,
   32417, 32427, 32438, 32448, 32458, 32467, 32476, 32485, 32494, 32502, 32510, 32518, 32526, 32533, 32540, 32547,
   32554, 32561, 32567, 32573, 32579, 32585, 32591, 32596, 32601, 32606, 32611, 32616, 32621, 32625, 32630, 32634,
   32638, 32642, 32646, 32650, 32653, 32657, 32660, 32664, 32667, 32670, 32673, 32676, 32679, 32681, 32684, 32687,
   32689, 32692, 32694, 32696, 32698, 32701, 32703, 32705, 32707, 32708, 32710, 32712, 32714, 32715, 32717, 32719,
   32720, 32722, 32723, 32724, 32726, 32727, 32728, 32730, 32731, 32732, 32733, 32734, 32735, 32736, 32737, 32738,
   32739, 32740, 32741, 32742, 32742, 32743, 32744, 32745, 32745, 32746, 32747, 32747, 32748, 32749, 32749, 32750,
   32750, 32751, 32751, 32752, 32752, 32753, 32753, 32754, 32754, 32755, 32755, 32756, 32756, 32756, 32757, 32757
};

static const int16_t tanh_lut_q15[512] = {
  -32746,-32745,-32745,-32744,-32743,-32742,-32741,-32741,-32740,-32739,-32738,-32737,-32736,-32735,-32734,-32733,
  -32732,-32731,-32729,-32728,-32727,-32726,-32724,-32723,-32721,-32720,-32718,-32717,-32715,-32714,-32712,-32710,
  -32708,-32706,-32704,-32702,-32700,-32698,-32696,-32694,-32691,-32689,-32686,-32684,-32681,-32678,-32675,-32672,
  -32669,-32666,-32663,-32660,-32656,-32653,-32649,-32645,-32641,-32637,-32633,-32629,-32624,-32620,-32615,-32610,
  -32605,-32600,-32595,-32589,-32584,-32578,-32572,-32566,-32559,-32553,-32546,-32539,-32531,-32524,-32516,-32508,
  -32500,-32491,-32483,-32474,-32464,-32455,-32445,-32435,-32424,-32413,-32402,-32390,-32378,-32366,-32353,-32340,
  -32327,-32313,-32299,-32284,-32268,-32253,-32236,-32220,-32202,-32184,-32166,-32147,-32128,-32107,-32087,-32065,
  -32043,-32020,-31997,-31973,-31948,-31922,-31895,-31868,-31840,-31811,-31781,-31750,-31718,-31685,-31651,-31616,
  -31580,-31543,-31505,-31465,-31425,-31383,-31340,-31296,-31250,-31203,-31154,-31104,-31053,-31000,-30945,-30889,
  -30831,-30771,-30709,-30646,-30581,-30513,-30444,-30373,-30300,-30224,-30147,-30067,-29984,-29900,-29813,-29723,
  -29631,-29536,-29438,-29338,-29235,-29129,-29020,-28907,-28792,-28673,-28552,-28426,-28298,-28165,-28030,-27890,
  -27747,-27600,-27449,-27294,-27135,-26971,-26804,-26632,-26455,-26274,-26089,-25899,-25704,-25504,-25299,-25090,
  -24875,-24655,-24430,-24199,-23963,-23722,-23475,-23222,-22964,-22700,-22430,-22155,-21873,-21586,-21293,-20993,
  -20688,-20376,-20058,-19735,-19405,-19068,-18726,-18377,-18023,-17662,-17295,-16922,-16542,-16157,-15766,-15369,
  -14966,-14557,-14142,-13722,-13296,-12865,-12428,-11986,-11540,-11088,-10631,-10170, -9704, -9234, -8759, -8281,
   -7799, -7313, -6824, -6332, -5837, -5339, -4838, -4335, -3830, -3323, -2815, -2305, -1794, -1282,  -769,  -256,
     256,   769,  1282,  1794,  2305,  2815,  3323,  3830,  4335,  4838,  5339,  5837,  6332,  6824,  7313,  7799,
    8281,  8759,  9234,  9704, 10170, 10631, 11088, 11540, 11986, 12428, 12865, 13296, 13722, 14142, 14557, 14966,
   15369, 15766, 16157, 16542, 16922, 17295, 17662, 18023, 18377, 18726, 19068, 19405, 19735, 20058, 20376, 20688,
   20993, 21293, 21586, 21873, 22155, 22430, 22700, 22964, 23222, 23475, 23722, 23963, 24199, 24430, 24655, 24875,
   25090, 25299, 25504, 25704, 25899, 26089, 26274, 26455, 26632, 26804, 26971, 27135, 27294, 27449, 27600, 27747,
   27890, 28030, 28165, 28298, 28426, 28552, 28673, 28792, 28907, 29020, 29129, 29235, 29338, 29438, 29536, 29631,
   29723, 29813, 29900, 29984, 30067, 30147, 30224, 30300, 30373, 30444, 30513, 30581, 30646, 30709, 30771, 30831,
   30889, 30945, 31000, 31053, 31104, 31154, 31203, 31250, 31296, 31340, 31383, 31425, 31465, 31505, 31543, 31580,
   31616, 31651, 31685, 31718, 31750, 31781, 31811, 31840, 31868, 31895, 31922, 31948, 31973, 31997, 32020, 32043,
   32065, 32087, 32107, 32128, 32147, 32166, 32184, 32202, 32220, 32236, 32253, 32268, 32284, 32299, 32313, 32327,
   32340, 32353, 32366, 32378, 32390, 32402, 32413, 32424, 32435, 32445, 32455, 32464, 32474, 32483, 32491, 32500,
   32508, 32516, 32524, 32531, 32539, 32546, 32553, 32559, 32566, 32572, 32578, 32584, 32589, 32595, 32600, 32605,
   32610, 32615, 32620, 32624, 32629, 32633, 32637, 32641, 32645, 32649, 32653, 32656, 32660, 32663, 32666, 32669,
   32672, 32675, 32678, 32681, 32684, 32686, 32689, 32691, 32694, 32696, 32698, 32700, 32702, 32704, 32706, 32708,
   32710, 32712, 32714, 32715, 32717, 32718, 32720, 32721, 32723, 32724, 32726, 32727, 32728, 32729, 32731, 32732,
   32733, 32734, 32735, 32736, 32737, 32738, 32739, 32740, 32741, 32741, 32742, 32743, 32744, 32745, 32745, 32746
};

/* LUT dimensions */
#define SIG_LUT_SIZE   512
#define SIG_Q10_MIN   (-8192)
#define SIG_Q10_MAX    8192
#define SIG_Q10_RANGE  16384
#define SIG_Q10_SHIFT  14

#define TANH_LUT_SIZE   512
#define TANH_Q10_MIN   (-4096)
#define TANH_Q10_MAX    4096
#define TANH_Q10_RANGE  8192
#define TANH_Q10_SHIFT  13

/* Q10 → Q15 LUT lookup with linear interpolation. Pure integer. */
static inline int16_t sigmoid_q15(int32_t q10) {
    if (q10 <= SIG_Q10_MIN) return 0;
    if (q10 >= SIG_Q10_MAX) return 32767;
    int32_t pos = (q10 - SIG_Q10_MIN) * (SIG_LUT_SIZE - 1);
    int32_t idx = pos >> SIG_Q10_SHIFT;
    int32_t frac = pos & (SIG_Q10_RANGE - 1);
    int64_t interp = (int64_t)sigmoid_lut_q15[idx] * (SIG_Q10_RANGE - frac)
                   + (int64_t)sigmoid_lut_q15[idx + 1] * frac;
    return (int16_t)((interp + (SIG_Q10_RANGE >> 1)) >> SIG_Q10_SHIFT);
}

static inline int16_t tanh_q15(int32_t q10) {
    if (q10 <= TANH_Q10_MIN) return -32768;
    if (q10 >= TANH_Q10_MAX) return 32767;
    int32_t pos = (q10 - TANH_Q10_MIN) * (TANH_LUT_SIZE - 1);
    int32_t idx = pos >> TANH_Q10_SHIFT;
    int32_t frac = pos & (TANH_Q10_RANGE - 1);
    int64_t interp = (int64_t)tanh_lut_q15[idx] * (TANH_Q10_RANGE - frac)
                   + (int64_t)tanh_lut_q15[idx + 1] * frac;
    return (int16_t)((interp + (TANH_Q10_RANGE >> 1)) >> TANH_Q10_SHIFT);
}

/* Direct Q20→Q15: skip float entirely. x in s32f20, output in Q15. */
static inline uint16_t sigmoid_q20_to_q15(int32_t q20) {
    /* Q20→Q10: >>10 (exact, no rounding needed — LUT absorbs 1-low-bit error) */
    return (uint16_t)sigmoid_q15(q20 >> 10);
}

static inline int16_t tanh_q20_to_q15(int32_t q20) {
    return tanh_q15(q20 >> 10);
}

/* Drop-in float wrappers (keep existing call sites working).
 * Float→Q10→LUT→Q15→float roundtrip. Prefer direct Q20→Q15 path. */
static inline float sigmoidf_fp(float x) {
    if (x >= 8.0f) return 1.0f;
    if (x <= -8.0f) return 0.0f;
    return (float)sigmoid_q15((int32_t)(x * 1024.0f)) * (1.0f / 32768.0f);
}

static inline float tanhf_fp(float x) {
    if (x >= 4.0f) return 1.0f;
    if (x <= -4.0f) return -1.0f;
    return (float)tanh_q15((int32_t)(x * 1024.0f)) * (1.0f / 32768.0f);
}

/* ================================================================
 * Dimension Constants
 * ================================================================ */

#define N_FFT       512
#define WIN_LEN     512
#define WIN_INC     256
#define N_BINS      257
#define N_BINS_BM   129
#define N_BINS_MID  65
#define N_BINS_SMALL 33

/* UL-UNAS specific channel dimensions */
#define CH_IN       1     /* single log-mag channel */
#define CH_XCONV    12    /* XConv output channels */
#define CH_XMB0     24    /* XMB0 / XDWS0 channels */
#define CH_XMB1     32    /* XMB1 channels */
#define CH_XDWS1    16    /* XDWS1 / GDPRNN channels */
#define CH_OUT      2     /* final CRM output (I, Q) */

/* cTFA GRU hidden dimensions */
#define CTA_XCONV_HID   24  /* XConv cTFA_ta GRU hidden */
#define CTA_XMB0_HID    48  /* XMB0 / XDWS0 cTFA_ta GRU hidden */
#define CTA_XMB1_HID    64  /* XMB1 cTFA_ta GRU hidden */
#define CTA_XDWS1_HID   32  /* XDWS1 cTFA_ta GRU hidden */
#define CTA_FA_HID      4   /* cTFA_fa BiGRU hidden (all modules) */
#define CTA_FA_SEG      17  /* cTFA_fa segments (ceil(68/4)=17) */

/* Intra/Inter RNN dimensions */
#define INTRA_GRU_HID   4   /* Intra-RNN BiGRU hidden per group */
#define INTER_GRU_HID   8   /* Inter-RNN GRU hidden per group */
#define INTRA_FC_DIM    8   /* Intra-RNN FC output dim */
#define INTER_FC_DIM    8   /* Inter-RNN FC output dim */

/* Decoder specific */
#define CH_DEC_XDWS0    64  /* De_XDWS0 cTFA_ta hidden (= XMB1 hid) */
#define CH_DEC_XMB0     48  /* De_XMB0 / De_XDWS1 cTFA_ta hidden */
#define CH_DEC_XMB1     24  /* De_XMB1 cTFA_ta hidden */
#define CH_DEC_XCONV    2   /* De_XConv cTFA_ta hidden */

/* TConv cache dimensions */
#define CACHE_XCONV_ROWS    2   /* XConv: stores 2 prev frames, H_in=3 */
#define CACHE_XMB0_ROWS     1   /* XMB0: stores 1 prev frame, H_in=2 */
#define CACHE_XDWS0_ROWS    1   /* XDWS0: stores 1 prev frame, H_in=2 */
#define CACHE_DEXDWS1_ROWS  1   /* De_XDWS1: stores 1 prev frame, H_in=2 */
#define CACHE_DEXMB1_ROWS   1   /* De_XMB1: stores 1 prev frame, H_in=2 */
#define CACHE_DEXCONV_ROWS  2   /* De_XConv: stores 2 prev frames, H_in=3 */

/* ================================================================
 * Model State (extern — allocated by user)
 * ================================================================ */

typedef struct {
    /* ---- Encoder Conv caches ---- */
    /* XConv: TConv cache (2 rows, 1 chan, 129 freq) = 258 */
    int32_t enc_xconv_cache[CACHE_XCONV_ROWS * 1 * N_BINS_BM];
    /* XMB0: TConv cache (1 row, 24 ch, 65 freq) = 1560 */
    int32_t enc_xmb0_cache[1 * CH_XMB0 * N_BINS_MID];
    /* XDWS0: TConv cache (1 row, 24 ch, 33 freq) = 792 */
    int32_t enc_xdws0_cache[1 * CH_XMB0 * N_BINS_SMALL];

    /* ---- Encoder cTFA_ta GRU hidden states ---- */
    int16_t enc_xconv_ta_h[CTA_XCONV_HID];       /* (24,) */
    int16_t enc_xmb0_ta_h[CTA_XMB0_HID];          /* (48,) */
    int16_t enc_xdws0_ta_h[CTA_XMB0_HID];         /* (48,) */
    int16_t enc_xmb1_ta_h[CTA_XMB1_HID];          /* (64,) */
    int16_t enc_xdws1_ta_h[CTA_XDWS1_HID];        /* (32,) */

    /* ---- Decoder cTFA_ta GRU hidden states ---- */
    int16_t dec_xdws0_ta_h[CH_DEC_XDWS0];         /* (64,) */
    int16_t dec_xmb0_ta_h[CH_DEC_XMB0];           /* (48,) */
    int16_t dec_xdws1_ta_h[CH_DEC_XMB0];          /* (48,) */
    int16_t dec_xmb1_ta_h[CH_DEC_XMB1];           /* (24,) */
    int16_t dec_xconv_ta_h[CH_DEC_XCONV];         /* (2,) */

    /* ---- Decoder Conv caches ---- */
    /* De_XDWS1: GTConv cache (1 row, 24 ch, 33 freq) = 792 */
    int32_t dec_xdws1_cache[1 * CH_XMB0 * N_BINS_SMALL];
    /* De_XMB1: GTConv cache (1 row, 12 ch, 33 freq) = 396 */
    int32_t dec_xmb1_cache[1 * CH_XCONV * N_BINS_SMALL];
    /* De_XConv: TConv cache (2 rows, 12 ch, 65 freq) = 1560 */
    int32_t dec_xconv_cache[CACHE_DEXCONV_ROWS * CH_XCONV * N_BINS_MID];

    /* ---- Inter-RNN hidden states ---- */
    int16_t inter_prev0[N_BINS_SMALL * CH_XDWS1];  /* (33, 16) */
    int16_t inter_prev1[N_BINS_SMALL * CH_XDWS1];  /* (33, 16) */

} ulunas_state_t;

/* ================================================================
 * Function Prototypes — Basic Ops
 * ================================================================ */

/* conv2d: (Cin, Win) → (Cout, Wout), weight (Cout, Cin, Hk, Wk) */
void conv2d_fixed(const int32_t *x, int Cin, int Win,
                  const int16_t *weight, const int32_t *bias,
                  int Cout, int Wout, int Hk, int Wk,
                  int stride, int pad_w, int qr,
                  int32_t *y);

/* tconv2d: transposed conv2d */
void tconv2d_fixed(const int32_t *x, int Cin, int Win,
                   const int16_t *weight, const int32_t *bias,
                   int Cout, int Wout, int Hk, int Wk,
                   int stride, int qr,
                   int32_t *y);

/* pconv2d: point-wise conv2d (1×1), weights (Cout, Cin, 1, 1) */
void pconv2d_fixed(const int32_t *x, int Cin, int Win,
                   const int16_t *weight, const int32_t *bias,
                   int Cout, int stride, int qr,
                   int32_t *y);

/* ptconv2d: point-wise transposed conv2d (1×1) */
void ptconv2d_fixed(const int32_t *x, int Cin, int Win,
                    const int16_t *weight, const int32_t *bias,
                    int Cout, int qr,
                    int32_t *y);

/* gconv2d: grouped conv2d (Cin/groups channels per group) */
void gconv2d_fixed(const int32_t *x, int Cin, int Win,
                   const int16_t *weight, const int32_t *bias,
                   int Cout, int Wout, int Hk, int Wk,
                   int stride, int pad_w, int groups, int qr,
                   int32_t *y);

/* gtconv2d: grouped transposed conv2d */
void gtconv2d_fixed(const int32_t *x, int Cin, int Win,
                    const int16_t *weight, const int32_t *bias,
                    int Cout, int Wout, int Hk, int Wk,
                    int stride, int groups, int qr,
                    int32_t *y);

/* non_gconv2d: grouped conv2d without dilation (for nonTConv blocks) */
void non_gconv2d_fixed(const int32_t *x, int Cin, int Win,
                       const int16_t *weight, const int32_t *bias,
                       int Cout, int Wout, int Hk, int Wk,
                       int stride, int pad_h, int pad_w, int groups, int qr,
                       int32_t *y);

/* non_gtconv2d: grouped transposed conv2d without dilation */
void non_gtconv2d_fixed(const int32_t *x, int Cin, int Win,
                        const int16_t *weight, const int32_t *bias,
                        int Cout, int Wout, int Hk, int Wk,
                        int stride, int pad_h, int pad_w, int groups, int qr,
                        int32_t *y);

/* BatchNorm: y = ((x - mean) * var_inv)>>qr1 * weight>>qr2 + bias */
void bn_fixed(int32_t *x, int C, int Win,
              const uint16_t *weight, const int32_t *bias,
              const int32_t *running_mean, const uint16_t *running_var,
              int qr1, int qr2);

/* AffinePReLU: affine transform + PReLU */
void affine_prelu_fixed(int32_t *x, int C, int Win,
                        const int16_t *affine_weight, const int32_t *affine_bias,
                        const int16_t *slope_weight,
                        int qr_affine, int qr_slope);

/* PReLU: y(x<0) = x * weight >> qr */
void prelu_fixed(int32_t *x, int C, int Win,
                 const int16_t *weight, int qr);

/* LayerNorm: computed online (mean/var from data) */
void ln_fixed(int32_t *x, int C, int Win,
              const int16_t *weight, const int32_t *bias, int qr);

/* GRU internal diagnostics */
void gru_diag_start(const char *filename);
void gru_diag_stop(void);

/* GRU: single time-step, (input_dim) → (hidden_dim)
 * ⚠️ biases are int32_t: cTFA stores s16f10 (cast needed), DPRNN stores s32f20 */
void gru_step_fixed(const int32_t *x_t, int input_dim, int hidden_dim,
                    const int16_t *ih_weight, const int32_t *ih_bias,
                    const int16_t *hh_weight, const int32_t *hh_bias,
                    int16_t *y, int16_t *h_prev, int qr1, int qr2);

/* GRU: full sequence (seq_len, input_dim) with shared h_prev */
void gru_sequence_fixed(const int32_t *x, int seq_len, int input_dim, int hidden_dim,
                        const int16_t *ih_weight, const int32_t *ih_bias,
                        const int16_t *hh_weight, const int32_t *hh_bias,
                        int16_t *y, int16_t *h_prev, int qr1, int qr2);

/* BiGRU: bidirectional GRU for (seq_len, input_dim) */
void bigru_fixed(const int32_t *x, int seq_len, int input_dim, int hidden_dim,
                 const int16_t *ih_w, const int32_t *ih_b,
                 const int16_t *hh_w, const int32_t *hh_b,
                 const int16_t *re_ih_w, const int32_t *re_ih_b,
                 const int16_t *re_hh_w, const int32_t *re_hh_b,
                 int16_t *y, int qr1, int qr2);

/* ================================================================
 * Function Prototypes — UL-UNAS Specific Ops
 * ================================================================ */

/* log_gen: log10 magnitude compression, (1, 257) → s32f20 */
void log_gen_fixed(const float *real_in, const float *imag_in,
                   int N, int32_t *y);

/* BM: Band Merging, (1, 257) → (1, 129), ERB filterbank */
void BM_fixed(const int32_t *x, const uint16_t *weight, int32_t *y);

/* BS: Band Splitting, (2, 129) → (2, 257), inverse ERB */
void BS_fixed(const int16_t *x, const uint16_t *weight, int16_t *y);

/* MASK: Complex Ratio Mask, s16f15 mask × s32f20 spec → output */
void MASK_fixed(const int16_t *mask, const int32_t *real_in,
                const int32_t *imag_in, int32_t *y);

/* sigmoid_fixed: sigmoid on s32f20 → u16f15 */
void sigmoid_fixed(const int32_t *x, int N, uint16_t *y);

/* ================================================================
 * Function Prototypes — cTFA Attention
 * ================================================================ */

/* cTFA_ta: Time-axis channel-wise attention (GRU single-step)
 * ⚠️ y is uint16_t[hidden_dim] — u16f15 gate values, NOT int32!
 * biases are int32_t (Q20 values stored as int32 in .mat files) */
void cTFA_ta_module(const int32_t *x, int C, int W,
                    int hidden_dim,
                    const int16_t *ih_w, const int32_t *ih_b,
                    const int16_t *hh_w, const int32_t *hh_b,
                    const int16_t *fc_w, const int32_t *fc_b,
                    int16_t *h_prev, uint16_t *y,
                    int qr1, int qr2, int fc_shift);

/* cTFA_fa: Frequency-axis channel-wise attention (BiGRU per-segment)
 * ⚠️ Qr values VARY per module! Enc:-13,-8  De0/1:-12,-7  De3:-11,-6 */
void cTFA_fa_module(const int32_t *x, int C, int W,
                    const int16_t *ih_w, const int32_t *ih_b,
                    const int16_t *hh_w, const int32_t *hh_b,
                    const int16_t *re_ih_w, const int32_t *re_ih_b,
                    const int16_t *re_hh_w, const int32_t *re_hh_b,
                    const int16_t *fc_w, const int32_t *fc_b,
                    int32_t *y,
                    int qr1, int qr2, int fc_shift);

/* cTFA apply: y = (x * ta_gate >> 15) * fa_gate >> 15
 * ta_gate: u16f15, fa_gate: u16f15 stored as int32 */
void cTFA_apply(const int32_t *x, const uint16_t *ta_gate,
                const int32_t *fa_gate, int C, int W,
                int32_t *y);

/* ================================================================
 * Function Prototypes — Encoder Sub-modules
 * ================================================================ */

/* TConv block: Temporal Conv2d + BN + AffinePReLU
 * conv_qr: QR for conv (varies! -14 for XConv/XMB0, -13 for XDWS0)
 * groups: >1 for grouped conv (XMB0=12, XDWS0=12, etc.) */
void TConv_block(const int32_t *x, int32_t *conv_cache,
                 int Cin, int Cout, int Win, int Wout,
                 int Hk, int Wk, int stride_h, int stride_w,
                 int cache_rows, int conv_qr, int groups,
                 int bn_qr1, int bn_qr2,
                 const int16_t *conv_w, const int32_t *conv_b,
                 const uint16_t *bn_w, const int32_t *bn_b,
                 const int32_t *bn_mean, const uint16_t *bn_var,
                 const int16_t *affine_w, const int32_t *affine_b,
                 const int16_t *slope_w,
                 int32_t *y);

/* PConv block: Pointwise Conv1x1 + BN + AffinePReLU */
void PConv_block(const int32_t *x, int Cin, int Cout, int Win,
                 const int16_t *conv_w, const int32_t *conv_b,
                 const uint16_t *bn_w, const int32_t *bn_b,
                 const int32_t *bn_mean, const uint16_t *bn_var,
                 const int16_t *affine_w, const int32_t *affine_b,
                 const int16_t *slope_w,
                 int32_t *y);

/* nonTConv block: Non-dilated grouped conv2d + BN + AffinePReLU
 * conv_qr, bn_qr1, bn_qr2 now parameterized (were hardcoded -13, -14, -14) */
void nonTConv_block(const int32_t *x, int Cin, int Cout, int Win, int Wout,
                    int Hk, int Wk, int stride, int groups,
                    int conv_qr, int bn_qr1, int bn_qr2,
                    const int16_t *conv_w, const int32_t *conv_b,
                    const uint16_t *bn_w, const int32_t *bn_b,
                    const int32_t *bn_mean, const uint16_t *bn_var,
                    const int16_t *affine_w, const int32_t *affine_b,
                    const int16_t *slope_w,
                    int32_t *y);

/* Shuffle: interleave deinterleave */
void shuffle_interleave(const int32_t *x, int half_C, int W, int32_t *y);
void shuffle_deinterleave(const int32_t *x, int half_C, int W, int32_t *y);

/* XConv module */
void XConv_module(const int32_t *x, int32_t *conv_cache, int16_t *ta_h,
                  int32_t *y);

/* XMB0 module */
void XMB0_module(const int32_t *x, int32_t *conv_cache, int16_t *ta_h,
                 int32_t *y);

/* XDWS0 module */
void XDWS0_module(const int32_t *x, int32_t *conv_cache, int16_t *ta_h,
                  int32_t *y);

/* XMB1 module */
void XMB1_module(const int32_t *x, int16_t *ta_h, int32_t *y);

/* XDWS1 module */
void XDWS1_module(const int32_t *x, int16_t *ta_h, int32_t *y);

/* ================================================================
 * Function Prototypes — Decoder Sub-modules
 * ================================================================ */

/* nonTConv_block (transposed): grouped transposed conv + BN + AffinePReLU
 * conv_qr, bn_qr1, bn_qr2 now parameterized (were hardcoded -13, -14, -14) */
void nonGTConv_block(const int32_t *x, int Cin, int Cout, int Win, int Wout,
                     int Hk, int Wk, int stride, int groups,
                     int conv_qr, int bn_qr1, int bn_qr2,
                     const int16_t *conv_w, const int32_t *conv_b,
                     const uint16_t *bn_w, const int32_t *bn_b,
                     const int32_t *bn_mean, const uint16_t *bn_var,
                     const int16_t *affine_w, const int32_t *affine_b,
                     const int16_t *slope_w,
                     int32_t *y);

/* GTConv_block: grouped transposed conv block (for Decoder TConv blocks) */
void GTConv_block(const int32_t *x, int32_t *conv_cache,
                  int Cin, int Cout, int Win, int Wout,
                  int Hk, int Wk, int stride_h, int stride_w,
                  int cache_rows, int conv_qr, int groups,
                  int bn_qr1, int bn_qr2,
                  const int16_t *conv_w, const int32_t *conv_b,
                  const uint16_t *bn_w, const int32_t *bn_b,
                  const int32_t *bn_mean, const uint16_t *bn_var,
                  const int16_t *affine_w, const int32_t *affine_b,
                  const int16_t *slope_w,
                  int32_t *y);

/* De_XDWS0 module */
void De_XDWS0_module(const int32_t *x, const int32_t *x_skip,
                     int16_t *ta_h, int32_t *y);

/* De_XMB0 module */
void De_XMB0_module(const int32_t *x, const int32_t *x_skip,
                    int16_t *ta_h, int32_t *y);

/* De_XDWS1 module */
void De_XDWS1_module(const int32_t *x, const int32_t *x_skip,
                     int32_t *conv_cache, int16_t *ta_h, int32_t *y);

/* De_XMB1 module */
void De_XMB1_module(const int32_t *x, const int32_t *x_skip,
                    int32_t *conv_cache, int16_t *ta_h, int32_t *y);

/* De_XConv module */
void De_XConv_module(const int32_t *x, const int32_t *x_skip,
                     int32_t *conv_cache, int16_t *ta_h, int32_t *y);

/* ================================================================
 * Function Prototypes — DPRNN
 * ================================================================ */

/* Intra_RNN: BiGRU(2 groups) + FC + LN + residual */
void Intra_RNN_module(const int32_t *x, int gdprnn_idx, int32_t *y);

/* Inter_RNN: GRU(2 groups, stateful) + FC + LN + residual */
void Inter_RNN_module(const int32_t *x, int16_t *h_prev, int gdprnn_idx, int32_t *y);

/* GDPRNN: Intra-RNN → Inter-RNN */
void GDPRNN_module(const int32_t *x, int16_t *inter_prev, int gdprnn_idx, int32_t *y);

/* ================================================================
 * Function Prototypes — High-Level Pipeline
 * ================================================================ */

/* Encoder: XConv → XMB0 → XDWS0 → XMB1 → XDWS1 */
void Encoder_module(const int32_t *x, ulunas_state_t *state,
                    int32_t *y_e0, int32_t *y_e1, int32_t *y_e2,
                    int32_t *y_e3, int32_t *y_e4);

/* Decoder: De_XDWS0 → De_XMB0 → De_XDWS1 → De_XMB1 → De_XConv */
void Decoder_module(const int32_t *x, ulunas_state_t *state,
                    const int32_t *y_e0, const int32_t *y_e1,
                    const int32_t *y_e2, const int32_t *y_e3,
                    const int32_t *y_e4,
                    int32_t *y);

/* Main inference: single-frame processing */
void ulunas_infer_frame(const float *real_in, const float *imag_in,
                        ulunas_state_t *state,
                        const uint16_t *erbfc_w, const uint16_t *ierbfc_w,
                        int32_t *crm_out);

/* Initialize state to zero */
void ulunas_state_init(ulunas_state_t *state);

#ifdef __cplusplus
}
#endif

#endif /* ULUNAS_FP_H */
