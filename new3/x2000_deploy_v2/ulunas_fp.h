/**
 * ulunas_fp.h — UL-UNAS MATLAB→C Fixed-Point Inference
 * ======================================================
 * Pure fixed-point implementation using int16_t/int32_t/int64_t.
 * No float/double in convolution, BN, GRU computation paths.
 *
 * Q-format (matching MATLAB Fix_point.m exactly):
 *   Q_ACT  = 20: s32f20 — main activations, conv I/O, BN intermediates
 *   Q_LOG  = 20: u32f20 — log_gen output, cTFA aggregation
 *   Q_GRU_H = 15: s16f15 — GRU/BiGRU hidden state, tanh output
 *   Q_SIG  = 15: u16f15 — sigmoid output, cTFA attention masks
 *   Q_LN_VAR = 11: u16f11 — LN 1/sqrt(var)
 *
 * Target: Ingenic X2000 (MIPS32R2, no FPU) + PC gcc verification
 *
 * Data layout: row-major (C, W) for 2D tensors
 *   Indexing: data[c * W + w]
 *   3D tensors (C, H, W): data[c * H * W + h * W + w]
 */

#ifndef ULUNAS_FP_H
#define ULUNAS_FP_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Q-format Constants
 * ================================================================ */
#define Q_ACT           20    /* s32f20: main activations */
#define Q_LOG           20    /* u32f20: log output, cTFA agg */
#define Q_GRU_H         15    /* s16f15: GRU hidden state, tanh */
#define Q_SIG           15    /* u16f15: sigmoid output */
#define Q_LN_VAR        11    /* u16f11: LN 1/sqrt(var) */

/* Weight Q-formats (from para_in_mat_FP/) */
#define Q_WGT_CONV      14    /* s16f14: Conv/DeConv weights */
#define Q_WGT_PCONV     13    /* s16f13: PConv weights */
#define Q_WGT_GCONV     13    /* s16f13: GConv/nonGConv/TConv weights */
#define Q_WGT_GRU       12    /* s16f12: GRU ih/hh weights */
#define Q_WGT_GRU_BIAS  10    /* s16f10: GRU bias */
#define Q_WGT_FC        13    /* s16f13: FC weights */
#define Q_WGT_FC_BIAS   20    /* s32f20: FC bias */
#define Q_WGT_BN_W      14    /* s16f14/u16f14: BN weight */
#define Q_WGT_BN_B      20    /* s32f20: BN bias */
#define Q_WGT_BN_M      20    /* s32f20: BN running_mean */
#define Q_WGT_BN_V      14    /* u16f14: BN running_var (varies per block) */
#define Q_WGT_LN        12    /* s16f12: LN weight/bias */
#define Q_WGT_AP_W      14    /* s16f14: AffinePReLU weight */
#define Q_WGT_AP_B      20    /* s32f20: AffinePReLU bias */
#define Q_WGT_AP_S      13    /* s16f13: AffinePReLU slope */
#define Q_WGT_TA_FC     13    /* s16f13: cTFA TA FC weight */
#define Q_WGT_TA_FC_B   20    /* s32f20: cTFA TA FC bias */
#define Q_WGT_FA_FC     13    /* s16f13: cTFA FA FC weight */
#define Q_WGT_FA_FC_B   20    /* s32f20: cTFA FA FC bias */
#define Q_WGT_DPRNN_FC  13    /* s16f13: DPRNN FC weight */
#define Q_WGT_DPRNN_FC_B 20   /* s32f20: DPRNN FC bias */
#define Q_WGT_ERB       15    /* u16f15: ERB merge/split weights */
#define Q_WGT_SIGMOID   15    /* u16f15: sigmoid/tanh output */

/* Scale constants */
#define Q20_SCALE       1048576   /* 2^20 */
#define Q15_SCALE       32768     /* 2^15 */
#define Q14_SCALE       16384     /* 2^14 */
#define Q13_SCALE       8192      /* 2^13 */
#define Q12_SCALE       4096      /* 2^12 */
#define Q11_SCALE       2048      /* 2^11 */
#define Q10_SCALE       1024      /* 2^10 */

#define Q15_ONE         32768     /* 1.0 in u16f15 */

/* ================================================================
 * Rounding helper: round(x / 2^N) with half-up rounding
 * ================================================================ */
static inline int32_t round_shr(int64_t x, int N) {
    int64_t half = ((int64_t)1 << (N - 1));
    if (x >= 0) return (int32_t)((x + half) >> N);
    else        return (int32_t)((x - half) >> N);
}

/* Saturate to int32_t */
static inline int32_t sat_i32(int64_t x) {
    if (x > INT32_MAX) return INT32_MAX;
    if (x < INT32_MIN) return INT32_MIN;
    return (int32_t)x;
}

/* Saturate to int16_t */
static inline int16_t sat_i16(int32_t x) {
    if (x > INT16_MAX) return INT16_MAX;
    if (x < INT16_MIN) return INT16_MIN;
    return (int16_t)x;
}

/* Saturate to uint16_t */
static inline uint16_t sat_u16(int32_t x) {
    if (x > UINT16_MAX) return UINT16_MAX;
    if (x < 0) return 0;
    return (uint16_t)x;
}

/* Clamp int32_t to int16_t range */
static inline int16_t clamp_i16(int32_t x) {
    if (x > 32767)  return 32767;
    if (x < -32768) return -32768;
    return (int16_t)x;
}

/* Clamp to uint16_t range */
static inline uint16_t clamp_u16(int32_t x) {
    if (x > 65535) return 65535;
    if (x < 0)     return 0;
    return (uint16_t)x;
}

static inline int32_t sat_s20(int64_t x) {
    if (x >  1073741824LL) return  1073741824;
    if (x < -1073741824LL) return -1073741824;
    return (int32_t)x;
}

/* ================================================================
 * LUT function declarations (defined in ulunas_lut.c)
 * ================================================================ */
uint16_t sigmoid_q20_to_q15(int32_t x_q20);
int16_t  tanh_q20_to_q15(int32_t x_q20);
int32_t  log10_q20(int32_t x_q20);
uint32_t sqrt_q40_to_q20(uint64_t x_q40);

uint32_t sigmoid_q20_to_q20(int32_t x_q20);
int32_t  tanh_q20_to_q20(int32_t x_q20);

/* ================================================================
 * Model State Structure — all frame-persistent caches
 * ================================================================ */
typedef struct {
    /* Temporal convolution caches (Encoder) */
    int32_t conv_cache_e0[2 * 129];
    int32_t conv_cache_e1[24 * 65];
    int32_t conv_cache_e2[24 * 33];

    /* Temporal convolution caches (Decoder) */
    int32_t conv_cache_d0[24 * 33];
    int32_t conv_cache_d1[12 * 33];
    int32_t conv_cache_d2[12 * 2 * 65];

    /* cTFA TA GRU hidden state caches (Encoder) */
    int32_t tfa_cache_e0[24];
    int32_t tfa_cache_e1[48];
    int32_t tfa_cache_e2[48];
    int32_t tfa_cache_e3[64];
    int32_t tfa_cache_e4[32];

    /* cTFA TA GRU hidden state caches (Decoder) -- int32_t Q20 */
    int32_t tfa_cache_d0[64];
    int32_t tfa_cache_d1[48];
    int32_t tfa_cache_d2[48];
    int32_t tfa_cache_d3[24];
    int32_t tfa_cache_d4[2];

    /* GDPRNN Inter-RNN hidden state caches -- int32_t Q20 */
    int32_t inter_cache_0[33 * 16];
    int32_t inter_cache_1[33 * 16];

} ulunas_state_t;

/* ================================================================
 * Initialization
 * ================================================================ */
void ulunas_state_init(ulunas_state_t *st);

/* ================================================================
 * Low-level Operator Declarations (defined in ulunas_fp.c)
 * ================================================================ */

/* --- Convolution Operators --- */

/**
 * conv2d_func: Standard 2D convolution
 * x[1][Wx], weight[Cout][Cin][Kh][Kw], bias[Cout]
 * Output: y[Cout][Wout] in Q20
 * Matching MATLAB: temp = round(x_kernel .* kernel_chan * 2^Qr);
 *                  conv_result = sum(temp, 'all');
 */
void conv2d_func(const int32_t *x, int Cin, int Cout, int Hout, int Wout,
                 int Kh, int Kw, int stride_h, int stride_w,
                 const int16_t *weight, const int32_t *bias, int Qr,
                 int32_t *y);

/**
 * pconv2d_func: Point-wise (1×1) convolution
 * x[Cin][W], weight[Cout][Cin][1][1], bias[Cout]
 * Output: y[Cout][W] in Q20
 */
void pconv2d_func(const int32_t *x, int Cin, int Cout, int Hout, int Wout,
                  const int16_t *weight, const int32_t *bias, int Qr,
                  int weight_stride,
                  int32_t *y);

/**
 * gconv2d_func: Grouped temporal conv (with cache concatenation)
 * x[Cout][Wx], cache[Cout][Wcache]
 * Input = [cache; x] concatenated along time dimension
 * weight[Cout][1][Kh][Kw], bias[Cout]
 * Output: y[Cout][Wout] in Q20
 * Updates: cache = x (for next frame)
 */
void gconv2d_func(const int32_t *x, int Cout, int Hout, int Wout,
                  int Kh, int Kw, int stride_h, int stride_w,
                  const int16_t *weight, const int32_t *bias, int Qr,
                  int32_t *cache, int32_t *y);

/**
 * non_gconv2d_func: Grouped non-temporal conv (no cache, freq-dim kernel)
 * x[Cout][Wx], weight[Cout][1][1][Kw] (kernel along frequency only)
 * Output: y[Cout][Wout] in Q20
 */
void non_gconv2d_func(const int32_t *x, int Cout, int Hout, int Wout,
                      int Kh, int Kw, int stride_h, int stride_w,
                      const int16_t *weight, const int32_t *bias, int Qr,
                      int32_t *y);

/**
 * tconv2d_func: Transposed 2D convolution (zero-insertion + rot90(kernel,2))
 * x[Cin][Hx][Wx], weight[Cin][Cout][Kh][Kw], bias[Cout]
 * Output: y[Cout][Wout] in Q20
 */
void tconv2d_func(const int32_t *x, int Cin, int Cout, int Hout, int Wout,
                  int Kh, int Kw, int stride_h, int stride_w,
                  const int16_t *weight, const int32_t *bias, int Qr,
                  int32_t *y);

/**
 * gtconv2d_func: Grouped transposed temporal conv (cache + zero-insertion + rot90(kernel,2))
 * x[Cout][Wx], cache[Cout][Wcache]
 * Input = [cache; x] concatenated, then zero-insertion
 * weight[Cout][1][Kh][Kw], bias[Cout]
 * Updates: cache = x
 */
void gtconv2d_func(const int32_t *x, int Cout, int Hout, int Wout,
                   int Kh, int Kw, int stride_h, int stride_w,
                   const int16_t *weight, const int32_t *bias, int Qr,
                   int32_t *cache, int32_t *y);

/**
 * non_gtconv2d_func: Grouped non-temporal transposed conv (zero-insertion + rot90(kernel,90))
 * No cache concatenation
 */
void non_gtconv2d_func(const int32_t *x, int Cout, int Hout, int Wout,
                       int Kh, int Kw, int stride_h, int stride_w,
                       const int16_t *weight, const int32_t *bias, int Qr,
                       int32_t *y);

/* --- Normalization Operators --- */

/**
 * bn_func: Batch Normalization
 * x[N], weight[N] (Q14), bias[N] (Q20), running_mean[N] (Q20), running_var[N] (Q11-Q14)
 * x_norm = round((x - mean) * var * 2^Qr1)
 * y = round(x_norm * weight * 2^Qr2) + bias
 */
void bn_func(const int32_t *x, const int16_t *weight, const int32_t *bias,
             const int32_t *running_mean, const uint16_t *running_var,
             int Qr1, int Qr2, int C, int N, int32_t *y);
/* bn_func with signed weight (int16_t) */
void bn_func_sw(const int32_t *x, const int16_t *weight, const int32_t *bias,
                const int32_t *running_mean, const uint16_t *running_var,
                int Qr1, int Qr2, int C, int N, int32_t *y);

/* bn_func with unsigned weight (uint16_t) */
void bn_func_uw(const int32_t *x, const uint16_t *weight, const int32_t *bias,
                const int32_t *running_mean, const uint16_t *running_var,
                int Qr1, int Qr2, int C, int N, int32_t *y);

/**
 * ln_func: Layer Normalization
 * Computes mean/var (may use float intermediate as MATLAB does x*2^-20 dequant)
 * x_norm = round((x - mean) * running_var * 2^(-11))
 * y = round(x_norm * weight * 2^Qr) + bias
 */
void ln_func(const int32_t *x, const int16_t *weight, const int32_t *bias,
             int Qr, int C, int N, int32_t *y);

/* --- Activation Operators --- */

/**
 * affineprelu_func: Affine PReLU
 * Negative part: x(idx) = round(x(idx) * slope(row) * 2^Qr1)
 * Affine + residual: y = round(x_copy * weight * 2^Qr2) + bias + x_neg_processed
 * Note: slope is per-channel (row)
 */
void affineprelu_func(const int32_t *x, const int16_t *weight, const int32_t *bias,
                      const int16_t *slope, int Qr1, int Qr2,
                      int C, int W, int32_t *y);

/* --- RNN Operators --- */

/**
 * gru_module: Single-direction GRU
 * x_t[1][in_dim] in Q20, h_cache[1][nHidden] in Q15
 * ih_weight[in_dim][3*nHidden] in Q12, ih_bias[3*nHidden] in Q10
 * hh_weight[nHidden][3*nHidden] in Q12, hh_bias[3*nHidden] in Q10
 * Qr1 = -13 (ih path), Qr2 = -8 (hh path)
 * Output: y[1][nHidden] in Q15, h_cache updated in Q15
 */
void gru_module(const int32_t *x_t, int nHidden, int in_dim,
                int16_t *h_cache,
                const int16_t *ih_weight, const int32_t *ih_bias,
                const int16_t *hh_weight, const int32_t *hh_bias,
                int Qr1, int Qr2,
                int16_t *y);

/**
 * bigru_module: Bidirectional GRU
 * Input x[T][in_dim] in Q20, output y[T][2*nHidden] in Q15
 * Forward: processes t=0..T-1
 * Backward: processes t=T-1..0, then flips
 * Concatenates: y = [y_forward, y_backward_flipped]
 */
void bigru_module(const int32_t *x, int T, int nHidden, int in_dim,
                  const int16_t *ih_weight, const int32_t *ih_bias,
                  const int16_t *hh_weight, const int32_t *hh_bias,
                  const int16_t *re_ih_weight, const int32_t *re_ih_bias,
                  const int16_t *re_hh_weight, const int32_t *re_hh_bias,
                  int Qr1, int Qr2,
                  int16_t *y);

/* Q20 GRU */
void gru_module_q20(const int32_t *x_t, int nHidden, int in_dim,
                    int32_t *h_cache,
                    const int16_t *ih_weight, const int32_t *ih_bias,
                    const int16_t *hh_weight, const int32_t *hh_bias,
                    int Qr1, int Qr2, int16_t *y);
void bigru_module_q20(const int32_t *x, int T, int nHidden, int in_dim,
                      const int16_t *ih_weight, const int32_t *ih_bias,
                      const int16_t *hh_weight, const int32_t *hh_bias,
                      const int16_t *re_ih_weight, const int32_t *re_ih_bias,
                      const int16_t *re_hh_weight, const int32_t *re_hh_bias,
                      int Qr1, int Qr2, int16_t *y);

/* --- cTFA Channel-wise Time-Frequency Attention --- */

/**
 * ctfa_ta_module: Time Attention
 * x[C][W] in Q20 → square → mean over W → GRU → FC → sigmoid → u16f15
 * Output: y[1][C] in Q15 (attention weights per channel)
 * h_cache[C_ta_nhid] in Q15 (GRU state, updated)
 */
void ctfa_ta_module(const int32_t *x, int C, int W, int nHidden,
                    int32_t *h_cache,
                    const int16_t *ih_weight, const int32_t *ih_bias,
                    const int16_t *hh_weight, const int32_t *hh_bias,
                    const int16_t *fc_weight, const int32_t *fc_bias,
                    int Qr1, int Qr2, int fc_Qr,
                    uint32_t *y);

/**
 * ctfa_fa_module: Frequency Attention
 * x[C][W] in Q20 → square → mean over C → pad → reshape → BiGRU → FC → reshape → sigmoid
 * Output: y[1][W] in Q20 (attention weights per frequency)
 */
void ctfa_fa_module(const int32_t *x, int C, int W, int nHidden,
                    int group, int seg, int pad_len,
                    const int16_t *ih_weight, const int32_t *ih_bias,
                    const int16_t *hh_weight, const int32_t *hh_bias,
                    const int16_t *re_ih_weight, const int32_t *re_ih_bias,
                    const int16_t *re_hh_weight, const int32_t *re_hh_bias,
                    const int16_t *fc_weight, const int32_t *fc_bias,
                    int Qr1, int Qr2, int fc_Qr,
                    uint32_t *y);

/* --- Shuffle Operators --- */

/**
 * shuffle_interleave: MATLAB pattern y(1:2:end) = x(1:N/2); y(2:2:end) = x(N/2+1:end)
 */
void shuffle_interleave(const int32_t *src, int C, int W, int32_t *dst);

/**
 * shuffle_deinterleave: Reverse of shuffle_interleave
 */
void shuffle_deinterleave(const int32_t *src, int C, int W, int32_t *dst);

/* --- Signal Processing --- */

/**
 * log_gen_fixed: Log-magnitude compression
 * Input: real[W] (Q20), imag[W] (Q20)
 * Output: log10(sqrt(real² + imag²)) in Q20
 */
void log_gen_fixed(const int32_t *real, const int32_t *imag, int W, int32_t *out);

/**
 * bm_fixed: ERB Band Merging
 * Low freq (0-2000Hz, bins 0-64): pass-through
 * High freq (2031-8000Hz, bins 65-256): merge to bins 65-128 via matrix multiply
 */
void bm_fixed(const int32_t *x, const uint16_t *weight, int W_in, int W_out, int32_t *y);

/**
 * bs_fixed: ERB Band Splitting (inverse of bm_fixed)
 */
void bs_fixed(const uint16_t *x, const uint16_t *weight, int W_in, int W_out, int16_t *y);

/**
 * mask_fixed: Apply mask to complex spectrum
 * y_real = round(x_real * mask * 2^(-15))
 * y_imag = round(x_imag * mask * 2^(-15))
 * Returns concatenated [y_real; y_imag] in Q20
 */
void mask_fixed(const int16_t *mask, const int32_t *x_real, const int32_t *x_imag,
                int W, int32_t *y);

/* ================================================================
 * Module-Level Function Declarations (defined in ulunas_modules.c)
 * ================================================================ */

/* --- Encoder --- */
void encoder_layer0_xconv(const int32_t *x, ulunas_state_t *st, int32_t *y);
void encoder_layer1_xmb0(const int32_t *x, ulunas_state_t *st, int32_t *y);
void encoder_layer2_xdws0(const int32_t *x, ulunas_state_t *st, int32_t *y);
void encoder_layer3_xmb1(const int32_t *x, ulunas_state_t *st, int32_t *y);
void encoder_layer4_xdws1(const int32_t *x, ulunas_state_t *st, int32_t *y);
void encoder_module(const int32_t *x, ulunas_state_t *st,
                    int32_t *e0, int32_t *e1, int32_t *e2, int32_t *e3, int32_t *e4);

/* --- GDPRNN --- */
void intra_rnn_module(const int32_t *x, int gdprnn_idx, int32_t *y);
void inter_rnn_module(const int32_t *x, int32_t *h_cache, int gdprnn_idx, int32_t *y);
void gdprnn_module(const int32_t *x, int32_t *h_cache, int gdprnn_idx, int32_t *y);

/* --- Decoder --- */
void decoder_layer0_de_xdws0(const int32_t *x, const int32_t *skip, ulunas_state_t *st, int32_t *y);
void decoder_layer1_de_xmb0(const int32_t *x, const int32_t *skip, ulunas_state_t *st, int32_t *y);
void decoder_layer2_de_xdws1(const int32_t *x, const int32_t *skip, ulunas_state_t *st, int32_t *y);
void decoder_layer3_de_xmb1(const int32_t *x, const int32_t *skip, ulunas_state_t *st, int32_t *y);
void decoder_layer4_de_xconv(const int32_t *x, const int32_t *skip, ulunas_state_t *st, int32_t *y);
void decoder_module(const int32_t *x, ulunas_state_t *st,
                    const int32_t *e0, const int32_t *e1, const int32_t *e2,
                    const int32_t *e3, const int32_t *e4,
                    int32_t *y);

/* ================================================================
 * Top-Level Inference (defined in ulunas_infer.c)
 * ================================================================ */

/**
 * ulunas_infer_frame: Single-frame inference
 * Input:  spec_real[257] (float STFT real part), spec_imag[257] (float STFT imag part)
 * Output: y_mask[2][257] in Q20 ([real_masked; imag_masked])
 * State:  st (updated with new caches)
 *
 * Pipeline: log_gen → BM → Encoder → GDPRNN×2 → Decoder → Sigmoid → BS → MASK
 */
void ulunas_infer_frame(const float *spec_real, const float *spec_imag,
                        ulunas_state_t *st, int32_t *y_mask);

#ifdef __cplusplus
}
#endif

#endif /* ULUNAS_FP_H */
