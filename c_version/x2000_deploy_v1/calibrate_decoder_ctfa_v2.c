/**
 * calibrate_decoder_ctfa_v2.c — Decoder cTFA Calibration v2
 * ==========================================================
 * KEY INSIGHT: Calibrate each decoder module's cTFA QRs by measuring
 * the FINAL decoder SNR against frame0_dec.bin, working backward
 * from d4→d0. This ensures downstream propagation effects are
 * properly accounted for.
 *
 * Unlike v1 (which optimized per-module golden match and found
 * numerically unstable extreme QRs), v2 directly optimizes the
 * end-to-end metric.
 *
 * Build:
 *   gcc -O2 -std=c99 -o calibrate_decoder_ctfa_v2 \
 *       calibrate_decoder_ctfa_v2.c ulunas_fp.c ulunas_modules.c \
 *       ulunas_infer.c ulunas_matlab_weights.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"

/* ================================================================
 * Helpers
 * ================================================================ */
static double snr_db(const int32_t *g, const int32_t *t, int n) {
    double s = 0, e = 0;
    for (int i = 0; i < n; i++) {
        double gv = g[i], d = gv - t[i];
        s += gv * gv; e += d * d;
    }
    return e < 1e-30 ? 999 : 10 * log10(s / e);
}

static int32_t *load_int32(const char *path, int n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    int32_t *b = malloc(n * sizeof(int32_t));
    if (fread(b, sizeof(int32_t), n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    fclose(f); return b;
}

static float *load_float(const char *path, int n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    float *b = malloc(n * sizeof(float));
    if (fread(b, sizeof(float), n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    fclose(f); return b;
}

/* ================================================================
 * Decoder cTFA QR state — mutable, updated as we calibrate
 * ================================================================ */
typedef struct {
    int ta_qr1, ta_qr2, ta_fc;
    int fa_qr1, fa_qr2, fa_fc;
} ctfa_qr_t;

/* Current best QRs for each module (initialized from ulunas_modules.c defaults) */
static ctfa_qr_t qr_d0 = {-13, -8, -9,  -12, -7, -9};
static ctfa_qr_t qr_d1 = {-13, -8, -9,  -12, -7, -9};
static ctfa_qr_t qr_d2 = {-13, -8, -8,  -13, -8, -9};
static ctfa_qr_t qr_d3 = {-13, -8, -9,  -11, -6, -9};
static ctfa_qr_t qr_d4 = {-13, -8, -8,  -13, -8, -9};

/* Forward declarations of parameterized decoder modules */
static void De_XDWS0_qr(const int32_t *x, const int32_t *x_skip, int16_t *ta_h,
                        int32_t *y, ctfa_qr_t qr);
static void De_XMB0_qr(const int32_t *x, const int32_t *x_skip, int16_t *ta_h,
                       int32_t *y, ctfa_qr_t qr);
static void De_XDWS1_qr(const int32_t *x, const int32_t *x_skip,
                        int32_t *conv_cache, int16_t *ta_h, int32_t *y, ctfa_qr_t qr);
static void De_XMB1_qr(const int32_t *x, const int32_t *x_skip,
                       int32_t *conv_cache, int16_t *ta_h, int32_t *y, ctfa_qr_t qr);
static void De_XConv_qr(const int32_t *x, const int32_t *x_skip,
                        int32_t *conv_cache, int16_t *ta_h, int32_t *y, ctfa_qr_t qr);

/* ================================================================
 * Full decoder with parameterized QRs
 * ================================================================ */
static void Decoder_qr(const int32_t *x, const int32_t *y_e0, const int32_t *y_e1,
                       const int32_t *y_e2, const int32_t *y_e3, const int32_t *y_e4,
                       int32_t *cache_d2, int32_t *cache_d3,
                       int32_t *cache_d4,
                       int16_t *ta_h_d0, int16_t *ta_h_d1, int16_t *ta_h_d2,
                       int16_t *ta_h_d3, int16_t *ta_h_d4,
                       int32_t *y) {
    int32_t y_d0[32*33], y_d1[24*33], y_d2[24*33], y_d3[12*65];

    De_XDWS0_qr(x, y_e4, ta_h_d0, y_d0, qr_d0);
    De_XMB0_qr(y_d0, y_e3, ta_h_d1, y_d1, qr_d1);
    De_XDWS1_qr(y_d1, y_e2, cache_d2, ta_h_d2, y_d2, qr_d2);
    De_XMB1_qr(y_d2, y_e1, cache_d3, ta_h_d3, y_d3, qr_d3);
    De_XConv_qr(y_d3, y_e0, cache_d4, ta_h_d4, y, qr_d4);
}

/* ================================================================
 * Parameterized De_XDWS0_module
 * ================================================================ */
static void De_XDWS0_qr(const int32_t *x, const int32_t *x_skip, int16_t *ta_h,
                        int32_t *y, ctfa_qr_t qr) {
    int32_t x_cat[32*33];
    for (int i = 0; i < 16*33; i++) {
        x_cat[i] = sat32((int64_t)x[i] + (int64_t)x_skip[i]);
        x_cat[16*33+i] = sat32((int64_t)x[16*33+i] + (int64_t)x_skip[16*33+i]);
    }

    int32_t y_pconv[32*33], y_shuf[32*33], y_tconv[32*33];
    { int32_t y0[16*33], y1[16*33];
      pconv2d_fixed(x_cat, 8, 33, decoder_de_convs_0_pconv_0_weight, decoder_de_convs_0_pconv_0_bias, 16, -14, y0);
      pconv2d_fixed(x_cat+8*33, 8, 33, decoder_de_convs_0_pconv_0_weight+16*8, decoder_de_convs_0_pconv_0_bias+16, 16, -14, y1);
      for (int c=0;c<16;c++) { memcpy(y_pconv+c*33,y0+c*33,33*sizeof(int32_t)); memcpy(y_pconv+(16+c)*33,y1+c*33,33*sizeof(int32_t)); }
      bn_fixed(y_pconv, 32, 33, decoder_de_convs_0_pconv_1_weight, decoder_de_convs_0_pconv_1_bias,
               decoder_de_convs_0_pconv_1_running_mean, decoder_de_convs_0_pconv_1_running_var, -14, -14);
      affine_prelu_fixed(y_pconv, 32, 33, decoder_de_convs_0_pconv_2_affine_weight,
                         decoder_de_convs_0_pconv_2_affine_bias, decoder_de_convs_0_pconv_2_slope_weight, -13, -13); }

    shuffle_interleave(y_pconv, 16, 33, y_shuf);
    nonGTConv_block(y_shuf, 32, 32, 33, 33, 1, 5, 1, 32, -17, -15, -16,
                    decoder_de_convs_0_dconv_1_weight, decoder_de_convs_0_dconv_1_bias,
                    decoder_de_convs_0_dconv_2_weight, decoder_de_convs_0_dconv_2_bias,
                    decoder_de_convs_0_dconv_2_running_mean, decoder_de_convs_0_dconv_2_running_var,
                    decoder_de_convs_0_dconv_3_affine_weight, decoder_de_convs_0_dconv_3_affine_bias,
                    decoder_de_convs_0_dconv_3_slope_weight, y_tconv);

    uint16_t ta_gate[32]; int32_t fa_gate[32*33];
    cTFA_ta_module(y_tconv, 32, 33, CH_DEC_XDWS0,
                   decoder_de_convs_0_dconv_4_ta_gru_weight_ih_l0, decoder_de_convs_0_dconv_4_ta_gru_bias_ih_l0,
                   decoder_de_convs_0_dconv_4_ta_gru_weight_hh_l0, decoder_de_convs_0_dconv_4_ta_gru_bias_hh_l0,
                   decoder_de_convs_0_dconv_4_ta_fc_weight, decoder_de_convs_0_dconv_4_ta_fc_bias,
                   ta_h, ta_gate, qr.ta_qr1, qr.ta_qr2, qr.ta_fc);
    cTFA_fa_module(y_tconv, 32, 33,
                   decoder_de_convs_0_dconv_4_fa_gru_weight_ih_l0, decoder_de_convs_0_dconv_4_fa_gru_bias_ih_l0,
                   decoder_de_convs_0_dconv_4_fa_gru_weight_hh_l0, decoder_de_convs_0_dconv_4_fa_gru_bias_hh_l0,
                   decoder_de_convs_0_dconv_4_fa_gru_weight_ih_l0_reverse, decoder_de_convs_0_dconv_4_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_0_dconv_4_fa_gru_weight_hh_l0_reverse, decoder_de_convs_0_dconv_4_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_0_dconv_4_fa_fc_weight, decoder_de_convs_0_dconv_4_fa_fc_bias,
                   fa_gate, qr.fa_qr1, qr.fa_qr2, qr.fa_fc);
    cTFA_apply(y_tconv, ta_gate, fa_gate, 32, 33, y);
}

/* ================================================================
 * Parameterized De_XMB0_module
 * ================================================================ */
static void De_XMB0_qr(const int32_t *x, const int32_t *x_skip, int16_t *ta_h,
                       int32_t *y, ctfa_qr_t qr) {
    int32_t x_cat[32*33];
    for (int i = 0; i < 32*33; i++)
        x_cat[i] = sat32((int64_t)x[i] + (int64_t)x_skip[i]);

    int32_t y_pconv0[24*33], y_shuf[24*33], y_tconv[24*33], y_pconv1[24*33];
    { int32_t y0[12*33], y1[12*33];
      pconv2d_fixed(x_cat, 16, 33, decoder_de_convs_1_pconv1_0_weight, decoder_de_convs_1_pconv1_0_bias, 12, -13, y0);
      pconv2d_fixed(x_cat+16*33, 16, 33, decoder_de_convs_1_pconv1_0_weight+12*16, decoder_de_convs_1_pconv1_0_bias+12, 12, -13, y1);
      for (int c=0;c<12;c++) { memcpy(y_pconv0+c*33,y0+c*33,33*sizeof(int32_t)); memcpy(y_pconv0+(12+c)*33,y1+c*33,33*sizeof(int32_t)); }
      bn_fixed(y_pconv0, 24, 33, decoder_de_convs_1_pconv1_1_weight, decoder_de_convs_1_pconv1_1_bias,
               decoder_de_convs_1_pconv1_1_running_mean, decoder_de_convs_1_pconv1_1_running_var, -11, -14);
      affine_prelu_fixed(y_pconv0, 24, 33, decoder_de_convs_1_pconv1_2_affine_weight,
                         decoder_de_convs_1_pconv1_2_affine_bias, decoder_de_convs_1_pconv1_2_slope_weight, -13, -13); }

    shuffle_interleave(y_pconv0, 12, 33, y_shuf);
    nonGTConv_block(y_shuf, 24, 24, 33, 33, 1, 5, 1, 24, -15, -15, -14,
                    decoder_de_convs_1_dconv_1_weight, decoder_de_convs_1_dconv_1_bias,
                    decoder_de_convs_1_dconv_2_weight, decoder_de_convs_1_dconv_2_bias,
                    decoder_de_convs_1_dconv_2_running_mean, decoder_de_convs_1_dconv_2_running_var,
                    decoder_de_convs_1_dconv_3_affine_weight, decoder_de_convs_1_dconv_3_affine_bias,
                    decoder_de_convs_1_dconv_3_slope_weight, y_tconv);

    { int32_t y0[12*33], y1[12*33];
      pconv2d_fixed(y_tconv, 12, 33, decoder_de_convs_1_pconv2_0_weight, decoder_de_convs_1_pconv2_0_bias, 12, -14, y0);
      pconv2d_fixed(y_tconv+12*33, 12, 33, decoder_de_convs_1_pconv2_0_weight+12*12, decoder_de_convs_1_pconv2_0_bias+12, 12, -14, y1);
      for (int c=0;c<12;c++) { memcpy(y_pconv1+c*33,y0+c*33,33*sizeof(int32_t)); memcpy(y_pconv1+(12+c)*33,y1+c*33,33*sizeof(int32_t)); }
      bn_fixed(y_pconv1, 24, 33, decoder_de_convs_1_pconv2_1_weight, decoder_de_convs_1_pconv2_1_bias,
               decoder_de_convs_1_pconv2_1_running_mean, decoder_de_convs_1_pconv2_1_running_var, -11, -11); }

    uint16_t ta_gate[24]; int32_t fa_gate[24*33], y_attn[24*33];
    cTFA_ta_module(y_pconv1, 24, 33, CH_DEC_XMB0,
                   decoder_de_convs_1_pconv2_2_ta_gru_weight_ih_l0, decoder_de_convs_1_pconv2_2_ta_gru_bias_ih_l0,
                   decoder_de_convs_1_pconv2_2_ta_gru_weight_hh_l0, decoder_de_convs_1_pconv2_2_ta_gru_bias_hh_l0,
                   decoder_de_convs_1_pconv2_2_ta_fc_weight, decoder_de_convs_1_pconv2_2_ta_fc_bias,
                   ta_h, ta_gate, qr.ta_qr1, qr.ta_qr2, qr.ta_fc);
    cTFA_fa_module(y_pconv1, 24, 33,
                   decoder_de_convs_1_pconv2_2_fa_gru_weight_ih_l0, decoder_de_convs_1_pconv2_2_fa_gru_bias_ih_l0,
                   decoder_de_convs_1_pconv2_2_fa_gru_weight_hh_l0, decoder_de_convs_1_pconv2_2_fa_gru_bias_hh_l0,
                   decoder_de_convs_1_pconv2_2_fa_gru_weight_ih_l0_reverse, decoder_de_convs_1_pconv2_2_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_1_pconv2_2_fa_gru_weight_hh_l0_reverse, decoder_de_convs_1_pconv2_2_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_1_pconv2_2_fa_fc_weight, decoder_de_convs_1_pconv2_2_fa_fc_bias,
                   fa_gate, qr.fa_qr1, qr.fa_qr2, qr.fa_fc);
    cTFA_apply(y_pconv1, ta_gate, fa_gate, 24, 33, y_attn);
    shuffle_deinterleave(y_attn, 12, 33, y);
}

/* ================================================================
 * Parameterized De_XDWS1_module
 * ================================================================ */
static void De_XDWS1_qr(const int32_t *x, const int32_t *x_skip,
                        int32_t *conv_cache, int16_t *ta_h, int32_t *y, ctfa_qr_t qr) {
    int32_t x_cat[24*33];
    for (int i = 0; i < 24*33; i++)
        x_cat[i] = sat32((int64_t)x[i] + (int64_t)x_skip[i]);

    int32_t y_pconv[24*33], y_shuf[24*33], y_tconv[24*33];
    { int32_t y0[12*33], y1[12*33];
      pconv2d_fixed(x_cat, 12, 33, decoder_de_convs_2_pconv_0_weight, decoder_de_convs_2_pconv_0_bias, 12, -14, y0);
      pconv2d_fixed(x_cat+12*33, 12, 33, decoder_de_convs_2_pconv_0_weight+12*12, decoder_de_convs_2_pconv_0_bias+12, 12, -14, y1);
      for (int c=0;c<12;c++) { memcpy(y_pconv+c*33,y0+c*33,33*sizeof(int32_t)); memcpy(y_pconv+(12+c)*33,y1+c*33,33*sizeof(int32_t)); }
      bn_fixed(y_pconv, 24, 33, decoder_de_convs_2_pconv_1_weight, decoder_de_convs_2_pconv_1_bias,
               decoder_de_convs_2_pconv_1_running_mean, decoder_de_convs_2_pconv_1_running_var, -11, -14);
      affine_prelu_fixed(y_pconv, 24, 33, decoder_de_convs_2_pconv_2_affine_weight,
                         decoder_de_convs_2_pconv_2_affine_bias, decoder_de_convs_2_pconv_2_slope_weight, -13, -13); }

    shuffle_interleave(y_pconv, 12, 33, y_shuf);
    GTConv_block(y_shuf, conv_cache, 24, 24, 33, 33, 1, 3, 1, 1, CACHE_DEXDWS1_ROWS, -12, 24, -18, -14,
                 decoder_de_convs_2_dconv_1_weight, decoder_de_convs_2_dconv_1_bias,
                 decoder_de_convs_2_dconv_2_weight, decoder_de_convs_2_dconv_2_bias,
                 decoder_de_convs_2_dconv_2_running_mean, decoder_de_convs_2_dconv_2_running_var,
                 decoder_de_convs_2_dconv_3_affine_weight, decoder_de_convs_2_dconv_3_affine_bias,
                 decoder_de_convs_2_dconv_3_slope_weight, y_tconv);

    uint16_t ta_gate[24]; int32_t fa_gate[24*33];
    cTFA_ta_module(y_tconv, 24, 33, CH_DEC_XMB0,
                   decoder_de_convs_2_dconv_4_ta_gru_weight_ih_l0, decoder_de_convs_2_dconv_4_ta_gru_bias_ih_l0,
                   decoder_de_convs_2_dconv_4_ta_gru_weight_hh_l0, decoder_de_convs_2_dconv_4_ta_gru_bias_hh_l0,
                   decoder_de_convs_2_dconv_4_ta_fc_weight, decoder_de_convs_2_dconv_4_ta_fc_bias,
                   ta_h, ta_gate, qr.ta_qr1, qr.ta_qr2, qr.ta_fc);
    cTFA_fa_module(y_tconv, 24, 33,
                   decoder_de_convs_2_dconv_4_fa_gru_weight_ih_l0, decoder_de_convs_2_dconv_4_fa_gru_bias_ih_l0,
                   decoder_de_convs_2_dconv_4_fa_gru_weight_hh_l0, decoder_de_convs_2_dconv_4_fa_gru_bias_hh_l0,
                   decoder_de_convs_2_dconv_4_fa_gru_weight_ih_l0_reverse, decoder_de_convs_2_dconv_4_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_2_dconv_4_fa_gru_weight_hh_l0_reverse, decoder_de_convs_2_dconv_4_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_2_dconv_4_fa_fc_weight, decoder_de_convs_2_dconv_4_fa_fc_bias,
                   fa_gate, qr.fa_qr1, qr.fa_qr2, qr.fa_fc);
    cTFA_apply(y_tconv, ta_gate, fa_gate, 24, 33, y);
}

/* ================================================================
 * Parameterized De_XMB1_module
 * ================================================================ */
static void De_XMB1_qr(const int32_t *x, const int32_t *x_skip,
                       int32_t *conv_cache, int16_t *ta_h, int32_t *y, ctfa_qr_t qr) {
    int32_t x_cat[24*33];
    for (int i = 0; i < 24*33; i++)
        x_cat[i] = sat32((int64_t)x[i] + (int64_t)x_skip[i]);

    int32_t y_pconv0[12*33], y_shuf[12*33], y_tconv[12*65], y_pconv1[12*65];
    { int32_t y0[6*33], y1[6*33];
      pconv2d_fixed(x_cat, 12, 33, decoder_de_convs_3_pconv1_0_weight, decoder_de_convs_3_pconv1_0_bias, 6, -14, y0);
      pconv2d_fixed(x_cat+12*33, 12, 33, decoder_de_convs_3_pconv1_0_weight+6*12, decoder_de_convs_3_pconv1_0_bias+6, 6, -14, y1);
      for (int c=0;c<6;c++) { memcpy(y_pconv0+c*33,y0+c*33,33*sizeof(int32_t)); memcpy(y_pconv0+(6+c)*33,y1+c*33,33*sizeof(int32_t)); }
      bn_fixed(y_pconv0, 12, 33, decoder_de_convs_3_pconv1_1_weight, decoder_de_convs_3_pconv1_1_bias,
               decoder_de_convs_3_pconv1_1_running_mean, decoder_de_convs_3_pconv1_1_running_var, -11, -14);
      affine_prelu_fixed(y_pconv0, 12, 33, decoder_de_convs_3_pconv1_2_affine_weight,
                         decoder_de_convs_3_pconv1_2_affine_bias, decoder_de_convs_3_pconv1_2_slope_weight, -13, -13); }

    shuffle_interleave(y_pconv0, 6, 33, y_shuf);
    GTConv_block(y_shuf, conv_cache, 12, 12, 33, 65, 1, 3, 1, 2, CACHE_DEXMB1_ROWS, -17, 12, -18, -16,
                 decoder_de_convs_3_dconv_1_weight, decoder_de_convs_3_dconv_1_bias,
                 decoder_de_convs_3_dconv_2_weight, decoder_de_convs_3_dconv_2_bias,
                 decoder_de_convs_3_dconv_2_running_mean, decoder_de_convs_3_dconv_2_running_var,
                 decoder_de_convs_3_dconv_3_affine_weight, decoder_de_convs_3_dconv_3_affine_bias,
                 decoder_de_convs_3_dconv_3_slope_weight, y_tconv);

    { int32_t y0[6*65], y1[6*65];
      pconv2d_fixed(y_tconv, 6, 65, decoder_de_convs_3_pconv2_0_weight, decoder_de_convs_3_pconv2_0_bias, 6, -14, y0);
      pconv2d_fixed(y_tconv+6*65, 6, 65, decoder_de_convs_3_pconv2_0_weight+6*6, decoder_de_convs_3_pconv2_0_bias+6, 6, -14, y1);
      for (int c=0;c<6;c++) { memcpy(y_pconv1+c*65,y0+c*65,65*sizeof(int32_t)); memcpy(y_pconv1+(6+c)*65,y1+c*65,65*sizeof(int32_t)); }
      bn_fixed(y_pconv1, 12, 65, decoder_de_convs_3_pconv2_1_weight, decoder_de_convs_3_pconv2_1_bias,
               decoder_de_convs_3_pconv2_1_running_mean, decoder_de_convs_3_pconv2_1_running_var, -11, -11); }

    uint16_t ta_gate[12]; int32_t fa_gate[12*65], y_attn[12*65];
    cTFA_ta_module(y_pconv1, 12, 65, CH_DEC_XMB1,
                   decoder_de_convs_3_pconv2_2_ta_gru_weight_ih_l0, decoder_de_convs_3_pconv2_2_ta_gru_bias_ih_l0,
                   decoder_de_convs_3_pconv2_2_ta_gru_weight_hh_l0, decoder_de_convs_3_pconv2_2_ta_gru_bias_hh_l0,
                   decoder_de_convs_3_pconv2_2_ta_fc_weight, decoder_de_convs_3_pconv2_2_ta_fc_bias,
                   ta_h, ta_gate, qr.ta_qr1, qr.ta_qr2, qr.ta_fc);
    cTFA_fa_module(y_pconv1, 12, 65,
                   decoder_de_convs_3_pconv2_2_fa_gru_weight_ih_l0, decoder_de_convs_3_pconv2_2_fa_gru_bias_ih_l0,
                   decoder_de_convs_3_pconv2_2_fa_gru_weight_hh_l0, decoder_de_convs_3_pconv2_2_fa_gru_bias_hh_l0,
                   decoder_de_convs_3_pconv2_2_fa_gru_weight_ih_l0_reverse, decoder_de_convs_3_pconv2_2_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_3_pconv2_2_fa_gru_weight_hh_l0_reverse, decoder_de_convs_3_pconv2_2_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_3_pconv2_2_fa_fc_weight, decoder_de_convs_3_pconv2_2_fa_fc_bias,
                   fa_gate, qr.fa_qr1, qr.fa_qr2, qr.fa_fc);
    cTFA_apply(y_pconv1, ta_gate, fa_gate, 12, 65, y_attn);
    shuffle_deinterleave(y_attn, 6, 65, y);
}

/* ================================================================
 * Parameterized De_XConv_module
 * ================================================================ */
static void De_XConv_qr(const int32_t *x, const int32_t *x_skip,
                        int32_t *conv_cache, int16_t *ta_h, int32_t *y, ctfa_qr_t qr) {
    int32_t x_cat[12*65];
    for (int i = 0; i < 12*65; i++)
        x_cat[i] = sat32((int64_t)x[i] + (int64_t)x_skip[i]);

    int H_in = 3;
    int32_t *x_full = calloc(12 * H_in * 65, sizeof(int32_t));
    for (int h = 0; h < 2; h++)
        for (int c = 0; c < 12; c++)
            for (int w = 0; w < 65; w++)
                x_full[(h*12+c)*65+w] = conv_cache[(h*12+c)*65+w];
    for (int c = 0; c < 12; c++)
        for (int w = 0; w < 65; w++)
            x_full[(2*12+c)*65+w] = x_cat[c*65+w];

    int Cin_d4 = 12, Cout_d4 = 1, Win_d4 = 65, Wout_d4 = 129, Hk_d4 = 3, Wk_d4 = 3;
    int stride_w = 2, pad_w = 1, conv_qr_d4 = -14;
    int W_insert = Win_d4 + (Win_d4 - 1) * (stride_w - 1);
    int W_padded = W_insert + 2 * pad_w;

    int32_t y_tconv[1*129];
    for (int co = 0; co < Cout_d4; co++) {
        for (int wo = 0; wo < Wout_d4; wo++)
            y_tconv[co * Wout_d4 + wo] = decoder_de_convs_4_ops_1_bias[co];
        for (int ci = 0; ci < Cin_d4; ci++) {
            for (int hk = 0; hk < Hk_d4; hk++) {
                const int32_t *frame = x_full + (hk * Cin_d4 + ci) * Win_d4;
                int32_t x_insert[131]; memset(x_insert, 0, sizeof(x_insert));
                for (int w = 0; w < Win_d4; w++)
                    x_insert[pad_w + w * stride_w] = frame[w];
                for (int wo = 0; wo < Wout_d4; wo++) {
                    int64_t acc = 0;
                    for (int wk = 0; wk < Wk_d4; wk++) {
                        int wi = wo + wk;
                        int32_t xv = (wi >= 0 && wi < W_padded) ? x_insert[wi] : 0;
                        int kidx = ci + Cin_d4 * co
                                 + Cin_d4 * Cout_d4 * (Hk_d4 - 1 - hk)
                                 + Cin_d4 * Cout_d4 * Hk_d4 * (Wk_d4 - 1 - wk);
                        acc += (int64_t)xv * (int64_t)decoder_de_convs_4_ops_1_weight[kidx];
                    }
                    int shift = -conv_qr_d4;
                    int32_t val = (int32_t)((acc + ((int64_t)1 << (shift - 1))) >> shift);
                    y_tconv[co * Wout_d4 + wo] = sat32((int64_t)y_tconv[co * Wout_d4 + wo] + val);
                }
            }
        }
    }
    free(x_full);

    bn_fixed(y_tconv, Cout_d4, 129,
             decoder_de_convs_4_ops_2_weight, decoder_de_convs_4_ops_2_bias,
             decoder_de_convs_4_ops_2_running_mean, decoder_de_convs_4_ops_2_running_var, -11, -11);

    uint16_t ta_gate[1]; int32_t fa_gate[1*129];
    cTFA_ta_module(y_tconv, 1, 129, CH_DEC_XCONV,
                   decoder_de_convs_4_ops_4_ta_gru_weight_ih_l0, decoder_de_convs_4_ops_4_ta_gru_bias_ih_l0,
                   decoder_de_convs_4_ops_4_ta_gru_weight_hh_l0, decoder_de_convs_4_ops_4_ta_gru_bias_hh_l0,
                   decoder_de_convs_4_ops_4_ta_fc_weight, decoder_de_convs_4_ops_4_ta_fc_bias,
                   ta_h, ta_gate, qr.ta_qr1, qr.ta_qr2, qr.ta_fc);
    cTFA_fa_module(y_tconv, 1, 129,
                   decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0, decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0,
                   decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0, decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0,
                   decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0_reverse, decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0_reverse, decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_4_ops_4_fa_fc_weight, decoder_de_convs_4_ops_4_fa_fc_bias,
                   fa_gate, qr.fa_qr1, qr.fa_qr2, qr.fa_fc);
    cTFA_apply(y_tconv, ta_gate, fa_gate, 1, 129, y);
}

/* ================================================================
 * Calibrate one module — optimize its cTFA QRs for final decoder SNR
 * ================================================================ */
static void calibrate_module(const char *name, ctfa_qr_t *qr,
                             const int32_t *r2,
                             const int32_t *e0, const int32_t *e1,
                             const int32_t *e2, const int32_t *e3, const int32_t *e4,
                             const int32_t *golden_dec) {
    printf("--- %s ---\n", name);

    /* Decoder state */
    int32_t cache_d2[1 * 24 * 33]; memset(cache_d2, 0, sizeof(cache_d2));
    int32_t cache_d3[1 * 12 * 33]; memset(cache_d3, 0, sizeof(cache_d3));
    int32_t cache_d4[2 * 12 * 65]; memset(cache_d4, 0, sizeof(cache_d4));
    int16_t ta_h_d0[CH_DEC_XDWS0], ta_h_d1[CH_DEC_XMB0], ta_h_d2[CH_DEC_XMB0];
    int16_t ta_h_d3[CH_DEC_XMB1], ta_h_d4[CH_DEC_XCONV];
    memset(ta_h_d0, 0, sizeof(ta_h_d0)); memset(ta_h_d1, 0, sizeof(ta_h_d1));
    memset(ta_h_d2, 0, sizeof(ta_h_d2)); memset(ta_h_d3, 0, sizeof(ta_h_d3));
    memset(ta_h_d4, 0, sizeof(ta_h_d4));

    int32_t y_dec[1*129];

    /* Baseline */
    Decoder_qr(r2, e0, e1, e2, e3, e4,
               cache_d2, cache_d3, cache_d4,
               ta_h_d0, ta_h_d1, ta_h_d2, ta_h_d3, ta_h_d4, y_dec);
    double base = snr_db(y_dec, golden_dec, 129);
    printf("  base=%5.1f dB  current(%d,%d,%d | %d,%d,%d)\n",
           base, qr->ta_qr1, qr->ta_qr2, qr->ta_fc, qr->fa_qr1, qr->fa_qr2, qr->fa_fc);

    /* Coarse grid — bounded to reasonable ranges to avoid numerical instability */
    double best = -999;
    ctfa_qr_t best_qr = *qr;

    /* Reasonable QR ranges: ta_qr1 [-18,-4], ta_qr2 [-14,-4], ta_fc [-12,-4]
     *                       fa_qr1 [-18,-4], fa_qr2 [-14,-4], fa_fc [-12,-4] */
    for (int ta1 = -18; ta1 <= -6; ta1 += 2)
    for (int ta2 = -12; ta2 <= -4; ta2 += 2)
    for (int tfc = -12; tfc <= -4; tfc += 2)
    for (int fa1 = -18; fa1 <= -6; fa1 += 2)
    for (int fa2 = -12; fa2 <= -4; fa2 += 2)
    for (int ffc = -12; ffc <= -4; ffc += 2) {
        ctfa_qr_t test = {ta1, ta2, tfc, fa1, fa2, ffc};
        *qr = test;
        Decoder_qr(r2, e0, e1, e2, e3, e4,
                   cache_d2, cache_d3, cache_d4,
                   ta_h_d0, ta_h_d1, ta_h_d2, ta_h_d3, ta_h_d4, y_dec);
        double s = snr_db(y_dec, golden_dec, 129);
        if (s > best) { best = s; best_qr = test; }
    }

    /* Fine grid around best */
    for (int ta1 = best_qr.ta_qr1-1; ta1 <= best_qr.ta_qr1+1; ta1++)
    for (int ta2 = best_qr.ta_qr2-1; ta2 <= best_qr.ta_qr2+1; ta2++)
    for (int tfc = best_qr.ta_fc-1; tfc <= best_qr.ta_fc+1; tfc++)
    for (int fa1 = best_qr.fa_qr1-1; fa1 <= best_qr.fa_qr1+1; fa1++)
    for (int fa2 = best_qr.fa_qr2-1; fa2 <= best_qr.fa_qr2+1; fa2++)
    for (int ffc = best_qr.fa_fc-1; ffc <= best_qr.fa_fc+1; ffc++) {
        ctfa_qr_t test = {ta1, ta2, tfc, fa1, fa2, ffc};
        *qr = test;
        Decoder_qr(r2, e0, e1, e2, e3, e4,
                   cache_d2, cache_d3, cache_d4,
                   ta_h_d0, ta_h_d1, ta_h_d2, ta_h_d3, ta_h_d4, y_dec);
        double s = snr_db(y_dec, golden_dec, 129);
        if (s > best) { best = s; best_qr = test; }
    }

    /* Apply best QRs permanently */
    *qr = best_qr;
    printf("  best=%5.1f dB  Δ=%+.1f  (%d,%d,%d | %d,%d,%d)\n\n",
           best, best - base,
           best_qr.ta_qr1, best_qr.ta_qr2, best_qr.ta_fc,
           best_qr.fa_qr1, best_qr.fa_qr2, best_qr.fa_fc);
}

/* ================================================================
 * Main
 * ================================================================ */
int main() {
    printf("=== Decoder cTFA Calibration v2 (Final-Output Metric) ===\n\n");

    /* Load STFT */
    float *real_in = load_float("dump_matlab/frame0_stft_real.bin", 257);
    float *imag_in = load_float("dump_matlab/frame0_stft_imag.bin", 257);
    if (!real_in || !imag_in) { printf("ERROR: missing STFT\n"); return 1; }

    int32_t *golden_dec = load_int32("dump_matlab/frame0_dec.bin", 1*129);
    if (!golden_dec) { printf("ERROR: missing frame0_dec.bin\n"); return 1; }

    /* Run encoder + DPRNN once (doesn't depend on decoder QRs) */
    int32_t x_log[1*257]; log_gen_fixed(real_in, imag_in, 257, x_log);
    int32_t x_bm[1*129];  BM_fixed(x_log, erb_erb_fc_weight, x_bm);

    ulunas_state_t st; ulunas_state_init(&st);
    int32_t e0[12*65], e1[24*33], e2[24*33], e3[32*33], e4[16*33];
    Encoder_module(x_bm, &st, e0, e1, e2, e3, e4);

    int32_t r1[16*33], r2[16*33];
    GDPRNN_module(e4, st.inter_prev0, 0, r1);
    GDPRNN_module(r1, st.inter_prev1, 1, r2);
    printf("Encoder+DPRNN done.\n\n");

    /* Calibrate backward: d4 → d3 → d2 → d1 → d0
     * Each module's QRs affect downstream, so we calibrate the LAST
     * module first and work backward. */
    printf("Calibrating d4→d0 (backward pass)...\n\n");
    calibrate_module("d4 De_XConv",   &qr_d4, r2, e0, e1, e2, e3, e4, golden_dec);
    calibrate_module("d3 De_XMB1",    &qr_d3, r2, e0, e1, e2, e3, e4, golden_dec);
    calibrate_module("d2 De_XDWS1",   &qr_d2, r2, e0, e1, e2, e3, e4, golden_dec);
    calibrate_module("d1 De_XMB0",    &qr_d1, r2, e0, e1, e2, e3, e4, golden_dec);
    calibrate_module("d0 De_XDWS0",   &qr_d0, r2, e0, e1, e2, e3, e4, golden_dec);

    /* Final SNR with all best QRs */
    printf("=== UPDATE ulunas_modules.c ===\n");
    printf("d0: ta=(%d,%d,%d) fa=(%d,%d,%d)\n", qr_d0.ta_qr1, qr_d0.ta_qr2, qr_d0.ta_fc, qr_d0.fa_qr1, qr_d0.fa_qr2, qr_d0.fa_fc);
    printf("d1: ta=(%d,%d,%d) fa=(%d,%d,%d)\n", qr_d1.ta_qr1, qr_d1.ta_qr2, qr_d1.ta_fc, qr_d1.fa_qr1, qr_d1.fa_qr2, qr_d1.fa_fc);
    printf("d2: ta=(%d,%d,%d) fa=(%d,%d,%d)\n", qr_d2.ta_qr1, qr_d2.ta_qr2, qr_d2.ta_fc, qr_d2.fa_qr1, qr_d2.fa_qr2, qr_d2.fa_fc);
    printf("d3: ta=(%d,%d,%d) fa=(%d,%d,%d)\n", qr_d3.ta_qr1, qr_d3.ta_qr2, qr_d3.ta_fc, qr_d3.fa_qr1, qr_d3.fa_qr2, qr_d3.fa_fc);
    printf("d4: ta=(%d,%d,%d) fa=(%d,%d,%d)\n", qr_d4.ta_qr1, qr_d4.ta_qr2, qr_d4.ta_fc, qr_d4.fa_qr1, qr_d4.fa_qr2, qr_d4.fa_fc);

    free(real_in); free(imag_in); free(golden_dec);
    return 0;
}
