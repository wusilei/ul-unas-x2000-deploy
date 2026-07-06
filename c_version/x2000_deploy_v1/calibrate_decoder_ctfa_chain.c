/**
 * calibrate_decoder_ctfa_chain.c — Decoder cTFA Full-Chain Calibration
 * ====================================================================
 * KEY DIFFERENCE from calibrate_ctfa.c:
 *   calibrate_ctfa.c uses MATLAB golden as cTFA input
 *   THIS tool runs the C encoder+DPRNN chain to produce the ACTUAL
 *   cTFA inputs that decoder modules see in production.
 *
 * Each decoder module is calibrated sequentially (d0→d4) so that
 * improved QRs in earlier modules propagate to downstream modules.
 *
 * Golden reference: frame0_dec_d{0-4}.bin (full module output from MATLAB)
 *
 * Build:
 *   gcc -O2 -std=c99 -o calibrate_decoder_ctfa_chain \
 *       calibrate_decoder_ctfa_chain.c ulunas_fp.c ulunas_matlab_weights.c -lm
 * Usage:
 *   ./calibrate_decoder_ctfa_chain
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"

/* ================================================================
 * SNR helpers
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
 * Full-Chain Calibration State
 * ================================================================ */
typedef struct {
    /* cTFA input captured from C chain */
    int32_t *ctfa_in;    /* the tensor right before cTFA_ta_module */
    int C, W;
    int hidden_dim;
    /* cTFA weights */
    const int16_t *ta_ih_w; const int32_t *ta_ih_b;
    const int16_t *ta_hh_w; const int32_t *ta_hh_b;
    const int16_t *ta_fc_w; const int32_t *ta_fc_b;
    const int16_t *fa_ih_w; const int32_t *fa_ih_b;
    const int16_t *fa_hh_w; const int32_t *fa_hh_b;
    const int16_t *fa_re_ih_w; const int32_t *fa_re_ih_b;
    const int16_t *fa_re_hh_w; const int32_t *fa_re_hh_b;
    const int16_t *fa_fc_w; const int32_t *fa_fc_b;
    /* Golden full-module output */
    int32_t *golden_out;
    /* Default QRs */
    int def_ta_qr1, def_ta_qr2, def_ta_fc;
    int def_fa_qr1, def_fa_qr2, def_fa_fc;
    /* Module name */
    const char *name;
    /* Whether to apply shuffle_deinterleave after cTFA */
    int shuffle_half_C;
} dec_ctfa_calib_t;

/* ================================================================
 * Run cTFA + optional shuffle, return SNR vs golden
 * ================================================================ */
static double eval_ctfa(const dec_ctfa_calib_t *m,
                        int ta_qr1, int ta_qr2, int ta_fc,
                        int fa_qr1, int fa_qr2, int fa_fc,
                        int16_t *ta_h_buf) {
    int total = m->C * m->W;
    uint16_t ta_gate[m->C];
    int32_t fa_gate[total];
    int32_t y[total];

    memset(ta_h_buf, 0, m->hidden_dim * sizeof(int16_t));

    cTFA_ta_module(m->ctfa_in, m->C, m->W, m->hidden_dim,
                   m->ta_ih_w, m->ta_ih_b,
                   m->ta_hh_w, m->ta_hh_b,
                   m->ta_fc_w, m->ta_fc_b,
                   ta_h_buf, ta_gate,
                   ta_qr1, ta_qr2, ta_fc);

    cTFA_fa_module(m->ctfa_in, m->C, m->W,
                   m->fa_ih_w, m->fa_ih_b,
                   m->fa_hh_w, m->fa_hh_b,
                   m->fa_re_ih_w, m->fa_re_ih_b,
                   m->fa_re_hh_w, m->fa_re_hh_b,
                   m->fa_fc_w, m->fa_fc_b,
                   fa_gate, fa_qr1, fa_qr2, fa_fc);

    cTFA_apply(m->ctfa_in, ta_gate, fa_gate, m->C, m->W, y);

    /* Apply shuffle_deinterleave if needed (d1, d3) */
    if (m->shuffle_half_C > 0) {
        int hc = m->shuffle_half_C;
        int32_t tmp[total];
        memcpy(tmp, y, total * sizeof(int32_t));
        shuffle_deinterleave(tmp, hc, m->W, y);
    }

    return snr_db(y, m->golden_out, total);
}

/* ================================================================
 * Brute-force search: coarse grid → fine grid
 * ================================================================ */
static void calibrate_one(dec_ctfa_calib_t *m) {
    int total = m->C * m->W;
    int16_t *ta_h_buf = calloc(m->hidden_dim, sizeof(int16_t));

    /* Baseline */
    double base = eval_ctfa(m, m->def_ta_qr1, m->def_ta_qr2, m->def_ta_fc,
                            m->def_fa_qr1, m->def_fa_qr2, m->def_fa_fc, ta_h_buf);
    printf("  %-12s base=%6.1f dB  default(%d,%d,%d | %d,%d,%d)\n",
           m->name, base,
           m->def_ta_qr1, m->def_ta_qr2, m->def_ta_fc,
           m->def_fa_qr1, m->def_fa_qr2, m->def_fa_fc);

    /* Coarse grid search */
    double best = -999;
    int bta1 = m->def_ta_qr1, bta2 = m->def_ta_qr2, btfc = m->def_ta_fc;
    int bfa1 = m->def_fa_qr1, bfa2 = m->def_fa_qr2, bffc = m->def_fa_fc;

    for (int ta1 = -20; ta1 <= -2; ta1 += 2)
    for (int ta2 = -14; ta2 <= -2; ta2 += 2)
    for (int tfc = -12; tfc <= -1; tfc += 2)
    for (int fa1 = -20; fa1 <= -2; fa1 += 2)
    for (int fa2 = -14; fa2 <= -2; fa2 += 2)
    for (int ffc = -12; ffc <= -1; ffc += 2) {
        double s = eval_ctfa(m, ta1, ta2, tfc, fa1, fa2, ffc, ta_h_buf);
        if (s > best) { best = s; bta1 = ta1; bta2 = ta2; btfc = tfc; bfa1 = fa1; bfa2 = fa2; bffc = ffc; }
    }

    /* Fine grid around best */
    for (int ta1 = bta1-1; ta1 <= bta1+1; ta1++)
    for (int ta2 = bta2-1; ta2 <= bta2+1; ta2++)
    for (int tfc = btfc-1; tfc <= btfc+1; tfc++)
    for (int fa1 = bfa1-1; fa1 <= bfa1+1; fa1++)
    for (int fa2 = bfa2-1; fa2 <= bfa2+1; fa2++)
    for (int ffc = bffc-1; ffc <= bffc+1; ffc++) {
        double s = eval_ctfa(m, ta1, ta2, tfc, fa1, fa2, ffc, ta_h_buf);
        if (s > best) { best = s; bta1 = ta1; bta2 = ta2; btfc = tfc; bfa1 = fa1; bfa2 = fa2; bffc = ffc; }
    }

    printf("  %-12s best=%6.1f dB Δ=%+.1f  (%d,%d,%d | %d,%d,%d)\n\n",
           m->name, best, best - base,
           bta1, bta2, btfc, bfa1, bfa2, bffc);

    /* Store best QRs back for sequential propagation */
    m->def_ta_qr1 = bta1; m->def_ta_qr2 = bta2; m->def_ta_fc = btfc;
    m->def_fa_qr1 = bfa1; m->def_fa_qr2 = bfa2; m->def_fa_fc = bffc;

    free(ta_h_buf);

    /* Print the update snippet for ulunas_modules.c */
    printf("  → UPDATE ulunas_modules.c: %s cTFA ta=(%d,%d,%d) fa=(%d,%d,%d)\n",
           m->name, bta1, bta2, btfc, bfa1, bfa2, bffc);
}

/* ================================================================
 * De_XDWS0 — cTFA input capture
 * ================================================================
 * Returns cTFA input (y_tconv from nonGTConv_block)
 */
static int32_t *capture_d0_ctfa_in(const int32_t *x, const int32_t *x_skip) {
    int32_t x_cat[32*33];
    for (int i = 0; i < 16*33; i++) {
        x_cat[i] = sat32((int64_t)x[i] + (int64_t)x_skip[i]);
        x_cat[16*33+i] = sat32((int64_t)x[16*33+i] + (int64_t)x_skip[16*33+i]);
    }

    int32_t y_pconv[32*33], y_shuf[32*33];
    /* PConv: Cin=8+8 per group, Cout=16+16, qr=-14 */
    { int32_t y0[16*33], y1[16*33];
      pconv2d_fixed(x_cat, 8, 33, decoder_de_convs_0_pconv_0_weight, decoder_de_convs_0_pconv_0_bias, 16, -14, y0);
      pconv2d_fixed(x_cat+8*33, 8, 33, decoder_de_convs_0_pconv_0_weight+16*8, decoder_de_convs_0_pconv_0_bias+16, 16, -14, y1);
      for (int c=0;c<16;c++) { memcpy(y_pconv+c*33,y0+c*33,33*sizeof(int32_t)); memcpy(y_pconv+(16+c)*33,y1+c*33,33*sizeof(int32_t)); }
      bn_fixed(y_pconv, 32, 33, decoder_de_convs_0_pconv_1_weight, decoder_de_convs_0_pconv_1_bias,
               decoder_de_convs_0_pconv_1_running_mean, decoder_de_convs_0_pconv_1_running_var, -14, -14);
      affine_prelu_fixed(y_pconv, 32, 33, decoder_de_convs_0_pconv_2_affine_weight,
                         decoder_de_convs_0_pconv_2_affine_bias, decoder_de_convs_0_pconv_2_slope_weight, -13, -13); }

    shuffle_interleave(y_pconv, 16, 33, y_shuf);

    /* nonGTConv */
    int32_t *y_tconv = calloc(32*33, sizeof(int32_t));
    nonGTConv_block(y_shuf, 32, 32, 33, 33, 1, 5, 1, 32, -17, -15, -16,
                    decoder_de_convs_0_dconv_1_weight, decoder_de_convs_0_dconv_1_bias,
                    decoder_de_convs_0_dconv_2_weight, decoder_de_convs_0_dconv_2_bias,
                    decoder_de_convs_0_dconv_2_running_mean, decoder_de_convs_0_dconv_2_running_var,
                    decoder_de_convs_0_dconv_3_affine_weight, decoder_de_convs_0_dconv_3_affine_bias,
                    decoder_de_convs_0_dconv_3_slope_weight, y_tconv);
    return y_tconv;
}

/* ================================================================
 * De_XMB0 — cTFA input capture
 * ================================================================
 * Returns cTFA input (y_pconv1 after BN)
 */
static int32_t *capture_d1_ctfa_in(const int32_t *x, const int32_t *x_skip) {
    int32_t x_cat[32*33];
    for (int i = 0; i < 32*33; i++)
        x_cat[i] = sat32((int64_t)x[i] + (int64_t)x_skip[i]);

    int32_t y_pconv0[24*33], y_shuf[24*33], y_tconv[24*33];
    /* PConv0: Cin=16 per group, Cout=12 per group, total 32→24 */
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

    /* PConv1: grouped 12→12, BN only */
    int32_t *y_pconv1 = calloc(24*33, sizeof(int32_t));
    { int32_t y0[12*33], y1[12*33];
      pconv2d_fixed(y_tconv, 12, 33, decoder_de_convs_1_pconv2_0_weight, decoder_de_convs_1_pconv2_0_bias, 12, -14, y0);
      pconv2d_fixed(y_tconv+12*33, 12, 33, decoder_de_convs_1_pconv2_0_weight+12*12, decoder_de_convs_1_pconv2_0_bias+12, 12, -14, y1);
      for (int c=0;c<12;c++) { memcpy(y_pconv1+c*33,y0+c*33,33*sizeof(int32_t)); memcpy(y_pconv1+(12+c)*33,y1+c*33,33*sizeof(int32_t)); }
      bn_fixed(y_pconv1, 24, 33, decoder_de_convs_1_pconv2_1_weight, decoder_de_convs_1_pconv2_1_bias,
               decoder_de_convs_1_pconv2_1_running_mean, decoder_de_convs_1_pconv2_1_running_var, -11, -11); }
    return y_pconv1;
}

/* ================================================================
 * De_XDWS1 — cTFA input capture
 * ================================================================
 * Returns cTFA input (y_tconv from GTConv_block)
 */
static int32_t *capture_d2_ctfa_in(const int32_t *x, const int32_t *x_skip, int32_t *conv_cache) {
    int32_t x_cat[24*33];
    for (int i = 0; i < 24*33; i++)
        x_cat[i] = sat32((int64_t)x[i] + (int64_t)x_skip[i]);

    int32_t y_pconv[24*33], y_shuf[24*33];
    /* PConv: grouped 12→12 */
    { int32_t y0[12*33], y1[12*33];
      pconv2d_fixed(x_cat, 12, 33, decoder_de_convs_2_pconv_0_weight, decoder_de_convs_2_pconv_0_bias, 12, -14, y0);
      pconv2d_fixed(x_cat+12*33, 12, 33, decoder_de_convs_2_pconv_0_weight+12*12, decoder_de_convs_2_pconv_0_bias+12, 12, -14, y1);
      for (int c=0;c<12;c++) { memcpy(y_pconv+c*33,y0+c*33,33*sizeof(int32_t)); memcpy(y_pconv+(12+c)*33,y1+c*33,33*sizeof(int32_t)); }
      bn_fixed(y_pconv, 24, 33, decoder_de_convs_2_pconv_1_weight, decoder_de_convs_2_pconv_1_bias,
               decoder_de_convs_2_pconv_1_running_mean, decoder_de_convs_2_pconv_1_running_var, -11, -14);
      affine_prelu_fixed(y_pconv, 24, 33, decoder_de_convs_2_pconv_2_affine_weight,
                         decoder_de_convs_2_pconv_2_affine_bias, decoder_de_convs_2_pconv_2_slope_weight, -13, -13); }

    shuffle_interleave(y_pconv, 12, 33, y_shuf);

    int32_t *y_tconv = calloc(24*33, sizeof(int32_t));
    GTConv_block(y_shuf, conv_cache, 24, 24, 33, 33, 1, 3, 1, 1, CACHE_DEXDWS1_ROWS, -12, 24, -18, -14,
                 decoder_de_convs_2_dconv_1_weight, decoder_de_convs_2_dconv_1_bias,
                 decoder_de_convs_2_dconv_2_weight, decoder_de_convs_2_dconv_2_bias,
                 decoder_de_convs_2_dconv_2_running_mean, decoder_de_convs_2_dconv_2_running_var,
                 decoder_de_convs_2_dconv_3_affine_weight, decoder_de_convs_2_dconv_3_affine_bias,
                 decoder_de_convs_2_dconv_3_slope_weight, y_tconv);
    return y_tconv;
}

/* ================================================================
 * De_XMB1 — cTFA input capture
 * ================================================================
 * Returns cTFA input (y_pconv1 after BN)
 */
static int32_t *capture_d3_ctfa_in(const int32_t *x, const int32_t *x_skip, int32_t *conv_cache) {
    int32_t x_cat[24*33];
    for (int i = 0; i < 24*33; i++)
        x_cat[i] = sat32((int64_t)x[i] + (int64_t)x_skip[i]);

    int32_t y_pconv0[12*33], y_shuf[12*33], y_tconv[12*65];
    /* PConv0: Cin=12 per group, Cout=6 per group, total 24→12 */
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

    /* PConv1: grouped 6→6, BN only */
    int32_t *y_pconv1 = calloc(12*65, sizeof(int32_t));
    { int32_t y0[6*65], y1[6*65];
      pconv2d_fixed(y_tconv, 6, 65, decoder_de_convs_3_pconv2_0_weight, decoder_de_convs_3_pconv2_0_bias, 6, -14, y0);
      pconv2d_fixed(y_tconv+6*65, 6, 65, decoder_de_convs_3_pconv2_0_weight+6*6, decoder_de_convs_3_pconv2_0_bias+6, 6, -14, y1);
      for (int c=0;c<6;c++) { memcpy(y_pconv1+c*65,y0+c*65,65*sizeof(int32_t)); memcpy(y_pconv1+(6+c)*65,y1+c*65,65*sizeof(int32_t)); }
      bn_fixed(y_pconv1, 12, 65, decoder_de_convs_3_pconv2_1_weight, decoder_de_convs_3_pconv2_1_bias,
               decoder_de_convs_3_pconv2_1_running_mean, decoder_de_convs_3_pconv2_1_running_var, -11, -11); }
    return y_pconv1;
}

/* ================================================================
 * De_XConv — cTFA input capture
 * ================================================================
 * Returns cTFA input (y_tconv after TConv + BN)
 */
static int32_t *capture_d4_ctfa_in(const int32_t *x, const int32_t *x_skip, int32_t *conv_cache) {
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
    int stride_w = 2, pad_w = 1;
    int W_insert = Win_d4 + (Win_d4 - 1) * (stride_w - 1);
    int W_padded = W_insert + 2 * pad_w;
    int conv_qr_d4 = -14;

    int32_t *y_tconv = calloc(1*129, sizeof(int32_t));
    for (int co = 0; co < Cout_d4; co++) {
        for (int wo = 0; wo < Wout_d4; wo++)
            y_tconv[co * Wout_d4 + wo] = decoder_de_convs_4_ops_1_bias[co];

        for (int ci = 0; ci < Cin_d4; ci++) {
            for (int hk = 0; hk < Hk_d4; hk++) {
                const int32_t *frame = x_full + (hk * Cin_d4 + ci) * Win_d4;
                int32_t x_insert[131];
                memset(x_insert, 0, sizeof(x_insert));
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
             decoder_de_convs_4_ops_2_running_mean, decoder_de_convs_4_ops_2_running_var,
             -11, -11);
    return y_tconv;
}

/* ================================================================
 * Run full decoder module with given cTFA QRs
 * ================================================================ */
static void run_d0_full(const int32_t *ctfa_in, int ta1, int ta2, int tfc, int fa1, int fa2, int ffc,
                        int16_t *ta_h, int32_t *y) {
    uint16_t ta_gate[32]; int32_t fa_gate[32*33];
    cTFA_ta_module(ctfa_in, 32, 33, CH_DEC_XDWS0,
                   decoder_de_convs_0_dconv_4_ta_gru_weight_ih_l0, decoder_de_convs_0_dconv_4_ta_gru_bias_ih_l0,
                   decoder_de_convs_0_dconv_4_ta_gru_weight_hh_l0, decoder_de_convs_0_dconv_4_ta_gru_bias_hh_l0,
                   decoder_de_convs_0_dconv_4_ta_fc_weight, decoder_de_convs_0_dconv_4_ta_fc_bias,
                   ta_h, ta_gate, ta1, ta2, tfc);
    cTFA_fa_module(ctfa_in, 32, 33,
                   decoder_de_convs_0_dconv_4_fa_gru_weight_ih_l0, decoder_de_convs_0_dconv_4_fa_gru_bias_ih_l0,
                   decoder_de_convs_0_dconv_4_fa_gru_weight_hh_l0, decoder_de_convs_0_dconv_4_fa_gru_bias_hh_l0,
                   decoder_de_convs_0_dconv_4_fa_gru_weight_ih_l0_reverse, decoder_de_convs_0_dconv_4_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_0_dconv_4_fa_gru_weight_hh_l0_reverse, decoder_de_convs_0_dconv_4_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_0_dconv_4_fa_fc_weight, decoder_de_convs_0_dconv_4_fa_fc_bias,
                   fa_gate, fa1, fa2, ffc);
    cTFA_apply(ctfa_in, ta_gate, fa_gate, 32, 33, y);
}

static void run_d1_full(const int32_t *ctfa_in, int ta1, int ta2, int tfc, int fa1, int fa2, int ffc,
                        int16_t *ta_h, int32_t *y) {
    uint16_t ta_gate[24]; int32_t fa_gate[24*33], y_attn[24*33];
    cTFA_ta_module(ctfa_in, 24, 33, CH_DEC_XMB0,
                   decoder_de_convs_1_pconv2_2_ta_gru_weight_ih_l0, decoder_de_convs_1_pconv2_2_ta_gru_bias_ih_l0,
                   decoder_de_convs_1_pconv2_2_ta_gru_weight_hh_l0, decoder_de_convs_1_pconv2_2_ta_gru_bias_hh_l0,
                   decoder_de_convs_1_pconv2_2_ta_fc_weight, decoder_de_convs_1_pconv2_2_ta_fc_bias,
                   ta_h, ta_gate, ta1, ta2, tfc);
    cTFA_fa_module(ctfa_in, 24, 33,
                   decoder_de_convs_1_pconv2_2_fa_gru_weight_ih_l0, decoder_de_convs_1_pconv2_2_fa_gru_bias_ih_l0,
                   decoder_de_convs_1_pconv2_2_fa_gru_weight_hh_l0, decoder_de_convs_1_pconv2_2_fa_gru_bias_hh_l0,
                   decoder_de_convs_1_pconv2_2_fa_gru_weight_ih_l0_reverse, decoder_de_convs_1_pconv2_2_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_1_pconv2_2_fa_gru_weight_hh_l0_reverse, decoder_de_convs_1_pconv2_2_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_1_pconv2_2_fa_fc_weight, decoder_de_convs_1_pconv2_2_fa_fc_bias,
                   fa_gate, fa1, fa2, ffc);
    cTFA_apply(ctfa_in, ta_gate, fa_gate, 24, 33, y_attn);
    shuffle_deinterleave(y_attn, 12, 33, y);
}

static void run_d2_full(const int32_t *ctfa_in, int ta1, int ta2, int tfc, int fa1, int fa2, int ffc,
                        int16_t *ta_h, int32_t *y) {
    uint16_t ta_gate[24]; int32_t fa_gate[24*33];
    cTFA_ta_module(ctfa_in, 24, 33, CH_DEC_XMB0,
                   decoder_de_convs_2_dconv_4_ta_gru_weight_ih_l0, decoder_de_convs_2_dconv_4_ta_gru_bias_ih_l0,
                   decoder_de_convs_2_dconv_4_ta_gru_weight_hh_l0, decoder_de_convs_2_dconv_4_ta_gru_bias_hh_l0,
                   decoder_de_convs_2_dconv_4_ta_fc_weight, decoder_de_convs_2_dconv_4_ta_fc_bias,
                   ta_h, ta_gate, ta1, ta2, tfc);
    cTFA_fa_module(ctfa_in, 24, 33,
                   decoder_de_convs_2_dconv_4_fa_gru_weight_ih_l0, decoder_de_convs_2_dconv_4_fa_gru_bias_ih_l0,
                   decoder_de_convs_2_dconv_4_fa_gru_weight_hh_l0, decoder_de_convs_2_dconv_4_fa_gru_bias_hh_l0,
                   decoder_de_convs_2_dconv_4_fa_gru_weight_ih_l0_reverse, decoder_de_convs_2_dconv_4_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_2_dconv_4_fa_gru_weight_hh_l0_reverse, decoder_de_convs_2_dconv_4_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_2_dconv_4_fa_fc_weight, decoder_de_convs_2_dconv_4_fa_fc_bias,
                   fa_gate, fa1, fa2, ffc);
    cTFA_apply(ctfa_in, ta_gate, fa_gate, 24, 33, y);
}

static void run_d3_full(const int32_t *ctfa_in, int ta1, int ta2, int tfc, int fa1, int fa2, int ffc,
                        int16_t *ta_h, int32_t *y) {
    uint16_t ta_gate[12]; int32_t fa_gate[12*65], y_attn[12*65];
    cTFA_ta_module(ctfa_in, 12, 65, CH_DEC_XMB1,
                   decoder_de_convs_3_pconv2_2_ta_gru_weight_ih_l0, decoder_de_convs_3_pconv2_2_ta_gru_bias_ih_l0,
                   decoder_de_convs_3_pconv2_2_ta_gru_weight_hh_l0, decoder_de_convs_3_pconv2_2_ta_gru_bias_hh_l0,
                   decoder_de_convs_3_pconv2_2_ta_fc_weight, decoder_de_convs_3_pconv2_2_ta_fc_bias,
                   ta_h, ta_gate, ta1, ta2, tfc);
    cTFA_fa_module(ctfa_in, 12, 65,
                   decoder_de_convs_3_pconv2_2_fa_gru_weight_ih_l0, decoder_de_convs_3_pconv2_2_fa_gru_bias_ih_l0,
                   decoder_de_convs_3_pconv2_2_fa_gru_weight_hh_l0, decoder_de_convs_3_pconv2_2_fa_gru_bias_hh_l0,
                   decoder_de_convs_3_pconv2_2_fa_gru_weight_ih_l0_reverse, decoder_de_convs_3_pconv2_2_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_3_pconv2_2_fa_gru_weight_hh_l0_reverse, decoder_de_convs_3_pconv2_2_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_3_pconv2_2_fa_fc_weight, decoder_de_convs_3_pconv2_2_fa_fc_bias,
                   fa_gate, fa1, fa2, ffc);
    cTFA_apply(ctfa_in, ta_gate, fa_gate, 12, 65, y_attn);
    shuffle_deinterleave(y_attn, 6, 65, y);
}

static void run_d4_full(const int32_t *ctfa_in, int ta1, int ta2, int tfc, int fa1, int fa2, int ffc,
                        int16_t *ta_h, int32_t *y) {
    uint16_t ta_gate[1]; int32_t fa_gate[1*129];
    cTFA_ta_module(ctfa_in, 1, 129, CH_DEC_XCONV,
                   decoder_de_convs_4_ops_4_ta_gru_weight_ih_l0, decoder_de_convs_4_ops_4_ta_gru_bias_ih_l0,
                   decoder_de_convs_4_ops_4_ta_gru_weight_hh_l0, decoder_de_convs_4_ops_4_ta_gru_bias_hh_l0,
                   decoder_de_convs_4_ops_4_ta_fc_weight, decoder_de_convs_4_ops_4_ta_fc_bias,
                   ta_h, ta_gate, ta1, ta2, tfc);
    cTFA_fa_module(ctfa_in, 1, 129,
                   decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0, decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0,
                   decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0, decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0,
                   decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0_reverse, decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0_reverse, decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_4_ops_4_fa_fc_weight, decoder_de_convs_4_ops_4_fa_fc_bias,
                   fa_gate, fa1, fa2, ffc);
    cTFA_apply(ctfa_in, ta_gate, fa_gate, 1, 129, y);
}

/* ================================================================
 * Main
 * ================================================================ */
int main() {
    printf("=== Decoder cTFA Full-Chain Calibration ===\n");
    printf("Uses C encoder+DPRNN chain → captures real cTFA inputs\n\n");

    const char *dir = "dump_matlab/";

    /* Load STFT input */
    float *real_in = load_float("dump_matlab/frame0_stft_real.bin", 257);
    float *imag_in = load_float("dump_matlab/frame0_stft_imag.bin", 257);
    if (!real_in || !imag_in) { printf("ERROR: missing STFT input\n"); return 1; }

    /* log_gen + BM */
    int32_t x_log[1*257]; log_gen_fixed(real_in, imag_in, 257, x_log);
    int32_t x_bm[1*129]; BM_fixed(x_log, erb_erb_fc_weight, x_bm);

    /* Encoder */
    ulunas_state_t st; ulunas_state_init(&st);
    int32_t e0[12*65], e1[24*33], e2[24*33], e3[32*33], e4[16*33];
    Encoder_module(x_bm, &st, e0, e1, e2, e3, e4);
    printf("Encoder done.\n");

    /* DPRNN */
    int32_t r1[16*33], r2[16*33];
    GDPRNN_module(e4, st.inter_prev0, 0, r1);
    GDPRNN_module(r1, st.inter_prev1, 1, r2);
    printf("DPRNN done.\n\n");

    /* Load golden module outputs */
    int32_t *gd0 = load_int32("dump_matlab/frame0_dec_d0.bin", 32*33);
    int32_t *gd1 = load_int32("dump_matlab/frame0_dec_d1.bin", 24*33);
    int32_t *gd2 = load_int32("dump_matlab/frame0_dec_d2.bin", 24*33);
    int32_t *gd3 = load_int32("dump_matlab/frame0_dec_d3.bin", 12*65);
    int32_t *gd4 = load_int32("dump_matlab/frame0_dec.bin", 1*129);  /* d4 == final decoder output */
    if (!gd0 || !gd1 || !gd2 || !gd3 || !gd4) { printf("ERROR: missing golden files\n"); return 1; }

    /* ================================================================
     * d0: De_XDWS0
     * ================================================================ */
    printf("--- d0 De_XDWS0 ---\n");
    {
        int32_t *ctfa_in = capture_d0_ctfa_in(r2, e4);

        dec_ctfa_calib_t m = {
            .ctfa_in = ctfa_in, .C = 32, .W = 33, .hidden_dim = CH_DEC_XDWS0,
            .ta_ih_w = decoder_de_convs_0_dconv_4_ta_gru_weight_ih_l0,
            .ta_ih_b = decoder_de_convs_0_dconv_4_ta_gru_bias_ih_l0,
            .ta_hh_w = decoder_de_convs_0_dconv_4_ta_gru_weight_hh_l0,
            .ta_hh_b = decoder_de_convs_0_dconv_4_ta_gru_bias_hh_l0,
            .ta_fc_w = decoder_de_convs_0_dconv_4_ta_fc_weight,
            .ta_fc_b = decoder_de_convs_0_dconv_4_ta_fc_bias,
            .fa_ih_w = decoder_de_convs_0_dconv_4_fa_gru_weight_ih_l0,
            .fa_ih_b = decoder_de_convs_0_dconv_4_fa_gru_bias_ih_l0,
            .fa_hh_w = decoder_de_convs_0_dconv_4_fa_gru_weight_hh_l0,
            .fa_hh_b = decoder_de_convs_0_dconv_4_fa_gru_bias_hh_l0,
            .fa_re_ih_w = decoder_de_convs_0_dconv_4_fa_gru_weight_ih_l0_reverse,
            .fa_re_ih_b = decoder_de_convs_0_dconv_4_fa_gru_bias_ih_l0_reverse,
            .fa_re_hh_w = decoder_de_convs_0_dconv_4_fa_gru_weight_hh_l0_reverse,
            .fa_re_hh_b = decoder_de_convs_0_dconv_4_fa_gru_bias_hh_l0_reverse,
            .fa_fc_w = decoder_de_convs_0_dconv_4_fa_fc_weight,
            .fa_fc_b = decoder_de_convs_0_dconv_4_fa_fc_bias,
            .golden_out = gd0,
            .def_ta_qr1 = -13, .def_ta_qr2 = -8, .def_ta_fc = -9,
            .def_fa_qr1 = -12, .def_fa_qr2 = -7, .def_fa_fc = -9,
            .name = "dec_d0", .shuffle_half_C = 0
        };
        calibrate_one(&m);

        /* Generate d0 output with best QRs for downstream */
        int16_t ta_h[CH_DEC_XDWS0]; memset(ta_h, 0, sizeof(ta_h));
        run_d0_full(ctfa_in, m.def_ta_qr1, m.def_ta_qr2, m.def_ta_fc,
                    m.def_fa_qr1, m.def_fa_qr2, m.def_fa_fc, ta_h, gd0);
        free(ctfa_in);
    }

    /* ================================================================
     * d1: De_XMB0 (uses d0 output from above)
     * ================================================================ */
    printf("--- d1 De_XMB0 ---\n");
    {
        int32_t *ctfa_in = capture_d1_ctfa_in(gd0, e3);

        dec_ctfa_calib_t m = {
            .ctfa_in = ctfa_in, .C = 24, .W = 33, .hidden_dim = CH_DEC_XMB0,
            .ta_ih_w = decoder_de_convs_1_pconv2_2_ta_gru_weight_ih_l0,
            .ta_ih_b = decoder_de_convs_1_pconv2_2_ta_gru_bias_ih_l0,
            .ta_hh_w = decoder_de_convs_1_pconv2_2_ta_gru_weight_hh_l0,
            .ta_hh_b = decoder_de_convs_1_pconv2_2_ta_gru_bias_hh_l0,
            .ta_fc_w = decoder_de_convs_1_pconv2_2_ta_fc_weight,
            .ta_fc_b = decoder_de_convs_1_pconv2_2_ta_fc_bias,
            .fa_ih_w = decoder_de_convs_1_pconv2_2_fa_gru_weight_ih_l0,
            .fa_ih_b = decoder_de_convs_1_pconv2_2_fa_gru_bias_ih_l0,
            .fa_hh_w = decoder_de_convs_1_pconv2_2_fa_gru_weight_hh_l0,
            .fa_hh_b = decoder_de_convs_1_pconv2_2_fa_gru_bias_hh_l0,
            .fa_re_ih_w = decoder_de_convs_1_pconv2_2_fa_gru_weight_ih_l0_reverse,
            .fa_re_ih_b = decoder_de_convs_1_pconv2_2_fa_gru_bias_ih_l0_reverse,
            .fa_re_hh_w = decoder_de_convs_1_pconv2_2_fa_gru_weight_hh_l0_reverse,
            .fa_re_hh_b = decoder_de_convs_1_pconv2_2_fa_gru_bias_hh_l0_reverse,
            .fa_fc_w = decoder_de_convs_1_pconv2_2_fa_fc_weight,
            .fa_fc_b = decoder_de_convs_1_pconv2_2_fa_fc_bias,
            .golden_out = gd1,
            .def_ta_qr1 = -13, .def_ta_qr2 = -8, .def_ta_fc = -9,
            .def_fa_qr1 = -12, .def_fa_qr2 = -7, .def_fa_fc = -9,
            .name = "dec_d1", .shuffle_half_C = 12
        };
        calibrate_one(&m);

        int16_t ta_h[CH_DEC_XMB0]; memset(ta_h, 0, sizeof(ta_h));
        run_d1_full(ctfa_in, m.def_ta_qr1, m.def_ta_qr2, m.def_ta_fc,
                    m.def_fa_qr1, m.def_fa_qr2, m.def_fa_fc, ta_h, gd1);
        free(ctfa_in);
    }

    /* ================================================================
     * d2: De_XDWS1 (uses d1 output from above)
     * ================================================================ */
    printf("--- d2 De_XDWS1 ---\n");
    {
        int32_t dec_xdws1_cache[1 * CH_XMB0 * N_BINS_SMALL];
        memset(dec_xdws1_cache, 0, sizeof(dec_xdws1_cache));
        int32_t *ctfa_in = capture_d2_ctfa_in(gd1, e2, dec_xdws1_cache);

        dec_ctfa_calib_t m = {
            .ctfa_in = ctfa_in, .C = 24, .W = 33, .hidden_dim = CH_DEC_XMB0,
            .ta_ih_w = decoder_de_convs_2_dconv_4_ta_gru_weight_ih_l0,
            .ta_ih_b = decoder_de_convs_2_dconv_4_ta_gru_bias_ih_l0,
            .ta_hh_w = decoder_de_convs_2_dconv_4_ta_gru_weight_hh_l0,
            .ta_hh_b = decoder_de_convs_2_dconv_4_ta_gru_bias_hh_l0,
            .ta_fc_w = decoder_de_convs_2_dconv_4_ta_fc_weight,
            .ta_fc_b = decoder_de_convs_2_dconv_4_ta_fc_bias,
            .fa_ih_w = decoder_de_convs_2_dconv_4_fa_gru_weight_ih_l0,
            .fa_ih_b = decoder_de_convs_2_dconv_4_fa_gru_bias_ih_l0,
            .fa_hh_w = decoder_de_convs_2_dconv_4_fa_gru_weight_hh_l0,
            .fa_hh_b = decoder_de_convs_2_dconv_4_fa_gru_bias_hh_l0,
            .fa_re_ih_w = decoder_de_convs_2_dconv_4_fa_gru_weight_ih_l0_reverse,
            .fa_re_ih_b = decoder_de_convs_2_dconv_4_fa_gru_bias_ih_l0_reverse,
            .fa_re_hh_w = decoder_de_convs_2_dconv_4_fa_gru_weight_hh_l0_reverse,
            .fa_re_hh_b = decoder_de_convs_2_dconv_4_fa_gru_bias_hh_l0_reverse,
            .fa_fc_w = decoder_de_convs_2_dconv_4_fa_fc_weight,
            .fa_fc_b = decoder_de_convs_2_dconv_4_fa_fc_bias,
            .golden_out = gd2,
            .def_ta_qr1 = -13, .def_ta_qr2 = -8, .def_ta_fc = -8,
            .def_fa_qr1 = -13, .def_fa_qr2 = -8, .def_fa_fc = -9,
            .name = "dec_d2", .shuffle_half_C = 0
        };
        calibrate_one(&m);

        int16_t ta_h[CH_DEC_XMB0]; memset(ta_h, 0, sizeof(ta_h));
        run_d2_full(ctfa_in, m.def_ta_qr1, m.def_ta_qr2, m.def_ta_fc,
                    m.def_fa_qr1, m.def_fa_qr2, m.def_fa_fc, ta_h, gd2);
        free(ctfa_in);
    }

    /* ================================================================
     * d3: De_XMB1 (uses d2 output from above)
     * ================================================================ */
    printf("--- d3 De_XMB1 ---\n");
    {
        int32_t dec_xmb1_cache[1 * CH_XCONV * N_BINS_SMALL];
        memset(dec_xmb1_cache, 0, sizeof(dec_xmb1_cache));
        int32_t *ctfa_in = capture_d3_ctfa_in(gd2, e1, dec_xmb1_cache);

        dec_ctfa_calib_t m = {
            .ctfa_in = ctfa_in, .C = 12, .W = 65, .hidden_dim = CH_DEC_XMB1,
            .ta_ih_w = decoder_de_convs_3_pconv2_2_ta_gru_weight_ih_l0,
            .ta_ih_b = decoder_de_convs_3_pconv2_2_ta_gru_bias_ih_l0,
            .ta_hh_w = decoder_de_convs_3_pconv2_2_ta_gru_weight_hh_l0,
            .ta_hh_b = decoder_de_convs_3_pconv2_2_ta_gru_bias_hh_l0,
            .ta_fc_w = decoder_de_convs_3_pconv2_2_ta_fc_weight,
            .ta_fc_b = decoder_de_convs_3_pconv2_2_ta_fc_bias,
            .fa_ih_w = decoder_de_convs_3_pconv2_2_fa_gru_weight_ih_l0,
            .fa_ih_b = decoder_de_convs_3_pconv2_2_fa_gru_bias_ih_l0,
            .fa_hh_w = decoder_de_convs_3_pconv2_2_fa_gru_weight_hh_l0,
            .fa_hh_b = decoder_de_convs_3_pconv2_2_fa_gru_bias_hh_l0,
            .fa_re_ih_w = decoder_de_convs_3_pconv2_2_fa_gru_weight_ih_l0_reverse,
            .fa_re_ih_b = decoder_de_convs_3_pconv2_2_fa_gru_bias_ih_l0_reverse,
            .fa_re_hh_w = decoder_de_convs_3_pconv2_2_fa_gru_weight_hh_l0_reverse,
            .fa_re_hh_b = decoder_de_convs_3_pconv2_2_fa_gru_bias_hh_l0_reverse,
            .fa_fc_w = decoder_de_convs_3_pconv2_2_fa_fc_weight,
            .fa_fc_b = decoder_de_convs_3_pconv2_2_fa_fc_bias,
            .golden_out = gd3,
            .def_ta_qr1 = -13, .def_ta_qr2 = -8, .def_ta_fc = -9,
            .def_fa_qr1 = -11, .def_fa_qr2 = -6, .def_fa_fc = -9,
            .name = "dec_d3", .shuffle_half_C = 6
        };
        calibrate_one(&m);

        int16_t ta_h[CH_DEC_XMB1]; memset(ta_h, 0, sizeof(ta_h));
        run_d3_full(ctfa_in, m.def_ta_qr1, m.def_ta_qr2, m.def_ta_fc,
                    m.def_fa_qr1, m.def_fa_qr2, m.def_fa_fc, ta_h, gd3);
        free(ctfa_in);
    }

    /* ================================================================
     * d4: De_XConv (uses d3 output from above)
     * ================================================================ */
    printf("--- d4 De_XConv ---\n");
    {
        int32_t dec_xconv_cache[CACHE_DEXCONV_ROWS * CH_XCONV * N_BINS_MID];
        memset(dec_xconv_cache, 0, sizeof(dec_xconv_cache));
        int32_t *ctfa_in = capture_d4_ctfa_in(gd3, e0, dec_xconv_cache);

        dec_ctfa_calib_t m = {
            .ctfa_in = ctfa_in, .C = 1, .W = 129, .hidden_dim = CH_DEC_XCONV,
            .ta_ih_w = decoder_de_convs_4_ops_4_ta_gru_weight_ih_l0,
            .ta_ih_b = decoder_de_convs_4_ops_4_ta_gru_bias_ih_l0,
            .ta_hh_w = decoder_de_convs_4_ops_4_ta_gru_weight_hh_l0,
            .ta_hh_b = decoder_de_convs_4_ops_4_ta_gru_bias_hh_l0,
            .ta_fc_w = decoder_de_convs_4_ops_4_ta_fc_weight,
            .ta_fc_b = decoder_de_convs_4_ops_4_ta_fc_bias,
            .fa_ih_w = decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0,
            .fa_ih_b = decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0,
            .fa_hh_w = decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0,
            .fa_hh_b = decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0,
            .fa_re_ih_w = decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0_reverse,
            .fa_re_ih_b = decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0_reverse,
            .fa_re_hh_w = decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0_reverse,
            .fa_re_hh_b = decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0_reverse,
            .fa_fc_w = decoder_de_convs_4_ops_4_fa_fc_weight,
            .fa_fc_b = decoder_de_convs_4_ops_4_fa_fc_bias,
            .golden_out = gd4,
            .def_ta_qr1 = -13, .def_ta_qr2 = -8, .def_ta_fc = -8,
            .def_fa_qr1 = -13, .def_fa_qr2 = -8, .def_fa_fc = -9,
            .name = "dec_d4", .shuffle_half_C = 0
        };
        calibrate_one(&m);
        free(ctfa_in);
    }

    printf("=== Done ===\n");
    free(real_in); free(imag_in);
    return 0;
}
