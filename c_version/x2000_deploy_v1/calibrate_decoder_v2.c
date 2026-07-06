/**
 * calibrate_decoder_v2.c — Decoder per-layer QR calibration
 * ==========================================================
 * Uses MATLAB-exported decoder TConv golden files.
 *
 * Build: gcc -O2 -o calibrate_decoder_v2 calibrate_decoder_v2.c
 *        ulunas_fp.c ulunas_matlab_weights.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"

static double snr_db(const int32_t *g, const int32_t *t, int n) {
    double s = 0, e = 0;
    for (int i = 0; i < n; i++) { double gv = g[i], d = gv - t[i]; s += gv * gv; e += d * d; }
    return e < 1e-30 ? 999 : 10 * log10(s / e);
}

/* ===== Calibration replicas of decoder conv blocks with adjustable QR ===== */

/* nonGTConv conv only (non_gtconv2d with adjustable qr) */
static void nongtconv_cal(const int32_t *x, int Cin, int Win,
                           const int16_t *conv_w, const int32_t *conv_b,
                           int Cout, int Wout, int Hk, int Wk,
                           int stride, int groups, int conv_qr,
                           int32_t *y_conv) {
    int C_per_group = Cin / groups;
    int Cout_per_group = Cout / groups;
    int pad_w = 2;
    int W_insert = Win + (Win - 1) * (stride - 1);
    int W_padded = W_insert + 2 * pad_w;
    int shift = -conv_qr;
    for (int g = 0; g < groups; g++) {
        for (int ci_l = 0; ci_l < C_per_group; ci_l++) {
            int ci = g * C_per_group + ci_l;
            const int32_t *x_chan = x + ci * Win;
            int32_t *x_insert = (int32_t *)calloc(W_padded, sizeof(int32_t));
            for (int w = 0; w < Win; w++) x_insert[pad_w + w * stride] = x_chan[w];
            for (int co_l = 0; co_l < Cout_per_group; co_l++) {
                int co = g * Cout_per_group + co_l;
                if (ci_l == 0) for (int wo = 0; wo < Wout; wo++) y_conv[co * Wout + wo] = conv_b[co];
                for (int wo = 0; wo < Wout; wo++) {
                    int64_t acc = 0;
                    for (int hk = 0; hk < Hk; hk++) {
                        for (int wk = 0; wk < Wk; wk++) {
                            int wi = wo + wk;
                            int32_t xv = (wi >= 0 && wi < W_padded) ? x_insert[wi] : 0;
                            int wk_rev = Wk - 1 - wk;
                            int kidx = ((ci_l * Cout_per_group + co_l) * Hk + hk) * Wk + wk_rev;
                            int g_off = g * C_per_group * Cout_per_group * Hk * Wk;
                            int64_t prod = (int64_t)xv * (int64_t)conv_w[g_off + kidx];
                            prod = (prod + ((int64_t)1 << (shift - 1))) >> shift;
                            acc += prod;
                        }
                    }
                    y_conv[co * Wout + wo] = sat32((int64_t)y_conv[co * Wout + wo] + acc);
                }
            }
            free(x_insert);
        }
    }
}

/* GTConv conv only (gtconv2d with adjustable qr) */
static void gtconv_cal(const int32_t *x, int Cin, int Win,
                        const int16_t *conv_w, const int32_t *conv_b,
                        int Cout, int Wout, int Hk, int Wk,
                        int stride, int groups, int conv_qr,
                        int32_t *y_conv) {
    nongtconv_cal(x, Cin, Win, conv_w, conv_b, Cout, Wout, Hk, Wk, stride, groups, conv_qr, y_conv);
}

/* BN with adjustable qr1, qr2 */
static void bn_cal(int32_t *x, int C, int Win,
                    const uint16_t *weight, const int32_t *bias,
                    const int32_t *running_mean, const uint16_t *running_var,
                    int qr1, int qr2) {
    for (int c = 0; c < C; c++) {
        int32_t *ch = x + c * Win;
        for (int w = 0; w < Win; w++) {
            int64_t diff = (int64_t)ch[w] - (int64_t)running_mean[c];
            int64_t norm = diff * (int64_t)running_var[c];
            int shift1 = -qr1;
            int32_t x_norm = (int32_t)((norm + ((int64_t)1 << (shift1 - 1))) >> shift1);
            int64_t scaled = (int64_t)x_norm * (int64_t)weight[c];
            int shift2 = -qr2;
            int32_t y_val = (int32_t)((scaled + ((int64_t)1 << (shift2 - 1))) >> shift2);
            ch[w] = sat32((int64_t)y_val + (int64_t)bias[c]);
        }
    }
}

/* AffinePReLU (qr=-13,-13 always) */
static void ap_cal(int32_t *x, int C, int Win,
                    const int16_t *affine_weight, const int32_t *affine_bias,
                    const int16_t *slope_weight) {
    for (int c = 0; c < C; c++) {
        int32_t *ch = x + c * Win;
        int16_t wt = affine_weight[c];
        int32_t bias_c = affine_bias[c];
        int16_t slope = slope_weight[c];
        for (int w = 0; w < Win; w++) {
            int32_t x_orig = ch[w];
            int32_t x_mod;
            if (x_orig < 0) {
                int64_t neg_prod = (int64_t)x_orig * (int64_t)slope;
                x_mod = (int32_t)((neg_prod + ((int64_t)1 << 12)) >> 13);
            } else { x_mod = x_orig; }
            int64_t affine_prod = (int64_t)x_orig * (int64_t)wt;
            int32_t affine_val = (int32_t)((affine_prod + ((int64_t)1 << 12)) >> 13);
            ch[w] = sat32((int64_t)affine_val + (int64_t)bias_c + (int64_t)x_mod);
        }
    }
}

int main() {
    /* Load STFT frame0 */
    float ri[257], ii[257];
    { FILE *f = fopen("dump_matlab/frame0_stft_real.bin", "rb"); fread(ri, sizeof(float), 257, f); fclose(f); }
    { FILE *f = fopen("dump_matlab/frame0_stft_imag.bin", "rb"); fread(ii, sizeof(float), 257, f); fclose(f); }

    /* log_gen → BM */
    int32_t xl[257]; log_gen_fixed(ri, ii, 257, xl);
    int32_t xb[129]; BM_fixed(xl, erb_erb_fc_weight, xb);

    /* Load all golden intermediate files */
    int32_t *ge0, *ge1, *ge2, *ge3, *ge4, *gr2;
    ge0 = malloc(12*65*4); { FILE *f = fopen("dump_matlab/frame0_enc_e0.bin","rb"); fread(ge0,4,12*65,f); fclose(f); }
    ge1 = malloc(24*33*4); { FILE *f = fopen("dump_matlab/frame0_enc_e1.bin","rb"); fread(ge1,4,24*33,f); fclose(f); }
    ge2 = malloc(24*33*4); { FILE *f = fopen("dump_matlab/frame0_enc_e2.bin","rb"); fread(ge2,4,24*33,f); fclose(f); }
    ge3 = malloc(32*33*4); { FILE *f = fopen("dump_matlab/frame0_enc_e3.bin","rb"); fread(ge3,4,32*33,f); fclose(f); }
    ge4 = malloc(16*33*4); { FILE *f = fopen("dump_matlab/frame0_enc_e4.bin","rb"); fread(ge4,4,16*33,f); fclose(f); }
    gr2 = malloc(16*33*4); { FILE *f = fopen("dump_matlab/frame0_rnn2.bin","rb"); fread(gr2,4,16*33,f); fclose(f); }

    /* Load decoder TConv goldens */
    int32_t gd0[32*33], gd1[24*33], gd2[24*33], gd3[12*65], gd4[1*129];
    { FILE *f = fopen("dump_matlab/frame0_dec_d0_tconv.bin","rb"); fread(gd0,4,32*33,f); fclose(f); }
    { FILE *f = fopen("dump_matlab/frame0_dec_d1_tconv.bin","rb"); fread(gd1,4,24*33,f); fclose(f); }
    { FILE *f = fopen("dump_matlab/frame0_dec_d2_tconv.bin","rb"); fread(gd2,4,24*33,f); fclose(f); }
    { FILE *f = fopen("dump_matlab/frame0_dec_d3_tconv.bin","rb"); fread(gd3,4,12*65,f); fclose(f); }
    { FILE *f = fopen("dump_matlab/frame0_dec_d4_tconv.bin","rb"); fread(gd4,4,1*129,f); fclose(f); }

    printf("=== Decoder Per-Layer QR Calibration (Frame 0) ===\n\n");

    /* ====================================================================
     * d0: De_XDWS0 nonGTConv_block
     * Input: rnn2(16,33) + e4(16,33) → concat(32,33) → PConv + shuffle → y_shuf(32,33)
     * nonGTConv: Cin=32, Cout=32, Hk=1, Wk=5, stride=1, groups=32
     * ==================================================================== */
    printf("--- d0 De_XDWS0 nonGTConv: Cin=32 Cout=32 Win=33 Wout=33 Hk=1 Wk=5 stride=1 groups=32 ---\n");
    {
        /* Build input: x_cat = rnn2 + e4 */
        int32_t x_cat[32*33];
        for (int i = 0; i < 16*33; i++) {
            x_cat[i] = sat32((int64_t)gr2[i] + (int64_t)ge4[i]);
            x_cat[16*33+i] = sat32((int64_t)gr2[16*33+i] + (int64_t)ge4[16*33+i]);
        }
        /* PConv0: group0 Cin=8→Cout=16, group1 Cin=8→Cout=16, total 16→32 */
        int32_t y_pconv[32*33];
        { int32_t y0[16*33], y1[16*33];
          pconv2d_fixed(x_cat, 8, 33, decoder_de_convs_0_pconv_0_weight, decoder_de_convs_0_pconv_0_bias, 16, -14, y0);
          pconv2d_fixed(x_cat+8*33, 8, 33, decoder_de_convs_0_pconv_0_weight+16*8, decoder_de_convs_0_pconv_0_bias+16, 16, -14, y1);
          for (int c = 0; c < 16; c++) {
              memcpy(y_pconv + c * 33, y0 + c * 33, 33 * sizeof(int32_t));
              memcpy(y_pconv + (16 + c) * 33, y1 + c * 33, 33 * sizeof(int32_t));
          }
          bn_fixed(y_pconv, 32, 33, decoder_de_convs_0_pconv_1_weight, decoder_de_convs_0_pconv_1_bias,
                   decoder_de_convs_0_pconv_1_running_mean, decoder_de_convs_0_pconv_1_running_var, -14, -14);
          affine_prelu_fixed(y_pconv, 32, 33, decoder_de_convs_0_pconv_2_affine_weight,
                             decoder_de_convs_0_pconv_2_affine_bias, decoder_de_convs_0_pconv_2_slope_weight, -13, -13);
        }
        int32_t y_shuf[32*33]; shuffle_interleave(y_pconv, 16, 33, y_shuf);

        int Cin = 32, Cout = 32, Win = 33, Wout = 33, Hk = 1, Wk = 5, stride = 1, groups = 32;
        double best = -999; int bc = -13, bb1 = -14, bb2 = -14;
        for (int cqr = -12; cqr >= -17; cqr--) {
            for (int bqr1 = -11; bqr1 >= -18; bqr1--) {
                for (int bqr2 = -11; bqr2 >= -16; bqr2--) {
                    int32_t y_conv[Cout * Wout]; memset(y_conv, 0, sizeof(y_conv));
                    nongtconv_cal(y_shuf, Cin, Win, decoder_de_convs_0_dconv_1_weight, decoder_de_convs_0_dconv_1_bias,
                                  Cout, Wout, Hk, Wk, stride, groups, cqr, y_conv);
                    bn_cal(y_conv, Cout, Wout, decoder_de_convs_0_dconv_2_weight, decoder_de_convs_0_dconv_2_bias,
                           decoder_de_convs_0_dconv_2_running_mean, decoder_de_convs_0_dconv_2_running_var, bqr1, bqr2);
                    ap_cal(y_conv, Cout, Wout, decoder_de_convs_0_dconv_3_affine_weight, decoder_de_convs_0_dconv_3_affine_bias,
                           decoder_de_convs_0_dconv_3_slope_weight);
                    double sn = snr_db(gd0, y_conv, Cout * Wout);
                    if (sn > best) { best = sn; bc = cqr; bb1 = bqr1; bb2 = bqr2; }
                }
            }
        }
        printf("  Best: conv_qr=%d bn_qr1=%d bn_qr2=%d SNR=%.2f dB\n\n", bc, bb1, bb2, best);
    }

    /* ====================================================================
     * d1: De_XMB0 nonGTConv_block
     * Input: d0_full_output(32,33) + e3(32,33) → PConv0(32→24) → shuffle → y_shuf(24,33)
     * nonGTConv: Cin=24, Cout=24, Hk=1, Wk=5, stride=1, groups=24
     * ==================================================================== */
    printf("--- d1 De_XMB0 nonGTConv: Cin=24 Cout=24 Win=33 Wout=33 Hk=1 Wk=5 stride=1 groups=24 ---\n");
    {
        /* Load d0_full_output golden (frame0_dec_d0.bin) */
        int32_t d0_full[32*33];
        { FILE *f = fopen("dump_matlab/frame0_dec_d0.bin","rb");
          if (f) { fread(d0_full,4,32*33,f); fclose(f); }
          else { fprintf(stderr,"WARNING: no frame0_dec_d0.bin, using rnn2+e4 as approx\n");
                 for(int i=0;i<32*33;i++) d0_full[i]=0; }
        }

        int32_t x_cat[32*33];
        for (int i = 0; i < 32*33; i++) x_cat[i] = sat32((int64_t)d0_full[i] + (int64_t)ge3[i]);

        /* PConv0: Cin=16 per group, Cout=12 per group, total 32→24 */
        int32_t y_pconv0[24*33];
        { int32_t y0[12*33], y1[12*33];
          pconv2d_fixed(x_cat, 16, 33, decoder_de_convs_1_pconv1_0_weight, decoder_de_convs_1_pconv1_0_bias, 12, -13, y0);
          pconv2d_fixed(x_cat+16*33, 16, 33, decoder_de_convs_1_pconv1_0_weight+12*16, decoder_de_convs_1_pconv1_0_bias+12, 12, -13, y1);
          for (int c = 0; c < 12; c++) {
              memcpy(y_pconv0 + c * 33, y0 + c * 33, 33 * sizeof(int32_t));
              memcpy(y_pconv0 + (12 + c) * 33, y1 + c * 33, 33 * sizeof(int32_t));
          }
          bn_fixed(y_pconv0, 24, 33, decoder_de_convs_1_pconv1_1_weight, decoder_de_convs_1_pconv1_1_bias,
                   decoder_de_convs_1_pconv1_1_running_mean, decoder_de_convs_1_pconv1_1_running_var, -11, -14);
          affine_prelu_fixed(y_pconv0, 24, 33, decoder_de_convs_1_pconv1_2_affine_weight,
                             decoder_de_convs_1_pconv1_2_affine_bias, decoder_de_convs_1_pconv1_2_slope_weight, -13, -13);
        }
        int32_t y_shuf[24*33]; shuffle_interleave(y_pconv0, 12, 33, y_shuf);

        int Cin = 24, Cout = 24, Win = 33, Wout = 33, Hk = 1, Wk = 5, stride = 1, groups = 24;
        double best = -999; int bc = -13, bb1 = -14, bb2 = -14;
        for (int cqr = -12; cqr >= -17; cqr--) {
            for (int bqr1 = -11; bqr1 >= -18; bqr1--) {
                for (int bqr2 = -11; bqr2 >= -16; bqr2--) {
                    int32_t y_conv[Cout * Wout]; memset(y_conv, 0, sizeof(y_conv));
                    nongtconv_cal(y_shuf, Cin, Win, decoder_de_convs_1_dconv_1_weight, decoder_de_convs_1_dconv_1_bias,
                                  Cout, Wout, Hk, Wk, stride, groups, cqr, y_conv);
                    bn_cal(y_conv, Cout, Wout, decoder_de_convs_1_dconv_2_weight, decoder_de_convs_1_dconv_2_bias,
                           decoder_de_convs_1_dconv_2_running_mean, decoder_de_convs_1_dconv_2_running_var, bqr1, bqr2);
                    ap_cal(y_conv, Cout, Wout, decoder_de_convs_1_dconv_3_affine_weight, decoder_de_convs_1_dconv_3_affine_bias,
                           decoder_de_convs_1_dconv_3_slope_weight);
                    double sn = snr_db(gd1, y_conv, Cout * Wout);
                    if (sn > best) { best = sn; bc = cqr; bb1 = bqr1; bb2 = bqr2; }
                }
            }
        }
        printf("  Best: conv_qr=%d bn_qr1=%d bn_qr2=%d SNR=%.2f dB\n\n", bc, bb1, bb2, best);
    }

    /* ====================================================================
     * d2: De_XDWS1 GTConv_block
     * Input: d1_full_output(24,33) + e2(24,33) → PConv(24→24) → shuffle → y_shuf(24,33)
     * GTConv: Cin=24, Cout=24, Hk=1, Wk=3, stride=1, groups=24, cache_rows=1
     * ==================================================================== */
    printf("--- d2 De_XDWS1 GTConv: Cin=24 Cout=24 Win=33 Wout=33 Hk=1 Wk=3 stride=1 groups=24 ---\n");
    {
        int32_t d1_full[24*33];
        { FILE *f = fopen("dump_matlab/frame0_dec_d1.bin","rb");
          if (f) { fread(d1_full,4,24*33,f); fclose(f); }
          else { for(int i=0;i<24*33;i++) d1_full[i]=0; }
        }

        int32_t x_cat[24*33];
        for (int i = 0; i < 24*33; i++) x_cat[i] = sat32((int64_t)d1_full[i] + (int64_t)ge2[i]);

        int32_t y_pconv[24*33];
        { int32_t y0[12*33], y1[12*33];
          pconv2d_fixed(x_cat, 12, 33, decoder_de_convs_2_pconv_0_weight, decoder_de_convs_2_pconv_0_bias, 12, -14, y0);
          pconv2d_fixed(x_cat+12*33, 12, 33, decoder_de_convs_2_pconv_0_weight+12*12, decoder_de_convs_2_pconv_0_bias+12, 12, -14, y1);
          for (int c = 0; c < 12; c++) {
              memcpy(y_pconv + c * 33, y0 + c * 33, 33 * sizeof(int32_t));
              memcpy(y_pconv + (12 + c) * 33, y1 + c * 33, 33 * sizeof(int32_t));
          }
          bn_fixed(y_pconv, 24, 33, decoder_de_convs_2_pconv_1_weight, decoder_de_convs_2_pconv_1_bias,
                   decoder_de_convs_2_pconv_1_running_mean, decoder_de_convs_2_pconv_1_running_var, -11, -14);
          affine_prelu_fixed(y_pconv, 24, 33, decoder_de_convs_2_pconv_2_affine_weight,
                             decoder_de_convs_2_pconv_2_affine_bias, decoder_de_convs_2_pconv_2_slope_weight, -13, -13);
        }
        int32_t y_shuf[24*33]; shuffle_interleave(y_pconv, 12, 33, y_shuf);

        /* GTConv: first frame, cache is zero */
        int Cin = 24, Cout = 24, Win = 33, Wout = 33, Hk = 1, Wk = 3, stride = 1, groups = 24;
        double best = -999; int bc = -13, bb1 = -14, bb2 = -14;
        for (int cqr = -12; cqr >= -17; cqr--) {
            for (int bqr1 = -11; bqr1 >= -18; bqr1--) {
                for (int bqr2 = -11; bqr2 >= -16; bqr2--) {
                    int32_t y_conv[Cout * Wout]; memset(y_conv, 0, sizeof(y_conv));
                    gtconv_cal(y_shuf, Cin, Win, decoder_de_convs_2_dconv_1_weight, decoder_de_convs_2_dconv_1_bias,
                               Cout, Wout, Hk, Wk, stride, groups, cqr, y_conv);
                    bn_cal(y_conv, Cout, Wout, decoder_de_convs_2_dconv_2_weight, decoder_de_convs_2_dconv_2_bias,
                           decoder_de_convs_2_dconv_2_running_mean, decoder_de_convs_2_dconv_2_running_var, bqr1, bqr2);
                    ap_cal(y_conv, Cout, Wout, decoder_de_convs_2_dconv_3_affine_weight, decoder_de_convs_2_dconv_3_affine_bias,
                           decoder_de_convs_2_dconv_3_slope_weight);
                    double sn = snr_db(gd2, y_conv, Cout * Wout);
                    if (sn > best) { best = sn; bc = cqr; bb1 = bqr1; bb2 = bqr2; }
                }
            }
        }
        printf("  Best: conv_qr=%d bn_qr1=%d bn_qr2=%d SNR=%.2f dB\n\n", bc, bb1, bb2, best);
    }

    /* ====================================================================
     * d3: De_XMB1 GTConv_block
     * Input: d2_full_output(24,33) + e1(24,33) → PConv0(24→12) → shuffle → y_shuf(12,33)
     * GTConv: Cin=12, Cout=12, Win=33, Wout=65, Hk=1, Wk=3, stride=2, groups=12, cache_rows=1
     * ==================================================================== */
    printf("--- d3 De_XMB1 GTConv: Cin=12 Cout=12 Win=33 Wout=65 Hk=1 Wk=3 stride=2 groups=12 ---\n");
    {
        int32_t d2_full[24*33];
        { FILE *f = fopen("dump_matlab/frame0_dec_d2.bin","rb");
          if (f) { fread(d2_full,4,24*33,f); fclose(f); }
          else { for(int i=0;i<24*33;i++) d2_full[i]=0; }
        }

        int32_t x_cat[24*33];
        for (int i = 0; i < 24*33; i++) x_cat[i] = sat32((int64_t)d2_full[i] + (int64_t)ge1[i]);

        int32_t y_pconv0[12*33];
        { int32_t y0[6*33], y1[6*33];
          pconv2d_fixed(x_cat, 12, 33, decoder_de_convs_3_pconv1_0_weight, decoder_de_convs_3_pconv1_0_bias, 6, -14, y0);
          pconv2d_fixed(x_cat+12*33, 12, 33, decoder_de_convs_3_pconv1_0_weight+6*12, decoder_de_convs_3_pconv1_0_bias+6, 6, -14, y1);
          for (int c = 0; c < 6; c++) {
              memcpy(y_pconv0 + c * 33, y0 + c * 33, 33 * sizeof(int32_t));
              memcpy(y_pconv0 + (6 + c) * 33, y1 + c * 33, 33 * sizeof(int32_t));
          }
          bn_fixed(y_pconv0, 12, 33, decoder_de_convs_3_pconv1_1_weight, decoder_de_convs_3_pconv1_1_bias,
                   decoder_de_convs_3_pconv1_1_running_mean, decoder_de_convs_3_pconv1_1_running_var, -11, -14);
          affine_prelu_fixed(y_pconv0, 12, 33, decoder_de_convs_3_pconv1_2_affine_weight,
                             decoder_de_convs_3_pconv1_2_affine_bias, decoder_de_convs_3_pconv1_2_slope_weight, -13, -13);
        }
        int32_t y_shuf[12*33]; shuffle_interleave(y_pconv0, 6, 33, y_shuf);

        int Cin = 12, Cout = 12, Win = 33, Wout = 65, Hk = 1, Wk = 3, stride = 2, groups = 12;
        double best = -999; int bc = -14, bb1 = -11, bb2 = -11;
        for (int cqr = -12; cqr >= -17; cqr--) {
            for (int bqr1 = -11; bqr1 >= -18; bqr1--) {
                for (int bqr2 = -11; bqr2 >= -16; bqr2--) {
                    int32_t y_conv[Cout * Wout]; memset(y_conv, 0, sizeof(y_conv));
                    gtconv_cal(y_shuf, Cin, Win, decoder_de_convs_3_dconv_1_weight, decoder_de_convs_3_dconv_1_bias,
                               Cout, Wout, Hk, Wk, stride, groups, cqr, y_conv);
                    bn_cal(y_conv, Cout, Wout, decoder_de_convs_3_dconv_2_weight, decoder_de_convs_3_dconv_2_bias,
                           decoder_de_convs_3_dconv_2_running_mean, decoder_de_convs_3_dconv_2_running_var, bqr1, bqr2);
                    ap_cal(y_conv, Cout, Wout, decoder_de_convs_3_dconv_3_affine_weight, decoder_de_convs_3_dconv_3_affine_bias,
                           decoder_de_convs_3_dconv_3_slope_weight);
                    double sn = snr_db(gd3, y_conv, Cout * Wout);
                    if (sn > best) { best = sn; bc = cqr; bb1 = bqr1; bb2 = bqr2; }
                }
            }
        }
        printf("  Best: conv_qr=%d bn_qr1=%d bn_qr2=%d SNR=%.2f dB\n\n", bc, bb1, bb2, best);
    }

    /* ====================================================================
     * d4: De_XConv hybrid conv(H) + tconv(W)
     * ====================================================================
     * Input: d3_full_output(12,65) + e0(12,65) → x_cat(12,65) → 3D cache → hybrid
     *
     * ✅ CORRECTED 2026-07-06: uses hybrid conv(H)+tconv(W) to match De_XConv_module
     *   - H direction: regular conv, H_in=3 (2 cache + 1 current), Hk=3 → H_out=1
     *   - W direction: transposed conv, Win=65, Wk=3, stride=2, pad=1 → Wout=129
     *   - Weight layout: (Cin=12, Cout=1, Hk=3, Wk=3) tconv convention
     *   - Weight index: ((ci*Cout+co)*Hk + hk)*Wk + wk_rev
     *   - Golden: frame0_dec_d4_tconv.bin (TConv+BN output, before cTFA)
     * ==================================================================== */
    printf("--- d4 De_XConv hybrid: Cin=12 Cout=1 Win=65 Wout=129 Hk=3 Wk=3 stride_w=2 ---\n");
    {
        int32_t d3_full[12*65];
        { FILE *f = fopen("dump_matlab/frame0_dec_d3.bin","rb");
          if (f) { fread(d3_full,4,12*65,f); fclose(f); }
          else { for(int i=0;i<12*65;i++) d3_full[i]=0; }
        }

        int32_t x_cat[12*65];
        for (int i = 0; i < 12*65; i++) x_cat[i] = sat32((int64_t)d3_full[i] + (int64_t)ge0[i]);

        /* Build 3D input: H_in=3 (cache rows 0,1 = zero for frame 0; row 2 = current) */
        int H_in = 3, Cin = 12, Win = 65, Cout = 1, Wout = 129, Hk = 3, Wk = 3;
        int stride_w = 2, pad_w = 1;
        int W_insert = Win + (Win - 1) * (stride_w - 1);  /* 65 + 64 = 129 */
        int W_padded = W_insert + 2 * pad_w;                /* 129 + 2 = 131 */

        int32_t x_full[H_in * Cin * Win];
        memset(x_full, 0, sizeof(x_full));
        /* Cache rows 0,1 are zero for frame 0. Current frame at row 2 */
        for (int c = 0; c < Cin; c++)
            memcpy(x_full + (2 * Cin + c) * Win, x_cat + c * Win, Win * sizeof(int32_t));

        double best = -999; int bc = -14, bb1 = -11, bb2 = -11;
        /* Extended search ranges to cover aggressive quantization */
        for (int cqr = -12; cqr >= -19; cqr--) {
            for (int bqr1 = -9; bqr1 >= -18; bqr1--) {
                for (int bqr2 = -9; bqr2 >= -18; bqr2--) {
                    int32_t y_conv[Cout * Wout]; memset(y_conv, 0, sizeof(y_conv));

                    /* Hybrid conv(H) + tconv(W) — exact replica of De_XConv_module */
                    for (int co = 0; co < Cout; co++) {
                        /* Init with bias */
                        for (int wo = 0; wo < Wout; wo++)
                            y_conv[co * Wout + wo] = decoder_de_convs_4_ops_1_bias[co];

                        for (int ci = 0; ci < Cin; ci++) {
                            for (int hk = 0; hk < Hk; hk++) {
                                /* Build upsampled input for this (ci, hk) temporal slice */
                                const int32_t *frame = x_full + (hk * Cin + ci) * Win;
                                int32_t x_insert[131];  /* W_padded = 131 */
                                memset(x_insert, 0, sizeof(x_insert));
                                for (int w = 0; w < Win; w++)
                                    x_insert[pad_w + w * stride_w] = frame[w];

                                for (int wo = 0; wo < Wout; wo++) {
                                    int64_t acc = 0;
                                    for (int wk = 0; wk < Wk; wk++) {
                                        int wi = wo + wk;
                                        int32_t xv = (wi >= 0 && wi < W_padded) ? x_insert[wi] : 0;
                                        /* MATLAB col-major + rot90(kernel,2): flip both Hk and Wk */
                                        int kidx = ci + Cin * co
                                                 + Cin * Cout * (Hk - 1 - hk)
                                                 + Cin * Cout * Hk * (Wk - 1 - wk);
                                        acc += (int64_t)xv * (int64_t)decoder_de_convs_4_ops_1_weight[kidx];
                                    }
                                    int shift = -cqr;
                                    int32_t val = (int32_t)((acc + ((int64_t)1 << (shift - 1))) >> shift);
                                    y_conv[co * Wout + wo] = sat32((int64_t)y_conv[co * Wout + wo] + val);
                                }
                            }
                        }
                    }

                    /* BN (De_XConv has BN only, no AffinePReLU) */
                    bn_cal(y_conv, Cout, Wout,
                           decoder_de_convs_4_ops_2_weight, decoder_de_convs_4_ops_2_bias,
                           decoder_de_convs_4_ops_2_running_mean, decoder_de_convs_4_ops_2_running_var,
                           bqr1, bqr2);

                    double sn = snr_db(gd4, y_conv, Cout * Wout);
                    if (sn > best) { best = sn; bc = cqr; bb1 = bqr1; bb2 = bqr2; }
                }
            }
        }
        printf("  Best: conv_qr=%d bn_qr1=%d bn_qr2=%d SNR=%.2f dB\n\n", bc, bb1, bb2, best);
    }

    printf("=== DONE ===\n");

    free(ge0); free(ge1); free(ge2); free(ge3); free(ge4); free(gr2);
    return 0;
}
