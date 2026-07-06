/**
 * find_all_qr_v2.c — Complete per-layer QR calibration
 * =====================================================
 * Calibrates conv_qr, bn_qr1 for ALL layers:
 *   Encoder: e0(TConv), e1(TConv), e2(TConv), e3(nonTConv), e4(nonTConv)
 *   Decoder: d0(nonGTConv), d1(nonGTConv), d2(GTConv), d3(GTConv), d4(De_XConv)
 *
 * Build: gcc -O2 -o find_all_qr_v2 find_all_qr_v2.c ulunas_fp.c ulunas_matlab_weights.c -lm
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

/* ===== Simplified calibration blocks (replicating ulunas_fp.c logic with QR params) ===== */

/* Replicate TConv_block conv logic with adjustable conv_qr */
static void tconv_conv_only(const int32_t *x_full, int H_in, int Cin, int Win,
                             const int16_t *conv_w, const int32_t *conv_b,
                             int Cout, int Wout, int Hk, int Wk,
                             int stride, int groups, int conv_qr,
                             int32_t *y_conv) {
    int pad_w = (Wk - 1) / 2;
    int Cin_pg = Cin / groups;
    int Cout_pg = Cout / groups;
    int shift = -conv_qr;

    for (int g = 0; g < groups; g++) {
        int ci_off = g * Cin_pg;
        int co_off = g * Cout_pg;
        int w_off = g * Cout_pg * Cin_pg * Hk * Wk;

        for (int co_l = 0; co_l < Cout_pg; co_l++) {
            int co = co_off + co_l;
            for (int wo = 0; wo < Wout; wo++)
                y_conv[co * Wout + wo] = conv_b[co];

            for (int ci_l = 0; ci_l < Cin_pg; ci_l++) {
                int ci = ci_off + ci_l;
                for (int wo = 0; wo < Wout; wo++) {
                    int64_t acc = 0;
                    for (int hk = 0; hk < Hk; hk++) {
                        int hi = hk;
                        if (hi < 0 || hi >= H_in) continue;
                        for (int wk = 0; wk < Wk; wk++) {
                            int wi = wo * stride + wk - pad_w;
                            if (wi < 0 || wi >= Win) continue;
                            int32_t xv = x_full[(hi * Cin + ci) * Win + wi];
                            int kidx = w_off + ((co_l * Cin_pg + ci_l) * Hk + hk) * Wk + wk;
                            int64_t prod = (int64_t)xv * (int64_t)conv_w[kidx];
                            prod = (prod + ((int64_t)1 << (shift - 1))) >> shift;
                            acc += prod;
                        }
                    }
                    y_conv[co * Wout + wo] = sat32((int64_t)y_conv[co * Wout + wo] + acc);
                }
            }
        }
    }
}

/* Replicate non_gconv2d with adjustable conv_qr */
static void nongconv_conv_only(const int32_t *x, int Cin, int Win,
                                const int16_t *conv_w, const int32_t *conv_b,
                                int Cout, int Wout, int Hk, int Wk,
                                int stride, int groups, int conv_qr,
                                int32_t *y_conv) {
    int C_per_group = Cin / groups;
    int Cout_per_group = Cout / groups;
    int pad_w = (Wk - 1) / 2;
    int shift = -conv_qr;

    for (int g = 0; g < groups; g++) {
        for (int ci_l = 0; ci_l < C_per_group; ci_l++) {
            int ci = g * C_per_group + ci_l;
            const int32_t *x_chan = x + ci * Win;
            for (int co_l = 0; co_l < Cout_per_group; co_l++) {
                int co = g * Cout_per_group + co_l;
                if (ci_l == 0) {
                    for (int wo = 0; wo < Wout; wo++)
                        y_conv[co * Wout + wo] = conv_b[co];
                }
                for (int wo = 0; wo < Wout; wo++) {
                    int64_t acc = 0;
                    for (int hk = 0; hk < Hk; hk++) {
                        for (int wk = 0; wk < Wk; wk++) {
                            int wi = wo * stride + wk - pad_w;
                            int32_t xv = (wi >= 0 && wi < Win) ? x_chan[wi] : 0;
                            int kidx = ((co_l * C_per_group + ci_l) * Hk + hk) * Wk + wk;
                            int g_off = g * Cout_per_group * C_per_group * Hk * Wk;
                            int64_t prod = (int64_t)xv * (int64_t)conv_w[g_off + kidx];
                            prod = (prod + ((int64_t)1 << (shift - 1))) >> shift;
                            acc += prod;
                        }
                    }
                    y_conv[co * Wout + wo] = sat32((int64_t)y_conv[co * Wout + wo] + acc);
                }
            }
        }
    }
}

/* Replicate non_gtconv2d with adjustable conv_qr */
static void nongtconv_conv_only(const int32_t *x, int Cin, int Win,
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
                if (ci_l == 0) {
                    for (int wo = 0; wo < Wout; wo++)
                        y_conv[co * Wout + wo] = conv_b[co];
                }
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

/* Replicate gtconv2d with adjustable conv_qr (for GTConv_block) */
static void gtconv_conv_only(const int32_t *x, int Cin, int Win,
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
                if (ci_l == 0) {
                    for (int wo = 0; wo < Wout; wo++)
                        y_conv[co * Wout + wo] = conv_b[co];
                }
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

/* BN with adjustable qr1, qr2 */
static void bn_calibrate(int32_t *x, int C, int Win,
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

/* AffinePReLU (always qr=-13,-13) */
static void ap_calibrate(int32_t *x, int C, int Win,
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
            } else {
                x_mod = x_orig;
            }
            int64_t affine_prod = (int64_t)x_orig * (int64_t)wt;
            int32_t affine_val = (int32_t)((affine_prod + ((int64_t)1 << 12)) >> 13);
            ch[w] = sat32((int64_t)affine_val + (int64_t)bias_c + (int64_t)x_mod);
        }
    }
}

int main() {
    /* Load STFT */
    float ri[257], ii[257];
    { FILE *f = fopen("dump_matlab/frame0_stft_real.bin", "rb"); fread(ri, sizeof(float), 257, f); fclose(f); }
    { FILE *f = fopen("dump_matlab/frame0_stft_imag.bin", "rb"); fread(ii, sizeof(float), 257, f); fclose(f); }

    /* log_gen → BM */
    int32_t xl[257]; log_gen_fixed(ri, ii, 257, xl);
    int32_t xb[129]; BM_fixed(xl, erb_erb_fc_weight, xb);

    /* Load all TConv goldens */
    int32_t g0[12*65], g1[24*33], g2[24*33], g3[32*33], g4[16*33];
    { FILE *f = fopen("dump_matlab/frame0_enc_e0_tconv.bin", "rb"); fread(g0, sizeof(int32_t), 12*65, f); fclose(f); }
    { FILE *f = fopen("dump_matlab/frame0_enc_e1_tconv.bin", "rb"); fread(g1, sizeof(int32_t), 24*33, f); fclose(f); }
    { FILE *f = fopen("dump_matlab/frame0_enc_e2_tconv.bin", "rb"); fread(g2, sizeof(int32_t), 24*33, f); fclose(f); }
    { FILE *f = fopen("dump_matlab/frame0_enc_e3_tconv.bin", "rb"); fread(g3, sizeof(int32_t), 32*33, f); fclose(f); }
    { FILE *f = fopen("dump_matlab/frame0_enc_e4_tconv.bin", "rb"); fread(g4, sizeof(int32_t), 16*33, f); fclose(f); }

    /* ========================================================================
     * ENCODER CALIBRATION
     * ======================================================================== */

    /* ---- e0: XConv TConv ---- */
    printf("=== Encoder e0 (XConv TConv): Cin=1 Cout=12 Win=129 Wout=65 Hk=3 Wk=3 stride=2 cache=2 groups=1 ===\n");
    {
        int Cin = 1, Cout = 12, Win = 129, Wout = 65, Hk = 3, Wk = 3, stride = 2, cache_rows = 2, groups = 1;
        int H_in = cache_rows + 1;
        int32_t x_full[H_in * Cin * Win]; memset(x_full, 0, sizeof(x_full));
        /* First frame: all cache rows are zeros, current frame at row 2 */
        for (int c = 0; c < Cin; c++)
            memcpy(x_full + (cache_rows * Cin + c) * Win, xb + c * Win, Win * sizeof(int32_t));

        double best = -999; int bc = -14, bb1 = -14;
        for (int cqr = -12; cqr >= -19; cqr--) {
            for (int bqr1 = -11; bqr1 >= -18; bqr1--) {
                int32_t y_conv[Cout * Wout]; memset(y_conv, 0, sizeof(y_conv));
                tconv_conv_only(x_full, H_in, Cin, Win,
                                encoder_en_convs_0_ops_1_weight, encoder_en_convs_0_ops_1_bias,
                                Cout, Wout, Hk, Wk, stride, groups, cqr, y_conv);
                bn_calibrate(y_conv, Cout, Wout,
                             encoder_en_convs_0_ops_2_weight, encoder_en_convs_0_ops_2_bias,
                             encoder_en_convs_0_ops_2_running_mean, encoder_en_convs_0_ops_2_running_var,
                             bqr1, -14);
                ap_calibrate(y_conv, Cout, Wout,
                             encoder_en_convs_0_ops_3_affine_weight, encoder_en_convs_0_ops_3_affine_bias,
                             encoder_en_convs_0_ops_3_slope_weight);
                double sn = snr_db(g0, y_conv, Cout * Wout);
                if (sn > best) { best = sn; bc = cqr; bb1 = bqr1; }
            }
        }
        printf("  Best: conv_qr=%d bn_qr1=%d bn_qr2=-14 SNR=%.2f dB\n\n", bc, bb1, best);
    }

    /* ---- Compute e0 output (using calibrated QR for subsequent layers) ---- */
    /* Use calibrated values from above: need to run separately. For now use module output golden. */
    int32_t *e0_full = (int32_t *)malloc(12 * 65 * sizeof(int32_t));
    { FILE *f = fopen("dump_matlab/frame0_enc_e0.bin", "rb"); fread(e0_full, sizeof(int32_t), 12*65, f); fclose(f); }

    /* ---- e1: XMB0 TConv ---- */
    printf("=== Encoder e1 (XMB0 TConv): Cin=24 Cout=24 Win=65 Wout=33 Hk=2 Wk=3 stride=2 cache=1 groups=24 ===\n");
    {
        /* Compute input to e1 TConv: PConv0 + shuffle using e0_full */
        int32_t y_pconv0[24*65];
        PConv_block(e0_full, 12, 24, 65,
                    encoder_en_convs_1_pconv1_0_weight, encoder_en_convs_1_pconv1_0_bias,
                    encoder_en_convs_1_pconv1_1_weight, encoder_en_convs_1_pconv1_1_bias,
                    encoder_en_convs_1_pconv1_1_running_mean, encoder_en_convs_1_pconv1_1_running_var,
                    encoder_en_convs_1_pconv1_2_affine_weight, encoder_en_convs_1_pconv1_2_affine_bias,
                    encoder_en_convs_1_pconv1_2_slope_weight, y_pconv0);
        int32_t y_shuf[24*65]; shuffle_interleave(y_pconv0, 12, 65, y_shuf);

        int Cin = 24, Cout = 24, Win = 65, Wout = 33, Hk = 2, Wk = 3, stride = 2, cache_rows = 1, groups = 24;
        int H_in = cache_rows + 1;
        int32_t x_full[H_in * Cin * Win]; memset(x_full, 0, sizeof(x_full));
        for (int c = 0; c < Cin; c++)
            memcpy(x_full + (cache_rows * Cin + c) * Win, y_shuf + c * Win, Win * sizeof(int32_t));

        double best = -999; int bc = -14, bb1 = -14;
        for (int cqr = -12; cqr >= -19; cqr--) {
            for (int bqr1 = -11; bqr1 >= -18; bqr1--) {
                int32_t y_conv[Cout * Wout]; memset(y_conv, 0, sizeof(y_conv));
                tconv_conv_only(x_full, H_in, Cin, Win,
                                encoder_en_convs_1_dconv_1_weight, encoder_en_convs_1_dconv_1_bias,
                                Cout, Wout, Hk, Wk, stride, groups, cqr, y_conv);
                bn_calibrate(y_conv, Cout, Wout,
                             encoder_en_convs_1_dconv_2_weight, encoder_en_convs_1_dconv_2_bias,
                             encoder_en_convs_1_dconv_2_running_mean, encoder_en_convs_1_dconv_2_running_var,
                             bqr1, -14);
                ap_calibrate(y_conv, Cout, Wout,
                             encoder_en_convs_1_dconv_3_affine_weight, encoder_en_convs_1_dconv_3_affine_bias,
                             encoder_en_convs_1_dconv_3_slope_weight);
                double sn = snr_db(g1, y_conv, Cout * Wout);
                if (sn > best) { best = sn; bc = cqr; bb1 = bqr1; }
            }
        }
        printf("  Best: conv_qr=%d bn_qr1=%d bn_qr2=-14 SNR=%.2f dB\n\n", bc, bb1, best);
    }

    /* ---- e2: XDWS0 TConv ---- */
    printf("=== Encoder e2 (XDWS0 TConv): Cin=24 Cout=24 Win=33 Wout=33 Hk=2 Wk=3 stride=1 cache=1 groups=24 ===\n");
    {
        /* Load e1 golden and compute PConv + shuffle to get e2 TConv input */
        int32_t *e1_golden = (int32_t *)malloc(24 * 33 * sizeof(int32_t));
        { FILE *f = fopen("dump_matlab/frame0_enc_e1.bin", "rb"); fread(e1_golden, sizeof(int32_t), 24*33, f); fclose(f); }

        int32_t y_pconv[24*33];
        PConv_block(e1_golden, 24, 24, 33,
                    encoder_en_convs_2_pconv_0_weight, encoder_en_convs_2_pconv_0_bias,
                    encoder_en_convs_2_pconv_1_weight, encoder_en_convs_2_pconv_1_bias,
                    encoder_en_convs_2_pconv_1_running_mean, encoder_en_convs_2_pconv_1_running_var,
                    encoder_en_convs_2_pconv_2_affine_weight, encoder_en_convs_2_pconv_2_affine_bias,
                    encoder_en_convs_2_pconv_2_slope_weight, y_pconv);
        int32_t y_shuf[24*33]; shuffle_interleave(y_pconv, 12, 33, y_shuf);

        int Cin = 24, Cout = 24, Win = 33, Wout = 33, Hk = 2, Wk = 3, stride = 1, cache_rows = 1, groups = 24;
        int H_in = cache_rows + 1;
        int32_t x_full[H_in * Cin * Win]; memset(x_full, 0, sizeof(x_full));
        for (int c = 0; c < Cin; c++)
            memcpy(x_full + (cache_rows * Cin + c) * Win, y_shuf + c * Win, Win * sizeof(int32_t));

        double best = -999; int bc = -14, bb1 = -14;
        for (int cqr = -12; cqr >= -19; cqr--) {
            for (int bqr1 = -11; bqr1 >= -18; bqr1--) {
                int32_t y_conv[Cout * Wout]; memset(y_conv, 0, sizeof(y_conv));
                tconv_conv_only(x_full, H_in, Cin, Win,
                                encoder_en_convs_2_dconv_1_weight, encoder_en_convs_2_dconv_1_bias,
                                Cout, Wout, Hk, Wk, stride, groups, cqr, y_conv);
                bn_calibrate(y_conv, Cout, Wout,
                             encoder_en_convs_2_dconv_2_weight, encoder_en_convs_2_dconv_2_bias,
                             encoder_en_convs_2_dconv_2_running_mean, encoder_en_convs_2_dconv_2_running_var,
                             bqr1, -14);
                ap_calibrate(y_conv, Cout, Wout,
                             encoder_en_convs_2_dconv_3_affine_weight, encoder_en_convs_2_dconv_3_affine_bias,
                             encoder_en_convs_2_dconv_3_slope_weight);
                double sn = snr_db(g2, y_conv, Cout * Wout);
                if (sn > best) { best = sn; bc = cqr; bb1 = bqr1; }
            }
        }
        printf("  Best: conv_qr=%d bn_qr1=%d bn_qr2=-14 SNR=%.2f dB\n\n", bc, bb1, best);
        free(e1_golden);
    }

    /* ---- e3: XMB1 nonTConv ---- */
    printf("=== Encoder e3 (XMB1 nonTConv): Cin=32 Cout=32 Win=33 Wout=33 Hk=1 Wk=5 stride=1 groups=32 ===\n");
    {
        /* Load e2 golden, compute XMB1 preprocessing up to nonTConv */
        int32_t *e2_golden = (int32_t *)malloc(24 * 33 * sizeof(int32_t));
        { FILE *f = fopen("dump_matlab/frame0_enc_e2.bin", "rb"); fread(e2_golden, sizeof(int32_t), 24*33, f); fclose(f); }

        /* PConv0: grouped 12→16 per group */
        int32_t y_pconv0[32*33];
        { int32_t y0[16*33], y1[16*33];
          pconv2d_fixed(e2_golden, 12, 33, encoder_en_convs_3_pconv1_0_weight, encoder_en_convs_3_pconv1_0_bias, 16, -13, y0);
          pconv2d_fixed(e2_golden+12*33, 12, 33, encoder_en_convs_3_pconv1_0_weight+16*12, encoder_en_convs_3_pconv1_0_bias+16, 16, -13, y1);
          for (int c = 0; c < 16; c++) {
              memcpy(y_pconv0 + c * 33, y0 + c * 33, 33 * sizeof(int32_t));
              memcpy(y_pconv0 + (16 + c) * 33, y1 + c * 33, 33 * sizeof(int32_t));
          }
          bn_fixed(y_pconv0, 32, 33, encoder_en_convs_3_pconv1_1_weight, encoder_en_convs_3_pconv1_1_bias,
                   encoder_en_convs_3_pconv1_1_running_mean, encoder_en_convs_3_pconv1_1_running_var, -11, -14);
          affine_prelu_fixed(y_pconv0, 32, 33, encoder_en_convs_3_pconv1_2_affine_weight,
                             encoder_en_convs_3_pconv1_2_affine_bias, encoder_en_convs_3_pconv1_2_slope_weight, -13, -13);
        }
        int32_t y_shuf[32*33]; shuffle_interleave(y_pconv0, 16, 33, y_shuf);

        int Cin = 32, Cout = 32, Win = 33, Wout = 33, Hk = 1, Wk = 5, stride = 1, groups = 32;

        double best = -999; int bc = -13, bb1 = -14;
        for (int cqr = -12; cqr >= -19; cqr--) {
            for (int bqr1 = -11; bqr1 >= -18; bqr1--) {
                int32_t y_conv[Cout * Wout]; memset(y_conv, 0, sizeof(y_conv));
                nongconv_conv_only(y_shuf, Cin, Win,
                                   encoder_en_convs_3_dconv_1_weight, encoder_en_convs_3_dconv_1_bias,
                                   Cout, Wout, Hk, Wk, stride, groups, cqr, y_conv);
                bn_calibrate(y_conv, Cout, Wout,
                             encoder_en_convs_3_dconv_2_weight, encoder_en_convs_3_dconv_2_bias,
                             encoder_en_convs_3_dconv_2_running_mean, encoder_en_convs_3_dconv_2_running_var,
                             bqr1, -14);
                ap_calibrate(y_conv, Cout, Wout,
                             encoder_en_convs_3_dconv_3_affine_weight, encoder_en_convs_3_dconv_3_affine_bias,
                             encoder_en_convs_3_dconv_3_slope_weight);
                double sn = snr_db(g3, y_conv, Cout * Wout);
                if (sn > best) { best = sn; bc = cqr; bb1 = bqr1; }
            }
        }
        printf("  Best: conv_qr=%d bn_qr1=%d bn_qr2=-14 SNR=%.2f dB\n\n", bc, bb1, best);
        free(e2_golden);
    }

    /* ---- e4: XDWS1 nonTConv ---- */
    printf("=== Encoder e4 (XDWS1 nonTConv): Cin=16 Cout=16 Win=33 Wout=33 Hk=1 Wk=5 stride=1 groups=16 ===\n");
    {
        /* Load e3 golden, compute XDWS1 preprocessing up to nonTConv */
        int32_t *e3_golden = (int32_t *)malloc(32 * 33 * sizeof(int32_t));
        { FILE *f = fopen("dump_matlab/frame0_enc_e3.bin", "rb"); fread(e3_golden, sizeof(int32_t), 32*33, f); fclose(f); }

        /* PConv: group0: Cin=16→Cout=8, group1: Cin=16→Cout=8 */
        int32_t y_pconv[16*33];
        { int32_t y0[8*33], y1[8*33];
          pconv2d_fixed(e3_golden, 16, 33, encoder_en_convs_4_pconv_0_weight, encoder_en_convs_4_pconv_0_bias, 8, -14, y0);
          pconv2d_fixed(e3_golden+16*33, 16, 33, encoder_en_convs_4_pconv_0_weight+8*16, encoder_en_convs_4_pconv_0_bias+8, 8, -14, y1);
          for (int c = 0; c < 8; c++) {
              memcpy(y_pconv + c * 33, y0 + c * 33, 33 * sizeof(int32_t));
              memcpy(y_pconv + (8 + c) * 33, y1 + c * 33, 33 * sizeof(int32_t));
          }
          bn_fixed(y_pconv, 16, 33, encoder_en_convs_4_pconv_1_weight, encoder_en_convs_4_pconv_1_bias,
                   encoder_en_convs_4_pconv_1_running_mean, encoder_en_convs_4_pconv_1_running_var, -11, -14);
          affine_prelu_fixed(y_pconv, 16, 33, encoder_en_convs_4_pconv_2_affine_weight,
                             encoder_en_convs_4_pconv_2_affine_bias, encoder_en_convs_4_pconv_2_slope_weight, -13, -13);
        }
        int32_t y_shuf[16*33]; shuffle_interleave(y_pconv, 8, 33, y_shuf);

        int Cin = 16, Cout = 16, Win = 33, Wout = 33, Hk = 1, Wk = 5, stride = 1, groups = 16;

        double best = -999; int bc = -13, bb1 = -14;
        for (int cqr = -12; cqr >= -19; cqr--) {
            for (int bqr1 = -11; bqr1 >= -18; bqr1--) {
                int32_t y_conv[Cout * Wout]; memset(y_conv, 0, sizeof(y_conv));
                nongconv_conv_only(y_shuf, Cin, Win,
                                   encoder_en_convs_4_dconv_1_weight, encoder_en_convs_4_dconv_1_bias,
                                   Cout, Wout, Hk, Wk, stride, groups, cqr, y_conv);
                bn_calibrate(y_conv, Cout, Wout,
                             encoder_en_convs_4_dconv_2_weight, encoder_en_convs_4_dconv_2_bias,
                             encoder_en_convs_4_dconv_2_running_mean, encoder_en_convs_4_dconv_2_running_var,
                             bqr1, -14);
                ap_calibrate(y_conv, Cout, Wout,
                             encoder_en_convs_4_dconv_3_affine_weight, encoder_en_convs_4_dconv_3_affine_bias,
                             encoder_en_convs_4_dconv_3_slope_weight);
                double sn = snr_db(g4, y_conv, Cout * Wout);
                if (sn > best) { best = sn; bc = cqr; bb1 = bqr1; }
            }
        }
        printf("  Best: conv_qr=%d bn_qr1=%d bn_qr2=-14 SNR=%.2f dB\n\n", bc, bb1, best);
        free(e3_golden);
    }

    /* ========================================================================
     * DECODER CALIBRATION (no golden files — use full module golden as proxy)
     * ========================================================================
     * For decoder, we'll work with what we have:
     * - We can compute the inputs using the encoder goldens + RNN goldens
     * - But we only have frame*_dec.bin as the final decoder output
     * - This section calibrates decoder conv layers INDIRECTLY via full-pipeline SNR
     *
     * For now, we output recommended QR based on encoder patterns:
     * - nonGTConv: similar to nonTConv (encoder e3/e4)
     * - GTConv: similar to TConv but transposed
     */

    printf("=== DECODER LAYER CALIBRATION ===\n");
    printf("(No TConv golden files available for decoder — using encoder patterns as guidance)\n\n");

    /* ---- d0: De_XDWS0 nonGTConv ---- */
    printf("--- Decoder d0 (De_XDWS0 nonGTConv): Cin=32 Cout=32 Win=33 Wout=33 stride=1 Hk=1 Wk=5 groups=32 ---\n");
    printf("  Recommendation: conv_qr=-13 bn_qr1=-14 bn_qr2=-14 (same as encoder e3 nonTConv)\n\n");

    /* ---- d1: De_XMB0 nonGTConv ---- */
    printf("--- Decoder d1 (De_XMB0 nonGTConv): Cin=24 Cout=24 Win=33 Wout=33 stride=1 Hk=1 Wk=5 groups=24 ---\n");
    printf("  Recommendation: conv_qr=-13 bn_qr1=-14 bn_qr2=-14\n\n");

    /* ---- d2: De_XDWS1 GTConv ---- */
    printf("--- Decoder d2 (De_XDWS1 GTConv): Cin=24 Cout=24 Win=33 Wout=33 stride=1 Hk=1 Wk=3 groups=24 ---\n");
    printf("  Recommendation: conv_qr=-14 bn_qr1=-11 bn_qr2=-14 (per current code, needs verification)\n\n");

    /* ---- d3: De_XMB1 GTConv ---- */
    printf("--- Decoder d3 (De_XMB1 GTConv): Cin=12 Cout=12 Win=33 Wout=65 stride=2 Hk=1 Wk=3 groups=12 ---\n");
    printf("  Recommendation: conv_qr=-14 bn_qr1=-11 bn_qr2=-11 (per current code, needs verification)\n\n");

    /* ---- d4: De_XConv ---- */
    printf("--- Decoder d4 (De_XConv): Cin=12*3 Cout=1 Win=65 Wout=129 stride=2 Hk=3 Wk=3 ---\n");
    printf("  Recommendation: conv_qr=-14 bn_qr1=-11 bn_qr2=-11 (per current code, needs verification)\n\n");

    printf("=== DONE ===\n");
    printf("Summary: Encoder e0-e4 calibrated. Decoder needs MATLAB golden export for full calibration.\n");
    printf("Run: matlab -nodisplay -r \"export_all_layers\" to export decoder TConv goldens.\n");

    free(e0_full);
    return 0;
}
