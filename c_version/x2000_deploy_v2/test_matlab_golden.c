/**
 * test_matlab_golden.c — Layer-by-Layer Golden Verification
 * ==========================================================
 * Compares C inference output against MATLAB golden binary dumps.
 *
 * Usage: ./test_matlab_golden <golden_dir>
 * Example: ./test_matlab_golden ../x2000_deploy_v1/dump_matlab/
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "qr_config.h"
#include "layer_dims.h"
#include "ulunas_matlab_weights.h"

/* ================================================================
 * Helpers
 * ================================================================ */

static int32_t *load_int32(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    int32_t *buf = malloc(n * sizeof(int32_t));
    if (!buf) { fclose(f); return NULL; }
    size_t r = fread(buf, sizeof(int32_t), n, f);
    fclose(f);
    if (r != (size_t)n) { free(buf); return NULL; }
    return buf;
}

static uint16_t *load_uint16(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint16_t *buf = malloc(n * sizeof(uint16_t));
    if (!buf) { fclose(f); return NULL; }
    size_t r = fread(buf, sizeof(uint16_t), n, f);
    fclose(f);
    if (r != (size_t)n) { free(buf); return NULL; }
    return buf;
}

static int16_t *load_int16(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    int16_t *buf = malloc(n * sizeof(int16_t));
    if (!buf) { fclose(f); return NULL; }
    size_t r = fread(buf, sizeof(int16_t), n, f);
    fclose(f);
    if (r != (size_t)n) { free(buf); return NULL; }
    return buf;
}

static float *load_float(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    float *buf = malloc(n * sizeof(float));
    if (!buf) { fclose(f); return NULL; }
    size_t r = fread(buf, sizeof(float), n, f);
    fclose(f);
    if (r != (size_t)n) { free(buf); return NULL; }
    return buf;
}

/* SNR in dB between int32_t arrays */
static double snr_db_i32(const int32_t *golden, const int32_t *test, int n) {
    double s = 0.0, e = 0.0;
    for (int i = 0; i < n; i++) {
        double gv = (double)golden[i];
        double tv = (double)test[i];
        double d = gv - tv;
        s += gv * gv;
        e += d * d;
    }
    if (e < 1e-30) return 999.0;
    if (s < 1e-30) return 999.0;
    return 10.0 * log10(s / e);
}

/* SNR for uint16_t arrays */
static double snr_db_u16(const uint16_t *golden, const uint16_t *test, int n) {
    double s = 0.0, e = 0.0;
    for (int i = 0; i < n; i++) {
        double gv = (double)golden[i];
        double tv = (double)test[i];
        double d = gv - tv;
        s += gv * gv;
        e += d * d;
    }
    if (e < 1e-30) return 999.0;
    if (s < 1e-30) return 999.0;
    return 10.0 * log10(s / e);
}

/* SNR for int16_t arrays */
static double snr_db_i16(const int16_t *golden, const int16_t *test, int n) {
    double s = 0.0, e = 0.0;
    for (int i = 0; i < n; i++) {
        double gv = (double)golden[i];
        double tv = (double)test[i];
        double d = gv - tv;
        s += gv * gv;
        e += d * d;
    }
    if (e < 1e-30) return 999.0;
    if (s < 1e-30) return 999.0;
    return 10.0 * log10(s / e);
}

static double max_abs_err_i32(const int32_t *golden, const int32_t *test, int n) {
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)golden[i] - (double)test[i]);
        if (d > m) m = d;
    }
    return m;
}

static const char *status(double snr) {
    if (snr > 120.0) return "PERFECT";
    if (snr > 80.0)  return "PASS";
    if (snr > 40.0)  return "WARN";
    return "FAIL";
}

/**
 * snr_db_2d: compare 2D tensors with MATLAB col-major to C row-major conversion.
 * MATLAB stores (C, W) in column-major: golden[c + C*w]
 * C stores (C, W) in row-major: test[c*W + w]
 */
static double snr_db_2d_i32(const int32_t *golden, const int32_t *test, int C, int W) {
    int N = C * W;
    int32_t *gt = malloc(N * sizeof(int32_t));
    for (int c = 0; c < C; c++)
        for (int w = 0; w < W; w++)
            gt[c * W + w] = golden[c + C * w];  /* col-major → row-major */
    double s = snr_db_i32(gt, test, N);
    free(gt);
    return s;
}

/* ================================================================
 * Main
 * ================================================================ */
int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : "../x2000_deploy_v1/dump_matlab/";
    char path[512];
    int tested = 0, passed = 0;

    printf("=== UL-UNAS C vs MATLAB Golden Layer SNR ===\n\n");

    /* State persists ACROSS frames — do NOT re-init per frame */
    ulunas_state_t st;
    ulunas_state_init(&st);

    /* Process frames 0-4 */
    for (int frame = 0; frame < 5; frame++) {
        /* Load STFT input from golden dump */
        snprintf(path, sizeof(path), "%s/frame%d_stft_real.bin", dir, frame);
        float *real_in = load_float(path, 257);
        snprintf(path, sizeof(path), "%s/frame%d_stft_imag.bin", dir, frame);
        float *imag_in = load_float(path, 257);

        if (!real_in || !imag_in) {
            printf("Frame %d: SKIP (no STFT input)\n", frame);
            free(real_in); free(imag_in);
            continue;
        }

        printf("--- Frame %d ---\n", frame);

        /* Step 1: log_gen */
        int32_t x_log[1 * 257];
        {
            /* Quantize to Q20 */
            int32_t real_q20[257], imag_q20[257];
            for (int i = 0; i < 257; i++) {
                real_q20[i] = (int32_t)round(real_in[i] * 1048576.0f);
                imag_q20[i] = (int32_t)round(imag_in[i] * 1048576.0f);
            }
            log_gen_fixed(real_q20, imag_q20, 257, x_log);
        }

        /* Step 2: BM */
        int32_t x_bm[1 * 129];
        bm_fixed(x_log, erb_erb_fc_weight, 257, 129, x_bm);

        snprintf(path, sizeof(path), "%s/frame%d_bm.bin", dir, frame);
        int32_t *gbm = load_int32(path, 129);
        if (gbm) {
            double snr = snr_db_i32(gbm, x_bm, 129);
            printf("  BM        : SNR=%7.2f dB  MAX=%6.1f  [%s]\n",
                   snr, max_abs_err_i32(gbm, x_bm, 129), status(snr));
            tested++; if (snr > 80.0) passed++;
            free(gbm);
        }

        /* Step 3a: E0 sub-block golden (before encoder to use correct cache state) */
        {
            int W_in = 129;
            int32_t x_c[3 * 129];
            memcpy(x_c, st.conv_cache_e0, 2 * W_in * sizeof(int32_t));
            memcpy(x_c + 2 * W_in, x_bm, 1 * W_in * sizeof(int32_t));

            /* Save cache for restoration */
            int32_t saved_conv_e0[2 * 129];
            int16_t saved_tfa_e0[24];
            memcpy(saved_conv_e0, st.conv_cache_e0, sizeof(saved_conv_e0));
            memcpy(saved_tfa_e0, st.tfa_cache_e0, sizeof(saved_tfa_e0));

            /* E0.1: TConv */
            int32_t y_tconv[12 * 65];
            conv2d_func(x_c, 1, 12, 1, 65, 3, 3, 1, 2,
                        encoder_en_convs_0_ops_1_weight, encoder_en_convs_0_ops_1_bias,
                        E0_TCONV_CONV_QR, y_tconv);

            /* E0.2: BN */
            int32_t y_bn[12 * 65];
            bn_func(y_tconv, encoder_en_convs_0_ops_2_weight, encoder_en_convs_0_ops_2_bias,
                    encoder_en_convs_0_ops_2_running_mean, encoder_en_convs_0_ops_2_running_var,
                    E0_TCONV_BN_QR1, E0_TCONV_BN_QR2, 12, 12 * 65, y_bn);

            /* E0.3: AffinePReLU */
            int32_t y_ap[12 * 65];
            affineprelu_func(y_bn, encoder_en_convs_0_ops_3_affine_weight,
                             encoder_en_convs_0_ops_3_affine_bias,
                             encoder_en_convs_0_ops_3_slope_weight,
                             E0_TCONV_AFFINE_QR1, E0_TCONV_AFFINE_QR2, 12, 65, y_ap);

            /* Check TConv+BN+AffinePReLU golden */
            snprintf(path, sizeof(path), "%s/frame%d_enc_e0_ctfa_in.bin", dir, frame);
            int32_t *g_ctfa_in = load_int32(path, 12 * 65);
            if (g_ctfa_in) {
                double snr = snr_db_2d_i32(g_ctfa_in, y_ap, 12, 65);
                printf("  E0.tconv : SNR=%7.2f dB  MAX=%6.1f  [%s]\n",
                       snr, max_abs_err_i32(g_ctfa_in, y_ap, 12 * 65), status(snr));
                free(g_ctfa_in);
            }

            /* E0.4: cTFA TA */
            uint16_t y_ta[12];
            ctfa_ta_module(y_ap, 12, 65, E0_CTFA_TA_GRU_NHID,
                           st.tfa_cache_e0,
                           encoder_en_convs_0_ops_4_ta_gru_weight_ih_l0,
                           encoder_en_convs_0_ops_4_ta_gru_bias_ih_l0,
                           encoder_en_convs_0_ops_4_ta_gru_weight_hh_l0,
                           encoder_en_convs_0_ops_4_ta_gru_bias_hh_l0,
                           encoder_en_convs_0_ops_4_ta_fc_weight,
                           encoder_en_convs_0_ops_4_ta_fc_bias,
                           E0_CTFA_TA_GRU_QR1, E0_CTFA_TA_GRU_QR2, E0_CTFA_TA_FC_QR, y_ta);

            snprintf(path, sizeof(path), "%s/frame%d_enc_e0_ctfa_ta.bin", dir, frame);
            uint16_t *g_ta = load_uint16(path, 12);
            if (g_ta) {
                double snr = snr_db_u16(g_ta, y_ta, 12);
                printf("  E0.ta    : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                free(g_ta);
            }

            /* --- E0 TA Sub-Step Diagnostics (frame 0 only) --- */
            if (frame == 0) {
                int TA_C = 12, TA_W = 65, TA_nH = E0_CTFA_TA_GRU_NHID;

                /* TA Step 1: Square + mean over frequency → [C] Q20 */
                int32_t ta_agg[12];
                for (int c = 0; c < TA_C; c++) {
                    int64_t sum_sq = 0;
                    for (int w = 0; w < TA_W; w++) {
                        int32_t val = y_ap[c * TA_W + w];
                        sum_sq += (int64_t)val * val;
                    }
                    int64_t denom = (int64_t)TA_W * 1048576LL;
                    int64_t half = denom / 2;
                    if (sum_sq >= 0) ta_agg[c] = (int32_t)((sum_sq + half) / denom);
                    else             ta_agg[c] = (int32_t)((sum_sq - half) / denom);
                    if (ta_agg[c] < 0) ta_agg[c] = 0;
                }

                /* TA Step 2: GRU (fresh cache = zeros for frame 0) */
                int16_t ta_gru_out[24];
                int16_t h_cache_ta_diag[24] = {0};
                gru_module(ta_agg, TA_nH, TA_C, h_cache_ta_diag,
                           encoder_en_convs_0_ops_4_ta_gru_weight_ih_l0,
                           encoder_en_convs_0_ops_4_ta_gru_bias_ih_l0,
                           encoder_en_convs_0_ops_4_ta_gru_weight_hh_l0,
                           encoder_en_convs_0_ops_4_ta_gru_bias_hh_l0,
                           E0_CTFA_TA_GRU_QR1, E0_CTFA_TA_GRU_QR2, ta_gru_out);

                snprintf(path, sizeof(path), "%s/frame0_e0_ta_gru.bin", dir);
                int16_t *g_ta_gru = load_int16(path, 24);
                if (g_ta_gru) {
                    double snr = snr_db_i16(g_ta_gru, ta_gru_out, 24);
                    printf("  E0.ta.gru : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                    printf("    [0..7] C:"); for(int i=0;i<8;i++) printf(" %6d", ta_gru_out[i]); printf("\n");
                    printf("    [0..7] G:"); for(int i=0;i<8;i++) printf(" %6d", g_ta_gru[i]); printf("\n");
                    free(g_ta_gru);
                }

                /* TA Step 3: FC + sigmoid */
                int ta_shift_fc = -E0_CTFA_TA_FC_QR;
                int64_t ta_r_fc = ((int64_t)1 << (ta_shift_fc - 1));
                uint16_t ta_sig_diag[12];
                for (int c = 0; c < TA_C; c++) {
                    int64_t acc = 0;
                    for (int j = 0; j < TA_nH; j++) {
                        int64_t prod = (int64_t)ta_gru_out[j] *
                            encoder_en_convs_0_ops_4_ta_fc_weight[j + TA_nH * c];
                        if (prod >= 0) acc += (prod + ta_r_fc) >> ta_shift_fc;
                        else           acc += (prod - ta_r_fc) >> ta_shift_fc;
                    }
                    int32_t fc_out = sat_i32(acc + encoder_en_convs_0_ops_4_ta_fc_bias[c]);
                    ta_sig_diag[c] = sigmoid_q20_to_q15(fc_out);
                }

                snprintf(path, sizeof(path), "%s/frame0_e0_ta_out.bin", dir);
                uint16_t *g_ta_out = load_uint16(path, 12);
                if (g_ta_out) {
                    double snr = snr_db_u16(g_ta_out, ta_sig_diag, 12);
                    printf("  E0.ta.sig : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                    free(g_ta_out);
                }
            }

            /* E0.5: cTFA FA */
            uint16_t y_fa[65];
            ctfa_fa_module(y_ap, 12, 65, E0_CTFA_FA_GRU_NHID,
                           E0_CTFA_FA_GROUP, E0_CTFA_FA_SEG, E0_CTFA_FA_PAD,
                           encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0,
                           encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0,
                           encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0,
                           encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0,
                           encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0_reverse,
                           encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0_reverse,
                           encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0_reverse,
                           encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0_reverse,
                           encoder_en_convs_0_ops_4_fa_fc_weight,
                           encoder_en_convs_0_ops_4_fa_fc_bias,
                           E0_CTFA_FA_GRU_QR1, E0_CTFA_FA_GRU_QR2, E0_CTFA_FA_FC_QR, y_fa);

            snprintf(path, sizeof(path), "%s/frame%d_enc_e0_ctfa_fa.bin", dir, frame);
            uint16_t *g_fa = load_uint16(path, 65);
            if (g_fa) {
                double snr = snr_db_u16(g_fa, y_fa, 65);
                printf("  E0.fa    : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                free(g_fa);
            }

            /* --- E0 FA Sub-Step Diagnostics (frame 0 only) --- */
            if (frame == 0) {
                int FA_C = 12, FA_W = 65, FA_nH = E0_CTFA_FA_GRU_NHID;
                int FA_G = E0_CTFA_FA_GROUP, FA_S = E0_CTFA_FA_SEG, FA_P = E0_CTFA_FA_PAD;

                /* FA Step 1: Square + mean over channels → [W] Q20 */
                int32_t fa_agg[65];
                for (int w = 0; w < FA_W; w++) {
                    int64_t sum_sq = 0;
                    for (int c = 0; c < FA_C; c++) {
                        int32_t val = y_ap[c * FA_W + w];
                        sum_sq += (int64_t)val * val;
                    }
                    int64_t denom = (int64_t)FA_C * 1048576LL;
                    int64_t half = denom / 2;
                    if (sum_sq >= 0) fa_agg[w] = (int32_t)((sum_sq + half) / denom);
                    else             fa_agg[w] = (int32_t)((sum_sq - half) / denom);
                    if (fa_agg[w] < 0) fa_agg[w] = 0;
                }

                snprintf(path, sizeof(path), "%s/frame0_e0_fa_agg.bin", dir);
                int32_t *g_fa_agg = load_int32(path, FA_W);
                if (g_fa_agg) {
                    double snr = snr_db_i32(g_fa_agg, fa_agg, FA_W);
                    printf("  E0.fa.agg : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                    free(g_fa_agg);
                }

                /* FA Step 2: Pad + Reshape to [seg][group] */
                int32_t fa_pad[68];
                memcpy(fa_pad, fa_agg, FA_W * sizeof(int32_t));
                memset(fa_pad + FA_W, 0, FA_P * sizeof(int32_t));
                int32_t fa_reshaped[68];
                for (int s = 0; s < FA_S; s++)
                    for (int g = 0; g < FA_G; g++)
                        fa_reshaped[s * FA_G + g] = fa_pad[g + FA_G * s];

                /* FA Step 3: BiGRU forward (per timestep, zero init) */
                int16_t fa_gru_fwd[17 * 4];
                int16_t h_fwd_diag[4] = {0};
                for (int t = 0; t < FA_S; t++) {
                    gru_module(&fa_reshaped[t * FA_G], FA_nH, FA_G, h_fwd_diag,
                               encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0,
                               encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0,
                               encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0,
                               encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0,
                               E0_CTFA_FA_GRU_QR1, E0_CTFA_FA_GRU_QR2,
                               &fa_gru_fwd[t * FA_nH]);
                }

                snprintf(path, sizeof(path), "%s/frame0_e0_fa_gru0.bin", dir);
                int16_t *g_fa_gru0 = load_int16(path, FA_S * FA_nH);
                if (g_fa_gru0) {
                    /* Golden is MATLAB column-major (hidden-first): g[t + T*h]
                     * C is row-major (time-first): c[t*nH + h]
                     * Reorder golden to match C */
                    int16_t g_reordered[68];
                    for (int t = 0; t < FA_S; t++)
                        for (int h = 0; h < FA_nH; h++)
                            g_reordered[t * FA_nH + h] = g_fa_gru0[t + FA_S * h];
                    double snr = snr_db_i16(g_reordered, fa_gru_fwd, FA_S * FA_nH);
                    printf("  E0.fa.gru0 : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                    printf("    [0..7] C:"); for(int i=0;i<8;i++) printf(" %6d", fa_gru_fwd[i]); printf("\n");
                    printf("    [0..7] G:"); for(int i=0;i<8;i++) printf(" %6d", g_reordered[i]); printf("\n");
                    free(g_fa_gru0);
                }

                /* FA Step 4: BiGRU reverse (process t=16..0, zero init) */
                int16_t fa_gru_rev_raw[17 * 4]; /* rev processing order */
                int16_t h_rev_diag[4] = {0};
                for (int t = 0; t < FA_S; t++) {
                    int t_rev = FA_S - 1 - t;
                    gru_module(&fa_reshaped[t_rev * FA_G], FA_nH, FA_G, h_rev_diag,
                               encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0_reverse,
                               encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0_reverse,
                               encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0_reverse,
                               encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0_reverse,
                               E0_CTFA_FA_GRU_QR1, E0_CTFA_FA_GRU_QR2,
                               &fa_gru_rev_raw[t * FA_nH]);
                }

                snprintf(path, sizeof(path), "%s/frame0_e0_fa_gru1.bin", dir);
                int16_t *g_fa_gru1 = load_int16(path, FA_S * FA_nH);
                if (g_fa_gru1) {
                    /* Golden is MATLAB column-major: g[t_fwd + T*h]
                     * C raw rev order: fa_gru_rev_raw[step*nH+h] where step=0 → t_fwd=T-1
                     * Build rev in forward time order, then compare */
                    int16_t g_reordered[68];
                    int16_t rev_fwd[68];
                    for (int t = 0; t < FA_S; t++)
                        for (int h = 0; h < FA_nH; h++) {
                            g_reordered[t * FA_nH + h] = g_fa_gru1[t + FA_S * h];
                            rev_fwd[t * FA_nH + h] = fa_gru_rev_raw[(FA_S - 1 - t) * FA_nH + h];
                        }
                    double snr = snr_db_i16(g_reordered, rev_fwd, FA_S * FA_nH);
                    printf("  E0.fa.gru1 : SNR=%7.2f dB  [%s] (fwd time order)\n", snr, status(snr));
                    free(g_fa_gru1);
                }

                /* Build concatenated BiGRU output: cat(fwd, flipped_rev) */
                int16_t fa_gru_cat[17 * 8];
                for (int t = 0; t < FA_S; t++) {
                    memcpy(&fa_gru_cat[t * (2 * FA_nH)], &fa_gru_fwd[t * FA_nH],
                           FA_nH * sizeof(int16_t));
                    memcpy(&fa_gru_cat[t * (2 * FA_nH) + FA_nH],
                           &fa_gru_rev_raw[(FA_S - 1 - t) * FA_nH],
                           FA_nH * sizeof(int16_t));
                }

                /* FA Step 5: FC [seg][2*nH] → [seg][group] */
                int fa_shift_fc = -E0_CTFA_FA_FC_QR;
                int64_t fa_r_fc = ((int64_t)1 << (fa_shift_fc - 1));
                int FA_in_fc = 2 * FA_nH;
                int32_t fa_fc_2d[17 * 4];
                for (int s = 0; s < FA_S; s++) {
                    for (int g = 0; g < FA_G; g++) {
                        int64_t acc = 0;
                        for (int i = 0; i < FA_in_fc; i++) {
                            int64_t prod = (int64_t)fa_gru_cat[s * FA_in_fc + i] *
                                encoder_en_convs_0_ops_4_fa_fc_weight[i + FA_in_fc * g];
                            if (prod >= 0) acc += (prod + fa_r_fc) >> fa_shift_fc;
                            else           acc += (prod - fa_r_fc) >> fa_shift_fc;
                        }
                        fa_fc_2d[s * FA_G + g] = sat_i32(acc +
                            encoder_en_convs_0_ops_4_fa_fc_bias[g]);
                    }
                }
                /* Reshape: x_flat[g + G*s] = x_fc_2d[s*G + g] */
                int32_t fa_fc_flat[68];
                for (int g = 0; g < FA_G; g++)
                    for (int s = 0; s < FA_S; s++)
                        fa_fc_flat[g + FA_G * s] = fa_fc_2d[s * FA_G + g];

                snprintf(path, sizeof(path), "%s/frame0_e0_fa_fc.bin", dir);
                int32_t *g_fa_fc = load_int32(path, 68);
                if (g_fa_fc) {
                    double snr = snr_db_i32(g_fa_fc, fa_fc_flat, 68);
                    printf("  E0.fa.fc  : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                    free(g_fa_fc);
                }

                /* FA Step 6: Sigmoid [0:W-1] */
                uint16_t fa_sig_diag[65];
                for (int w = 0; w < FA_W; w++)
                    fa_sig_diag[w] = sigmoid_q20_to_q15(fa_fc_flat[w]);

                snprintf(path, sizeof(path), "%s/frame0_e0_fa_out.bin", dir);
                uint16_t *g_fa_out = load_uint16(path, FA_W);
                if (g_fa_out) {
                    double snr = snr_db_u16(g_fa_out, fa_sig_diag, FA_W);
                    printf("  E0.fa.sig : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                    free(g_fa_out);
                }
            }

            /* E0.6: cTFA fusion */
            int32_t y_ctfa[12 * 65];
            int64_t r_f = 16384;
            for (int c = 0; c < 12; c++) {
                for (int w = 0; w < 65; w++) {
                    int64_t p1 = (int64_t)y_ap[c * 65 + w] * y_ta[c];
                    int32_t yt;
                    if (p1 >= 0) yt = (int32_t)((p1 + r_f) >> 15);
                    else         yt = (int32_t)((p1 - r_f) >> 15);
                    int64_t p2 = (int64_t)yt * y_fa[w];
                    if (p2 >= 0) y_ctfa[c * 65 + w] = (int32_t)((p2 + r_f) >> 15);
                    else         y_ctfa[c * 65 + w] = (int32_t)((p2 - r_f) >> 15);
                }
            }

            snprintf(path, sizeof(path), "%s/frame%d_enc_e0_ctfa_out.bin", dir, frame);
            int32_t *g_ctfa_out = load_int32(path, 12 * 65);
            if (g_ctfa_out) {
                double snr = snr_db_2d_i32(g_ctfa_out, y_ctfa, 12, 65);
                printf("  E0.ctfa  : SNR=%7.2f dB  MAX=%6.1f  [%s]\n",
                       snr, max_abs_err_i32(g_ctfa_out, y_ctfa, 12 * 65), status(snr));
                free(g_ctfa_out);
            }

            /* Restore cache so encoder_module starts from correct state */
            memcpy(st.conv_cache_e0, saved_conv_e0, sizeof(saved_conv_e0));
            memcpy(st.tfa_cache_e0, saved_tfa_e0, sizeof(saved_tfa_e0));
        }

        /* Step 3: Encoder */
        int32_t e0[12 * 65], e1[24 * 33], e2[24 * 33], e3[32 * 33], e4[16 * 33];
        encoder_module(x_bm, &st, e0, e1, e2, e3, e4);

        /* Check each encoder layer */
        const char *enames[] = {"enc_e0", "enc_e1", "enc_e2", "enc_e3", "enc_e4"};
        int32_t *eouts[] = {e0, e1, e2, e3, e4};
        int eC[] = {12, 24, 24, 32, 16};
        int eW[] = {65, 33, 33, 33, 33};

        for (int s = 0; s < 5; s++) {
            snprintf(path, sizeof(path), "%s/frame%d_%s.bin", dir, frame, enames[s]);
            int32_t *g = load_int32(path, eC[s] * eW[s]);
            if (g) {
                double snr = snr_db_2d_i32(g, eouts[s], eC[s], eW[s]);
                printf("  %-7s   : SNR=%7.2f dB  MAX=%6.1f  [%s]\n",
                       enames[s], snr, max_abs_err_i32(g, eouts[s], eC[s] * eW[s]), status(snr));
                tested++; if (snr > 80.0) passed++;
                free(g);
            } else {
                printf("  %-7s   : SKIP (no golden)\n", enames[s]);
            }
        }

        /* --- Layer Isolation Test (frame 0 only, fresh state = zero caches) --- */
        if (frame == 0) {
            printf("\n  --- Layer Isolation (golden input -> C layer -> golden output) ---\n");
            const char *inames[] = {"enc_e0", "enc_e1", "enc_e2", "enc_e3", "enc_e4"};
            int iC[] = {12, 24, 24, 32, 16};
            int iW[] = {65, 33, 33, 33, 33};
            /* Test each encoder layer with golden input */
            for (int layer = 0; layer < 5; layer++) {
                /* Determine golden input file */
                const char *input_name = (layer == 0) ? "bm" : inames[layer - 1];
                int input_sz = (layer == 0) ? 129 : iC[layer - 1] * iW[layer - 1];
                snprintf(path, sizeof(path), "%s/frame0_%s.bin", dir, input_name);
                int32_t *golden_in = load_int32(path, input_sz);
                if (!golden_in) { printf("  E%d iso    : SKIP (no golden input)\n", layer); continue; }

                /* Load golden output for comparison */
                snprintf(path, sizeof(path), "%s/frame0_%s.bin", dir, inames[layer]);
                int32_t *golden_out = load_int32(path, iC[layer] * iW[layer]);
                if (!golden_out) { printf("  E%d iso    : SKIP (no golden output)\n", layer); free(golden_in); continue; }

                /* Convert golden input from MATLAB column-major to C row-major.
                 * Golden is [C, W] col-major: golden[c + C*w]
                 * C expects [C, W] row-major: x[c*W + w] */
                int32_t *input_rm = malloc(input_sz * sizeof(int32_t));
                if (layer == 0) {
                    memcpy(input_rm, golden_in, input_sz * sizeof(int32_t)); /* 1D, no change */
                } else {
                    for (int c = 0; c < iC[layer - 1]; c++)
                        for (int w = 0; w < iW[layer - 1]; w++)
                            input_rm[c * iW[layer - 1] + w] = golden_in[c + iC[layer - 1] * w];
                }

                /* Fresh state (zero caches = frame 0 cold start) */
                ulunas_state_t iso_st;
                ulunas_state_init(&iso_st);

                int32_t *c_out = malloc(iC[layer] * iW[layer] * sizeof(int32_t));
                if (!c_out) { free(golden_in); free(golden_out); free(input_rm); continue; }

                switch (layer) {
                case 0: encoder_layer0_xconv(input_rm, &iso_st, c_out); break;
                case 1: encoder_layer1_xmb0(input_rm, &iso_st, c_out); break;
                case 2: encoder_layer2_xdws0(input_rm, &iso_st, c_out); break;
                case 3: encoder_layer3_xmb1(input_rm, &iso_st, c_out); break;
                case 4: encoder_layer4_xdws1(input_rm, &iso_st, c_out); break;
                }

                double snr = snr_db_2d_i32(golden_out, c_out, iC[layer], iW[layer]);
                printf("  E%d iso    : SNR=%7.2f dB  [%s] (golden input)\n",
                       layer, snr, status(snr));

                /* E1 sub-step diagnostics (full pipeline, frame 0 only) */
                if (layer == 1) {
                    int E1_C = 24;
                    /* ================================================
                     * E1 Pipeline (inline, matching encoder_layer1_xmb0):
                     *   golden_in[12×65] → PConv0(g=2) → BN → AP
                     *   → Shuffle → TConv(gconv,2×3,s=2) → BN → AP
                     *   → PConv1(g=2) → BN
                     *   → cTFA(TA+FA) → Fusion → Shuffle → [24×33]
                     * ================================================ */

                    /* ---- Stage 1: PConv0 (2 groups) ---- */
                    uint16_t e1_ta_sig[24]; /* declared here for fusion scope access */
                    uint16_t e1_fa_sig[33];
                    int32_t y_pconv0_raw[24 * 65];
                    pconv2d_func(input_rm, 6, 12, 1, 65,
                                 encoder_en_convs_1_pconv1_0_weight,
                                 encoder_en_convs_1_pconv1_0_bias,
                                 E1_PCONV0_CONV_QR, 24, y_pconv0_raw);
                    pconv2d_func(input_rm + 6 * 65, 6, 12, 1, 65,
                                 encoder_en_convs_1_pconv1_0_weight + 12,
                                 encoder_en_convs_1_pconv1_0_bias + 12,
                                 E1_PCONV0_CONV_QR, 24, y_pconv0_raw + 12 * 65);

                    /* ---- Stage 2: BN ---- */
                    int32_t y_bn0[24 * 65];
                    bn_func_uw(y_pconv0_raw, encoder_en_convs_1_pconv1_1_weight,
                            encoder_en_convs_1_pconv1_1_bias,
                            encoder_en_convs_1_pconv1_1_running_mean,
                            encoder_en_convs_1_pconv1_1_running_var,
                            E1_PCONV0_BN_QR1, E1_PCONV0_BN_QR2, 24, 24 * 65, y_bn0);

                    /* ---- Stage 3: AffinePReLU ---- */
                    int32_t y_ap0[24 * 65];
                    affineprelu_func(y_bn0, encoder_en_convs_1_pconv1_2_affine_weight,
                                     encoder_en_convs_1_pconv1_2_affine_bias,
                                     encoder_en_convs_1_pconv1_2_slope_weight,
                                     E1_PCONV0_AFFINE_QR1, E1_PCONV0_AFFINE_QR2, 24, 65, y_ap0);

                    /* Check PConv0+BN+AP golden */
                    snprintf(path, sizeof(path), "%s/frame0_e1_pconv0.bin", dir);
                    int32_t *g_e1_pc0 = load_int32(path, 24 * 65);
                    if (g_e1_pc0) {
                        double snr = snr_db_2d_i32(g_e1_pc0, y_ap0, 24, 65);
                        printf("  E1.pconv0  : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                        printf("    [0..7] C:"); for(int i=0;i<8;i++) printf(" %d", y_ap0[i]); printf("\n");
                        printf("    [0..7] G:"); for(int i=0;i<8;i++) printf(" %d", g_e1_pc0[i]); printf("\n");
                        free(g_e1_pc0);
                    }

                    /* ---- Stage 4: Shuffle (interleave) ---- */
                    int32_t y_shuf0[24 * 65];
                    shuffle_interleave(y_ap0, 24, 65, y_shuf0);

                    snprintf(path, sizeof(path), "%s/frame0_e1_shuf.bin", dir);
                    int32_t *g_e1_shuf = load_int32(path, 24 * 65);
                    if (g_e1_shuf) {
                        double snr = snr_db_2d_i32(g_e1_shuf, y_shuf0, 24, 65);
                        printf("  E1.shuf    : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                        free(g_e1_shuf);
                    }

                    /* ---- Stage 5: TConv (gconv, 2×3, s=[1,2], with cache) ---- */
                    int32_t y_tconv_raw[24 * 33];
                    {
                        ulunas_state_t e1s_st;
                        ulunas_state_init(&e1s_st);
                        gconv2d_func(y_shuf0, 24, 1, 33, 2, 3, 1, 2,
                                     encoder_en_convs_1_dconv_1_weight,
                                     encoder_en_convs_1_dconv_1_bias,
                                     E1_TCONV_CONV_QR, e1s_st.conv_cache_e1, y_tconv_raw);
                    }

                    /* ---- Stage 6: TConv BN ---- */
                    int32_t y_tconv_bn[24 * 33];
                    bn_func(y_tconv_raw, encoder_en_convs_1_dconv_2_weight,
                            encoder_en_convs_1_dconv_2_bias,
                            encoder_en_convs_1_dconv_2_running_mean,
                            encoder_en_convs_1_dconv_2_running_var,
                            E1_TCONV_BN_QR1, E1_TCONV_BN_QR2, 24, 24 * 33, y_tconv_bn);

                    /* ---- Stage 7: TConv AffinePReLU ---- */
                    int32_t y_tconv_ap[24 * 33];
                    affineprelu_func(y_tconv_bn, encoder_en_convs_1_dconv_3_affine_weight,
                                     encoder_en_convs_1_dconv_3_affine_bias,
                                     encoder_en_convs_1_dconv_3_slope_weight,
                                     E1_TCONV_AFFINE_QR1, E1_TCONV_AFFINE_QR2, 24, 33, y_tconv_ap);

                    /* Check TConv golden */
                    snprintf(path, sizeof(path), "%s/frame0_enc_e1_tconv.bin", dir);
                    int32_t *g_e1_tconv = load_int32(path, 24 * 33);
                    if (g_e1_tconv) {
                        double snr = snr_db_2d_i32(g_e1_tconv, y_tconv_ap, 24, 33);
                        printf("  E1.tconv   : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                        printf("    [0..7] C:"); for(int i=0;i<8;i++) printf(" %d", y_tconv_ap[i]); printf("\n");
                        printf("    [0..7] G:"); for(int i=0;i<8;i++) printf(" %d", g_e1_tconv[i]); printf("\n");
                        free(g_e1_tconv);
                    } else {
                        /* Dump first values for manual inspection */
                        printf("  E1.tconv   : (no golden)  [0..7]=");
                        for(int i=0;i<8;i++) printf(" %d", y_tconv_ap[i]); printf("\n");
                    }

                    /* ---- Stage 8: PConv1 (2 groups) ---- */
                    int32_t y_pconv1_raw[24 * 33];
                    pconv2d_func(y_tconv_ap, 12, 12, 1, 33,
                                 encoder_en_convs_1_pconv2_0_weight,
                                 encoder_en_convs_1_pconv2_0_bias,
                                 E1_PCONV1_CONV_QR, 24, y_pconv1_raw);
                    pconv2d_func(y_tconv_ap + 12 * 33, 12, 12, 1, 33,
                                 encoder_en_convs_1_pconv2_0_weight + 12,
                                 encoder_en_convs_1_pconv2_0_bias + 12,
                                 E1_PCONV1_CONV_QR, 24, y_pconv1_raw + 12 * 33);

                    /* ---- Stage 9: PConv1 BN (in-place on y_pconv1_cTFA) ---- */
                    int32_t y_ctfa_in[24 * 33];
                    bn_func(y_pconv1_raw, encoder_en_convs_1_pconv2_1_weight,
                            encoder_en_convs_1_pconv2_1_bias,
                            encoder_en_convs_1_pconv2_1_running_mean,
                            encoder_en_convs_1_pconv2_1_running_var,
                            E1_PCONV1_BN_QR1, E1_PCONV1_BN_QR2, 24, 24 * 33, y_ctfa_in);

                    /* Check cTFA input golden */
                    snprintf(path, sizeof(path), "%s/frame0_enc_e1_ctfa_in.bin", dir);
                    int32_t *g_e1_ctfa_in = load_int32(path, 24 * 33);
                    if (g_e1_ctfa_in) {
                        double snr = snr_db_2d_i32(g_e1_ctfa_in, y_ctfa_in, 24, 33);
                        printf("  E1.ctfa_in : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                        printf("    [0..7] C:"); for(int i=0;i<8;i++) printf(" %d", y_ctfa_in[i]); printf("\n");
                        printf("    [0..7] G:"); for(int i=0;i<8;i++) printf(" %d", g_e1_ctfa_in[i]); printf("\n");
                        free(g_e1_ctfa_in);
                    } else {
                        printf("  E1.ctfa_in : (no golden)  [0..7]=");
                        for(int i=0;i<8;i++) printf(" %d", y_ctfa_in[i]); printf("\n");
                    }

                    /* ================================================
                     * TA sub-step diagnostics
                     * ================================================ */
                    {
                        int TA_C = 24, TA_W = 33, TA_nH = E1_CTFA_TA_GRU_NHID; /* nH=48 */

                        /* TA Step 1: square + mean over frequency → [C] Q20 */
                        int32_t ta_agg[24];
                        for (int c = 0; c < TA_C; c++) {
                            int64_t sum_sq = 0;
                            for (int w = 0; w < TA_W; w++) {
                                int32_t val = y_ctfa_in[c * TA_W + w];
                                sum_sq += (int64_t)val * val;
                            }
                            int64_t denom = (int64_t)TA_W * 1048576LL;
                            int64_t half = denom / 2;
                            if (sum_sq >= 0) ta_agg[c] = (int32_t)((sum_sq + half) / denom);
                            else             ta_agg[c] = (int32_t)((sum_sq - half) / denom);
                            if (ta_agg[c] < 0) ta_agg[c] = 0;
                        }

                        /* TA Step 2: GRU (fresh cache, single time step) */
                        int16_t ta_gru_out[48]; /* nH=48 */
                        int16_t h_cache_ta_diag[48] = {0};
                        gru_module(ta_agg, TA_nH, TA_C, h_cache_ta_diag,
                                   encoder_en_convs_1_pconv2_2_ta_gru_weight_ih_l0,
                                   encoder_en_convs_1_pconv2_2_ta_gru_bias_ih_l0,
                                   encoder_en_convs_1_pconv2_2_ta_gru_weight_hh_l0,
                                   encoder_en_convs_1_pconv2_2_ta_gru_bias_hh_l0,
                                   E1_CTFA_TA_GRU_QR1, E1_CTFA_TA_GRU_QR2, ta_gru_out);

                        snprintf(path, sizeof(path), "%s/frame0_e1_ta_gru.bin", dir);
                        int16_t *g_ta_gru = load_int16(path, TA_nH);
                        if (g_ta_gru) {
                            double snr = snr_db_i16(g_ta_gru, ta_gru_out, TA_nH);
                            printf("  E1.ta.gru  : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                            printf("    [0..7] C:"); for(int i=0;i<8;i++) printf(" %6d", ta_gru_out[i]); printf("\n");
                            printf("    [0..7] G:"); for(int i=0;i<8;i++) printf(" %6d", g_ta_gru[i]); printf("\n");
                            free(g_ta_gru);
                        }

                        /* TA Step 3: FC + sigmoid → [C] Q15 */
                        int ta_shift_fc = -E1_CTFA_TA_FC_QR; /* shift=8 */
                        int64_t ta_r_fc = ((int64_t)1 << (ta_shift_fc - 1));
                        /* use e1_ta_sig declared at block scope */
                        for (int c = 0; c < TA_C; c++) {
                            int64_t acc = 0;
                            for (int j = 0; j < TA_nH; j++) {
                                int64_t prod = (int64_t)ta_gru_out[j] *
                                    encoder_en_convs_1_pconv2_2_ta_fc_weight[j + TA_nH * c];
                                if (prod >= 0) acc += (prod + ta_r_fc) >> ta_shift_fc;
                                else           acc += (prod - ta_r_fc) >> ta_shift_fc;
                            }
                            int32_t fc_out = sat_i32(acc + encoder_en_convs_1_pconv2_2_ta_fc_bias[c]);
                            e1_ta_sig[c] = sigmoid_q20_to_q15(fc_out);
                        }

                        snprintf(path, sizeof(path), "%s/frame0_e1_ta_out.bin", dir);
                        uint16_t *g_ta_out = load_uint16(path, TA_C);
                        if (g_ta_out) {
                            double snr = snr_db_u16(g_ta_out, e1_ta_sig, TA_C);
                            printf("  E1.ta.sig  : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                            free(g_ta_out);
                        }

                        /* Compare with full TA module output */
                        snprintf(path, sizeof(path), "%s/frame0_enc_e1_ctfa_ta.bin", dir);
                        uint16_t *g_e1_ta = load_uint16(path, TA_C);
                        if (g_e1_ta) {
                            double snr = snr_db_u16(g_e1_ta, e1_ta_sig, TA_C);
                            printf("  E1.ta      : SNR=%7.2f dB  [%s] (full module)\n", snr, status(snr));
                            printf("    C: "); for(int i=0;i<8;i++) printf(" %d", e1_ta_sig[i]); printf("\n");
                            printf("    G: "); for(int i=0;i<8;i++) printf(" %d", g_e1_ta[i]); printf("\n");
                            free(g_e1_ta);
                        }
                    }

                    /* ================================================
                     * FA sub-step diagnostics
                     * ================================================ */
                    {
                        int FA_C = 24, FA_W = 33, FA_nH = E1_CTFA_FA_GRU_NHID; /* nH=4 */
                        int FA_G = E1_CTFA_FA_GROUP; /* 4 */
                        int FA_S = E1_CTFA_FA_SEG;   /* 9 */
                        int FA_P = E1_CTFA_FA_PAD;   /* 3 */

                        /* FA Step 1: square + mean over channels → [W] Q20 */
                        int32_t fa_agg[33];
                        for (int w = 0; w < FA_W; w++) {
                            int64_t sum_sq = 0;
                            for (int c = 0; c < FA_C; c++) {
                                int32_t val = y_ctfa_in[c * FA_W + w];
                                sum_sq += (int64_t)val * val;
                            }
                            int64_t denom = (int64_t)FA_C * 1048576LL;
                            int64_t half = denom / 2;
                            if (sum_sq >= 0) fa_agg[w] = (int32_t)((sum_sq + half) / denom);
                            else             fa_agg[w] = (int32_t)((sum_sq - half) / denom);
                            if (fa_agg[w] < 0) fa_agg[w] = 0;
                        }

                        snprintf(path, sizeof(path), "%s/frame0_e1_fa_agg.bin", dir);
                        int32_t *g_fa_agg = load_int32(path, FA_W);
                        if (g_fa_agg) {
                            double snr = snr_db_i32(g_fa_agg, fa_agg, FA_W);
                            printf("  E1.fa.agg  : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                            free(g_fa_agg);
                        }

                        /* FA Step 2: Pad + Reshape to [seg][group] */
                        int32_t fa_pad[36]; /* 33+3=36 */
                        memcpy(fa_pad, fa_agg, FA_W * sizeof(int32_t));
                        memset(fa_pad + FA_W, 0, FA_P * sizeof(int32_t));
                        int32_t fa_reshaped[36]; /* [9][4] */
                        for (int s = 0; s < FA_S; s++)
                            for (int g = 0; g < FA_G; g++)
                                fa_reshaped[s * FA_G + g] = fa_pad[g + FA_G * s];

                        /* FA Step 3: BiGRU forward (per timestep, zero init) */
                        int16_t fa_gru_fwd[36]; /* [9][4] */
                        int16_t h_fwd[4] = {0};
                        for (int t = 0; t < FA_S; t++) {
                            gru_module(&fa_reshaped[t * FA_G], FA_nH, FA_G, h_fwd,
                                       encoder_en_convs_1_pconv2_2_fa_gru_weight_ih_l0,
                                       encoder_en_convs_1_pconv2_2_fa_gru_bias_ih_l0,
                                       encoder_en_convs_1_pconv2_2_fa_gru_weight_hh_l0,
                                       encoder_en_convs_1_pconv2_2_fa_gru_bias_hh_l0,
                                       E1_CTFA_FA_GRU_QR1, E1_CTFA_FA_GRU_QR2,
                                       &fa_gru_fwd[t * FA_nH]);
                        }

                        snprintf(path, sizeof(path), "%s/frame0_e1_fa_gru0.bin", dir);
                        int16_t *g_fa_gru0 = load_int16(path, FA_S * FA_nH);
                        if (g_fa_gru0) {
                            double snr = snr_db_i16(g_fa_gru0, fa_gru_fwd, FA_S * FA_nH);
                            printf("  E1.fa.gru0 : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                            printf("    [0..7] C:"); for(int i=0;i<8;i++) printf(" %6d", fa_gru_fwd[i]); printf("\n");
                            printf("    [0..7] G:"); for(int i=0;i<8;i++) printf(" %6d", g_fa_gru0[i]); printf("\n");
                            free(g_fa_gru0);
                        }

                        /* FA Step 4: BiGRU reverse (process t=8..0, zero init) */
                        int16_t fa_gru_rev[36]; /* [9][4] in reverse processing order */
                        int16_t h_rev[4] = {0};
                        for (int t = 0; t < FA_S; t++) {
                            int t_rev = FA_S - 1 - t;
                            gru_module(&fa_reshaped[t_rev * FA_G], FA_nH, FA_G, h_rev,
                                       encoder_en_convs_1_pconv2_2_fa_gru_weight_ih_l0_reverse,
                                       encoder_en_convs_1_pconv2_2_fa_gru_bias_ih_l0_reverse,
                                       encoder_en_convs_1_pconv2_2_fa_gru_weight_hh_l0_reverse,
                                       encoder_en_convs_1_pconv2_2_fa_gru_bias_hh_l0_reverse,
                                       E1_CTFA_FA_GRU_QR1, E1_CTFA_FA_GRU_QR2,
                                       &fa_gru_rev[t * FA_nH]);
                        }

                        snprintf(path, sizeof(path), "%s/frame0_e1_fa_gru1.bin", dir);
                        int16_t *g_fa_gru1 = load_int16(path, FA_S * FA_nH);
                        if (g_fa_gru1) {
                            double snr = snr_db_i16(g_fa_gru1, fa_gru_rev, FA_S * FA_nH);
                            printf("  E1.fa.gru1 : SNR=%7.2f dB  [%s] (raw rev order)\n", snr, status(snr));
                            free(g_fa_gru1);
                        }

                        /* FA Step 5: Concat forward + flipped reverse → [seg][2*nH] */
                        int16_t fa_gru_cat[72]; /* [9][8] */
                        for (int t = 0; t < FA_S; t++) {
                            memcpy(&fa_gru_cat[t * (2 * FA_nH)], &fa_gru_fwd[t * FA_nH],
                                   FA_nH * sizeof(int16_t));
                            memcpy(&fa_gru_cat[t * (2 * FA_nH) + FA_nH],
                                   &fa_gru_rev[(FA_S - 1 - t) * FA_nH],
                                   FA_nH * sizeof(int16_t));
                        }

                        /* FA Step 6: FC [seg][2*nH] → [seg][group] */
                        int fa_shift_fc = -E1_CTFA_FA_FC_QR; /* shift=9 */
                        int64_t fa_r_fc = ((int64_t)1 << (fa_shift_fc - 1));
                        int FA_in_fc = 2 * FA_nH; /* 8 */
                        int32_t fa_fc_2d[36]; /* [9][4] */
                        for (int s = 0; s < FA_S; s++) {
                            for (int g = 0; g < FA_G; g++) {
                                int64_t acc = 0;
                                for (int i = 0; i < FA_in_fc; i++) {
                                    int64_t prod = (int64_t)fa_gru_cat[s * FA_in_fc + i] *
                                        encoder_en_convs_1_pconv2_2_fa_fc_weight[i + FA_in_fc * g];
                                    if (prod >= 0) acc += (prod + fa_r_fc) >> fa_shift_fc;
                                    else           acc += (prod - fa_r_fc) >> fa_shift_fc;
                                }
                                fa_fc_2d[s * FA_G + g] = sat_i32(acc +
                                    encoder_en_convs_1_pconv2_2_fa_fc_bias[g]);
                            }
                        }
                        /* Reshape: x_flat[g + G*s] = fc_2d[s*G + g] */
                        int32_t fa_fc_flat[36];
                        for (int g = 0; g < FA_G; g++)
                            for (int s = 0; s < FA_S; s++)
                                fa_fc_flat[g + FA_G * s] = fa_fc_2d[s * FA_G + g];

                        snprintf(path, sizeof(path), "%s/frame0_e1_fa_fc.bin", dir);
                        int32_t *g_fa_fc = load_int32(path, FA_S * FA_G);
                        if (g_fa_fc) {
                            double snr = snr_db_i32(g_fa_fc, fa_fc_flat, FA_S * FA_G);
                            printf("  E1.fa.fc   : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                            free(g_fa_fc);
                        }

                        /* FA Step 7: Sigmoid [0:W-1] (de-pad: keep first W elements) */
                        /* use e1_fa_sig declared at block scope */
                        for (int w = 0; w < FA_W; w++)
                            e1_fa_sig[w] = sigmoid_q20_to_q15(fa_fc_flat[w]);

                        snprintf(path, sizeof(path), "%s/frame0_e1_fa_out.bin", dir);
                        uint16_t *g_fa_out = load_uint16(path, FA_W);
                        if (g_fa_out) {
                            double snr = snr_db_u16(g_fa_out, e1_fa_sig, FA_W);
                            printf("  E1.fa.sig  : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                            free(g_fa_out);
                        }

                        /* Compare with full FA module output */
                        snprintf(path, sizeof(path), "%s/frame0_enc_e1_ctfa_fa.bin", dir);
                        uint16_t *g_e1_fa = load_uint16(path, FA_W);
                        if (g_e1_fa) {
                            double snr = snr_db_u16(g_e1_fa, e1_fa_sig, FA_W);
                            printf("  E1.fa      : SNR=%7.2f dB  [%s] (full module)\n", snr, status(snr));
                            free(g_e1_fa);
                        }

                        /* ================================================
                         * cTFA Fusion + Final Shuffle
                         * y_t = round(y_ctfa_in .* ta' * 2^(-15))
                         * y   = shuffle(round(y_t .* fa * 2^(-15)))
                         * ================================================ */
                        int32_t y_fusion[24 * 33];
                        int64_t r_f = 16384;
                        for (int c = 0; c < 24; c++) {
                            for (int w = 0; w < 33; w++) {
                                int64_t p1 = (int64_t)y_ctfa_in[c * 33 + w] * e1_ta_sig[c];
                                int32_t yt;
                                if (p1 >= 0) yt = (int32_t)((p1 + r_f) >> 15);
                                else         yt = (int32_t)((p1 - r_f) >> 15);
                                int64_t p2 = (int64_t)yt * e1_fa_sig[w];
                                if (p2 >= 0) y_fusion[c * 33 + w] = (int32_t)((p2 + r_f) >> 15);
                                else         y_fusion[c * 33 + w] = (int32_t)((p2 - r_f) >> 15);
                            }
                        }

                        /* Final shuffle (interleave) */
                        int32_t y_e1_final[24 * 33];
                        shuffle_interleave(y_fusion, 24, 33, y_e1_final);

                        /* Check fusion+shuffle against enc_e1 golden = final E1 output */
                        snprintf(path, sizeof(path), "%s/frame0_enc_e1.bin", dir);
                        int32_t *g_e1_out = load_int32(path, 24 * 33);
                        if (g_e1_out) {
                            double snr = snr_db_2d_i32(g_e1_out, y_e1_final, 24, 33);
                            printf("  E1.final   : SNR=%7.2f dB  [%s] (fusion+shuffle vs golden)\n",
                                   snr, status(snr));
                            printf("    [0..7] C:"); for(int i=0;i<8;i++) printf(" %d", y_e1_final[i]); printf("\n");
                            printf("    [0..7] G:"); for(int i=0;i<8;i++) printf(" %d", g_e1_out[i]); printf("\n");
                            free(g_e1_out);
                        }
                    }
                }

                free(golden_in); free(golden_out); free(c_out); free(input_rm);
            }
        }

        /* Step 4: GDPRNN */
        int32_t r1[16 * 33], r2[16 * 33];
        gdprnn_module(e4, st.inter_cache_0, 0, r1);
        gdprnn_module(r1, st.inter_cache_1, 1, r2);

        /* --- RNN1 Sub-Step Diagnostics (frame 0 only, golden E4 input) --- */
        if (frame == 0) {
            printf("\n  --- RNN1 Intra Sub-Steps (golden E4 input) ---\n");

            /* Load golden E4 as input (like E1 iso test) to isolate RNN bugs */
            snprintf(path, sizeof(path), "%s/frame0_enc_e4.bin", dir);
            int32_t *golden_e4 = load_int32(path, 16 * 33);
            if (!golden_e4) {
                printf("  rnn1 iso   : SKIP (no golden E4)\n");
            } else {
            /* Transpose: golden_e4 is [16][33] (MATLAB col-major) → x_tpose[33][16] */
            int32_t x_tpose[33 * 16];
            for (int t = 0; t < 33; t++)
                for (int c = 0; c < 16; c++)
                    x_tpose[t * 16 + c] = golden_e4[c * 33 + t];

            /* Check intra input */
            snprintf(path, sizeof(path), "%s/frame0_rnn1_intra_in.bin", dir);
            int32_t *g_intra_in = load_int32(path, 33 * 16);
            if (g_intra_in) {
                double snr = snr_db_i32(g_intra_in, x_tpose, 33 * 16);
                printf("  rnn1.intra_in : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                free(g_intra_in);
            }

            /* Split into 2 groups */
            int32_t x0[33 * 8], x1[33 * 8];
            for (int t = 0; t < 33; t++) {
                memcpy(&x0[t * 8], &x_tpose[t * 16], 8 * sizeof(int32_t));
                memcpy(&x1[t * 8], &x_tpose[t * 16 + 8], 8 * sizeof(int32_t));
            }

            /* BiGRU group 0: nHidden=4, in_dim=8 */
            int16_t x0_gru[33 * 8];
            bigru_module(x0, 33, 4, 8,
                         dpgrnn_0_intra_rnn_rnn1_weight_ih_l0,
                         dpgrnn_0_intra_rnn_rnn1_bias_ih_l0,
                         dpgrnn_0_intra_rnn_rnn1_weight_hh_l0,
                         dpgrnn_0_intra_rnn_rnn1_bias_hh_l0,
                         dpgrnn_0_intra_rnn_rnn1_weight_ih_l0_reverse,
                         dpgrnn_0_intra_rnn_rnn1_bias_ih_l0_reverse,
                         dpgrnn_0_intra_rnn_rnn1_weight_hh_l0_reverse,
                         dpgrnn_0_intra_rnn_rnn1_bias_hh_l0_reverse,
                         DPRNN_GRU_QR1, DPRNN_GRU_QR2, x0_gru);

            snprintf(path, sizeof(path), "%s/frame0_rnn1_intra_gru0.bin", dir);
            int16_t *g_ig0 = load_int16(path, 33 * 8);
            if (g_ig0) {
                double snr = snr_db_i16(g_ig0, x0_gru, 33 * 8);
                printf("  rnn1.intra_gru0: SNR=%7.2f dB  [%s]\n", snr, status(snr));
                printf("    [0..7] C:"); for(int i=0;i<8;i++) printf(" %6d", x0_gru[i]); printf("\n");
                printf("    [0..7] G:"); for(int i=0;i<8;i++) printf(" %6d", g_ig0[i]); printf("\n");
                free(g_ig0);
            }

            /* BiGRU group 1 */
            int16_t x1_gru[33 * 8];
            bigru_module(x1, 33, 4, 8,
                         dpgrnn_0_intra_rnn_rnn2_weight_ih_l0,
                         dpgrnn_0_intra_rnn_rnn2_bias_ih_l0,
                         dpgrnn_0_intra_rnn_rnn2_weight_hh_l0,
                         dpgrnn_0_intra_rnn_rnn2_bias_hh_l0,
                         dpgrnn_0_intra_rnn_rnn2_weight_ih_l0_reverse,
                         dpgrnn_0_intra_rnn_rnn2_bias_ih_l0_reverse,
                         dpgrnn_0_intra_rnn_rnn2_weight_hh_l0_reverse,
                         dpgrnn_0_intra_rnn_rnn2_bias_hh_l0_reverse,
                         DPRNN_GRU_QR1, DPRNN_GRU_QR2, x1_gru);

            snprintf(path, sizeof(path), "%s/frame0_rnn1_intra_gru1.bin", dir);
            int16_t *g_ig1 = load_int16(path, 33 * 8);
            if (g_ig1) {
                double snr = snr_db_i16(g_ig1, x1_gru, 33 * 8);
                printf("  rnn1.intra_gru1: SNR=%7.2f dB  [%s]\n", snr, status(snr));
                free(g_ig1);
            }

            /* Concat */
            int16_t x_cat[33 * 16];
            for (int t = 0; t < 33; t++) {
                memcpy(&x_cat[t * 16], &x0_gru[t * 8], 8 * sizeof(int16_t));
                memcpy(&x_cat[t * 16 + 8], &x1_gru[t * 8], 8 * sizeof(int16_t));
            }

            snprintf(path, sizeof(path), "%s/frame0_rnn1_intra_cat.bin", dir);
            int16_t *g_icat = load_int16(path, 33 * 16);
            if (g_icat) {
                double snr = snr_db_i16(g_icat, x_cat, 33 * 16);
                printf("  rnn1.intra_cat : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                free(g_icat);
            }

            /* Intra FC */
            int shift_fc_i = -DPRNN_INTRA_FC_QR; /* 9 */
            int64_t r_fc_i = ((int64_t)1 << (shift_fc_i - 1));
            int32_t x_fc_i[33 * 16];
            for (int t = 0; t < 33; t++) {
                for (int o = 0; o < 16; o++) {
                    int64_t acc = 0;
                    for (int i = 0; i < 16; i++) {
                        int64_t prod = (int64_t)x_cat[t * 16 + i] * dpgrnn_0_intra_fc_weight[i + 16 * o];
                        if (prod >= 0) acc += (prod + r_fc_i) >> shift_fc_i;
                        else           acc += (prod - r_fc_i) >> shift_fc_i;
                    }
                    x_fc_i[t * 16 + o] = sat_i32(acc + dpgrnn_0_intra_fc_bias[o]);
                }
            }

            snprintf(path, sizeof(path), "%s/frame0_rnn1_intra_fc.bin", dir);
            int32_t *g_ifc = load_int32(path, 33 * 16);
            if (g_ifc) {
                double snr = snr_db_i32(g_ifc, x_fc_i, 33 * 16);
                printf("  rnn1.intra_fc  : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                printf("    [0..7] C:"); for(int i=0;i<8;i++) printf(" %d", x_fc_i[i]); printf("\n");
                printf("    [0..7] G:"); for(int i=0;i<8;i++) printf(" %d", g_ifc[i]); printf("\n");
                free(g_ifc);
            }

            /* Intra LN */
            int32_t x_ln_i[33 * 16];
            ln_func(x_fc_i, dpgrnn_0_intra_ln_weight, dpgrnn_0_intra_ln_bias,
                    DPRNN_INTRA_LN_QR, 16, 33 * 16, x_ln_i);

            snprintf(path, sizeof(path), "%s/frame0_rnn1_intra_ln.bin", dir);
            int32_t *g_iln = load_int32(path, 33 * 16);
            if (g_iln) {
                double snr = snr_db_i32(g_iln, x_ln_i, 33 * 16);
                printf("  rnn1.intra_ln  : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                printf("    [0..7] C:"); for(int i=0;i<8;i++) printf(" %d", x_ln_i[i]); printf("\n");
                printf("    [0..7] G:"); for(int i=0;i<8;i++) printf(" %d", g_iln[i]); printf("\n");
                free(g_iln);
            }

            /* Intra residual: y_intra = x + x_ln, output as [16][33] */
            int32_t y_intra[16 * 33];
            for (int t = 0; t < 33; t++)
                for (int c = 0; c < 16; c++)
                    y_intra[c * 33 + t] = sat_i32((int64_t)golden_e4[c * 33 + t] + x_ln_i[t * 16 + c]);

            snprintf(path, sizeof(path), "%s/frame0_rnn1_intra_out.bin", dir);
            int32_t *g_iout = load_int32(path, 33 * 16);
            if (g_iout) {
                /* Golden is [33][16], C output is [16][33] — transpose for comparison */
                int32_t *gt = malloc(33 * 16 * sizeof(int32_t));
                for (int t = 0; t < 33; t++)
                    for (int c = 0; c < 16; c++)
                        gt[t * 16 + c] = g_iout[t * 16 + c];
                double snr = snr_db_2d_i32(gt, y_intra, 16, 33);
                printf("  rnn1.intra_out : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                free(gt); free(g_iout);
            }

            /* ================================================
             * Inter-RNN sub-steps
             * ================================================ */
            printf("\n  --- RNN1 Inter Sub-Steps ---\n");

            /* Transpose y_intra[16][33] → x_inter[33][16] */
            int32_t x_inter[33 * 16];
            for (int t = 0; t < 33; t++)
                for (int c = 0; c < 16; c++)
                    x_inter[t * 16 + c] = y_intra[c * 33 + t];

            snprintf(path, sizeof(path), "%s/frame0_rnn1_inter_in.bin", dir);
            int32_t *g_inter_in = load_int32(path, 33 * 16);
            if (g_inter_in) {
                double snr = snr_db_i32(g_inter_in, x_inter, 33 * 16);
                printf("  rnn1.inter_in  : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                free(g_inter_in);
            }

            /* Split */
            int32_t xi0[33 * 8], xi1[33 * 8];
            for (int t = 0; t < 33; t++) {
                memcpy(&xi0[t * 8], &x_inter[t * 16], 8 * sizeof(int32_t));
                memcpy(&xi1[t * 8], &x_inter[t * 16 + 8], 8 * sizeof(int32_t));
            }

            /* Inter GRU: per-time-step independent hidden states (matching MATLAB parallel) */
            int16_t xi0_gru[33 * 8], xi1_gru[33 * 8];
            int16_t inter_cache_diag[33 * 16];
            memset(inter_cache_diag, 0, 33 * 16 * sizeof(int16_t));
            for (int t = 0; t < 33; t++) {
                int16_t h0[8], h1[8];
                memcpy(h0, &inter_cache_diag[t * 8], 8 * sizeof(int16_t));
                memcpy(h1, &inter_cache_diag[33 * 8 + t * 8], 8 * sizeof(int16_t));
                gru_module(&xi0[t * 8], 8, 8, h0,
                           dpgrnn_0_inter_rnn_rnn1_weight_ih_l0,
                           dpgrnn_0_inter_rnn_rnn1_bias_ih_l0,
                           dpgrnn_0_inter_rnn_rnn1_weight_hh_l0,
                           dpgrnn_0_inter_rnn_rnn1_bias_hh_l0,
                           DPRNN_GRU_QR1, DPRNN_GRU_QR2, &xi0_gru[t * 8]);
                gru_module(&xi1[t * 8], 8, 8, h1,
                           dpgrnn_0_inter_rnn_rnn2_weight_ih_l0,
                           dpgrnn_0_inter_rnn_rnn2_bias_ih_l0,
                           dpgrnn_0_inter_rnn_rnn2_weight_hh_l0,
                           dpgrnn_0_inter_rnn_rnn2_bias_hh_l0,
                           DPRNN_GRU_QR1, DPRNN_GRU_QR2, &xi1_gru[t * 8]);
                memcpy(&inter_cache_diag[t * 8], h0, 8 * sizeof(int16_t));
                memcpy(&inter_cache_diag[33 * 8 + t * 8], h1, 8 * sizeof(int16_t));
            }

            snprintf(path, sizeof(path), "%s/frame0_rnn1_inter_gru0.bin", dir);
            int16_t *g_eg0 = load_int16(path, 33 * 8);
            if (g_eg0) {
                double snr = snr_db_i16(g_eg0, xi0_gru, 33 * 8);
                printf("  rnn1.inter_gru0: SNR=%7.2f dB  [%s]\n", snr, status(snr));
                printf("    [0..7] C:"); for(int i=0;i<8;i++) printf(" %6d", xi0_gru[i]); printf("\n");
                printf("    [0..7] G:"); for(int i=0;i<8;i++) printf(" %6d", g_eg0[i]); printf("\n");
                free(g_eg0);
            }

            snprintf(path, sizeof(path), "%s/frame0_rnn1_inter_gru1.bin", dir);
            int16_t *g_eg1 = load_int16(path, 33 * 8);
            if (g_eg1) {
                double snr = snr_db_i16(g_eg1, xi1_gru, 33 * 8);
                printf("  rnn1.inter_gru1: SNR=%7.2f dB  [%s]\n", snr, status(snr));
                free(g_eg1);
            }

            /* Inter Concat */
            int16_t xi_cat[33 * 16];
            for (int t = 0; t < 33; t++) {
                memcpy(&xi_cat[t * 16], &xi0_gru[t * 8], 8 * sizeof(int16_t));
                memcpy(&xi_cat[t * 16 + 8], &xi1_gru[t * 8], 8 * sizeof(int16_t));
            }

            snprintf(path, sizeof(path), "%s/frame0_rnn1_inter_cat.bin", dir);
            int16_t *g_ecat = load_int16(path, 33 * 16);
            if (g_ecat) {
                double snr = snr_db_i16(g_ecat, xi_cat, 33 * 16);
                printf("  rnn1.inter_cat : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                free(g_ecat);
            }

            /* Inter FC */
            int shift_fc_e = -DPRNN_INTER_FC_QR; /* 9 */
            int64_t r_fc_e = ((int64_t)1 << (shift_fc_e - 1));
            int32_t x_fc_e[33 * 16];
            for (int t = 0; t < 33; t++) {
                for (int o = 0; o < 16; o++) {
                    int64_t acc = 0;
                    for (int i = 0; i < 16; i++) {
                        int64_t prod = (int64_t)xi_cat[t * 16 + i] * dpgrnn_0_inter_fc_weight[i + 16 * o];
                        if (prod >= 0) acc += (prod + r_fc_e) >> shift_fc_e;
                        else           acc += (prod - r_fc_e) >> shift_fc_e;
                    }
                    x_fc_e[t * 16 + o] = sat_i32(acc + dpgrnn_0_inter_fc_bias[o]);
                }
            }

            snprintf(path, sizeof(path), "%s/frame0_rnn1_inter_fc.bin", dir);
            int32_t *g_efc = load_int32(path, 33 * 16);
            if (g_efc) {
                double snr = snr_db_i32(g_efc, x_fc_e, 33 * 16);
                printf("  rnn1.inter_fc  : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                printf("    [0..7] C:"); for(int i=0;i<8;i++) printf(" %d", x_fc_e[i]); printf("\n");
                printf("    [0..7] G:"); for(int i=0;i<8;i++) printf(" %d", g_efc[i]); printf("\n");
                free(g_efc);
            }

            /* Inter LN */
            int32_t x_ln_e[33 * 16];
            ln_func(x_fc_e, dpgrnn_0_inter_ln_weight, dpgrnn_0_inter_ln_bias,
                    DPRNN_INTER_LN_QR, 16, 33 * 16, x_ln_e);

            snprintf(path, sizeof(path), "%s/frame0_rnn1_inter_ln.bin", dir);
            int32_t *g_eln = load_int32(path, 33 * 16);
            if (g_eln) {
                double snr = snr_db_i32(g_eln, x_ln_e, 33 * 16);
                printf("  rnn1.inter_ln  : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                printf("    [0..7] C:"); for(int i=0;i<8;i++) printf(" %d", x_ln_e[i]); printf("\n");
                printf("    [0..7] G:"); for(int i=0;i<8;i++) printf(" %d", g_eln[i]); printf("\n");
                free(g_eln);
            }

            /* Inter residual → output as [16][33] */
            int32_t y_inter[16 * 33];
            for (int t = 0; t < 33; t++)
                for (int c = 0; c < 16; c++)
                    y_inter[c * 33 + t] = sat_i32((int64_t)y_intra[c * 33 + t] + x_ln_e[t * 16 + c]);

            snprintf(path, sizeof(path), "%s/frame0_rnn1_inter_out.bin", dir);
            int32_t *g_eout = load_int32(path, 33 * 16);
            if (g_eout) {
                int32_t *gt = malloc(33 * 16 * sizeof(int32_t));
                for (int t = 0; t < 33; t++)
                    for (int c = 0; c < 16; c++)
                        gt[t * 16 + c] = g_eout[t * 16 + c];
                double snr = snr_db_2d_i32(gt, y_inter, 16, 33);
                printf("  rnn1.inter_out : SNR=%7.2f dB  [%s]\n", snr, status(snr));
                free(gt); free(g_eout);
            }
            }
            free(golden_e4);
        } /* end RNN1 frame 0 diagnostics */

        snprintf(path, sizeof(path), "%s/frame%d_rnn1.bin", dir, frame);
        int32_t *gr1 = load_int32(path, 16 * 33);
        if (gr1) {
            double snr = snr_db_2d_i32(gr1, r1, 16, 33);
            printf("  rnn1      : SNR=%7.2f dB  [%s]\n", snr, status(snr));
            tested++; if (snr > 80.0) passed++;
            free(gr1);
        }

        snprintf(path, sizeof(path), "%s/frame%d_rnn2.bin", dir, frame);
        int32_t *gr2 = load_int32(path, 16 * 33);
        if (gr2) {
            double snr = snr_db_2d_i32(gr2, r2, 16, 33);
            printf("  rnn2      : SNR=%7.2f dB  [%s]\n", snr, status(snr));
            tested++; if (snr > 80.0) passed++;
            free(gr2);
        }

        /* Step 5-8: Decoder + Sigmoid + BS + MASK */
        int32_t y_dec[1 * 129];
        decoder_module(r2, &st, e0, e1, e2, e3, e4, y_dec);

        snprintf(path, sizeof(path), "%s/frame%d_dec.bin", dir, frame);
        int32_t *gd = load_int32(path, 1 * 129);
        if (gd) {
            double snr = snr_db_i32(gd, y_dec, 1 * 129);
            printf("  dec       : SNR=%7.2f dB  [%s]\n", snr, status(snr));
            tested++; if (snr > 80.0) passed++;
            free(gd);
        }

        /* Sigmoid */
        uint16_t y_sig[1 * 129];
        for (int i = 0; i < 1 * 129; i++) {
            y_sig[i] = sigmoid_q20_to_q15(y_dec[i]);
        }

        /* BS → [1, 257] single-channel mask */
        int16_t y_bs[257];
        bs_fixed(y_sig, erb_ierb_fc_weight, 129, 257, y_bs);

        /* MASK */
        {
            int32_t real_q20[257], imag_q20[257];
            for (int i = 0; i < 257; i++) {
                real_q20[i] = (int32_t)round(real_in[i] * 1048576.0f);
                imag_q20[i] = (int32_t)round(imag_in[i] * 1048576.0f);
            }
            int32_t y_mask[2 * 257];
            mask_fixed(y_bs, real_q20, imag_q20, 257, y_mask);

            snprintf(path, sizeof(path), "%s/frame%d_mask.bin", dir, frame);
            int32_t *gm = load_int32(path, 2 * 257);
            if (gm) {
                double snr = snr_db_i32(gm, y_mask, 2 * 257);
                printf("  MASK      : SNR=%7.2f dB  MAX=%6.1f  [%s]\n",
                       snr, max_abs_err_i32(gm, y_mask, 2 * 257), status(snr));
                tested++; if (snr > 80.0) passed++;
                free(gm);
            }
        }

        free(real_in); free(imag_in);
        printf("\n");
    }

    printf("=== Result: %d/%d layers PASS (>= 80dB) ===\n", passed, tested);
    return (passed == tested) ? 0 : 1;
}
