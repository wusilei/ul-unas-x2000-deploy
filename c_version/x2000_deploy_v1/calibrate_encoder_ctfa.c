/**
 * calibrate_encoder_ctfa.c — Encoder cTFA Full-Chain Calibration
 * ==============================================================
 * Recalibrates encoder cTFA QRs (e0-e4) using the v3 method:
 * real Encoder_module + DPRNN + Decoder_module, optimizing
 * final dec SNR against frame0_dec.bin.
 *
 * Forward calibration (e0→e4): each module's output feeds the
 * next, so earlier modules affect entire downstream chain.
 *
 * Build:
 *   gcc -O2 -std=c99 -DQR_CALIBRATION_MODE -o calibrate_encoder_ctfa \
 *       calibrate_encoder_ctfa.c ulunas_fp.c ulunas_modules.c \
 *       ulunas_infer.c ulunas_matlab_weights.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"
#include "ulunas_ctfa_qr.h"

static double snr_db(const int32_t *g, const int32_t *t, int n) {
    double s = 0, e = 0;
    for (int i = 0; i < n; i++) { double gv = g[i], d = gv - t[i]; s += gv * gv; e += d * d; }
    return e < 1e-30 ? 999 : 10 * log10(s / e);
}
static int32_t *load_int32(const char *p, int n) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    int32_t *b = malloc(n * sizeof(int32_t));
    if (fread(b, sizeof(int32_t), n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    fclose(f); return b;
}
static float *load_float(const char *p, int n) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    float *b = malloc(n * sizeof(float));
    if (fread(b, sizeof(float), n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    fclose(f); return b;
}

int main() {
    printf("=== Encoder cTFA Full-Chain Calibration ===\n\n");

    float *ri = load_float("dump_matlab/frame0_stft_real.bin", 257);
    float *ii = load_float("dump_matlab/frame0_stft_imag.bin", 257);
    int32_t *gd = load_int32("dump_matlab/frame0_dec.bin", 129);
    if (!ri || !ii || !gd) { printf("ERROR: missing files\n"); return 1; }

    int32_t xl[257]; log_gen_fixed(ri, ii, 257, xl);
    int32_t xb[129];  BM_fixed(xl, erb_erb_fc_weight, xb);

    /* Baseline */
    {
        ulunas_state_t st; ulunas_state_init(&st);
        int32_t e0[12*65], e1[24*33], e2[24*33], e3[32*33], e4[16*33];
        Encoder_module(xb, &st, e0, e1, e2, e3, e4);
        int32_t r1[16*33], r2[16*33];
        GDPRNN_module(e4, st.inter_prev0, 0, r1);
        GDPRNN_module(r1, st.inter_prev1, 1, r2);
        int32_t yd[129];
        Decoder_module(r2, &st, e0, e1, e2, e3, e4, yd);
        printf("Baseline dec SNR = %.2f dB\n\n", snr_db(yd, gd, 129));
    }

    /* Encoder module descriptors */
    ctfa_qr_t *qrs[] = {&g_qr_e0, &g_qr_e1, &g_qr_e2, &g_qr_e3, &g_qr_e4};
    const char *names[] = {"e0 XConv", "e1 XMB0", "e2 XDWS0", "e3 XMB1", "e4 XDWS1"};

    /* Coarse cTFA sweep grid */
    int ta1s[] = {-22, -19, -16, -12, -8, -4};
    int ta2s[] = {-14, -10, -6, -2};
    int tfcs[] = {-24, -19, -12, -8, -4};
    int fa1s[] = {-22, -16, -12, -8, -4};
    int fa2s[] = {-22, -14, -10, -6, -2};
    int ffcs[] = {-12, -8, -4};
    int n1 = 6, n2 = 4, n3 = 5, n4 = 5, n5 = 5, n6 = 3;
    int total = n1 * n2 * n3 * n4 * n5 * n6;  /* 9000 */

    /* Forward calibration */
    for (int m = 0; m < 5; m++) {
        ctfa_qr_t *qr = qrs[m];
        ctfa_qr_t orig = *qr;
        printf("--- %s ---\n", names[m]);
        printf("  current: ta=(%d,%d,%d) fa=(%d,%d,%d)\n",
               qr->ta_qr1, qr->ta_qr2, qr->ta_fc, qr->fa_qr1, qr->fa_qr2, qr->fa_fc);

        double best = -999;
        ctfa_qr_t best_qr = *qr;
        int tested = 0;

        for (int i1 = 0; i1 < n1; i1++)
        for (int i2 = 0; i2 < n2; i2++)
        for (int i3 = 0; i3 < n3; i3++)
        for (int j1 = 0; j1 < n4; j1++)
        for (int j2 = 0; j2 < n5; j2++)
        for (int j3 = 0; j3 < n6; j3++) {
            *qr = (ctfa_qr_t){ta1s[i1], ta2s[i2], tfcs[i3], fa1s[j1], fa2s[j2], ffcs[j3]};

            ulunas_state_t st; ulunas_state_init(&st);
            int32_t e0[12*65], e1[24*33], e2[24*33], e3[32*33], e4[16*33];
            Encoder_module(xb, &st, e0, e1, e2, e3, e4);
            int32_t r1[16*33], r2[16*33];
            GDPRNN_module(e4, st.inter_prev0, 0, r1);
            GDPRNN_module(r1, st.inter_prev1, 1, r2);
            int32_t yd[129];
            Decoder_module(r2, &st, e0, e1, e2, e3, e4, yd);
            double s = snr_db(yd, gd, 129);
            tested++;

            if (s > best) { best = s; best_qr = *qr; }
        }

        *qr = best_qr;
        printf("  best: SNR=%.2f dB  ta=(%d,%d,%d) fa=(%d,%d,%d)  [%d tested]\n\n",
               best, best_qr.ta_qr1, best_qr.ta_qr2, best_qr.ta_fc,
               best_qr.fa_qr1, best_qr.fa_qr2, best_qr.fa_fc, tested);
    }

    /* Final measurement */
    {
        ulunas_state_t st; ulunas_state_init(&st);
        int32_t e0[12*65], e1[24*33], e2[24*33], e3[32*33], e4[16*33];
        Encoder_module(xb, &st, e0, e1, e2, e3, e4);
        int32_t r1[16*33], r2[16*33];
        GDPRNN_module(e4, st.inter_prev0, 0, r1);
        GDPRNN_module(r1, st.inter_prev1, 1, r2);
        int32_t yd[129];
        Decoder_module(r2, &st, e0, e1, e2, e3, e4, yd);
        printf("=== Final dec SNR: %.2f dB ===\n\n", snr_db(yd, gd, 129));
    }

    printf("=== UPDATE ulunas_modules.c ===\n");
    for (int m = 0; m < 5; m++) {
        ctfa_qr_t *qr = qrs[m];
        printf("%s: ta=(%d,%d,%d) fa=(%d,%d,%d)\n",
               names[m], qr->ta_qr1, qr->ta_qr2, qr->ta_fc, qr->fa_qr1, qr->fa_qr2, qr->fa_fc);
    }

    free(ri); free(ii); free(gd);
    return 0;
}
