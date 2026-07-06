/**
 * calibrate_decoder_ctfa_v3.c — Definitive Chain Calibration
 * ==========================================================
 * KEY INSIGHT: Uses the REAL Decoder_module (no code duplication)
 * by compiling with -DQR_CALIBRATION_MODE. Sets global g_qr_d0..d4
 * variables that ulunas_modules.c reads via macros.
 *
 * Build:
 *   gcc -O2 -std=c99 -DQR_CALIBRATION_MODE -o calibrate_decoder_ctfa_v3 \
 *       calibrate_decoder_ctfa_v3.c ulunas_fp.c ulunas_modules.c \
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

/* Pointer to the QR we're currently calibrating */
static ctfa_qr_t *calib_qr = NULL;

static void calibrate_one(const char *name, ctfa_qr_t *qr,
                          const int32_t *r2,
                          const int32_t *e0, const int32_t *e1,
                          const int32_t *e2, const int32_t *e3, const int32_t *e4,
                          const int32_t *golden_dec) {
    printf("--- %s ---\n", name);
    calib_qr = qr;

    /* Fresh state each calibration pass */
    ulunas_state_t st; ulunas_state_init(&st);
    int32_t y_dec[1*129];

    /* Baseline */
    Decoder_module(r2, &st, e0, e1, e2, e3, e4, y_dec);
    double base = snr_db(y_dec, golden_dec, 129);
    printf("  base=%5.1f dB  (%d,%d,%d | %d,%d,%d)\n",
           base, qr->ta_qr1, qr->ta_qr2, qr->ta_fc,
           qr->fa_qr1, qr->fa_qr2, qr->fa_fc);

    /* Coarse grid */
    double best = base;
    ctfa_qr_t best_qr = *qr;
    for (int ta1 = -18; ta1 <= -4; ta1 += 2)
    for (int ta2 = -14; ta2 <= -2; ta2 += 2)
    for (int tfc = -12; tfc <= -2; tfc += 2)
    for (int fa1 = -18; fa1 <= -4; fa1 += 2)
    for (int fa2 = -14; fa2 <= -2; fa2 += 2)
    for (int ffc = -12; ffc <= -2; ffc += 2) {
        *qr = (ctfa_qr_t){ta1, ta2, tfc, fa1, fa2, ffc};
        ulunas_state_init(&st);
        Decoder_module(r2, &st, e0, e1, e2, e3, e4, y_dec);
        double s = snr_db(y_dec, golden_dec, 129);
        if (s > best) { best = s; best_qr = *qr; }
    }

    /* Fine grid */
    for (int ta1 = best_qr.ta_qr1-1; ta1 <= best_qr.ta_qr1+1; ta1++)
    for (int ta2 = best_qr.ta_qr2-1; ta2 <= best_qr.ta_qr2+1; ta2++)
    for (int tfc = best_qr.ta_fc-1; tfc <= best_qr.ta_fc+1; tfc++)
    for (int fa1 = best_qr.fa_qr1-1; fa1 <= best_qr.fa_qr1+1; fa1++)
    for (int fa2 = best_qr.fa_qr2-1; fa2 <= best_qr.fa_qr2+1; fa2++)
    for (int ffc = best_qr.fa_fc-1; ffc <= best_qr.fa_fc+1; ffc++) {
        *qr = (ctfa_qr_t){ta1, ta2, tfc, fa1, fa2, ffc};
        ulunas_state_init(&st);
        Decoder_module(r2, &st, e0, e1, e2, e3, e4, y_dec);
        double s = snr_db(y_dec, golden_dec, 129);
        if (s > best) { best = s; best_qr = *qr; }
    }

    *qr = best_qr;
    printf("  best=%5.1f dB  Δ=%+.1f  (%d,%d,%d | %d,%d,%d)\n\n",
           best, best - base,
           best_qr.ta_qr1, best_qr.ta_qr2, best_qr.ta_fc,
           best_qr.fa_qr1, best_qr.fa_qr2, best_qr.fa_fc);
}

int main() {
    printf("=== Decoder cTFA Calibration v3 (Real Decoder_module) ===\n\n");

    float *real_in = load_float("dump_matlab/frame0_stft_real.bin", 257);
    float *imag_in = load_float("dump_matlab/frame0_stft_imag.bin", 257);
    int32_t *golden_dec = load_int32("dump_matlab/frame0_dec.bin", 1*129);
    if (!real_in || !imag_in || !golden_dec) { printf("ERROR: missing input files\n"); return 1; }

    /* Run encoder+DPRNN once */
    int32_t x_log[257]; log_gen_fixed(real_in, imag_in, 257, x_log);
    int32_t x_bm[129];  BM_fixed(x_log, erb_erb_fc_weight, x_bm);

    ulunas_state_t st; ulunas_state_init(&st);
    int32_t e0[12*65], e1[24*33], e2[24*33], e3[32*33], e4[16*33];
    Encoder_module(x_bm, &st, e0, e1, e2, e3, e4);

    int32_t r1[16*33], r2[16*33];
    GDPRNN_module(e4, st.inter_prev0, 0, r1);
    GDPRNN_module(r1, st.inter_prev1, 1, r2);
    printf("Encoder+DPRNN done.\n\n");

    /* Calibrate backward: d4 → d3 → d2 → d1 → d0 */
    printf("Backward calibration (d4→d0):\n\n");
    calibrate_one("d4 De_XConv",  &g_qr_d4, r2, e0, e1, e2, e3, e4, golden_dec);
    calibrate_one("d3 De_XMB1",   &g_qr_d3, r2, e0, e1, e2, e3, e4, golden_dec);
    calibrate_one("d2 De_XDWS1",  &g_qr_d2, r2, e0, e1, e2, e3, e4, golden_dec);
    calibrate_one("d1 De_XMB0",   &g_qr_d1, r2, e0, e1, e2, e3, e4, golden_dec);
    calibrate_one("d0 De_XDWS0",  &g_qr_d0, r2, e0, e1, e2, e3, e4, golden_dec);

    /* Final verification */
    ulunas_state_init(&st);
    int32_t y_dec[1*129];
    Decoder_module(r2, &st, e0, e1, e2, e3, e4, y_dec);
    double final_snr = snr_db(y_dec, golden_dec, 129);
    printf("=== Final decoder SNR: %.2f dB ===\n\n", final_snr);

    printf("=== UPDATE ulunas_modules.c with these values ===\n");
    printf("d0: ta=(%d,%d,%d) fa=(%d,%d,%d)\n", g_qr_d0.ta_qr1, g_qr_d0.ta_qr2, g_qr_d0.ta_fc, g_qr_d0.fa_qr1, g_qr_d0.fa_qr2, g_qr_d0.fa_fc);
    printf("d1: ta=(%d,%d,%d) fa=(%d,%d,%d)\n", g_qr_d1.ta_qr1, g_qr_d1.ta_qr2, g_qr_d1.ta_fc, g_qr_d1.fa_qr1, g_qr_d1.fa_qr2, g_qr_d1.fa_fc);
    printf("d2: ta=(%d,%d,%d) fa=(%d,%d,%d)\n", g_qr_d2.ta_qr1, g_qr_d2.ta_qr2, g_qr_d2.ta_fc, g_qr_d2.fa_qr1, g_qr_d2.fa_qr2, g_qr_d2.fa_fc);
    printf("d3: ta=(%d,%d,%d) fa=(%d,%d,%d)\n", g_qr_d3.ta_qr1, g_qr_d3.ta_qr2, g_qr_d3.ta_fc, g_qr_d3.fa_qr1, g_qr_d3.fa_qr2, g_qr_d3.fa_fc);
    printf("d4: ta=(%d,%d,%d) fa=(%d,%d,%d)\n", g_qr_d4.ta_qr1, g_qr_d4.ta_qr2, g_qr_d4.ta_fc, g_qr_d4.fa_qr1, g_qr_d4.fa_qr2, g_qr_d4.fa_fc);

    free(real_in); free(imag_in); free(golden_dec);
    return 0;
}
