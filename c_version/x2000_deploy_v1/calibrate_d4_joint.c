/**
 * calibrate_d4_joint.c — d4 TConv + cTFA Joint Optimizer
 * =======================================================
 * Sweeps d4 TConv (conv_qr, bn_qr1, bn_qr2) + d4 cTFA (6 params)
 * jointly, measuring FINAL decoder SNR against frame0_dec.bin.
 *
 * Phase 1: Coarse grid over TConv + d4 cTFA → top candidates
 * Phase 2: Full v3 backward pass (d4→d0) for each top candidate
 *
 * Build:
 *   gcc -O2 -std=c99 -DQR_CALIBRATION_MODE -DJOINT_CALIBRATION_MODE \
 *       -o calibrate_d4_joint calibrate_d4_joint.c \
 *       ulunas_fp.c ulunas_modules.c ulunas_infer.c ulunas_matlab_weights.c -lm
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

/* ================================================================
 * Phase 1: Coarse d4 TConv + d4 cTFA joint sweep
 * ================================================================
 * Evaluates 125 TConv combos × 729 d4 cTFA combos = 91k evals
 * Each eval: Decoder_module call (~fast since only d4 differ)
 */
typedef struct {
    int conv_qr, bn_qr1, bn_qr2;
    int ta1, ta2, tfc, fa1, fa2, ffc;
    double snr;
} candidate_t;

static int cmp_cand(const void *a, const void *b) {
    double d = ((candidate_t*)b)->snr - ((candidate_t*)a)->snr;
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}

int main() {
    printf("=== d4 TConv + cTFA Joint Optimization ===\n\n");

    /* Load inputs */
    float *ri = load_float("dump_matlab/frame0_stft_real.bin", 257);
    float *ii = load_float("dump_matlab/frame0_stft_imag.bin", 257);
    int32_t *gd = load_int32("dump_matlab/frame0_dec.bin", 1*129);
    if (!ri || !ii || !gd) { printf("ERROR: missing input files\n"); return 1; }

    /* Pre-compute encoder + DPRNN (same for all evals) */
    int32_t xl[257]; log_gen_fixed(ri, ii, 257, xl);
    int32_t xb[129];  BM_fixed(xl, erb_erb_fc_weight, xb);

    ulunas_state_t st_enc; ulunas_state_init(&st_enc);
    int32_t e0[12*65], e1[24*33], e2[24*33], e3[32*33], e4[16*33];
    Encoder_module(xb, &st_enc, e0, e1, e2, e3, e4);

    int32_t r1[16*33], r2[16*33];
    GDPRNN_module(e4, st_enc.inter_prev0, 0, r1);
    GDPRNN_module(r1, st_enc.inter_prev1, 1, r2);
    printf("Encoder+DPRNN pre-computed.\n");

    /* Baseline */
    {
        ulunas_state_t st; ulunas_state_init(&st);
        int32_t yd[129];
        Decoder_module(r2, &st, e0, e1, e2, e3, e4, yd);
        printf("Baseline dec SNR = %.2f dB\n\n", snr_db(yd, gd, 129));
    }

    /* Phase 1: Coarse joint sweep */
    printf("Phase 1: Coarse TConv + d4 cTFA sweep...\n");

    /* TConv ranges: step=2 for coarse */
    int tconv_cq[] = {-10, -12, -14, -16, -18};
    int tconv_b1[] = {-8, -11, -14, -17, -20};
    int tconv_b2[] = {-8, -11, -14, -17, -20};
    int n_tconv = 5 * 5 * 5;  /* 125 */

    /* cTFA ranges: coarse step=2 */
    int ta1s[] = {-16, -12, -8};  /* 3 */
    int ta2s[] = {-14, -10, -6, -2};  /* 4 */
    int tfcs[] = {-8, -4, -1};   /* 3 */
    int fa1s[] = {-16, -12, -8, -4};  /* 4 */
    int fa2s[] = {-14, -10, -6, -2};  /* 4 */
    int ffcs[] = {-8, -4, -1};   /* 3 */
    int n_ctfa = 3 * 4 * 3 * 4 * 4 * 3;  /* 1728 */

    int n_total = n_tconv * n_ctfa;  /* 216,000 */
    candidate_t *candidates = malloc(n_total * sizeof(candidate_t));
    int ci = 0;

    /* Store original d0-d3 cTFA QRs (fixed during phase 1) */
    ctfa_qr_t save_d0 = g_qr_d0, save_d1 = g_qr_d1;
    ctfa_qr_t save_d2 = g_qr_d2, save_d3 = g_qr_d3;

    for (int icq = 0; icq < 5; icq++) {
    for (int ib1 = 0; ib1 < 5; ib1++) {
    for (int ib2 = 0; ib2 < 5; ib2++) {
        g_d4_tconv.conv_qr = tconv_cq[icq];
        g_d4_tconv.bn_qr1  = tconv_b1[ib1];
        g_d4_tconv.bn_qr2  = tconv_b2[ib2];

        for (int i1 = 0; i1 < 3; i1++)
        for (int i2 = 0; i2 < 4; i2++)
        for (int i3 = 0; i3 < 3; i3++)
        for (int j1 = 0; j1 < 4; j1++)
        for (int j2 = 0; j2 < 4; j2++)
        for (int j3 = 0; j3 < 3; j3++) {
            g_qr_d4.ta_qr1 = ta1s[i1]; g_qr_d4.ta_qr2 = ta2s[i2]; g_qr_d4.ta_fc = tfcs[i3];
            g_qr_d4.fa_qr1 = fa1s[j1]; g_qr_d4.fa_qr2 = fa2s[j2]; g_qr_d4.fa_fc = ffcs[j3];

            ulunas_state_t st; ulunas_state_init(&st);
            int32_t yd[129];
            Decoder_module(r2, &st, e0, e1, e2, e3, e4, yd);
            double s = snr_db(yd, gd, 129);

            candidates[ci++] = (candidate_t){
                tconv_cq[icq], tconv_b1[ib1], tconv_b2[ib2],
                ta1s[i1], ta2s[i2], tfcs[i3],
                fa1s[j1], fa2s[j2], ffcs[j3], s
            };
        }
    }}}

    /* Sort by SNR descending */
    qsort(candidates, ci, sizeof(candidate_t), cmp_cand);

    printf("Top 10 coarse candidates:\n");
    for (int i = 0; i < 10 && i < ci; i++) {
        candidate_t *c = &candidates[i];
        printf("  #%d: SNR=%.2f  TConv(%d,%d,%d) cTFA(%d,%d,%d | %d,%d,%d)\n",
               i+1, c->snr,
               c->conv_qr, c->bn_qr1, c->bn_qr2,
               c->ta1, c->ta2, c->tfc, c->fa1, c->fa2, c->ffc);
    }

    /* Phase 2: Full backward pass for top candidates */
    printf("\nPhase 2: Full v3 backward pass for top 5...\n");
    for (int top = 0; top < 5 && top < ci; top++) {
        candidate_t *c = &candidates[top];
        g_d4_tconv.conv_qr = c->conv_qr;
        g_d4_tconv.bn_qr1  = c->bn_qr1;
        g_d4_tconv.bn_qr2  = c->bn_qr2;

        /* Restore d0-d3 to known good, set d4 to candidate */
        g_qr_d0 = save_d0; g_qr_d1 = save_d1;
        g_qr_d2 = save_d2; g_qr_d3 = save_d3;
        g_qr_d4.ta_qr1 = c->ta1; g_qr_d4.ta_qr2 = c->ta2; g_qr_d4.ta_fc = c->tfc;
        g_qr_d4.fa_qr1 = c->fa1; g_qr_d4.fa_qr2 = c->fa2; g_qr_d4.fa_fc = c->ffc;

        printf("\n--- Top #%d: TConv(%d,%d,%d) base cTFA(%d,%d,%d | %d,%d,%d) ---\n",
               top+1, c->conv_qr, c->bn_qr1, c->bn_qr2,
               c->ta1, c->ta2, c->tfc, c->fa1, c->fa2, c->ffc);

        /* Re-calibrate d4 cTFA with fine grid */
        double best_d4 = c->snr;
        ctfa_qr_t best_qr4 = g_qr_d4;
        for (int ta1 = c->ta1-2; ta1 <= c->ta1+2; ta1++)
        for (int ta2 = c->ta2-2; ta2 <= c->ta2+2; ta2++)
        for (int tfc = c->tfc-2; tfc <= c->tfc+2; tfc++)
        for (int fa1 = c->fa1-2; fa1 <= c->fa1+2; fa1++)
        for (int fa2 = c->fa2-2; fa2 <= c->fa2+2; fa2++)
        for (int ffc = c->ffc-2; ffc <= c->ffc+2; ffc++) {
            g_qr_d4 = (ctfa_qr_t){ta1, ta2, tfc, fa1, fa2, ffc};
            ulunas_state_t st; ulunas_state_init(&st);
            int32_t yd[129];
            Decoder_module(r2, &st, e0, e1, e2, e3, e4, yd);
            double s = snr_db(yd, gd, 129);
            if (s > best_d4) { best_d4 = s; best_qr4 = g_qr_d4; }
        }
        g_qr_d4 = best_qr4;
        printf("  After d4 refine: SNR=%.2f  cTFA(%d,%d,%d | %d,%d,%d)\n",
               best_d4, best_qr4.ta_qr1, best_qr4.ta_qr2, best_qr4.ta_fc,
               best_qr4.fa_qr1, best_qr4.fa_qr2, best_qr4.fa_fc);

        /* Now re-calibrate d3→d0 backward */
        ctfa_qr_t *qrs[] = {&g_qr_d3, &g_qr_d2, &g_qr_d1, &g_qr_d0};
        const char *names[] = {"d3","d2","d1","d0"};
        for (int m = 0; m < 4; m++) {
            ctfa_qr_t *qr = qrs[m];
            ctfa_qr_t orig = *qr;
            double best_m = -999;
            ctfa_qr_t best_m_qr = *qr;

            for (int ta1 = orig.ta_qr1-1; ta1 <= orig.ta_qr1+1; ta1++)
            for (int ta2 = orig.ta_qr2-1; ta2 <= orig.ta_qr2+1; ta2++)
            for (int tfc = orig.ta_fc-1; tfc <= orig.ta_fc+1; tfc++)
            for (int fa1 = orig.fa_qr1-1; fa1 <= orig.fa_qr1+1; fa1++)
            for (int fa2 = orig.fa_qr2-1; fa2 <= orig.fa_qr2+1; fa2++)
            for (int ffc = orig.fa_fc-1; ffc <= orig.fa_fc+1; ffc++) {
                *qr = (ctfa_qr_t){ta1, ta2, tfc, fa1, fa2, ffc};
                ulunas_state_t st; ulunas_state_init(&st);
                int32_t yd[129];
                Decoder_module(r2, &st, e0, e1, e2, e3, e4, yd);
                double s = snr_db(yd, gd, 129);
                if (s > best_m) { best_m = s; best_m_qr = *qr; }
            }
            *qr = best_m_qr;
            printf("  %s refined: SNR=%.2f  cTFA(%d,%d,%d | %d,%d,%d)\n",
                   names[m], best_m, best_m_qr.ta_qr1, best_m_qr.ta_qr2, best_m_qr.ta_fc,
                   best_m_qr.fa_qr1, best_m_qr.fa_qr2, best_m_qr.fa_fc);
        }

        /* Final SNR for this TConv combo */
        ulunas_state_t st; ulunas_state_init(&st);
        int32_t yd[129];
        Decoder_module(r2, &st, e0, e1, e2, e3, e4, yd);
        double final_snr = snr_db(yd, gd, 129);
        printf("  >>> FINAL for TConv(%d,%d,%d): %.2f dB <<<\n",
               g_d4_tconv.conv_qr, g_d4_tconv.bn_qr1, g_d4_tconv.bn_qr2, final_snr);
    }

    printf("\n=== DONE ===\n");
    free(candidates); free(ri); free(ii); free(gd);
    return 0;
}
