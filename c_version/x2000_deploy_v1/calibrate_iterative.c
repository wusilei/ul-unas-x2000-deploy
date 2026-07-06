/**
 * calibrate_iterative.c — MASK-Targeted Iterative Encoder+Decoder Calibration
 * ===========================================================================
 * Optimizes all 10 encoder+decoder cTFA modules against MASK SNR
 * (not decoder SNR), using alternating forward/backward passes.
 *
 * Iterates until convergence (Δ < 0.05 dB).
 * QR ranges constrained to [-18,-4] to avoid extreme values.
 *
 * Build:
 *   gcc -O2 -std=c99 -DQR_CALIBRATION_MODE -o calibrate_iterative \
 *       calibrate_iterative.c ulunas_fp.c ulunas_modules.c \
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
static int32_t *L(const char *p, int n) { FILE *f = fopen(p,"rb"); if(!f) return NULL;
    int32_t *b = malloc(n*4); if(fread(b,4,n,f)!=(size_t)n){free(b);fclose(f);return NULL;} fclose(f); return b; }
static float *LF(const char *p, int n) { FILE *f = fopen(p,"rb"); if(!f) return NULL;
    float *b = malloc(n*4); if(fread(b,4,n,f)!=(size_t)n){free(b);fclose(f);return NULL;} fclose(f); return b; }

/* Run full pipeline and return MASK SNR */
static double measure_mask_snr(const float *ri, const float *ii,
                                const int32_t *gmask, const uint16_t *erb_w, const uint16_t *ierb_w) {
    int32_t xl[257]; log_gen_fixed(ri, ii, 257, xl);
    int32_t xb[129]; BM_fixed(xl, erb_w, xb);

    ulunas_state_t st; ulunas_state_init(&st);
    int32_t e0[12*65], e1[24*33], e2[24*33], e3[32*33], e4[16*33];
    Encoder_module(xb, &st, e0, e1, e2, e3, e4);

    int32_t r1[16*33], r2[16*33];
    GDPRNN_module(e4, st.inter_prev0, 0, r1);
    GDPRNN_module(r1, st.inter_prev1, 1, r2);

    int32_t yd[129];
    Decoder_module(r2, &st, e0, e1, e2, e3, e4, yd);

    /* sigmoid → BS → MASK */
    uint16_t ys[129]; sigmoid_fixed(yd, 129, ys);
    int16_t ys_s16[129]; for(int i=0;i<129;i++) ys_s16[i]=(int16_t)ys[i];
    int16_t yb[257]; BS_fixed(ys_s16, ierb_w, yb);

    int32_t rq[257], iq[257];
    for(int i=0;i<257;i++){rq[i]=F2Q20(ri[i]); iq[i]=F2Q20(ii[i]);}
    int32_t crm[514]; MASK_fixed(yb, rq, iq, crm);

    return snr_db(crm, gmask, 514);
}

/* Calibrate one module: sweep 3^6=729 coarse + 3^6 fine */
static double calibrate_one(ctfa_qr_t *qr, const float *ri, const float *ii,
                             const int32_t *gmask, const uint16_t *erb_w, const uint16_t *ierb_w,
                             const char *name) {
    ctfa_qr_t orig = *qr;
    double base = measure_mask_snr(ri, ii, gmask, erb_w, ierb_w);
    printf("  %-12s base=%.2f  (%d,%d,%d | %d,%d,%d)\n",
           name, base, qr->ta_qr1, qr->ta_qr2, qr->ta_fc, qr->fa_qr1, qr->fa_qr2, qr->fa_fc);

    /* Coarse grid: step=2 */
    int vals[] = {-18, -14, -10, -6, -4};
    int nv = 5;
    double best = base;
    ctfa_qr_t best_qr = *qr;

    for (int i1=0;i1<nv;i1++) for (int i2=0;i2<nv;i2++) for (int i3=0;i3<nv;i3++)
    for (int j1=0;j1<nv;j1++) for (int j2=0;j2<nv;j2++) for (int j3=0;j3<nv;j3++) {
        *qr = (ctfa_qr_t){vals[i1],vals[i2],vals[i3], vals[j1],vals[j2],vals[j3]};
        double s = measure_mask_snr(ri, ii, gmask, erb_w, ierb_w);
        if (s > best) { best = s; best_qr = *qr; }
    }

    /* Fine: step=1 around best */
    for (int i1=best_qr.ta_qr1-1; i1<=best_qr.ta_qr1+1; i1++)
    for (int i2=best_qr.ta_qr2-1; i2<=best_qr.ta_qr2+1; i2++)
    for (int i3=best_qr.ta_fc-1; i3<=best_qr.ta_fc+1; i3++)
    for (int j1=best_qr.fa_qr1-1; j1<=best_qr.fa_qr1+1; j1++)
    for (int j2=best_qr.fa_qr2-1; j2<=best_qr.fa_qr2+1; j2++)
    for (int j3=best_qr.fa_fc-1; j3<=best_qr.fa_fc+1; j3++) {
        *qr = (ctfa_qr_t){i1,i2,i3, j1,j2,j3};
        double s = measure_mask_snr(ri, ii, gmask, erb_w, ierb_w);
        if (s > best) { best = s; best_qr = *qr; }
    }

    *qr = best_qr;
    printf("  %-12s best=%.2f Δ=%+.2f  (%d,%d,%d | %d,%d,%d)\n",
           name, best, best-base,
           best_qr.ta_qr1,best_qr.ta_qr2,best_qr.ta_fc,
           best_qr.fa_qr1,best_qr.fa_qr2,best_qr.fa_fc);
    return best;
}

int main() {
    printf("=== Iterative MASK-SNR Calibration ===\n\n");

    float *ri = LF("dump_matlab/frame0_stft_real.bin", 257);
    float *ii = LF("dump_matlab/frame0_stft_imag.bin", 257);
    int32_t *gm = L("dump_matlab/frame0_mask.bin", 514);
    if (!ri || !ii || !gm) { printf("ERROR: missing files\n"); return 1; }

    ctfa_qr_t *enc_qrs[] = {&g_qr_e0, &g_qr_e1, &g_qr_e2, &g_qr_e3, &g_qr_e4};
    ctfa_qr_t *dec_qrs[] = {&g_qr_d4, &g_qr_d3, &g_qr_d2, &g_qr_d1, &g_qr_d0};
    const char *en[] = {"e0","e1","e2","e3","e4"};
    const char *dn[] = {"d4","d3","d2","d1","d0"};

    double prev = measure_mask_snr(ri, ii, gm, erb_erb_fc_weight, erb_ierb_fc_weight);
    printf("Initial MASK SNR = %.2f dB\n\n", prev);

    for (int iter = 1; iter <= 5; iter++) {
        printf("=== Iteration %d ===\n", iter);

        /* Encoder pass (forward: e0→e4) */
        printf("-- Encoder (forward) --\n");
        for (int m = 0; m < 5; m++)
            calibrate_one(enc_qrs[m], ri, ii, gm, erb_erb_fc_weight, erb_ierb_fc_weight, en[m]);

        /* Decoder pass (backward: d4→d0) */
        printf("-- Decoder (backward) --\n");
        for (int m = 0; m < 5; m++)
            calibrate_one(dec_qrs[m], ri, ii, gm, erb_erb_fc_weight, erb_ierb_fc_weight, dn[m]);

        double curr = measure_mask_snr(ri, ii, gm, erb_erb_fc_weight, erb_ierb_fc_weight);
        printf("Iter %d final MASK SNR = %.2f dB (Δ=%.3f)\n\n", iter, curr, curr - prev);
        if (curr - prev < 0.05) { printf("Converged.\n"); break; }
        prev = curr;
    }

    printf("=== Final QRs ===\n");
    printf("Encoder:\n");
    for (int m=0;m<5;m++) printf("  %s: ta=(%d,%d,%d) fa=(%d,%d,%d)\n", en[m],
        enc_qrs[m]->ta_qr1,enc_qrs[m]->ta_qr2,enc_qrs[m]->ta_fc,
        enc_qrs[m]->fa_qr1,enc_qrs[m]->fa_qr2,enc_qrs[m]->fa_fc);
    printf("Decoder:\n");
    for (int m=0;m<5;m++) printf("  %s: ta=(%d,%d,%d) fa=(%d,%d,%d)\n", dn[m],
        dec_qrs[m]->ta_qr1,dec_qrs[m]->ta_qr2,dec_qrs[m]->ta_fc,
        dec_qrs[m]->fa_qr1,dec_qrs[m]->fa_qr2,dec_qrs[m]->fa_fc);

    free(ri); free(ii); free(gm);
    return 0;
}
