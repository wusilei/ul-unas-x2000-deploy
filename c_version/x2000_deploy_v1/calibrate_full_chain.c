/**
 * calibrate_full_chain.c — Zero-Gap Full-Chain Calibrator
 * ========================================================
 * Runs the COMPLETE pipeline (STM→BM→Encoder→DPRNN→Decoder→Sigmoid→BS→MASK)
 * and optimizes final MASK SNR directly. Uses the EXACT same code path
 * as ulunas_infer.c — zero measurement bias by construction.
 *
 * Eliminates:
 *  - intermediate golden input dependency
 *  - calibrator-vs-test code path divergence
 *  - manual QR transfer copy errors
 *  - frame-0-only optimization bias
 *
 * Build:
 *   gcc -O2 -std=c99 -DQR_CALIBRATION_MODE -DJOINT_CALIBRATION_MODE \
 *       -o calibrate_full_chain calibrate_full_chain.c \
 *       ulunas_fp.c ulunas_modules.c ulunas_infer.c ulunas_matlab_weights.c -lm
 *
 * Usage:
 *   ./calibrate_full_chain [module] [--mask]
 *   module: all | encoder | decoder | d4 (default: decoder)
 *   --mask: optimize MASK SNR instead of decoder SNR
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

/* Run full chain: STFT → log_gen → BM → Encoder → DPRNN → Decoder → Sigmoid → BS → MASK
 * Returns MASK SNR (or decoder SNR if dec_only).
 * dec_snr_floor: if use_mask_target and dec SNR drops below this, return -999 */
static double run_full_chain(const float *real_in, const float *imag_in,
                              const int32_t *golden_mask, const int32_t *golden_dec,
                              int use_mask_target, double dec_snr_floor,
                              double *dec_snr_out) {
    int32_t x_log[257]; log_gen_fixed(real_in, imag_in, 257, x_log);
    int32_t x_bm[129];  BM_fixed(x_log, erb_erb_fc_weight, x_bm);

    ulunas_state_t st; ulunas_state_init(&st);
    int32_t e0[12*65], e1[24*33], e2[24*33], e3[32*33], e4[16*33];
    Encoder_module(x_bm, &st, e0, e1, e2, e3, e4);

    int32_t r1[16*33], r2[16*33];
    GDPRNN_module(e4, st.inter_prev0, 0, r1);
    GDPRNN_module(r1, st.inter_prev1, 1, r2);

    int32_t y_dec[1*129];
    Decoder_module(r2, &st, e0, e1, e2, e3, e4, y_dec);

    double dec_snr = snr_db(golden_dec, y_dec, 129);  /* golden as signal reference */
    if (dec_snr_out) *dec_snr_out = dec_snr;

    if (!use_mask_target) return dec_snr;

    /* Reject candidates that collapse decoder beyond floor */
    if (dec_snr < dec_snr_floor) return -999.0;

    /* Full MASK chain */
    uint16_t y_sig[1*N_BINS_BM];
    sigmoid_fixed(y_dec, 1*N_BINS_BM, y_sig);
    int16_t y_sig_s16[1*N_BINS_BM];
    for (int i = 0; i < 1*N_BINS_BM; i++) y_sig_s16[i] = (int16_t)y_sig[i];

    int16_t y_bs[1*N_BINS];
    BS_fixed(y_sig_s16, erb_ierb_fc_weight, y_bs);

    /* Compute spec from STFT (same as ulunas_infer) */
    int32_t real_q[257], imag_q[257];
    for (int i = 0; i < 257; i++) {
        real_q[i] = F2Q20(real_in[i]);
        imag_q[i] = F2Q20(imag_in[i]);
    }

    int32_t crm_out[2*257];
    MASK_fixed(y_bs, real_q, imag_q, crm_out);

    return snr_db(golden_mask, crm_out, 2*257);  /* golden as signal reference */
}

/* Multi-frame evaluator helper */
static double eval_multi(const float **ri, const float **ii,
                          const int32_t **gm, const int32_t **gd,
                          int n_frames, int use_mask, double dec_snr_floor,
                          double *dec_out) {
    double sum = 0, dsum = 0;
    for (int f = 0; f < n_frames; f++) {
        double d;
        sum += run_full_chain(ri[f], ii[f], gm[f], gd[f], use_mask, dec_snr_floor, &d);
        dsum += d;
    }
    if (dec_out) *dec_out = dsum / n_frames;
    return sum / n_frames;
}

/* Coarse+fine grid search over 6 cTFA QR params, multi-frame aware */
static double search_ctfa_6d_multi(ctfa_qr_t *qr,
                                    const float **ri, const float **ii,
                                    const int32_t **gm, const int32_t **gd,
                                    int n_frames, int use_mask, double dec_snr_floor,
                                    double *base_out) {
    double base = eval_multi(ri, ii, gm, gd, n_frames, use_mask, dec_snr_floor, NULL);
    if (base_out) *base_out = base;

    double best = base;
    ctfa_qr_t best_qr = *qr;

    /* Coarse: step=2 */
    for (int ta1 = -20; ta1 <= -2; ta1 += 2)
    for (int ta2 = -14; ta2 <= -2; ta2 += 2)
    for (int tfc = -12; tfc <= -2; tfc += 2)
    for (int fa1 = -20; fa1 <= -2; fa1 += 2)
    for (int fa2 = -14; fa2 <= -2; fa2 += 2)
    for (int ffc = -12; ffc <= -2; ffc += 2) {
        *qr = (ctfa_qr_t){ta1, ta2, tfc, fa1, fa2, ffc};
        double s = eval_multi(ri, ii, gm, gd, n_frames, use_mask, dec_snr_floor, NULL);
        if (s > best) { best = s; best_qr = *qr; }
    }

    /* Fine: step=1 around best */
    for (int ta1 = best_qr.ta_qr1-1; ta1 <= best_qr.ta_qr1+1; ta1++)
    for (int ta2 = best_qr.ta_qr2-1; ta2 <= best_qr.ta_qr2+1; ta2++)
    for (int tfc = best_qr.ta_fc-1; tfc <= best_qr.ta_fc+1; tfc++)
    for (int fa1 = best_qr.fa_qr1-1; fa1 <= best_qr.fa_qr1+1; fa1++)
    for (int fa2 = best_qr.fa_qr2-1; fa2 <= best_qr.fa_qr2+1; fa2++)
    for (int ffc = best_qr.fa_fc-1; ffc <= best_qr.fa_fc+1; ffc++) {
        *qr = (ctfa_qr_t){ta1, ta2, tfc, fa1, fa2, ffc};
        double s = eval_multi(ri, ii, gm, gd, n_frames, use_mask, dec_snr_floor, NULL);
        if (s > best) { best = s; best_qr = *qr; }
    }

    *qr = best_qr;
    return best;
}

/* Print QR in copy-paste ready format for ulunas_ctfa_qr.h */
static void print_qr_macro(const char *name, ctfa_qr_t *qr, double snr, double base, double dec_snr) {
    printf("#define %-4s_TA %3d, %3d, %3d\n", name, qr->ta_qr1, qr->ta_qr2, qr->ta_fc);
    printf("#define %-4s_FA %3d, %3d, %3d\n", name, qr->fa_qr1, qr->fa_qr2, qr->fa_fc);
    printf("  SNR: %.2f dB (Δ=%+.2f)  dec=%.2f\n\n", snr, snr - base, dec_snr);
}

int main(int argc, char **argv) {
    int use_mask = 0, n_frames = 1;
    double floor_delta = 3.0;
    const char *mode = "decoder";

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mask")) use_mask = 1;
        else if (!strncmp(argv[i], "--floor=", 8)) floor_delta = atof(argv[i]+8);
        else if (!strncmp(argv[i], "--frames=", 9)) n_frames = atoi(argv[i]+9);
        else mode = argv[i];
    }
    if (n_frames < 1) n_frames = 1; if (n_frames > 5) n_frames = 5;

    printf("=== Full-Chain Calibrator (zero gap, %d frames) ===\n", n_frames);
    printf("Target: %s SNR, Mode: %s", use_mask ? "MASK" : "decoder", mode);
    if (use_mask) printf(", floor=baseline-%.1fdB", floor_delta);
    printf("\n\n");

    /* Load all frames */
    float *ri[5], *ii[5]; int32_t *gd[5], *gm[5];
    for (int f = 0; f < n_frames; f++) {
        char p[256];
        snprintf(p, sizeof(p), "dump_matlab/frame%d_stft_real.bin", f); ri[f] = load_float(p, 257);
        snprintf(p, sizeof(p), "dump_matlab/frame%d_stft_imag.bin", f); ii[f] = load_float(p, 257);
        snprintf(p, sizeof(p), "dump_matlab/frame%d_dec.bin", f);      gd[f] = load_int32(p, 1*129);
        snprintf(p, sizeof(p), "dump_matlab/frame%d_mask.bin", f);     gm[f] = load_int32(p, 2*257);
        if (!ri[f] || !ii[f] || !gd[f] || !gm[f]) { printf("ERROR: missing frame %d\n", f); return 1; }
    }

    /* Baseline: average over all frames */
    double sum_snr = 0, sum_dec = 0;
    for (int f = 0; f < n_frames; f++) {
        double d;
        sum_snr += run_full_chain(ri[f], ii[f], gm[f], gd[f], use_mask, -999.0, &d);
        sum_dec += d;
    }
    double base_snr = sum_snr / n_frames;
    double dec_snr_base = sum_dec / n_frames;
    printf("Baseline (%d frames): %s=%.2f dB  dec=%.2f dB\n\n",
           n_frames, use_mask ? "mask" : "dec", base_snr, dec_snr_base);

    /* MASK-targeted: allow configurable decoder degradation */
    double dec_snr_floor = use_mask ? dec_snr_base - floor_delta : -999.0;
    if (use_mask) printf("Decoder SNR floor: %.2f dB (Δ=%.1f)\n\n", dec_snr_floor, floor_delta);

    /* Build frame pointer arrays for multi-frame eval */
    const float  *rip[5], *iip[5];
    const int32_t *gmp[5], *gdp[5];
    for (int f = 0; f < n_frames; f++) { rip[f] = ri[f]; iip[f] = ii[f]; gmp[f] = gm[f]; gdp[f] = gd[f]; }

    /* ================================================================
     * Decoder cTFA calibration (backward: d4→d0)
     * ================================================================ */
    if (!strcmp(mode, "decoder") || !strcmp(mode, "all")) {
        printf("=== Decoder cTFA (d4→d0 backward) ===\n\n");

        ctfa_qr_t *dec_qrs[] = {&g_qr_d4, &g_qr_d3, &g_qr_d2, &g_qr_d1, &g_qr_d0};
        const char *dec_names[] = {"D4","D3","D2","D1","D0"};

        int n_passes = use_mask ? 3 : 1;  /* mask: multi-pass tightening */
        for (int pass = 0; pass < n_passes; pass++) {
            double floor = dec_snr_floor;
            if (use_mask && pass > 0) {
                /* Tighten floor after each pass */
                double cur_dec = eval_multi(rip, iip, gmp, gdp, n_frames, 0, -999.0, NULL);
                floor = cur_dec - (floor_delta - pass * 1.0);
                if (floor < dec_snr_base - 1.0) floor = dec_snr_base - 1.0;
                printf("\n--- Pass %d: floor=%.1f dB ---\n\n", pass + 1, floor);
            }

            for (int m = 0; m < 5; m++) {
                double base;
                double best = search_ctfa_6d_multi(dec_qrs[m],
                    rip, iip, gmp, gdp, n_frames, use_mask, floor, &base);
                double d;
                eval_multi(rip, iip, gmp, gdp, n_frames, use_mask, floor, &d);
                print_qr_macro(dec_names[m], dec_qrs[m], best, base, d);
            }
        }
    }

    /* ================================================================
     * d4 TConv QR calibration (joint with d4 cTFA)
     * ================================================================ */
    if (!strcmp(mode, "d4") || !strcmp(mode, "all")) {
        printf("=== d4 TConv QR (conv, bn1, bn2) ===\n\n");

        int tconv_cq[] = {-10,-12,-14,-16,-18};
        int tconv_b1[] = {-8,-11,-14,-17,-20};
        int tconv_b2[] = {-8,-11,-14,-17,-20};

        double best_tc = base_snr;
        int best_cq = g_d4_tconv.conv_qr, best_b1 = g_d4_tconv.bn_qr1, best_b2 = g_d4_tconv.bn_qr2;

        for (int ic = 0; ic < 5; ic++)
        for (int ib1 = 0; ib1 < 5; ib1++)
        for (int ib2 = 0; ib2 < 5; ib2++) {
            g_d4_tconv.conv_qr = tconv_cq[ic];
            g_d4_tconv.bn_qr1  = tconv_b1[ib1];
            g_d4_tconv.bn_qr2  = tconv_b2[ib2];
            double s = eval_multi(rip, iip, gmp, gdp, n_frames, use_mask, dec_snr_floor, NULL);
            if (s > best_tc) {
                best_tc = s;
                best_cq = tconv_cq[ic]; best_b1 = tconv_b1[ib1]; best_b2 = tconv_b2[ib2];
            }
        }

        g_d4_tconv.conv_qr = best_cq; g_d4_tconv.bn_qr1 = best_b1; g_d4_tconv.bn_qr2 = best_b2;
        double d;
        double s = eval_multi(rip, iip, gmp, gdp, n_frames, use_mask, dec_snr_floor, &d);
        printf("Best d4 TConv: conv=%d bn=(%d,%d)  %s=%.2f dB  dec=%.2f\n\n",
               best_cq, best_b1, best_b2, use_mask ? "mask" : "dec", s, d);
    }

    /* ================================================================
     * Print final QR values for copy-paste into ulunas_ctfa_qr.h
     * ================================================================ */
    printf("=== Final QR Values (copy to ulunas_ctfa_qr.h #else branch) ===\n");
    printf("#define D0_TA %d, %d, %d\n", g_qr_d0.ta_qr1, g_qr_d0.ta_qr2, g_qr_d0.ta_fc);
    printf("#define D0_FA %d, %d, %d\n", g_qr_d0.fa_qr1, g_qr_d0.fa_qr2, g_qr_d0.fa_fc);
    printf("#define D1_TA %d, %d, %d\n", g_qr_d1.ta_qr1, g_qr_d1.ta_qr2, g_qr_d1.ta_fc);
    printf("#define D1_FA %d, %d, %d\n", g_qr_d1.fa_qr1, g_qr_d1.fa_qr2, g_qr_d1.fa_fc);
    printf("#define D2_TA %d, %d, %d\n", g_qr_d2.ta_qr1, g_qr_d2.ta_qr2, g_qr_d2.ta_fc);
    printf("#define D2_FA %d, %d, %d\n", g_qr_d2.fa_qr1, g_qr_d2.fa_qr2, g_qr_d2.fa_fc);
    printf("#define D3_TA %d, %d, %d\n", g_qr_d3.ta_qr1, g_qr_d3.ta_qr2, g_qr_d3.ta_fc);
    printf("#define D3_FA %d, %d, %d\n", g_qr_d3.fa_qr1, g_qr_d3.fa_qr2, g_qr_d3.fa_fc);
    printf("#define D4_TA %d, %d, %d\n", g_qr_d4.ta_qr1, g_qr_d4.ta_qr2, g_qr_d4.ta_fc);
    printf("#define D4_FA %d, %d, %d\n", g_qr_d4.fa_qr1, g_qr_d4.fa_qr2, g_qr_d4.fa_fc);
    printf("#define D4_TCONV_CQR %d\n", g_d4_tconv.conv_qr);
    printf("#define D4_TCONV_BN1 %d\n", g_d4_tconv.bn_qr1);
    printf("#define D4_TCONV_BN2 %d\n", g_d4_tconv.bn_qr2);

    double final_snr = eval_multi(rip, iip, gmp, gdp, n_frames, use_mask, dec_snr_floor, &dec_snr_base);
    printf("\nFinal (%d frames): %s=%.2f dB  dec=%.2f dB\n",
           n_frames, use_mask ? "mask" : "dec", final_snr, dec_snr_base);

    for (int f = 0; f < n_frames; f++) { free(ri[f]); free(ii[f]); free(gd[f]); free(gm[f]); }
    return 0;
}
