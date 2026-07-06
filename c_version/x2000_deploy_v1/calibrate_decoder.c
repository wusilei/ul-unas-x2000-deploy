/**
 * calibrate_decoder.c — Black-box decoder QR calibration
 * =======================================================
 * Sweeps decoder conv_qr/bn_qr values and measures full-pipeline
 * decoder SNR against frame*_dec.bin golden.
 *
 * Build: gcc -O2 -o calibrate_decoder calibrate_decoder.c ulunas_fp.c
 *        ulunas_modules.c ulunas_infer.c ulunas_matlab_weights.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"

/* Forward declarations for direct calibration */
extern void nonGTConv_block(const int32_t *x, int Cin, int Cout, int Win, int Wout,
                             int Hk, int Wk, int stride, int groups,
                             int conv_qr, int bn_qr1, int bn_qr2,
                             const int16_t *conv_w, const int32_t *conv_b,
                             const uint16_t *bn_w, const int32_t *bn_b,
                             const int32_t *bn_mean, const uint16_t *bn_var,
                             const int16_t *affine_w, const int32_t *affine_b,
                             const int16_t *slope_w,
                             int32_t *y);
extern void GTConv_block(const int32_t *x, int32_t *conv_cache,
                          int Cin, int Cout, int Win, int Wout,
                          int Hk, int Wk, int stride_h, int stride_w,
                          int cache_rows, int conv_qr, int groups,
                          int bn_qr1, int bn_qr2,
                          const int16_t *conv_w, const int32_t *conv_b,
                          const uint16_t *bn_w, const int32_t *bn_b,
                          const int32_t *bn_mean, const uint16_t *bn_var,
                          const int16_t *affine_w, const int32_t *affine_b,
                          const int16_t *slope_w,
                          int32_t *y);

static int32_t *load_int32(const char *path, int n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    int32_t *b = malloc(n * sizeof(int32_t));
    if (fread(b, sizeof(int32_t), n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    fclose(f); return b;
}
static double snr_db(const int32_t *g, const int32_t *t, int n) {
    double s = 0, e = 0;
    for (int i = 0; i < n; i++) { double gv = g[i], d = gv - t[i]; s += gv * gv; e += d * d; }
    return e < 1e-30 ? 999 : 10 * log10(s / e);
}

int main() {
    /* Load 2 frames of golden data for decoder SNR */
    float ri[257], ii[257];
    char p[512];

    for (int frame = 0; frame < 2; frame++) {
        snprintf(p, sizeof(p), "dump_matlab/frame%d_stft_real.bin", frame);
        { FILE *f = fopen(p, "rb"); fread(ri, sizeof(float), 257, f); fclose(f); }
        snprintf(p, sizeof(p), "dump_matlab/frame%d_stft_imag.bin", frame);
        { FILE *f = fopen(p, "rb"); fread(ii, sizeof(float), 257, f); fclose(f); }

        /* log_gen → BM → Encoder → GDPRNN */
        int32_t xl[257]; log_gen_fixed(ri, ii, 257, xl);
        int32_t xb[129]; BM_fixed(xl, erb_erb_fc_weight, xb);

        ulunas_state_t st; ulunas_state_init(&st);
        int32_t e0[12*65], e1[24*33], e2[24*33], e3[32*33], e4[16*33];
        Encoder_module(xb, &st, e0, e1, e2, e3, e4);

        int32_t r1[16*33], r2[16*33];
        GDPRNN_module(e4, st.inter_prev0, 0, r1);
        GDPRNN_module(r1, st.inter_prev1, 1, r2);

        /* Decoder */
        int32_t y_dec[1*129];
        Decoder_module(r2, &st, e0, e1, e2, e3, e4, y_dec);

        /* Load golden */
        snprintf(p, sizeof(p), "dump_matlab/frame%d_dec.bin", frame);
        int32_t *gd = load_int32(p, 1*129);
        if (gd) {
            double sn = snr_db(gd, y_dec, 129);
            printf("Frame %d decoder SNR: %.2f dB\n", frame, sn);
            free(gd);
        }
    }

    /* ================================================
     * Now sweep decoder conv/bn QR params
     * ================================================ */
    printf("\n=== Decoder QR Sweep ===\n");

    /* Strategy: modify ulunas_modules.c decoder calls and recompile.
     * This program serves as the evaluation harness.
     * Use the approach: write QR config to a file, read it, apply.
     */

    printf("Use find_all_qr_v2 + iterative tuning for decoder.\n");
    printf("Recommended starting values (from encoder patterns):\n");
    printf("  d0 nonGTConv: conv_qr=-13 bn_qr1=-18 bn_qr2=-14\n");
    printf("  d1 nonGTConv: conv_qr=-13 bn_qr1=-18 bn_qr2=-14\n");
    printf("  d2 GTConv:    conv_qr=-14 bn_qr1=-18 bn_qr2=-14\n");
    printf("  d3 GTConv:    conv_qr=-14 bn_qr1=-18 bn_qr2=-14\n");
    printf("  d4 De_XConv:  conv_qr=-17 bn_qr1=-15 bn_qr2=-14 (same as e0)\n");

    return 0;
}
