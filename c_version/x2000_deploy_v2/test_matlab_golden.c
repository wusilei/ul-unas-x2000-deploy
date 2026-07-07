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

        /* Initialize state */
        ulunas_state_t st;
        ulunas_state_init(&st);

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

        /* Step 4: GDPRNN */
        int32_t r1[16 * 33], r2[16 * 33];
        gdprnn_module(e4, st.inter_cache_0, 0, r1);
        gdprnn_module(r1, st.inter_cache_1, 1, r2);

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
