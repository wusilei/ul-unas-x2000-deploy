/**
 * bm_test.c — ERB Band Merging (standalone)
 * ==========================================
 * Matches MATLAB BM_module.m exactly:
 *   y(1:65) = x(1:65)                    [passthrough]
 *   y(66:129) = round(x(66:257) * W * 2^(-15))  [matrix multiply]
 *
 * Input:  log_gen output (Q20, 257 bins) — using BM golden bins 0-64 as log_gen proxy
 * Output: 129 bins Q20
 *
 * Build: gcc -O2 -std=c99 -o bm_test bm_test.c -lm
 * Usage: ./bm_test <golden_dir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define HI_IN   192
#define HI_OUT  64
#define LOW_BINS 65
#define N_IN    257
#define N_OUT   129

#include "erb_weights.h"

static int32_t sat_i32(int64_t x) {
    if (x > INT32_MAX) return INT32_MAX;
    if (x < INT32_MIN) return INT32_MIN;
    return (int32_t)x;
}

static void bm_fixed(const int32_t *x, int W_in, int W_out, int32_t *y) {
    memcpy(y, x, LOW_BINS * sizeof(int32_t));
    int64_t r = 16384;  /* 2^14 for >> 15 */
    for (int o = 0; o < HI_OUT; o++) {
        int64_t acc = 0;
        for (int i = 0; i < HI_IN; i++) {
            int64_t prod = (int64_t)x[LOW_BINS + i] * erb_erb_fc_weight[i + HI_IN * o];
            if (prod >= 0) acc += (prod + r) >> 15;
            else           acc += (prod - r) >> 15;
        }
        y[LOW_BINS + o] = sat_i32(acc);
    }
}

static int32_t *load_i32(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    int32_t *buf = malloc(n * sizeof(int32_t));
    if (!buf) { fclose(f); return NULL; }
    fread(buf, sizeof(int32_t), n, f);
    fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : ".";

    /* Use BM golden bins 0-64 as log_gen proxy input.
     * bins 65-256: we need the actual log_gen output.
     * Strategy: read BM golden for passthrough, but we need full 257-bin log_gen.
     * Better: reconstruct log_gen from STFT input. */
    char path[512];

    /* Actually, the cleanest approach: load STFT, run log_gen with float math,
     * then run BM, then compare with BM golden. */
    snprintf(path, sizeof(path), "%s/frame0_stft_real.bin", dir);
    FILE *fr = fopen(path, "rb");
    if (!fr) { fprintf(stderr, "Cannot open %s\n", path); return 1; }
    float *stft_real = malloc(N_IN * sizeof(float));
    float *stft_imag = malloc(N_IN * sizeof(float));
    fread(stft_real, sizeof(float), N_IN, fr);
    fclose(fr);
    snprintf(path, sizeof(path), "%s/frame0_stft_imag.bin", dir);
    fr = fopen(path, "rb");
    fread(stft_imag, sizeof(float), N_IN, fr);
    fclose(fr);

    /* log_gen with Q20 quantization (matching MATLAB Fix_point then log_gen exactly) */
    int32_t x_log[N_IN];
    for (int i = 0; i < N_IN; i++) {
        /* Step 1: quantize to Q20 (matching Fix_point(stft, 's32f20')) */
        int32_t real_q20 = (int32_t)round((double)stft_real[i] * 1048576.0);
        int32_t imag_q20 = (int32_t)round((double)stft_imag[i] * 1048576.0);
        /* Step 2: dequant to double for sqrt+log (matching x*2^(-20) in log_gen.m) */
        double real = (double)real_q20 / 1048576.0;
        double imag = (double)imag_q20 / 1048576.0;
        double mag = sqrt(real * real + imag * imag);
        double clamped = (mag < 1e-12) ? 1e-12 : mag;
        x_log[i] = (int32_t)round(log10(clamped) * 1048576.0);
    }

    /* BM */
    int32_t y_bm[N_OUT];
    bm_fixed(x_log, N_IN, N_OUT, y_bm);

    /* Load golden */
    snprintf(path, sizeof(path), "%s/frame0_bm.bin", dir);
    int32_t *golden = load_i32(path, N_OUT);

    /* Output comparison */
    printf("Bin,C_Q20,Golden_Q20,Diff_LSB,EffBits_C,EffBits_G,Float_C,Float_G,Error_pct\n");
    int sel_bins[] = {0,8,16,32,48,64,65,72,80,96,112,128};
    for (int si = 0; si < 12; si++) {
        int b = sel_bins[si];
        int32_t cv = y_bm[b], gv = golden[b];
        int32_t diff = cv - gv;
        int eb_c = (cv == 0) ? 20 : (int)ceil(log2(fabs((double)cv))) + 20;
        int eb_g = (gv == 0) ? 20 : (int)ceil(log2(fabs((double)gv))) + 20;
        float fc = cv / 1048576.0f, fg = gv / 1048576.0f;
        float pct = (fg != 0.0f) ? 100.0f * fabsf(fc - fg) / fabsf(fg) : 0.0f;
        printf("%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.4f\n", b, cv, gv, diff, eb_c, eb_g, fc, fg, pct);
    }

    /* Full SNR */
    double s = 0, e = 0;
    for (int i = 0; i < N_OUT; i++) { double g = golden[i], c = y_bm[i], d = g-c; s += g*g; e += d*d; }
    fprintf(stderr, "BM SNR: %.2f dB (bins 0-64 bit-exact: %s)\n",
            10.0*log10(s/e), memcmp(y_bm, golden, LOW_BINS*4)==0 ? "YES" : "NO");

    /* Dump ALL bins to stderr for full table */
    for (int b = 0; b < N_OUT; b++) {
        int32_t cv = y_bm[b], gv = golden[b], diff = cv - gv;
        int eb = (cv == 0) ? 20 : (int)ceil(log2(fabs((double)cv))) + 20;
        fprintf(stderr, "bin %3d: C=%10d G=%10d diff=%6d float_C=%12.6f float_G=%12.6f eff=%d\n",
                b, cv, gv, diff, cv/1048576.0, gv/1048576.0, eb);
    }

    free(stft_real); free(stft_imag); free(golden);
    return 0;
}
