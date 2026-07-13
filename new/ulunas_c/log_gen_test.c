/**
 * log_gen_test.c — Fixed-point Log-Magnitude Compression (standalone)
 * ====================================================================
 * Matches MATLAB log_gen.m:
 *   mag = sqrt(real.^2 + imag.^2)     [integer sqrt, Q40→Q20]
 *   clamped = max(mag, 1e-12)         [clamp to 1 LSB]
 *   y = log10(clamped)                [LUT + interpolation]
 *   y = Fix_point(y, 's32f20')        [already Q20]
 *
 * Input:  frame0_stft_real.bin + frame0_stft_imag.bin (257 float32 each)
 * Output: CSV of per-bin: Bin, C_real_Q20, C_imag_Q20, C_log_Q20, Golden_log_Q20
 *
 * Build: gcc -O2 -std=c99 -o log_gen_test log_gen_test.c -lm
 * Usage: ./log_gen_test <golden_dir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define N_BINS 257

/* ================================================================
 * Integer sqrt: Q40 → Q20 via binary search
 * ================================================================ */
static uint32_t isqrt_q40_to_q20(uint64_t x) {
    uint64_t lo = 0, hi = (x > 0xFFFFFFFEULL) ? 0xFFFFFFFFULL : (x + 1);
    if (hi > 0xFFFFFFFFULL) hi = 0xFFFFFFFFULL;
    while (lo + 1 < hi) {
        uint64_t mid = (lo + hi) >> 1;
        if (mid * mid <= x) lo = mid;
        else hi = mid;
    }
    return (uint32_t)(lo > 0 ? lo : 1);
}

/* ================================================================
 * log10 LUT: log10(x) * 2^20 for x in [1, 2^32)
 * ================================================================
 * Table covers integer powers of 2: log10(2^k) * 2^20
 * log10(2) * 1048576 = 315652.9... ≈ 315653
 * Between table entries: linear interpolation
 */
#define LOG10_LUT_SIZE 32
static const int32_t log10_lut_base[LOG10_LUT_SIZE] = {
         0,    315653,   631306,   946959,  1262612,  1578265,  1893918,  2209571,
   2525224,  2840877,  3156530,  3472183,  3787836,  4103489,  4419142,  4734795,
   5050448,  5366101,  5681754,  5997407,  6313060,  6628713,  6944366,  7260019,
   7575672,  7891325,  8206978,  8522631,  8838284,  9153937,  9469590,  9785243
};

/**
 * log10_q20 — PC verification version using float (bit-exact with MATLAB)
 * Input:  mag_q20 — magnitude in Q20
 * Output: log10(mag_q20 / 2^20) * 2^20
 *
 * NOTE: For X2000 (no FPU), replace with 2048-entry LUT + linear interpolation
 * covering [1, 2^32) with log10(x/2^20)*2^20 directly.
 */
static int32_t log10_q20(int32_t mag_q20) {
    if (mag_q20 <= 0) return -22000000;
    double mag = (double)mag_q20 / 1048576.0;
    double clamped = (mag < 1e-12) ? 1e-12 : mag;
    return (int32_t)round(log10(clamped) * 1048576.0);
}

/* ================================================================
 * log_gen: Main function
 * ================================================================ */
/**
 * log_gen: PC verification version using float sqrt+log10 for bit-exact comparison.
 * For X2000: use isqrt_q40_to_q20 + 2048-entry log10 LUT.
 */
static void log_gen_fixed(const int32_t *real_q20, const int32_t *imag_q20, int W, int32_t *out) {
    for (int w = 0; w < W; w++) {
        double real_f = (double)real_q20[w] / 1048576.0;
        double imag_f = (double)imag_q20[w] / 1048576.0;
        double mag = sqrt(real_f * real_f + imag_f * imag_f);
        double clamped = (mag < 1e-12) ? 1e-12 : mag;
        out[w] = (int32_t)round(log10(clamped) * 1048576.0);
    }
}

/* ================================================================
 * Helpers
 * ================================================================ */
static float *load_float(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    float *buf = malloc(n * sizeof(float));
    fread(buf, sizeof(float), n, f);
    fclose(f);
    return buf;
}

static int32_t *load_i32(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    int32_t *buf = malloc(n * sizeof(int32_t));
    fread(buf, sizeof(int32_t), n, f);
    fclose(f);
    return buf;
}

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : ".";

    /* Load STFT golden input */
    char path[512];
    snprintf(path, sizeof(path), "%s/frame0_stft_real.bin", dir);
    float *stft_real = load_float(path, N_BINS);
    snprintf(path, sizeof(path), "%s/frame0_stft_imag.bin", dir);
    float *stft_imag = load_float(path, N_BINS);

    if (!stft_real || !stft_imag) {
        fprintf(stderr, "Error: cannot load STFT input from %s/\n", dir);
        return 1;
    }

    /* Quantize to Q20 */
    int32_t real_q20[N_BINS], imag_q20[N_BINS];
    for (int i = 0; i < N_BINS; i++) {
        real_q20[i] = (int32_t)round(stft_real[i] * 1048576.0f);
        imag_q20[i] = (int32_t)round(stft_imag[i] * 1048576.0f);
    }

    /* Run log_gen */
    int32_t c_log[N_BINS];
    log_gen_fixed(real_q20, imag_q20, N_BINS, c_log);

    /* Load BM golden (bins 0-64 are passthrough = log_gen golden) */
    snprintf(path, sizeof(path), "%s/frame0_bm.bin", dir);
    int32_t *bm_golden = load_i32(path, 129);

    /* Output CSV: Bin, C_log_Q20, Golden_log_Q20, diff, eff_bits */
    printf("Bin,C_log_Q20,Golden_log_Q20,Diff_LSB,EffectiveBits_C,EffectiveBits_G,Float_C,Float_G,Error_pct\n");
    for (int b = 0; b < 65 && b < N_BINS; b++) {
        int32_t cv = c_log[b];
        int32_t gv = bm_golden ? bm_golden[b] : 0;
        int32_t diff = cv - gv;
        int eff_c = (cv == 0) ? 20 : (int)ceil(log2(fabs((double)cv))) + 20;
        int eff_g = (gv == 0) ? 20 : (int)ceil(log2(fabs((double)gv))) + 20;
        float fc = cv / 1048576.0f;
        float fg = gv / 1048576.0f;
        float pct = (fg != 0.0f) ? 100.0f * fabsf(fc - fg) / fabsf(fg) : 0.0f;
        printf("%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.4f\n",
               b, cv, gv, diff, eff_c, eff_g, fc, fg, pct);
    }

    /* Also dump all 257 bins of C output for reference */
    fprintf(stderr, "# Full 257-bin C log_gen output:\n");
    for (int b = 0; b < N_BINS; b++) {
        fprintf(stderr, "  bin %3d: Q20=%10d  float=%12.6f  eff_bits=%d\n",
                b, c_log[b], c_log[b]/1048576.0f,
                (c_log[b]==0) ? 20 : (int)ceil(log2(fabs((double)c_log[b])))+20);
    }

    free(stft_real); free(stft_imag); free(bm_golden);
    return 0;
}
