/**
 * diag_frames.c — Per-frame STFT energy + decoder/MASK diagnostics
 * Build: gcc -O2 -std=c99 -o diag_frames diag_frames.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float *load_float(const char *p, int n) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    float *b = malloc(n * sizeof(float));
    if (fread(b, sizeof(float), n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    fclose(f); return b;
}
static int32_t *load_int32(const char *p, int n) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
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
    for (int f = 0; f < 5; f++) {
        char p[256];
        float *ri, *ii; int32_t *gd, *gm;

        snprintf(p, sizeof(p), "dump_matlab/frame%d_stft_real.bin", f); ri = load_float(p, 257);
        snprintf(p, sizeof(p), "dump_matlab/frame%d_stft_imag.bin", f); ii = load_float(p, 257);
        snprintf(p, sizeof(p), "dump_matlab/frame%d_dec.bin", f);      gd = load_int32(p, 129);
        snprintf(p, sizeof(p), "dump_matlab/frame%d_mask.bin", f);     gm = load_int32(p, 514);

        if (!ri || !ii || !gd || !gm) { printf("Frame %d: MISSING\n", f); continue; }

        /* STFT energy (dB relative to full scale) */
        double stft_pow = 0;
        for (int i = 0; i < 257; i++) stft_pow += ri[i]*ri[i] + ii[i]*ii[i];
        double stft_db = 10 * log10(stft_pow / 257);

        /* STFT magnitude stats */
        double max_mag = 0, mean_mag = 0;
        for (int i = 0; i < 257; i++) {
            double mag = sqrt(ri[i]*ri[i] + ii[i]*ii[i]);
            mean_mag += mag;
            if (mag > max_mag) max_mag = mag;
        }
        mean_mag /= 257;
        double mag_db = 20 * log10(mean_mag);

        /* Decoder output stats (Q20) */
        double dec_pow = 0, dec_max = 0;
        for (int i = 0; i < 129; i++) {
            double v = fabs((double)gd[i]);
            dec_pow += v * v;
            if (v > dec_max) dec_max = v;
        }
        double dec_rms_db = 10 * log10(dec_pow / 129) - 20*log10(1<<20);  /* dB relative to Q20 */

        /* Golden mask stats (Q20) */
        double mask_pow = 0, mask_max = 0;
        for (int i = 0; i < 514; i++) {
            double v = fabs((double)gm[i]);
            mask_pow += v * v;
            if (v > mask_max) mask_max = v;
        }
        double mask_rms_db = 10 * log10(mask_pow / 514) - 20*log10(1<<20);

        /* Compute decoder and mask error distributions */
        // Decoder error vs golden
        double dec_err_by_band[129] = {0};
        for (int i = 0; i < 129; i++) dec_err_by_band[i] = 0; // placeholder

        printf("Frame %d: STFT energy=%.1f dB  mean_mag=%.1f dBFS  max_mag=%.3f\n",
               f, stft_db, mag_db, max_mag);
        printf("  golden dec rms=%.1f dB  max=%.0f  golden mask rms=%.1f dB  max=%.0f\n",
               dec_rms_db, dec_max, mask_rms_db, mask_max);

        /* Correlation: dec error vs STFT magnitude per band */
        // MASK error per bin (real+imag)
        // For each ERB band in decoder output, find corresponding FFT bins
        // Simplified: just look at overall stats
        printf("  dec/golden SNR would be computed from the above\n\n");

        free(ri); free(ii); free(gd); free(gm);
    }
    return 0;
}
