/**
 * sigmoid_bs_mask_test.c — Sigmoid + BS + MASK (standalone)
 * ==========================================================
 * Matches MATLAB:
 *   sigmoid_func: y = 1/(1+exp(-x)) → u16f15
 *   BS_module:    Band Splitting via ERB matrix
 *   MASK_module:  y = round(real.*mask*2^(-15)), round(imag.*mask*2^(-15))
 *
 * Input:  decoder output frame0_dec.bin (Q20, 129 bins)
 *         STFT float input (frame0_stft_real/imag.bin)
 * Output: per-bin comparison for sigmoid, BS, MASK
 *
 * Build: gcc -O2 -std=c99 -o sigmoid_bs_mask_test sigmoid_bs_mask_test.c -lm
 * Usage: ./sigmoid_bs_mask_test <golden_dir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define LOW_BINS  65
#define BS_HI_IN  64
#define BS_HI_OUT 192
#define N_IN_BS   129
#define N_OUT_BS  257

#include "erb_weights.h"

/* ──── sigmoid: float-based for PC verification ──── */
static uint16_t sigmoid_float_to_q15(double x) {
    double s = 1.0 / (1.0 + exp(-x));
    int32_t v = (int32_t)round(s * 32768.0);
    if (v < 0) v = 0; if (v > 65535) v = 65535;
    return (uint16_t)v;
}

/* ──── BS: Band Splitting ──── */
static void bs_fixed(const uint16_t *x, int W_in, int W_out, int16_t *y) {
    for (int i = 0; i < LOW_BINS; i++)
        y[i] = (int16_t)x[i];
    int64_t r = 16384;
    for (int o = 0; o < BS_HI_OUT; o++) {
        int64_t acc = 0;
        for (int i = 0; i < BS_HI_IN; i++) {
            int64_t prod = (int64_t)x[LOW_BINS + i] * erb_ierb_fc_weight[i + BS_HI_IN * o];
            if (prod >= 0) acc += (prod + r) >> 15;
            else           acc += (prod - r) >> 15;
        }
        if (acc > 32767) acc = 32767;
        if (acc < -32768) acc = -32768;
        y[LOW_BINS + o] = (int16_t)acc;
    }
}

/* ──── MASK: Apply mask to complex spectrum ──── */
static void mask_fixed(const int16_t *mask, const int32_t *x_real, const int32_t *x_imag,
                       int W, int32_t *y) {
    int64_t r = 16384;
    for (int i = 0; i < W; i++) {
        int64_t pr = (int64_t)x_real[i] * mask[i];
        int64_t pi = (int64_t)x_imag[i] * mask[i];
        y[i]     = (pr >= 0) ? (int32_t)((pr + r) >> 15) : (int32_t)((pr - r) >> 15);
        y[W + i] = (pi >= 0) ? (int32_t)((pi + r) >> 15) : (int32_t)((pi - r) >> 15);
    }
}

/* ──── Helpers ──── */
static int32_t *load_i32(const char *path, int n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    int32_t *b = malloc(n * sizeof(int32_t)); fread(b, sizeof(int32_t), n, f); fclose(f); return b;
}
static uint16_t *load_u16(const char *path, int n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    uint16_t *b = malloc(n * sizeof(uint16_t)); fread(b, sizeof(uint16_t), n, f); fclose(f); return b;
}
static int16_t *load_i16(const char *path, int n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    int16_t *b = malloc(n * sizeof(int16_t)); fread(b, sizeof(int16_t), n, f); fclose(f); return b;
}

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : ".";
    char path[512];

    /* Load decoder golden output (129 bins Q20) */
    snprintf(path, sizeof(path), "%s/frame0_dec.bin", dir);
    int32_t *dec = load_i32(path, 129);
    if (!dec) { fprintf(stderr, "No frame0_dec.bin\n"); return 1; }

    /* Load STFT golden for MASK */
    snprintf(path, sizeof(path), "%s/frame0_stft_real.bin", dir);
    FILE *fr = fopen(path, "rb"); float *stft_r = malloc(257*sizeof(float)); fread(stft_r, 4, 257, fr); fclose(fr);
    snprintf(path, sizeof(path), "%s/frame0_stft_imag.bin", dir);
    FILE *fi = fopen(path, "rb"); float *stft_i = malloc(257*sizeof(float)); fread(stft_i, 4, 257, fi); fclose(fi);

    /* ──── Step 1: Sigmoid ──── */
    uint16_t sig_c[129];
    for (int i = 0; i < 129; i++) {
        double x = (double)dec[i] / 1048576.0;
        sig_c[i] = sigmoid_float_to_q15(x);
    }
    snprintf(path, sizeof(path), "%s/frame0_sig.bin", dir);
    uint16_t *sig_g = load_u16(path, 129);

    printf("=== sigmoid_func ===\n");
    printf("Bin,C_u16f15,Golden_u16f15,Diff_LSB,Float_C,Float_G\n");
    int sel_sig[] = {0,4,8,16,32,48,64,72,80,96,112,128};
    for (int si = 0; si < 12; si++) {
        int b = sel_sig[si];
        printf("%d,%u,%u,%d,%.6f,%.6f\n", b, sig_c[b], sig_g[b],
               (int)sig_c[b]-(int)sig_g[b], sig_c[b]/32768.0f, sig_g[b]/32768.0f);
    }
    double s=0,e=0;
    for(int i=0;i<129;i++){double g=sig_g[i],c=sig_c[i],d=g-c;s+=g*g;e+=d*d;}
    fprintf(stderr,"Sigmoid SNR: %.2f dB\n", 10.0*log10(s/e));

    /* ──── Step 2: BS ──── */
    int16_t bs_c[257];
    bs_fixed(sig_c, 129, 257, bs_c);
    snprintf(path, sizeof(path), "%s/frame0_bs.bin", dir);
    int16_t *bs_g = load_i16(path, 257);

    printf("\n=== BS_module ===\n");
    printf("Bin,C_s16f15,Golden_s16f15,Diff_LSB,Float_C,Float_G\n");
    int sel_bs[] = {0,8,16,32,48,64,65,72,80,96,128,160,200,240,256};
    for (int si = 0; si < 15; si++) {
        int b = sel_bs[si];
        printf("%d,%d,%d,%d,%.6f,%.6f\n", b, bs_c[b], bs_g[b],
               (int)bs_c[b]-(int)bs_g[b], bs_c[b]/32768.0f, bs_g[b]/32768.0f);
    }
    s=0;e=0;
    for(int i=0;i<257;i++){double g=bs_g[i],c=bs_c[i],d=g-c;s+=g*g;e+=d*d;}
    fprintf(stderr,"BS SNR: %.2f dB\n", 10.0*log10(s/e));

    /* ──── Step 3: MASK ──── */
    int32_t real_q20[257], imag_q20[257];
    for (int i = 0; i < 257; i++) {
        real_q20[i] = (int32_t)round((double)stft_r[i] * 1048576.0);
        imag_q20[i] = (int32_t)round((double)stft_i[i] * 1048576.0);
    }
    int32_t mask_c[514];
    mask_fixed(bs_c, real_q20, imag_q20, 257, mask_c);
    snprintf(path, sizeof(path), "%s/frame0_mask.bin", dir);
    int32_t *mask_g_raw = load_i32(path, 514);  /* 514 = 257×2 interleaved: [R0,I0,R1,I1,...] */

    /* Deinterleave golden: raw[0]=R0, raw[1]=I0, raw[2]=R1, raw[3]=I1, ... */
    int32_t mask_g_real[257], mask_g_imag[257];
    for (int i = 0; i < 257; i++) {
        mask_g_real[i] = mask_g_raw[i * 2];
        mask_g_imag[i] = mask_g_raw[i * 2 + 1];
    }

    printf("\n=== MASK_module ===\n");
    printf("Bin,C_Real_Q20,Golden_Real_Q20,Diff_Re,C_Imag_Q20,Golden_Imag_Q20,Diff_Im,EffBits_Re,EffBits_Im\n");
    int sel_mask[] = {0,4,8,16,32,46,64,80,96,128,160,200,240,256};
    for (int si = 0; si < 14; si++) {
        int b = sel_mask[si];
        int32_t cr = mask_c[b], gr = mask_g_real[b];
        int32_t ci = mask_c[257+b], gi = mask_g_imag[b];
        int ebr = (cr==0)?20:(int)ceil(log2(fabs((double)cr)))+20;
        int ebi = (ci==0)?20:(int)ceil(log2(fabs((double)ci)))+20;
        printf("%d,%d,%d,%d,%d,%d,%d,%d,%d\n", b, cr, gr, cr-gr, ci, gi, ci-gi, ebr, ebi);
    }
    /* SNR: compare deinterleaved golden with C output (interleaved) */
    s=0;e=0;
    for(int i=0;i<257;i++){
        double gr=mask_g_real[i],cr=mask_c[i],dg=gr-cr; s+=gr*gr; e+=dg*dg;
        double gi=mask_g_imag[i],ci=mask_c[257+i],di=gi-ci; s+=gi*gi; e+=di*di;
    }
    fprintf(stderr,"MASK SNR: %.2f dB\n", 10.0*log10(s/e));

    /* dump full MASK for guide tables */
    for(int b=0;b<257;b++){
        int32_t cr=mask_c[b],gr=mask_g_real[b],ci=mask_c[257+b],gi=mask_g_imag[b];
        int eb=(cr==0)?20:(int)ceil(log2(fabs((double)cr)))+20;
        fprintf(stderr,"mask bin %3d: C_re=%10d G_re=%10d C_im=%10d G_im=%10d eff=%d re_float=%.6f im_float=%.6f\n",
                b,cr,gr,ci,gi,eb,cr/1048576.0,ci/1048576.0);
    }

    free(dec); free(sig_g); free(bs_g); free(mask_g_raw); free(stft_r); free(stft_i);
    return 0;
}
