/**
 * test_layer_snr.c — Per-layer SNR vs MATLAB golden (NEW FILE, does not modify existing code)
 * ==========================================================================================
 * Tests linux_api9 operators against golden binary dumps frame by frame.
 * Uses existing linux_api9 source files (compiled separately, linked at build time).
 *
 * Build:
 *   gcc -O2 -std=c99 -I. -DPC_PLATFORM -o test_layer_snr \
 *       test_layer_snr.c ulunas_fp.c ulunas_lut.c ulunas_modules.c ulunas_matlab_weights.c -lm
 *
 * Usage:
 *   ./test_layer_snr <golden_dir>
 */
#include "ulunas_fp.h"
#include "ulunas_lut.h"
#include "qr_config.h"
#include "layer_dims.h"
#include "ulunas_matlab_weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ─── Helpers ─── */
static double snr_db_i32(const int32_t *golden, const int32_t *test, int n) {
    double s_num = 0.0, s_den = 0.0;
    for (int i = 0; i < n; i++) {
        double gv = (double)golden[i];
        double tv = (double)test[i];
        s_num += gv * gv;
        double diff = gv - tv;
        s_den += diff * diff;
    }
    if (s_den < 1e-30) return 999.0;
    return 10.0 * log10(s_num / s_den);
}

static double snr_db_i16(const int16_t *golden, const int16_t *test, int n) {
    double s_num = 0.0, s_den = 0.0;
    for (int i = 0; i < n; i++) {
        double gv = (double)golden[i];
        double tv = (double)test[i];
        s_num += gv * gv;
        double diff = gv - tv;
        s_den += diff * diff;
    }
    if (s_den < 1e-30) return 999.0;
    return 10.0 * log10(s_num / s_den);
}

static int32_t *load_i32(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    int32_t *data = (int32_t*)malloc(n * sizeof(int32_t));
    if (!data) { fclose(f); return NULL; }
    size_t rd = fread(data, sizeof(int32_t), n, f);
    fclose(f);
    if (rd != (size_t)n) { free(data); return NULL; }
    return data;
}

static int16_t *load_i16_as_i32(const char *path, int n) {
    /* Golden BS output is int16 */
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    int16_t *data = (int16_t*)malloc(n * sizeof(int16_t));
    if (!data) { fclose(f); return NULL; }
    size_t rd = fread(data, sizeof(int16_t), n, f);
    fclose(f);
    if (rd != (size_t)n) { free(data); return NULL; }
    return data;
}

static uint16_t *load_u16(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint16_t *data = (uint16_t*)malloc(n * sizeof(uint16_t));
    if (!data) { fclose(f); return NULL; }
    size_t rd = fread(data, sizeof(uint16_t), n, f);
    fclose(f);
    if (rd != (size_t)n) { free(data); return NULL; }
    return data;
}

#define SNR_TEST(label, golden, test, n, type) do { \
    double s = snr_db_##type(golden, test, n); \
    printf("  %-30s %8.2f dB  %s\n", label, s, (s >= 80.0) ? "PASS" : (s >= 30.0 ? "WARN" : "FAIL")); \
    if (s >= 80.0) passed++; else if (s < 30.0) failed++; \
    total++; \
} while(0)

/* ─── Main ─── */
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <golden_dir>\n", argv[0]);
        return 1;
    }
    const char *dir = argv[1];
    char path[1024];
    int total = 0, passed = 0, failed = 0;

    /* We need STFT real/imag input for log_gen. If not available, skip those steps.
     * For now, start from BM golden as input. */
    printf("=== UL-UNAS linux_api9 Per-Layer SNR vs MATLAB Golden ===\n\n");

    /* ── Load BM input ── */
    snprintf(path, sizeof(path), "%s/frame001_bm.bin", dir);
    int32_t *x_bm = load_i32(path, BM_OUT_BINS);  /* 129 */
    if (!x_bm) {
        printf("FATAL: cannot load %s\n", path);
        return 1;
    }
    printf("Input: frame001_bm.bin (%d elements)\n\n", BM_OUT_BINS);

    /* ── Encoder ── */
    printf("─── Encoder ───\n");
    {
        ulunas_state_t st;
        ulunas_state_init(&st);

        int32_t e0[12 * 65], e1[24 * 33], e2[24 * 33], e3[32 * 33], e4[16 * 33];
        encoder_module(x_bm, &st, e0, e1, e2, e3, e4);

        /* E0 */
        snprintf(path, sizeof(path), "%s/frame001_e0.bin", dir);
        int32_t *g_e0 = load_i32(path, 12 * 65);
        if (g_e0) { SNR_TEST("E0 (XConv)", g_e0, e0, 12 * 65, i32); free(g_e0); }
        else printf("  E0: SKIP (no golden)\n");

        /* E1 */
        snprintf(path, sizeof(path), "%s/frame001_e1.bin", dir);
        int32_t *g_e1 = load_i32(path, 24 * 33);
        if (g_e1) { SNR_TEST("E1 (XMB0)", g_e1, e1, 24 * 33, i32); free(g_e1); }
        else printf("  E1: SKIP (no golden)\n");

        /* E2 */
        snprintf(path, sizeof(path), "%s/frame001_e2.bin", dir);
        int32_t *g_e2 = load_i32(path, 24 * 33);
        if (g_e2) { SNR_TEST("E2 (XDWS0)", g_e2, e2, 24 * 33, i32); free(g_e2); }
        else printf("  E2: SKIP (no golden)\n");

        /* E3 */
        snprintf(path, sizeof(path), "%s/frame001_e3.bin", dir);
        int32_t *g_e3 = load_i32(path, 32 * 33);
        if (g_e3) { SNR_TEST("E3 (XMB1)", g_e3, e3, 32 * 33, i32); free(g_e3); }
        else printf("  E3: SKIP (no golden)\n");

        /* E4 (Full Encoder output) */
        snprintf(path, sizeof(path), "%s/frame001_e4.bin", dir);
        int32_t *g_e4 = load_i32(path, 16 * 33);
        if (g_e4) { SNR_TEST("E4 (XDWS1)", g_e4, e4, 16 * 33, i32); free(g_e4); }
        else printf("  E4: SKIP (no golden)\n");

        printf("\n─── GDPRNN ───\n");

        /* RNN1 */
        int32_t rnn1[16 * 33];
        gdprnn_module(e4, st.inter_cache_0, 0, rnn1);

        snprintf(path, sizeof(path), "%s/frame001_rnn1.bin", dir);
        int32_t *g_rnn1 = load_i32(path, 16 * 33);
        if (g_rnn1) { SNR_TEST("RNN1 (GDPRNN Block0)", g_rnn1, rnn1, 16 * 33, i32); free(g_rnn1); }
        else printf("  RNN1: SKIP (no golden)\n");

        /* RNN2 */
        int32_t rnn2[16 * 33];
        gdprnn_module(rnn1, st.inter_cache_1, 1, rnn2);

        snprintf(path, sizeof(path), "%s/frame001_rnn2.bin", dir);
        int32_t *g_rnn2 = load_i32(path, 16 * 33);
        if (g_rnn2) { SNR_TEST("RNN2 (GDPRNN Block1)", g_rnn2, rnn2, 16 * 33, i32); free(g_rnn2); }
        else printf("  RNN2: SKIP (no golden)\n");

        printf("\n─── Decoder ───\n");

        /* Full decoder */
        int32_t y_dec[1 * 129];
        decoder_module(rnn2, &st, e0, e1, e2, e3, e4, y_dec);

        snprintf(path, sizeof(path), "%s/frame001_dec.bin", dir);
        int32_t *g_dec = load_i32(path, 1 * 129);
        if (g_dec) { SNR_TEST("Decoder (final)", g_dec, y_dec, 1 * 129, i32); free(g_dec); }
        else printf("  Dec: SKIP (no golden)\n");

        /* D0 */
        snprintf(path, sizeof(path), "%s/frame001_d0.bin", dir);
        int32_t *g_d0 = load_i32(path, 32 * 33);
        if (g_d0) {
            /* Re-run decoder with layer isolation: call individual layers */
            ulunas_state_t st2; ulunas_state_init(&st2);
            int32_t d0_out[32 * 33];
            decoder_layer0_de_xdws0(rnn2, e3, &st2, d0_out);
            SNR_TEST("D0 (De_XDWS0)", g_d0, d0_out, 32 * 33, i32);
            free(g_d0);
        } else printf("  D0: SKIP (no golden)\n");

        /* D1 */
        snprintf(path, sizeof(path), "%s/frame001_d1.bin", dir);
        int32_t *g_d1 = load_i32(path, 24 * 33);
        if (g_d1) {
            ulunas_state_t st2; ulunas_state_init(&st2);
            int32_t d0_out[32 * 33], d1_out[24 * 33];
            decoder_layer0_de_xdws0(rnn2, e3, &st2, d0_out);
            decoder_layer1_de_xmb0(d0_out, e2, &st2, d1_out);
            SNR_TEST("D1 (De_XMB0)", g_d1, d1_out, 24 * 33, i32);
            free(g_d1);
        } else printf("  D1: SKIP (no golden)\n");

        /* D2 */
        snprintf(path, sizeof(path), "%s/frame001_d2.bin", dir);
        int32_t *g_d2 = load_i32(path, 24 * 33);
        if (g_d2) {
            ulunas_state_t st2; ulunas_state_init(&st2);
            int32_t d0_out[32 * 33], d1_out[24 * 33], d2_out[24 * 33];
            decoder_layer0_de_xdws0(rnn2, e3, &st2, d0_out);
            decoder_layer1_de_xmb0(d0_out, e2, &st2, d1_out);
            decoder_layer2_de_xdws1(d1_out, e1, &st2, d2_out);
            SNR_TEST("D2 (De_XDWS1)", g_d2, d2_out, 24 * 33, i32);
            free(g_d2);
        } else printf("  D2: SKIP (no golden)\n");

        /* D3 */
        snprintf(path, sizeof(path), "%s/frame001_d3.bin", dir);
        int32_t *g_d3 = load_i32(path, 12 * 65);
        if (g_d3) {
            ulunas_state_t st2; ulunas_state_init(&st2);
            int32_t d0_out[32 * 33], d1_out[24 * 33], d2_out[24 * 33], d3_out[12 * 65];
            decoder_layer0_de_xdws0(rnn2, e3, &st2, d0_out);
            decoder_layer1_de_xmb0(d0_out, e2, &st2, d1_out);
            decoder_layer2_de_xdws1(d1_out, e1, &st2, d2_out);
            decoder_layer3_de_xmb1(d2_out, e0, &st2, d3_out);
            SNR_TEST("D3 (De_XMB1)", g_d3, d3_out, 12 * 65, i32);
            free(g_d3);
        } else printf("  D3: SKIP (no golden)\n");

        printf("\n─── Output Stages ───\n");

        /* Sigmoid */
        uint16_t y_sig[129];
        for (int i = 0; i < 129; i++) y_sig[i] = sigmoid_q20_to_q15(y_dec[i]);

        snprintf(path, sizeof(path), "%s/frame001_sig.bin", dir);
        uint16_t *g_sig = load_u16(path, 129);
        if (g_sig) {
            /* Compare as int16 for SNR computation */
            int16_t *sig_i16 = (int16_t*)malloc(129 * sizeof(int16_t));
            int16_t *gsig_i16 = (int16_t*)malloc(129 * sizeof(int16_t));
            for (int i = 0; i < 129; i++) {
                sig_i16[i] = (int16_t)y_sig[i];
                gsig_i16[i] = (int16_t)g_sig[i];
            }
            SNR_TEST("Sigmoid (dec→mask)", gsig_i16, sig_i16, 129, i16);
            free(sig_i16); free(gsig_i16); free(g_sig);
        } else printf("  Sigmoid: SKIP (no golden)\n");

        /* BS */
        int16_t y_bs[257];
        bs_fixed(y_sig, erb_ierb_fc_weight, 129, 257, y_bs);

        snprintf(path, sizeof(path), "%s/frame001_bs.bin", dir);
        int16_t *g_bs = load_i16_as_i32(path, 257);
        if (g_bs) {
            SNR_TEST("BS (Band Split)", g_bs, y_bs, 257, i16);
            free(g_bs);
        } else printf("  BS: SKIP (no golden)\n");
    }

    free(x_bm);

    printf("\n=== Result: %d/%d PASS (>=80dB), %d FAIL (<30dB), %d total ===\n",
           passed, total, failed, total);
    return failed ? 1 : 0;
}
