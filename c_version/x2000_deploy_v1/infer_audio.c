/**
 * infer_audio.c — Batch STFT→MASK inference for full audio evaluation
 * ====================================================================
 * Reads STFT frames from binary files, runs UL-UNAS model,
 * writes MASK outputs.
 *
 * Build: make infer_audio
 * Usage: ./infer_audio <input_dir> <output_dir> <n_frames>
 *   input_dir:  contains frame0_stft_real.bin, frame0_stft_imag.bin, ...
 *   output_dir: will contain frame0_mask.bin, frame0_dec.bin, ...
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"
#include "ulunas_ctfa_qr.h"

/* Runtime QR override from qr_config.bin.
 * Format: 30 int32 values = [D0_TA(3), D0_FA(3), D1_TA(3), D1_FA(3), ..., D4_TA(3), D4_FA(3)]
 * If file exists, overrides decoder cTFA QR globals at runtime. */
static void load_qr_config(const char *dir) {
    char path[512];
    snprintf(path, sizeof(path), "%s/qr_config.bin", dir);
    FILE *f = fopen(path, "rb");
    if (!f) return;

    int32_t buf[30];
    if (fread(buf, sizeof(int32_t), 30, f) == 30) {
        #ifdef QR_CALIBRATION_MODE
        g_qr_d0 = (ctfa_qr_t){buf[0], buf[1], buf[2], buf[3], buf[4], buf[5]};
        g_qr_d1 = (ctfa_qr_t){buf[6], buf[7], buf[8], buf[9], buf[10], buf[11]};
        g_qr_d2 = (ctfa_qr_t){buf[12], buf[13], buf[14], buf[15], buf[16], buf[17]};
        g_qr_d3 = (ctfa_qr_t){buf[18], buf[19], buf[20], buf[21], buf[22], buf[23]};
        g_qr_d4 = (ctfa_qr_t){buf[24], buf[25], buf[26], buf[27], buf[28], buf[29]};
        printf("  [QR config loaded: D0-D4 overridden]\n");
        #endif
    }
    fclose(f);
}

static float *load_float(const char *p, int n) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    float *b = malloc(n * sizeof(float));
    if (fread(b, sizeof(float), n, f) != (size_t)n) { free(b); fclose(f); return NULL; }
    fclose(f); return b;
}
static void save_int32(const char *p, const int32_t *d, int n) {
    FILE *f = fopen(p, "wb"); if (!f) { perror(p); return; }
    fwrite(d, sizeof(int32_t), n, f); fclose(f);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input_dir> <output_dir> <n_frames> [dec_bias_q20]\n", argv[0]);
        fprintf(stderr, "  dec_bias_q20: global decoder bias in Q20 (default 0, try 2000000-6000000)\n");
        return 1;
    }
    const char *indir = argv[1], *outdir = argv[2];
    int n_frames = atoi(argv[3]);
    int32_t dec_bias_q20 = argc > 4 ? atoi(argv[4]) : 0;

    ulunas_state_t st; ulunas_state_init(&st);

    /* Load runtime QR config if present */
    load_qr_config(indir);

    for (int f = 0; f < n_frames; f++) {
        char p[512];

        /* Load STFT frame */
        snprintf(p, sizeof(p), "%s/frame%d_stft_real.bin", indir, f);
        float *ri = load_float(p, 257);
        snprintf(p, sizeof(p), "%s/frame%d_stft_imag.bin", indir, f);
        float *ii = load_float(p, 257);
        if (!ri || !ii) { printf("Frame %d: missing STFT, stopping\n", f); free(ri); free(ii); break; }

        /* Run model (full chain, no golden comparison) */
        int32_t x_log[257]; log_gen_fixed(ri, ii, 257, x_log);
        int32_t x_bm[129];  BM_fixed(x_log, erb_erb_fc_weight, x_bm);

        int32_t e0[12*65], e1[24*33], e2[24*33], e3[32*33], e4[16*33];
        Encoder_module(x_bm, &st, e0, e1, e2, e3, e4);

        int32_t r1[16*33], r2[16*33];
        GDPRNN_module(e4, st.inter_prev0, 0, r1);
        GDPRNN_module(r1, st.inter_prev1, 1, r2);

        int32_t y_dec[1*129];
        Decoder_module(r2, &st, e0, e1, e2, e3, e4, y_dec);

        /* Apply global decoder bias (shift sigmoid operating point) */
        if (dec_bias_q20 != 0) {
            for (int i = 0; i < 129; i++)
                y_dec[i] = sat32((int64_t)y_dec[i] + (int64_t)dec_bias_q20);
        }

        /* Save decoder output (with bias applied) */
        snprintf(p, sizeof(p), "%s/frame%d_dec.bin", outdir, f);
        save_int32(p, y_dec, 129);

        /* Sigmoid → BS → MASK */
        uint16_t y_sig[1*129]; sigmoid_fixed(y_dec, 129, y_sig);
        int16_t y_sig_s16[129]; for (int i = 0; i < 129; i++) y_sig_s16[i] = (int16_t)y_sig[i];
        int16_t y_bs[1*257]; BS_fixed(y_sig_s16, erb_ierb_fc_weight, y_bs);

        int32_t real_q[257], imag_q[257];
        for (int i = 0; i < 257; i++) { real_q[i] = F2Q20(ri[i]); imag_q[i] = F2Q20(ii[i]); }
        int32_t crm[2*257];
        MASK_fixed(y_bs, real_q, imag_q, crm);

        /* Save mask output (Q20 complex) */
        snprintf(p, sizeof(p), "%s/frame%d_mask.bin", outdir, f);
        save_int32(p, crm, 2*257);

        free(ri); free(ii);

        if (f % 50 == 0) printf("  Processed %d/%d frames\n", f, n_frames);
    }
    printf("Done: %d frames → %s/\n", n_frames, outdir);
    return 0;
}
