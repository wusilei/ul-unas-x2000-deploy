/* dump_float.c — Run pipeline Frame 0, dump every layer as float */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_lut.h"
#include "qr_config.h"
#include "layer_dims.h"
#include "ulunas_matlab_weights.h"

static float *load_f32(const char *path, int n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    float *b = (float*)malloc(n * sizeof(float));
    fread(b, sizeof(float), n, f); fclose(f); return b;
}

static void dump_f32(const char *name, const int32_t *data, int n, int q) {
    printf("%s = [", name);
    for (int i = 0; i < n; i++) {
        printf("%.8f", (double)data[i] / (double)(1 << q));
        if (i < n - 1) printf(", ");
    }
    printf("];\n");
}

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : "dump_matlab";
    char path[512];
    int frame = 0;

    snprintf(path, sizeof(path), "%s/frame%d_stft_real.bin", dir, frame);
    float *sr = load_f32(path, 257);
    snprintf(path, sizeof(path), "%s/frame%d_stft_imag.bin", dir, frame);
    float *si = load_f32(path, 257);
    if (!sr || !si) { printf("No input\n"); return 1; }

    ulunas_state_t st; ulunas_state_init(&st);

    /* Quantize STFT to Q20 */
    int32_t rq[257], iq[257];
    for (int i = 0; i < 257; i++) {
        rq[i] = (int32_t)round(sr[i] * 1048576.0f);
        iq[i] = (int32_t)round(si[i] * 1048576.0f);
    }

    /* [1] STFT — dump float input */
    printf("=== C Frame 0 Layer Float Dump ===\n");
    printf("STFT_real = ["); for(int i=0;i<257;i++) printf("%.8f%s", sr[i], i<256?", ":""); printf("];\n");
    printf("STFT_imag = ["); for(int i=0;i<257;i++) printf("%.8f%s", si[i], i<256?", ":""); printf("];\n");

    /* [2] log_gen */
    int32_t x_log[257];
    log_gen_fixed(rq, iq, 257, x_log);
    dump_f32("log_gen", x_log, 257, 20);

    /* [3] BM */
    int32_t x_bm[129];
    bm_fixed(x_log, erb_erb_fc_weight, 257, 129, x_bm);
    dump_f32("BM", x_bm, 129, 20);

    /* [4] Encoder */
    int32_t e0[12*65], e1[24*33], e2[24*33], e3[32*33], e4[16*33];
    encoder_module(x_bm, &st, e0, e1, e2, e3, e4);
    dump_f32("E0", e0, 12*65, 20);
    dump_f32("E1", e1, 24*33, 20);
    dump_f32("E2", e2, 24*33, 20);
    dump_f32("E3", e3, 32*33, 20);
    dump_f32("E4", e4, 16*33, 20);

    /* [5] RNN */
    int32_t r1[16*33], r2[16*33];
    gdprnn_module(e4, st.inter_cache_0, 0, r1);
    gdprnn_module(r1, st.inter_cache_1, 1, r2);
    dump_f32("RNN1", r1, 16*33, 20);
    dump_f32("RNN2", r2, 16*33, 20);

    /* [6] Decoder */
    int32_t y_dec[129];
    decoder_module(r2, &st, e0, e1, e2, e3, e4, y_dec);
    dump_f32("Decoder", y_dec, 129, 20);

    /* [7] Sigmoid */
    uint16_t y_sig[129];
    for (int i = 0; i < 129; i++) y_sig[i] = sigmoid_q20_to_q15(y_dec[i]);
    printf("Sigmoid = ["); for(int i=0;i<129;i++) printf("%.8f%s", (double)y_sig[i]/32768.0, i<128?", ":""); printf("];\n");

    /* [8] BS */
    int16_t y_bs[257];
    bs_fixed(y_sig, erb_ierb_fc_weight, 129, 257, y_bs);
    printf("BS = ["); for(int i=0;i<257;i++) printf("%.8f%s", (double)y_bs[i]/32768.0, i<256?", ":""); printf("];\n");

    /* [9] MASK */
    int32_t y_mask[514];
    mask_fixed(y_bs, rq, iq, 257, y_mask);
    printf("MASK_real = ["); for(int i=0;i<257;i++) printf("%.8f%s", (double)y_mask[i]/1048576.0, i<256?", ":""); printf("];\n");
    printf("MASK_imag = ["); for(int i=0;i<257;i++) printf("%.8f%s", (double)y_mask[257+i]/1048576.0, i<256?", ":""); printf("];\n");

    free(sr); free(si);
    return 0;
}
