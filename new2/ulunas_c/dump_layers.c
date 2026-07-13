#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../../UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/ulunas_fp.h"
#include "../../UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/ulunas_lut.h"
#include "../../UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/qr_config.h"
#include "../../UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/layer_dims.h"
#include "../../UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/ulunas_matlab_weights.h"

static void write_i32(const char *path, const int32_t *data, int n) {
    FILE *f = fopen(path, "wb");
    if (!f) return;
    fwrite(data, sizeof(int32_t), n, f);
    fclose(f);
}

static float *load_float(const char *path, int n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    float *b = (float*)malloc(n * sizeof(float));
    if (!b) { fclose(f); return NULL; }
    fread(b, sizeof(float), n, f);
    fclose(f);
    return b;
}

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : "dump_matlab";
    char path[512];
    int frame = 0;

    snprintf(path, sizeof(path), "%s/frame%d_stft_real.bin", dir, frame);
    float *sr = load_float(path, 257);
    snprintf(path, sizeof(path), "%s/frame%d_stft_imag.bin", dir, frame);
    float *si = load_float(path, 257);
    if (!sr || !si) { printf("No STFT input\n"); return 1; }

    int32_t rq[257], iq[257];
    for (int i = 0; i < 257; i++) {
        rq[i] = (int32_t)round(sr[i] * 1048576.0f);
        iq[i] = (int32_t)round(si[i] * 1048576.0f);
    }
    free(sr); free(si);

    ulunas_state_t st;
    ulunas_state_init(&st);

    /* log_gen */
    int32_t x_log[257];
    log_gen_fixed(rq, iq, 257, x_log);
    write_i32("dump_c/frame0_log_gen_c.bin", x_log, 257);

    /* BM */
    int32_t x_bm[129];
    bm_fixed(x_log, erb_erb_fc_weight, 257, 129, x_bm);
    write_i32("dump_c/frame0_bm_c.bin", x_bm, 129);

    /* Encoder */
    int32_t e0[12*65], e1[24*33], e2[24*33], e3[32*33], e4[16*33];
    encoder_module(x_bm, &st, e0, e1, e2, e3, e4);
    write_i32("dump_c/frame0_e0_xconv_c.bin", e0, 12*65);
    write_i32("dump_c/frame0_e1_xmb0_c.bin", e1, 24*33);
    write_i32("dump_c/frame0_e2_xdws0_c.bin", e2, 24*33);
    write_i32("dump_c/frame0_e3_xmb1_c.bin", e3, 32*33);
    write_i32("dump_c/frame0_e4_xdws1_c.bin", e4, 16*33);

    /* RNN */
    int32_t r1[16*33], r2[16*33];
    gdprnn_module(e4, st.inter_cache_0, 0, r1);
    write_i32("dump_c/frame0_gdprnn1_c.bin", r1, 16*33);
    gdprnn_module(r1, st.inter_cache_1, 1, r2);
    write_i32("dump_c/frame0_gdprnn2_c.bin", r2, 16*33);

    /* Decoder */
    int32_t y_dec[129];
    decoder_module(r2, &st, e0, e1, e2, e3, e4, y_dec);
    write_i32("dump_c/frame0_decoder_c.bin", y_dec, 129);

    /* Sigmoid */
    uint16_t y_sig[129];
    for (int i = 0; i < 129; i++)
        y_sig[i] = sigmoid_q20_to_q15(y_dec[i]);
    write_i32("dump_c/frame0_sigmoid_c.bin", (int32_t*)y_sig, 129);

    /* BS */
    int16_t y_bs[257];
    bs_fixed(y_sig, erb_ierb_fc_weight, 129, 257, y_bs);
    write_i32("dump_c/frame0_bs_c.bin", (int32_t*)y_bs, 257);

    /* MASK */
    int32_t y_mask[514];
    mask_fixed(y_bs, rq, iq, 257, y_mask);
    write_i32("dump_c/frame0_mask_c.bin", y_mask, 514);

    printf("All 12 layers dumped to dump_c/\n");
    return 0;
}
