/* check_rnn.c — Run pipeline, inspect inter_cache values after RNN */
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
    float *b = (float*)malloc(n*4); fread(b,4,n,f); fclose(f); return b;
}

int main() {
    /* Load STFT */
    float *sr = load_f32("dump_matlab/frame0_stft_real.bin", 257);
    float *si = load_f32("dump_matlab/frame0_stft_imag.bin", 257);
    if (!sr || !si) { printf("No input\n"); return 1; }

    ulunas_state_t st;
    ulunas_state_init(&st);

    /* Check state after init */
    printf("=== After init ===\n");
    printf("inter_cache_0[0..7]: ");
    for(int i=0;i<8;i++) printf("%d ", st.inter_cache_0[i]);
    printf("\n");

    /* Run up to RNN1 */
    int32_t rq[257], iq[257];
    for (int i = 0; i < 257; i++) {
        rq[i] = (int32_t)round(sr[i] * 1048576.0f);
        iq[i] = (int32_t)round(si[i] * 1048576.0f);
    }

    int32_t x_log[257];
    log_gen_fixed(rq, iq, 257, x_log);

    int32_t x_bm[129];
    bm_fixed(x_log, erb_erb_fc_weight, 257, 129, x_bm);

    int32_t e0[12*65], e1[24*33], e2[24*33], e3[32*33], e4[16*33];
    encoder_module(x_bm, &st, e0, e1, e2, e3, e4);

    printf("=== After encoder, before RNN ===\n");
    printf("inter_cache_0[0..7]: ");
    for(int i=0;i<8;i++) printf("%d ", st.inter_cache_0[i]);
    printf("\n");

    /* Check what the RNN expects: gdprnn_module receives inter_cache_0 as int16_t* */
    printf("\n=== Before RNN1 ===\n");
    printf("inter_cache_0 first 8 values (int16_t): ");
    for(int i=0;i<8;i++) printf("%d ", st.inter_cache_0[i]);
    printf("\n");
    printf("inter_cache_0 last 8 values (int16_t): ");
    for(int i=520;i<528;i++) printf("%d ", st.inter_cache_0[i]);
    printf("\n");

    /* Run RNN1 */
    int32_t r1[16*33];
    gdprnn_module(e4, st.inter_cache_0, 0, r1);

    printf("\n=== After RNN1 ===\n");
    printf("inter_cache_0 first 8 values (int16_t): ");
    for(int i=0;i<8;i++) printf("%d ", st.inter_cache_0[i]);
    printf("\n");
    printf("inter_cache_0 last 8 values (int16_t): ");
    for(int i=520;i<528;i++) printf("%d ", st.inter_cache_0[i]);
    printf("\n");

    /* Check for values near int16_t limits */
    int clip_count = 0, max_val = 0;
    for(int i=0;i<528;i++) {
        int v = abs(st.inter_cache_0[i]);
        if (v > max_val) max_val = v;
        if (v >= 32767) clip_count++;
    }
    printf("inter_cache_0 after RNN1: max_abs=%d, clipped(>=32767)=%d\n", max_val, clip_count);

    /* Check RNN1 output range */
    printf("\n=== RNN1 output ===\n");
    int r1_max = 0, r1_clip = 0;
    for(int i=0;i<16*33;i++) {
        int v = abs(r1[i]);
        if (v > r1_max) r1_max = v;
        if (v >= 2147483647) r1_clip++;
    }
    printf("RNN1 output: max_abs=%d (Q20: %.2f), int32_clipped=%d\n",
           r1_max, (double)r1_max/1048576.0, r1_clip);

    /* Show E4 input to RNN */
    printf("\n=== E4 (RNN input) stats ===\n");
    int e4_max = 0;
    for(int i=0;i<16*33;i++) if(abs(e4[i]) > e4_max) e4_max = abs(e4[i]);
    printf("E4 max_abs=%d (Q20: %.4f)\n", e4_max, (double)e4_max/1048576.0);

    /* Check tfa_cache — did encoder write int32_t through int16_t pointer? */
    printf("\n=== tfa_cache after encoder ===\n");
    printf("tfa_cache_e0 (int16_t, max 32767): first 8 = ");
    for(int i=0;i<8;i++) printf("%d ", st.tfa_cache_e0[i]);
    int tfa_max=0; for(int i=0;i<24;i++) if(abs(st.tfa_cache_e0[i])>tfa_max) tfa_max=abs(st.tfa_cache_e0[i]);
    printf(" max_abs=%d\n", tfa_max);

    /* Read tfa_cache as int32_t to detect write mismatch */
    int32_t *tfa32 = (int32_t*)st.tfa_cache_e0;
    printf("tfa_cache_e0 as int32_t[0..5]: %d %d %d %d %d %d\n",
           tfa32[0], tfa32[1], tfa32[2], tfa32[3], tfa32[4], tfa32[5]);
    printf("(if values >> 32767, int32_t wrote through int16_t pointer)\n");

    /* Also check inter_cache as int32_t */
    printf("\n=== inter_cache as int32_t ===\n");
    int32_t *ic32 = (int32_t*)st.inter_cache_0;
    printf("First 4 as int32_t: %d %d %d %d\n", ic32[0], ic32[1], ic32[2], ic32[3]);

    free(sr); free(si);
    return 0;
}
