#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_lut.h"
#include "qr_config.h"
#include "layer_dims.h"
#include "ulunas_matlab_weights.h"

static float *lf(const char *p, int n) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    float *b = (float*)malloc(n*4); fread(b,4,n,f); fclose(f); return b;
}

int main() {
    float *sr = lf("dump_matlab/frame0_stft_real.bin", 257);
    float *si = lf("dump_matlab/frame0_stft_imag.bin", 257);
    if (!sr || !si) { printf("No input
"); return 1; }

    ulunas_state_t st;
    ulunas_state_init(&st);

    int32_t rq[257], iq[257];
    for (int i = 0; i < 257; i++) {
        rq[i] = (int32_t)round(sr[i] * 1048576.0f);
        iq[i] = (int32_t)round(si[i] * 1048576.0f);
    }

    int32_t x_log[257], x_bm[129];
    log_gen_fixed(rq, iq, 257, x_log);
    bm_fixed(x_log, erb_erb_fc_weight, 257, 129, x_bm);

    int32_t e0[12*65], e1[24*33], e2[24*33], e3[32*33], e4[16*33];
    encoder_module(x_bm, &st, e0, e1, e2, e3, e4);

    // Check tfa_cache after encoder
    printf("=== tfa_cache after encoder (should be GRU hidden states) ===
");
    printf("tfa_cache_e0 (int16_t, max 32767): first 8 = ");
    for(int i=0;i<8;i++) printf("%d ", st.tfa_cache_e0[i]);
    int max_e0=0; for(int i=0;i<24;i++) if(abs(st.tfa_cache_e0[i])>max_e0) max_e0=abs(st.tfa_cache_e0[i]);
    printf(" max_abs=%d
", max_e0);

    printf("tfa_cache_e1: max_abs=");
    int max_e1=0; for(int i=0;i<48;i++) if(abs(st.tfa_cache_e1[i])>max_e1) max_e1=abs(st.tfa_cache_e1[i]);
    printf("%d
", max_e1);

    printf("tfa_cache_e2: max_abs=");
    int max_e2=0; for(int i=0;i<48;i++) if(abs(st.tfa_cache_e2[i])>max_e2) max_e2=abs(st.tfa_cache_e2[i]);
    printf("%d
", max_e2);

    // Check what type tfa_cache_e0 actually is
    printf("
sizeof(tfa_cache_e0) = %zu, sizeof(element) = %zu
",
           sizeof(st.tfa_cache_e0), sizeof(st.tfa_cache_e0[0]));

    // Now check: does ctfa_ta_module write int32_t through int16_t pointer?
    // The ctfa_ta_module receives int16_t* h_cache but internally gru_module writes int32_t?
    printf("
=== Checking for int32_t writes through int16_t pointer ===
");
    int32_t *as_i32 = (int32_t*)st.tfa_cache_e0;
    printf("tfa_cache_e0 as int32_t[0..3]: %d %d %d %d
",
           as_i32[0], as_i32[1], as_i32[2], as_i32[3]);
    printf("If these are huge, int32_t was written through int16_t*
");

    free(sr); free(si);
    return 0;
}
