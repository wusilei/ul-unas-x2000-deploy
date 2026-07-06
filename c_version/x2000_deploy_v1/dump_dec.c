#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define QR_CALIBRATION_MODE
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"
#include "ulunas_ctfa_qr.h"
static int32_t *load_int32(const char *path, int n) {
    FILE *f = fopen(path, "rb"); if(!f) return NULL;
    int32_t *b = malloc(n*sizeof(int32_t));
    if(fread(b,sizeof(int32_t),n,f)!=(size_t)n){free(b);fclose(f);return NULL;}
    fclose(f); return b;
}
static float *load_float(const char *path, int n) {
    FILE *f = fopen(path, "rb"); if(!f) return NULL;
    float *b = malloc(n*sizeof(float));
    if(fread(b,sizeof(float),n,f)!=(size_t)n){free(b);fclose(f);return NULL;}
    fclose(f); return b;
}
int main() {
    float *real_in = load_float("dump_matlab/frame0_stft_real.bin", 257);
    float *imag_in = load_float("dump_matlab/frame0_stft_imag.bin", 257);
    int32_t x_log[1*257]; log_gen_fixed(real_in,imag_in,257,x_log);
    int32_t x_bm[1*129]; BM_fixed(x_log,erb_erb_fc_weight,x_bm);
    
    ulunas_state_t st; ulunas_state_init(&st);
    int32_t e0[12*65],e1[24*33],e2[24*33],e3[32*33],e4[16*33];
    Encoder_module(x_bm,&st,e0,e1,e2,e3,e4);
    int32_t r1[16*33],r2[16*33];
    GDPRNN_module(e4,st.inter_prev0,0,r1);
    GDPRNN_module(r1,st.inter_prev1,1,r2);
    
    int32_t y_dec[1*129];
    Decoder_module(r2,&st,e0,e1,e2,e3,e4,y_dec);
    
    /* Print first 10 decoder output values */
    printf("dec output first 10:");
    for(int i=0;i<10;i++)printf(" %d",y_dec[i]);
    printf("\n");
    
    int32_t *gd = load_int32("dump_matlab/frame0_dec.bin", 1*129);
    if(gd){
        printf("golden first 10:");
        for(int i=0;i<10;i++)printf(" %d",gd[i]);
        printf("\n");
        double s=0,e=0;
        for(int i=0;i<129;i++){double gv=gd[i],d=gv-y_dec[i];s+=gv*gv;e+=d*d;}
        printf("SNR=%.2f dB\n",e<1e-30?999:10*log10(s/e));
        free(gd);
    }
    free(real_in);free(imag_in);
    return 0;
}
