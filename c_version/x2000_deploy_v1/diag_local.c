#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define QR_CALIBRATION_MODE
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"
#include "ulunas_ctfa_qr.h"
static double snr_db(const int32_t *g, const int32_t *t, int n) {
    double s=0,e=0; for(int i=0;i<n;i++){double gv=g[i],d=gv-t[i];s+=gv*gv;e+=d*d;}
    return e<1e-30?999:10*log10(s/e);
}
static int32_t *L(const char *p, int n){FILE *f=fopen(p,"rb");int32_t *b=malloc(n*4);fread(b,4,n,f);fclose(f);return b;}
static float *LF(const char *p, int n){FILE *f=fopen(p,"rb");float *b=malloc(n*4);fread(b,4,n,f);fclose(f);return b;}
int main() {
    float *ri=LF("dump_matlab/frame0_stft_real.bin",257),*ii=LF("dump_matlab/frame0_stft_imag.bin",257);
    int32_t xl[257]; log_gen_fixed(ri,ii,257,xl);
    int32_t xb[129]; BM_fixed(xl,erb_erb_fc_weight,xb);
    ulunas_state_t st; ulunas_state_init(&st);
    int32_t e0[12*65],e1[24*33],e2[24*33],e3[32*33],e4[16*33];
    Encoder_module(xb,&st,e0,e1,e2,e3,e4);
    int32_t r1[16*33],r2[16*33];
    GDPRNN_module(e4,st.inter_prev0,0,r1);
    GDPRNN_module(r1,st.inter_prev1,1,r2);
    int32_t yd[1*129];
    Decoder_module(r2,&st,e0,e1,e2,e3,e4,yd);
    int32_t *gd=L("dump_matlab/frame0_dec.bin",129);
    printf("Local diag dec SNR = %.2f dB\n", snr_db(yd,gd,129));
    free(ri);free(ii);free(gd); return 0;
}
