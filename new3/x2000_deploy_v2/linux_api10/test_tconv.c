#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_lut.h"
#include "ulunas_matlab_weights.h"

static double snr_i32(const int32_t *g, const int32_t *t, int n) {
    double sn=0,sd=0;
    for(int i=0;i<n;i++){double gv=g[i],tv=t[i];sn+=gv*gv;sd+=(gv-tv)*(gv-tv);}
    return sd<1e-30?999:10*log10(sn/sd);
}
static int32_t* load_t(const char *p, int rows, int cols) {
    FILE *f=fopen(p,"rb");if(!f)return NULL;
    int n=rows*cols,*tmp=malloc(n*4),*d=malloc(n*4);
    fread(tmp,4,n,f);fclose(f);
    if(rows==1)memcpy(d,tmp,n*4);
    else for(int r=0;r<rows;r++)for(int c=0;c<cols;c++)d[r*cols+c]=tmp[c*rows+r];
    free(tmp);return d;
}
int main() {
    /* Load BM golden as input */
    int32_t *bm=load_t("dump_matlab/frame0_bm.bin",1,129);
    /* Load E0 tconv golden (12×65, before cTFA) */
    int32_t *gtconv=load_t("dump_matlab/frame0_enc_e0_tconv.bin",12,65);
    if(!bm||!gtconv){printf("FATAL\n");return 1;}

    /* Run encoder_xconv_module to get tconv output */
    /* encoder_xconv_module(bm, conv_cache, tfa_cache, y_e0) */
    int32_t cc[2*129]={0}, tc[24]={0}, e0[12*65];
    encoder_xconv_module(bm, cc, tc, e0);  /* e0 is post-cTFA */

    /* We need the TConv output before cTFA. But encoder_xconv_module
       only outputs the final e0. Let's compare what we can. */
    printf("E0 tconv SNR: %.2f dB\n", snr_i32(gtconv, e0, 12*65));

    /* Also compare E0 final (should match previous test) */
    int32_t *ge0=load_t("dump_matlab/frame0_enc_e0.bin",12,65);
    if(ge0) printf("E0 final SNR: %.2f dB\n", snr_i32(ge0, e0, 12*65));
    free(bm);free(gtconv);free(ge0);
    return 0;
}
