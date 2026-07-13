/* RNN1 sub-step diagnostic: pinpoint SNR degradation source */
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
static double snr_i16(const int16_t *g, const int16_t *t, int n) {
    double sn=0,sd=0;
    for(int i=0;i<n;i++){double gv=g[i],tv=t[i];sn+=gv*gv;sd+=(gv-tv)*(gv-tv);}
    return sd<1e-30?999:10*log10(sn/sd);
}

static int32_t* load32_t(const char *p, int rows, int cols) {
    FILE *f=fopen(p,"rb");if(!f)return NULL;
    int n=rows*cols,*tmp=malloc(n*4),*d=malloc(n*4);
    fread(tmp,4,n,f);fclose(f);
    if(rows==1)memcpy(d,tmp,n*4);
    else for(int r=0;r<rows;r++)for(int c=0;c<cols;c++)d[r*cols+c]=tmp[c*rows+r];
    free(tmp);return d;
}
static int16_t* load16_t(const char *p, int rows, int cols) {
    FILE *f=fopen(p,"rb");if(!f)return NULL;
    int n=rows*cols;
    int16_t *tmp=(int16_t*)malloc(n*2),*d=(int16_t*)malloc(n*2);
    fread(tmp,2,n,f);fclose(f);
    if(rows==1)memcpy(d,tmp,n*2);
    else for(int r=0;r<rows;r++)for(int c=0;c<cols;c++)d[r*cols+c]=tmp[c*rows+r];
    free(tmp);return d;
}
#define R32(lbl,g,t,n) do{double s=snr_i32(g,t,n);printf("  %-30s %8.2f dB  %s\n",lbl,s,s>=80?"PASS":s>=30?"WARN":"FAIL");}while(0)
#define R16(lbl,g,t,n) do{double s=snr_i16(g,t,n);printf("  %-30s %8.2f dB  %s\n",lbl,s,s>=80?"PASS":s>=30?"WARN":"FAIL");}while(0)

int main() {
    const char *d="dump_matlab";
    char b[512];
    int T=33;

    printf("=== RNN1 Sub-Step Diagnostic ===\n\n");

    /* ── Load intra_in golden (33,16) int32 Q20 ── */
    snprintf(b,sizeof(b),"%s/frame0_rnn1_intra_in.bin",d);
    printf("Loading %s...\n",b);
    int32_t *gin=load32_t(b, T, 16);
    if(!gin){printf("FATAL: %s\n",b);return 1;}
    printf("Loaded.\n");
    printf("─── Intra RNN ───\n");

    /* Split into groups: x0 = columns 0..7, x1 = columns 8..15 */
    int32_t x0[T*8], x1[T*8];
    for(int t=0;t<T;t++){
        for(int c=0;c<8;c++){x0[t*8+c]=gin[t*16+c];x1[t*8+c]=gin[t*16+8+c];}
    }

    /* ── BiGRU group 0 ── */
    int nH=4, in_dim=8;
    int16_t g0[T*8];
    bigru_module(x0, T, nH, in_dim,
        dpgrnn_0_intra_rnn_rnn1_weight_ih_l0, dpgrnn_0_intra_rnn_rnn1_bias_ih_l0,
        dpgrnn_0_intra_rnn_rnn1_weight_hh_l0, dpgrnn_0_intra_rnn_rnn1_bias_hh_l0,
        dpgrnn_0_intra_rnn_rnn1_weight_ih_l0_reverse, dpgrnn_0_intra_rnn_rnn1_bias_ih_l0_reverse,
        dpgrnn_0_intra_rnn_rnn1_weight_hh_l0_reverse, dpgrnn_0_intra_rnn_rnn1_bias_hh_l0_reverse,
        -13, -8, g0);

    snprintf(b,sizeof(b),"%s/frame0_rnn1_intra_gru0.bin",d);
    int16_t *gg0=load16_t(b, T, 8);
    if(gg0){R16("  gru0 BiGRU", gg0, g0, T*8);free(gg0);}else printf("  gru0: SKIP\n");

    /* ── BiGRU group 1 ── */
    int16_t g1[T*8];
    bigru_module(x1, T, nH, in_dim,
        dpgrnn_0_intra_rnn_rnn2_weight_ih_l0, dpgrnn_0_intra_rnn_rnn2_bias_ih_l0,
        dpgrnn_0_intra_rnn_rnn2_weight_hh_l0, dpgrnn_0_intra_rnn_rnn2_bias_hh_l0,
        dpgrnn_0_intra_rnn_rnn2_weight_ih_l0_reverse, dpgrnn_0_intra_rnn_rnn2_bias_ih_l0_reverse,
        dpgrnn_0_intra_rnn_rnn2_weight_hh_l0_reverse, dpgrnn_0_intra_rnn_rnn2_bias_hh_l0_reverse,
        -13, -8, g1);

    snprintf(b,sizeof(b),"%s/frame0_rnn1_intra_gru1.bin",d);
    int16_t *gg1=load16_t(b, T, 8);
    if(gg1){R16("  gru1 BiGRU", gg1, g1, T*8);free(gg1);}else printf("  gru1: SKIP\n");

    /* ── Concat ── */
    int16_t cat[T*16];
    for(int t=0;t<T;t++){for(int c=0;c<8;c++){cat[t*16+c]=g0[t*8+c];cat[t*16+8+c]=g1[t*8+c];}}

    snprintf(b,sizeof(b),"%s/frame0_rnn1_intra_cat.bin",d);
    int16_t *gcat=load16_t(b, T, 16);
    if(gcat){R16("  cat", gcat, cat, T*16);free(gcat);}else printf("  cat: SKIP\n");

    /* ── FC: per-product shift + accumulate, bias added AFTER ── */
    int32_t fc[T*16];
    for(int t=0;t<T;t++){
        for(int co=0;co<16;co++){
            int64_t acc=0;
            for(int ci=0;ci<16;ci++){
                int64_t prod=(int64_t)(int32_t)cat[t*16+ci]*dpgrnn_0_intra_fc_weight[ci+16*co];
                if(prod>=0) acc+=(prod+256)>>9;
                else        acc+=(prod-256)>>9;
            }
            fc[t*16+co]=sat_i32(acc+dpgrnn_0_intra_fc_bias[co]);
        }
    }
    snprintf(b,sizeof(b),"%s/frame0_rnn1_intra_fc.bin",d);
    int32_t *gfc=load32_t(b, T, 16);
    if(gfc){
        printf("  C FC[0..3]: %d %d %d %d\n", fc[0],fc[1],fc[2],fc[3]);
        printf("  G FC[0..3]: %d %d %d %d\n", gfc[0],gfc[1],gfc[2],gfc[3]);
        R32("  FC (qr=-9)", gfc, fc, T*16);free(gfc);
    }else printf("  FC: SKIP\n");

    /* ── LN (qr=-14) on FC output ── */
    int32_t ln_out[T*16];
    ln_func(fc, dpgrnn_0_intra_ln_weight, dpgrnn_0_intra_ln_bias, -14, 16, T*16, ln_out);

    snprintf(b,sizeof(b),"%s/frame0_rnn1_intra_ln.bin",d);
    int32_t *gln=load32_t(b, T, 16);
    if(gln){
        R32("  LN (qr=-14)", gln, ln_out, T*16);
        /* Double-precision LN reference */
        double dvals[T*16];
        for(int i=0;i<T*16;i++) dvals[i] = fc[i] / 1048576.0;
        double dmean=0; for(int i=0;i<T*16;i++) dmean+=dvals[i]; dmean/=(T*16);
        double dvar=0; for(int i=0;i<T*16;i++){double d=dvals[i]-dmean;dvar+=d*d;} dvar=dvar/(T*16)+1e-8;
        double istd = 1.0/sqrt(dvar);
        double s=0,sd=0;
        for(int i=0;i<T*16;i++){
            int c=i%16;
            double xnorm = (dvals[i]-dmean)*istd;
            int32_t xn_q = (int32_t)round(xnorm * 1048576.0);
            double sc = xnorm * (dpgrnn_0_intra_ln_weight[c]/16384.0);
            int32_t ys = (int32_t)round(sc * 16384.0);
            int32_t ref_val = ys + dpgrnn_0_intra_ln_bias[c];
            double gv=gln[i],tv=ref_val;s+=gv*gv;sd+=(gv-tv)*(gv-tv);
        }
        printf("  LN (double ref)                 %8.2f dB\n", sd<1e-30?999:10*log10(s/sd));
        free(gln);
    }else printf("  LN: SKIP\n");

    /* ── Residual: intra_out = intra_in + ln ── */
    int32_t iout[T*16];
    for(int i=0;i<T*16;i++) iout[i]=sat_i32((int64_t)gin[i]+ln_out[i]);

    snprintf(b,sizeof(b),"%s/frame0_rnn1_intra_out.bin",d);
    int32_t *gio=load32_t(b, T, 16);
    if(gio){R32("  intra_out (residual)", gio, iout, T*16);free(gio);}else printf("  intra_out: SKIP\n");

    free(gin);
    printf("\n=== Done ===\n");
    return 0;
}
