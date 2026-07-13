/* Verify GRU algorithm: dequantize weights to double → run in double precision → compare SNR.
 * Proves C GRU algorithm = MATLAB, isolated from int16 weight quantization. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_matlab_weights.h"

static double sigmoid(double x) { return 1.0/(1.0+exp(-x)); }
static double tanh_d(double x) { return tanh(x); }

static double snr_db(const double *g, const double *t, int n) {
    double sn=0,sd=0;
    for(int i=0;i<n;i++){sn+=g[i]*g[i];sd+=(g[i]-t[i])*(g[i]-t[i]);}
    return sd<1e-30?999:10*log10(sn/sd);
}
#define R(lbl,g,t,n) do{double s=snr_db(g,t,n);printf("  %-40s %8.2f dB  %s\n",lbl,s,s>=80?"PASS":s>=30?"WARN":"FAIL");}while(0)

/* Load int32 golden: col-major → row-major transpose */
static double* load32_t(const char *p, int rows, int cols) {
    FILE *f=fopen(p,"rb");if(!f)return NULL;
    int n=rows*cols;
    int32_t *tmp=malloc(n*4);fread(tmp,4,n,f);fclose(f);
    double *d=malloc(n*8);
    if(rows==1)for(int i=0;i<n;i++)d[i]=tmp[i]/1048576.0;
    else for(int r=0;r<rows;r++)for(int c=0;c<cols;c++)d[r*cols+c]=tmp[c*rows+r]/1048576.0;
    free(tmp);return d;
}
static double* load16_t(const char *p, int rows, int cols) {
    FILE *f=fopen(p,"rb");if(!f)return NULL;
    int n=rows*cols;
    int16_t *tmp=malloc(n*2);fread(tmp,2,n,f);fclose(f);
    double *d=malloc(n*8);
    /* Keep as raw Q15 integer values for SNR comparison (not normalized) */
    if(rows==1)for(int i=0;i<n;i++)d[i]=tmp[i];
    else for(int r=0;r<rows;r++)for(int c=0;c<cols;c++)d[r*cols+c]=tmp[c*rows+r];
    free(tmp);return d;
}

/* Dequantize int16 weights to double */
static void dq_w16(const int16_t *w, int n, int Q, double *out) {
    double s=pow(2.0,Q);
    for(int i=0;i<n;i++) out[i]=w[i]/s;
}
static void dq_b32(const int32_t *b, int n, int Q, double *out) {
    double s=pow(2.0,Q);
    for(int i=0;i<n;i++) out[i]=b[i]/s;
}

/* GRU step in double precision (matching MATLAB GRU_module.m).
 * x_t: Q20 actual values (from load32_t: Q20_int / 2^20).
 * h_cache: Q15 actual values (0..32767 range).
 * ih_w/hh_w: Q12 actual values (weight_int / 2^12).
 * ih_b/hh_b: Q20 actual values (bias_int / 2^20).
 *
 * Scale factors for round(x * w * 2^Qr):
 *   ih path: x(Q20) * ih_w(Q12) = Q32, round(*2^(-13)) → need 2^19 for actual values
 *   hh path: h(Q15) * hh_w(Q12) = Q27, round(*2^(-8)) → need 2^19 for actual values */
static void gru_double(const double *x_t, int nHidden, int in_dim,
                       double *h_cache,
                       const double *ih_w, const double *ih_b,
                       const double *hh_w, const double *hh_b,
                       double Qr1, double Qr2, double *y) {
    /* For actual values: x*w is int_product/2^(20+12), need to match round(int_prod * 2^Qr) */
    double s1 = pow(2.0, 20 + 12 + Qr1);  /* 2^19 for Qr1=-13 */
    double s2 = pow(2.0, 15 + 12 + Qr2);  /* 2^19 for Qr2=-8 */
    (void)Qr1; (void)Qr2;

    /* Reset gate R */
    for(int j=0;j<nHidden;j++){
        double sum_ih=0,sum_hh=0;
        for(int i=0;i<in_dim;i++) sum_ih+=x_t[i]*ih_w[i+in_dim*j];
        for(int i=0;i<nHidden;i++) sum_hh+=h_cache[i]*hh_w[i+nHidden*j];
        y[j]=round(sum_ih*s1)+round(sum_hh*s2)+ih_b[j]+hh_b[j];
    }
    for(int j=0;j<nHidden;j++){
        y[j]=sigmoid(y[j]*pow(2.0,-20))*32768.0; /* Q15 output */
    }
    double r_t[64]; memcpy(r_t,y,nHidden*8);

    /* Update gate Z */
    for(int j=0;j<nHidden;j++){
        double sum_ih=0,sum_hh=0;
        for(int i=0;i<in_dim;i++) sum_ih+=x_t[i]*ih_w[i+in_dim*(nHidden+j)];
        for(int i=0;i<nHidden;i++) sum_hh+=h_cache[i]*hh_w[i+nHidden*(nHidden+j)];
        y[j]=round(sum_ih*s1)+round(sum_hh*s2)+ih_b[nHidden+j]+hh_b[nHidden+j];
    }
    for(int j=0;j<nHidden;j++){
        y[j]=sigmoid(y[j]*pow(2.0,-20))*32768.0;
    }
    double z_t[64]; memcpy(z_t,y,nHidden*8);

    /* Candidate hidden H */
    double h_t[64];
    for(int j=0;j<nHidden;j++){
        double sum_hh=0;
        for(int i=0;i<nHidden;i++) sum_hh+=h_cache[i]*hh_w[i+nHidden*(2*nHidden+j)];
        h_t[j]=round(sum_hh*s2)+hh_b[2*nHidden+j];
    }
    for(int j=0;j<nHidden;j++){
        double sum_ih=0;
        for(int i=0;i<in_dim;i++) sum_ih+=x_t[i]*ih_w[i+in_dim*(2*nHidden+j)];
        y[j]=round(sum_ih*s1)+round(r_t[j]*h_t[j]/32768.0)+ih_b[2*nHidden+j];
    }
    for(int j=0;j<nHidden;j++){
        y[j]=tanh_d(y[j]*pow(2.0,-20))*32768.0; /* Q15 */
    }
    double n_t[64]; memcpy(n_t,y,nHidden*8);

    /* Hidden update: h = round((32768-z).*n/32768) + round(z.*h_prev/32768) */
    for(int j=0;j<nHidden;j++){
        double t1=round((32768.0-z_t[j])*n_t[j]/32768.0);
        double t2=round(z_t[j]*h_cache[j]/32768.0);
        h_cache[j]=fmax(-32768,fmin(32767,t1+t2));
        y[j]=h_cache[j];
    }
}

int main() {
    const char *d="dump_matlab";
    char b[512]; int T=33;

    printf("=== GRU Double-Precision Verification ===\n\n");

    /* Load intra_in golden */
    snprintf(b,sizeof(b),"%s/frame0_rnn1_intra_in.bin",d);
    double *gin=load32_t(b,T,16);
    if(!gin){printf("FATAL\n");return 1;}

    /* Dequantize weights */
    /* Group 0: forward + reverse weights */
    double ih_w0[8*12], ih_b0[12], hh_w0[4*12], hh_b0[12];
    dq_w16(dpgrnn_0_intra_rnn_rnn1_weight_ih_l0, 8*12, 12, ih_w0);
    dq_b32(dpgrnn_0_intra_rnn_rnn1_bias_ih_l0, 12, 20, ih_b0);
    dq_w16(dpgrnn_0_intra_rnn_rnn1_weight_hh_l0, 4*12, 12, hh_w0);
    dq_b32(dpgrnn_0_intra_rnn_rnn1_bias_hh_l0, 12, 20, hh_b0);
    double rih_w0[8*12], rih_b0[12], rhh_w0[4*12], rhh_b0[12];
    dq_w16(dpgrnn_0_intra_rnn_rnn1_weight_ih_l0_reverse, 8*12, 12, rih_w0);
    dq_b32(dpgrnn_0_intra_rnn_rnn1_bias_ih_l0_reverse, 12, 20, rih_b0);
    dq_w16(dpgrnn_0_intra_rnn_rnn1_weight_hh_l0_reverse, 4*12, 12, rhh_w0);
    dq_b32(dpgrnn_0_intra_rnn_rnn1_bias_hh_l0_reverse, 12, 20, rhh_b0);

    /* Group 1: forward + reverse weights */
    double ih_w1[8*12], ih_b1[12], hh_w1[4*12], hh_b1[12];
    dq_w16(dpgrnn_0_intra_rnn_rnn2_weight_ih_l0, 8*12, 12, ih_w1);
    dq_b32(dpgrnn_0_intra_rnn_rnn2_bias_ih_l0, 12, 20, ih_b1);
    dq_w16(dpgrnn_0_intra_rnn_rnn2_weight_hh_l0, 4*12, 12, hh_w1);
    dq_b32(dpgrnn_0_intra_rnn_rnn2_bias_hh_l0, 12, 20, hh_b1);
    double rih_w1[8*12], rih_b1[12], rhh_w1[4*12], rhh_b1[12];
    dq_w16(dpgrnn_0_intra_rnn_rnn2_weight_ih_l0_reverse, 8*12, 12, rih_w1);
    dq_b32(dpgrnn_0_intra_rnn_rnn2_bias_ih_l0_reverse, 12, 20, rih_b1);
    dq_w16(dpgrnn_0_intra_rnn_rnn2_weight_hh_l0_reverse, 4*12, 12, rhh_w1);
    dq_b32(dpgrnn_0_intra_rnn_rnn2_bias_hh_l0_reverse, 12, 20, rhh_b1);

    /* Split input: x0 = cols 0..7, x1 = cols 8..15 */
    double x0[T*8], x1[T*8];
    for(int t=0;t<T;t++){for(int c=0;c<8;c++){x0[t*8+c]=gin[t*16+c];x1[t*8+c]=gin[t*16+8+c];}}

    /* ── Forward GRU group 0 ── */
    double h0_fwd[4]={0}, y0_fwd[T*4];
    for(int t=0;t<T;t++) gru_double(x0+t*8,4,8,h0_fwd,ih_w0,ih_b0,hh_w0,hh_b0,-13,-8,y0_fwd+t*4);

    /* ── Backward GRU group 0 ── */
    double h0_bwd[4]={0}, y0_bwd_raw[T*4];
    for(int t=T-1;t>=0;t--) gru_double(x0+t*8,4,8,h0_bwd,rih_w0,rih_b0,rhh_w0,rhh_b0,-13,-8,y0_bwd_raw+t*4);

    /* BiGRU output group 0: concat [fwd, bwd] */
    double g0_out[T*8];
    for(int t=0;t<T;t++){for(int c=0;c<4;c++){g0_out[t*8+c]=y0_fwd[t*4+c];g0_out[t*8+4+c]=y0_bwd_raw[t*4+c];}}

    /* Load golden gru0 */
    snprintf(b,sizeof(b),"%s/frame0_rnn1_intra_gru0.bin",d);
    double *gg0=load16_t(b,T,8);
    if(gg0){R("  gru0 BiGRU (double)",gg0,g0_out,T*8);free(gg0);}else printf("  gru0: SKIP\n");

    /* Repeat for group 1 with different weights */
    double h1_fwd[4]={0}, y1_fwd[T*4], h1_bwd[4]={0}, y1_bwd_raw[T*4];
    for(int t=0;t<T;t++) gru_double(x1+t*8,4,8,h1_fwd,ih_w1,ih_b1,hh_w1,hh_b1,-13,-8,y1_fwd+t*4);
    for(int t=T-1;t>=0;t--) gru_double(x1+t*8,4,8,h1_bwd,rih_w1,rih_b1,rhh_w1,rhh_b1,-13,-8,y1_bwd_raw+t*4);
    double g1_out[T*8];
    for(int t=0;t<T;t++){for(int c=0;c<4;c++){g1_out[t*8+c]=y1_fwd[t*4+c];g1_out[t*8+4+c]=y1_bwd_raw[t*4+c];}}

    snprintf(b,sizeof(b),"%s/frame0_rnn1_intra_gru1.bin",d);
    double *gg1=load16_t(b,T,8);
    if(gg1){R("  gru1 BiGRU (double)",gg1,g1_out,T*8);free(gg1);}else printf("  gru1: SKIP\n");

    /* Concat */
    double cat[T*16];
    for(int t=0;t<T;t++){for(int c=0;c<8;c++){cat[t*16+c]=g0_out[t*8+c];cat[t*16+8+c]=g1_out[t*8+c];}}

    snprintf(b,sizeof(b),"%s/frame0_rnn1_intra_cat.bin",d);
    double *gcat=load16_t(b,T,16);
    if(gcat){R("  cat (double)",gcat,cat,T*16);free(gcat);}else printf("  cat: SKIP\n");

    /* FC in double: cat(Q15 actual) * fc_w(Q13 actual) = Q28 in int, >>9.
     * Actual: cat_act * fc_w_act * 2^(15+13) = Q28 → round(*2^(-9)) → need 2^(15+13-9)=2^19 */
    double fc_w[256], fc_b[16];
    dq_w16(dpgrnn_0_intra_fc_weight,256,13,fc_w);
    dq_b32(dpgrnn_0_intra_fc_bias,16,20,fc_b);
    double fc_s = pow(2.0, 15 + 13 + (-9));  /* 2^19 */
    double fc[T*16];
    for(int t=0;t<T;t++) for(int co=0;co<16;co++){
        double s=0;
        for(int ci=0;ci<16;ci++) s+=cat[t*16+ci]*fc_w[ci+16*co];
        fc[t*16+co]=round(s*fc_s)+fc_b[co];
    }
    snprintf(b,sizeof(b),"%s/frame0_rnn1_intra_fc.bin",d);
    double *gfc=load32_t(b,T,16);
    if(gfc){R("  FC (double)",gfc,fc,T*16);free(gfc);}else printf("  FC: SKIP\n");

    /* LN in double */
    double ln_w[16], ln_b[16];
    dq_w16(dpgrnn_0_intra_ln_weight,16,12,ln_w);
    dq_b32(dpgrnn_0_intra_ln_bias,16,20,ln_b);
    double lmean=0; for(int i=0;i<T*16;i++) lmean+=fc[i]; lmean/=T*16;
    double lvar=0; for(int i=0;i<T*16;i++){double d=fc[i]-lmean;lvar+=d*d;} lvar=lvar/(T*16)+1e-8;
    double istd=1.0/sqrt(lvar);
    double ln_out[T*16];
    for(int i=0;i<T*16;i++){
        int c=i%16;
        double xn=(fc[i]-lmean)*istd;
        double xq=round(xn*1048576.0);  /* Q20 int */
        /* MATLAB: round(xq * weight_int * 2^(-14)) + bias_int
         * weight_int = ln_w_act * 2^12. /16384 * 2^12 = /4 */
        ln_out[i]=round(xq*ln_w[c]/4.0)+ln_b[c];
    }
    snprintf(b,sizeof(b),"%s/frame0_rnn1_intra_ln.bin",d);
    double *gln=load32_t(b,T,16);
    if(gln){R("  LN (double)",gln,ln_out,T*16);free(gln);}else printf("  LN: SKIP\n");

    free(gin);
    printf("\n=== Done ===\n");
    return 0;
}
