/**
 * test_matlab_golden.c — Layer-by-Layer SNR Verification
 * =======================================================
 * Build: make pc
 * Run:   ./test_matlab_golden dump_matlab/
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"

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
static double snr_db(const int32_t *g, const int32_t *t, int n) {
    double s=0,e=0;
    for(int i=0;i<n;i++){double gv=g[i],d=gv-t[i];s+=gv*gv;e+=d*d;}
    if(e<1e-30)return 999.0;
    return 10.0*log10(s/e);
}
static const char *ok(double s){return s>120?"PERFECT":s>80?"PASS":s>40?"WARN":"FAIL";}

/* SNR for 2D layers: transpose golden from MATLAB col-major (W,C) to C row-major (C,W) */
static double snr_db_2d(const int32_t *golden_wc, const int32_t *c_out_cw, int C, int W) {
    int n = C * W;
    int32_t *gT = malloc(n * sizeof(int32_t));
    for (int c = 0; c < C; c++)
        for (int w = 0; w < W; w++)
            gT[c * W + w] = golden_wc[c + C * w];  /* col-major (C,W) → row-major (C,W) */
    double s = snr_db(gT, c_out_cw, n);
    free(gT);
    return s;
}

/* Per-channel SNR for 2D layers: returns SNR dB for each channel */
static void snr_db_per_channel(const int32_t *golden_wc, const int32_t *c_out_cw,
                                int C, int W, double *snr_out) {
    int32_t *gT = malloc(W * sizeof(int32_t));
    for (int c = 0; c < C; c++) {
        for (int w = 0; w < W; w++)
            gT[w] = golden_wc[c + C * w];  /* col-major: golden[c,w] = golden[c + C*w] */
        snr_out[c] = snr_db(gT, c_out_cw + c * W, W);
    }
    free(gT);
}

int main(int argc, char **argv) {
    const char *dir = argc>1?argv[1]:"dump_matlab/";
    char p[512];
    int tested=0,passed=0;

    printf("=== UL-UNAS Layer SNR (golden col-major → transposed) ===\n\n");

    for(int f=0;f<5;f++){
        /* Load STFT input */
        snprintf(p,sizeof(p),"%s/frame%d_stft_real.bin",dir,f);
        float *real_in=load_float(p,257);
        snprintf(p,sizeof(p),"%s/frame%d_stft_imag.bin",dir,f);
        float *imag_in=load_float(p,257);
        if(!real_in||!imag_in){printf("Frame %d: no STFT input\n",f);free(real_in);free(imag_in);continue;}

        printf("--- Frame %d ---\n",f);

        /* log_gen */
        int32_t x_log[1*257];
        log_gen_fixed(real_in,imag_in,257,x_log);

        /* BM */
        int32_t x_bm[1*129]; BM_fixed(x_log,erb_erb_fc_weight,x_bm);
        snprintf(p,sizeof(p),"%s/frame%d_bm.bin",dir,f);
        int32_t *gbm=load_int32(p,1*129);
        if(gbm){double s=snr_db(gbm,x_bm,129);printf("  BM      : SNR=%7.2f dB [%s]\n",s,ok(s));tested++;if(s>80)passed++;free(gbm);}

        /* Encoder */
        ulunas_state_t st; ulunas_state_init(&st);
        int32_t e0[12*65],e1[24*33],e2[24*33],e3[32*33],e4[16*33];
        Encoder_module(x_bm,&st,e0,e1,e2,e3,e4);

        const char *en[]={"enc_e0","enc_e1","enc_e2","enc_e3","enc_e4"};
        int32_t *eo[]={e0,e1,e2,e3,e4};
        int eC[]={12,24,24,32,16};
        int eW[]={65,33,33,33,33};
        for(int s=0;s<5;s++){
            snprintf(p,sizeof(p),"%s/frame%d_%s.bin",dir,f,en[s]);
            int32_t *g=load_int32(p,eC[s]*eW[s]);
            if(g){double sn=snr_db_2d(g,eo[s],eC[s],eW[s]);printf("  %-7s: SNR=%7.2f dB [%s]\n",en[s],sn,ok(sn));tested++;if(sn>80)passed++;free(g);}
            else printf("  %-7s: SKIP\n",en[s]);
        }

        /* ── Per-channel SNR: e2 (24ch × 33W) ── */
        {
            snprintf(p,sizeof(p),"%s/frame%d_enc_e2.bin",dir,f);
            int32_t *g2=load_int32(p,24*33);
            if(g2){
                double ch_snr[24];
                snr_db_per_channel(g2, e2, 24, 33, ch_snr);
                printf("\n  e2 per-channel SNR:\n");
                for(int r=0;r<6;r++){
                    printf("    ch %02d-%02d:", r*4, r*4+3);
                    for(int c=r*4;c<r*4+4 && c<24;c++)
                        printf(" %+6.1f", ch_snr[c]);
                    printf("\n");
                }
                free(g2);
            }
        }

        /* ── Per-channel SNR: e3 (32ch × 33W) ── */
        {
            snprintf(p,sizeof(p),"%s/frame%d_enc_e3.bin",dir,f);
            int32_t *g3=load_int32(p,32*33);
            if(g3){
                double ch_snr[32];
                snr_db_per_channel(g3, e3, 32, 33, ch_snr);
                printf("\n  e3 per-channel SNR:\n");
                for(int r=0;r<8;r++){
                    printf("    ch %02d-%02d:", r*4, r*4+3);
                    for(int c=r*4;c<r*4+4 && c<32;c++)
                        printf(" %+6.1f", ch_snr[c]);
                    printf("\n");
                }
                free(g3);
            }
        }

        /* ── Per-channel SNR breakdown: group 0 vs group 1 ── */
        printf("\n  Group summary:\n");
        {
            double ch_snr[32];  /* max channels */
            /* e2: 24ch, group0=ch0-11, group1=ch12-23 */
            snprintf(p,sizeof(p),"%s/frame%d_enc_e2.bin",dir,f);
            int32_t *g2=load_int32(p,24*33);
            if(g2){
                snr_db_per_channel(g2, e2, 24, 33, ch_snr);
                double g0=0,g1=0; int n0=0,n1=0;
                for(int c=0;c<12;c++){g0+=ch_snr[c];n0++;}
                for(int c=12;c<24;c++){g1+=ch_snr[c];n1++;}
                double min_snr=ch_snr[0],max_snr=ch_snr[0];
                for(int c=1;c<24;c++){if(ch_snr[c]<min_snr)min_snr=ch_snr[c];if(ch_snr[c]>max_snr)max_snr=ch_snr[c];}
                printf("  e2: group0(0-11)=%+.1f  group1(12-23)=%+.1f  range=[%+.1f, %+.1f]  spread=%.1f dB\n",
                    g0/n0, g1/n1, min_snr, max_snr, max_snr-min_snr);
                free(g2);
            }
            /* e3: 32ch, group0=ch0-15, group1=ch16-31 */
            snprintf(p,sizeof(p),"%s/frame%d_enc_e3.bin",dir,f);
            int32_t *g3=load_int32(p,32*33);
            if(g3){
                snr_db_per_channel(g3, e3, 32, 33, ch_snr);
                double g0=0,g1=0; int n0=0,n1=0;
                for(int c=0;c<16;c++){g0+=ch_snr[c];n0++;}
                for(int c=16;c<32;c++){g1+=ch_snr[c];n1++;}
                double min_snr=ch_snr[0],max_snr=ch_snr[0];
                for(int c=1;c<32;c++){if(ch_snr[c]<min_snr)min_snr=ch_snr[c];if(ch_snr[c]>max_snr)max_snr=ch_snr[c];}
                printf("  e3: group0(0-15)=%+.1f  group1(16-31)=%+.1f  range=[%+.1f, %+.1f]  spread=%.1f dB\n",
                    g0/n0, g1/n1, min_snr, max_snr, max_snr-min_snr);
                free(g3);
            }
        }

        /* GDPRNN */
        int32_t r1[16*33],r2[16*33];
        GDPRNN_module(e4,st.inter_prev0,0,r1);
        GDPRNN_module(r1,st.inter_prev1,1,r2);
        snprintf(p,sizeof(p),"%s/frame%d_rnn1.bin",dir,f);
        int32_t *gr1=load_int32(p,16*33);
        if(gr1){double sn=snr_db_2d(gr1,r1,16,33);printf("  rnn1    : SNR=%7.2f dB [%s]\n",sn,ok(sn));tested++;if(sn>80)passed++;free(gr1);}
        snprintf(p,sizeof(p),"%s/frame%d_rnn2.bin",dir,f);
        int32_t *gr2=load_int32(p,16*33);
        if(gr2){double sn=snr_db_2d(gr2,r2,16,33);printf("  rnn2    : SNR=%7.2f dB [%s]\n",sn,ok(sn));tested++;if(sn>80)passed++;free(gr2);}

        /* Decoder */
        int32_t y_dec[1*129];
        Decoder_module(r2,&st,e0,e1,e2,e3,e4,y_dec);
        snprintf(p,sizeof(p),"%s/frame%d_dec.bin",dir,f);
        int32_t *gd=load_int32(p,1*129);
        if(gd){double sn=snr_db(gd,y_dec,129);printf("  dec     : SNR=%7.2f dB [%s]\n",sn,ok(sn));tested++;if(sn>80)passed++;free(gd);}

        /* Sigmoid */
        uint16_t y_sig[1*129]; sigmoid_fixed(y_dec,129,y_sig);
        int16_t y_sig_s16[129]; for(int i=0;i<129;i++)y_sig_s16[i]=(int16_t)y_sig[i];

        /* BS */
        int16_t y_bs[1*257]; BS_fixed(y_sig_s16,erb_ierb_fc_weight,y_bs);

        /* MASK */
        int32_t real_q[257],imag_q[257];
        for(int i=0;i<257;i++){real_q[i]=F2Q20(real_in[i]);imag_q[i]=F2Q20(imag_in[i]);}
        int32_t crm[2*257];
        MASK_fixed(y_bs,real_q,imag_q,crm);
        snprintf(p,sizeof(p),"%s/frame%d_mask.bin",dir,f);
        int32_t *gm=load_int32(p,2*257);
        if(gm){double sn=snr_db(gm,crm,514);printf("  MASK    : SNR=%7.2f dB [%s]\n",sn,ok(sn));tested++;if(sn>80)passed++;free(gm);}

        free(real_in); free(imag_in);
        printf("\n");
    }

    printf("=== %d/%d layers >= 80dB ===\n",passed,tested);
    return (passed==tested)?0:1;
}
