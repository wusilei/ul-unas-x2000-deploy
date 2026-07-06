/**
#define QR_CALIBRATION_MODE
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

int main(int argc, char **argv) {
    const char *dir = argc>1?argv[1]:"dump_matlab/";
    char p[512];
    int tested=0,passed=0;

    printf("=== UL-UNAS Layer SNR ===\n\n");

    for(int f=0;f<2;f++){
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
        int el[]={12*65,24*33,24*33,32*33,16*33};
        for(int s=0;s<5;s++){
            snprintf(p,sizeof(p),"%s/frame%d_%s.bin",dir,f,en[s]);
            int32_t *g=load_int32(p,el[s]);
            if(g){double sn=snr_db(g,eo[s],el[s]);printf("  %-7s: SNR=%7.2f dB [%s]\n",en[s],sn,ok(sn));tested++;if(sn>80)passed++;free(g);}
            else printf("  %-7s: SKIP\n",en[s]);
        }

        /* GDPRNN */
        int32_t r1[16*33],r2[16*33];
        GDPRNN_module(e4,st.inter_prev0,0,r1);
        GDPRNN_module(r1,st.inter_prev1,1,r2);
        snprintf(p,sizeof(p),"%s/frame%d_rnn1.bin",dir,f);
        int32_t *gr1=load_int32(p,16*33);
        if(gr1){double sn=snr_db(gr1,r1,528);printf("  rnn1    : SNR=%7.2f dB [%s]\n",sn,ok(sn));tested++;if(sn>80)passed++;free(gr1);}
        snprintf(p,sizeof(p),"%s/frame%d_rnn2.bin",dir,f);
        int32_t *gr2=load_int32(p,16*33);
        if(gr2){double sn=snr_db(gr2,r2,528);printf("  rnn2    : SNR=%7.2f dB [%s]\n",sn,ok(sn));tested++;if(sn>80)passed++;free(gr2);}

        /* Decoder */
        int32_t y_dec[1*129];
        Decoder_module(r2,&st,e0,e1,e2,e3,e4,y_dec);
        snprintf(p,sizeof(p),"%s/frame%d_dec.bin",dir,f);
        int32_t *gd=load_int32(p,1*129);
        if(gd){double sn=snr_db(gd,y_dec,129);printf("  dec     : SNR=%7.2f dB [%s]\n",sn,ok(sn));tested++;if(sn>80)passed++;free(gd);}
        { int32_t *gd2=load_int32(p,1*129); if(gd2){printf("  DIAG: dec SNR with QR_MODE = %.2f dB\n",snr_db(gd2,y_dec,129));free(gd2);} }

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
