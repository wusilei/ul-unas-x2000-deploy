/* test_snr_v2.c — linux_api10 per-layer SNR vs v2 dump_matlab golden */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_lut.h"
#include "layer_dims.h"
#include "ulunas_matlab_weights.h"

static double snr_i32(const int32_t *g, const int32_t *t, int n) {
    double sn=0,sd=0;
    for(int i=0;i<n;i++){double gv=g[i],tv=t[i];sn+=gv*gv;double d=gv-tv;sd+=d*d;}
    return sd<1e-30?999.0:10.0*log10(sn/sd);
}
/* Load golden and transpose col-major (MATLAB) → row-major (C) */
static int32_t* load_t(const char *p, int rows, int cols) {
    FILE *f=fopen(p,"rb");if(!f)return NULL;
    int n=rows*cols;
    int32_t *tmp=malloc(n*4),*d=malloc(n*4);
    if(!tmp||!d){free(tmp);free(d);fclose(f);return NULL;}
    fread(tmp,4,n,f);fclose(f);
    if(rows==1){memcpy(d,tmp,n*4);}
    else{for(int r=0;r<rows;r++)for(int c=0;c<cols;c++)d[r*cols+c]=tmp[c*rows+r];}
    free(tmp);return d;
}
#define R(lbl,g,t,rows,cols) do{double s=snr_i32(g,t,(rows)*(cols));printf("  %-30s %8.2f dB  %s\n",lbl,s,s>=80?"PASS":s>=30?"WARN":"FAIL");if(s>=80)p++;else if(s<30)f++;}while(0)

int main(int argc, char **argv) {
    const char *d=argc>1?argv[1]:"dump_matlab";
    char b[512]; int p=0,f=0;
    printf("=== linux_api10 Per-Layer SNR vs Golden ===\n\n");

    /* Load BM (1D, no transpose needed) */
    snprintf(b,sizeof(b),"%s/frame0_bm.bin",d);
    int32_t *bm=load_t(b,1,129);
    if(!bm){fprintf(stderr,"FATAL: %s\n",b);return 1;}

    /* ── Encoder ── */
    printf("─── Encoder ───\n");
    {
        ulunas_state_t st; ulunas_state_init(&st);
        int32_t e0[12*65],e1[24*33],e2[24*33],e3[32*33],e4[16*33];
        encoder_module(bm,&st,e0,e1,e2,e3,e4);

        snprintf(b,sizeof(b),"%s/frame0_enc_e0.bin",d);
        int32_t *g=load_t(b,12,65); if(g){R("E0 (XConv)",g,e0,12,65);free(g);}else printf("  E0: SKIP\n");

        snprintf(b,sizeof(b),"%s/frame0_enc_e1.bin",d);
        g=load_t(b,24,33); if(g){R("E1 (XMB0)",g,e1,24,33);free(g);}else printf("  E1: SKIP\n");

        snprintf(b,sizeof(b),"%s/frame0_enc_e2.bin",d);
        g=load_t(b,24,33); if(g){R("E2 (XDWS0)",g,e2,24,33);free(g);}else printf("  E2: SKIP\n");

        snprintf(b,sizeof(b),"%s/frame0_enc_e3.bin",d);
        g=load_t(b,32,33); if(g){R("E3 (XMB1)",g,e3,32,33);free(g);}else printf("  E3: SKIP\n");

        snprintf(b,sizeof(b),"%s/frame0_enc_e4.bin",d);
        g=load_t(b,16,33); if(g){R("E4 (XDWS1)",g,e4,16,33);free(g);}else printf("  E4: SKIP\n");

        /* ── GDPRNN ── */
        printf("\n─── GDPRNN ───\n");
        int32_t rnn1[16*33],rnn2[16*33];
        gdprnn_module(e4,st.inter_cache_0,0,rnn1);
        gdprnn_module(rnn1,st.inter_cache_1,1,rnn2);

        snprintf(b,sizeof(b),"%s/frame0_rnn1.bin",d);
        g=load_t(b,16,33); if(g){R("RNN1 (Block0)",g,rnn1,16,33);free(g);}else printf("  RNN1: SKIP\n");

        snprintf(b,sizeof(b),"%s/frame0_rnn2.bin",d);
        g=load_t(b,16,33); if(g){R("RNN2 (Block1)",g,rnn2,16,33);free(g);}else printf("  RNN2: SKIP\n");

        /* ── Decoder ── */
        printf("\n─── Decoder ───\n");
        int32_t dec[1*129];
        decoder_module(rnn2,&st,e0,e1,e2,e3,e4,dec);

        snprintf(b,sizeof(b),"%s/frame0_dec_d0.bin",d);
        g=load_t(b,32,33); if(g){
            ulunas_state_t st2; ulunas_state_init(&st2);
            int32_t d0[32*33]; decoder_layer0_de_xdws0(rnn2,e4,&st2,d0);
            R("D0 (De_XDWS0)",g,d0,32,33);free(g);
        }else printf("  D0: SKIP\n");

        snprintf(b,sizeof(b),"%s/frame0_dec_d1.bin",d);
        g=load_t(b,24,33); if(g){
            ulunas_state_t st2; ulunas_state_init(&st2);
            int32_t d0[32*33],d1[24*33];
            decoder_layer0_de_xdws0(rnn2,e4,&st2,d0);
            decoder_layer1_de_xmb0(d0,e3,&st2,d1);
            R("D1 (De_XMB0)",g,d1,24,33);free(g);
        }else printf("  D1: SKIP\n");

        snprintf(b,sizeof(b),"%s/frame0_dec_d2.bin",d);
        g=load_t(b,24,33); if(g){
            ulunas_state_t st2; ulunas_state_init(&st2);
            int32_t d0[32*33],d1[24*33],d2[24*33];
            decoder_layer0_de_xdws0(rnn2,e4,&st2,d0);
            decoder_layer1_de_xmb0(d0,e3,&st2,d1);
            decoder_layer2_de_xdws1(d1,e2,&st2,d2);
            R("D2 (De_XDWS1)",g,d2,24,33);free(g);
        }else printf("  D2: SKIP\n");

        snprintf(b,sizeof(b),"%s/frame0_dec_d3.bin",d);
        g=load_t(b,12,65); if(g){
            ulunas_state_t st2; ulunas_state_init(&st2);
            int32_t d0[32*33],d1[24*33],d2[24*33],d3[12*65];
            decoder_layer0_de_xdws0(rnn2,e4,&st2,d0);
            decoder_layer1_de_xmb0(d0,e3,&st2,d1);
            decoder_layer2_de_xdws1(d1,e2,&st2,d2);
            decoder_layer3_de_xmb1(d2,e1,&st2,d3);
            R("D3 (De_XMB1)",g,d3,12,65);free(g);
        }else printf("  D3: SKIP\n");

        snprintf(b,sizeof(b),"%s/frame0_dec.bin",d);
        g=load_t(b,1,129); if(g){R("Decoder out",g,dec,1,129);free(g);}else printf("  Dec: SKIP\n");

        /* ── Sigmoid ── */
        printf("\n─── Output ───\n");
        uint16_t sig[129];
        for(int i=0;i<129;i++) sig[i]=sigmoid_q20_to_q15(dec[i]);
        snprintf(b,sizeof(b),"%s/frame0_sig.bin",d);
        FILE *fs=fopen(b,"rb");
        if(fs){uint16_t gs[129];fread(gs,2,129,fs);fclose(fs);
            double sn=0,sd=0;
            for(int i=0;i<129;i++){double gv=gs[i],tv=sig[i];sn+=gv*gv;sd+=(gv-tv)*(gv-tv);}
            double s=sd<1e-30?999:10*log10(sn/sd);
            printf("  %-30s %8.2f dB  %s\n","Sigmoid",s,s>=80?"PASS":s>=30?"WARN":"FAIL");
            if(s>=80)p++;else if(s<30)f++;
        }else printf("  Sigmoid: SKIP\n");
    }
    free(bm);
    printf("\n=== %d PASS (>=80dB), %d FAIL (<30dB) ===\n",p,f);
    return f?1:0;
}
