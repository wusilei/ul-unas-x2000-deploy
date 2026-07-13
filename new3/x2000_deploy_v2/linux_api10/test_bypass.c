/* Quick bypass test: STFT→ISTFT without NR, check if pipeline is transparent */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "ulunas_fp.h"
#include "fft_q15.h"

#define N_FFT 512
#define WIN_LEN 512
#define WIN_INC 200
#define N_BINS 257
#define FIFO_SZ (WIN_LEN*4)

/* Hann window */
static int16_t hann[WIN_LEN];
static void init_hann(void) {
    for(int i=0;i<WIN_LEN;i++)
        hann[i]=(int16_t)(sinf(3.14159265f*i/(WIN_LEN-1))*32767.0f+0.5f);
}

/* WOLA inv table for WIN_INC=200 */
static uint32_t wola_inv[WIN_INC];
static void init_wola(void) {
    for(int i=0;i<WIN_INC;i++){
        int64_t ss=0;
        for(int k=0;i+k*WIN_INC<WIN_LEN;k++){
            int32_t w=hann[i+k*WIN_INC];
            ss+=(int64_t)w*w;
        }
        wola_inv[i]=(ss>0)?(uint32_t)((1ULL<<60)/ss):1;
    }
}

int main(int argc, char **argv) {
    if(argc<3){fprintf(stderr,"Usage: %s <in.wav> <out.wav>\n",argv[0]);return 1;}

    /* Read WAV */
    FILE *fi=fopen(argv[1],"rb"); uint8_t h[44]; fread(h,1,44,fi);
    int n_samp=(h[40]|(h[41]<<8)|(h[42]<<16)|(h[43]<<24))/2;
    short *in=malloc(n_samp*2),*out=malloc(n_samp*2);
    fread(in,2,n_samp,fi); fclose(fi);

    init_hann(); init_wola();

    /* FIFO + OLA */
    short fifo[FIFO_SZ]; int fw=0,fc=0;
    int32_t ola[WIN_LEN+WIN_INC]; memset(ola,0,sizeof(ola));
    int ola_pos=0;
    short ofifo[FIFO_SZ]; int orp=0,oc=0;

    int in_pos=0, out_pos=0;
    while(in_pos<n_samp || fc>=WIN_LEN) {
        /* Feed input */
        while(fc<WIN_LEN && in_pos<n_samp) {
            fifo[fw]=in[in_pos++];
            fw=(fw+1)%FIFO_SZ; fc++;
        }
        if(fc<WIN_LEN) break;
        fc-=WIN_INC;

        /* STFT */
        int32_t fft_in[WIN_LEN];
        int start=(fw-fc-WIN_LEN+FIFO_SZ)%FIFO_SZ;
        for(int i=0;i<WIN_LEN;i++)
            fft_in[i]=(int32_t)(((int64_t)fifo[(start+i)%FIFO_SZ]*hann[i]+16384)>>15);
        int32_t fr[N_BINS],fi[N_BINS];
        fft_q15_forward(fft_in,fr,fi);

        /* BYPASS: no mask, use original spectrum */
        int32_t inv_r[N_BINS],inv_i[N_BINS];
        for(int i=0;i<N_BINS;i++){inv_r[i]=fr[i];inv_i[i]=fi[i];}

        /* ISTFT */
        int32_t ifft_out[WIN_LEN];
        fft_q15_inverse(inv_r,inv_i,ifft_out);

        /* Synthesis + OLA */
        for(int i=0;i<WIN_LEN;i++){
            int32_t s=(int32_t)(((int64_t)ifft_out[i]*hann[i]+8388608)>>24);
            ola[(ola_pos+i)%(WIN_LEN+WIN_INC)]+=s;
        }
        /* WOLA output */
        for(int i=0;i<WIN_INC;i++){
            int32_t v=ola[ola_pos]; ola[ola_pos]=0;
            int idx=ola_pos%WIN_INC;
            ola_pos=(ola_pos+1)%(WIN_LEN+WIN_INC);
            int32_t norm=(int32_t)(((int64_t)v*wola_inv[idx]+(1LL<<29))>>30);
            if(norm>32767)norm=32767;if(norm<-32768)norm=-32768;
            ofifo[(orp+oc)%FIFO_SZ]=(short)norm; oc++;
        }
    }
    /* Drain output */
    int n_out=oc;
    for(int i=0;i<n_out&&i<n_samp;i++){
        out[i]=ofifo[(orp+i)%FIFO_SZ];
    }
    /* Pad remaining */
    for(int i=n_out;i<n_samp;i++) out[i]=0;

    /* Write WAV */
    FILE *fo=fopen(argv[2],"wb");
    int sz=n_samp*2, fs=sz+36, bs=8000*2;
    uint8_t wh[44]={0x52,0x49,0x46,0x46,0,0,0,0,0x57,0x41,0x56,0x45,
        0x66,0x6d,0x74,0x20,16,0,0,0,1,0,1,0,0x40,0x1f,0,0,0,0,0,0,2,0,16,0,0x64,0x61,0x74,0x61,0,0,0,0};
    memcpy(wh+4,&fs,4); memcpy(wh+28,&bs,4); memcpy(wh+40,&sz,4);
    fwrite(wh,1,44,fo); fwrite(out,2,n_samp,fo); fclose(fo);
    printf("Bypass wrote %s: %d samples\n",argv[2],n_samp);

    /* Compare */
    double corr_s=0,corr_s1=0,corr_s2=0;
    int mn=n_out<n_samp?n_out:n_samp;
    for(int i=8000;i<mn;i++){  /* skip warmup */
        double a=in[i],b=out[i];
        corr_s+=a*b; corr_s1+=a*a; corr_s2+=b*b;
    }
    double corr=corr_s/sqrt(corr_s1*corr_s2);
    printf("Correlation (skip 1s): %.4f\n",corr);
    free(in);free(out);
    return 0;
}
