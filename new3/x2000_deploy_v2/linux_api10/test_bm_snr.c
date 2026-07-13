/* Quick test: load STFT input → log_gen → BM → compare against golden */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_lut.h"
#include "ulunas_matlab_weights.h"

static double snr(const int32_t *g, const int32_t *t, int n) {
    double sn=0,sd=0;
    for(int i=0;i<n;i++){double gv=g[i],tv=t[i];sn+=gv*gv;sd+=(gv-tv)*(gv-tv);}
    return sd<1e-30?999:10*log10(sn/sd);
}

int main() {
    /* Load STFT float input */
    float real_f[257], imag_f[257];
    FILE *fr=fopen("dump_matlab/frame0_stft_real.bin","rb");
    FILE *fi=fopen("dump_matlab/frame0_stft_imag.bin","rb");
    fread(real_f,4,257,fr); fclose(fr);
    fread(imag_f,4,257,fi); fclose(fi);

    /* Load golden BM */
    int32_t gbm[129];
    FILE *fb=fopen("dump_matlab/frame0_bm.bin","rb");
    fread(gbm,4,129,fb); fclose(fb);

    /* C path: float→Q20 → log_gen → BM */
    int32_t real_q20[257], imag_q20[257];
    for(int i=0;i<257;i++){
        real_q20[i]=(int32_t)round(real_f[i]*1048576.0f);
        imag_q20[i]=(int32_t)round(imag_f[i]*1048576.0f);
    }

    /* log_gen */
    int32_t x_log[257];
    log_gen_fixed(real_q20, imag_q20, 257, x_log);
    printf("log_gen[0]: C=%d\n", x_log[0]);

    /* Compare log_gen against golden if available */
    int32_t glog[257];
    FILE *fl=fopen("dump_matlab/frame0_log_gen.bin","rb");
    if(fl){fread(glog,4,257,fl);fclose(fl);
        printf("log_gen SNR: %.2f dB\n", snr(glog,x_log,257));
    }

    /* BM */
    int32_t x_bm[129];
    bm_fixed(x_log, erb_erb_fc_weight, 257, 129, x_bm);

    /* Golden BM from MATLAB */
    printf("BM C[0..2]: %d %d %d\n", x_bm[0], x_bm[1], x_bm[2]);
    printf("BM G[0..2]: %d %d %d\n", gbm[0], gbm[1], gbm[2]);
    printf("BM SNR: %.2f dB\n", snr(gbm, x_bm, 129));

    return 0;
}
