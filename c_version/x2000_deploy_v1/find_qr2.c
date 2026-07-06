#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"
static double snr_db(const int32_t *g, const int32_t *t, int n){
    double s=0,e=0;
    for(int i=0;i<n;i++){double gv=g[i],d=gv-t[i];s+=gv*gv;e+=d*d;}
    return e<1e-30?999:10*log10(s/e);
}
int main(){
    float ri[257],ii[257]; FILE *fr=fopen("dump_matlab/frame0_stft_real.bin","rb");fread(ri,sizeof(float),257,fr);fclose(fr);
    FILE *fi=fopen("dump_matlab/frame0_stft_imag.bin","rb");fread(ii,sizeof(float),257,fi);fclose(fi);
    int32_t xl[257];log_gen_fixed(ri,ii,257,xl);
    int32_t xb[129];BM_fixed(xl,erb_erb_fc_weight,xb);
    int32_t gg[12*65];FILE *f=fopen("dump_matlab/frame0_enc_e0_tconv.bin","rb");fread(gg,sizeof(int32_t),12*65,f);fclose(f);
    printf("=== 2D search conv_qr × bn_qr1 ===\n");
    double best=-999;int bc=-14,bb=-14;
    for(int cqr=-14;cqr>=-18;cqr--){
        for(int bqr=-11;bqr>=-17;bqr--){
            int Hi=3,Ci=1,Wi=129,Co=12,Wo=65,Hk=3,Wk=3,sw=2,pw=1;
            int sh=-cqr,shb=-bqr;
            int32_t xf[3*129];memset(xf,0,sizeof(xf));for(int i=0;i<129;i++)xf[2*129+i]=xb[i];
            int32_t yc[12*65];
            for(int co=0;co<Co;co++){for(int wo=0;wo<Wo;wo++)yc[co*Wo+wo]=encoder_en_convs_0_ops_1_bias[co];
                for(int ci=0;ci<Ci;ci++){for(int wo=0;wo<Wo;wo++){int64_t a=0;
                    for(int hk=0;hk<Hk;hk++){for(int wk=0;wk<Wk;wk++){
                        int wi=wo*sw+wk-pw;if(wi<0||wi>=Wi)continue;
                        a+=((int64_t)xf[(hk*Ci+ci)*Wi+wi]*(int64_t)encoder_en_convs_0_ops_1_weight[((co*Ci+ci)*Hk+hk)*Wk+wk]+(1<<(sh-1)))>>sh;
                    }}
                    yc[co*Wo+wo]=sat32((int64_t)yc[co*Wo+wo]+a);
        }}}
            for(int c=0;c<Co;c++)for(int w=0;w<Wo;w++){
                int64_t d=(int64_t)yc[c*Wo+w]-(int64_t)encoder_en_convs_0_ops_2_running_mean[c];
                int32_t xn=(int32_t)((d*(int64_t)encoder_en_convs_0_ops_2_running_var[c]+(1<<(shb-1)))>>shb);
                int64_t sc=((int64_t)xn*(int64_t)encoder_en_convs_0_ops_2_weight[c]+8192)>>14;
                yc[c*Wo+w]=sat32(sc+(int64_t)encoder_en_convs_0_ops_2_bias[c]);
            }
            for(int c=0;c<Co;c++)for(int w=0;w<Wo;w++){
                int32_t xo=yc[c*Wo+w];int32_t xm;
                if(xo<0){int64_t np=((int64_t)xo*(int64_t)encoder_en_convs_0_ops_3_slope_weight[c]+4096)>>13;xm=(int32_t)np;}
                else xm=xo;
                int64_t ap=((int64_t)xo*(int64_t)encoder_en_convs_0_ops_3_affine_weight[c]+4096)>>13;
                yc[c*Wo+w]=sat32(ap+(int64_t)encoder_en_convs_0_ops_3_affine_bias[c]+(int64_t)xm);
            }
            double sn=snr_db(gg,yc,12*65);
            if(sn>best){best=sn;bc=cqr;bb=bqr;}
        }
    }
    printf("Best: conv_qr=%d bn_qr1=%d SNR=%.2f dB\n",bc,bb,best);
    return 0;
}
