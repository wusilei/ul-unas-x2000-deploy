#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"

int main() {
    float real_in[257], imag_in[257];
    FILE *fr=fopen("dump_matlab/frame1_stft_real.bin","rb");fread(real_in,sizeof(float),257,fr);fclose(fr);
    FILE *fi=fopen("dump_matlab/frame1_stft_imag.bin","rb");fread(imag_in,sizeof(float),257,fi);fclose(fi);
    int32_t xl[257];log_gen_fixed(real_in,imag_in,257,xl);
    int32_t xb[129];BM_fixed(xl,erb_erb_fc_weight,xb);
    int32_t genc0[12*65];
    FILE *fg=fopen("dump_matlab/frame1_enc_e0.bin","rb");fread(genc0,sizeof(int32_t),12*65,fg);fclose(fg);

    /* Test: FULL XConv with different conv_qr values */
    for(int cqr=-13;cqr>=-16;cqr--){
        ulunas_state_t st; memset(&st,0,sizeof(st));
        int32_t e0[12*65],e1[24*33],e2[24*33],e3[32*33],e4[16*33];
        
        /* Temporarily patch TConv_block's conv_qr by modifying the call */
        /* We can't easily do this, so let's test manually */
        
        int H_in=3,Cin=1,Win=129,Cout=12,Wout=65,Hk=3,Wk=3,sw=2,pw=1;
        int shift=-cqr;
        int32_t xf[3*129];memset(xf,0,sizeof(xf));for(int i=0;i<129;i++)xf[2*129+i]=xb[i];
        int32_t yc[12*65];
        for(int co=0;co<Cout;co++){
            for(int wo=0;wo<Wout;wo++)yc[co*Wout+wo]=encoder_en_convs_0_ops_1_bias[co];
            for(int ci=0;ci<Cin;ci++){
                for(int wo=0;wo<Wout;wo++){
                    int64_t acc=0;
                    for(int hk=0;hk<Hk;hk++){
                        for(int wk=0;wk<Wk;wk++){
                            int wi=wo*sw+wk-pw;
                            if(wi<0||wi>=Win)continue;
                            int32_t xv=xf[(hk*Cin+ci)*Win+wi];
                            int kidx=((co*Cin+ci)*Hk+hk)*Wk+wk;
                            int16_t kv=encoder_en_convs_0_ops_1_weight[kidx];
                            acc+=((int64_t)xv*(int64_t)kv+(1<<(shift-1)))>>shift;
                        }
                    }
                    yc[co*Wout+wo]=sat32((int64_t)yc[co*Wout+wo]+acc);
                }
            }
        }
        /* BN with qr1=-14 */
        for(int c=0;c<Cout;c++)for(int w=0;w<Wout;w++){
            int64_t d=(int64_t)yc[c*Wout+w]-(int64_t)encoder_en_convs_0_ops_2_running_mean[c];
            int32_t xn=(int32_t)((d*(int64_t)encoder_en_convs_0_ops_2_running_var[c]+8192)>>14);
            int64_t sc=((int64_t)xn*(int64_t)encoder_en_convs_0_ops_2_weight[c]+8192)>>14;
            yc[c*Wout+w]=sat32(sc+(int64_t)encoder_en_convs_0_ops_2_bias[c]);
        }
        /* AffinePReLU */
        for(int c=0;c<Cout;c++)for(int w=0;w<Wout;w++){
            int32_t xo=yc[c*Wout+w];int32_t xm;
            if(xo<0){int64_t np=((int64_t)xo*(int64_t)encoder_en_convs_0_ops_3_slope_weight[c]+4096)>>13;xm=(int32_t)np;}
            else xm=xo;
            int64_t ap=((int64_t)xo*(int64_t)encoder_en_convs_0_ops_3_affine_weight[c]+4096)>>13;
            yc[c*Wout+w]=sat32(ap+(int64_t)encoder_en_convs_0_ops_3_affine_bias[c]+(int64_t)xm);
        }
        double sig=0,err=0;
        for(int i=0;i<12*65;i++){double g=genc0[i],d=g-yc[i];sig+=g*g;err+=d*d;}
        printf("conv_qr=%d: SNR=%.2f dB\n",cqr,10*log10(sig/err));
    }
    return 0;
}
