#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"

int main() {
    float real_in[257], imag_in[257];
    FILE *fr = fopen("dump_matlab/frame1_stft_real.bin", "rb"); fread(real_in, sizeof(float), 257, fr); fclose(fr);
    FILE *fi = fopen("dump_matlab/frame1_stft_imag.bin", "rb"); fread(imag_in, sizeof(float), 257, fi); fclose(fi);
    int32_t x_log[257]; log_gen_fixed(real_in, imag_in, 257, x_log);
    int32_t x_bm[129]; BM_fixed(x_log, erb_erb_fc_weight, x_bm);

    int32_t genc0[12*65];
    FILE *fg = fopen("dump_matlab/frame1_enc_e0.bin", "rb"); fread(genc0, sizeof(int32_t), 12*65, fg); fclose(fg);

    /* Test different conv QR and BN QR combos */
    for(int conv_qr = -14; conv_qr >= -16; conv_qr--) {
    for(int bn_qr1 = -14; bn_qr1 >= -16; bn_qr1--) {
        int H_in=3, Cin=1, Win=129, Cout=12, Wout=65, Hk=3, Wk=3, stride_w=2, pad_w=1;
        int shift_c = -conv_qr;
        int32_t x_full[3*1*129]; memset(x_full,0,sizeof(x_full));
        for(int i=0;i<129;i++) x_full[2*129+i]=x_bm[i];

        int32_t y_conv[12*65];
        for(int co=0;co<Cout;co++){
            for(int wo=0;wo<Wout;wo++) y_conv[co*Wout+wo]=encoder_en_convs_0_ops_1_bias[co];
            for(int ci=0;ci<Cin;ci++){
                for(int wo=0;wo<Wout;wo++){
                    int64_t acc=0;
                    for(int hk=0;hk<Hk;hk++){
                        int hi=hk;
                        for(int wk=0;wk<Wk;wk++){
                            int wi=wo*stride_w+wk-pad_w;
                            if(wi<0||wi>=Win) continue;
                            int32_t xv=x_full[(hi*Cin+ci)*Win+wi];
                            int kidx=((co*Cin+ci)*Hk+hk)*Wk+wk;
                            int16_t kv=encoder_en_convs_0_ops_1_weight[kidx];
                            acc += ((int64_t)xv*(int64_t)kv+(1<<(shift_c-1)))>>shift_c;
                        }
                    }
                    y_conv[co*Wout+wo]=sat32((int64_t)y_conv[co*Wout+wo]+acc);
                }
            }
        }
        /* BN with variable qr1 */
        for(int c=0;c<Cout;c++){
            for(int w=0;w<Wout;w++){
                int64_t diff=(int64_t)y_conv[c*Wout+w]-(int64_t)encoder_en_convs_0_ops_2_running_mean[c];
                int shift1=-bn_qr1;
                int32_t xn=(int32_t)((diff*(int64_t)encoder_en_convs_0_ops_2_running_var[c]+(1<<(shift1-1)))>>shift1);
                int64_t sc=((int64_t)xn*(int64_t)encoder_en_convs_0_ops_2_weight[c]+8192)>>14;
                y_conv[c*Wout+w]=sat32(sc+(int64_t)encoder_en_convs_0_ops_2_bias[c]);
            }
        }
        /* AffinePReLU */
        for(int c=0;c<Cout;c++){
            for(int w=0;w<Wout;w++){
                int32_t xo=y_conv[c*Wout+w]; int32_t xm;
                if(xo<0){int64_t np=((int64_t)xo*(int64_t)encoder_en_convs_0_ops_3_slope_weight[c]+4096)>>13;xm=(int32_t)np;}
                else xm=xo;
                int64_t ap=((int64_t)xo*(int64_t)encoder_en_convs_0_ops_3_affine_weight[c]+4096)>>13;
                y_conv[c*Wout+w]=sat32(ap+(int64_t)encoder_en_convs_0_ops_3_affine_bias[c]+(int64_t)xm);
            }
        }
        double sig=0,err=0;
        for(int i=0;i<12*65;i++){double g=genc0[i],d=g-y_conv[i];sig+=g*g;err+=d*d;}
        double snr=10*log10(sig/err);
        if(snr > -10 || (conv_qr==-14 && bn_qr1==-14))
            printf("conv_qr=%d bn_qr1=%d: SNR=%.2f dB\n", conv_qr, bn_qr1, snr);
    }}
    return 0;
}
