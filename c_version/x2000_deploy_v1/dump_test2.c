#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"

int main() {
    float real_in[257], imag_in[257];
    FILE *fr = fopen("dump_matlab/frame1_stft_real.bin", "rb");
    fread(real_in, sizeof(float), 257, fr); fclose(fr);
    FILE *fi = fopen("dump_matlab/frame1_stft_imag.bin", "rb");
    fread(imag_in, sizeof(float), 257, fi); fclose(fi);

    int32_t x_log[257]; log_gen_fixed(real_in, imag_in, 257, x_log);
    int32_t x_bm[129]; BM_fixed(x_log, erb_erb_fc_weight, x_bm);

    /* Manual conv step matching TConv_block internals */
    int H_in = 3, Cin = 1, Win = 129, Cout = 12, Wout = 65;
    int Hk = 3, Wk = 3, stride_w = 2, pad_w = 1, conv_qr = -14;
    int shift = -conv_qr;

    /* Build x_full: 2 zero cache rows + current BM */
    int32_t x_full[3*1*129];
    memset(x_full, 0, sizeof(x_full));
    for(int i=0;i<129;i++) x_full[2*129 + i] = x_bm[i];

    /* Print x_full first 5 */
    printf("x_full[2,0..4] (BM row): ");
    for(int i=0;i<5;i++) printf("%d ", x_full[2*129+i]);
    printf("\n");

    /* Manual conv */
    int32_t y_conv[12*65];
    for(int co=0;co<Cout;co++){
        for(int wo=0;wo<Wout;wo++) y_conv[co*Wout+wo] = encoder_en_convs_0_ops_1_bias[co];
        for(int ci=0;ci<Cin;ci++){
            for(int wo=0;wo<Wout;wo++){
                int64_t acc = 0;
                for(int hk=0;hk<Hk;hk++){
                    int hi = hk;
                    if(hi<0||hi>=H_in) continue;
                    for(int wk=0;wk<Wk;wk++){
                        int wi = wo*stride_w + wk - pad_w;
                        if(wi<0||wi>=Win) continue;
                        int32_t xv = x_full[(hi*Cin+ci)*Win+wi];
                        int kidx = ((co*Cin+ci)*Hk+hk)*Wk+wk;
                        int16_t kv = encoder_en_convs_0_ops_1_weight[kidx];
                        int64_t prod = ((int64_t)xv * (int64_t)kv + (1<<(shift-1))) >> shift;
                        acc += prod;
                    }
                }
                y_conv[co*Wout+wo] = sat32((int64_t)y_conv[co*Wout+wo] + acc);
            }
        }
    }

    printf("Conv result (first 5): ");
    for(int i=0;i<5;i++) printf("%d ", y_conv[i]);
    printf("\n");

    /* BN */
    for(int c=0;c<Cout;c++){
        for(int w=0;w<Wout;w++){
            int64_t diff = (int64_t)y_conv[c*Wout+w] - (int64_t)encoder_en_convs_0_ops_2_running_mean[c];
            int64_t norm = (diff * (int64_t)encoder_en_convs_0_ops_2_running_var[c] + 8192) >> 14;
            int64_t scaled = ((int64_t)norm * (int64_t)encoder_en_convs_0_ops_2_weight[c] + 8192) >> 14;
            y_conv[c*Wout+w] = sat32(scaled + (int64_t)encoder_en_convs_0_ops_2_bias[c]);
        }
    }
    printf("After BN (first 5): ");
    for(int i=0;i<5;i++) printf("%d ", y_conv[i]);
    printf("\n");

    /* AffinePReLU */
    for(int c=0;c<Cout;c++){
        for(int w=0;w<Wout;w++){
            int32_t xo = y_conv[c*Wout+w];
            int32_t xm;
            if(xo < 0){
                int64_t np = ((int64_t)xo * (int64_t)encoder_en_convs_0_ops_3_slope_weight[c] + 4096) >> 13;
                xm = (int32_t)np;
            } else { xm = xo; }
            int64_t ap = ((int64_t)xo * (int64_t)encoder_en_convs_0_ops_3_affine_weight[c] + 4096) >> 13;
            y_conv[c*Wout+w] = sat32(ap + (int64_t)encoder_en_convs_0_ops_3_affine_bias[c] + (int64_t)xm);
        }
    }
    printf("After AffinePReLU (first 5): ");
    for(int i=0;i<5;i++) printf("%d ", y_conv[i]);
    printf("\n");

    /* Compare with golden enc_e0 */
    int32_t genc0[12*65];
    FILE *fg = fopen("dump_matlab/frame1_enc_e0.bin", "rb");
    fread(genc0, sizeof(int32_t), 12*65, fg); fclose(fg);
    printf("MATLAB enc_e0 (first 5): ");
    for(int i=0;i<5;i++) printf("%d ", genc0[i]);
    printf("\n");

    double sig=0,err=0;
    for(int i=0;i<12*65;i++){double g=genc0[i],d=g-y_conv[i];sig+=g*g;err+=d*d;}
    printf("\nManual TConv SNR: %.2f dB\n", 10*log10(sig/err));
    return 0;
}
