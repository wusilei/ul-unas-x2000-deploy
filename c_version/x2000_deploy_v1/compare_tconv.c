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

int main() {
    float real_in[257], imag_in[257];
    FILE *fr=fopen("dump_matlab/frame0_stft_real.bin","rb");fread(real_in,sizeof(float),257,fr);fclose(fr);
    FILE *fi=fopen("dump_matlab/frame0_stft_imag.bin","rb");fread(imag_in,sizeof(float),257,fi);fclose(fi);

    int32_t xl[257];log_gen_fixed(real_in,imag_in,257,xl);
    int32_t xb[129];BM_fixed(xl,erb_erb_fc_weight,xb);

    ulunas_state_t st; ulunas_state_init(&st);

    // ===== XConv TConv (e0) =====
    {
        int32_t yc[12*65];
        TConv_block(xb, st.enc_xconv_cache, 1,12,129,65, 3,3,1,2, 2,-14,1,-14,-14,
            encoder_en_convs_0_ops_1_weight, encoder_en_convs_0_ops_1_bias,
            encoder_en_convs_0_ops_2_weight, encoder_en_convs_0_ops_2_bias,
            encoder_en_convs_0_ops_2_running_mean, encoder_en_convs_0_ops_2_running_var,
            encoder_en_convs_0_ops_3_affine_weight, encoder_en_convs_0_ops_3_affine_bias,
            encoder_en_convs_0_ops_3_slope_weight, yc);
        int32_t gg[12*65]; FILE *f=fopen("dump_matlab/frame0_enc_e0_tconv.bin","rb");fread(gg,sizeof(int32_t),12*65,f);fclose(f);
        printf("e0_tconv XConv   (12,65): SNR=%7.2f dB\n", snr_db(gg,yc,12*65));
    }

    // ===== XMB0 TConv (e1) — first need to compute y_e0 through XConv =====
    {
        int32_t e0[12*65];
        XConv_module(xb, st.enc_xconv_cache, st.enc_xconv_ta_h, e0);

        // PConv0 + shuffle
        int32_t yp0[24*65];
        PConv_block(e0,12,24,65,
            encoder_en_convs_1_pconv1_0_weight,encoder_en_convs_1_pconv1_0_bias,
            encoder_en_convs_1_pconv1_1_weight,encoder_en_convs_1_pconv1_1_bias,
            encoder_en_convs_1_pconv1_1_running_mean,encoder_en_convs_1_pconv1_1_running_var,
            encoder_en_convs_1_pconv1_2_affine_weight,encoder_en_convs_1_pconv1_2_affine_bias,
            encoder_en_convs_1_pconv1_2_slope_weight, yp0);
        int32_t ys[24*65]; shuffle_interleave(yp0,12,65,ys);

        int32_t yc[24*33];
        TConv_block(ys, st.enc_xmb0_cache, 24,24,65,33, 2,3,1,2, 1,-14,24,-11,-14,
            encoder_en_convs_1_dconv_1_weight,encoder_en_convs_1_dconv_1_bias,
            encoder_en_convs_1_dconv_2_weight,encoder_en_convs_1_dconv_2_bias,
            encoder_en_convs_1_dconv_2_running_mean,encoder_en_convs_1_dconv_2_running_var,
            encoder_en_convs_1_dconv_3_affine_weight,encoder_en_convs_1_dconv_3_affine_bias,
            encoder_en_convs_1_dconv_3_slope_weight, yc);
        int32_t gg[24*33]; FILE *f=fopen("dump_matlab/frame0_enc_e1_tconv.bin","rb");fread(gg,sizeof(int32_t),24*33,f);fclose(f);
        printf("e1_tconv XMB0    (24,33): SNR=%7.2f dB\n", snr_db(gg,yc,24*33));
    }

    // ===== XDWS0 TConv (e2) =====
    {
        // need e0,e1 first
        int32_t e0[12*65],e1[24*33];
        XConv_module(xb, st.enc_xconv_cache, st.enc_xconv_ta_h, e0);
        XMB0_module(e0, st.enc_xmb0_cache, st.enc_xmb0_ta_h, e1);

        int32_t yp[24*33];
        PConv_block(e1,24,24,33,
            encoder_en_convs_2_pconv_0_weight,encoder_en_convs_2_pconv_0_bias,
            encoder_en_convs_2_pconv_1_weight,encoder_en_convs_2_pconv_1_bias,
            encoder_en_convs_2_pconv_1_running_mean,encoder_en_convs_2_pconv_1_running_var,
            encoder_en_convs_2_pconv_2_affine_weight,encoder_en_convs_2_pconv_2_affine_bias,
            encoder_en_convs_2_pconv_2_slope_weight, yp);
        int32_t ys[24*33]; shuffle_interleave(yp,12,33,ys);

        int32_t yc[24*33];
        TConv_block(ys, st.enc_xdws0_cache, 24,24,33,33, 2,3,1,1, 1,-13,24,-14,-14,
            encoder_en_convs_2_dconv_1_weight,encoder_en_convs_2_dconv_1_bias,
            encoder_en_convs_2_dconv_2_weight,encoder_en_convs_2_dconv_2_bias,
            encoder_en_convs_2_dconv_2_running_mean,encoder_en_convs_2_dconv_2_running_var,
            encoder_en_convs_2_dconv_3_affine_weight,encoder_en_convs_2_dconv_3_affine_bias,
            encoder_en_convs_2_dconv_3_slope_weight, yc);
        int32_t gg[24*33]; FILE *f=fopen("dump_matlab/frame0_enc_e2_tconv.bin","rb");fread(gg,sizeof(int32_t),24*33,f);fclose(f);
        printf("e2_tconv XDWS0   (24,33): SNR=%7.2f dB\n", snr_db(gg,yc,24*33));
    }

    return 0;
}
