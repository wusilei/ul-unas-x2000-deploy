#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"

int main() {
    float real_in[257], imag_in[257];
    FILE *fr = fopen("dump_matlab/frame1_stft_real.bin", "rb");
    FILE *fi = fopen("dump_matlab/frame1_stft_imag.bin", "rb");
    fread(real_in, sizeof(float), 257, fr); fclose(fr);
    fread(imag_in, sizeof(float), 257, fi); fclose(fi);

    int32_t x_log[257];
    log_gen_fixed(real_in, imag_in, 257, x_log);
    int32_t x_bm[129];
    BM_fixed(x_log, erb_erb_fc_weight, x_bm);

    int32_t gbm[129];
    FILE *fg = fopen("dump_matlab/frame1_bm.bin", "rb");
    fread(gbm, sizeof(int32_t), 129, fg); fclose(fg);
    printf("=== BM (first 5) ===\n");
    for(int i=0;i<5;i++) printf("  [%d] C=%d  MATLAB=%d\n", i, x_bm[i], gbm[i]);

    /* Zero cache for first frame */
    int32_t conv_cache[2*1*129];
    memset(conv_cache, 0, sizeof(conv_cache));

    int32_t y_tconv[12*65];
    TConv_block(x_bm, conv_cache, 1, 12, 129, 65,
                3, 3, 1, 2, 2, -14, 1, -14, -14,
                encoder_en_convs_0_ops_1_weight, encoder_en_convs_0_ops_1_bias,
                encoder_en_convs_0_ops_2_weight, encoder_en_convs_0_ops_2_bias,
                encoder_en_convs_0_ops_2_running_mean, encoder_en_convs_0_ops_2_running_var,
                encoder_en_convs_0_ops_3_affine_weight, encoder_en_convs_0_ops_3_affine_bias,
                encoder_en_convs_0_ops_3_slope_weight, y_tconv);

    int32_t genc0[12*65];
    FILE *fg2 = fopen("dump_matlab/frame1_enc_e0.bin", "rb");
    fread(genc0, sizeof(int32_t), 12*65, fg2); fclose(fg2);

    printf("\n=== C TConv vs MATLAB enc_e0 (first 10) ===\n");
    for(int i=0;i<10;i++) printf("  [%d] C=%d  MATLAB=%d  diff=%d\n", i, y_tconv[i], genc0[i], y_tconv[i]-genc0[i]);

    double sig=0,err=0;
    for(int i=0;i<12*65;i++){double g=genc0[i],d=g-y_tconv[i];sig+=g*g;err+=d*d;}
    printf("\nTConv-only SNR: %.2f dB\n", 10*log10(sig/err));

    /* Also dump to file for Python analysis */
    FILE *fd = fopen("dump_matlab/frame1_c_tconv.bin", "wb");
    fwrite(y_tconv, sizeof(int32_t), 12*65, fd); fclose(fd);
    printf("C output dumped to dump_matlab/frame1_c_tconv.bin\n");
    return 0;
}
