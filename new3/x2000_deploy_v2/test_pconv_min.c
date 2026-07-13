#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"
#include <math.h>
int main() {
    FILE *f = fopen("dump_matlab/frame0_enc_e0.bin", "rb");
    int32_t *golden = malloc(12*65*4);
    fread(golden, 4, 12*65, f); fclose(f);
    int32_t *x = malloc(12*65*4);
    for (int c=0; c<12; c++) for (int w=0; w<65; w++) x[c*65+w]=golden[c+12*w];
    int32_t y0[12*65];
    pconv2d_func(x, 6, 12, 1, 65, encoder_en_convs_1_pconv1_0_weight,
                 encoder_en_convs_1_pconv1_0_bias, -14, 24, y0);
    printf("first 8: ");
    for(int i=0;i<8;i++) printf("%d ", y0[i]);
    printf("\n");
    FILE *f2 = fopen("dump_matlab/frame0_e1_pconv0.bin", "rb");
    int32_t *gp0 = malloc(24*65*4);
    fread(gp0, 4, 24*65, f2); fclose(f2);
    int32_t *g_rm = malloc(24*65*4);
    for (int c=0; c<24; c++) for (int w=0; w<65; w++) g_rm[c*65+w]=gp0[c+24*w];
    printf("first 8 G: ");
    for(int i=0;i<8;i++) printf("%d ", g_rm[i]);
    printf("\n");
    double ss=0,ssg=0,ds=0;
    for(int i=0;i<12*65;i++){double c=y0[i],g=g_rm[i];ss+=c*c;ssg+=g*g;ds+=(c-g)*(c-g);}
    printf("PConv0 g0 SNR=%.2f dB\n", 10*log10(ssg/ds));
    free(golden);free(x);free(gp0);free(g_rm);
    return 0;
}
