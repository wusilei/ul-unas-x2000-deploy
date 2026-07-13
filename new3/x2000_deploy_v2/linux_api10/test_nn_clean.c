#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "ulunas_fp.h"
#include "nn_infer.h"
#include "ulunas_matlab_weights.h"

static double snr_i32(const int32_t *g, const int32_t *t, int n) {
    double sn=0,sd=0;
    for(int i=0;i<n;i++){double gv=g[i],tv=t[i];sn+=gv*gv;sd+=(gv-tv)*(gv-tv);}
    return sd<1e-30?999:10*log10(sn/sd);
}
int main() {
    int32_t x[24*65]; for(int i=0;i<24*65;i++)x[i]=(i-780)*300;
    int32_t y1[12*65],y2[12*65];
    pconv2d_func(x,12,6,1,65,decoder_de_convs_3_pconv1_0_weight,decoder_de_convs_3_pconv1_0_bias,-14,12,y1);
    nn_pconv2d(x,12,6,65,decoder_de_convs_3_pconv1_0_weight,decoder_de_convs_3_pconv1_0_bias,-14,12,y2);
    printf("pconv2d SNR: %.1f dB\n",snr_i32(y1,y2,12*65));

    /* Also test bn */
    int32_t bx[4]={10,20,30,40}, by1[4],by2[4];
    int16_t bw[2]={16384,16384}; int32_t bb[2]={0,0}, bm[2]={0,0}; uint16_t bv[2]={16384,16384};
    bn_func_sw(bx,bw,bb,bm,bv,-14,-14,2,2,by1);
    nn_bn_sw(bx,bw,bb,bm,bv,-14,-14,2,2,by2);
    printf("bn_sw SNR: %.1f dB\n",snr_i32(by1,by2,4));

    /* Test gru */
    int32_t gx[8]={1000,2000,3000,4000,5000,6000,7000,8000};
    int16_t gh[4]={0}, gy1[4],gy2[4];
    int16_t gw[8*12]; for(int i=0;i<96;i++)gw[i]=(i-48)*10;
    int32_t gb[12]={0};
    gru_module(gx,4,8,gh,gw,gb,gw,gb,-13,-8,gy1);
    gh[0]=gh[1]=gh[2]=gh[3]=0;
    nn_gru_step(gx,4,8,gh,gw,gb,gw,gb,-13,-8,gy2);
    printf("gru_step: %s\n",memcmp(gy1,gy2,8)==0?"PASS (bit-exact)":"FAIL");

    return 0;
}
