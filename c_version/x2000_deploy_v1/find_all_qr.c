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
    float ri[257],ii[257];
    FILE *fr=fopen("dump_matlab/frame0_stft_real.bin","rb");fread(ri,sizeof(float),257,fr);fclose(fr);
    FILE *fi=fopen("dump_matlab/frame0_stft_imag.bin","rb");fread(ii,sizeof(float),257,fi);fclose(fi);
    int32_t xl[257];log_gen_fixed(ri,ii,257,xl);
    int32_t xb[129];BM_fixed(xl,erb_erb_fc_weight,xb);

    // Load all TConv goldens
    int32_t g0[12*65],g1[24*33],g2[24*33],g3[32*33],g4[16*33];
    {FILE *f=fopen("dump_matlab/frame0_enc_e0_tconv.bin","rb");fread(g0,sizeof(int32_t),12*65,f);fclose(f);}
    {FILE *f=fopen("dump_matlab/frame0_enc_e1_tconv.bin","rb");fread(g1,sizeof(int32_t),24*33,f);fclose(f);}
    {FILE *f=fopen("dump_matlab/frame0_enc_e2_tconv.bin","rb");fread(g2,sizeof(int32_t),24*33,f);fclose(f);}
    {FILE *f=fopen("dump_matlab/frame0_enc_e3_tconv.bin","rb");fread(g3,sizeof(int32_t),32*33,f);fclose(f);}
    {FILE *f=fopen("dump_matlab/frame0_enc_e4_tconv.bin","rb");fread(g4,sizeof(int32_t),16*33,f);fclose(f);}

    // XConv e0: Cin=1 Cout=12 Win=129 Wout=65 Hk=3 Wk=3 stride=2 pad=1 cache_rows=2
    // XMB0  e1: Cin=24 Cout=24 Win=65 Wout=33 Hk=2 Wk=3 stride=2 pad=1 cache_rows=1
    // XDWS0 e2: Cin=24 Cout=24 Win=33 Wout=33 Hk=2 Wk=3 stride=1 pad=1 cache_rows=1
    // XMB1  e3: Cin=32 Cout=32 Win=33 Wout=33 Hk=1 Wk=5 stride=1 pad=2 cache_rows=0 (nonTConv)
    // XDWS1 e4: Cin=16 Cout=16 Win=33 Wout=33 Hk=1 Wk=5 stride=1 pad=2 cache_rows=0 (nonTConv)

    struct {const char*n;int32_t*g;int Ci,Co,Wi,Wo,Hk,Wk,sw,cr;int16_t *w;int32_t *b;
            uint16_t *bnw;int32_t *bnb,*bnm;uint16_t *bnv;int16_t *aw;int32_t *ab;int16_t *swgt;
            int grp;int *gl;} L[]={
    {"e0",g0,1,12,129,65,3,3,2,2, encoder_en_convs_0_ops_1_weight,encoder_en_convs_0_ops_1_bias,
     encoder_en_convs_0_ops_2_weight,encoder_en_convs_0_ops_2_bias,encoder_en_convs_0_ops_2_running_mean,encoder_en_convs_0_ops_2_running_var,
     encoder_en_convs_0_ops_3_affine_weight,encoder_en_convs_0_ops_3_affine_bias,encoder_en_convs_0_ops_3_slope_weight,1,&(int){0}},
    {"e1",g1,24,24,65,33,2,3,2,1, encoder_en_convs_1_dconv_1_weight,encoder_en_convs_1_dconv_1_bias,
     encoder_en_convs_1_dconv_2_weight,encoder_en_convs_1_dconv_2_bias,encoder_en_convs_1_dconv_2_running_mean,encoder_en_convs_1_dconv_2_running_var,
     encoder_en_convs_1_dconv_3_affine_weight,encoder_en_convs_1_dconv_3_affine_bias,encoder_en_convs_1_dconv_3_slope_weight,24,&(int){0}},
    {"e2",g2,24,24,33,33,2,3,1,1, encoder_en_convs_2_dconv_1_weight,encoder_en_convs_2_dconv_1_bias,
     encoder_en_convs_2_dconv_2_weight,encoder_en_convs_2_dconv_2_bias,encoder_en_convs_2_dconv_2_running_mean,encoder_en_convs_2_dconv_2_running_var,
     encoder_en_convs_2_dconv_3_affine_weight,encoder_en_convs_2_dconv_3_affine_bias,encoder_en_convs_2_dconv_3_slope_weight,24,&(int){0}},
    };

    int32_t e0[12*65],e1[24*33],e2[24*33];
    // Compute upstream layers to feed into e1/e2
    XConv_module(xb, (int32_t[2*1*129]){0}, (int16_t[24]){0}, e0);
    {int32_t yp0[24*65];PConv_block(e0,12,24,65,encoder_en_convs_1_pconv1_0_weight,encoder_en_convs_1_pconv1_0_bias,encoder_en_convs_1_pconv1_1_weight,encoder_en_convs_1_pconv1_1_bias,encoder_en_convs_1_pconv1_1_running_mean,encoder_en_convs_1_pconv1_1_running_var,encoder_en_convs_1_pconv1_2_affine_weight,encoder_en_convs_1_pconv1_2_affine_bias,encoder_en_convs_1_pconv1_2_slope_weight,yp0);
     int32_t ys[24*65];shuffle_interleave(yp0,12,65,ys);memcpy(e1,ys,24*65*4);}
    {int32_t yp[24*33];PConv_block(e1,24,24,33,encoder_en_convs_2_pconv_0_weight,encoder_en_convs_2_pconv_0_bias,encoder_en_convs_2_pconv_1_weight,encoder_en_convs_2_pconv_1_bias,encoder_en_convs_2_pconv_1_running_mean,encoder_en_convs_2_pconv_1_running_var,encoder_en_convs_2_pconv_2_affine_weight,encoder_en_convs_2_pconv_2_affine_bias,encoder_en_convs_2_pconv_2_slope_weight,yp);
     int32_t ys2[24*33];shuffle_interleave(yp,12,33,ys2);memcpy(e2,ys2,24*33*4);}

    int32_t *inputs[]={xb,e1,e2};

    for(int li=0;li<3;li++){
        int Ci=L[li].Ci, Co=L[li].Co, Wi=L[li].Wi, Wo=L[li].Wo, Hk=L[li].Hk, Wk=L[li].Wk, sw=L[li].sw, cr=L[li].cr;
        printf("=== %s: Cin=%d Cout=%d Win=%d Wout=%d Hk=%d Wk=%d stride=%d ===\n",L[li].n,Ci,Co,Wi,Wo,Hk,Wk,sw);
        double best=-999; int bc=-14,bb=-14;
        for(int cqr=-12;cqr>=-19;cqr--){
            for(int bqr=-11;bqr>=-18;bqr--){
                int sh=-cqr, shb=-bqr, pw=(Wk-1)/2;
                int Hi=cr+1;
                int32_t xf[Hi*Ci*Wi]; memset(xf,0,sizeof(xf));
                for(int c=0;c<Ci;c++)for(int w=0;w<Wi;w++)xf[(cr*Ci+c)*Wi+w]=inputs[li][c*Wi+w];
                int32_t yc[Co*Wo];
                for(int co=0;co<Co;co++){
                    for(int wo=0;wo<Wo;wo++)yc[co*Wo+wo]=L[li].b[co];
                    for(int ci=0;ci<Ci/Ci;ci++){ // depthwise: only self-channel
                        for(int wo=0;wo<Wo;wo++){int64_t a=0;
                            for(int hk=0;hk<Hk;hk++){
                                for(int wk=0;wk<Wk;wk++){
                                    int wi=wo*sw+wk-pw;if(wi<0||wi>=Wi)continue;
                                    int32_t xv=xf[(hk*Ci+ci)*Wi+wi];
                                    int kidx=((co*1+ci)*Hk+hk)*Wk+wk;
                                    a+=((int64_t)xv*(int64_t)L[li].w[kidx]+(1<<(sh-1)))>>sh;
                                }
                            }
                            yc[co*Wo+wo]=sat32((int64_t)yc[co*Wo+wo]+a);
                }}}
                for(int c=0;c<Co;c++)for(int w=0;w<Wo;w++){
                    int64_t d=(int64_t)yc[c*Wo+w]-(int64_t)L[li].bnm[c];
                    int32_t xn=(int32_t)((d*(int64_t)L[li].bnv[c]+(1<<(shb-1)))>>shb);
                    int64_t sc=((int64_t)xn*(int64_t)L[li].bnw[c]+8192)>>14;
                    yc[c*Wo+w]=sat32(sc+(int64_t)L[li].bnb[c]);
                }
                for(int c=0;c<Co;c++)for(int w=0;w<Wo;w++){
                    int32_t xo=yc[c*Wo+w];int32_t xm;
                    if(xo<0){int64_t np=((int64_t)xo*(int64_t)L[li].swgt[c]+4096)>>13;xm=(int32_t)np;}
                    else xm=xo;
                    int64_t ap=((int64_t)xo*(int64_t)L[li].aw[c]+4096)>>13;
                    yc[c*Wo+w]=sat32(ap+(int64_t)L[li].ab[c]+(int64_t)xm);
                }
                double sn=snr_db(L[li].g,yc,Co*Wo);
                if(sn>best){best=sn;bc=cqr;bb=bqr;}
            }
        }
        printf("  Best: conv_qr=%d bn_qr1=%d SNR=%.2f dB\n\n",bc,bb,best);
    }
    return 0;
}
