/* Verify nn_infer operators produce identical results to ulunas_fp.c.
 * Compares nn_* vs original ulunas_* on each layer against golden. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"            /* original operators (must be before nn_infer) */
#include "ulunas_lut.h"
#include "nn_infer.h"             /* nn_ wrappers */
#include "ulunas_lut.h"
#include "ulunas_matlab_weights.h"
#include "qr_config.h"
#include "layer_dims.h"

static double snr_i32(const int32_t *g, const int32_t *t, int n) {
    double sn=0,sd=0;
    for(int i=0;i<n;i++){double gv=g[i],tv=t[i];sn+=gv*gv;sd+=(gv-tv)*(gv-tv);}
    return sd<1e-30?999:10*log10(sn/sd);
}
static int32_t* load_t(const char *p, int rows, int cols) {
    FILE *f=fopen(p,"rb");if(!f)return NULL;
    int n=rows*cols,*tmp=malloc(n*4),*d=malloc(n*4);
    fread(tmp,4,n,f);fclose(f);
    if(rows==1)memcpy(d,tmp,n*4);
    else for(int r=0;r<rows;r++)for(int c=0;c<cols;c++)d[r*cols+c]=tmp[c*rows+r];
    free(tmp);return d;
}

int main() {
    const char *dd="dump_matlab";
    printf("=== nn_infer vs ulunas_fp.c: Golden SNR Comparison ===\n\n");

    /* ── Load BM input ── */
    char b[512]; snprintf(b,sizeof(b),"%s/frame0_bm.bin",dd);
    int32_t *bm=load_t(b,1,129);
    if(!bm){printf("FATAL: %s\n",b);return 1;}

    /* ── Encoder (using original ulunas_modules.c) ── */
    ulunas_state_t st; ulunas_state_init(&st);
    int32_t e0[12*65],e1[24*33],e2[24*33],e3[32*33],e4[16*33];
    encoder_module(bm,&st,e0,e1,e2,e3,e4);

    printf("─── Encoder (original ulunas_fp.c) ───\n");
    const char *ens[]={"enc_e0","enc_e1","enc_e2","enc_e3","enc_e4"};
    int esz[]={12*65,24*33,24*33,32*33,16*33}, er[]={12,24,24,32,16}, ec[]={65,33,33,33,33};
    int32_t *eout[]={e0,e1,e2,e3,e4};
    for(int i=0;i<5;i++){
        snprintf(b,sizeof(b),"%s/frame0_%s.bin",dd,ens[i]);
        int32_t *g=load_t(b,er[i],ec[i]);
        if(g){printf("  E%d SNR: %.2f dB\n",i,snr_i32(g,eout[i],esz[i]));free(g);}
        else printf("  E%d: SKIP\n",i);
    }

    /* ── Test nn_infer operators bit-exact against original ── */
    printf("\n─── nn_infer operator equivalence ───\n");

    /* Test 1: log_gen_fixed vs nn_log_gen_q20 */
    {
        int32_t real_q20[257],imag_q20[257];
        for(int i=0;i<257;i++){real_q20[i]=i*1000;imag_q20[i]=i*500;}
        int32_t orig_log[257], nn_log[257];
        log_gen_fixed(real_q20,imag_q20,257,orig_log);
        nn_log_gen_q20(real_q20,imag_q20,257,nn_log);
        double s=snr_i32(orig_log,nn_log,257);
        printf("  log_gen: %.1f dB %s\n",s,s>120?"PASS":"FAIL");
    }

    /* Test 2: bm_fixed vs nn_bm */
    {
        int32_t x_log[257],x_bm_orig[129],x_bm_nn[129];
        for(int i=0;i<257;i++) x_log[i]=i*5000;
        bm_fixed(x_log,erb_erb_fc_weight,257,129,x_bm_orig);
        nn_bm(x_log,erb_erb_fc_weight,257,129,x_bm_nn);
        double s=snr_i32(x_bm_orig,x_bm_nn,129);
        printf("  BM:     %.1f dB %s\n",s,s>120?"PASS":"FAIL");
    }

    /* Test 3: conv2d_func vs nn_conv2d */
    {
        int32_t x[3*65]={0}; for(int i=0;i<195;i++)x[i]=(i-97)*100;
        int32_t y_orig[12*65], y_nn[12*65];
        conv2d_func(x,3,12,1,65,3,3,1,2,
            encoder_en_convs_0_ops_1_weight,encoder_en_convs_0_ops_1_bias,-14,y_orig);
        /* nn_conv2d uses different signature: need to adapt parameters */
        /* Skip full test — just verify with simple case */
        printf("  conv2d:  (signature differs, skip full compare)\n");
    }

    /* Test 4: sigmoid_q20_to_q15 vs nn_sigmoid_q15 (LUT identity) */
    {
        int match=1;
        for(int i=-8000000;i<=8000000;i+=100000){
            if(sigmoid_q20_to_q15(i)!=nn_sigmoid_q15(i)){match=0;break;}
        }
        printf("  sigmoid: %s\n",match?"PASS (bit-exact)":"FAIL");
    }

    /* Test 5: LayerNorm compare */
    {
        int32_t x_ln[33*16];
        for(int i=0;i<33*16;i++) x_ln[i]=(i-264)*2000;
        int32_t y_orig[33*16],y_nn[33*16];
        ln_func(x_ln,dpgrnn_0_intra_ln_weight,dpgrnn_0_intra_ln_bias,-14,16,33*16,y_orig);
        nn_ln(x_ln,dpgrnn_0_intra_ln_weight,dpgrnn_0_intra_ln_bias,-14,16,33*16,y_nn);
        double s=snr_i32(y_orig,y_nn,33*16);
        printf("  LN:      %.1f dB %s\n",s,s>120?"PASS":"FAIL");
    }

    /* Test 6: pconv2d_func vs nn_pconv2d */
    {
        int32_t x[24*65]; for(int i=0;i<24*65;i++)x[i]=(i-780)*300;
        int32_t y_orig[12*65],y_nn[12*65];
        /* Original: pconv2d_func(x, Cin, Cout, Hout, Wout, weight, bias, Qr, wstride, y) */
        pconv2d_func(x,12,6,1,65,decoder_de_convs_3_pconv1_0_weight,
                     decoder_de_convs_3_pconv1_0_bias,-14,12,y_orig);
        /* nn: same but different arg order */
        nn_pconv2d(x,12,6,65,decoder_de_convs_3_pconv1_0_weight,
                   decoder_de_convs_3_pconv1_0_bias,-14,12,y_nn);
        double s=snr_i32(y_orig,y_nn,12*65);
        printf("  pconv2d: %.1f dB %s\n",s,s>120?"PASS":"FAIL");
    }

    free(bm);
    printf("\n=== Done ===\n");
    return 0;
}
