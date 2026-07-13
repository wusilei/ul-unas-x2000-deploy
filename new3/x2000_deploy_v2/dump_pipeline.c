#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_lut.h"
#include "qr_config.h"
#include "layer_dims.h"
#include "ulunas_matlab_weights.h"

static void w32(const char *p, const int32_t *d, int n) { FILE *f=fopen(p,"wb"); if(f){fwrite(d,4,n,f);fclose(f);} }
static float *lf(const char *p, int n) { FILE *f=fopen(p,"rb"); if(!f)return 0; float *b=malloc(n*4); fread(b,4,n,f); fclose(f); return b; }

int main(int argc, char **argv) {
    const char *dir = (argc>1)?argv[1]:"dump_matlab";
    char path[512];
    int frame=0;

    snprintf(path,sizeof(path),"%s/frame%d_stft_real.bin",dir,frame);
    float *real_in=lf(path,257);
    snprintf(path,sizeof(path),"%s/frame%d_stft_imag.bin",dir,frame);
    float *imag_in=lf(path,257);
    if(!real_in||!imag_in){printf("No input
");return 1;}

    ulunas_state_t st;
    ulunas_state_init(&st);

    /* EXACT COPY from test_matlab_golden.c main() pipeline */
    int32_t x_log[257];
    {
        int32_t real_q20[257], imag_q20[257];
        for (int i=0;i<257;i++){real_q20[i]=(int32_t)round(real_in[i]*1048576.0f);imag_q20[i]=(int32_t)round(imag_in[i]*1048576.0f);}
        log_gen_fixed(real_q20,imag_q20,257,x_log);
    }
    int32_t x_bm[129];
    bm_fixed(x_log,erb_erb_fc_weight,257,129,x_bm);
    w32("dump_c/frame0_bm_c.bin",x_bm,129);

    int32_t e0[12*65],e1[24*33],e2[24*33],e3[32*33],e4[16*33];
    encoder_module(x_bm,&st,e0,e1,e2,e3,e4);
    w32("dump_c/frame0_e0_c.bin",e0,12*65);
    w32("dump_c/frame0_e1_c.bin",e1,24*33);
    w32("dump_c/frame0_e2_c.bin",e2,24*33);
    w32("dump_c/frame0_e3_c.bin",e3,32*33);
    w32("dump_c/frame0_e4_c.bin",e4,16*33);

    int32_t r1[16*33],r2[16*33];
    gdprnn_module(e4,st.inter_cache_0,0,r1);
    w32("dump_c/frame0_r1_c.bin",r1,16*33);
    gdprnn_module(r1,st.inter_cache_1,1,r2);
    w32("dump_c/frame0_r2_c.bin",r2,16*33);

    int32_t y_dec[129];
    decoder_module(r2,&st,e0,e1,e2,e3,e4,y_dec);
    w32("dump_c/frame0_dec_c.bin",y_dec,129);

    uint16_t y_sig[129];
    for(int i=0;i<129;i++)y_sig[i]=sigmoid_q20_to_q15(y_dec[i]);
    int16_t y_bs[257];
    bs_fixed(y_sig,erb_ierb_fc_weight,129,257,y_bs);
    int32_t y_mask[514];
    mask_fixed(y_bs,real_q20,imag_q20,257,y_mask);

    printf("Dumped. BM[0]=%d E0[0]=%d R1[0]=%d Dec[0]=%d
",x_bm[0],e0[0],r1[0],y_dec[0]);
    free(real_in);free(imag_in);
    return 0;
}
