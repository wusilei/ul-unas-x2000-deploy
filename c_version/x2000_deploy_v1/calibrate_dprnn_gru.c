/**
 * calibrate_dprnn_gru.c — DPRNN GRU Q-format Calibration
 * =======================================================
 * Calibrates Intra_RNN and Inter_RNN GRU qr1/qr2 independently
 * for each GDPRNN block (gdprnn_idx=0,1).
 *
 * Build: gcc -O2 -o calibrate_dprnn_gru calibrate_dprnn_gru.c
 *        ulunas_fp.c ulunas_matlab_weights.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"

static double snr_db(const int32_t *g, const int32_t *t, int n) {
    double s = 0, e = 0;
    for (int i = 0; i < n; i++) { double gv = g[i], d = gv - t[i]; s += gv * gv; e += d * d; }
    return e < 1e-30 ? 999 : 10 * log10(s / e);
}

/* ================================================================
 * Parameterized Intra_RNN_module
 * ================================================================ */
static void intra_rnn_cal(const int32_t *x, int gdprnn_idx,
                          int intra_qr1, int intra_qr2,
                          int32_t *y) {
    int16_t x0_gru[33*8], x1_gru[33*8];

    if (gdprnn_idx == 0) {
        bigru_fixed(x,33,8,INTRA_GRU_HID,
            dpgrnn_0_intra_rnn_rnn1_weight_ih_l0,dpgrnn_0_intra_rnn_rnn1_bias_ih_l0,
            dpgrnn_0_intra_rnn_rnn1_weight_hh_l0,dpgrnn_0_intra_rnn_rnn1_bias_hh_l0,
            dpgrnn_0_intra_rnn_rnn1_weight_ih_l0_reverse,dpgrnn_0_intra_rnn_rnn1_bias_ih_l0_reverse,
            dpgrnn_0_intra_rnn_rnn1_weight_hh_l0_reverse,dpgrnn_0_intra_rnn_rnn1_bias_hh_l0_reverse,
            x0_gru,intra_qr1,intra_qr2);
        bigru_fixed(x+33*8,33,8,INTRA_GRU_HID,
            dpgrnn_0_intra_rnn_rnn2_weight_ih_l0,dpgrnn_0_intra_rnn_rnn2_bias_ih_l0,
            dpgrnn_0_intra_rnn_rnn2_weight_hh_l0,dpgrnn_0_intra_rnn_rnn2_bias_hh_l0,
            dpgrnn_0_intra_rnn_rnn2_weight_ih_l0_reverse,dpgrnn_0_intra_rnn_rnn2_bias_ih_l0_reverse,
            dpgrnn_0_intra_rnn_rnn2_weight_hh_l0_reverse,dpgrnn_0_intra_rnn_rnn2_bias_hh_l0_reverse,
            x1_gru,intra_qr1,intra_qr2);
    } else {
        bigru_fixed(x,33,8,INTRA_GRU_HID,
            dpgrnn_1_intra_rnn_rnn1_weight_ih_l0,dpgrnn_1_intra_rnn_rnn1_bias_ih_l0,
            dpgrnn_1_intra_rnn_rnn1_weight_hh_l0,dpgrnn_1_intra_rnn_rnn1_bias_hh_l0,
            dpgrnn_1_intra_rnn_rnn1_weight_ih_l0_reverse,dpgrnn_1_intra_rnn_rnn1_bias_ih_l0_reverse,
            dpgrnn_1_intra_rnn_rnn1_weight_hh_l0_reverse,dpgrnn_1_intra_rnn_rnn1_bias_hh_l0_reverse,
            x0_gru,intra_qr1,intra_qr2);
        bigru_fixed(x+33*8,33,8,INTRA_GRU_HID,
            dpgrnn_1_intra_rnn_rnn2_weight_ih_l0,dpgrnn_1_intra_rnn_rnn2_bias_ih_l0,
            dpgrnn_1_intra_rnn_rnn2_weight_hh_l0,dpgrnn_1_intra_rnn_rnn2_bias_hh_l0,
            dpgrnn_1_intra_rnn_rnn2_weight_ih_l0_reverse,dpgrnn_1_intra_rnn_rnn2_bias_ih_l0_reverse,
            dpgrnn_1_intra_rnn_rnn2_weight_hh_l0_reverse,dpgrnn_1_intra_rnn_rnn2_bias_hh_l0_reverse,
            x1_gru,intra_qr1,intra_qr2);
    }

    /* Concat: (33,8)+(33,8)→(33,16) */
    int16_t x_gru[33*16];
    for(int f=0;f<33;f++){for(int h=0;h<8;h++){x_gru[f*16+h]=x0_gru[f*8+h];x_gru[f*16+8+h]=x1_gru[f*8+h];}}

    /* FC: (16→16), Qr=-9 */
    const int16_t *fc_w=(gdprnn_idx==0)?dpgrnn_0_intra_fc_weight:dpgrnn_1_intra_fc_weight;
    const int32_t *fc_b=(gdprnn_idx==0)?dpgrnn_0_intra_fc_bias:dpgrnn_1_intra_fc_bias;
    int32_t x_fc[33*16];
    for(int f=0;f<33;f++){for(int o=0;o<16;o++){int64_t acc=fc_b[o];
      for(int i=0;i<16;i++)acc+=(int64_t)x_gru[f*16+i]*(int64_t)fc_w[o*16+i];
      x_fc[f*16+o]=sat32((acc+((int64_t)1<<8))>>9);}}

    /* LN: Qr=-14 */
    const int16_t *ln_w=(gdprnn_idx==0)?dpgrnn_0_intra_ln_weight:dpgrnn_1_intra_ln_weight;
    const int32_t *ln_b=(gdprnn_idx==0)?dpgrnn_0_intra_ln_bias:dpgrnn_1_intra_ln_bias;
    int32_t x_ln[33*16]; memcpy(x_ln,x_fc,33*16*sizeof(int32_t));
    ln_fixed(x_ln,16,33,ln_w,ln_b,-14);

    /* Residual: y = x + x_ln */
    for(int i=0;i<33*16;i++)y[i]=sat32((int64_t)x[i]+(int64_t)x_ln[i]);
}

/* ================================================================
 * Parameterized Inter_RNN_module
 * ================================================================ */
static void inter_rnn_cal(const int32_t *x, int16_t *h_prev, int gdprnn_idx,
                          int inter_qr1, int inter_qr2,
                          int32_t *y) {
    int16_t x0_gru[33*8], x1_gru[33*8];

    if(gdprnn_idx==0){
        for(int f=0;f<33;f++){
            gru_step_fixed(x+f*16,8,INTER_GRU_HID,
                dpgrnn_0_inter_rnn_rnn1_weight_ih_l0,dpgrnn_0_inter_rnn_rnn1_bias_ih_l0,
                dpgrnn_0_inter_rnn_rnn1_weight_hh_l0,dpgrnn_0_inter_rnn_rnn1_bias_hh_l0,
                x0_gru+f*8,h_prev+f*16,inter_qr1,inter_qr2);}
        for(int f=0;f<33;f++){
            gru_step_fixed(x+f*16+8,8,INTER_GRU_HID,
                dpgrnn_0_inter_rnn_rnn2_weight_ih_l0,dpgrnn_0_inter_rnn_rnn2_bias_ih_l0,
                dpgrnn_0_inter_rnn_rnn2_weight_hh_l0,dpgrnn_0_inter_rnn_rnn2_bias_hh_l0,
                x1_gru+f*8,h_prev+f*16+8,inter_qr1,inter_qr2);}
    }else{
        for(int f=0;f<33;f++){
            gru_step_fixed(x+f*16,8,INTER_GRU_HID,
                dpgrnn_1_inter_rnn_rnn1_weight_ih_l0,dpgrnn_1_inter_rnn_rnn1_bias_ih_l0,
                dpgrnn_1_inter_rnn_rnn1_weight_hh_l0,dpgrnn_1_inter_rnn_rnn1_bias_hh_l0,
                x0_gru+f*8,h_prev+f*16,inter_qr1,inter_qr2);}
        for(int f=0;f<33;f++){
            gru_step_fixed(x+f*16+8,8,INTER_GRU_HID,
                dpgrnn_1_inter_rnn_rnn2_weight_ih_l0,dpgrnn_1_inter_rnn_rnn2_bias_ih_l0,
                dpgrnn_1_inter_rnn_rnn2_weight_hh_l0,dpgrnn_1_inter_rnn_rnn2_bias_hh_l0,
                x1_gru+f*8,h_prev+f*16+8,inter_qr1,inter_qr2);}
    }

    int16_t x_gru[33*16];
    for(int f=0;f<33;f++){for(int h=0;h<8;h++){x_gru[f*16+h]=x0_gru[f*8+h];x_gru[f*16+8+h]=x1_gru[f*8+h];}}

    const int16_t *fc_w=(gdprnn_idx==0)?dpgrnn_0_inter_fc_weight:dpgrnn_1_inter_fc_weight;
    const int32_t *fc_b=(gdprnn_idx==0)?dpgrnn_0_inter_fc_bias:dpgrnn_1_inter_fc_bias;
    int32_t x_fc[33*16];
    for(int f=0;f<33;f++){for(int o=0;o<16;o++){int64_t acc=fc_b[o];
      for(int i=0;i<16;i++)acc+=(int64_t)x_gru[f*16+i]*(int64_t)fc_w[o*16+i];
      x_fc[f*16+o]=sat32((acc+((int64_t)1<<8))>>9);}}

    const int16_t *ln_w=(gdprnn_idx==0)?dpgrnn_0_inter_ln_weight:dpgrnn_1_inter_ln_weight;
    const int32_t *ln_b=(gdprnn_idx==0)?dpgrnn_0_inter_ln_bias:dpgrnn_1_inter_ln_bias;
    int32_t x_ln[33*16]; memcpy(x_ln,x_fc,33*16*sizeof(int32_t));
    ln_fixed(x_ln,16,33,ln_w,ln_b,-13);

    for(int i=0;i<33*16;i++)y[i]=sat32((int64_t)x[i]+(int64_t)x_ln[i]);
}

/* ================================================================
 * Parameterized GDPRNN_module
 * ================================================================ */
static void gdprnn_cal(const int32_t *x, int16_t *inter_prev, int gdprnn_idx,
                       int intra_qr1, int intra_qr2,
                       int inter_qr1, int inter_qr2,
                       int32_t *y) {
    /* Transpose: (16,33) → (33,16) */
    int32_t x_t[33*16];
    for(int c=0;c<16;c++)for(int w=0;w<33;w++)x_t[w*16+c]=x[c*33+w];

    int32_t y_intra[33*16];
    intra_rnn_cal(x_t, gdprnn_idx, intra_qr1, intra_qr2, y_intra);

    int32_t y_inter[33*16];
    inter_rnn_cal(y_intra, inter_prev, gdprnn_idx, inter_qr1, inter_qr2, y_inter);

    /* Transpose back: (33,16) → (16,33) */
    for(int c=0;c<16;c++)for(int w=0;w<33;w++)y[c*33+w]=y_inter[w*16+c];
}

int main() {
    /* Load e4 encoder output (input to GDPRNN) and golden rnn1/rnn2 */
    int32_t ge4[16*33], gr1[16*33], gr2[16*33];
    { FILE *f = fopen("dump_matlab/frame0_enc_e4.bin","rb"); fread(ge4,4,16*33,f); fclose(f); }
    { FILE *f = fopen("dump_matlab/frame0_rnn1.bin","rb"); fread(gr1,4,16*33,f); fclose(f); }
    { FILE *f = fopen("dump_matlab/frame0_rnn2.bin","rb"); fread(gr2,4,16*33,f); fclose(f); }

    printf("=== DPRNN GRU QR Calibration ===\n\n");

    /* Baseline: current qr1=-13, qr2=-8 */
    {
        int16_t inter_prev0[33*16] = {0};
        int16_t inter_prev1[33*16] = {0};
        int32_t r1[16*33], r2[16*33];
        gdprnn_cal(ge4, inter_prev0, 0, -13, -8, -13, -8, r1);
        gdprnn_cal(r1, inter_prev1, 1, -13, -8, -13, -8, r2);
        printf("Baseline (qr=-13,-8):  rnn1=%.2f dB  rnn2=%.2f dB\n",
               snr_db(gr1, r1, 528), snr_db(gr2, r2, 528));
    }

    /* ================================================================
     * GDPRNN Block 0 (rnn1) calibration
     * ================================================================ */
    printf("\n--- GDPRNN Block 0 (rnn1) ---\n");
    printf("Scanning intra_qr1∈[-15,-11] intra_qr2∈[-10,-6] "
           "inter_qr1∈[-15,-11] inter_qr2∈[-10,-6]\n");

    double best0 = -999;
    int bi1=0, bi2=0, be1=0, be2=0;
    for (int iqr1 = -18; iqr1 <= -8; iqr1++) {
        for (int iqr2 = -12; iqr2 <= -4; iqr2++) {
            for (int eqr1 = -18; eqr1 <= -8; eqr1++) {
                for (int eqr2 = -12; eqr2 <= -4; eqr2++) {
                    int16_t inter_prev[33*16] = {0};
                    int32_t r1[16*33];
                    gdprnn_cal(ge4, inter_prev, 0, iqr1, iqr2, eqr1, eqr2, r1);
                    double sn = snr_db(gr1, r1, 528);
                    if (sn > best0) {
                        best0 = sn; bi1 = iqr1; bi2 = iqr2; be1 = eqr1; be2 = eqr2;
                    }
                }
            }
        }
    }
    printf("Best: intra_qr1=%d intra_qr2=%d inter_qr1=%d inter_qr2=%d SNR=%.2f dB\n",
           bi1, bi2, be1, be2, best0);

    /* ================================================================
     * GDPRNN Block 1 (rnn2) calibration
     * Uses calibrated block 0 output as input
     * ================================================================ */
    printf("\n--- GDPRNN Block 1 (rnn2) ---\n");
    printf("Scanning intra_qr1∈[-15,-11] intra_qr2∈[-10,-6] "
           "inter_qr1∈[-15,-11] inter_qr2∈[-10,-6]\n");

    /* First compute block 0 output with calibrated QR */
    int16_t inter_prev0[33*16] = {0};
    int32_t r1_cal[16*33];
    gdprnn_cal(ge4, inter_prev0, 0, bi1, bi2, be1, be2, r1_cal);

    double best1 = -999;
    int b2i1=0, b2i2=0, b2e1=0, b2e2=0;
    for (int iqr1 = -18; iqr1 <= -8; iqr1++) {
        for (int iqr2 = -12; iqr2 <= -4; iqr2++) {
            for (int eqr1 = -18; eqr1 <= -8; eqr1++) {
                for (int eqr2 = -12; eqr2 <= -4; eqr2++) {
                    int16_t inter_prev[33*16] = {0};
                    int32_t r2[16*33];
                    gdprnn_cal(r1_cal, inter_prev, 1, iqr1, iqr2, eqr1, eqr2, r2);
                    double sn = snr_db(gr2, r2, 528);
                    if (sn > best1) {
                        best1 = sn; b2i1 = iqr1; b2i2 = iqr2; b2e1 = eqr1; b2e2 = eqr2;
                    }
                }
            }
        }
    }
    printf("Best: intra_qr1=%d intra_qr2=%d inter_qr1=%d inter_qr2=%d SNR=%.2f dB\n",
           b2i1, b2i2, b2e1, b2e2, best1);

    /* ================================================================
     * Summary and recommended updates
     * ================================================================ */
    printf("\n=== CALIBRATION SUMMARY ===\n");
    printf("GDPRNN Block 0: intra_qr1=%d intra_qr2=%d inter_qr1=%d inter_qr2=%d → rnn1 SNR=%.2f dB\n",
           bi1, bi2, be1, be2, best0);
    printf("GDPRNN Block 1: intra_qr1=%d intra_qr2=%d inter_qr1=%d inter_qr2=%d → rnn2 SNR=%.2f dB\n",
           b2i1, b2i2, b2e1, b2e2, best1);
    printf("Original: rnn1=%.2f (qr=-13,-8), rnn2=%.2f (qr=-13,-8)\n",
           -3.29, -2.63);  /* from SUMMARY */
    printf("Improvement: rnn1 Δ=%.2f dB, rnn2 Δ=%.2f dB\n",
           best0 - (-3.29), best1 - (-2.63));

    printf("\nTo apply: update Intra_RNN_module and Inter_RNN_module in ulunas_modules.c\n");
    printf("  Block 0: Intra bigru_fixed qr=%d,%d  Inter gru_step_fixed qr=%d,%d\n",
           bi1, bi2, be1, be2);
    printf("  Block 1: Intra bigru_fixed qr=%d,%d  Inter gru_step_fixed qr=%d,%d\n",
           b2i1, b2i2, b2e1, b2e2);

    return 0;
}
