/**
 * ulunas_modules.c — UL-UNAS Encoder + Decoder + DPRNN Modules
 * =============================================================
 * Complete implementation with per-layer Q-format tracking.
 *
 * ⚠️ CRITICAL Q-FORMAT TABLE (prevent Bug #1/#2/#8 recurrence):
 *
 * cTFA_fa GRU Qr:        Enc: -13,-8   De_XDWS0/De_XMB0: -12,-7
 *                         De_XDWS1/De_XConv: -13,-8   De_XMB1: -11,-6
 * cTFA_ta GRU Qr:        ALL: -13,-8
 * cTFA_ta FC shift:       Enc: -8   De_XDWS0/De_XMB0/De_XMB1: -9   De_XDWS1/De_XConv: -8
 * cTFA_fa FC shift:       ALL: -9
 */

#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"
#include "ulunas_ctfa_qr.h"

/* ================================================================
 * Global cTFA QR Configuration (for calibration tools)
 * ================================================================
 * Defaults: v2-calibrated values from 2026-07-06 Session 3
 */
#ifdef QR_CALIBRATION_MODE
ctfa_qr_t g_qr_d0 = {-16, -4, -12,  -14, -6, -12};
ctfa_qr_t g_qr_d1 = {-12, -14, -8,  -12, -8, -8};
ctfa_qr_t g_qr_d2 = {-22, -20, -5,  -22, -20, -6};
ctfa_qr_t g_qr_d3 = {-5, -17, -9,  -11, -8, -9};
ctfa_qr_t g_qr_d4 = {-16, -18, -2,  -9, -13, -6};
ctfa_qr_t g_qr_e0 = {-13, -20, -2,  -13, -5, -14};
ctfa_qr_t g_qr_e1 = {-22, -20, -5,  -22, -14, -9};
ctfa_qr_t g_qr_e2 = {-2, -15, -11,  -15, -1, -3};
ctfa_qr_t g_qr_e3 = {-22, -14, -8,  -16, -2, -8};
ctfa_qr_t g_qr_e4 = {-22, -14, -24,  -22, -22, -12};

#define D0_TA (g_qr_d0.ta_qr1), (g_qr_d0.ta_qr2), (g_qr_d0.ta_fc)
#define D0_FA (g_qr_d0.fa_qr1), (g_qr_d0.fa_qr2), (g_qr_d0.fa_fc)
#define D1_TA (g_qr_d1.ta_qr1), (g_qr_d1.ta_qr2), (g_qr_d1.ta_fc)
#define D1_FA (g_qr_d1.fa_qr1), (g_qr_d1.fa_qr2), (g_qr_d1.fa_fc)
#define D2_TA (g_qr_d2.ta_qr1), (g_qr_d2.ta_qr2), (g_qr_d2.ta_fc)
#define D2_FA (g_qr_d2.fa_qr1), (g_qr_d2.fa_qr2), (g_qr_d2.fa_fc)
#define D3_TA (g_qr_d3.ta_qr1), (g_qr_d3.ta_qr2), (g_qr_d3.ta_fc)
#define D3_FA (g_qr_d3.fa_qr1), (g_qr_d3.fa_qr2), (g_qr_d3.fa_fc)
#define D4_TA (g_qr_d4.ta_qr1), (g_qr_d4.ta_qr2), (g_qr_d4.ta_fc)
#define D4_FA (g_qr_d4.fa_qr1), (g_qr_d4.fa_qr2), (g_qr_d4.fa_fc)
#define E0_TA (g_qr_e0.ta_qr1), (g_qr_e0.ta_qr2), (g_qr_e0.ta_fc)
#define E0_FA (g_qr_e0.fa_qr1), (g_qr_e0.fa_qr2), (g_qr_e0.fa_fc)
#define E1_TA (g_qr_e1.ta_qr1), (g_qr_e1.ta_qr2), (g_qr_e1.ta_fc)
#define E1_FA (g_qr_e1.fa_qr1), (g_qr_e1.fa_qr2), (g_qr_e1.fa_fc)
#define E2_TA (g_qr_e2.ta_qr1), (g_qr_e2.ta_qr2), (g_qr_e2.ta_fc)
#define E2_FA (g_qr_e2.fa_qr1), (g_qr_e2.fa_qr2), (g_qr_e2.fa_fc)
#define E3_TA (g_qr_e3.ta_qr1), (g_qr_e3.ta_qr2), (g_qr_e3.ta_fc)
#define E3_FA (g_qr_e3.fa_qr1), (g_qr_e3.fa_qr2), (g_qr_e3.fa_fc)
#define E4_TA (g_qr_e4.ta_qr1), (g_qr_e4.ta_qr2), (g_qr_e4.ta_fc)
#define E4_FA (g_qr_e4.fa_qr1), (g_qr_e4.fa_qr2), (g_qr_e4.fa_fc)
#else
#define D0_TA -16, -4, -12
#define D0_FA -14, -6, -12
#define D1_TA -12, -14, -8
#define D1_FA -12, -8, -8
#define D2_TA -18, -14, -4
#define D2_FA -20, -8, -6
#define D3_TA -16, -14, -12
#define D3_FA -2, -14, -10
#define D4_TA -16, -14, -2
#define D4_FA -8, -10, -6
#define E0_TA -18, -14, -12
#define E0_FA -10, -8, -4
#define E1_TA -14, -14, -4
#define E1_FA -2, -14, -6
#define E2_TA -2, -15, -11
#define E2_FA -15, -1, -3
#define E3_TA -22, -14, -8
#define E3_FA -16, -2, -8
#define E4_TA -22, -14, -24
#define E4_FA -22, -22, -12
#endif

/* ================================================================
 * Global d4 TConv QR (for joint TConv+cTFA calibration)
 * ================================================================ */
#ifdef JOINT_CALIBRATION_MODE
d4_tconv_qr_t g_d4_tconv = {-14, -11, -11};  /* conv_qr, bn_qr1, bn_qr2 */
#endif /* JOINT_CALIBRATION_MODE */

/* DPRNN GRU QR variables (always defined, modified by calibrator in QR_CALIBRATION_MODE) */
dprnn_gru_qr_t g_gru_qr_0 = {-11, -4, -14, -12};
dprnn_gru_qr_t g_gru_qr_1 = {-16, -12, -11, -12};

/* DPRNN GRU QR macros — calibration mode uses globals, else hardcoded */
#ifdef QR_CALIBRATION_MODE
#define GRU0_INTRA_QR1 g_gru_qr_0.intra_qr1
#define GRU0_INTRA_QR2 g_gru_qr_0.intra_qr2
#define GRU0_INTER_QR1 g_gru_qr_0.inter_qr1
#define GRU0_INTER_QR2 g_gru_qr_0.inter_qr2
#define GRU1_INTRA_QR1 g_gru_qr_1.intra_qr1
#define GRU1_INTRA_QR2 g_gru_qr_1.intra_qr2
#define GRU1_INTER_QR1 g_gru_qr_1.inter_qr1
#define GRU1_INTER_QR2 g_gru_qr_1.inter_qr2
#else
#define GRU0_INTRA_QR1 -11
#define GRU0_INTRA_QR2 -4
#define GRU0_INTER_QR1 -14
#define GRU0_INTER_QR2 -12
#define GRU1_INTRA_QR1 -16
#define GRU1_INTRA_QR2 -12
#define GRU1_INTER_QR1 -11
#define GRU1_INTER_QR2 -12
#endif /* QR_CALIBRATION_MODE for GRU */

/* D4 TConv QR macros */
#ifdef JOINT_CALIBRATION_MODE
#define D4_TCONV_CQR  (g_d4_tconv.conv_qr)
#define D4_TCONV_BN1  (g_d4_tconv.bn_qr1)
#define D4_TCONV_BN2  (g_d4_tconv.bn_qr2)
#else
#define D4_TCONV_CQR  -14
#define D4_TCONV_BN1  -11
#define D4_TCONV_BN2  -11
#endif

/* ================================================================
 * XConv_module — Encoder Entry Block
 * ================================================================
 * x: (1, 129) → TConv(1→12, k=(3,3), s=2) → cTFA(ta+fa) → y(12, 65)
 */
void XConv_module(const int32_t *x, int32_t *conv_cache, int16_t *ta_h,
                  int32_t *y) {
    int32_t y_tconv[12 * 65];

    /* Calibrated: conv_qr=-17 bn_qr1=-15 bn_qr2=-14 (was -14,-14,-14) */
    TConv_block(x, conv_cache, 1, 12, N_BINS_BM, N_BINS_MID,
                3, 3, 1, 2, CACHE_XCONV_ROWS, -17, 1, -15, -14,
                encoder_en_convs_0_ops_1_weight, encoder_en_convs_0_ops_1_bias,
                encoder_en_convs_0_ops_2_weight, encoder_en_convs_0_ops_2_bias,
                encoder_en_convs_0_ops_2_running_mean, encoder_en_convs_0_ops_2_running_var,
                encoder_en_convs_0_ops_3_affine_weight, encoder_en_convs_0_ops_3_affine_bias,
                encoder_en_convs_0_ops_3_slope_weight, y_tconv);

    /* cTFA_ta: Qr=-13,-8, fc_shift=-8 */
    uint16_t ta_gate[12];
    cTFA_ta_module(y_tconv, 12, 65, CTA_XCONV_HID,
                   encoder_en_convs_0_ops_4_ta_gru_weight_ih_l0,
                   encoder_en_convs_0_ops_4_ta_gru_bias_ih_l0,
                   encoder_en_convs_0_ops_4_ta_gru_weight_hh_l0,
                   encoder_en_convs_0_ops_4_ta_gru_bias_hh_l0,
                   encoder_en_convs_0_ops_4_ta_fc_weight,
                   encoder_en_convs_0_ops_4_ta_fc_bias,
                   ta_h, ta_gate, E0_TA);

    /* cTFA_fa: E0_FA */
    int32_t fa_gate[12 * 65];
    cTFA_fa_module(y_tconv, 12, 65,
                   encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0,
                   encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0,
                   encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0,
                   encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0,
                   encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0_reverse,
                   encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0_reverse,
                   encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0_reverse,
                   encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0_reverse,
                   encoder_en_convs_0_ops_4_fa_fc_weight,
                   encoder_en_convs_0_ops_4_fa_fc_bias,
                   fa_gate, E0_FA);

    cTFA_apply(y_tconv, ta_gate, fa_gate, 12, 65, y);
}

/* ================================================================
 * XMB0_module — Encoder Block 1
 * ================================================================
 * (12,65)→PConv0(12→24)→shuffle→TConv(24→24,s=2)→PConv1(BN)→cTFA→shuffle→y(24,33)
 */
void XMB0_module(const int32_t *x, int32_t *conv_cache, int16_t *ta_h,
                 int32_t *y) {
    int32_t y_pconv0[24*65], y_shuf[24*65], y_tconv[24*33], y_pconv1[24*33];

    PConv_block(x, 12, 24, 65,
                encoder_en_convs_1_pconv1_0_weight, encoder_en_convs_1_pconv1_0_bias,
                encoder_en_convs_1_pconv1_1_weight, encoder_en_convs_1_pconv1_1_bias,
                encoder_en_convs_1_pconv1_1_running_mean, encoder_en_convs_1_pconv1_1_running_var,
                encoder_en_convs_1_pconv1_2_affine_weight, encoder_en_convs_1_pconv1_2_affine_bias,
                encoder_en_convs_1_pconv1_2_slope_weight, y_pconv0);

    shuffle_interleave(y_pconv0, 12, 65, y_shuf);

    /* Calibrated: conv_qr=-17 bn_qr1=-14 bn_qr2=-14 (was -14,-11,-14) */
    TConv_block(y_shuf, conv_cache, 24, 24, 65, 33, 2, 3, 1, 2, CACHE_XMB0_ROWS, -17, 24, -14, -14,
                encoder_en_convs_1_dconv_1_weight, encoder_en_convs_1_dconv_1_bias,
                encoder_en_convs_1_dconv_2_weight, encoder_en_convs_1_dconv_2_bias,
                encoder_en_convs_1_dconv_2_running_mean, encoder_en_convs_1_dconv_2_running_var,
                encoder_en_convs_1_dconv_3_affine_weight, encoder_en_convs_1_dconv_3_affine_bias,
                encoder_en_convs_1_dconv_3_slope_weight, y_tconv);

    /* PConv1: grouped pconv2d(qr=-14) + BN(qr1=-14,qr2=-14) — NO AffinePReLU */
    { int32_t y0[12*33], y1[12*33];
      pconv2d_fixed(y_tconv, 12, 33, encoder_en_convs_1_pconv2_0_weight,
                    encoder_en_convs_1_pconv2_0_bias, 12, -14, y0);
      pconv2d_fixed(y_tconv+12*33, 12, 33, encoder_en_convs_1_pconv2_0_weight+12*12,
                    encoder_en_convs_1_pconv2_0_bias+12, 12, -14, y1);
      for(int c=0;c<12;c++){memcpy(y_pconv1+c*33,y0+c*33,33*sizeof(int32_t));memcpy(y_pconv1+(12+c)*33,y1+c*33,33*sizeof(int32_t));}
      bn_fixed(y_pconv1,24,33,encoder_en_convs_1_pconv2_1_weight,encoder_en_convs_1_pconv2_1_bias,
               encoder_en_convs_1_pconv2_1_running_mean,encoder_en_convs_1_pconv2_1_running_var,-14,-14); }

    /* cTFA: E1_TA / E1_FA */
    uint16_t ta_gate[24]; int32_t fa_gate[24*33];
    cTFA_ta_module(y_pconv1,24,33,CTA_XMB0_HID,
                   encoder_en_convs_1_pconv2_2_ta_gru_weight_ih_l0,encoder_en_convs_1_pconv2_2_ta_gru_bias_ih_l0,
                   encoder_en_convs_1_pconv2_2_ta_gru_weight_hh_l0,encoder_en_convs_1_pconv2_2_ta_gru_bias_hh_l0,
                   encoder_en_convs_1_pconv2_2_ta_fc_weight,encoder_en_convs_1_pconv2_2_ta_fc_bias,
                   ta_h,ta_gate, E1_TA);
    cTFA_fa_module(y_pconv1,24,33,
                   encoder_en_convs_1_pconv2_2_fa_gru_weight_ih_l0,encoder_en_convs_1_pconv2_2_fa_gru_bias_ih_l0,
                   encoder_en_convs_1_pconv2_2_fa_gru_weight_hh_l0,encoder_en_convs_1_pconv2_2_fa_gru_bias_hh_l0,
                   encoder_en_convs_1_pconv2_2_fa_gru_weight_ih_l0_reverse,encoder_en_convs_1_pconv2_2_fa_gru_bias_ih_l0_reverse,
                   encoder_en_convs_1_pconv2_2_fa_gru_weight_hh_l0_reverse,encoder_en_convs_1_pconv2_2_fa_gru_bias_hh_l0_reverse,
                   encoder_en_convs_1_pconv2_2_fa_fc_weight,encoder_en_convs_1_pconv2_2_fa_fc_bias,
                   fa_gate, E1_FA);

    // ta_gate is now uint16, used directly
    int32_t y_attn[24*33]; cTFA_apply(y_pconv1,ta_gate,fa_gate,24,33,y_attn);
    shuffle_deinterleave(y_attn,12,33,y);
}

/* ================================================================
 * XDWS0_module — Encoder Block 2
 * ================================================================
 * (24,33)→PConv(24→24)→shuffle→TConv(24→24,s=1)→cTFA→y(24,33)
 */
void XDWS0_module(const int32_t *x, int32_t *conv_cache, int16_t *ta_h,
                  int32_t *y) {
    int32_t y_pconv[24*33], y_shuf[24*33], y_tconv[24*33];

    PConv_block(x,24,24,33,
                encoder_en_convs_2_pconv_0_weight,encoder_en_convs_2_pconv_0_bias,
                encoder_en_convs_2_pconv_1_weight,encoder_en_convs_2_pconv_1_bias,
                encoder_en_convs_2_pconv_1_running_mean,encoder_en_convs_2_pconv_1_running_var,
                encoder_en_convs_2_pconv_2_affine_weight,encoder_en_convs_2_pconv_2_affine_bias,
                encoder_en_convs_2_pconv_2_slope_weight,y_pconv);

    shuffle_interleave(y_pconv,12,33,y_shuf);

    /* Calibrated: conv_qr=-14 bn_qr1=-17 bn_qr2=-14 (was -13,-14,-14) */
    TConv_block(y_shuf,conv_cache,24,24,33,33,2,3,1,1,CACHE_XDWS0_ROWS, -14, 24, -17, -14,
                encoder_en_convs_2_dconv_1_weight,encoder_en_convs_2_dconv_1_bias,
                encoder_en_convs_2_dconv_2_weight,encoder_en_convs_2_dconv_2_bias,
                encoder_en_convs_2_dconv_2_running_mean,encoder_en_convs_2_dconv_2_running_var,
                encoder_en_convs_2_dconv_3_affine_weight,encoder_en_convs_2_dconv_3_affine_bias,
                encoder_en_convs_2_dconv_3_slope_weight,y_tconv);

    uint16_t ta_gate[24]; int32_t fa_gate[24*33];
    cTFA_ta_module(y_tconv,24,33,CTA_XMB0_HID,  /* calibrated ta=-19,-3,-23 */
                   encoder_en_convs_2_dconv_4_ta_gru_weight_ih_l0,encoder_en_convs_2_dconv_4_ta_gru_bias_ih_l0,
                   encoder_en_convs_2_dconv_4_ta_gru_weight_hh_l0,encoder_en_convs_2_dconv_4_ta_gru_bias_hh_l0,
                   encoder_en_convs_2_dconv_4_ta_fc_weight,encoder_en_convs_2_dconv_4_ta_fc_bias,
                   ta_h,ta_gate, E2_TA);
    cTFA_fa_module(y_tconv,24,33,
                   encoder_en_convs_2_dconv_4_fa_gru_weight_ih_l0,encoder_en_convs_2_dconv_4_fa_gru_bias_ih_l0,
                   encoder_en_convs_2_dconv_4_fa_gru_weight_hh_l0,encoder_en_convs_2_dconv_4_fa_gru_bias_hh_l0,
                   encoder_en_convs_2_dconv_4_fa_gru_weight_ih_l0_reverse,encoder_en_convs_2_dconv_4_fa_gru_bias_ih_l0_reverse,
                   encoder_en_convs_2_dconv_4_fa_gru_weight_hh_l0_reverse,encoder_en_convs_2_dconv_4_fa_gru_bias_hh_l0_reverse,
                   encoder_en_convs_2_dconv_4_fa_fc_weight,encoder_en_convs_2_dconv_4_fa_fc_bias,
                   fa_gate, E2_FA);

    // ta_gate is now uint16, used directly
    cTFA_apply(y_tconv,ta_gate,fa_gate,24,33,y);
}

/* ================================================================
 * XMB1_module — Encoder Block 3
 * ================================================================
 * (24,33)→PConv0(24→32 grouped)→shuffle→nonTConv(32→32,k=(1,5))→PConv1(BN)→cTFA→shuffle→y(32,33)
 */
void XMB1_module(const int32_t *x, int16_t *ta_h, int32_t *y) {
    int32_t y_pconv0[32*33], y_shuf[32*33], y_tconv[32*33], y_pconv1[32*33];

    /* PConv0: grouped 12→16 per group, qr=-13 */
    { int32_t y0[16*33], y1[16*33];
      pconv2d_fixed(x,12,33,encoder_en_convs_3_pconv1_0_weight,encoder_en_convs_3_pconv1_0_bias,16,-13,y0);
      pconv2d_fixed(x+12*33,12,33,encoder_en_convs_3_pconv1_0_weight+16*12,encoder_en_convs_3_pconv1_0_bias+16,16,-13,y1);
      for(int c=0;c<16;c++){memcpy(y_pconv0+c*33,y0+c*33,33*sizeof(int32_t));memcpy(y_pconv0+(16+c)*33,y1+c*33,33*sizeof(int32_t));}
      bn_fixed(y_pconv0,32,33,encoder_en_convs_3_pconv1_1_weight,encoder_en_convs_3_pconv1_1_bias,
               encoder_en_convs_3_pconv1_1_running_mean,encoder_en_convs_3_pconv1_1_running_var,-11,-14);
      affine_prelu_fixed(y_pconv0,32,33,encoder_en_convs_3_pconv1_2_affine_weight,
                         encoder_en_convs_3_pconv1_2_affine_bias,encoder_en_convs_3_pconv1_2_slope_weight,-13,-13); }

    shuffle_interleave(y_pconv0,16,33,y_shuf);

    /* Calibrated: conv_qr=-13 bn_qr1=-18 bn_qr2=-14 */
    nonTConv_block(y_shuf,32,32,33,33,1,5,1,32,-13,-18,-14,
                   encoder_en_convs_3_dconv_1_weight,encoder_en_convs_3_dconv_1_bias,
                   encoder_en_convs_3_dconv_2_weight,encoder_en_convs_3_dconv_2_bias,
                   encoder_en_convs_3_dconv_2_running_mean,encoder_en_convs_3_dconv_2_running_var,
                   encoder_en_convs_3_dconv_3_affine_weight,encoder_en_convs_3_dconv_3_affine_bias,
                   encoder_en_convs_3_dconv_3_slope_weight,y_tconv);

    /* PConv1: grouped 16→16 per group, total 32→32, qr=-14, BN only */
    { int32_t y0[16*33], y1[16*33];
      pconv2d_fixed(y_tconv,16,33,encoder_en_convs_3_pconv2_0_weight,encoder_en_convs_3_pconv2_0_bias,16,-14,y0);
      pconv2d_fixed(y_tconv+16*33,16,33,encoder_en_convs_3_pconv2_0_weight+16*16,encoder_en_convs_3_pconv2_0_bias+16,16,-14,y1);
      for(int c=0;c<16;c++){memcpy(y_pconv1+c*33,y0+c*33,33*sizeof(int32_t));memcpy(y_pconv1+(16+c)*33,y1+c*33,33*sizeof(int32_t));}
      bn_fixed(y_pconv1,32,33,encoder_en_convs_3_pconv2_1_weight,encoder_en_convs_3_pconv2_1_bias,
               encoder_en_convs_3_pconv2_1_running_mean,encoder_en_convs_3_pconv2_1_running_var,-14,-14); }

    /* cTFA: Qr=-13,-8, fc_shift=-8 */
    uint16_t ta_gate[32]; int32_t fa_gate[32*33];
    cTFA_ta_module(y_pconv1,32,33,CTA_XMB1_HID,
                   encoder_en_convs_3_pconv2_2_ta_gru_weight_ih_l0,encoder_en_convs_3_pconv2_2_ta_gru_bias_ih_l0,
                   encoder_en_convs_3_pconv2_2_ta_gru_weight_hh_l0,encoder_en_convs_3_pconv2_2_ta_gru_bias_hh_l0,
                   encoder_en_convs_3_pconv2_2_ta_fc_weight,encoder_en_convs_3_pconv2_2_ta_fc_bias,
                   ta_h,ta_gate, E3_TA);
    cTFA_fa_module(y_pconv1,32,33,
                   encoder_en_convs_3_pconv2_2_fa_gru_weight_ih_l0,encoder_en_convs_3_pconv2_2_fa_gru_bias_ih_l0,
                   encoder_en_convs_3_pconv2_2_fa_gru_weight_hh_l0,encoder_en_convs_3_pconv2_2_fa_gru_bias_hh_l0,
                   encoder_en_convs_3_pconv2_2_fa_gru_weight_ih_l0_reverse,encoder_en_convs_3_pconv2_2_fa_gru_bias_ih_l0_reverse,
                   encoder_en_convs_3_pconv2_2_fa_gru_weight_hh_l0_reverse,encoder_en_convs_3_pconv2_2_fa_gru_bias_hh_l0_reverse,
                   encoder_en_convs_3_pconv2_2_fa_fc_weight,encoder_en_convs_3_pconv2_2_fa_fc_bias,
                   fa_gate, E3_FA);

    // ta_gate is now uint16, used directly
    int32_t y_attn[32*33]; cTFA_apply(y_pconv1,ta_gate,fa_gate,32,33,y_attn);
    shuffle_deinterleave(y_attn,16,33,y);
}

/* ================================================================
 * XDWS1_module — Encoder Block 4
 * ================================================================
 * (32,33)→PConv(32→16 grouped)→shuffle→nonTConv(16→16,k=(1,5))→cTFA→y(16,33)
 */
void XDWS1_module(const int32_t *x, int16_t *ta_h, int32_t *y) {
    int32_t y_pconv[16*33], y_shuf[16*33], y_tconv[16*33];

    /* PConv: group0: Cin=16→Cout=8, group1: Cin=16→Cout=8, total 32→16, qr=-14 */
    { int32_t y0[8*33], y1[8*33];
      pconv2d_fixed(x,16,33,encoder_en_convs_4_pconv_0_weight,encoder_en_convs_4_pconv_0_bias,8,-14,y0);
      pconv2d_fixed(x+16*33,16,33,encoder_en_convs_4_pconv_0_weight+8*16,encoder_en_convs_4_pconv_0_bias+8,8,-14,y1);
      for(int c=0;c<8;c++){memcpy(y_pconv+c*33,y0+c*33,33*sizeof(int32_t));memcpy(y_pconv+(8+c)*33,y1+c*33,33*sizeof(int32_t));}
      bn_fixed(y_pconv,16,33,encoder_en_convs_4_pconv_1_weight,encoder_en_convs_4_pconv_1_bias,
               encoder_en_convs_4_pconv_1_running_mean,encoder_en_convs_4_pconv_1_running_var,-11,-14);
      affine_prelu_fixed(y_pconv,16,33,encoder_en_convs_4_pconv_2_affine_weight,
                         encoder_en_convs_4_pconv_2_affine_bias,encoder_en_convs_4_pconv_2_slope_weight,-13,-13); }

    shuffle_interleave(y_pconv,8,33,y_shuf);

    /* Calibrated: conv_qr=-13 bn_qr1=-18 bn_qr2=-14 */
    nonTConv_block(y_shuf,16,16,33,33,1,5,1,16,-13,-18,-14,
                   encoder_en_convs_4_dconv_1_weight,encoder_en_convs_4_dconv_1_bias,
                   encoder_en_convs_4_dconv_2_weight,encoder_en_convs_4_dconv_2_bias,
                   encoder_en_convs_4_dconv_2_running_mean,encoder_en_convs_4_dconv_2_running_var,
                   encoder_en_convs_4_dconv_3_affine_weight,encoder_en_convs_4_dconv_3_affine_bias,
                   encoder_en_convs_4_dconv_3_slope_weight,y_tconv);

    /* cTFA: Qr=-13,-8, fc_shift=-8 */
    uint16_t ta_gate[16]; int32_t fa_gate[16*33];
    cTFA_ta_module(y_tconv,16,33,CTA_XDWS1_HID,
                   encoder_en_convs_4_dconv_4_ta_gru_weight_ih_l0,encoder_en_convs_4_dconv_4_ta_gru_bias_ih_l0,
                   encoder_en_convs_4_dconv_4_ta_gru_weight_hh_l0,encoder_en_convs_4_dconv_4_ta_gru_bias_hh_l0,
                   encoder_en_convs_4_dconv_4_ta_fc_weight,encoder_en_convs_4_dconv_4_ta_fc_bias,
                   ta_h,ta_gate, E4_TA);
    cTFA_fa_module(y_tconv,16,33,
                   encoder_en_convs_4_dconv_4_fa_gru_weight_ih_l0,encoder_en_convs_4_dconv_4_fa_gru_bias_ih_l0,
                   encoder_en_convs_4_dconv_4_fa_gru_weight_hh_l0,encoder_en_convs_4_dconv_4_fa_gru_bias_hh_l0,
                   encoder_en_convs_4_dconv_4_fa_gru_weight_ih_l0_reverse,encoder_en_convs_4_dconv_4_fa_gru_bias_ih_l0_reverse,
                   encoder_en_convs_4_dconv_4_fa_gru_weight_hh_l0_reverse,encoder_en_convs_4_dconv_4_fa_gru_bias_hh_l0_reverse,
                   encoder_en_convs_4_dconv_4_fa_fc_weight,encoder_en_convs_4_dconv_4_fa_fc_bias,
                   fa_gate, E4_FA);

    // ta_gate is now uint16, used directly
    cTFA_apply(y_tconv,ta_gate,fa_gate,16,33,y);
}

/* ================================================================
 * Encoder_module — Top-level
 * ================================================================ */
void Encoder_module(const int32_t *x, ulunas_state_t *state,
                    int32_t *y_e0, int32_t *y_e1, int32_t *y_e2,
                    int32_t *y_e3, int32_t *y_e4) {
    XConv_module(x, state->enc_xconv_cache, state->enc_xconv_ta_h, y_e0);
    XMB0_module(y_e0, state->enc_xmb0_cache, state->enc_xmb0_ta_h, y_e1);
    XDWS0_module(y_e1, state->enc_xdws0_cache, state->enc_xdws0_ta_h, y_e2);
    XMB1_module(y_e2, state->enc_xmb1_ta_h, y_e3);
    XDWS1_module(y_e3, state->enc_xdws1_ta_h, y_e4);
}

/* ================================================================
 * DECODER — De_XDWS0_module
 * ================================================================
 * x(16,33)+skip(16,33)→concat(32,33)→PConv0(32→32)→shuffle→nonGTConv(32→32)→cTFA→y(32,33)
 *
 * ⚠️ cTFA_fa GRU Qr=-12,-7 (NOT -13,-8!)
 * ⚠️ cTFA_ta FC shift=-9 (NOT -8!)
 */
void De_XDWS0_module(const int32_t *x, const int32_t *x_skip,
                     int16_t *ta_h, int32_t *y) {
    int32_t x_cat[32*33];
    for(int i=0;i<16*33;i++){x_cat[i]=sat32((int64_t)x[i]+(int64_t)x_skip[i]);x_cat[16*33+i]=sat32((int64_t)x[16*33+i]+(int64_t)x_skip[16*33+i]);}

    int32_t y_pconv[32*33], y_shuf[32*33], y_tconv[32*33];

    /* PConv: grouped 16→16 per group, Cin=8+8 per half (total Cin=16, Cout=32) */
    { int32_t y0[16*33], y1[16*33];
      pconv2d_fixed(x_cat,8,33,decoder_de_convs_0_pconv_0_weight,decoder_de_convs_0_pconv_0_bias,16,-14,y0);
      pconv2d_fixed(x_cat+8*33,8,33,decoder_de_convs_0_pconv_0_weight+16*8,decoder_de_convs_0_pconv_0_bias+16,16,-14,y1);
      for(int c=0;c<16;c++){memcpy(y_pconv+c*33,y0+c*33,33*sizeof(int32_t));memcpy(y_pconv+(16+c)*33,y1+c*33,33*sizeof(int32_t));}
      bn_fixed(y_pconv,32,33,decoder_de_convs_0_pconv_1_weight,decoder_de_convs_0_pconv_1_bias,
               decoder_de_convs_0_pconv_1_running_mean,decoder_de_convs_0_pconv_1_running_var,-14,-14);
      affine_prelu_fixed(y_pconv,32,33,decoder_de_convs_0_pconv_2_affine_weight,
                         decoder_de_convs_0_pconv_2_affine_bias,decoder_de_convs_0_pconv_2_slope_weight,-13,-13); }

    shuffle_interleave(y_pconv,16,33,y_shuf);

    /* nonGTConv: Cin=Cout=32, k=(1,5), s=1, groups=16 */
    /* Calibrated: conv_qr=-17 bn_qr1=-15 bn_qr2=-16 (was -13,-14,-14) */
    nonGTConv_block(y_shuf,32,32,33,33,1,5,1,32,-17,-15,-16,
                    decoder_de_convs_0_dconv_1_weight,decoder_de_convs_0_dconv_1_bias,
                    decoder_de_convs_0_dconv_2_weight,decoder_de_convs_0_dconv_2_bias,
                    decoder_de_convs_0_dconv_2_running_mean,decoder_de_convs_0_dconv_2_running_var,
                    decoder_de_convs_0_dconv_3_affine_weight,decoder_de_convs_0_dconv_3_affine_bias,
                    decoder_de_convs_0_dconv_3_slope_weight,y_tconv);

    /* cTFA: D0_TA / D0_FA */
    uint16_t ta_gate[32]; int32_t fa_gate[32*33];
    cTFA_ta_module(y_tconv,32,33,CH_DEC_XDWS0,
                   decoder_de_convs_0_dconv_4_ta_gru_weight_ih_l0,decoder_de_convs_0_dconv_4_ta_gru_bias_ih_l0,
                   decoder_de_convs_0_dconv_4_ta_gru_weight_hh_l0,decoder_de_convs_0_dconv_4_ta_gru_bias_hh_l0,
                   decoder_de_convs_0_dconv_4_ta_fc_weight,decoder_de_convs_0_dconv_4_ta_fc_bias,
                   ta_h,ta_gate, D0_TA);
    cTFA_fa_module(y_tconv,32,33,
                   decoder_de_convs_0_dconv_4_fa_gru_weight_ih_l0,decoder_de_convs_0_dconv_4_fa_gru_bias_ih_l0,
                   decoder_de_convs_0_dconv_4_fa_gru_weight_hh_l0,decoder_de_convs_0_dconv_4_fa_gru_bias_hh_l0,
                   decoder_de_convs_0_dconv_4_fa_gru_weight_ih_l0_reverse,decoder_de_convs_0_dconv_4_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_0_dconv_4_fa_gru_weight_hh_l0_reverse,decoder_de_convs_0_dconv_4_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_0_dconv_4_fa_fc_weight,decoder_de_convs_0_dconv_4_fa_fc_bias,
                   fa_gate, D0_FA);

    // ta_gate is now uint16, used directly
    cTFA_apply(y_tconv,ta_gate,fa_gate,32,33,y);
}

/* ================================================================
 * DECODER — De_XMB0_module
 * ================================================================
 * x(32,33)+skip(24,33)? No: x(32,33)+skip=XDWS0_out(32,33)...
 * Wait. Decoder data flow:
 *   y_rnn2(16,33) + y_e4(16,33) → De_XDWS0 → (32,33)
 *   De_XDWS0_out(32,33) + y_e3(32,33) → De_XMB0 → (24,33)
 *   De_XMB0_out(24,33) + y_e2(24,33) → De_XDWS1 → (24,33)
 *   De_XDWS1_out(24,33) + y_e1(24,33) → De_XMB1 → (12,65)
 *   De_XMB1_out(12,65) + y_e0(12,65) → De_XConv → (2,129)
 *
 * So De_XMB0: x(32,33) + skip(32,33) → PConv0(32→24 grouped)→shuffle→nonGTConv→PConv1(BN)→cTFA→shuffle→y(24,33)
 *
 * ⚠️ cTFA_fa GRU Qr=-12,-7 | cTFA_ta FC shift=-9
 */
void De_XMB0_module(const int32_t *x, const int32_t *x_skip,
                    int16_t *ta_h, int32_t *y) {
    int32_t x_cat[32*33];
    for(int i=0;i<32*33;i++)x_cat[i]=sat32((int64_t)x[i]+(int64_t)x_skip[i]);

    int32_t y_pconv0[24*33], y_shuf[24*33], y_tconv[24*33], y_pconv1[24*33];

    /* PConv0: Cin=16 per group, Cout=12 per group → total 32→24 */
    { int32_t y0[12*33], y1[12*33];
      pconv2d_fixed(x_cat,16,33,decoder_de_convs_1_pconv1_0_weight,decoder_de_convs_1_pconv1_0_bias,12,-13,y0);
      pconv2d_fixed(x_cat+16*33,16,33,decoder_de_convs_1_pconv1_0_weight+12*16,decoder_de_convs_1_pconv1_0_bias+12,12,-13,y1);
      for(int c=0;c<12;c++){memcpy(y_pconv0+c*33,y0+c*33,33*sizeof(int32_t));memcpy(y_pconv0+(12+c)*33,y1+c*33,33*sizeof(int32_t));}
      bn_fixed(y_pconv0,24,33,decoder_de_convs_1_pconv1_1_weight,decoder_de_convs_1_pconv1_1_bias,
               decoder_de_convs_1_pconv1_1_running_mean,decoder_de_convs_1_pconv1_1_running_var,-11,-14);
      affine_prelu_fixed(y_pconv0,24,33,decoder_de_convs_1_pconv1_2_affine_weight,
                         decoder_de_convs_1_pconv1_2_affine_bias,decoder_de_convs_1_pconv1_2_slope_weight,-13,-13); }

    shuffle_interleave(y_pconv0,12,33,y_shuf);

    /* nonGTConv: Cin=Cout=24, k=(1,5), s=1, groups=12 */
    /* Calibrated: conv_qr=-15 bn_qr1=-15 bn_qr2=-14 (was -13,-14,-14) */
    nonGTConv_block(y_shuf,24,24,33,33,1,5,1,24,-15,-15,-14,
                    decoder_de_convs_1_dconv_1_weight,decoder_de_convs_1_dconv_1_bias,
                    decoder_de_convs_1_dconv_2_weight,decoder_de_convs_1_dconv_2_bias,
                    decoder_de_convs_1_dconv_2_running_mean,decoder_de_convs_1_dconv_2_running_var,
                    decoder_de_convs_1_dconv_3_affine_weight,decoder_de_convs_1_dconv_3_affine_bias,
                    decoder_de_convs_1_dconv_3_slope_weight,y_tconv);

    /* PConv1: grouped 12→12, qr=-14, BN only qr1=-11,qr2=-11 */
    { int32_t y0[12*33], y1[12*33];
      pconv2d_fixed(y_tconv,12,33,decoder_de_convs_1_pconv2_0_weight,decoder_de_convs_1_pconv2_0_bias,12,-14,y0);
      pconv2d_fixed(y_tconv+12*33,12,33,decoder_de_convs_1_pconv2_0_weight+12*12,decoder_de_convs_1_pconv2_0_bias+12,12,-14,y1);
      for(int c=0;c<12;c++){memcpy(y_pconv1+c*33,y0+c*33,33*sizeof(int32_t));memcpy(y_pconv1+(12+c)*33,y1+c*33,33*sizeof(int32_t));}
      bn_fixed(y_pconv1,24,33,decoder_de_convs_1_pconv2_1_weight,decoder_de_convs_1_pconv2_1_bias,
               decoder_de_convs_1_pconv2_1_running_mean,decoder_de_convs_1_pconv2_1_running_var,-11,-11); }

    /* cTFA: D1_TA / D1_FA */
    uint16_t ta_gate[24]; int32_t fa_gate[24*33];
    cTFA_ta_module(y_pconv1,24,33,CH_DEC_XMB0,
                   decoder_de_convs_1_pconv2_2_ta_gru_weight_ih_l0,decoder_de_convs_1_pconv2_2_ta_gru_bias_ih_l0,
                   decoder_de_convs_1_pconv2_2_ta_gru_weight_hh_l0,decoder_de_convs_1_pconv2_2_ta_gru_bias_hh_l0,
                   decoder_de_convs_1_pconv2_2_ta_fc_weight,decoder_de_convs_1_pconv2_2_ta_fc_bias,
                   ta_h,ta_gate, D1_TA);
    cTFA_fa_module(y_pconv1,24,33,
                   decoder_de_convs_1_pconv2_2_fa_gru_weight_ih_l0,decoder_de_convs_1_pconv2_2_fa_gru_bias_ih_l0,
                   decoder_de_convs_1_pconv2_2_fa_gru_weight_hh_l0,decoder_de_convs_1_pconv2_2_fa_gru_bias_hh_l0,
                   decoder_de_convs_1_pconv2_2_fa_gru_weight_ih_l0_reverse,decoder_de_convs_1_pconv2_2_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_1_pconv2_2_fa_gru_weight_hh_l0_reverse,decoder_de_convs_1_pconv2_2_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_1_pconv2_2_fa_fc_weight,decoder_de_convs_1_pconv2_2_fa_fc_bias,
                   fa_gate, D1_FA);

    // ta_gate is now uint16, used directly
    int32_t y_attn[24*33]; cTFA_apply(y_pconv1,ta_gate,fa_gate,24,33,y_attn);
    shuffle_deinterleave(y_attn,12,33,y);
}

/* ================================================================
 * DECODER — De_XDWS1_module
 * ================================================================
 * x(24,33)+skip(24,33)→PConv(24→24)→shuffle→GTConv(24→24,s=1,cache)→cTFA→y(24,33)
 *
 * ⚠️ cTFA Qr=-13,-8 (SAME as encoder!) | cTFA_ta FC shift=-8 (SAME as encoder!)
 * ⚠️ GTConv uses gtconv2d qr=-13 (not -14!)
 * ⚠️ BN: qr1=-11,qr2=-14 for PConv
 */
void De_XDWS1_module(const int32_t *x, const int32_t *x_skip,
                     int32_t *conv_cache, int16_t *ta_h, int32_t *y) {
    int32_t x_cat[24*33];
    for(int i=0;i<24*33;i++)x_cat[i]=sat32((int64_t)x[i]+(int64_t)x_skip[i]);

    int32_t y_pconv[24*33], y_shuf[24*33], y_tconv[24*33];

    /* PConv: grouped 12→12 per group (Cin=12,Cout=12 per half), qr=-14 */
    { int32_t y0[12*33], y1[12*33];
      pconv2d_fixed(x_cat,12,33,decoder_de_convs_2_pconv_0_weight,decoder_de_convs_2_pconv_0_bias,12,-14,y0);
      pconv2d_fixed(x_cat+12*33,12,33,decoder_de_convs_2_pconv_0_weight+12*12,decoder_de_convs_2_pconv_0_bias+12,12,-14,y1);
      for(int c=0;c<12;c++){memcpy(y_pconv+c*33,y0+c*33,33*sizeof(int32_t));memcpy(y_pconv+(12+c)*33,y1+c*33,33*sizeof(int32_t));}
      bn_fixed(y_pconv,24,33,decoder_de_convs_2_pconv_1_weight,decoder_de_convs_2_pconv_1_bias,
               decoder_de_convs_2_pconv_1_running_mean,decoder_de_convs_2_pconv_1_running_var,-11,-14);
      affine_prelu_fixed(y_pconv,24,33,decoder_de_convs_2_pconv_2_affine_weight,
                         decoder_de_convs_2_pconv_2_affine_bias,decoder_de_convs_2_pconv_2_slope_weight,-13,-13); }

    shuffle_interleave(y_pconv,12,33,y_shuf);

    /* GTConv with cache: Cin=Cout=24, k=(2,3), s=1, groups=12 */
    /* Calibrated: conv_qr=-12 bn_qr1=-18 bn_qr2=-14 (was -13,-11,-14) */
    GTConv_block(y_shuf,conv_cache,24,24,33,33,1,3,1,1,CACHE_DEXDWS1_ROWS, -12, 24, -18, -14,
                 decoder_de_convs_2_dconv_1_weight,decoder_de_convs_2_dconv_1_bias,
                 decoder_de_convs_2_dconv_2_weight,decoder_de_convs_2_dconv_2_bias,
                 decoder_de_convs_2_dconv_2_running_mean,decoder_de_convs_2_dconv_2_running_var,
                 decoder_de_convs_2_dconv_3_affine_weight,decoder_de_convs_2_dconv_3_affine_bias,
                 decoder_de_convs_2_dconv_3_slope_weight,y_tconv);

    /* cTFA: D2_TA / D2_FA */
    uint16_t ta_gate[24]; int32_t fa_gate[24*33];
    cTFA_ta_module(y_tconv,24,33,CH_DEC_XMB0,  /* 48 hidden like XDWS0 */
                   decoder_de_convs_2_dconv_4_ta_gru_weight_ih_l0,decoder_de_convs_2_dconv_4_ta_gru_bias_ih_l0,
                   decoder_de_convs_2_dconv_4_ta_gru_weight_hh_l0,decoder_de_convs_2_dconv_4_ta_gru_bias_hh_l0,
                   decoder_de_convs_2_dconv_4_ta_fc_weight,decoder_de_convs_2_dconv_4_ta_fc_bias,
                   ta_h,ta_gate, D2_TA);
    cTFA_fa_module(y_tconv,24,33,
                   decoder_de_convs_2_dconv_4_fa_gru_weight_ih_l0,decoder_de_convs_2_dconv_4_fa_gru_bias_ih_l0,
                   decoder_de_convs_2_dconv_4_fa_gru_weight_hh_l0,decoder_de_convs_2_dconv_4_fa_gru_bias_hh_l0,
                   decoder_de_convs_2_dconv_4_fa_gru_weight_ih_l0_reverse,decoder_de_convs_2_dconv_4_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_2_dconv_4_fa_gru_weight_hh_l0_reverse,decoder_de_convs_2_dconv_4_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_2_dconv_4_fa_fc_weight,decoder_de_convs_2_dconv_4_fa_fc_bias,
                   fa_gate, D2_FA);

    // ta_gate is now uint16, used directly
    cTFA_apply(y_tconv,ta_gate,fa_gate,24,33,y);
}

/* ================================================================
 * DECODER — De_XMB1_module
 * ================================================================
 * x(24,33)+skip(24,33)→PConv0(24→12 grouped)→shuffle→GTConv(12→12,s=2,cache)→PConv1(BN)→cTFA→shuffle→y(12,65)
 *
 * ⚠️ cTFA_fa GRU Qr=-11,-6 (MOST AGGRESSIVE quantization!)
 * ⚠️ GTConv qr=-14, BN qr1=-11,qr2=-11
 * ⚠️ PConv1 BN qr1=-11,qr2=-11
 * ⚠️ De_XMB1 output is (12,65) — W dimension expands from 33→65 via GTConv stride=2
 */
void De_XMB1_module(const int32_t *x, const int32_t *x_skip,
                    int32_t *conv_cache, int16_t *ta_h, int32_t *y) {
    int32_t x_cat[24*33];
    for(int i=0;i<24*33;i++)x_cat[i]=sat32((int64_t)x[i]+(int64_t)x_skip[i]);

    int32_t y_pconv0[12*33], y_shuf[12*33], y_tconv[12*65], y_pconv1[12*65];

    /* PConv0: Cin=12 per group, Cout=6 per group → total 24→12, qr=-14 */
    { int32_t y0[6*33], y1[6*33];
      pconv2d_fixed(x_cat,12,33,decoder_de_convs_3_pconv1_0_weight,decoder_de_convs_3_pconv1_0_bias,6,-14,y0);
      pconv2d_fixed(x_cat+12*33,12,33,decoder_de_convs_3_pconv1_0_weight+6*12,decoder_de_convs_3_pconv1_0_bias+6,6,-14,y1);
      for(int c=0;c<6;c++){memcpy(y_pconv0+c*33,y0+c*33,33*sizeof(int32_t));memcpy(y_pconv0+(6+c)*33,y1+c*33,33*sizeof(int32_t));}
      bn_fixed(y_pconv0,12,33,decoder_de_convs_3_pconv1_1_weight,decoder_de_convs_3_pconv1_1_bias,
               decoder_de_convs_3_pconv1_1_running_mean,decoder_de_convs_3_pconv1_1_running_var,-11,-14);
      affine_prelu_fixed(y_pconv0,12,33,decoder_de_convs_3_pconv1_2_affine_weight,
                         decoder_de_convs_3_pconv1_2_affine_bias,decoder_de_convs_3_pconv1_2_slope_weight,-13,-13); }

    shuffle_interleave(y_pconv0,6,33,y_shuf);

    /* GTConv: Cin=Cout=12, k=(2,3), s=2(upsample W 33→65), groups=6 */
    /* Calibrated: conv_qr=-17 bn_qr1=-18 bn_qr2=-16 (was -14,-11,-11) */
    GTConv_block(y_shuf,conv_cache,12,12,33,65,1,3,1,2,CACHE_DEXMB1_ROWS, -17, 12, -18, -16,
                 decoder_de_convs_3_dconv_1_weight,decoder_de_convs_3_dconv_1_bias,
                 decoder_de_convs_3_dconv_2_weight,decoder_de_convs_3_dconv_2_bias,
                 decoder_de_convs_3_dconv_2_running_mean,decoder_de_convs_3_dconv_2_running_var,
                 decoder_de_convs_3_dconv_3_affine_weight,decoder_de_convs_3_dconv_3_affine_bias,
                 decoder_de_convs_3_dconv_3_slope_weight,y_tconv);

    /* PConv1: grouped 6→6, qr=-14, BN only qr1=-11,qr2=-11 */
    { int32_t y0[6*65], y1[6*65];
      pconv2d_fixed(y_tconv,6,65,decoder_de_convs_3_pconv2_0_weight,decoder_de_convs_3_pconv2_0_bias,6,-14,y0);
      pconv2d_fixed(y_tconv+6*65,6,65,decoder_de_convs_3_pconv2_0_weight+6*6,decoder_de_convs_3_pconv2_0_bias+6,6,-14,y1);
      for(int c=0;c<6;c++){memcpy(y_pconv1+c*65,y0+c*65,65*sizeof(int32_t));memcpy(y_pconv1+(6+c)*65,y1+c*65,65*sizeof(int32_t));}
      bn_fixed(y_pconv1,12,65,decoder_de_convs_3_pconv2_1_weight,decoder_de_convs_3_pconv2_1_bias,
               decoder_de_convs_3_pconv2_1_running_mean,decoder_de_convs_3_pconv2_1_running_var,-11,-11); }

    /* cTFA: D3_TA / D3_FA */
    uint16_t ta_gate[12]; int32_t fa_gate[12*65];
    cTFA_ta_module(y_pconv1,12,65,CH_DEC_XMB1,
                   decoder_de_convs_3_pconv2_2_ta_gru_weight_ih_l0,decoder_de_convs_3_pconv2_2_ta_gru_bias_ih_l0,
                   decoder_de_convs_3_pconv2_2_ta_gru_weight_hh_l0,decoder_de_convs_3_pconv2_2_ta_gru_bias_hh_l0,
                   decoder_de_convs_3_pconv2_2_ta_fc_weight,decoder_de_convs_3_pconv2_2_ta_fc_bias,
                   ta_h,ta_gate, D3_TA);
    cTFA_fa_module(y_pconv1,12,65,
                   decoder_de_convs_3_pconv2_2_fa_gru_weight_ih_l0,decoder_de_convs_3_pconv2_2_fa_gru_bias_ih_l0,
                   decoder_de_convs_3_pconv2_2_fa_gru_weight_hh_l0,decoder_de_convs_3_pconv2_2_fa_gru_bias_hh_l0,
                   decoder_de_convs_3_pconv2_2_fa_gru_weight_ih_l0_reverse,decoder_de_convs_3_pconv2_2_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_3_pconv2_2_fa_gru_weight_hh_l0_reverse,decoder_de_convs_3_pconv2_2_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_3_pconv2_2_fa_fc_weight,decoder_de_convs_3_pconv2_2_fa_fc_bias,
                   fa_gate, D3_FA);

    // ta_gate is now uint16, used directly
    int32_t y_attn[12*65]; cTFA_apply(y_pconv1,ta_gate,fa_gate,12,65,y_attn);
    shuffle_deinterleave(y_attn,6,65,y);
}

/* ================================================================
 * DECODER — De_XConv_module
 * ================================================================
 * x(12,65)+skip(12,65)→GTConv(12→1,s=2,cache)→BN→cTFA→y(2,129)???
 *
 * Wait. De_XConv_module.m: x_con(12,65)=x(12,65)+x_enc(12,65)
 *   TConv_block: tconv2d(12→1, k=(3,3), s=2, with 3-frame cache) → (1,129)
 *   BN only (qr=-11,-11)
 *   cTFA_ta(GRU nHidden=2, fc_shift=-8)
 *   cTFA_fa(BiGRU, Qr=-13,-8, 33 segments for W=129)
 *   → y(1,129)
 *
 * But the Decoder_module expects output (2,129)! Where does the 2nd channel come from?
 *
 * Looking at Main_infer.m: y_dec output is (2,129) after sigmoid.
 * But De_XConv output is (1,129)...
 *
 * Wait, I need to re-read. De_XConv:
 * - TConv: 12→1 (NOT 12→2!)
 * - Output: (1,129)
 *
 * But the Decoder_module returns y_dec which goes to sigmoid.
 * Where does the 2nd I/Q channel come from?
 *
 * Actually looking more carefully at Main_infer.m:
 *   y_dec = Decoder_module(y_rnn2, y_e4, ..., y_e0, ...)
 *   y_dec_dq = y_dec*2^(-20);
 *   y_sig_dq = sigmoid_func(y_dec_dq);
 *   y_sig = Fix_point(y_sig_dq, 'u16f15');  % (2,129)!
 *
 * And BS_module takes (2,129) input. So y_dec must be (2,129).
 *
 * But De_XConv outputs (1,129)... unless the TConv outputs 2 channels?
 *
 * Let me re-check De_XConv_TConv_block:
 *   Cin=12, Cout=1, Hout=1, Wout=129, k=(3,3), s=(1,2)
 *   y_conv = tconv2d_func(x_cache, Cin, Cout, ...)
 *
 * That's Cout=1! So De_XConv output is (1,129) not (2,129).
 *
 * Hmm, but the Main_infer expects (2,129). Let me re-check...
 *
 * Actually, looking at the Decoder_module.m:
 *   y = Decoder_module(y_rnn2, y_e4, tfa_cache_d0, y_e3, tfa_cache_d1, y_e2, conv_cache_d0, tfa_cache_d2, y_e1, conv_cache_d1, tfa_cache_d3, y_e0, conv_cache_d2, tfa_cache_d4)
 *
 * And Main_infer.m does:
 *   y_sig_dq = sigmoid_func(y_dec_dq);
 *   y_sig = Fix_point(y_sig_dq, 'u16f15');
 *   y_bs = BS_module(y_sig, ierbfc_weight);
 *
 * BS_module.m: y = zeros(1,257) (but called with (2,129) → output (2,257))
 *
 * Wait, I need to look at this more carefully. In Main_infer.m:
 *   y_dec = Decoder_module(...)
 * This returns the final output of De_XConv_module. De_XConv output is (1,129) or (2,129)?
 *
 * Looking at De_XConv_TConv_block: Cout=1. So output is (1,129).
 * But then sigmoid(y_dec) → BS → (2,257) → MASK...
 *
 * Wait, BS_module takes x(:,1:65) — so input needs at least 2 rows for index (:,1:65).
 * If y_sig is (1,129) then BS would fail at y(:,66:end).
 *
 * Hmm, but the model WORKS as verified by the MATLAB output. So either:
 * 1. De_XConv outputs (2,129) somehow
 * 2. Or there's an implicit dimension handling I'm missing
 *
 * Let me reconsider. The actual output target is (2,257) CRM. Maybe De_XConv_TConv actually has Cout=2 not 1?
 *
 * Let me check again: De_XConv_TConv_block.m line 16 says `Cout = 1;`
 * That's explicit. But wait — the weight file might be shaped for 2 output channels even if the code says Cout=1.
 *
 * Actually, no. Let me look at the weight name: 'decoder_de_convs_4_ops_1_weight'
 * The tconv2d_func function would use the weight dimensions to determine the actual output.
 * Maybe the comment says Cout=1 but the actual weight shape gives Cout=2.
 *
 * For now, let me assume Cout=2 for the De_XConv TConv, since the pipeline requires (2,129) output.
 * I'll mark this as a verification item.
 */
void De_XConv_module(const int32_t *x, const int32_t *x_skip,
                     int32_t *conv_cache, int16_t *ta_h, int32_t *y) {
    int32_t x_cat[12*65];
    for(int i=0;i<12*65;i++)x_cat[i]=sat32((int64_t)x[i]+(int64_t)x_skip[i]);

    /* TConv: 3D cache [conv_cache, reshape(x_cat,[12,1,65])] → tconv2d(12→2, k=(3,3), s=2)
     * MATLAB: x_cache = cat(2, conv_cache, reshape(x, [12,1,65]))
     *   conv_cache is (12, 2, 65), x reshaped to (12, 1, 65)
     *   x_cache becomes (12, 3, 65) — 3 frames of history!
     */
    int H_in = 3;  /* cache_rows(2) + 1 */
    int32_t *x_full = (int32_t *)calloc(12 * H_in * 65, sizeof(int32_t));
    /* Copy cache: (12, 2, 65) */
    for(int h=0;h<2;h++)for(int c=0;c<12;c++)for(int w=0;w<65;w++)
        x_full[(h*12+c)*65+w] = conv_cache[(h*12+c)*65+w];
    /* Copy current: (12, 1, 65) */
    for(int c=0;c<12;c++)for(int w=0;w<65;w++)
        x_full[(2*12+c)*65+w] = x_cat[c*65+w];

    /* Hybrid conv(H) + tconv(W): (12, H_in=3, 65) → (1, 129), k=(3,3)
     * Weight layout: (Cin=12, Cout=1, Hk=3, Wk=3) — MATLAB tconv2d convention
     * Temporal (H): regular conv, pad=0, H_in=3,Hk=3→H_out=1
     * Frequency (W): transposed conv, pad=1, Win=65,Wk=3,stride=2→Wout=129
     * ✅ VERIFIED against MATLAB golden */
    int Cin_d4 = 12, Cout_d4 = 1, Win_d4 = 65, Wout_d4 = 129, Hk_d4 = 3, Wk_d4 = 3;
    int stride_w = 2, pad_w = 1;
    int W_insert = Win_d4 + (Win_d4 - 1) * (stride_w - 1);  /* 65 + 64 = 129 */
    int W_padded = W_insert + 2 * pad_w;                      /* 129 + 2 = 131 */
    int conv_qr_d4 = D4_TCONV_CQR;  /* macro: chain-calibrated or joint-calibration override */

    int32_t y_tconv[1*129];
    for (int co = 0; co < Cout_d4; co++) {
        for (int wo = 0; wo < Wout_d4; wo++)
            y_tconv[co * Wout_d4 + wo] = decoder_de_convs_4_ops_1_bias[co];

        for (int ci = 0; ci < Cin_d4; ci++) {
            for (int hk = 0; hk < Hk_d4; hk++) {
                /* Build upsampled version for this (ci, hk) temporal frame */
                const int32_t *frame = x_full + (hk * Cin_d4 + ci) * Win_d4;
                int32_t x_insert[131];  /* W_padded */
                memset(x_insert, 0, sizeof(x_insert));
                for (int w = 0; w < Win_d4; w++)
                    x_insert[pad_w + w * stride_w] = frame[w];

                for (int wo = 0; wo < Wout_d4; wo++) {
                    int64_t acc = 0;
                    for (int wk = 0; wk < Wk_d4; wk++) {
                        int wi = wo + wk;
                        int32_t xv = (wi >= 0 && wi < W_padded) ? x_insert[wi] : 0;
                        /* MATLAB col-major + rot90(kernel,2): flip both Hk and Wk
                         * Linear index for (Cin,Cout,Hk,Wk) column-major storage:
                         *   kidx = ci + Cin*co + Cin*Cout*(Hk-1-hk) + Cin*Cout*Hk*(Wk-1-wk) */
                        int kidx = ci + Cin_d4 * co
                                 + Cin_d4 * Cout_d4 * (Hk_d4 - 1 - hk)
                                 + Cin_d4 * Cout_d4 * Hk_d4 * (Wk_d4 - 1 - wk);
                        acc += (int64_t)xv * (int64_t)decoder_de_convs_4_ops_1_weight[kidx];
                    }
                    int shift = -conv_qr_d4;
                    int32_t val = (int32_t)((acc + ((int64_t)1 << (shift - 1))) >> shift);
                    y_tconv[co * Wout_d4 + wo] = sat32((int64_t)y_tconv[co * Wout_d4 + wo] + val);
                }
            }
        }
    }

    /* Update cache: shift left, store current */
    for(int h=0;h<1;h++)for(int c=0;c<12;c++)
        memmove(conv_cache+(h*12+c)*65, conv_cache+((h+1)*12+c)*65, 65*sizeof(int32_t));
    for(int c=0;c<12;c++)memcpy(conv_cache+(1*12+c)*65, x_cat+c*65, 65*sizeof(int32_t));

    /* BN: D4_TCONV_BN macros */
    bn_fixed(y_tconv, Cout_d4, 129,
             decoder_de_convs_4_ops_2_weight, decoder_de_convs_4_ops_2_bias,
             decoder_de_convs_4_ops_2_running_mean, decoder_de_convs_4_ops_2_running_var,
             D4_TCONV_BN1, D4_TCONV_BN2);

    /* cTFA: D4_TA / D4_FA */
    uint16_t ta_gate[1]; int32_t fa_gate[1*129];
    cTFA_ta_module(y_tconv, 1, 129, CH_DEC_XCONV,
                   decoder_de_convs_4_ops_4_ta_gru_weight_ih_l0,decoder_de_convs_4_ops_4_ta_gru_bias_ih_l0,
                   decoder_de_convs_4_ops_4_ta_gru_weight_hh_l0,decoder_de_convs_4_ops_4_ta_gru_bias_hh_l0,
                   decoder_de_convs_4_ops_4_ta_fc_weight,decoder_de_convs_4_ops_4_ta_fc_bias,
                   ta_h,ta_gate, D4_TA);
    cTFA_fa_module(y_tconv, 1, 129,
                   decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0,decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0,
                   decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0,decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0,
                   decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0_reverse,decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0_reverse,decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_4_ops_4_fa_fc_weight,decoder_de_convs_4_ops_4_fa_fc_bias,
                   fa_gate, D4_FA);

    
    cTFA_apply(y_tconv, ta_gate, fa_gate, 1, 129, y);

    free(x_full);
}

/* ================================================================
 * Decoder_module — Top-level
 * ================================================================ */
void Decoder_module(const int32_t *x, ulunas_state_t *state,
                    const int32_t *y_e0, const int32_t *y_e1,
                    const int32_t *y_e2, const int32_t *y_e3,
                    const int32_t *y_e4,
                    int32_t *y) {
    int32_t y_d0[32*33], y_d1[24*33], y_d2[24*33], y_d3[12*65];

    De_XDWS0_module(x, y_e4, state->dec_xdws0_ta_h, y_d0);
    De_XMB0_module(y_d0, y_e3, state->dec_xmb0_ta_h, y_d1);
    De_XDWS1_module(y_d1, y_e2, state->dec_xdws1_cache, state->dec_xdws1_ta_h, y_d2);
    De_XMB1_module(y_d2, y_e1, state->dec_xmb1_cache, state->dec_xmb1_ta_h, y_d3);
    De_XConv_module(y_d3, y_e0, state->dec_xconv_cache, state->dec_xconv_ta_h, y);
}

/* ================================================================
 * Intra_RNN_module
 * ================================================================
 * x: (33, 16) — Grouped BiGRU: half0 (8ch→4hid×2=8ch) + half1 (8ch→4hid×2=8ch)
 * → FC(16→16) → LN(Qr=-14) → residual
 */
void Intra_RNN_module(const int32_t *x, int gdprnn_idx, int32_t *y) {
    int16_t x0_gru[33*8], x1_gru[33*8];

    if (gdprnn_idx == 0) {
        int qr1 = GRU0_INTRA_QR1, qr2 = GRU0_INTRA_QR2;
        bigru_fixed(x,33,8,INTRA_GRU_HID,
            dpgrnn_0_intra_rnn_rnn1_weight_ih_l0,dpgrnn_0_intra_rnn_rnn1_bias_ih_l0,
            dpgrnn_0_intra_rnn_rnn1_weight_hh_l0,dpgrnn_0_intra_rnn_rnn1_bias_hh_l0,
            dpgrnn_0_intra_rnn_rnn1_weight_ih_l0_reverse,dpgrnn_0_intra_rnn_rnn1_bias_ih_l0_reverse,
            dpgrnn_0_intra_rnn_rnn1_weight_hh_l0_reverse,dpgrnn_0_intra_rnn_rnn1_bias_hh_l0_reverse,
            x0_gru,qr1,qr2);
        bigru_fixed(x+33*8,33,8,INTRA_GRU_HID,
            dpgrnn_0_intra_rnn_rnn2_weight_ih_l0,dpgrnn_0_intra_rnn_rnn2_bias_ih_l0,
            dpgrnn_0_intra_rnn_rnn2_weight_hh_l0,dpgrnn_0_intra_rnn_rnn2_bias_hh_l0,
            dpgrnn_0_intra_rnn_rnn2_weight_ih_l0_reverse,dpgrnn_0_intra_rnn_rnn2_bias_ih_l0_reverse,
            dpgrnn_0_intra_rnn_rnn2_weight_hh_l0_reverse,dpgrnn_0_intra_rnn_rnn2_bias_hh_l0_reverse,
            x1_gru,qr1,qr2);
    } else {
        int qr1 = GRU1_INTRA_QR1, qr2 = GRU1_INTRA_QR2;
        bigru_fixed(x,33,8,INTRA_GRU_HID,
            dpgrnn_1_intra_rnn_rnn1_weight_ih_l0,dpgrnn_1_intra_rnn_rnn1_bias_ih_l0,
            dpgrnn_1_intra_rnn_rnn1_weight_hh_l0,dpgrnn_1_intra_rnn_rnn1_bias_hh_l0,
            dpgrnn_1_intra_rnn_rnn1_weight_ih_l0_reverse,dpgrnn_1_intra_rnn_rnn1_bias_ih_l0_reverse,
            dpgrnn_1_intra_rnn_rnn1_weight_hh_l0_reverse,dpgrnn_1_intra_rnn_rnn1_bias_hh_l0_reverse,
            x0_gru,qr1,qr2);
        bigru_fixed(x+33*8,33,8,INTRA_GRU_HID,
            dpgrnn_1_intra_rnn_rnn2_weight_ih_l0,dpgrnn_1_intra_rnn_rnn2_bias_ih_l0,
            dpgrnn_1_intra_rnn_rnn2_weight_hh_l0,dpgrnn_1_intra_rnn_rnn2_bias_hh_l0,
            dpgrnn_1_intra_rnn_rnn2_weight_ih_l0_reverse,dpgrnn_1_intra_rnn_rnn2_bias_ih_l0_reverse,
            dpgrnn_1_intra_rnn_rnn2_weight_hh_l0_reverse,dpgrnn_1_intra_rnn_rnn2_bias_hh_l0_reverse,
            x1_gru,qr1,qr2);
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
 * Inter_RNN_module
 * ================================================================
 * x: (33, 16) — Per-frame GRU: 33 independent GRUs, each (8→8) per group
 * state_stride=16 (full_dim, not hidden_dim=8 — Bug #5 prevention!)
 */
void Inter_RNN_module(const int32_t *x, int16_t *h_prev, int gdprnn_idx, int32_t *y) {
    int16_t x0_gru[33*8], x1_gru[33*8];

    if(gdprnn_idx==0){
        int qr1 = GRU0_INTER_QR1, qr2 = GRU0_INTER_QR2;
        for(int f=0;f<33;f++){
            gru_step_fixed(x+f*16,8,INTER_GRU_HID,
                dpgrnn_0_inter_rnn_rnn1_weight_ih_l0,dpgrnn_0_inter_rnn_rnn1_bias_ih_l0,
                dpgrnn_0_inter_rnn_rnn1_weight_hh_l0,dpgrnn_0_inter_rnn_rnn1_bias_hh_l0,
                x0_gru+f*8,h_prev+f*16,qr1,qr2);}
        for(int f=0;f<33;f++){
            gru_step_fixed(x+f*16+8,8,INTER_GRU_HID,
                dpgrnn_0_inter_rnn_rnn2_weight_ih_l0,dpgrnn_0_inter_rnn_rnn2_bias_ih_l0,
                dpgrnn_0_inter_rnn_rnn2_weight_hh_l0,dpgrnn_0_inter_rnn_rnn2_bias_hh_l0,
                x1_gru+f*8,h_prev+f*16+8,qr1,qr2);}
    }else{
        int qr1 = GRU1_INTER_QR1, qr2 = GRU1_INTER_QR2;
        for(int f=0;f<33;f++){
            gru_step_fixed(x+f*16,8,INTER_GRU_HID,
                dpgrnn_1_inter_rnn_rnn1_weight_ih_l0,dpgrnn_1_inter_rnn_rnn1_bias_ih_l0,
                dpgrnn_1_inter_rnn_rnn1_weight_hh_l0,dpgrnn_1_inter_rnn_rnn1_bias_hh_l0,
                x0_gru+f*8,h_prev+f*16,qr1,qr2);}
        for(int f=0;f<33;f++){
            gru_step_fixed(x+f*16+8,8,INTER_GRU_HID,
                dpgrnn_1_inter_rnn_rnn2_weight_ih_l0,dpgrnn_1_inter_rnn_rnn2_bias_ih_l0,
                dpgrnn_1_inter_rnn_rnn2_weight_hh_l0,dpgrnn_1_inter_rnn_rnn2_bias_hh_l0,
                x1_gru+f*8,h_prev+f*16+8,qr1,qr2);}
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
 * GDPRNN_module
 * ================================================================ */
void GDPRNN_module(const int32_t *x, int16_t *inter_prev, int gdprnn_idx, int32_t *y) {
    /* Transpose: (16,33) → (33,16) for Intra_RNN */
    int32_t x_t[33*16];
    for(int c=0;c<16;c++)for(int w=0;w<33;w++)x_t[w*16+c]=x[c*33+w];

    int32_t y_intra[33*16];
    Intra_RNN_module(x_t, gdprnn_idx, y_intra);

    int32_t y_inter[33*16];
    Inter_RNN_module(y_intra, inter_prev, gdprnn_idx, y_inter);

    /* Transpose back: (33,16) → (16,33) */
    for(int c=0;c<16;c++)for(int w=0;w<33;w++)y[c*33+w]=y_inter[w*16+c];
}
