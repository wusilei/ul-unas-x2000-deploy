/**
 * ulunas_modules.c — UL-UNAS Module Assembly
 * ===========================================
 * Encoder/Decoder/GDPRNN module wrappers that call low-level operators
 * with the correct weights, dimensions, and Qr parameters.
 *
 * Each function reads weights from the globally-declared arrays
 * in ulunas_matlab_weights.h and calls the corresponding operators
 * matching the MATLAB block files 1:1.
 */

#include "ulunas_fp.h"
#include "ulunas_lut.h"
#include "qr_config.h"
#include "layer_dims.h"

/* Forward declare all weight arrays (defined in ulunas_matlab_weights.c) */
#include "ulunas_matlab_weights.h"

/* ================================================================
 * Encoder Layer 0: XConv
 * ================================================================
 * TConv(1→12, 3×3, s=[1,2]) → BN → AffinePReLU → cTFA
 * Input:  [1, 129]
 * Output: [12, 65]
 */
void encoder_layer0_xconv(const int32_t *x, ulunas_state_t *st, int32_t *y) {
    int32_t y_tconv[12 * 65];

    /* TConv: conv2d with cache */
    /* MATLAB XConv_TConv_block:
     *   x_c = [conv_cache; x]  → [2+1, 129] = [3, 129]
     *   y_conv = conv2d_func(x_c, Cin=1, Cout=12, Hout=1, Wout=65, [3,3], [1,2], weight, bias, -14)
     *   conv_cache = x_c(2:3,:)  → keep last 2 rows
     */
    int W_in = 129;
    int cache_h = 2;
    /* Build x_c: [cache; x], x is [1, 129] */
    int32_t x_c[3 * 129];  /* [3, 129] */
    memcpy(x_c, st->conv_cache_e0, cache_h * W_in * sizeof(int32_t));
    memcpy(x_c + cache_h * W_in, x, 1 * W_in * sizeof(int32_t));

    conv2d_func(x_c, 1, 12, 1, 65, 3, 3, 1, 2,
                encoder_en_convs_0_ops_1_weight, encoder_en_convs_0_ops_1_bias,
                E0_TCONV_CONV_QR, y_tconv);

    /* Update cache: keep last 2 rows of x_c */
    memcpy(st->conv_cache_e0, x_c + (3 - 2) * W_in, 2 * W_in * sizeof(int32_t));

    /* BN */
    int32_t y_bn[12 * 65];
    bn_func(y_tconv, encoder_en_convs_0_ops_2_weight, encoder_en_convs_0_ops_2_bias,
            encoder_en_convs_0_ops_2_running_mean, encoder_en_convs_0_ops_2_running_var,
            E0_TCONV_BN_QR1, E0_TCONV_BN_QR2, 12, 12 * 65, y_bn);

    /* AffinePReLU */
    int32_t y_ap[12 * 65];
    affineprelu_func(y_bn, encoder_en_convs_0_ops_3_affine_weight,
                     encoder_en_convs_0_ops_3_affine_bias,
                     encoder_en_convs_0_ops_3_slope_weight,
                     E0_TCONV_AFFINE_QR1, E0_TCONV_AFFINE_QR2, 12, 65, y_ap);

    /* cTFA TA */
    uint16_t y_ta[12];
    ctfa_ta_module(y_ap, 12, 65, E0_CTFA_TA_GRU_NHID,
                   st->tfa_cache_e0,
                   encoder_en_convs_0_ops_4_ta_gru_weight_ih_l0,
                   encoder_en_convs_0_ops_4_ta_gru_bias_ih_l0,
                   encoder_en_convs_0_ops_4_ta_gru_weight_hh_l0,
                   encoder_en_convs_0_ops_4_ta_gru_bias_hh_l0,
                   encoder_en_convs_0_ops_4_ta_fc_weight,
                   encoder_en_convs_0_ops_4_ta_fc_bias,
                   E0_CTFA_TA_GRU_QR1, E0_CTFA_TA_GRU_QR2, E0_CTFA_TA_FC_QR,
                   y_ta);

    /* cTFA FA */
    uint16_t y_fa[65];
    ctfa_fa_module(y_ap, 12, 65, E0_CTFA_FA_GRU_NHID,
                   E0_CTFA_FA_GROUP, E0_CTFA_FA_SEG, E0_CTFA_FA_PAD,
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
                   E0_CTFA_FA_GRU_QR1, E0_CTFA_FA_GRU_QR2, E0_CTFA_FA_FC_QR,
                   y_fa);

    /* cTFA fusion: y_t = round(tconv .* ta' * 2^(-15)); y = round(y_t .* fa * 2^(-15)) */
    int64_t r_fusion = 16384;  /* 2^14 for >> 15 */
    for (int c = 0; c < 12; c++) {
        for (int w = 0; w < 65; w++) {
            int64_t prod1 = (int64_t)y_ap[c * 65 + w] * y_ta[c];
            int32_t y_t;
            if (prod1 >= 0) y_t = (int32_t)((prod1 + r_fusion) >> 15);
            else            y_t = (int32_t)((prod1 - r_fusion) >> 15);
            int64_t prod2 = (int64_t)y_t * y_fa[w];
            if (prod2 >= 0) y[c * 65 + w] = (int32_t)((prod2 + r_fusion) >> 15);
            else            y[c * 65 + w] = (int32_t)((prod2 - r_fusion) >> 15);
        }
    }
}

/* ================================================================
 * Encoder Layer 1: XMB0
 * ================================================================
 * PConv0(groups=2, 12→24) → Shuffle → TConv(k=2×3, s=[1,2]) → PConv1(groups=2) → cTFA → Shuffle
 * Input:  [12, 65]
 * Output: [24, 33]
 */
void encoder_layer1_xmb0(const int32_t *x, ulunas_state_t *st, int32_t *y) {
    int32_t y_pconv0[24 * 65];
    int32_t y_s[24 * 65];
    int32_t y_tconv[24 * 33];
    int32_t y_pconv1[24 * 33];

    /* PConv0: grouped PConv (2 groups, Cin=6→Cout=12 each) */
    /* Group 0: x[0:6, :] → pconv with weight[0:12, :] */
    pconv2d_func(x, 6, 12, 1, 65,
                 encoder_en_convs_1_pconv1_0_weight,      /* first 12 output channels */
                 encoder_en_convs_1_pconv1_0_bias,
                 E1_PCONV0_CONV_QR, y_pconv0);
    /* Group 1: x[6:12, :] → pconv with weight[12:24, :] */
    pconv2d_func(x + 6 * 65, 6, 12, 1, 65,
                 encoder_en_convs_1_pconv1_0_weight + 12 * 6,  /* next 12 output channels */
                 encoder_en_convs_1_pconv1_0_bias + 12,
                 E1_PCONV0_CONV_QR, y_pconv0 + 12 * 65);

    /* BN */
    int32_t y_bn[24 * 65];
    bn_func(y_pconv0, encoder_en_convs_1_pconv1_1_weight, encoder_en_convs_1_pconv1_1_bias,
            encoder_en_convs_1_pconv1_1_running_mean, encoder_en_convs_1_pconv1_1_running_var,
            E1_PCONV0_BN_QR1, E1_PCONV0_BN_QR2, 24, 24 * 65, y_bn);

    /* AffinePReLU */
    int32_t y_ap[24 * 65];
    affineprelu_func(y_bn, encoder_en_convs_1_pconv1_2_affine_weight,
                     encoder_en_convs_1_pconv1_2_affine_bias,
                     encoder_en_convs_1_pconv1_2_slope_weight,
                     E1_PCONV0_AFFINE_QR1, E1_PCONV0_AFFINE_QR2, 24, 65, y_ap);

    /* Shuffle: interleave */
    shuffle_interleave(y_ap, 24, 65, y_s);

    /* TConv: gconv with cache [24, 65] → [24, 33] (s=[1,2]) */
    /* Use x=y_s (current frame, 24×65), cache=conv_cache_e1 (24×65) */
    gconv2d_func(y_s, 24, 1, 33, 2, 3, 1, 2,
                 encoder_en_convs_1_dconv_1_weight, encoder_en_convs_1_dconv_1_bias,
                 E1_TCONV_CONV_QR,
                 st->conv_cache_e1, y_tconv);  /* cache updated inside */

    /* BN */
    int32_t y_tconv_bn[24 * 33];
    bn_func(y_tconv, encoder_en_convs_1_dconv_2_weight, encoder_en_convs_1_dconv_2_bias,
            encoder_en_convs_1_dconv_2_running_mean, encoder_en_convs_1_dconv_2_running_var,
            E1_TCONV_BN_QR1, E1_TCONV_BN_QR2, 24, 24 * 33, y_tconv_bn);

    /* AffinePReLU */
    int32_t y_tconv_ap[24 * 33];
    affineprelu_func(y_tconv_bn, encoder_en_convs_1_dconv_3_affine_weight,
                     encoder_en_convs_1_dconv_3_affine_bias,
                     encoder_en_convs_1_dconv_3_slope_weight,
                     E1_TCONV_AFFINE_QR1, E1_TCONV_AFFINE_QR2, 24, 33, y_tconv_ap);

    /* PConv1: grouped PConv (2 groups, Cin=12→Cout=12 each), BN only */
    pconv2d_func(y_tconv_ap, 12, 12, 1, 33,
                 encoder_en_convs_1_pconv2_0_weight,
                 encoder_en_convs_1_pconv2_0_bias,
                 E1_PCONV1_CONV_QR, y_pconv1);
    pconv2d_func(y_tconv_ap + 12 * 33, 12, 12, 1, 33,
                 encoder_en_convs_1_pconv2_0_weight + 12 * 12,
                 encoder_en_convs_1_pconv2_0_bias + 12,
                 E1_PCONV1_CONV_QR, y_pconv1 + 12 * 33);
    bn_func(y_pconv1, encoder_en_convs_1_pconv2_1_weight, encoder_en_convs_1_pconv2_1_bias,
            encoder_en_convs_1_pconv2_1_running_mean, encoder_en_convs_1_pconv2_1_running_var,
            E1_PCONV1_BN_QR1, E1_PCONV1_BN_QR2, 24, 24 * 33, y_pconv1);  /* in-place BN on y_pconv1 */

    /* cTFA TA (on y_pconv1) */
    uint16_t y_ta[24];
    ctfa_ta_module(y_pconv1, 24, 33, E1_CTFA_TA_GRU_NHID,
                   st->tfa_cache_e1,
                   encoder_en_convs_1_pconv2_2_ta_gru_weight_ih_l0,
                   encoder_en_convs_1_pconv2_2_ta_gru_bias_ih_l0,
                   encoder_en_convs_1_pconv2_2_ta_gru_weight_hh_l0,
                   encoder_en_convs_1_pconv2_2_ta_gru_bias_hh_l0,
                   encoder_en_convs_1_pconv2_2_ta_fc_weight,
                   encoder_en_convs_1_pconv2_2_ta_fc_bias,
                   E1_CTFA_TA_GRU_QR1, E1_CTFA_TA_GRU_QR2, E1_CTFA_TA_FC_QR, y_ta);

    /* cTFA FA */
    uint16_t y_fa[33];
    ctfa_fa_module(y_pconv1, 24, 33, E1_CTFA_FA_GRU_NHID,
                   E1_CTFA_FA_GROUP, E1_CTFA_FA_SEG, E1_CTFA_FA_PAD,
                   encoder_en_convs_1_pconv2_2_fa_gru_weight_ih_l0,
                   encoder_en_convs_1_pconv2_2_fa_gru_bias_ih_l0,
                   encoder_en_convs_1_pconv2_2_fa_gru_weight_hh_l0,
                   encoder_en_convs_1_pconv2_2_fa_gru_bias_hh_l0,
                   encoder_en_convs_1_pconv2_2_fa_gru_weight_ih_l0_reverse,
                   encoder_en_convs_1_pconv2_2_fa_gru_bias_ih_l0_reverse,
                   encoder_en_convs_1_pconv2_2_fa_gru_weight_hh_l0_reverse,
                   encoder_en_convs_1_pconv2_2_fa_gru_bias_hh_l0_reverse,
                   encoder_en_convs_1_pconv2_2_fa_fc_weight,
                   encoder_en_convs_1_pconv2_2_fa_fc_bias,
                   E1_CTFA_FA_GRU_QR1, E1_CTFA_FA_GRU_QR2, E1_CTFA_FA_FC_QR, y_fa);

    /* cTFA fusion + final shuffle */
    int32_t y_ctfa[24 * 33];
    int64_t r = 16384;
    for (int c = 0; c < 24; c++) {
        for (int w = 0; w < 33; w++) {
            int64_t p1 = (int64_t)y_pconv1[c * 33 + w] * y_ta[c];
            int32_t yt;
            if (p1 >= 0) yt = (int32_t)((p1 + r) >> 15);
            else         yt = (int32_t)((p1 - r) >> 15);
            int64_t p2 = (int64_t)yt * y_fa[w];
            if (p2 >= 0) y_ctfa[c * 33 + w] = (int32_t)((p2 + r) >> 15);
            else         y_ctfa[c * 33 + w] = (int32_t)((p2 - r) >> 15);
        }
    }

    /* Final shuffle: interleave for output */
    shuffle_interleave(y_ctfa, 24, 33, y);
}

/* ================================================================
 * Encoder Layer 2: XDWS0
 * ================================================================
 * PConv(groups=2, 24→24) → Shuffle → TConv(k=2×3, s=[1,1]) → cTFA
 * Input:  [24, 33]
 * Output: [24, 33]
 */
void encoder_layer2_xdws0(const int32_t *x, ulunas_state_t *st, int32_t *y) {
    int32_t y_pconv[24 * 33];
    int32_t y_s[24 * 33];
    int32_t y_tconv[24 * 33];

    /* PConv: groups=2, Cin=12→Cout=12 each */
    pconv2d_func(x, 12, 12, 1, 33,
                 encoder_en_convs_2_pconv_0_weight,
                 encoder_en_convs_2_pconv_0_bias,
                 E2_PCONV_CONV_QR, y_pconv);
    pconv2d_func(x + 12 * 33, 12, 12, 1, 33,
                 encoder_en_convs_2_pconv_0_weight + 12 * 12,
                 encoder_en_convs_2_pconv_0_bias + 12,
                 E2_PCONV_CONV_QR, y_pconv + 12 * 33);

    /* BN */
    int32_t y_bn[24 * 33];
    bn_func(y_pconv, encoder_en_convs_2_pconv_1_weight, encoder_en_convs_2_pconv_1_bias,
            encoder_en_convs_2_pconv_1_running_mean, encoder_en_convs_2_pconv_1_running_var,
            E2_PCONV_BN_QR1, E2_PCONV_BN_QR2, 24, 24 * 33, y_bn);

    /* AffinePReLU */
    int32_t y_ap[24 * 33];
    affineprelu_func(y_bn, encoder_en_convs_2_pconv_2_affine_weight,
                     encoder_en_convs_2_pconv_2_affine_bias,
                     encoder_en_convs_2_pconv_2_slope_weight,
                     E2_PCONV_AFFINE_QR1, E2_PCONV_AFFINE_QR2, 24, 33, y_ap);

    /* Shuffle */
    shuffle_interleave(y_ap, 24, 33, y_s);

    /* TConv: gconv, s=[1,1] → output same W */
    gconv2d_func(y_s, 24, 1, 33, 2, 3, 1, 1,
                 encoder_en_convs_2_dconv_1_weight, encoder_en_convs_2_dconv_1_bias,
                 E2_TCONV_CONV_QR,
                 st->conv_cache_e2, y_tconv);

    /* BN */
    int32_t y_tconv_bn[24 * 33];
    bn_func(y_tconv, encoder_en_convs_2_dconv_2_weight, encoder_en_convs_2_dconv_2_bias,
            encoder_en_convs_2_dconv_2_running_mean, encoder_en_convs_2_dconv_2_running_var,
            E2_TCONV_BN_QR1, E2_TCONV_BN_QR2, 24, 24 * 33, y_tconv_bn);

    /* AffinePReLU */
    int32_t y_tconv_ap[24 * 33];
    affineprelu_func(y_tconv_bn, encoder_en_convs_2_dconv_3_affine_weight,
                     encoder_en_convs_2_dconv_3_affine_bias,
                     encoder_en_convs_2_dconv_3_slope_weight,
                     E2_TCONV_AFFINE_QR1, E2_TCONV_AFFINE_QR2, 24, 33, y_tconv_ap);

    /* cTFA TA */
    uint16_t y_ta[24];
    ctfa_ta_module(y_tconv_ap, 24, 33, E2_CTFA_TA_GRU_NHID,
                   st->tfa_cache_e2,
                   encoder_en_convs_2_dconv_4_ta_gru_weight_ih_l0,
                   encoder_en_convs_2_dconv_4_ta_gru_bias_ih_l0,
                   encoder_en_convs_2_dconv_4_ta_gru_weight_hh_l0,
                   encoder_en_convs_2_dconv_4_ta_gru_bias_hh_l0,
                   encoder_en_convs_2_dconv_4_ta_fc_weight,
                   encoder_en_convs_2_dconv_4_ta_fc_bias,
                   E2_CTFA_TA_GRU_QR1, E2_CTFA_TA_GRU_QR2, E2_CTFA_TA_FC_QR, y_ta);

    /* cTFA FA */
    uint16_t y_fa[33];
    ctfa_fa_module(y_tconv_ap, 24, 33, E2_CTFA_FA_GRU_NHID,
                   E2_CTFA_FA_GROUP, E2_CTFA_FA_SEG, E2_CTFA_FA_PAD,
                   encoder_en_convs_2_dconv_4_fa_gru_weight_ih_l0,
                   encoder_en_convs_2_dconv_4_fa_gru_bias_ih_l0,
                   encoder_en_convs_2_dconv_4_fa_gru_weight_hh_l0,
                   encoder_en_convs_2_dconv_4_fa_gru_bias_hh_l0,
                   encoder_en_convs_2_dconv_4_fa_gru_weight_ih_l0_reverse,
                   encoder_en_convs_2_dconv_4_fa_gru_bias_ih_l0_reverse,
                   encoder_en_convs_2_dconv_4_fa_gru_weight_hh_l0_reverse,
                   encoder_en_convs_2_dconv_4_fa_gru_bias_hh_l0_reverse,
                   encoder_en_convs_2_dconv_4_fa_fc_weight,
                   encoder_en_convs_2_dconv_4_fa_fc_bias,
                   E2_CTFA_FA_GRU_QR1, E2_CTFA_FA_GRU_QR2, E2_CTFA_FA_FC_QR, y_fa);

    /* cTFA fusion */
    int64_t r = 16384;
    for (int c = 0; c < 24; c++) {
        for (int w = 0; w < 33; w++) {
            int64_t p1 = (int64_t)y_tconv_ap[c * 33 + w] * y_ta[c];
            int32_t yt;
            if (p1 >= 0) yt = (int32_t)((p1 + r) >> 15);
            else         yt = (int32_t)((p1 - r) >> 15);
            int64_t p2 = (int64_t)yt * y_fa[w];
            if (p2 >= 0) y[c * 33 + w] = (int32_t)((p2 + r) >> 15);
            else         y[c * 33 + w] = (int32_t)((p2 - r) >> 15);
        }
    }
}

/* ================================================================
 * Encoder Layer 3: XMB1
 * ================================================================
 * PConv0(groups=2, 24→32) → Shuffle → nonTConv(1×5) → PConv1(groups=2) → cTFA → Shuffle
 * Input:  [24, 33]
 * Output: [32, 33]
 */
void encoder_layer3_xmb1(const int32_t *x, ulunas_state_t *st, int32_t *y) {
    int32_t y_pconv0[32 * 33];
    int32_t y_s[32 * 33];
    int32_t y_tconv[32 * 33];
    int32_t y_pconv1[32 * 33];

    /* PConv0: groups=2, Cin=12→Cout=16 each */
    pconv2d_func(x, 12, 16, 1, 33,
                 encoder_en_convs_3_pconv1_0_weight,
                 encoder_en_convs_3_pconv1_0_bias,
                 E3_PCONV0_CONV_QR, y_pconv0);
    pconv2d_func(x + 12 * 33, 12, 16, 1, 33,
                 encoder_en_convs_3_pconv1_0_weight + 16 * 12,
                 encoder_en_convs_3_pconv1_0_bias + 16,
                 E3_PCONV0_CONV_QR, y_pconv0 + 16 * 33);

    /* BN */
    int32_t y_bn0[32 * 33];
    bn_func(y_pconv0, encoder_en_convs_3_pconv1_1_weight, encoder_en_convs_3_pconv1_1_bias,
            encoder_en_convs_3_pconv1_1_running_mean, encoder_en_convs_3_pconv1_1_running_var,
            E3_PCONV0_BN_QR1, E3_PCONV0_BN_QR2, 32, 32 * 33, y_bn0);

    /* AffinePReLU */
    int32_t y_ap0[32 * 33];
    affineprelu_func(y_bn0, encoder_en_convs_3_pconv1_2_affine_weight,
                     encoder_en_convs_3_pconv1_2_affine_bias,
                     encoder_en_convs_3_pconv1_2_slope_weight,
                     E3_PCONV0_AFFINE_QR1, E3_PCONV0_AFFINE_QR2, 32, 33, y_ap0);

    /* Shuffle */
    shuffle_interleave(y_ap0, 32, 33, y_s);

    /* nonTConv: no cache, 1×5 kernel */
    non_gconv2d_func(y_s, 32, 1, 33, 1, 5, 1, 1,
                     encoder_en_convs_3_dconv_1_weight, encoder_en_convs_3_dconv_1_bias,
                     E3_NONTCONV_CONV_QR, y_tconv);

    /* BN */
    int32_t y_tconv_bn[32 * 33];
    bn_func(y_tconv, encoder_en_convs_3_dconv_2_weight, encoder_en_convs_3_dconv_2_bias,
            encoder_en_convs_3_dconv_2_running_mean, encoder_en_convs_3_dconv_2_running_var,
            E3_NONTCONV_BN_QR1, E3_NONTCONV_BN_QR2, 32, 32 * 33, y_tconv_bn);

    /* AffinePReLU */
    int32_t y_tconv_ap[32 * 33];
    affineprelu_func(y_tconv_bn, encoder_en_convs_3_dconv_3_affine_weight,
                     encoder_en_convs_3_dconv_3_affine_bias,
                     encoder_en_convs_3_dconv_3_slope_weight,
                     E3_NONTCONV_AFFINE_QR1, E3_NONTCONV_AFFINE_QR2, 32, 33, y_tconv_ap);

    /* PConv1: groups=2, Cin=16→Cout=16 each, BN only */
    pconv2d_func(y_tconv_ap, 16, 16, 1, 33,
                 encoder_en_convs_3_pconv2_0_weight,
                 encoder_en_convs_3_pconv2_0_bias,
                 E3_PCONV1_CONV_QR, y_pconv1);
    pconv2d_func(y_tconv_ap + 16 * 33, 16, 16, 1, 33,
                 encoder_en_convs_3_pconv2_0_weight + 16 * 16,
                 encoder_en_convs_3_pconv2_0_bias + 16,
                 E3_PCONV1_CONV_QR, y_pconv1 + 16 * 33);
    bn_func(y_pconv1, encoder_en_convs_3_pconv2_1_weight, encoder_en_convs_3_pconv2_1_bias,
            encoder_en_convs_3_pconv2_1_running_mean, encoder_en_convs_3_pconv2_1_running_var,
            E3_PCONV1_BN_QR1, E3_PCONV1_BN_QR2, 32, 32 * 33, y_pconv1);

    /* cTFA TA */
    uint16_t y_ta[32];
    ctfa_ta_module(y_pconv1, 32, 33, E3_CTFA_TA_GRU_NHID,
                   st->tfa_cache_e3,
                   encoder_en_convs_3_pconv2_2_ta_gru_weight_ih_l0,
                   encoder_en_convs_3_pconv2_2_ta_gru_bias_ih_l0,
                   encoder_en_convs_3_pconv2_2_ta_gru_weight_hh_l0,
                   encoder_en_convs_3_pconv2_2_ta_gru_bias_hh_l0,
                   encoder_en_convs_3_pconv2_2_ta_fc_weight,
                   encoder_en_convs_3_pconv2_2_ta_fc_bias,
                   E3_CTFA_TA_GRU_QR1, E3_CTFA_TA_GRU_QR2, E3_CTFA_TA_FC_QR, y_ta);

    /* cTFA FA */
    uint16_t y_fa[33];
    ctfa_fa_module(y_pconv1, 32, 33, E3_CTFA_FA_GRU_NHID,
                   E3_CTFA_FA_GROUP, E3_CTFA_FA_SEG, E3_CTFA_FA_PAD,
                   encoder_en_convs_3_pconv2_2_fa_gru_weight_ih_l0,
                   encoder_en_convs_3_pconv2_2_fa_gru_bias_ih_l0,
                   encoder_en_convs_3_pconv2_2_fa_gru_weight_hh_l0,
                   encoder_en_convs_3_pconv2_2_fa_gru_bias_hh_l0,
                   encoder_en_convs_3_pconv2_2_fa_gru_weight_ih_l0_reverse,
                   encoder_en_convs_3_pconv2_2_fa_gru_bias_ih_l0_reverse,
                   encoder_en_convs_3_pconv2_2_fa_gru_weight_hh_l0_reverse,
                   encoder_en_convs_3_pconv2_2_fa_gru_bias_hh_l0_reverse,
                   encoder_en_convs_3_pconv2_2_fa_fc_weight,
                   encoder_en_convs_3_pconv2_2_fa_fc_bias,
                   E3_CTFA_FA_GRU_QR1, E3_CTFA_FA_GRU_QR2, E3_CTFA_FA_FC_QR, y_fa);

    /* cTFA fusion */
    int32_t y_ctfa[32 * 33];
    int64_t r = 16384;
    for (int c = 0; c < 32; c++) {
        for (int w = 0; w < 33; w++) {
            int64_t p1 = (int64_t)y_pconv1[c * 33 + w] * y_ta[c];
            int32_t yt;
            if (p1 >= 0) yt = (int32_t)((p1 + r) >> 15);
            else         yt = (int32_t)((p1 - r) >> 15);
            int64_t p2 = (int64_t)yt * y_fa[w];
            if (p2 >= 0) y_ctfa[c * 33 + w] = (int32_t)((p2 + r) >> 15);
            else         y_ctfa[c * 33 + w] = (int32_t)((p2 - r) >> 15);
        }
    }

    /* Final shuffle */
    shuffle_interleave(y_ctfa, 32, 33, y);
}

/* ================================================================
 * Encoder Layer 4: XDWS1
 * ================================================================
 * PConv(groups=2, 32→16) → Shuffle → nonTConv(1×5) → cTFA
 * Input:  [32, 33]
 * Output: [16, 33]
 */
void encoder_layer4_xdws1(const int32_t *x, ulunas_state_t *st, int32_t *y) {
    int32_t y_pconv[16 * 33];
    int32_t y_s[16 * 33];
    int32_t y_tconv[16 * 33];

    /* PConv: groups=2, Cin=16→Cout=8 each */
    pconv2d_func(x, 16, 8, 1, 33,
                 encoder_en_convs_4_pconv_0_weight,
                 encoder_en_convs_4_pconv_0_bias,
                 E4_PCONV_CONV_QR, y_pconv);
    pconv2d_func(x + 16 * 33, 16, 8, 1, 33,
                 encoder_en_convs_4_pconv_0_weight + 8 * 16,
                 encoder_en_convs_4_pconv_0_bias + 8,
                 E4_PCONV_CONV_QR, y_pconv + 8 * 33);

    /* BN */
    int32_t y_bn[16 * 33];
    bn_func(y_pconv, encoder_en_convs_4_pconv_1_weight, encoder_en_convs_4_pconv_1_bias,
            encoder_en_convs_4_pconv_1_running_mean, encoder_en_convs_4_pconv_1_running_var,
            E4_PCONV_BN_QR1, E4_PCONV_BN_QR2, 16, 16 * 33, y_bn);

    /* AffinePReLU */
    int32_t y_ap[16 * 33];
    affineprelu_func(y_bn, encoder_en_convs_4_pconv_2_affine_weight,
                     encoder_en_convs_4_pconv_2_affine_bias,
                     encoder_en_convs_4_pconv_2_slope_weight,
                     E4_PCONV_AFFINE_QR1, E4_PCONV_AFFINE_QR2, 16, 33, y_ap);

    /* Shuffle */
    shuffle_interleave(y_ap, 16, 33, y_s);

    /* nonTConv */
    non_gconv2d_func(y_s, 16, 1, 33, 1, 5, 1, 1,
                     encoder_en_convs_4_dconv_1_weight, encoder_en_convs_4_dconv_1_bias,
                     E4_NONTCONV_CONV_QR, y_tconv);

    /* BN */
    int32_t y_tconv_bn[16 * 33];
    bn_func(y_tconv, encoder_en_convs_4_dconv_2_weight, encoder_en_convs_4_dconv_2_bias,
            encoder_en_convs_4_dconv_2_running_mean, encoder_en_convs_4_dconv_2_running_var,
            E4_NONTCONV_BN_QR1, E4_NONTCONV_BN_QR2, 16, 16 * 33, y_tconv_bn);

    /* AffinePReLU */
    int32_t y_tconv_ap[16 * 33];
    affineprelu_func(y_tconv_bn, encoder_en_convs_4_dconv_3_affine_weight,
                     encoder_en_convs_4_dconv_3_affine_bias,
                     encoder_en_convs_4_dconv_3_slope_weight,
                     E4_NONTCONV_AFFINE_QR1, E4_NONTCONV_AFFINE_QR2, 16, 33, y_tconv_ap);

    /* cTFA TA */
    uint16_t y_ta[16];
    ctfa_ta_module(y_tconv_ap, 16, 33, E4_CTFA_TA_GRU_NHID,
                   st->tfa_cache_e4,
                   encoder_en_convs_4_dconv_4_ta_gru_weight_ih_l0,
                   encoder_en_convs_4_dconv_4_ta_gru_bias_ih_l0,
                   encoder_en_convs_4_dconv_4_ta_gru_weight_hh_l0,
                   encoder_en_convs_4_dconv_4_ta_gru_bias_hh_l0,
                   encoder_en_convs_4_dconv_4_ta_fc_weight,
                   encoder_en_convs_4_dconv_4_ta_fc_bias,
                   E4_CTFA_TA_GRU_QR1, E4_CTFA_TA_GRU_QR2, E4_CTFA_TA_FC_QR, y_ta);

    /* cTFA FA */
    uint16_t y_fa[33];
    ctfa_fa_module(y_tconv_ap, 16, 33, E4_CTFA_FA_GRU_NHID,
                   E4_CTFA_FA_GROUP, E4_CTFA_FA_SEG, E4_CTFA_FA_PAD,
                   encoder_en_convs_4_dconv_4_fa_gru_weight_ih_l0,
                   encoder_en_convs_4_dconv_4_fa_gru_bias_ih_l0,
                   encoder_en_convs_4_dconv_4_fa_gru_weight_hh_l0,
                   encoder_en_convs_4_dconv_4_fa_gru_bias_hh_l0,
                   encoder_en_convs_4_dconv_4_fa_gru_weight_ih_l0_reverse,
                   encoder_en_convs_4_dconv_4_fa_gru_bias_ih_l0_reverse,
                   encoder_en_convs_4_dconv_4_fa_gru_weight_hh_l0_reverse,
                   encoder_en_convs_4_dconv_4_fa_gru_bias_hh_l0_reverse,
                   encoder_en_convs_4_dconv_4_fa_fc_weight,
                   encoder_en_convs_4_dconv_4_fa_fc_bias,
                   E4_CTFA_FA_GRU_QR1, E4_CTFA_FA_GRU_QR2, E4_CTFA_FA_FC_QR, y_fa);

    /* cTFA fusion */
    int64_t r = 16384;
    for (int c = 0; c < 16; c++) {
        for (int w = 0; w < 33; w++) {
            int64_t p1 = (int64_t)y_tconv_ap[c * 33 + w] * y_ta[c];
            int32_t yt;
            if (p1 >= 0) yt = (int32_t)((p1 + r) >> 15);
            else         yt = (int32_t)((p1 - r) >> 15);
            int64_t p2 = (int64_t)yt * y_fa[w];
            if (p2 >= 0) y[c * 33 + w] = (int32_t)((p2 + r) >> 15);
            else         y[c * 33 + w] = (int32_t)((p2 - r) >> 15);
        }
    }
}

/* ================================================================
 * Encoder Module: all 5 layers
 * ================================================================ */
void encoder_module(const int32_t *x, ulunas_state_t *st,
                    int32_t *e0, int32_t *e1, int32_t *e2, int32_t *e3, int32_t *e4) {
    encoder_layer0_xconv(x, st, e0);
    encoder_layer1_xmb0(e0, st, e1);
    encoder_layer2_xdws0(e1, st, e2);
    encoder_layer3_xmb1(e2, st, e3);
    encoder_layer4_xdws1(e3, st, e4);
}

/* ================================================================
 * GDPRNN: Intra-RNN
 * ================================================================
 * Split x[33][16] into x0[33][8], x1[33][8]
 * Each: BiGRU(nHidden=4) → [33][8]
 * Concat → [33][16]
 * FC(Qr=-9) → LN(Qr=-14) → residual
 */
void intra_rnn_module(const int32_t *x, int gdprnn_idx, int32_t *y) {
    /* Select weights based on gdprnn_idx */
    const int16_t *rnn1_ih_w, *rnn1_hh_w, *rnn1_re_ih_w, *rnn1_re_hh_w;
    const int32_t *rnn1_ih_b, *rnn1_hh_b, *rnn1_re_ih_b, *rnn1_re_hh_b;
    const int16_t *rnn2_ih_w, *rnn2_hh_w, *rnn2_re_ih_w, *rnn2_re_hh_w;
    const int32_t *rnn2_ih_b, *rnn2_hh_b, *rnn2_re_ih_b, *rnn2_re_hh_b;
    const int16_t *fc_w, *ln_w;
    const int32_t *ln_b;
    const int32_t *fc_b;

    if (gdprnn_idx == 0) {
        rnn1_ih_w = dpgrnn_0_intra_rnn_rnn1_weight_ih_l0;
        rnn1_ih_b = dpgrnn_0_intra_rnn_rnn1_bias_ih_l0;
        rnn1_hh_w = dpgrnn_0_intra_rnn_rnn1_weight_hh_l0;
        rnn1_hh_b = dpgrnn_0_intra_rnn_rnn1_bias_hh_l0;
        rnn1_re_ih_w = dpgrnn_0_intra_rnn_rnn1_weight_ih_l0_reverse;
        rnn1_re_ih_b = dpgrnn_0_intra_rnn_rnn1_bias_ih_l0_reverse;
        rnn1_re_hh_w = dpgrnn_0_intra_rnn_rnn1_weight_hh_l0_reverse;
        rnn1_re_hh_b = dpgrnn_0_intra_rnn_rnn1_bias_hh_l0_reverse;
        rnn2_ih_w = dpgrnn_0_intra_rnn_rnn2_weight_ih_l0;
        rnn2_ih_b = dpgrnn_0_intra_rnn_rnn2_bias_ih_l0;
        rnn2_hh_w = dpgrnn_0_intra_rnn_rnn2_weight_hh_l0;
        rnn2_hh_b = dpgrnn_0_intra_rnn_rnn2_bias_hh_l0;
        rnn2_re_ih_w = dpgrnn_0_intra_rnn_rnn2_weight_ih_l0_reverse;
        rnn2_re_ih_b = dpgrnn_0_intra_rnn_rnn2_bias_ih_l0_reverse;
        rnn2_re_hh_w = dpgrnn_0_intra_rnn_rnn2_weight_hh_l0_reverse;
        rnn2_re_hh_b = dpgrnn_0_intra_rnn_rnn2_bias_hh_l0_reverse;
        fc_w = dpgrnn_0_intra_fc_weight;
        fc_b = dpgrnn_0_intra_fc_bias;
        ln_w = dpgrnn_0_intra_ln_weight;
        ln_b = dpgrnn_0_intra_ln_bias;
    } else {
        rnn1_ih_w = dpgrnn_1_intra_rnn_rnn1_weight_ih_l0;
        rnn1_ih_b = dpgrnn_1_intra_rnn_rnn1_bias_ih_l0;
        rnn1_hh_w = dpgrnn_1_intra_rnn_rnn1_weight_hh_l0;
        rnn1_hh_b = dpgrnn_1_intra_rnn_rnn1_bias_hh_l0;
        rnn1_re_ih_w = dpgrnn_1_intra_rnn_rnn1_weight_ih_l0_reverse;
        rnn1_re_ih_b = dpgrnn_1_intra_rnn_rnn1_bias_ih_l0_reverse;
        rnn1_re_hh_w = dpgrnn_1_intra_rnn_rnn1_weight_hh_l0_reverse;
        rnn1_re_hh_b = dpgrnn_1_intra_rnn_rnn1_bias_hh_l0_reverse;
        rnn2_ih_w = dpgrnn_1_intra_rnn_rnn2_weight_ih_l0;
        rnn2_ih_b = dpgrnn_1_intra_rnn_rnn2_bias_ih_l0;
        rnn2_hh_w = dpgrnn_1_intra_rnn_rnn2_weight_hh_l0;
        rnn2_hh_b = dpgrnn_1_intra_rnn_rnn2_bias_hh_l0;
        rnn2_re_ih_w = dpgrnn_1_intra_rnn_rnn2_weight_ih_l0_reverse;
        rnn2_re_ih_b = dpgrnn_1_intra_rnn_rnn2_bias_ih_l0_reverse;
        rnn2_re_hh_w = dpgrnn_1_intra_rnn_rnn2_weight_hh_l0_reverse;
        rnn2_re_hh_b = dpgrnn_1_intra_rnn_rnn2_bias_hh_l0_reverse;
        fc_w = dpgrnn_1_intra_fc_weight;
        fc_b = dpgrnn_1_intra_fc_bias;
        ln_w = dpgrnn_1_intra_ln_weight;
        ln_b = dpgrnn_1_intra_ln_bias;
    }

    /* x is [33][16] in Q20, split into [33][8] each */
    /* x0 = x[:, 0:8], x1 = x[:, 8:16] */
    /* Extract columns: need to reorganize from row-major [C=16, W=33] where C=channels, W=frames */
    /* Input from encoder: [16, 33]. Transpose to [33, 16] for RNN processing. */
    /* Actually x is [16, 33] (channels × frames). For RNN: each row becomes a time step.
     * MATLAB: x0 = x(:,1:8) → [33, 8] (time steps × features)
     * So we need to transpose: x_rnn[t][c] = x[c][t]
     */

    /* Reorganize: split columns */
    int32_t x0[33 * 8], x1[33 * 8];
    for (int t = 0; t < 33; t++) {
        for (int c = 0; c < 8; c++) {
            x0[t * 8 + c] = x[c * 33 + t];       /* first group */
            x1[t * 8 + c] = x[(8 + c) * 33 + t]; /* second group */
        }
    }

    /* BiGRU on each group */
    int16_t x0_gru[33 * 8];   /* [33][8] (2*nHidden=8) */
    int16_t x1_gru[33 * 8];

    bigru_module(x0, 33, 4, 8,
                 rnn1_ih_w, rnn1_ih_b, rnn1_hh_w, rnn1_hh_b,
                 rnn1_re_ih_w, rnn1_re_ih_b, rnn1_re_hh_w, rnn1_re_hh_b,
                 DPRNN_GRU_QR1, DPRNN_GRU_QR2, x0_gru);

    bigru_module(x1, 33, 4, 8,
                 rnn2_ih_w, rnn2_ih_b, rnn2_hh_w, rnn2_hh_b,
                 rnn2_re_ih_w, rnn2_re_ih_b, rnn2_re_hh_w, rnn2_re_hh_b,
                 DPRNN_GRU_QR1, DPRNN_GRU_QR2, x1_gru);

    /* Concat: [33][16] */
    int16_t x_cat[33 * 16];
    for (int t = 0; t < 33; t++) {
        memcpy(&x_cat[t * 16], &x0_gru[t * 8], 8 * sizeof(int16_t));
        memcpy(&x_cat[t * 16 + 8], &x1_gru[t * 8], 8 * sizeof(int16_t));
    }

    /* FC: round(x_gru * fc_weight * 2^(-9)) + fc_bias */
    int shift_fc = -DPRNN_INTRA_FC_QR;  /* 9 */
    int64_t r_fc = ((int64_t)1 << (shift_fc - 1));
    int32_t x_fc[33 * 16];
    for (int t = 0; t < 33; t++) {
        for (int o = 0; o < 16; o++) {
            int64_t acc = 0;
            for (int i = 0; i < 16; i++) {
                int64_t prod = (int64_t)x_cat[t * 16 + i] * fc_w[i + 16 * o];
                if (prod >= 0) acc += (prod + r_fc) >> shift_fc;
                else           acc += (prod - r_fc) >> shift_fc;
            }
            x_fc[t * 16 + o] = sat_i32(acc + fc_b[o]);
        }
    }

    /* LN(Qr=-14) */
    int32_t x_ln[33 * 16];
    ln_func(x_fc, ln_w, ln_b, DPRNN_INTRA_LN_QR, 16, 33 * 16, x_ln);

    /* Residual: y = x + x_ln (x still in Q20, x_ln in Q20) */
    /* But wait: x_fc is in Q20, but x (the residual) was never dequantized.
     * In MATLAB: y = x + x_ln where both are the same type.
     * x is in original format (Q20 from encoder output), x_ln is from LN output (Q20).
     * So we add them directly.
     */
    for (int t = 0; t < 33; t++) {
        for (int c = 0; c < 16; c++) {
            int32_t x_val = x[c * 33 + t];  /* original in [C][W] layout */
            y[c * 33 + t] = sat_i32((int64_t)x_val + x_ln[t * 16 + c]);
        }
    }
}

/* ================================================================
 * GDPRNN: Inter-RNN
 * ================================================================
 * Split y_intra[33][16] → x0[33][8], x1[33][8]
 * Each: GRU(nHidden=8, with cached state)
 * Concat → FC(Qr=-9) → LN(Qr=-13) → residual
 */
void inter_rnn_module(const int32_t *x, int16_t *h_cache, int gdprnn_idx, int32_t *y) {
    const int16_t *rnn1_ih_w, *rnn1_hh_w, *rnn2_ih_w, *rnn2_hh_w;
    const int32_t *rnn1_ih_b, *rnn1_hh_b, *rnn2_ih_b, *rnn2_hh_b;
    const int16_t *fc_w, *ln_w;
    const int32_t *ln_b;
    const int32_t *fc_b;

    if (gdprnn_idx == 0) {
        rnn1_ih_w = dpgrnn_0_inter_rnn_rnn1_weight_ih_l0;
        rnn1_ih_b = dpgrnn_0_inter_rnn_rnn1_bias_ih_l0;
        rnn1_hh_w = dpgrnn_0_inter_rnn_rnn1_weight_hh_l0;
        rnn1_hh_b = dpgrnn_0_inter_rnn_rnn1_bias_hh_l0;
        rnn2_ih_w = dpgrnn_0_inter_rnn_rnn2_weight_ih_l0;
        rnn2_ih_b = dpgrnn_0_inter_rnn_rnn2_bias_ih_l0;
        rnn2_hh_w = dpgrnn_0_inter_rnn_rnn2_weight_hh_l0;
        rnn2_hh_b = dpgrnn_0_inter_rnn_rnn2_bias_hh_l0;
        fc_w = dpgrnn_0_inter_fc_weight;
        fc_b = dpgrnn_0_inter_fc_bias;
        ln_w = dpgrnn_0_inter_ln_weight;
        ln_b = dpgrnn_0_inter_ln_bias;
    } else {
        rnn1_ih_w = dpgrnn_1_inter_rnn_rnn1_weight_ih_l0;
        rnn1_ih_b = dpgrnn_1_inter_rnn_rnn1_bias_ih_l0;
        rnn1_hh_w = dpgrnn_1_inter_rnn_rnn1_weight_hh_l0;
        rnn1_hh_b = dpgrnn_1_inter_rnn_rnn1_bias_hh_l0;
        rnn2_ih_w = dpgrnn_1_inter_rnn_rnn2_weight_ih_l0;
        rnn2_ih_b = dpgrnn_1_inter_rnn_rnn2_bias_ih_l0;
        rnn2_hh_w = dpgrnn_1_inter_rnn_rnn2_weight_hh_l0;
        rnn2_hh_b = dpgrnn_1_inter_rnn_rnn2_bias_hh_l0;
        fc_w = dpgrnn_1_inter_fc_weight;
        fc_b = dpgrnn_1_inter_fc_bias;
        ln_w = dpgrnn_1_inter_ln_weight;
        ln_b = dpgrnn_1_inter_ln_bias;
    }

    /* x is [33][16], split into [33][8] each */
    int32_t x0[33 * 8], x1[33 * 8];
    for (int t = 0; t < 33; t++) {
        for (int c = 0; c < 8; c++) {
            x0[t * 8 + c] = x[c * 33 + t];
            x1[t * 8 + c] = x[(8 + c) * 33 + t];
        }
    }

    /* GRU on each group, per time step, with persistent h_cache */
    int16_t x0_gru[33 * 8], x1_gru[33 * 8];
    int16_t h0[8], h1[8];
    memcpy(h0, h_cache, 8 * sizeof(int16_t));
    memcpy(h1, h_cache + 8, 8 * sizeof(int16_t));

    for (int t = 0; t < 33; t++) {
        gru_module(&x0[t * 8], 8, 8, h0,
                   rnn1_ih_w, rnn1_ih_b, rnn1_hh_w, rnn1_hh_b,
                   DPRNN_GRU_QR1, DPRNN_GRU_QR2, &x0_gru[t * 8]);

        gru_module(&x1[t * 8], 8, 8, h1,
                   rnn2_ih_w, rnn2_ih_b, rnn2_hh_w, rnn2_hh_b,
                   DPRNN_GRU_QR1, DPRNN_GRU_QR2, &x1_gru[t * 8]);
    }

    /* Update persistent h_cache */
    memcpy(h_cache, h0, 8 * sizeof(int16_t));
    memcpy(h_cache + 8, h1, 8 * sizeof(int16_t));

    /* Concat */
    int16_t x_cat[33 * 16];
    for (int t = 0; t < 33; t++) {
        memcpy(&x_cat[t * 16], &x0_gru[t * 8], 8 * sizeof(int16_t));
        memcpy(&x_cat[t * 16 + 8], &x1_gru[t * 8], 8 * sizeof(int16_t));
    }

    /* FC */
    int shift_fc = -DPRNN_INTER_FC_QR;  /* 9 */
    int64_t r_fc = ((int64_t)1 << (shift_fc - 1));
    int32_t x_fc[33 * 16];
    for (int t = 0; t < 33; t++) {
        for (int o = 0; o < 16; o++) {
            int64_t acc = 0;
            for (int i = 0; i < 16; i++) {
                int64_t prod = (int64_t)x_cat[t * 16 + i] * fc_w[i + 16 * o];
                if (prod >= 0) acc += (prod + r_fc) >> shift_fc;
                else           acc += (prod - r_fc) >> shift_fc;
            }
            x_fc[t * 16 + o] = sat_i32(acc + fc_b[o]);
        }
    }

    /* LN(Qr=-13) */
    int32_t x_ln[33 * 16];
    ln_func(x_fc, ln_w, ln_b, DPRNN_INTER_LN_QR, 16, 33 * 16, x_ln);

    /* Residual */
    for (int t = 0; t < 33; t++) {
        for (int c = 0; c < 16; c++) {
            int32_t x_val = x[c * 33 + t];
            y[c * 33 + t] = sat_i32((int64_t)x_val + x_ln[t * 16 + c]);
        }
    }
}

/* ================================================================
 * GDPRNN Module: Intra → Inter
 * Input:  [16, 33] (channels × frames)
 * Output: [16, 33] (transposed back)
 */
void gdprnn_module(const int32_t *x, int16_t *h_cache, int gdprnn_idx, int32_t *y) {
    int32_t y_intra[16 * 33];
    int32_t y_inter[16 * 33];

    /* Intra-RNN: x is [16, 33] → internally transposed to [33, 16] */
    intra_rnn_module(x, gdprnn_idx, y_intra);

    /* Inter-RNN: y_intra is [16, 33] → internally transposed */
    inter_rnn_module(y_intra, h_cache, gdprnn_idx, y_inter);

    /* Output stays as [16, 33] (C×W) */
    memcpy(y, y_inter, 16 * 33 * sizeof(int32_t));
}

/* ================================================================
 * Decoder Layer 0: De_XDWS0
 * ================================================================
 * skip(y_e4) → PConv(2 groups, 16→32) → Shuffle → nonGTConv(1×5) → cTFA
 */
void decoder_layer0_de_xdws0(const int32_t *x, const int32_t *skip, ulunas_state_t *st, int32_t *y) {
    /* Skip connection: x_con = x + skip */
    int32_t x_con[16 * 33];
    for (int i = 0; i < 16 * 33; i++) {
        x_con[i] = sat_i32((int64_t)x[i] + skip[i]);
    }

    int32_t y_pconv[32 * 33];
    /* PConv: groups=2, Cin=8→Cout=16 each */
    pconv2d_func(x_con, 8, 16, 1, 33,
                 decoder_de_convs_0_pconv_0_weight,
                 decoder_de_convs_0_pconv_0_bias,
                 D0_PCONV_CONV_QR, y_pconv);
    pconv2d_func(x_con + 8 * 33, 8, 16, 1, 33,
                 decoder_de_convs_0_pconv_0_weight + 16 * 8,
                 decoder_de_convs_0_pconv_0_bias + 16,
                 D0_PCONV_CONV_QR, y_pconv + 16 * 33);

    /* BN */
    int32_t y_bn[32 * 33];
    bn_func(y_pconv, decoder_de_convs_0_pconv_1_weight, decoder_de_convs_0_pconv_1_bias,
            decoder_de_convs_0_pconv_1_running_mean, decoder_de_convs_0_pconv_1_running_var,
            D0_PCONV_BN_QR1, D0_PCONV_BN_QR2, 32, 32 * 33, y_bn);

    /* AffinePReLU */
    int32_t y_ap[32 * 33];
    affineprelu_func(y_bn, decoder_de_convs_0_pconv_2_affine_weight,
                     decoder_de_convs_0_pconv_2_affine_bias,
                     decoder_de_convs_0_pconv_2_slope_weight,
                     D0_PCONV_AFFINE_QR1, D0_PCONV_AFFINE_QR2, 32, 33, y_ap);

    /* Shuffle */
    int32_t y_s[32 * 33];
    shuffle_interleave(y_ap, 32, 33, y_s);

    /* nonGTConv */
    int32_t y_tconv[32 * 33];
    non_gtconv2d_func(y_s, 32, 1, 33, 1, 5, 1, 1,
                      decoder_de_convs_0_dconv_1_weight, decoder_de_convs_0_dconv_1_bias,
                      D0_NONTCONV_CONV_QR, y_tconv);

    /* BN */
    int32_t y_tconv_bn[32 * 33];
    bn_func(y_tconv, decoder_de_convs_0_dconv_2_weight, decoder_de_convs_0_dconv_2_bias,
            decoder_de_convs_0_dconv_2_running_mean, decoder_de_convs_0_dconv_2_running_var,
            D0_NONTCONV_BN_QR1, D0_NONTCONV_BN_QR2, 32, 32 * 33, y_tconv_bn);

    /* AffinePReLU */
    int32_t y_tconv_ap[32 * 33];
    affineprelu_func(y_tconv_bn, decoder_de_convs_0_dconv_3_affine_weight,
                     decoder_de_convs_0_dconv_3_affine_bias,
                     decoder_de_convs_0_dconv_3_slope_weight,
                     D0_NONTCONV_AFFINE_QR1, D0_NONTCONV_AFFINE_QR2, 32, 33, y_tconv_ap);

    /* cTFA TA */
    uint16_t y_ta[32];
    ctfa_ta_module(y_tconv_ap, 32, 33, D0_CTFA_TA_GRU_NHID,
                   st->tfa_cache_d0,
                   decoder_de_convs_0_dconv_4_ta_gru_weight_ih_l0,
                   decoder_de_convs_0_dconv_4_ta_gru_bias_ih_l0,
                   decoder_de_convs_0_dconv_4_ta_gru_weight_hh_l0,
                   decoder_de_convs_0_dconv_4_ta_gru_bias_hh_l0,
                   decoder_de_convs_0_dconv_4_ta_fc_weight,
                   decoder_de_convs_0_dconv_4_ta_fc_bias,
                   D0_CTFA_TA_GRU_QR1, D0_CTFA_TA_GRU_QR2, D0_CTFA_TA_FC_QR, y_ta);

    /* cTFA FA */
    uint16_t y_fa[33];
    ctfa_fa_module(y_tconv_ap, 32, 33, D0_CTFA_FA_GRU_NHID,
                   D0_CTFA_FA_GROUP, D0_CTFA_FA_SEG, D0_CTFA_FA_PAD,
                   decoder_de_convs_0_dconv_4_fa_gru_weight_ih_l0,
                   decoder_de_convs_0_dconv_4_fa_gru_bias_ih_l0,
                   decoder_de_convs_0_dconv_4_fa_gru_weight_hh_l0,
                   decoder_de_convs_0_dconv_4_fa_gru_bias_hh_l0,
                   decoder_de_convs_0_dconv_4_fa_gru_weight_ih_l0_reverse,
                   decoder_de_convs_0_dconv_4_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_0_dconv_4_fa_gru_weight_hh_l0_reverse,
                   decoder_de_convs_0_dconv_4_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_0_dconv_4_fa_fc_weight,
                   decoder_de_convs_0_dconv_4_fa_fc_bias,
                   D0_CTFA_FA_GRU_QR1, D0_CTFA_FA_GRU_QR2, D0_CTFA_FA_FC_QR, y_fa);

    /* cTFA fusion */
    int64_t r = 16384;
    for (int c = 0; c < 32; c++) {
        for (int w = 0; w < 33; w++) {
            int64_t p1 = (int64_t)y_tconv_ap[c * 33 + w] * y_ta[c];
            int32_t yt;
            if (p1 >= 0) yt = (int32_t)((p1 + r) >> 15);
            else         yt = (int32_t)((p1 - r) >> 15);
            int64_t p2 = (int64_t)yt * y_fa[w];
            if (p2 >= 0) y[c * 33 + w] = (int32_t)((p2 + r) >> 15);
            else         y[c * 33 + w] = (int32_t)((p2 - r) >> 15);
        }
    }
}

/* ================================================================
 * Decoder Layer 1: De_XMB0
 * ================================================================
 * skip(x_e3) → PConv0(groups=2, 32→24) → Shuffle → nonGTConv(1×5) → PConv1 → cTFA → Shuffle
 * Input:  [32, 33] + skip_e3[32, 33]
 * Output: [24, 33]
 */
void decoder_layer1_de_xmb0(const int32_t *x, const int32_t *skip, ulunas_state_t *st, int32_t *y) {
    /* Skip connection */
    int32_t x_con[32 * 33];
    for (int i = 0; i < 32 * 33; i++) {
        x_con[i] = sat_i32((int64_t)x[i] + skip[i]);
    }

    int32_t y_pconv0[24 * 33];
    /* PConv0: groups=2, Cin=16→Cout=12 each */
    pconv2d_func(x_con, 16, 12, 1, 33,
                 decoder_de_convs_1_pconv1_0_weight,
                 decoder_de_convs_1_pconv1_0_bias,
                 D1_PCONV0_CONV_QR, y_pconv0);
    pconv2d_func(x_con + 16 * 33, 16, 12, 1, 33,
                 decoder_de_convs_1_pconv1_0_weight + 12 * 16,
                 decoder_de_convs_1_pconv1_0_bias + 12,
                 D1_PCONV0_CONV_QR, y_pconv0 + 12 * 33);

    /* BN */
    int32_t y_bn0[24 * 33];
    bn_func(y_pconv0, decoder_de_convs_1_pconv1_1_weight, decoder_de_convs_1_pconv1_1_bias,
            decoder_de_convs_1_pconv1_1_running_mean, decoder_de_convs_1_pconv1_1_running_var,
            D1_PCONV0_BN_QR1, D1_PCONV0_BN_QR2, 24, 24 * 33, y_bn0);

    /* AffinePReLU */
    int32_t y_ap0[24 * 33];
    affineprelu_func(y_bn0, decoder_de_convs_1_pconv1_2_affine_weight,
                     decoder_de_convs_1_pconv1_2_affine_bias,
                     decoder_de_convs_1_pconv1_2_slope_weight,
                     D1_PCONV0_AFFINE_QR1, D1_PCONV0_AFFINE_QR2, 24, 33, y_ap0);

    /* Shuffle */
    int32_t y_s[24 * 33];
    shuffle_interleave(y_ap0, 24, 33, y_s);

    /* nonGTConv: [24,33]→[24,33], 1×5, no cache */
    int32_t y_tconv[24 * 33];
    non_gtconv2d_func(y_s, 24, 1, 33, 1, 5, 1, 1,
                      decoder_de_convs_1_dconv_1_weight, decoder_de_convs_1_dconv_1_bias,
                      D1_NONTCONV_CONV_QR, y_tconv);

    /* BN */
    int32_t y_tconv_bn[24 * 33];
    bn_func(y_tconv, decoder_de_convs_1_dconv_2_weight, decoder_de_convs_1_dconv_2_bias,
            decoder_de_convs_1_dconv_2_running_mean, decoder_de_convs_1_dconv_2_running_var,
            D1_NONTCONV_BN_QR1, D1_NONTCONV_BN_QR2, 24, 24 * 33, y_tconv_bn);

    /* AffinePReLU */
    int32_t y_tconv_ap[24 * 33];
    affineprelu_func(y_tconv_bn, decoder_de_convs_1_dconv_3_affine_weight,
                     decoder_de_convs_1_dconv_3_affine_bias,
                     decoder_de_convs_1_dconv_3_slope_weight,
                     D1_NONTCONV_AFFINE_QR1, D1_NONTCONV_AFFINE_QR2, 24, 33, y_tconv_ap);

    /* PConv1: groups=2, Cin=12→Cout=12 each */
    int32_t y_pconv1[24 * 33];
    pconv2d_func(y_tconv_ap, 12, 12, 1, 33,
                 decoder_de_convs_1_pconv2_0_weight,
                 decoder_de_convs_1_pconv2_0_bias,
                 D1_PCONV1_CONV_QR, y_pconv1);
    pconv2d_func(y_tconv_ap + 12 * 33, 12, 12, 1, 33,
                 decoder_de_convs_1_pconv2_0_weight + 12 * 12,
                 decoder_de_convs_1_pconv2_0_bias + 12,
                 D1_PCONV1_CONV_QR, y_pconv1 + 12 * 33);

    /* BN (in-place on y_pconv1 to get cTFA input) */
    bn_func(y_pconv1, decoder_de_convs_1_pconv2_1_weight, decoder_de_convs_1_pconv2_1_bias,
            decoder_de_convs_1_pconv2_1_running_mean, decoder_de_convs_1_pconv2_1_running_var,
            D1_PCONV1_BN_QR1, D1_PCONV1_BN_QR2, 24, 24 * 33, y_pconv1);

    /* cTFA TA */
    uint16_t y_ta[24];
    ctfa_ta_module(y_pconv1, 24, 33, D1_CTFA_TA_GRU_NHID,
                   st->tfa_cache_d1,
                   decoder_de_convs_1_pconv2_2_ta_gru_weight_ih_l0,
                   decoder_de_convs_1_pconv2_2_ta_gru_bias_ih_l0,
                   decoder_de_convs_1_pconv2_2_ta_gru_weight_hh_l0,
                   decoder_de_convs_1_pconv2_2_ta_gru_bias_hh_l0,
                   decoder_de_convs_1_pconv2_2_ta_fc_weight,
                   decoder_de_convs_1_pconv2_2_ta_fc_bias,
                   D1_CTFA_TA_GRU_QR1, D1_CTFA_TA_GRU_QR2, D1_CTFA_TA_FC_QR, y_ta);

    /* cTFA FA */
    uint16_t y_fa[33];
    ctfa_fa_module(y_pconv1, 24, 33, D1_CTFA_FA_GRU_NHID,
                   D1_CTFA_FA_GROUP, D1_CTFA_FA_SEG, D1_CTFA_FA_PAD,
                   decoder_de_convs_1_pconv2_2_fa_gru_weight_ih_l0,
                   decoder_de_convs_1_pconv2_2_fa_gru_bias_ih_l0,
                   decoder_de_convs_1_pconv2_2_fa_gru_weight_hh_l0,
                   decoder_de_convs_1_pconv2_2_fa_gru_bias_hh_l0,
                   decoder_de_convs_1_pconv2_2_fa_gru_weight_ih_l0_reverse,
                   decoder_de_convs_1_pconv2_2_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_1_pconv2_2_fa_gru_weight_hh_l0_reverse,
                   decoder_de_convs_1_pconv2_2_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_1_pconv2_2_fa_fc_weight,
                   decoder_de_convs_1_pconv2_2_fa_fc_bias,
                   D1_CTFA_FA_GRU_QR1, D1_CTFA_FA_GRU_QR2, D1_CTFA_FA_FC_QR, y_fa);

    /* cTFA fusion */
    int64_t r = 16384;
    int32_t y_ctfa[24 * 33];
    for (int c = 0; c < 24; c++) {
        for (int w = 0; w < 33; w++) {
            int64_t p1 = (int64_t)y_pconv1[c * 33 + w] * y_ta[c];
            int32_t yt;
            if (p1 >= 0) yt = (int32_t)((p1 + r) >> 15);
            else         yt = (int32_t)((p1 - r) >> 15);
            int64_t p2 = (int64_t)yt * y_fa[w];
            if (p2 >= 0) y_ctfa[c * 33 + w] = (int32_t)((p2 + r) >> 15);
            else         y_ctfa[c * 33 + w] = (int32_t)((p2 - r) >> 15);
        }
    }

    /* Final shuffle */
    shuffle_interleave(y_ctfa, 24, 33, y);
}

/* ================================================================
 * Decoder Layer 2: De_XDWS1
 * ================================================================
 * skip(x_e2) → PConv(groups=2, 24→24) → Shuffle → GTConv(2×3,s=[1,1]) → cTFA
 * Input:  [24, 33] + skip_e2[24, 33]
 * Output: [24, 33]
 */
void decoder_layer2_de_xdws1(const int32_t *x, const int32_t *skip, ulunas_state_t *st, int32_t *y) {
    /* Skip connection */
    int32_t x_con[24 * 33];
    for (int i = 0; i < 24 * 33; i++) {
        x_con[i] = sat_i32((int64_t)x[i] + skip[i]);
    }

    int32_t y_pconv[24 * 33];
    /* PConv: groups=2, Cin=12→Cout=12 each */
    pconv2d_func(x_con, 12, 12, 1, 33,
                 decoder_de_convs_2_pconv_0_weight,
                 decoder_de_convs_2_pconv_0_bias,
                 D2_PCONV_CONV_QR, y_pconv);
    pconv2d_func(x_con + 12 * 33, 12, 12, 1, 33,
                 decoder_de_convs_2_pconv_0_weight + 12 * 12,
                 decoder_de_convs_2_pconv_0_bias + 12,
                 D2_PCONV_CONV_QR, y_pconv + 12 * 33);

    /* BN */
    int32_t y_bn0[24 * 33];
    bn_func(y_pconv, decoder_de_convs_2_pconv_1_weight, decoder_de_convs_2_pconv_1_bias,
            decoder_de_convs_2_pconv_1_running_mean, decoder_de_convs_2_pconv_1_running_var,
            D2_PCONV_BN_QR1, D2_PCONV_BN_QR2, 24, 24 * 33, y_bn0);

    /* AffinePReLU */
    int32_t y_ap[24 * 33];
    affineprelu_func(y_bn0, decoder_de_convs_2_pconv_2_affine_weight,
                     decoder_de_convs_2_pconv_2_affine_bias,
                     decoder_de_convs_2_pconv_2_slope_weight,
                     D2_PCONV_AFFINE_QR1, D2_PCONV_AFFINE_QR2, 24, 33, y_ap);

    /* Shuffle */
    int32_t y_s[24 * 33];
    shuffle_interleave(y_ap, 24, 33, y_s);

    /* GTConv: transposed temporal conv with cache, 2×3, s=[1,1] */
    int32_t y_tconv[24 * 33];
    gtconv2d_func(y_s, 24, 1, 33, 2, 3, 1, 1,
                  decoder_de_convs_2_dconv_1_weight, decoder_de_convs_2_dconv_1_bias,
                  D2_TCONV_CONV_QR,
                  st->conv_cache_d0, y_tconv);

    /* BN */
    int32_t y_tconv_bn[24 * 33];
    bn_func(y_tconv, decoder_de_convs_2_dconv_2_weight, decoder_de_convs_2_dconv_2_bias,
            decoder_de_convs_2_dconv_2_running_mean, decoder_de_convs_2_dconv_2_running_var,
            D2_TCONV_BN_QR1, D2_TCONV_BN_QR2, 24, 24 * 33, y_tconv_bn);

    /* AffinePReLU */
    int32_t y_tconv_ap[24 * 33];
    affineprelu_func(y_tconv_bn, decoder_de_convs_2_dconv_3_affine_weight,
                     decoder_de_convs_2_dconv_3_affine_bias,
                     decoder_de_convs_2_dconv_3_slope_weight,
                     D2_TCONV_AFFINE_QR1, D2_TCONV_AFFINE_QR2, 24, 33, y_tconv_ap);

    /* cTFA TA */
    uint16_t y_ta[24];
    ctfa_ta_module(y_tconv_ap, 24, 33, D2_CTFA_TA_GRU_NHID,
                   st->tfa_cache_d2,
                   decoder_de_convs_2_dconv_4_ta_gru_weight_ih_l0,
                   decoder_de_convs_2_dconv_4_ta_gru_bias_ih_l0,
                   decoder_de_convs_2_dconv_4_ta_gru_weight_hh_l0,
                   decoder_de_convs_2_dconv_4_ta_gru_bias_hh_l0,
                   decoder_de_convs_2_dconv_4_ta_fc_weight,
                   decoder_de_convs_2_dconv_4_ta_fc_bias,
                   D2_CTFA_TA_GRU_QR1, D2_CTFA_TA_GRU_QR2, D2_CTFA_TA_FC_QR, y_ta);

    /* cTFA FA */
    uint16_t y_fa[33];
    ctfa_fa_module(y_tconv_ap, 24, 33, D2_CTFA_FA_GRU_NHID,
                   D2_CTFA_FA_GROUP, D2_CTFA_FA_SEG, D2_CTFA_FA_PAD,
                   decoder_de_convs_2_dconv_4_fa_gru_weight_ih_l0,
                   decoder_de_convs_2_dconv_4_fa_gru_bias_ih_l0,
                   decoder_de_convs_2_dconv_4_fa_gru_weight_hh_l0,
                   decoder_de_convs_2_dconv_4_fa_gru_bias_hh_l0,
                   decoder_de_convs_2_dconv_4_fa_gru_weight_ih_l0_reverse,
                   decoder_de_convs_2_dconv_4_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_2_dconv_4_fa_gru_weight_hh_l0_reverse,
                   decoder_de_convs_2_dconv_4_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_2_dconv_4_fa_fc_weight,
                   decoder_de_convs_2_dconv_4_fa_fc_bias,
                   D2_CTFA_FA_GRU_QR1, D2_CTFA_FA_GRU_QR2, D2_CTFA_FA_FC_QR, y_fa);

    /* cTFA fusion */
    int64_t r = 16384;
    for (int c = 0; c < 24; c++) {
        for (int w = 0; w < 33; w++) {
            int64_t p1 = (int64_t)y_tconv_ap[c * 33 + w] * y_ta[c];
            int32_t yt;
            if (p1 >= 0) yt = (int32_t)((p1 + r) >> 15);
            else         yt = (int32_t)((p1 - r) >> 15);
            int64_t p2 = (int64_t)yt * y_fa[w];
            if (p2 >= 0) y[c * 33 + w] = (int32_t)((p2 + r) >> 15);
            else         y[c * 33 + w] = (int32_t)((p2 - r) >> 15);
        }
    }
}

/* ================================================================
 * Decoder Layer 3: De_XMB1
 * ================================================================
 * skip(x_e1) → PConv0(groups=2, 24→12) → Shuffle → GTConv(2×3,s=[1,2]) → PConv1 → cTFA → Shuffle
 * Input:  [24, 33] + skip_e1[24, 33]
 * Output: [12, 65]
 */
void decoder_layer3_de_xmb1(const int32_t *x, const int32_t *skip, ulunas_state_t *st, int32_t *y) {
    /* Skip connection */
    int32_t x_con[24 * 33];
    for (int i = 0; i < 24 * 33; i++) {
        x_con[i] = sat_i32((int64_t)x[i] + skip[i]);
    }

    int32_t y_pconv0[12 * 33];
    /* PConv0: groups=2, Cin=12→Cout=6 each */
    pconv2d_func(x_con, 12, 6, 1, 33,
                 decoder_de_convs_3_pconv1_0_weight,
                 decoder_de_convs_3_pconv1_0_bias,
                 D3_PCONV0_CONV_QR, y_pconv0);
    pconv2d_func(x_con + 12 * 33, 12, 6, 1, 33,
                 decoder_de_convs_3_pconv1_0_weight + 6 * 12,
                 decoder_de_convs_3_pconv1_0_bias + 6,
                 D3_PCONV0_CONV_QR, y_pconv0 + 6 * 33);

    /* BN */
    int32_t y_bn0[12 * 33];
    bn_func(y_pconv0, decoder_de_convs_3_pconv1_1_weight, decoder_de_convs_3_pconv1_1_bias,
            decoder_de_convs_3_pconv1_1_running_mean, decoder_de_convs_3_pconv1_1_running_var,
            D3_PCONV0_BN_QR1, D3_PCONV0_BN_QR2, 12, 12 * 33, y_bn0);

    /* AffinePReLU */
    int32_t y_ap0[12 * 33];
    affineprelu_func(y_bn0, decoder_de_convs_3_pconv1_2_affine_weight,
                     decoder_de_convs_3_pconv1_2_affine_bias,
                     decoder_de_convs_3_pconv1_2_slope_weight,
                     D3_PCONV0_AFFINE_QR1, D3_PCONV0_AFFINE_QR2, 12, 33, y_ap0);

    /* Shuffle */
    int32_t y_s[12 * 33];
    shuffle_interleave(y_ap0, 12, 33, y_s);

    /* GTConv: transposed temporal conv with cache, 2×3, s=[1,2] */
    int32_t y_tconv[12 * 65];
    gtconv2d_func(y_s, 12, 1, 65, 2, 3, 1, 2,
                  decoder_de_convs_3_dconv_1_weight, decoder_de_convs_3_dconv_1_bias,
                  D3_TCONV_CONV_QR,
                  st->conv_cache_d1, y_tconv);

    /* BN */
    int32_t y_tconv_bn[12 * 65];
    bn_func(y_tconv, decoder_de_convs_3_dconv_2_weight, decoder_de_convs_3_dconv_2_bias,
            decoder_de_convs_3_dconv_2_running_mean, decoder_de_convs_3_dconv_2_running_var,
            D3_TCONV_BN_QR1, D3_TCONV_BN_QR2, 12, 12 * 65, y_tconv_bn);

    /* AffinePReLU */
    int32_t y_tconv_ap[12 * 65];
    affineprelu_func(y_tconv_bn, decoder_de_convs_3_dconv_3_affine_weight,
                     decoder_de_convs_3_dconv_3_affine_bias,
                     decoder_de_convs_3_dconv_3_slope_weight,
                     D3_TCONV_AFFINE_QR1, D3_TCONV_AFFINE_QR2, 12, 65, y_tconv_ap);

    /* PConv1: groups=2, Cin=6→Cout=6 each */
    int32_t y_pconv1[12 * 65];
    pconv2d_func(y_tconv_ap, 6, 6, 1, 65,
                 decoder_de_convs_3_pconv2_0_weight,
                 decoder_de_convs_3_pconv2_0_bias,
                 D3_PCONV1_CONV_QR, y_pconv1);
    pconv2d_func(y_tconv_ap + 6 * 65, 6, 6, 1, 65,
                 decoder_de_convs_3_pconv2_0_weight + 6 * 6,
                 decoder_de_convs_3_pconv2_0_bias + 6,
                 D3_PCONV1_CONV_QR, y_pconv1 + 6 * 65);

    /* BN (in-place) */
    bn_func(y_pconv1, decoder_de_convs_3_pconv2_1_weight, decoder_de_convs_3_pconv2_1_bias,
            decoder_de_convs_3_pconv2_1_running_mean, decoder_de_convs_3_pconv2_1_running_var,
            D3_PCONV1_BN_QR1, D3_PCONV1_BN_QR2, 12, 12 * 65, y_pconv1);

    /* cTFA TA */
    uint16_t y_ta[12];
    ctfa_ta_module(y_pconv1, 12, 65, D3_CTFA_TA_GRU_NHID,
                   st->tfa_cache_d3,
                   decoder_de_convs_3_pconv2_2_ta_gru_weight_ih_l0,
                   decoder_de_convs_3_pconv2_2_ta_gru_bias_ih_l0,
                   decoder_de_convs_3_pconv2_2_ta_gru_weight_hh_l0,
                   decoder_de_convs_3_pconv2_2_ta_gru_bias_hh_l0,
                   decoder_de_convs_3_pconv2_2_ta_fc_weight,
                   decoder_de_convs_3_pconv2_2_ta_fc_bias,
                   D3_CTFA_TA_GRU_QR1, D3_CTFA_TA_GRU_QR2, D3_CTFA_TA_FC_QR, y_ta);

    /* cTFA FA */
    uint16_t y_fa[65];
    ctfa_fa_module(y_pconv1, 12, 65, D3_CTFA_FA_GRU_NHID,
                   D3_CTFA_FA_GROUP, D3_CTFA_FA_SEG, D3_CTFA_FA_PAD,
                   decoder_de_convs_3_pconv2_2_fa_gru_weight_ih_l0,
                   decoder_de_convs_3_pconv2_2_fa_gru_bias_ih_l0,
                   decoder_de_convs_3_pconv2_2_fa_gru_weight_hh_l0,
                   decoder_de_convs_3_pconv2_2_fa_gru_bias_hh_l0,
                   decoder_de_convs_3_pconv2_2_fa_gru_weight_ih_l0_reverse,
                   decoder_de_convs_3_pconv2_2_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_3_pconv2_2_fa_gru_weight_hh_l0_reverse,
                   decoder_de_convs_3_pconv2_2_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_3_pconv2_2_fa_fc_weight,
                   decoder_de_convs_3_pconv2_2_fa_fc_bias,
                   D3_CTFA_FA_GRU_QR1, D3_CTFA_FA_GRU_QR2, D3_CTFA_FA_FC_QR, y_fa);

    /* cTFA fusion */
    int64_t r = 16384;
    int32_t y_ctfa[12 * 65];
    for (int c = 0; c < 12; c++) {
        for (int w = 0; w < 65; w++) {
            int64_t p1 = (int64_t)y_pconv1[c * 65 + w] * y_ta[c];
            int32_t yt;
            if (p1 >= 0) yt = (int32_t)((p1 + r) >> 15);
            else         yt = (int32_t)((p1 - r) >> 15);
            int64_t p2 = (int64_t)yt * y_fa[w];
            if (p2 >= 0) y_ctfa[c * 65 + w] = (int32_t)((p2 + r) >> 15);
            else         y_ctfa[c * 65 + w] = (int32_t)((p2 - r) >> 15);
        }
    }

    /* Final shuffle */
    shuffle_interleave(y_ctfa, 12, 65, y);
}

/* ================================================================
 * Decoder Layer 4: De_XConv
 * ================================================================
 * skip(x_e0) → TConv(12→1, 3×3, s=[1,2], with cache) → BN → cTFA
 * Input:  [12, 65] + skip_e0[12, 65]
 * Output: [1, 129]
 */
void decoder_layer4_de_xconv(const int32_t *x, const int32_t *skip, ulunas_state_t *st, int32_t *y) {
    /* Skip connection */
    int32_t x_con[12 * 65];
    for (int i = 0; i < 12 * 65; i++) {
        x_con[i] = sat_i32((int64_t)x[i] + skip[i]);
    }

    /* TConv: cache is [12, 2, 65], current x_con is [12, 1, 65] → x_cache [12, 3, 65] */
    /* MATLAB: x_cache = cat(2, conv_cache, reshape(x_con, [12,1,65]))
     * C: x_cache[ci*Hcache*65 + h*65 + w]
     * cache_d2 is [12, 2, 65]
     */
    int cache_h = 2;
    int x_h = 1;
    int W_in = 65;
    int32_t *cache_d2 = st->conv_cache_d2;  /* [12, 2, 65] */
    int32_t x_cache[12 * 3 * 65];  /* [12, 3, 65] */
    for (int c = 0; c < 12; c++) {
        /* Copy cache: 2 rows */
        memcpy(&x_cache[c * 3 * W_in], &cache_d2[c * cache_h * W_in], cache_h * W_in * sizeof(int32_t));
        /* Copy x_con: 1 row */
        memcpy(&x_cache[c * 3 * W_in + cache_h * W_in], &x_con[c * x_h * W_in], x_h * W_in * sizeof(int32_t));
    }

    /* tconv2d_func: standard transposed conv, Cin=12, Cout=1, Wout=129 */
    int32_t y_tconv[1 * 129];
    tconv2d_func(x_cache, 12, 1, 1, 129, 3, 3, 1, 2,
                 decoder_de_convs_4_ops_1_weight, decoder_de_convs_4_ops_1_bias,
                 D4_TCONV_CONV_QR, y_tconv);

    /* Update cache: x_cache(:, 2:3, :) → keep last 2 rows */
    for (int c = 0; c < 12; c++) {
        memcpy(&cache_d2[c * cache_h * W_in], &x_cache[c * 3 * W_in + 1 * W_in], cache_h * W_in * sizeof(int32_t));
    }

    /* BN */
    int32_t y_bn[1 * 129];
    bn_func(y_tconv, decoder_de_convs_4_ops_2_weight, decoder_de_convs_4_ops_2_bias,
            decoder_de_convs_4_ops_2_running_mean, decoder_de_convs_4_ops_2_running_var,
            D4_TCONV_BN_QR1, D4_TCONV_BN_QR2, 1, 1 * 129, y_bn);

    /* cTFA TA (nHidden=2, 1 output channel) */
    uint16_t y_ta[1];
    ctfa_ta_module(y_bn, 1, 129, D4_CTFA_TA_GRU_NHID,
                   st->tfa_cache_d4,
                   decoder_de_convs_4_ops_4_ta_gru_weight_ih_l0,
                   decoder_de_convs_4_ops_4_ta_gru_bias_ih_l0,
                   decoder_de_convs_4_ops_4_ta_gru_weight_hh_l0,
                   decoder_de_convs_4_ops_4_ta_gru_bias_hh_l0,
                   decoder_de_convs_4_ops_4_ta_fc_weight,
                   decoder_de_convs_4_ops_4_ta_fc_bias,
                   D4_CTFA_TA_GRU_QR1, D4_CTFA_TA_GRU_QR2, D4_CTFA_TA_FC_QR, y_ta);

    /* cTFA FA */
    uint16_t y_fa[129];
    ctfa_fa_module(y_bn, 1, 129, D4_CTFA_FA_GRU_NHID,
                   D4_CTFA_FA_GROUP, D4_CTFA_FA_SEG, D4_CTFA_FA_PAD,
                   decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0,
                   decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0,
                   decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0,
                   decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0,
                   decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0_reverse,
                   decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0_reverse,
                   decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0_reverse,
                   decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0_reverse,
                   decoder_de_convs_4_ops_4_fa_fc_weight,
                   decoder_de_convs_4_ops_4_fa_fc_bias,
                   D4_CTFA_FA_GRU_QR1, D4_CTFA_FA_GRU_QR2, D4_CTFA_FA_FC_QR, y_fa);

    /* cTFA fusion */
    int64_t r = 16384;
    for (int w = 0; w < 129; w++) {
        int64_t p1 = (int64_t)y_bn[w] * y_ta[0];
        int32_t yt;
        if (p1 >= 0) yt = (int32_t)((p1 + r) >> 15);
        else         yt = (int32_t)((p1 - r) >> 15);
        int64_t p2 = (int64_t)yt * y_fa[w];
        if (p2 >= 0) y[w] = (int32_t)((p2 + r) >> 15);
        else         y[w] = (int32_t)((p2 - r) >> 15);
    }
}

/* ================================================================
 * Decoder Module: all 5 layers with skip connections
 * ================================================================
 * Pipeline: De_XDWS0 → De_XMB0 → De_XDWS1 → De_XMB1 → De_XConv
 * Output: [1, 129] (single-channel mask in Q20)
 */
void decoder_module(const int32_t *x, ulunas_state_t *st,
                    const int32_t *e0, const int32_t *e1, const int32_t *e2,
                    const int32_t *e3, const int32_t *e4,
                    int32_t *y) {
    int32_t d0[32 * 33], d1[24 * 33], d2[24 * 33], d3[12 * 65];

    /* Layer 0: [16,33] + e4 → [32,33] */
    decoder_layer0_de_xdws0(x, e4, st, d0);

    /* Layer 1: [32,33] + e3 → [24,33] */
    decoder_layer1_de_xmb0(d0, e3, st, d1);

    /* Layer 2: [24,33] + e2 → [24,33] */
    decoder_layer2_de_xdws1(d1, e2, st, d2);

    /* Layer 3: [24,33] + e1 → [12,65] */
    decoder_layer3_de_xmb1(d2, e1, st, d3);

    /* Layer 4: [12,65] + e0 → [1,129] */
    decoder_layer4_de_xconv(d3, e0, st, y);
}
