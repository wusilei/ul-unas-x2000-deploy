/**
 * qr_config.h — Per-Layer Per-Operator Qr Shift Parameters
 * =========================================================
 * Auto-extracted from MATLAB source code.
 * Qr < 0 means right-shift by -Qr: round(x * 2^Qr) = round(x >> (-Qr))
 * Each value verified against the actual MATLAB block file.
 */

#ifndef QR_CONFIG_H
#define QR_CONFIG_H

/* ================================================================
 * Encoder Layer 0 (XConv)
 * ================================================================ */
#define E0_TCONV_CONV_QR         (-14)
#define E0_TCONV_BN_QR1          (-14)
#define E0_TCONV_BN_QR2          (-14)
#define E0_TCONV_AFFINE_QR1      (-13)
#define E0_TCONV_AFFINE_QR2      (-13)
#define E0_CTFA_TA_GRU_QR1       (-13)
#define E0_CTFA_TA_GRU_QR2       (-8)
#define E0_CTFA_TA_FC_QR         (-8)
#define E0_CTFA_FA_GRU_QR1       (-13)
#define E0_CTFA_FA_GRU_QR2       (-8)
#define E0_CTFA_FA_FC_QR         (-9)

/* ================================================================
 * Encoder Layer 1 (XMB0)
 * ================================================================ */
#define E1_PCONV0_CONV_QR        (-14)
#define E1_PCONV0_BN_QR1         (-11)
#define E1_PCONV0_BN_QR2         (-14)
#define E1_PCONV0_AFFINE_QR1     (-13)
#define E1_PCONV0_AFFINE_QR2     (-13)
#define E1_TCONV_CONV_QR         (-14)
#define E1_TCONV_BN_QR1          (-11)
#define E1_TCONV_BN_QR2          (-14)
#define E1_TCONV_AFFINE_QR1      (-13)
#define E1_TCONV_AFFINE_QR2      (-13)
#define E1_PCONV1_CONV_QR        (-14)
#define E1_PCONV1_BN_QR1         (-14)
#define E1_PCONV1_BN_QR2         (-14)
#define E1_CTFA_TA_GRU_QR1       (-13)
#define E1_CTFA_TA_GRU_QR2       (-8)
#define E1_CTFA_TA_FC_QR         (-8)
#define E1_CTFA_FA_GRU_QR1       (-13)
#define E1_CTFA_FA_GRU_QR2       (-8)
#define E1_CTFA_FA_FC_QR         (-9)

/* ================================================================
 * Encoder Layer 2 (XDWS0)
 * ================================================================ */
#define E2_PCONV_CONV_QR         (-14)
#define E2_PCONV_BN_QR1          (-11)
#define E2_PCONV_BN_QR2          (-14)
#define E2_PCONV_AFFINE_QR1      (-13)
#define E2_PCONV_AFFINE_QR2      (-13)
#define E2_TCONV_CONV_QR         (-13)
#define E2_TCONV_BN_QR1          (-14)
#define E2_TCONV_BN_QR2          (-14)
#define E2_TCONV_AFFINE_QR1      (-13)
#define E2_TCONV_AFFINE_QR2      (-13)
#define E2_CTFA_TA_GRU_QR1       (-13)
#define E2_CTFA_TA_GRU_QR2       (-8)
#define E2_CTFA_TA_FC_QR         (-8)
#define E2_CTFA_FA_GRU_QR1       (-13)
#define E2_CTFA_FA_GRU_QR2       (-8)
#define E2_CTFA_FA_FC_QR         (-9)

/* ================================================================
 * Encoder Layer 3 (XMB1)
 * ================================================================ */
#define E3_PCONV0_CONV_QR        (-13)
#define E3_PCONV0_BN_QR1         (-11)
#define E3_PCONV0_BN_QR2         (-14)
#define E3_PCONV0_AFFINE_QR1     (-13)
#define E3_PCONV0_AFFINE_QR2     (-13)
#define E3_NONTCONV_CONV_QR      (-13)
#define E3_NONTCONV_BN_QR1       (-14)
#define E3_NONTCONV_BN_QR2       (-14)
#define E3_NONTCONV_AFFINE_QR1   (-13)
#define E3_NONTCONV_AFFINE_QR2   (-13)
#define E3_PCONV1_CONV_QR        (-14)
#define E3_PCONV1_BN_QR1         (-14)
#define E3_PCONV1_BN_QR2         (-14)
#define E3_CTFA_TA_GRU_QR1       (-13)
#define E3_CTFA_TA_GRU_QR2       (-8)
#define E3_CTFA_TA_FC_QR         (-8)
#define E3_CTFA_FA_GRU_QR1       (-13)
#define E3_CTFA_FA_GRU_QR2       (-8)
#define E3_CTFA_FA_FC_QR         (-9)

/* ================================================================
 * Encoder Layer 4 (XDWS1)
 * ================================================================ */
#define E4_PCONV_CONV_QR         (-14)
#define E4_PCONV_BN_QR1          (-11)
#define E4_PCONV_BN_QR2          (-14)
#define E4_PCONV_AFFINE_QR1      (-13)
#define E4_PCONV_AFFINE_QR2      (-13)
#define E4_NONTCONV_CONV_QR      (-14)
#define E4_NONTCONV_BN_QR1       (-14)
#define E4_NONTCONV_BN_QR2       (-14)
#define E4_NONTCONV_AFFINE_QR1   (-13)
#define E4_NONTCONV_AFFINE_QR2   (-13)
#define E4_CTFA_TA_GRU_QR1       (-13)
#define E4_CTFA_TA_GRU_QR2       (-8)
#define E4_CTFA_TA_FC_QR         (-8)
#define E4_CTFA_FA_GRU_QR1       (-13)
#define E4_CTFA_FA_GRU_QR2       (-8)
#define E4_CTFA_FA_FC_QR         (-9)

/* ================================================================
 * Decoder Layer 0 (De_XDWS0)
 * ================================================================ */
#define D0_PCONV_CONV_QR         (-14)
#define D0_PCONV_BN_QR1          (-14)
#define D0_PCONV_BN_QR2          (-14)
#define D0_PCONV_AFFINE_QR1      (-13)
#define D0_PCONV_AFFINE_QR2      (-13)
#define D0_NONTCONV_CONV_QR      (-14)
#define D0_NONTCONV_BN_QR1       (-14)
#define D0_NONTCONV_BN_QR2       (-14)
#define D0_NONTCONV_AFFINE_QR1   (-13)
#define D0_NONTCONV_AFFINE_QR2   (-13)
#define D0_CTFA_TA_GRU_QR1       (-13)
#define D0_CTFA_TA_GRU_QR2       (-8)
#define D0_CTFA_TA_FC_QR         (-8)
#define D0_CTFA_FA_GRU_QR1       (-13)
#define D0_CTFA_FA_GRU_QR2       (-8)
#define D0_CTFA_FA_FC_QR         (-9)

/* ================================================================
 * Decoder Layer 1 (De_XMB0)
 * ================================================================ */
#define D1_PCONV0_CONV_QR        (-13)   /* De_XMB0_PConv_block_0.m */
#define D1_PCONV0_BN_QR1         (-11)
#define D1_PCONV0_BN_QR2         (-14)
#define D1_PCONV0_AFFINE_QR1     (-13)
#define D1_PCONV0_AFFINE_QR2     (-13)
#define D1_NONTCONV_CONV_QR      (-14)
#define D1_NONTCONV_BN_QR1       (-11)   /* De_XMB0_nonTConv_block.m */
#define D1_NONTCONV_BN_QR2       (-14)
#define D1_NONTCONV_AFFINE_QR1   (-13)
#define D1_NONTCONV_AFFINE_QR2   (-13)
#define D1_PCONV1_CONV_QR        (-14)
#define D1_PCONV1_BN_QR1         (-11)   /* De_XMB0_PConv_block_1.m */
#define D1_PCONV1_BN_QR2         (-11)
#define D1_CTFA_TA_GRU_QR1       (-13)
#define D1_CTFA_TA_GRU_QR2       (-8)
#define D1_CTFA_TA_FC_QR         (-8)
#define D1_CTFA_FA_GRU_QR1       (-13)
#define D1_CTFA_FA_GRU_QR2       (-8)
#define D1_CTFA_FA_FC_QR         (-9)

/* ================================================================
 * Decoder Layer 2 (De_XDWS1)
 * ================================================================ */
#define D2_PCONV_CONV_QR         (-14)
#define D2_PCONV_BN_QR1          (-11)
#define D2_PCONV_BN_QR2          (-14)
#define D2_PCONV_AFFINE_QR1      (-13)
#define D2_PCONV_AFFINE_QR2      (-13)
#define D2_TCONV_CONV_QR         (-13)
#define D2_TCONV_BN_QR1          (-11)
#define D2_TCONV_BN_QR2          (-14)
#define D2_TCONV_AFFINE_QR1      (-13)
#define D2_TCONV_AFFINE_QR2      (-13)
#define D2_CTFA_TA_GRU_QR1       (-13)
#define D2_CTFA_TA_GRU_QR2       (-8)
#define D2_CTFA_TA_FC_QR         (-8)
#define D2_CTFA_FA_GRU_QR1       (-13)
#define D2_CTFA_FA_GRU_QR2       (-8)
#define D2_CTFA_FA_FC_QR         (-9)

/* ================================================================
 * Decoder Layer 3 (De_XMB1)
 * ================================================================ */
#define D3_PCONV0_CONV_QR        (-14)
#define D3_PCONV0_BN_QR1         (-11)
#define D3_PCONV0_BN_QR2         (-14)
#define D3_PCONV0_AFFINE_QR1     (-13)
#define D3_PCONV0_AFFINE_QR2     (-13)
#define D3_TCONV_CONV_QR         (-14)
#define D3_TCONV_BN_QR1          (-11)
#define D3_TCONV_BN_QR2          (-11)
#define D3_TCONV_AFFINE_QR1      (-13)
#define D3_TCONV_AFFINE_QR2      (-13)
#define D3_PCONV1_CONV_QR        (-14)
#define D3_PCONV1_BN_QR1         (-11)   /* De_XMB1_PConv_block_1.m */
#define D3_PCONV1_BN_QR2         (-11)
#define D3_CTFA_TA_GRU_QR1       (-13)
#define D3_CTFA_TA_GRU_QR2       (-8)
#define D3_CTFA_TA_FC_QR         (-8)
#define D3_CTFA_FA_GRU_QR1       (-13)
#define D3_CTFA_FA_GRU_QR2       (-8)
#define D3_CTFA_FA_FC_QR         (-9)

/* ================================================================
 * Decoder Layer 4 (De_XConv)
 * ================================================================ */
#define D4_TCONV_CONV_QR         (-14)
#define D4_TCONV_BN_QR1          (-11)
#define D4_TCONV_BN_QR2          (-11)
#define D4_CTFA_TA_GRU_QR1       (-13)
#define D4_CTFA_TA_GRU_QR2       (-8)
#define D4_CTFA_TA_FC_QR         (-8)
#define D4_CTFA_FA_GRU_QR1       (-13)
#define D4_CTFA_FA_GRU_QR2       (-8)
#define D4_CTFA_FA_FC_QR         (-9)

/* ================================================================
 * GDPRNN
 * ================================================================ */
#define DPRNN_GRU_QR1            (-13)
#define DPRNN_GRU_QR2            (-8)
#define DPRNN_INTRA_FC_QR        (-9)
#define DPRNN_INTRA_LN_QR        (-14)
#define DPRNN_INTER_FC_QR        (-9)
#define DPRNN_INTER_LN_QR        (-13)

/* ================================================================
 * Shared / Global
 * ================================================================ */
#define BM_BS_QR                 (-15)
#define MASK_QR                  (-15)
#define CTFA_FUSION_QR           (-15)
#define GRU_GATE_QR              (-15)   /* round(r_t .* h_t * 2^(-15)) */
#define GRU_UPDATE_QR            (-15)   /* round(z_t .* h * 2^(-15)) */
#define GRU_ONE_MINUS_Z_Q15      32768   /* 1.0 in Q15 for (1-z)*n */

#endif /* QR_CONFIG_H */
