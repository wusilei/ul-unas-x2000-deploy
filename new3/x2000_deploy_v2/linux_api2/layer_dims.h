/**
 * layer_dims.h — Network Layer Dimensions
 * ========================================
 * Auto-generated from MATLAB source analysis.
 * Every dimension matches the MATLAB block files exactly.
 */

#ifndef LAYER_DIMS_H
#define LAYER_DIMS_H

/* ================================================================
 * Input dimensions
 * ================================================================ */
#define INPUT_FREQ_BINS     257     /* STFT frequency bins */
#define BM_OUT_BINS         129     /* After ERB band merging */

/* ================================================================
 * Encoder Layer 0 (XConv)
 * ================================================================ */
#define E0_CONV_CIN         1
#define E0_CONV_COUT        12
#define E0_CONV_HOUT        1
#define E0_CONV_WOUT        65
#define E0_CONV_KH          3
#define E0_CONV_KW          3
#define E0_CONV_STRIDE_H    1
#define E0_CONV_STRIDE_W    2
#define E0_CTFA_TA_GRU_NHID 24
#define E0_CTFA_FA_GRU_NHID 4
#define E0_CTFA_FA_GROUP    4
#define E0_CTFA_FA_SEG      17
#define E0_CTFA_FA_PAD      3

/* ================================================================
 * Encoder Layer 1 (XMB0)
 * ================================================================ */
#define E1_PCONV0_CIN       6   /* per group */
#define E1_PCONV0_COUT      12  /* per group */
#define E1_PCONV0_HOUT      1
#define E1_PCONV0_WOUT      65
#define E1_TCONV_CIN        24
#define E1_TCONV_COUT       24
#define E1_TCONV_HOUT       1
#define E1_TCONV_WOUT       33
#define E1_TCONV_KH         2
#define E1_TCONV_KW         3
#define E1_TCONV_STRIDE_H   1
#define E1_TCONV_STRIDE_W   2
#define E1_PCONV1_CIN       12  /* per group */
#define E1_PCONV1_COUT      12  /* per group */
#define E1_PCONV1_HOUT      1
#define E1_PCONV1_WOUT      33
#define E1_CTFA_TA_GRU_NHID 48
#define E1_CTFA_FA_GRU_NHID 4
#define E1_CTFA_FA_GROUP    4
#define E1_CTFA_FA_SEG      9   /* (33+3)/4 = 9 */
#define E1_CTFA_FA_PAD      3

/* ================================================================
 * Encoder Layer 2 (XDWS0)
 * ================================================================ */
#define E2_PCONV_CIN        12  /* per group */
#define E2_PCONV_COUT       12  /* per group */
#define E2_PCONV_HOUT       1
#define E2_PCONV_WOUT       33
#define E2_TCONV_CIN        24
#define E2_TCONV_COUT       24
#define E2_TCONV_HOUT       1
#define E2_TCONV_WOUT       33
#define E2_TCONV_KH         2
#define E2_TCONV_KW         3
#define E2_TCONV_STRIDE_H   1
#define E2_TCONV_STRIDE_W   1
#define E2_CTFA_TA_GRU_NHID 48
#define E2_CTFA_FA_GRU_NHID 4
#define E2_CTFA_FA_GROUP    4
#define E2_CTFA_FA_SEG      9
#define E2_CTFA_FA_PAD      3

/* ================================================================
 * Encoder Layer 3 (XMB1)
 * ================================================================ */
#define E3_PCONV0_CIN       12  /* per group */
#define E3_PCONV0_COUT      16  /* per group */
#define E3_PCONV0_HOUT      1
#define E3_PCONV0_WOUT      33
#define E3_NONTCONV_CIN     32
#define E3_NONTCONV_COUT    32
#define E3_NONTCONV_HOUT    1
#define E3_NONTCONV_WOUT    33
#define E3_NONTCONV_KH      1
#define E3_NONTCONV_KW      5
#define E3_NONTCONV_STRIDE_H 1
#define E3_NONTCONV_STRIDE_W 1
#define E3_PCONV1_CIN       16  /* per group */
#define E3_PCONV1_COUT      16  /* per group */
#define E3_PCONV1_HOUT      1
#define E3_PCONV1_WOUT      33
#define E3_CTFA_TA_GRU_NHID 64
#define E3_CTFA_FA_GRU_NHID 4
#define E3_CTFA_FA_GROUP    4
#define E3_CTFA_FA_SEG      9
#define E3_CTFA_FA_PAD      3

/* ================================================================
 * Encoder Layer 4 (XDWS1)
 * ================================================================ */
#define E4_PCONV_CIN        16  /* per group */
#define E4_PCONV_COUT       8   /* per group */
#define E4_PCONV_HOUT       1
#define E4_PCONV_WOUT       33
#define E4_NONTCONV_CIN     16
#define E4_NONTCONV_COUT    16
#define E4_NONTCONV_HOUT    1
#define E4_NONTCONV_WOUT    33
#define E4_NONTCONV_KH      1
#define E4_NONTCONV_KW      5
#define E4_NONTCONV_STRIDE_H 1
#define E4_NONTCONV_STRIDE_W 1
#define E4_CTFA_TA_GRU_NHID 32
#define E4_CTFA_FA_GRU_NHID 4
#define E4_CTFA_FA_GROUP    4
#define E4_CTFA_FA_SEG      9
#define E4_CTFA_FA_PAD      3

/* ================================================================
 * Decoder Layer 0 (De_XDWS0)
 * ================================================================ */
#define D0_PCONV_CIN        8   /* per group */
#define D0_PCONV_COUT       16  /* per group */
#define D0_PCONV_HOUT       1
#define D0_PCONV_WOUT       33
#define D0_NONTCONV_CIN     32
#define D0_NONTCONV_COUT    32
#define D0_NONTCONV_HOUT    1
#define D0_NONTCONV_WOUT    33
#define D0_NONTCONV_KH      1
#define D0_NONTCONV_KW      5
#define D0_NONTCONV_STRIDE_H 1
#define D0_NONTCONV_STRIDE_W 1
#define D0_CTFA_TA_GRU_NHID 64
#define D0_CTFA_FA_GRU_NHID 4
#define D0_CTFA_FA_GROUP    4
#define D0_CTFA_FA_SEG      9
#define D0_CTFA_FA_PAD      3

/* ================================================================
 * Decoder Layer 1 (De_XMB0)
 * ================================================================ */
#define D1_PCONV0_CIN       16  /* per group */
#define D1_PCONV0_COUT      12  /* per group */
#define D1_PCONV0_HOUT      1
#define D1_PCONV0_WOUT      33
#define D1_NONTCONV_CIN     24
#define D1_NONTCONV_COUT    24
#define D1_NONTCONV_HOUT    1
#define D1_NONTCONV_WOUT    33
#define D1_NONTCONV_KH      1
#define D1_NONTCONV_KW      5
#define D1_NONTCONV_STRIDE_H 1
#define D1_NONTCONV_STRIDE_W 1
#define D1_PCONV1_CIN       12  /* per group */
#define D1_PCONV1_COUT      12  /* per group */
#define D1_PCONV1_HOUT      1
#define D1_PCONV1_WOUT      33
#define D1_CTFA_TA_GRU_NHID 48
#define D1_CTFA_FA_GRU_NHID 4
#define D1_CTFA_FA_GROUP    4
#define D1_CTFA_FA_SEG      9
#define D1_CTFA_FA_PAD      3

/* ================================================================
 * Decoder Layer 2 (De_XDWS1)
 * ================================================================ */
#define D2_PCONV_CIN        12  /* per group */
#define D2_PCONV_COUT       12  /* per group */
#define D2_PCONV_HOUT       1
#define D2_PCONV_WOUT       33
#define D2_TCONV_CIN        24
#define D2_TCONV_COUT       24
#define D2_TCONV_HOUT       1
#define D2_TCONV_WOUT       33
#define D2_TCONV_KH         2
#define D2_TCONV_KW         3
#define D2_TCONV_STRIDE_H   1
#define D2_TCONV_STRIDE_W   1
#define D2_CTFA_TA_GRU_NHID 48
#define D2_CTFA_FA_GRU_NHID 4
#define D2_CTFA_FA_GROUP    4
#define D2_CTFA_FA_SEG      9
#define D2_CTFA_FA_PAD      3

/* ================================================================
 * Decoder Layer 3 (De_XMB1)
 * ================================================================ */
#define D3_PCONV0_CIN       12  /* per group */
#define D3_PCONV0_COUT      6   /* per group */
#define D3_PCONV0_HOUT      1
#define D3_PCONV0_WOUT      33
#define D3_TCONV_CIN        12
#define D3_TCONV_COUT       12
#define D3_TCONV_HOUT       1
#define D3_TCONV_WOUT       65
#define D3_TCONV_KH         2
#define D3_TCONV_KW         3
#define D3_TCONV_STRIDE_H   1
#define D3_TCONV_STRIDE_W   2
#define D3_PCONV1_CIN       6   /* per group */
#define D3_PCONV1_COUT      6   /* per group */
#define D3_PCONV1_HOUT      1
#define D3_PCONV1_WOUT      65
#define D3_CTFA_TA_GRU_NHID 24
#define D3_CTFA_FA_GRU_NHID 4
#define D3_CTFA_FA_GROUP    4
#define D3_CTFA_FA_SEG      17  /* (65+3)/4 = 17 */
#define D3_CTFA_FA_PAD      3

/* ================================================================
 * Decoder Layer 4 (De_XConv)
 * ================================================================ */
#define D4_TCONV_CIN        12
#define D4_TCONV_COUT       1
#define D4_TCONV_HOUT       1
#define D4_TCONV_WOUT       129
#define D4_TCONV_KH         3
#define D4_TCONV_KW         3
#define D4_TCONV_STRIDE_H   1
#define D4_TCONV_STRIDE_W   2
#define D4_CTFA_TA_GRU_NHID 2
#define D4_CTFA_FA_GRU_NHID 4
#define D4_CTFA_FA_GROUP    4
#define D4_CTFA_FA_SEG      33  /* (129+3)/4 = 33 */
#define D4_CTFA_FA_PAD      3

/* ================================================================
 * GDPRNN dimensions
 * ================================================================ */
#define DPRNN_CHANNELS      16
#define DPRNN_FRAMES        33
#define DPRNN_GROUP_SIZE    8
#define DPRNN_INTRA_NHID    4
#define DPRNN_INTER_NHID    8
#define DPRNN_FC_CIN        8   /* each group: nHidden*2 for BiGRU or nHidden for GRU */
#define DPRNN_INTRA_FC_CIN  16  /* BiGRU concat: 2 groups * 4*2 = 16 */
#define DPRNN_INTER_FC_CIN  16  /* GRU concat: 2 groups * 8 = 16 */

/* ================================================================
 * Cache sizes
 * ================================================================ */
#define CACHE_E0_SIZE       (2 * 129)       /* conv_cache_e0 */
#define CACHE_E1_SIZE       (24 * 65)       /* conv_cache_e1 */
#define CACHE_E2_SIZE       (24 * 33)       /* conv_cache_e2 */
#define CACHE_D0_SIZE       (24 * 33)       /* conv_cache_d0 */
#define CACHE_D1_SIZE       (12 * 33)       /* conv_cache_d1 */
#define CACHE_D2_SIZE       (12 * 2 * 65)   /* conv_cache_d2 */

#endif /* LAYER_DIMS_H */
