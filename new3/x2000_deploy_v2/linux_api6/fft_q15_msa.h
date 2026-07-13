/**
 * fft_q15_msa.h — MSA-optimized fixed-point 512-pt FFT
 * =====================================================
 * Uses MSA mul_q_w / madd_q_w / msub_q_w to replace int64 multiply path.
 * Processes 2 butterflies per inner loop iteration when stride permits.
 */
#ifndef FFT_Q15_MSA_H
#define FFT_Q15_MSA_H

#include <msa.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FFT_N 512
#define FFT_BINS 257
#define FFT_BITS 9

/* Twiddle factors: cos(2πk/512) and sin(2πk/512), k=0..255, Q15 */
static const int16_t fft_tw_cos[256] = {
     32767,  32765,  32757,  32745,  32728,  32705,  32678,  32646,  32609,  32567,  32521,  32469,  32412,  32351,  32285,  32213,
     32137,  32057,  31971,  31880,  31785,  31685,  31580,  31470,  31356,  31237,  31113,  30985,  30852,  30714,  30571,  30424,
     30273,  30117,  29956,  29791,  29621,  29447,  29268,  29085,  28898,  28706,  28510,  28310,  28105,  27896,  27683,  27466,
     27245,  27019,  26790,  26556,  26319,  26077,  25832,  25582,  25329,  25072,  24811,  24547,  24279,  24007,  23731,  23452,
     23170,  22884,  22594,  22301,  22005,  21705,  21403,  21096,  20787,  20475,  20159,  19841,  19519,  19195,  18868,  18537,
     18204,  17869,  17530,  17189,  16846,  16499,  16151,  15800,  15446,  15090,  14732,  14372,  14010,  13645,  13279,  12910,
     12539,  12167,  11793,  11417,  11039,  10659,  10278,   9896,   9512,   9126,   8739,   8351,   7962,   7571,   7179,   6786,
      6393,   5998,   5602,   5205,   4808,   4410,   4011,   3612,   3212,   2811,   2410,   2009,   1608,   1206,    804,    402,
         0,   -402,   -804,  -1206,  -1608,  -2009,  -2410,  -2811,  -3212,  -3612,  -4011,  -4410,  -4808,  -5205,  -5602,  -5998,
     -6393,  -6786,  -7179,  -7571,  -7962,  -8351,  -8739,  -9126,  -9512,  -9896, -10278, -10659, -11039, -11417, -11793, -12167,
    -12539, -12910, -13279, -13645, -14010, -14372, -14732, -15090, -15446, -15800, -16151, -16499, -16846, -17189, -17530, -17869,
    -18204, -18537, -18868, -19195, -19519, -19841, -20159, -20475, -20787, -21096, -21403, -21705, -22005, -22301, -22594, -22884,
    -23170, -23452, -23731, -24007, -24279, -24547, -24811, -25072, -25329, -25582, -25832, -26077, -26319, -26556, -26790, -27019,
    -27245, -27466, -27683, -27896, -28105, -28310, -28510, -28706, -28898, -29085, -29268, -29447, -29621, -29791, -29956, -30117,
    -30273, -30424, -30571, -30714, -30852, -30985, -31113, -31237, -31356, -31470, -31580, -31685, -31785, -31880, -31971, -32057,
    -32137, -32213, -32285, -32351, -32412, -32469, -32521, -32567, -32609, -32646, -32678, -32705, -32728, -32745, -32757, -32765
};

static const int16_t fft_tw_sin[256] = {
         0,    402,    804,   1206,   1608,   2009,   2410,   2811,   3212,   3612,   4011,   4410,   4808,   5205,   5602,   5998,
      6393,   6786,   7179,   7571,   7962,   8351,   8739,   9126,   9512,   9896,  10278,  10659,  11039,  11417,  11793,  12167,
     12539,  12910,  13279,  13645,  14010,  14372,  14732,  15090,  15446,  15800,  16151,  16499,  16846,  17189,  17530,  17869,
     18204,  18537,  18868,  19195,  19519,  19841,  20159,  20475,  20787,  21096,  21403,  21705,  22005,  22301,  22594,  22884,
     23170,  23452,  23731,  24007,  24279,  24547,  24811,  25072,  25329,  25582,  25832,  26077,  26319,  26556,  26790,  27019,
     27245,  27466,  27683,  27896,  28105,  28310,  28510,  28706,  28898,  29085,  29268,  29447,  29621,  29791,  29956,  30117,
     30273,  30424,  30571,  30714,  30852,  30985,  31113,  31237,  31356,  31470,  31580,  31685,  31785,  31880,  31971,  32057,
     32137,  32213,  32285,  32351,  32412,  32469,  32521,  32567,  32609,  32646,  32678,  32705,  32728,  32745,  32757,  32765,
     32767,  32765,  32757,  32745,  32728,  32705,  32678,  32646,  32609,  32567,  32521,  32469,  32412,  32351,  32285,  32213,
     32137,  32057,  31971,  31880,  31785,  31685,  31580,  31470,  31356,  31237,  31113,  30985,  30852,  30714,  30571,  30424,
     30273,  30117,  29956,  29791,  29621,  29447,  29268,  29085,  28898,  28706,  28510,  28310,  28105,  27896,  27683,  27466,
     27245,  27019,  26790,  26556,  26319,  26077,  25832,  25582,  25329,  25072,  24811,  24547,  24279,  24007,  23731,  23452,
     23170,  22884,  22594,  22301,  22005,  21705,  21403,  21096,  20787,  20475,  20159,  19841,  19519,  19195,  18868,  18537,
     18204,  17869,  17530,  17189,  16846,  16499,  16151,  15800,  15446,  15090,  14732,  14372,  14010,  13645,  13279,  12910,
     12539,  12167,  11793,  11417,  11039,  10659,  10278,   9896,   9512,   9126,   8739,   8351,   7962,   7571,   7179,   6786,
      6393,   5998,   5602,   5205,   4808,   4410,   4011,   3612,   3212,   2811,   2410,   2009,   1608,   1206,    804,    402
};

/* Bit-reversal table: 9-bit reversal for indices 0..511 */
static const int16_t fft_bit_rev[512] = {
       0,  256,  128,  384,   64,  320,  192,  448,   32,  288,  160,  416,   96,  352,  224,  480,
      16,  272,  144,  400,   80,  336,  208,  464,   48,  304,  176,  432,  112,  368,  240,  496,
       8,  264,  136,  392,   72,  328,  200,  456,   40,  296,  168,  424,  104,  360,  232,  488,
      24,  280,  152,  408,   88,  344,  216,  472,   56,  312,  184,  440,  120,  376,  248,  504,
       4,  260,  132,  388,   68,  324,  196,  452,   36,  292,  164,  420,  100,  356,  228,  484,
      20,  276,  148,  404,   84,  340,  212,  468,   52,  308,  180,  436,  116,  372,  244,  500,
      12,  268,  140,  396,   76,  332,  204,  460,   44,  300,  172,  428,  108,  364,  236,  492,
      28,  284,  156,  412,   92,  348,  220,  476,   60,  316,  188,  444,  124,  380,  252,  508,
       2,  258,  130,  386,   66,  322,  194,  450,   34,  290,  162,  418,   98,  354,  226,  482,
      18,  274,  146,  402,   82,  338,  210,  466,   50,  306,  178,  434,  114,  370,  242,  498,
      10,  266,  138,  394,   74,  330,  202,  458,   42,  298,  170,  426,  106,  362,  234,  490,
      26,  282,  154,  410,   90,  346,  218,  474,   58,  314,  186,  442,  122,  378,  250,  506,
       6,  262,  134,  390,   70,  326,  198,  454,   38,  294,  166,  422,  102,  358,  230,  486,
      22,  278,  150,  406,   86,  342,  214,  470,   54,  310,  182,  438,  118,  374,  246,  502,
      14,  270,  142,  398,   78,  334,  206,  462,   46,  302,  174,  430,  110,  366,  238,  494,
      30,  286,  158,  414,   94,  350,  222,  478,   62,  318,  190,  446,  126,  382,  254,  510,
       1,  257,  129,  385,   65,  321,  193,  449,   33,  289,  161,  417,   97,  353,  225,  481,
      17,  273,  145,  401,   81,  337,  209,  465,   49,  305,  177,  433,  113,  369,  241,  497,
       9,  265,  137,  393,   73,  329,  201,  457,   41,  297,  169,  425,  105,  361,  233,  489,
      25,  281,  153,  409,   89,  345,  217,  473,   57,  313,  185,  441,  121,  377,  249,  505,
       5,  261,  133,  389,   69,  325,  197,  453,   37,  293,  165,  421,  101,  357,  229,  485,
      21,  277,  149,  405,   85,  341,  213,  469,   53,  309,  181,  437,  117,  373,  245,  501,
      13,  269,  141,  397,   77,  333,  205,  461,   45,  301,  173,  429,  109,  365,  237,  493,
      29,  285,  157,  413,   93,  349,  221,  477,   61,  317,  189,  445,  125,  381,  253,  509,
       3,  259,  131,  387,   67,  323,  195,  451,   35,  291,  163,  419,   99,  355,  227,  483,
      19,  275,  147,  403,   83,  339,  211,  467,   51,  307,  179,  435,  115,  371,  243,  499,
      11,  267,  139,  395,   75,  331,  203,  459,   43,  299,  171,  426,  107,  363,  235,  491,
      27,  283,  155,  411,   91,  347,  219,  475,   59,  315,  187,  442,  123,  379,  251,  507,
       7,  263,  135,  391,   71,  327,  199,  455,   39,  295,  167,  423,  103,  359,  231,  487,
      23,  279,  151,  407,   87,  343,  215,  471,   55,  310,  183,  439,  119,  375,  247,  503,
      15,  271,  143,  399,   79,  335,  207,  463,   47,  303,  175,  431,  111,  367,  239,  495,
      31,  287,  159,  415,   95,  351,  223,  479,   63,  319,  191,  447,  127,  383,  255,  511
};

/* MSA-accelerated scalar Q15 multiply: (a * b) >> 15 with rounding.
 * Uses MSA mulr_q_w single instruction instead of int64 mult+mfhi+mflo+shift. */
static inline int32_t msa_mulr_q15(int32_t a, int16_t b) {
    v4i32 va = (v4i32)__msa_fill_w(a);
    v4i32 vb = (v4i32)__msa_fill_w((int32_t)b);
    v4i32 vr = __msa_mulr_q_w(va, vb);
    return __msa_copy_s_w(vr, 0);
}

/* MSA Q15 complex multiply: (a_r + j*a_i) * (b_r + j*b_i)
 * tr = (a_r * b_r - a_i * b_i) >> 15
 * ti = (a_r * b_i + a_i * b_r) >> 15
 * Uses MSA msubr_q_w and maddr_q_w for fused multiply-subtract/add with rounding. */
static inline void msa_cmul_q15(int32_t ar, int32_t ai, int16_t br, int16_t bi,
                                 int32_t *tr, int32_t *ti) {
    v4i32 va = (v4i32){ar, ar, ai, ai};
    v4i32 vb = (v4i32){(int32_t)br, (int32_t)bi, (int32_t)br, (int32_t)bi};

    /* tr = ar*br - ai*bi, ti = ar*bi + ai*br, both >> 15 with rounding */
    v4i32 v_ar_br = __msa_mulr_q_w((v4i32){ar, 0, 0, 0}, (v4i32){(int32_t)br, 0, 0, 0});
    v4i32 v_ai_bi = __msa_mulr_q_w((v4i32){ai, 0, 0, 0}, (v4i32){(int32_t)bi, 0, 0, 0});
    v4i32 v_ar_bi = __msa_mulr_q_w((v4i32){ar, 0, 0, 0}, (v4i32){(int32_t)bi, 0, 0, 0});
    v4i32 v_ai_br = __msa_mulr_q_w((v4i32){ai, 0, 0, 0}, (v4i32){(int32_t)br, 0, 0, 0});

    *tr = __msa_copy_s_w(v_ar_br, 0) - __msa_copy_s_w(v_ai_bi, 0);
    *ti = __msa_copy_s_w(v_ar_bi, 0) + __msa_copy_s_w(v_ai_br, 0);
}

/* Core FFT: 512-pt complex radix-2 DIT, in-place, MSA-optimized butterfly */
static inline void fft_q15_512(int32_t *real, int32_t *imag, int inverse) {
    /* Bit-reversal permutation (scalar, O(N) with low overhead) */
    for (int i = 0; i < FFT_N; i++) {
        int j = fft_bit_rev[i];
        if (i < j) {
            int32_t tr = real[i]; real[i] = real[j]; real[j] = tr;
            int32_t ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
        }
    }

    /* 9 stages of radix-2 butterflies, MSA-optimized inner multiply */
    for (int s = 0; s < FFT_BITS; s++) {
        int m = 1 << s;
        int m2 = m << 1;
        for (int k = 0; k < m; k++) {
            int tw_idx = k * (FFT_N / m2);
            int16_t wr = fft_tw_cos[tw_idx];
            int16_t wi = inverse ? fft_tw_sin[tw_idx] : (int16_t)(-fft_tw_sin[tw_idx]);

            /* Inner loop: try 2-butterfly MSA vectorization when stride permits */
            int j = k;
            int j_end = FFT_N;

            /* Stage 0 (m2=2): consecutive pairs — process 2 butterflies at once */
            if (m2 == 2) {
                for (; j < j_end; j += 4) {
                    int j2_0 = j + 1, j2_1 = j + 3;

                    /* Load 2 consecutive j (even) and j2 (odd) pairs */
                    v4i32 r_j = (v4i32){real[j], real[j+2], 0, 0};
                    v4i32 i_j = (v4i32){imag[j], imag[j+2], 0, 0};
                    v4i32 r_j2 = (v4i32){real[j2_0], real[j2_1], 0, 0};
                    v4i32 i_j2 = (v4i32){imag[j2_0], imag[j2_1], 0, 0};

                    /* Broadcast twiddle */
                    v4i32 v_wr = (v4i32){(int32_t)wr, (int32_t)wr, 0, 0};
                    v4i32 v_wi = (v4i32){(int32_t)wi, (int32_t)wi, 0, 0};

                    /* tr = (rj2 * wr - ij2 * wi) >> 15 */
                    v4i32 v_rj2_wr = __msa_mulr_q_w(r_j2, v_wr);
                    v4i32 v_ij2_wi = __msa_mulr_q_w(i_j2, v_wi);
                    v4i32 tr_vec = __msa_subv_w(v_rj2_wr, v_ij2_wi);

                    /* ti = (ij2 * wr + rj2 * wi) >> 15 */
                    v4i32 v_ij2_wr = __msa_mulr_q_w(i_j2, v_wr);
                    v4i32 v_rj2_wi = __msa_mulr_q_w(r_j2, v_wi);
                    v4i32 ti_vec = __msa_addv_w(v_ij2_wr, v_rj2_wi);

                    /* rj2' = rj - tr, rj' = rj + tr */
                    v4i32 r_j2_new = __msa_subv_w(r_j, tr_vec);
                    v4i32 r_j_new  = __msa_addv_w(r_j, tr_vec);

                    /* ij2' = ij - ti, ij' = ij + ti */
                    v4i32 i_j2_new = __msa_subv_w(i_j, ti_vec);
                    v4i32 i_j_new  = __msa_addv_w(i_j, ti_vec);

                    real[j]     = __msa_copy_s_w(r_j_new,  0);
                    real[j+2]   = __msa_copy_s_w(r_j_new,  1);
                    real[j2_0]  = __msa_copy_s_w(r_j2_new, 0);
                    real[j2_1]  = __msa_copy_s_w(r_j2_new, 1);
                    imag[j]     = __msa_copy_s_w(i_j_new,  0);
                    imag[j+2]   = __msa_copy_s_w(i_j_new,  1);
                    imag[j2_0]  = __msa_copy_s_w(i_j2_new, 0);
                    imag[j2_1]  = __msa_copy_s_w(i_j2_new, 1);
                }
            }
            /* Stage 1 (m2=4): interleaved pairs — process 2 butterflies */
            else if (m2 == 4) {
                for (; j < j_end; j += 8) {
                    int j2_0 = j + m, j2_1 = j + 4 + m;

                    v4i32 r_j = (v4i32){real[j], real[j+4], 0, 0};
                    v4i32 i_j = (v4i32){imag[j], imag[j+4], 0, 0};
                    v4i32 r_j2 = (v4i32){real[j2_0], real[j2_1], 0, 0};
                    v4i32 i_j2 = (v4i32){imag[j2_0], imag[j2_1], 0, 0};

                    v4i32 v_wr = (v4i32){(int32_t)wr, (int32_t)wr, 0, 0};
                    v4i32 v_wi = (v4i32){(int32_t)wi, (int32_t)wi, 0, 0};

                    v4i32 tr_vec = __msa_subv_w(__msa_mulr_q_w(r_j2, v_wr),
                                                __msa_mulr_q_w(i_j2, v_wi));
                    v4i32 ti_vec = __msa_addv_w(__msa_mulr_q_w(i_j2, v_wr),
                                                __msa_mulr_q_w(r_j2, v_wi));

                    v4i32 r_j2_new = __msa_subv_w(r_j, tr_vec);
                    v4i32 r_j_new  = __msa_addv_w(r_j, tr_vec);
                    v4i32 i_j2_new = __msa_subv_w(i_j, ti_vec);
                    v4i32 i_j_new  = __msa_addv_w(i_j, ti_vec);

                    real[j]     = __msa_copy_s_w(r_j_new,  0);
                    real[j+4]   = __msa_copy_s_w(r_j_new,  1);
                    real[j2_0]  = __msa_copy_s_w(r_j2_new, 0);
                    real[j2_1]  = __msa_copy_s_w(r_j2_new, 1);
                    imag[j]     = __msa_copy_s_w(i_j_new,  0);
                    imag[j+4]   = __msa_copy_s_w(i_j_new,  1);
                    imag[j2_0]  = __msa_copy_s_w(i_j2_new, 0);
                    imag[j2_1]  = __msa_copy_s_w(i_j2_new, 1);
                }
            }
            /* Remaining stages: MSA-accelerated scalar butterfly */
            else {
                for (; j < j_end; j += m2) {
                    int j2 = j + m;
                    int32_t tr, ti;

                    /* MSA Q15 complex multiply: single instruction replaces int64 path */
                    v4i32 rj2 = (v4i32){real[j2], 0, 0, 0};
                    v4i32 ij2 = (v4i32){imag[j2], 0, 0, 0};
                    v4i32 v_wr = (v4i32){(int32_t)wr, 0, 0, 0};
                    v4i32 v_wi = (v4i32){(int32_t)wi, 0, 0, 0};

                    tr = __msa_copy_s_w(__msa_msubr_q_w(rj2, v_wr,
                           __msa_mulr_q_w(ij2, v_wi)), 0);
                    ti = __msa_copy_s_w(__msa_maddr_q_w(ij2, v_wr,
                           __msa_mulr_q_w(rj2, v_wi)), 0);

                    real[j2] = real[j] - tr;
                    imag[j2] = imag[j] - ti;
                    real[j]  += tr;
                    imag[j]  += ti;
                }
            }
        }
    }
    /* NOTE: No 1/N scaling in inverse — matches KissFFT convention. */
}

/* Real forward FFT: 512 real → 257 complex */
static inline void fft_q15_forward(const int32_t *real_in,
                                    int32_t *real_out, int32_t *imag_out) {
    int32_t r_tmp[FFT_N], i_tmp[FFT_N];
    for (int i = 0; i < FFT_N; i++) { r_tmp[i] = real_in[i]; i_tmp[i] = 0; }
    fft_q15_512(r_tmp, i_tmp, 0);
    for (int i = 0; i < FFT_BINS; i++) { real_out[i] = r_tmp[i]; imag_out[i] = i_tmp[i]; }
}

/* Real inverse FFT: 257 complex → 512 real */
static inline void fft_q15_inverse(const int32_t *real_in, const int32_t *imag_in,
                                    int32_t *real_out) {
    int32_t r_tmp[FFT_N], i_tmp[FFT_N];
    r_tmp[0] = real_in[0]; i_tmp[0] = imag_in[0];
    r_tmp[FFT_N/2] = real_in[FFT_N/2]; i_tmp[FFT_N/2] = imag_in[FFT_N/2];
    for (int i = 1; i < FFT_N/2; i++) {
        r_tmp[i] = real_in[i]; i_tmp[i] = imag_in[i];
        r_tmp[FFT_N - i] = real_in[i]; i_tmp[FFT_N - i] = -imag_in[i];
    }
    fft_q15_512(r_tmp, i_tmp, 1);
    for (int i = 0; i < FFT_N; i++) real_out[i] = r_tmp[i];
}

#ifdef __cplusplus
}
#endif
#endif /* FFT_Q15_MSA_H */
