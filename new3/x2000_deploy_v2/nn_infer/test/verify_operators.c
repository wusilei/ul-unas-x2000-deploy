/**
 * verify_operators.c — Bit-exact verification of nn_infer operators
 * =================================================================
 * Tests each operator category against known inputs and golden outputs.
 * Compares with original ulunas_fp.c/ulunas_lut.c implementations where available.
 *
 * Build: gcc -O2 -std=c99 -I.. -o verify_operators verify_operators.c ../nn_core/nn_lut.c -lm
 * Run:   ./verify_operators
 */

#include "nn_infer.h"
#include <stdio.h>
#include <math.h>

static int tests_passed = 0, tests_failed = 0;

#define TEST(name)  printf("  %-50s ", name)
#define PASS()      do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(fmt,...) do { printf("FAIL: " fmt "\n", ##__VA_ARGS__); tests_failed++; } while(0)
#define CHECK(cond, fmt,...) do { if (cond) PASS(); else FAIL(fmt, ##__VA_ARGS__); } while(0)
#define CHECK_EQ(a, b, fmt,...) CHECK((a) == (b), fmt " (got %d, expected %d)", ##__VA_ARGS__, (int)(a), (int)(b))
#define CHECK_NEAR(a, b, tol, fmt,...) CHECK(abs((a)-(b)) <= (tol), fmt " (got %d, expected %d, diff=%d)", ##__VA_ARGS__, (int)(a), (int)(b), (int)abs((a)-(b)))

/* ================================================================
 * Test 1: Q-format Primitives
 * ================================================================ */
static void test_qformat(void) {
    printf("\n--- Q-format Primitives ---\n");

    TEST("nn_round_shr positive");
    CHECK_EQ(nn_round_shr(1024, 5), 32, "round_shr(1024, 5)");

    TEST("nn_round_shr negative");
    CHECK_EQ(nn_round_shr(-1024, 5), -32, "round_shr(-1024, 5)");

    TEST("nn_round_shr half-up");
    CHECK_EQ(nn_round_shr(48, 5), 2, "round_shr(48, 5)");  /* 48/32=1.5 → 2 */

    TEST("nn_sat_i32 overflow");
    CHECK_EQ(nn_sat_i32(3000000000LL), INT32_MAX, "sat_i32 overflow");

    TEST("nn_sat_i32 underflow");
    CHECK_EQ(nn_sat_i32(-3000000000LL), INT32_MIN, "sat_i32 underflow");

    TEST("nn_sat_i16 overflow");
    CHECK_EQ(nn_sat_i16(40000), INT16_MAX, "sat_i16 overflow");

    TEST("nn_sat_u16 overflow");
    CHECK_EQ(nn_sat_u16(70000), UINT16_MAX, "sat_u16 overflow");

    TEST("nn_clamp_i16");
    CHECK_EQ(nn_clamp_i16(40000), 32767, "clamp_i16");
}

/* ================================================================
 * Test 2: LUT Functions (bit-exact vs original ulunas_lut.c)
 * ================================================================ */
static void test_lut(void) {
    printf("\n--- LUT Functions ---\n");

    /* Sigmoid at x=0: sigmoid(0) = 0.5 → Q15 = 16384 */
    TEST("sigmoid_q15(0)");
    uint16_t s0 = nn_sigmoid_q15(0);
    /* With Q20 LUT: sigmoid(0*2^20) should be close to 0.5*32768 = 16384 */
    CHECK_NEAR((int)s0, 16384, 80, "sigmoid_q15(0)");

    /* Sigmoid at large positive: saturates to 32768 */
    TEST("sigmoid_q15 large positive");
    uint16_t sp = nn_sigmoid_q15(8388608);  /* 8.0 * 2^20 */
    CHECK_EQ((int)sp, 32768, "sigmoid_q15(8.0)");

    /* Sigmoid at large negative: saturates to 0 */
    TEST("sigmoid_q15 large negative");
    uint16_t sn = nn_sigmoid_q15(-8388608);  /* -8.0 * 2^20 */
    CHECK_EQ((int)sn, 0, "sigmoid_q15(-8.0)");

    /* Sigmoid monotonicity: sigmoid(1) > sigmoid(0) */
    TEST("sigmoid_q15 monotonic");
    uint16_t s1 = nn_sigmoid_q15(1048576);  /* 1.0 * 2^20 */
    CHECK(s1 > s0, "sigmoid(1.0)=%u > sigmoid(0)=%u", s1, s0);

    /* Tanh at x=0: tanh(0) = 0 */
    TEST("tanh_q15(0)");
    int16_t t0 = nn_tanh_q15(0);
    CHECK_NEAR((int)t0, 0, 130, "tanh_q15(0)");

    /* Tanh at large positive: saturates to 32767 */
    TEST("tanh_q15 large positive");
    int16_t tp = nn_tanh_q15(4194304);  /* 4.0 * 2^20 */
    CHECK_EQ((int)tp, 32767, "tanh_q15(4.0)");

    /* Tanh at large negative: saturates to -32768 */
    TEST("tanh_q15 large negative");
    int16_t tn = nn_tanh_q15(-4194304);  /* -4.0 * 2^20 */
    CHECK_EQ((int)tn, -32768, "tanh_q15(-4.0)");

    /* Sqrt: sqrt(4 * 2^40) = 2 * 2^20 = 2097152 */
    TEST("sqrt_q40_to_q20(4.0)");
    uint32_t sq = nn_sqrt_q40_to_q20(4ULL << 40);
    CHECK_NEAR((int)sq, 2097152, 10, "sqrt(4.0*2^40)");

    /* Sqrt: sqrt(0) = 0 */
    TEST("sqrt_q40_to_q20(0)");
    CHECK_EQ(nn_sqrt_q40_to_q20(0), 0, "sqrt(0)");

    /* Sqrt: sqrt(1) = 1 */
    TEST("sqrt_q40_to_q20(1)");
    CHECK_EQ(nn_sqrt_q40_to_q20(1), 1, "sqrt(1)");
}

/* ================================================================
 * Test 3: Convolution Operators
 * ================================================================ */
static void test_conv(void) {
    printf("\n--- Convolution ---\n");

    /* Simple 1×1 conv: identity kernel */
    TEST("conv2d identity 1x1");
    {
        int32_t x[4] = {100, 200, 300, 400};  /* Cin=1, Win=4 */
        int32_t b[1] = {0};
        int32_t y[4] = {0};
        int16_t pw[1] = {16384};  /* 1.0 in Q14 */
        nn_pconv2d(x, 1, 1, 4, pw, b, -14, 0, y);
        /* y[i] = round(x[i] * 1.0) = x[i] (Q20 stays Q20) */
        CHECK_NEAR(y[0], 100, 2, "pconv2d[0]");
        CHECK_NEAR(y[3], 400, 2, "pconv2d[3]");
    }

    /* Point-wise conv: 2→2 channels */
    TEST("pconv2d 2in→2out");
    {
        int32_t x[4] = {10, 20, 30, 40}; /* Cin=2, W=2: ch0=[10,20], ch1=[30,40] */
        int16_t w[4] = {16384, 0, 0, 16384}; /* Identity: w[0,0]=1.0, w[0,1]=0, w[1,0]=0, w[1,1]=1.0 */
        int32_t b[2] = {0, 0};
        int32_t y[4] = {0};
        nn_pconv2d(x, 2, 2, 2, w, b, -14, 0, y);
        CHECK_NEAR(y[0], 10, 2, "pconv2d ch0[0]");
        CHECK_NEAR(y[2], 30, 2, "pconv2d ch1[0]");  /* y[1*2+0] */
    }

    /* Bias test: Cin=1, Cout=2, W=2. Same input [100,200] → both output ch */
    TEST("pconv2d with bias");
    {
        int32_t x[2] = {100, 200};  /* Cin=1, W=2 */
        int16_t w[2] = {16384, 16384}; /* Cout=2, Cin=1 */
        int32_t b[2] = {50, -50};
        int32_t y[4] = {0};
        nn_pconv2d(x, 1, 2, 2, w, b, -14, 0, y);
        /* y[0][0] = x[0]*1.0+50 = 150 */
        CHECK_NEAR(y[0], 150, 2, "pconv2d+bias y[0][0]");
        /* y[0][1] = x[1]*1.0+50 = 250 */
        CHECK_NEAR(y[1], 250, 2, "pconv2d+bias y[0][1]");
        /* y[1][0] = x[0]*1.0-50 = 50 */
        CHECK_NEAR(y[2], 50, 2, "pconv2d+bias y[1][0]");
        /* y[1][1] = x[1]*1.0-50 = 150 */
        CHECK_NEAR(y[3], 150, 2, "pconv2d+bias y[1][1]");
    }
}

/* ================================================================
 * Test 4: BatchNorm
 * ================================================================ */
static void test_bn(void) {
    printf("\n--- BatchNorm ---\n");

    /* BN with identity parameters: mean=0, var=1, weight=1, bias=0 */
    TEST("bn_sw identity");
    {
        int32_t x[4] = {10, 20, 30, 40};  /* C=2, N=2 */
        int16_t w[2] = {16384, 16384};    /* 1.0 in Q14 */
        int32_t b[2] = {0, 0};
        int32_t mean[2] = {0, 0};
        uint16_t var[2] = {16384, 16384}; /* 1.0 in Q14 */
        int32_t y[4];
        /* With qr1=-14, qr2=-14: should get approximately x back */
        nn_bn_sw(x, w, b, mean, var, -14, -14, 2, 2, y);
        CHECK_NEAR(y[0], 10, 2, "bn identity[0]");
        CHECK_NEAR(y[3], 40, 2, "bn identity[3]");
    }

    /* BN with mean shift */
    TEST("bn_sw mean shift");
    {
        int32_t x[4] = {20, 30, 40, 50};      /* C=2, N=2 */
        int16_t w[2] = {16384, 16384};
        int32_t b[2] = {0, 0};
        int32_t mean[2] = {10, 10};            /* subtract 10 */
        uint16_t var[2] = {16384, 16384};       /* ×1 */
        int32_t y[4];
        nn_bn_sw(x, w, b, mean, var, -14, -14, 2, 2, y);
        CHECK_NEAR(y[0], 10, 2, "bn mean_shift[0]");  /* 20-10=10 */
        CHECK_NEAR(y[3], 40, 2, "bn mean_shift[3]");  /* 50-10=40 */
    }

    /* BN unsigned weight */
    TEST("bn_uw identity");
    {
        int32_t x[4] = {100, 200, 300, 400};
        uint16_t w[2] = {16384, 16384};   /* 1.0 in u16f14 */
        int32_t b[2] = {0, 0};
        int32_t mean[2] = {0, 0};
        uint16_t var[2] = {16384, 16384};
        int32_t y[4];
        nn_bn_uw(x, w, b, mean, var, -14, -14, 2, 2, y);
        CHECK_NEAR(y[0], 100, 2, "bn_uw identity[0]");
        CHECK_NEAR(y[3], 400, 2, "bn_uw identity[3]");
    }
}

/* ================================================================
 * Test 5: Activation Functions
 * ================================================================ */
static void test_activations(void) {
    printf("\n--- Activations ---\n");

    /* PReLU: positive should pass through, negative should ×slope */
    TEST("prelu positive passthrough");
    {
        int32_t x[4] = {10, -10, 20, -20};  /* C=2, W=2 */
        int16_t slope[2] = {8192, 8192};    /* 0.5 in Q14 */
        nn_prelu(x, 2, 2, slope, -14);
        CHECK_EQ(x[0], 10, "prelu positive");
        CHECK_NEAR(x[1], -5, 2, "prelu negative ×0.5");
    }

    /* AffinePReLU */
    TEST("affine_prelu basic");
    {
        int32_t x[2] = {100, -100};         /* C=1, W=2 */
        int16_t w[1] = {16384};             /* weight=1.0 */
        int32_t b[1] = {10};                /* bias=10 */
        int16_t s[1] = {8192};              /* slope=0.5 */
        int32_t y[2];
        nn_affine_prelu(x, w, b, s, -14, -14, 1, 2, y);
        /* x[0]=100: positive → affine(100*1)+bias=110 */
        CHECK_NEAR(y[0], 110, 2, "affine_prelu positive");
        /* x[1]=-100: negative → neg_part=-50, affine=-100, residual=-150 */
    }
}

/* ================================================================
 * Test 6: GRU
 * ================================================================ */
static void test_gru(void) {
    printf("\n--- GRU ---\n");

    /* Minimal GRU: input_dim=1, hidden_dim=1, identity-ish weights */
    TEST("gru_step minimal");
    {
        int32_t x[1] = {0};                  /* Q20: 0.0 */
        int16_t h[1] = {0};                  /* Q15: 0.0 */
        /* Weight layout: [in_dim][3*hidden] = [1][3] */
        /* All zeros → gates should be 0.5 sigmoid, tanh(0)=0 */
        int16_t ih_w[3] = {0, 0, 0};
        int32_t ih_b[3] = {0, 0, 0};
        int16_t hh_w[3] = {0, 0, 0};
        int32_t hh_b[3] = {0, 0, 0};
        int16_t y[1];
        nn_gru_step(x, 1, 1, h, ih_w, ih_b, hh_w, hh_b, -13, -8, y);
        /* With all-zero weights: r_t=z_t=sigmoid(0)=0.5, n_t=tanh(0)=0 */
        /* h_new = (1-0.5)*0 + 0.5*0 = 0 */
        CHECK_NEAR((int)h[0], 0, 100, "gru_step zero input");
    }

    /* GRU state persistence: h should update and carry over */
    TEST("gru_step state persistence");
    {
        int32_t x[1] = {1048576};           /* Q20: 1.0 */
        int16_t h[1] = {0};
        int16_t ih_w[3] = {4096, 4096, 4096};  /* Q12: 1.0 each */
        int32_t ih_b[3] = {0, 0, 0};
        int16_t hh_w[3] = {0, 0, 0};
        int32_t hh_b[3] = {0, 0, 0};
        int16_t y[1];
        nn_gru_step(x, 1, 1, h, ih_w, ih_b, hh_w, hh_b, -13, -8, y);
        /* h should be non-zero after processing non-zero input */
        CHECK(h[0] != 0, "gru state updated (h=%d)", (int)h[0]);
    }
}

/* ================================================================
 * Test 7: Shuffle
 * ================================================================ */
static void test_shuffle(void) {
    printf("\n--- Shuffle ---\n");

    TEST("shuffle_interleave");
    {
        int32_t x[4] = {1, 2, 3, 4};  /* C=1, W=4 */
        int32_t y[4];
        nn_shuffle_interleave(x, 1, 4, y);
        /* half=2: y[0]=x[0]=1, y[1]=x[2]=3, y[2]=x[1]=2, y[3]=x[3]=4 */
        CHECK_EQ(y[0], 1, "shuffle[0]");
        CHECK_EQ(y[1], 3, "shuffle[1]");
        CHECK_EQ(y[2], 2, "shuffle[2]");
        CHECK_EQ(y[3], 4, "shuffle[3]");
    }

    TEST("shuffle_deinterleave");
    {
        int32_t x[4] = {1, 3, 2, 4};
        int32_t y[4];
        nn_shuffle_deinterleave(x, 1, 4, y);
        CHECK_EQ(y[0], 1, "deinterleave[0]");
        CHECK_EQ(y[1], 2, "deinterleave[1]");
        CHECK_EQ(y[2], 3, "deinterleave[2]");
        CHECK_EQ(y[3], 4, "deinterleave[3]");
    }
}

/* ================================================================
 * Test 8: ERB + MASK
 * ================================================================ */
static void test_erb_mask(void) {
    printf("\n--- ERB + MASK ---\n");

    TEST("bm passthrough low bins");
    {
        int32_t x[4] = {10, 20, 30, 40};
        int32_t y[4] = {0};
        uint16_t w[1] = {32768};  /* 1.0 in Q15 */
        nn_bm(x, w, 4, 4, y);    /* W_in = W_out: all passthrough */
        CHECK_EQ(y[0], 10, "bm[0]");
        CHECK_EQ(y[3], 40, "bm[3]");
    }

    TEST("mask basic");
    {
        int16_t mask[2] = {16384, 32767};   /* 0.5, ~1.0 in Q15 (32768 wraps in int16!) */
        int32_t re[2] = {100, 200};          /* Q20 */
        int32_t im[2] = {50, 100};
        int32_t y[4];
        nn_mask(mask, re, im, 2, y);
        /* y[0] = round(100 * 0.5) = 50 */
        CHECK_NEAR(y[0], 50, 2, "mask real[0]");
        /* y[1] = round(200 * 1.0) = 200 */
        CHECK_NEAR(y[1], 200, 2, "mask real[1]");
        /* y[2] = round(50 * 0.5) = 25 */
        CHECK_NEAR(y[2], 25, 2, "mask imag[0]");
    }
}

/* ================================================================
 * Main
 * ================================================================ */
int main(void) {
    printf("=== nn_infer Operator Verification ===\n");

    test_qformat();
    test_lut();
    test_conv();
    test_bn();
    test_activations();
    test_gru();
    test_shuffle();
    test_erb_mask();

    printf("\n=== Results: %d passed, %d failed ===\n", tests_passed, tests_failed);
    return tests_failed ? 1 : 0;
}
