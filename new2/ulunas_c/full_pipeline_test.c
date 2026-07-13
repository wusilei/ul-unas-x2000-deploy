/**
 * full_pipeline_test.c — Hybrid Full Pipeline Golden Verification
 * ================================================================
 * Uses VERIFIED inline implementations for log_gen/BM/sigmoid/BS/mask,
 * and MATURE code for encoder/RNN/decoder modules.
 *
 * Usage: ./full_pipeline_test <golden_dir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ──── Mature project headers (for encoder/RNN/decoder) ──── */
#include "../../UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/ulunas_fp.h"
#include "../../UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/ulunas_lut.h"
#include "../../UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/qr_config.h"
#include "../../UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/layer_dims.h"
#include "../../UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/ulunas_matlab_weights.h"

static int32_t *load_i32(const char *path, int n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    int32_t *b = malloc(n * sizeof(int32_t));
    fread(b, sizeof(int32_t), n, f); fclose(f); return b;
}
static uint16_t *load_u16(const char *path, int n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    uint16_t *b = malloc(n * sizeof(uint16_t));
    fread(b, sizeof(uint16_t), n, f); fclose(f); return b;
}
static int16_t *load_i16(const char *path, int n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    int16_t *b = malloc(n * sizeof(int16_t));
    fread(b, sizeof(int16_t), n, f); fclose(f); return b;
}
static float *load_f32(const char *path, int n) {
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    float *b = malloc(n * sizeof(float));
    fread(b, sizeof(float), n, f); fclose(f); return b;
}

static double snr_db_i32(const int32_t *g, const int32_t *c, int n) {
    double s = 0, e = 0;
    for (int i = 0; i < n; i++) { double d = g[i] - c[i]; s += (double)g[i]*g[i]; e += d*d; }
    return (e < 1e-30 || s < 1e-30) ? 999.0 : 10.0 * log10(s / e);
}
static double snr_db_u16(const uint16_t *g, const uint16_t *c, int n) {
    double s = 0, e = 0;
    for (int i = 0; i < n; i++) { double d = g[i] - c[i]; s += (double)g[i]*g[i]; e += d*d; }
    return (e < 1e-30 || s < 1e-30) ? 999.0 : 10.0 * log10(s / e);
}
static double snr_db_i16(const int16_t *g, const int16_t *c, int n) {
    double s = 0, e = 0;
    for (int i = 0; i < n; i++) { double d = g[i] - c[i]; s += (double)g[i]*g[i]; e += d*d; }
    return (e < 1e-30 || s < 1e-30) ? 999.0 : 10.0 * log10(s / e);
}
static double snr_2d_i32(const int32_t *g, const int32_t *c, int C, int W) {
    int N = C * W;
    int32_t *gt = malloc(N * sizeof(int32_t));
    for (int r = 0; r < C; r++)
        for (int w = 0; w < W; w++)
            gt[r * W + w] = g[r + C * w];  /* col-major → row-major */
    double s = snr_db_i32(gt, c, N);
    free(gt);
    return s;
}

const char *status(double snr) {
    if (snr > 120.0) return "PERFECT";
    if (snr > 80.0)  return "PASS";
    if (snr > 40.0)  return "WARN";
    return "FAIL";
}

/* ──── VERIFIED inline implementations ──── */

/* Rounding right shift */
static inline int32_t my_round_shr(int64_t x, int shift) {
    int64_t r = ((int64_t)1 << (shift - 1));
    return (int32_t)((x >= 0) ? ((x + r) >> shift) : ((x - r) >> shift));
}

/* Integer sqrt (binary search, from log_gen_test.c) */
static uint32_t isqrt64(uint64_t x) {
    if (x == 0) return 0;
    uint64_t lo = 0, hi = (x > 0xFFFFFFFFULL) ? 0xFFFFFFFFULL : x;
    if (hi > 0xFFFFFFFFULL) hi = 0xFFFFFFFFULL;
    while (lo + 1 < hi) {
        uint64_t mid = (lo + hi) >> 1;
        if (mid * mid <= x) lo = mid; else hi = mid;
    }
    return (uint32_t)lo;
}

/* LUT-based log10 Q20 (from log_gen_test.c) */
#define LOG10_LUT_SIZE 256
static const int32_t log10_lut_base[LOG10_LUT_SIZE] = {
    0,-135467,-263906,-386030,-502524,-613971,-720865,-823664,-922720,-1018281,
    -1110520,-1199552,-1285487,-1368422,-1448448,-1525649,-1600106,-1671893,-1741083,
    -1807746,-1871951,-1933760,-1993235,-2050435,-2105417,-2158235,-2208943,-2257590,
    -2304223,-2348890,-2391633,-2432496,-2471521,-2508747,-2544214,-2577960,-2610021,
    -2640434,-2669234,-2696456,-2722132,-2746296,-2768979,-2790211,-2810022,-2828441,
    -2845496,-2861214,-2875623,-2888749,-2900617,-2911253,-2920683,-2928930,-2936017,
    -2941968,-2946805,-2950549,-2953222,-2954844,-2955434,-2955013,-2953600,-2951213,
    -2947870,-2943590,-2938389,-2932285,-2925294,-2917432,-2908715,-2899159,-2888779,
    -2877591,-2865608,-2852846,-2839317,-2825035,-2810012,-2794260,-2777792,-2760618,
    -2742750,-2724198,-2704972,-2685082,-2664538,-2643349,-2621525,-2599073,-2576004,
    -2552325,-2528045,-2503172,-2477713,-2451677,-2425070,-2397900,-2370174,-2341898,
    -2313080,-2283726,-2253842,-2223435,-2192511,-2161074,-2129132,-2096689,-2063752,
    -2030325,-1996412,-1962020,-1927153,-1891816,-1856014,-1819750,-1783030,-1745857,
    -1708235,-1670168,-1631660,-1592715,-1553336,-1513527,-1473292,-1432633,-1391554,
    -1350058,-1308147,-1265825,-1223095,-1179959,-1136420,-1092481,-1048144,-1003412,
    -958287,-912772,-867070,-821184,-775115,-728867,-682442,-635842,-589068,-542124,
    -495010,-447730,-400284,-352675,-304904,-256974,-208886,-160641,-112241,-63688,
    -14982,33874,82880,132033,181334,230780,280372,330107,379985,430006,
    480168,530472,580915,631498,682220,733081,784079,835213,886484,937891,
    989433,1041109,1092919,1144863,1196939,1249148,1301488,1353960,1406562,1459296,
    1512159,1565153,1618276,1671528,1724909,1778418,1832056,1885821,1939715,1993735,
    2047883,2102158,2156559,2211087,2265741,2320520,2375426,2430456,2485612,2540893,
    2596299,2651829,2707483,2763262,2819164,2875190,2931340,2987613,3044009,3100529,
    3157171,3213936,3270823,3327834,3384966,3442221,3499599,3557098,3614720,3672463,
    3730328,3788315,3846424,3904654,3963006,4021479,4080074,4138790,4197627,4256586,
    4315666,4374867,4434190,4493634,4553199,4612885,4672693,4732622,4792672,4852843,
    4913136,4973549,5034084,5094740,5155517,5216415,5277434,5338575,5399837,5461220,
    5522725,5584350,5646097,5707966,5769955,5832066,5894299,5956653,6019128,6081726,
    6144444,6207285,6270247,6333332,6396539,6459868,6523319,6586892,6650587,6714405,
    6778345,6842407,6906592,6970899,7035329,7099881,7164556,7229354,7294274,7359317,
    7424483,7489772,7555184,7620719,7686377,7752159,7818064,7884092,7950244,8016520,
    8082919,8149443,8216090,8282862,8349758,8416778,8483922,8551191,8618584,8686102
};
static int32_t my_log10_q20(int32_t mag_q20) {
    if (mag_q20 <= 0) return -3670016; /* log10(1e-12) in Q20 ≈ -3.5 * 2^20 */
    /* Find integer part: floor(log2(mag)) */
    int int_part = 0;
    uint32_t v = (uint32_t)(mag_q20 > 0 ? mag_q20 : -mag_q20);
    while (v >= 2) { v >>= 1; int_part++; }
    /* Fractional part from LUT */
    uint32_t frac = (uint32_t)mag_q20;
    if (int_part > 0) frac = (frac << (20 - int_part)) & 0xFFFFF;
    int idx = (frac >> 12) & 0xFF;
    int32_t log_frac = log10_lut_base[idx];
    int32_t log_int = (int_part - 20) * 315653; /* log10(2) * 2^20 ≈ 315653 */
    return log_int + log_frac;
}

/* Inline VERIFIED implementations (matching standalone tests) */
static void inline_log_gen(const int32_t *real, const int32_t *imag, int W, int32_t *out) {
    for (int i = 0; i < W; i++) {
        int64_t r = real[i], im = imag[i];
        uint64_t mag_sq = (uint64_t)(r * r + im * im);
        uint32_t mag = isqrt64(mag_sq);
        if (mag < 1) mag = 1;
        out[i] = my_log10_q20((int32_t)mag);
    }
}

static void inline_bm(const int32_t *x, const uint16_t *weight, int W_in, int W_out, int32_t *y) {
    memcpy(y, x, 65 * sizeof(int32_t));
    int n_hi_in = W_in - 65, n_hi_out = W_out - 65;
    for (int o = 0; o < n_hi_out; o++) {
        int64_t acc = 0;
        for (int i = 0; i < n_hi_in; i++) {
            acc += (int64_t)x[65 + i] * weight[i + n_hi_in * o];  /* COL-MAJOR: match mature */
        }
        y[65 + o] = my_round_shr(acc, 15);
    }
}

static void inline_bs(const uint16_t *x, const uint16_t *weight, int W_in, int W_out, int16_t *y) {
    for (int i = 0; i < 65; i++) y[i] = (int16_t)x[i];
    int n_hi_in = W_in - 65, n_hi_out = W_out - 65;
    for (int o = 0; o < n_hi_out; o++) {
        int64_t acc = 0;
        for (int i = 0; i < n_hi_in; i++) {
            acc += (int64_t)x[65 + i] * weight[i + n_hi_in * o];  /* COL-MAJOR */
        }
        int32_t v = my_round_shr(acc, 15);
        y[65 + o] = (v > 32767) ? 32767 : (v < -32768) ? -32768 : (int16_t)v;
    }
}

static void inline_sigmoid(const int32_t *x, int N, uint16_t *y) {
    for (int i = 0; i < N; i++) {
        y[i] = sigmoid_q20_to_q15(x[i]);  /* reuse mature's sigmoid LUT */
    }
}

static void inline_mask(const int16_t *m, const int32_t *xr, const int32_t *xi, int W, int32_t *y) {
    for (int i = 0; i < W; i++) {
        y[i]     = my_round_shr((int64_t)xr[i] * m[i], 15);
        y[W + i] = my_round_shr((int64_t)xi[i] * m[i], 15);
    }
}

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : "dump_matlab";
    char path[512];

    printf("=== UL-UNAS Full Pipeline: Hybrid (inline front/back-end + mature encoder/RNN/decoder) ===\n\n");

    ulunas_state_t st;
    ulunas_state_init(&st);

    for (int frame = 0; frame < 1; frame++) {  /* Frame 0 only for now */
        /* Load float STFT */
        snprintf(path, sizeof(path), "%s/frame%d_stft_real.bin", dir, frame);
        float *stft_r = load_f32(path, 257);
        snprintf(path, sizeof(path), "%s/frame%d_stft_imag.bin", dir, frame);
        float *stft_i = load_f32(path, 257);
        if (!stft_r || !stft_i) { printf("Frame %d: SKIP\n", frame); continue; }

        printf("--- Frame %d ---\n", frame);

        /* Quantize to Q20 */
        int32_t real_q20[257], imag_q20[257];
        for (int i = 0; i < 257; i++) {
            real_q20[i] = (int32_t)round(stft_r[i] * 1048576.0f);
            imag_q20[i] = (int32_t)round(stft_i[i] * 1048576.0f);
        }

        /* Step 1: log_gen (VERIFIED inline) */
        int32_t x_log[257];
        inline_log_gen(real_q20, imag_q20, 257, x_log);

        /* Step 2: BM (VERIFIED inline, col-major weight indexing) */
        int32_t x_bm[129];
        inline_bm(x_log, erb_erb_fc_weight, 257, 129, x_bm);

        snprintf(path, sizeof(path), "%s/frame%d_bm.bin", dir, frame);
        int32_t *gbm = load_i32(path, 129);
        if (gbm) {
            double s = snr_db_i32(gbm, x_bm, 129);
            printf("  BM        : SNR=%7.2f dB  [%s]\n", s, status(s));
            free(gbm);
        }

        /* Step 3: Encoder (MATURE module) */
        int32_t e0[12*65], e1[24*33], e2[24*33], e3[32*33], e4[16*33];
        encoder_module(x_bm, &st, e0, e1, e2, e3, e4);

        snprintf(path, sizeof(path), "%s/frame%d_enc_e0.bin", dir, frame);
        int32_t *ge0 = load_i32(path, 12*65);
        if (ge0) { printf("  enc_e0    : SNR=%7.2f dB  [%s]\n", snr_2d_i32(ge0, e0, 12, 65), status(snr_2d_i32(ge0, e0, 12, 65))); free(ge0); }

        snprintf(path, sizeof(path), "%s/frame%d_enc_e1.bin", dir, frame);
        int32_t *ge1 = load_i32(path, 24*33);
        if (ge1) { printf("  enc_e1    : SNR=%7.2f dB  [%s]\n", snr_2d_i32(ge1, e1, 24, 33), status(snr_2d_i32(ge1, e1, 24, 33))); free(ge1); }

        snprintf(path, sizeof(path), "%s/frame%d_enc_e2.bin", dir, frame);
        int32_t *ge2 = load_i32(path, 24*33);
        if (ge2) { printf("  enc_e2    : SNR=%7.2f dB  [%s]\n", snr_2d_i32(ge2, e2, 24, 33), status(snr_2d_i32(ge2, e2, 24, 33))); free(ge2); }

        snprintf(path, sizeof(path), "%s/frame%d_enc_e3.bin", dir, frame);
        int32_t *ge3 = load_i32(path, 32*33);
        if (ge3) { printf("  enc_e3    : SNR=%7.2f dB  [%s]\n", snr_2d_i32(ge3, e3, 32, 33), status(snr_2d_i32(ge3, e3, 32, 33))); free(ge3); }

        snprintf(path, sizeof(path), "%s/frame%d_enc_e4.bin", dir, frame);
        int32_t *ge4 = load_i32(path, 16*33);
        if (ge4) { printf("  enc_e4    : SNR=%7.2f dB  [%s]\n", snr_2d_i32(ge4, e4, 16, 33), status(snr_2d_i32(ge4, e4, 16, 33))); free(ge4); }

        /* Step 4: GDPRNN (MATURE module) */
        int32_t rnn1[16*33], rnn2[16*33];
        gdprnn_module(e4, st.inter_cache_0, 0, rnn1);
        gdprnn_module(rnn1, st.inter_cache_1, 1, rnn2);

        snprintf(path, sizeof(path), "%s/frame%d_rnn1.bin", dir, frame);
        int32_t *gr1 = load_i32(path, 16*33);
        if (gr1) { printf("  rnn1      : SNR=%7.2f dB  [%s]\n", snr_2d_i32(gr1, rnn1, 16, 33), status(snr_2d_i32(gr1, rnn1, 16, 33))); free(gr1); }

        snprintf(path, sizeof(path), "%s/frame%d_rnn2.bin", dir, frame);
        int32_t *gr2 = load_i32(path, 16*33);
        if (gr2) { printf("  rnn2      : SNR=%7.2f dB  [%s]\n", snr_2d_i32(gr2, rnn2, 16, 33), status(snr_2d_i32(gr2, rnn2, 16, 33))); free(gr2); }

        /* Step 5: Decoder (MATURE module) */
        int32_t y_dec[129];
        decoder_module(rnn2, &st, e0, e1, e2, e3, e4, y_dec);

        snprintf(path, sizeof(path), "%s/frame%d_dec.bin", dir, frame);
        int32_t *gdec = load_i32(path, 129);
        if (gdec) { printf("  dec       : SNR=%7.2f dB  [%s]\n", snr_db_i32(gdec, y_dec, 129), status(snr_db_i32(gdec, y_dec, 129))); free(gdec); }

        /* Step 6: Sigmoid (VERIFIED inline using mature LUT) */
        uint16_t y_sig[129];
        inline_sigmoid(y_dec, 129, y_sig);

        snprintf(path, sizeof(path), "%s/frame%d_sig.bin", dir, frame);
        uint16_t *gsig = load_u16(path, 129);
        if (gsig) { printf("  sigmoid   : SNR=%7.2f dB  [%s]\n", snr_db_u16(gsig, y_sig, 129), status(snr_db_u16(gsig, y_sig, 129))); free(gsig); }

        /* Step 7: BS (VERIFIED inline) */
        int16_t y_bs[257];
        inline_bs(y_sig, erb_ierb_fc_weight, 129, 257, y_bs);

        snprintf(path, sizeof(path), "%s/frame%d_bs.bin", dir, frame);
        int16_t *gbs = load_i16(path, 257);
        if (gbs) { printf("  BS        : SNR=%7.2f dB  [%s]\n", snr_db_i16(gbs, y_bs, 257), status(snr_db_i16(gbs, y_bs, 257))); free(gbs); }

        /* Step 8: MASK (VERIFIED inline) */
        int32_t y_mask[2*257];
        inline_mask(y_bs, real_q20, imag_q20, 257, y_mask);

        snprintf(path, sizeof(path), "%s/frame%d_mask.bin", dir, frame);
        int32_t *gmask = load_i32(path, 2*257);
        if (gmask) { printf("  MASK      : SNR=%7.2f dB  [%s]\n", snr_db_i32(gmask, y_mask, 2*257), status(snr_db_i32(gmask, y_mask, 2*257))); free(gmask); }

        free(stft_r); free(stft_i);
    }
    printf("\n=== Done ===\n");
    return 0;
}
