/**
 * ulunas_infer.c — UL-UNAS Top-Level Single-Frame Inference
 * ==========================================================
 * Pipeline: log_gen → BM → Encoder → GDPRNN×2 → Decoder → Sigmoid → BS → MASK
 */

#include "ulunas_fp.h"
#include "ulunas_lut.h"
#include "qr_config.h"
#include "layer_dims.h"
#include "ulunas_matlab_weights.h"

/* Sigmoid on Q20 int32_t input → uint16_t Q15 output, applied element-wise */
static void sigmoid_fixed(const int32_t *x, int N, uint16_t *y) {
    for (int i = 0; i < N; i++) {
        y[i] = sigmoid_q20_to_q15(x[i]);
    }
}

void ulunas_infer_frame(const float *spec_real, const float *spec_imag,
                        ulunas_state_t *st, int32_t *y_mask) {
    /* Step 0: Quantize STFT input float → Q20 */
    int32_t real_q20[257], imag_q20[257];
    for (int i = 0; i < 257; i++) {
        real_q20[i] = (int32_t)round(spec_real[i] * 1048576.0f);  /* ×2^20 */
        imag_q20[i] = (int32_t)round(spec_imag[i] * 1048576.0f);
    }

    /* Step 1: Log-magnitude Compression */
    int32_t x_log[257];
    log_gen_fixed(real_q20, imag_q20, 257, x_log);

    /* Step 2: BM — ERB Band Merging [257] → [129] */
    int32_t x_bm[129];
    bm_fixed(x_log, erb_erb_fc_weight, 257, 129, x_bm);

    /* Step 3: Encoder (5 layers) */
    int32_t e0[12 * 65], e1[24 * 33], e2[24 * 33], e3[32 * 33], e4[16 * 33];
    encoder_module(x_bm, st, e0, e1, e2, e3, e4);

    /* Step 4: GDPRNN ×2 */
    int32_t rnn1[16 * 33], rnn2[16 * 33];
    gdprnn_module(e4, st->inter_cache_0, 0, rnn1);
    gdprnn_module(rnn1, st->inter_cache_1, 1, rnn2);

    /* Step 5: Decoder → [1, 129] single-channel mask in Q20 */
    int32_t y_dec[1 * 129];
    decoder_module(rnn2, st, e0, e1, e2, e3, e4, y_dec);

    /* Step 6: Sigmoid → [1, 129] u16f15 */
    uint16_t y_sig[1 * 129];
    sigmoid_fixed(y_dec, 1 * 129, y_sig);

    /* Step 7: BS — ERB Band Splitting [1, 129] → [1, 257] */
    int16_t y_bs[257];
    bs_fixed(y_sig, erb_ierb_fc_weight, 129, 257, y_bs);

    /* Step 8: MASK → [2, 257] (real + imag masked output in Q20) */
    mask_fixed(y_bs, real_q20, imag_q20, 257, y_mask);
}
