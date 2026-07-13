/**
 * ulunas_nr.c — UL-UNAS 16kHz Native Noise Reduction
 * ==================================================
 * v8: 16kHz native, matching model training rate.
 *     Resampling handled externally by resample_8k_16k.h
 */
#include "noise_reduction.h"
#include "ulunas_fp.h"
#include "ulunas_lut.h"
#include "qr_config.h"
#include "layer_dims.h"
#include "ulunas_matlab_weights.h"
#include "fft_q15.h"
#include <stdlib.h>
#include <string.h>

#define N_FFT       512
#define WIN_LEN     512
#define WIN_INC     256
#define N_BINS      257
#define FRAME_IN    400      /* 25ms @ 16kHz */
#define FIFO_SZ     2048
#define WARMUP_MUTE 5
#define WARMUP_FADE 3
#define OUTPUT_GAIN_Q15 32768

static const int16_t stft_win_q15[512] = {
#include "stft_window_q15.inc"
};

static const uint32_t wola_inv_q30[256] = {
#include "wola_inv_q30.inc"
};

static ulunas_state_t g_state;
static int16_t g_fifo[FIFO_SZ];
static int g_fifo_wpos, g_fifo_count;
static int32_t g_ola[WIN_LEN + WIN_INC];
static int g_ola_pos;
static int16_t g_out_fifo[FIFO_SZ];
static int g_out_rpos, g_out_count;
static int g_frame_count;

void noise_init(void) {
    ulunas_state_init(&g_state);
    g_fifo_wpos = g_fifo_count = 0; g_ola_pos = 0;
    g_out_rpos = g_out_count = 0; g_frame_count = 0;
    memset(g_fifo, 0, sizeof(g_fifo));
    memset(g_ola, 0, sizeof(g_ola));
    memset(g_out_fifo, 0, sizeof(g_out_fifo));
}

void noise_deinit(void) { }

void noise_reduction(short *voiceIn, short *voiceOut) {
    /* Feed 16kHz PCM into FIFO */
    for (int i = 0; i < FRAME_IN; i++) {
        g_fifo[g_fifo_wpos] = voiceIn[i];
        g_fifo_wpos = (g_fifo_wpos + 1) % FIFO_SZ;
        g_fifo_count++;
    }

    /* Process STFT frames */
    int read_wpos = (g_fifo_wpos - g_fifo_count + FIFO_SZ) % FIFO_SZ;
    while (g_fifo_count >= WIN_LEN) {
        g_fifo_count -= WIN_INC;

        /* Window + FFT */
        int32_t fft_in[WIN_LEN];
        int start = (read_wpos - WIN_LEN + FIFO_SZ) % FIFO_SZ;
        for (int i = 0; i < WIN_LEN; i++) {
            int32_t v = g_fifo[(start + i) % FIFO_SZ];
            fft_in[i] = (int32_t)(((int64_t)v * stft_win_q15[i] + 16384) >> 15);
        }

        int32_t fwd_r[N_BINS], fwd_i[N_BINS];
        fft_q15_forward(fft_in, fwd_r, fwd_i);

        /* Q15→Q20 */
        int32_t real_q20[N_BINS], imag_q20[N_BINS];
        for (int j = 0; j < N_BINS; j++) {
            real_q20[j] = fwd_r[j] << 5;
            imag_q20[j] = fwd_i[j] << 5;
        }

        /* Full UL-UNAS inference */
        int32_t x_log[N_BINS];
        log_gen_fixed(real_q20, imag_q20, N_BINS, x_log);

        int32_t x_bm[129];
        bm_fixed(x_log, erb_erb_fc_weight, N_BINS, 129, x_bm);

        int32_t e0[12*65], e1[24*33], e2[24*33], e3[32*33], e4[16*33];
        encoder_module(x_bm, &g_state, e0, e1, e2, e3, e4);

        int32_t rnn1[16*33], rnn2[16*33];
        gdprnn_module(e4, g_state.inter_cache_0, 0, rnn1);
        gdprnn_module(rnn1, g_state.inter_cache_1, 1, rnn2);

        int32_t y_dec[1*129];
        decoder_module(rnn2, &g_state, e0, e1, e2, e3, e4, y_dec);

        uint16_t y_sig[1*129];
        for (int j = 0; j < 129; j++)
            y_sig[j] = sigmoid_q20_to_q15(y_dec[j]);

        int16_t y_bs[N_BINS];
        bs_fixed(y_sig, erb_ierb_fc_weight, 129, N_BINS, y_bs);

        int32_t crm[2*N_BINS];
        mask_fixed(y_bs, real_q20, imag_q20, N_BINS, crm);

        /* IFFT */
        int32_t inv_r[N_BINS], inv_i[N_BINS];
        for (int i = 0; i < N_BINS; i++) {
            inv_r[i] = (crm[i] + 16) >> 5;
            inv_i[i] = (crm[N_BINS + i] + 16) >> 5;
        }
        int32_t ifft_out[WIN_LEN];
        fft_q15_inverse(inv_r, inv_i, ifft_out);

        /* OLA */
        for (int i = 0; i < WIN_LEN; i++) {
            int32_t s = (int32_t)(((int64_t)ifft_out[i] * stft_win_q15[i] + 8388608) >> 24);
            int pos = (g_ola_pos + i) % (WIN_LEN + WIN_INC);
            g_ola[pos] += s;
        }

        /* WOLA output */
        for (int i = 0; i < WIN_INC; i++) {
            int32_t v = g_ola[g_ola_pos];
            g_ola[g_ola_pos] = 0;
            int idx = g_ola_pos % 256;
            g_ola_pos = (g_ola_pos + 1) % (WIN_LEN + WIN_INC);
            int64_t norm = ((int64_t)v * (int64_t)wola_inv_q30[idx] + 16384) >> 15;
            v = (int32_t)(((int64_t)norm * OUTPUT_GAIN_Q15 + 16384) >> 15);
            if (v > 32767) v = 32767; if (v < -32768) v = -32768;
            g_out_fifo[(g_out_rpos + g_out_count) % FIFO_SZ] = (int16_t)v;
            g_out_count++;
        }
        g_frame_count++;
        read_wpos = (read_wpos + WIN_INC) % FIFO_SZ;
    }

    /* Output */
    if (g_out_count >= FRAME_IN) {
        int32_t gain = 32768;
        if (g_frame_count < WARMUP_MUTE) gain = 0;
        else if (g_frame_count < WARMUP_MUTE + WARMUP_FADE)
            gain = ((g_frame_count - WARMUP_MUTE) * 32768) / WARMUP_FADE;
        for (int i = 0; i < FRAME_IN; i++) {
            int32_t o = ((int32_t)g_out_fifo[g_out_rpos] * gain + 16384) >> 15;
            g_out_rpos = (g_out_rpos + 1) % FIFO_SZ;
            if (o > 32767) o = 32767; if (o < -32768) o = -32768;
            voiceOut[i] = (int16_t)o;
        }
        g_out_count -= FRAME_IN;
    } else {
        memset(voiceOut, 0, FRAME_IN * sizeof(short));
    }
}
