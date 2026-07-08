/**
 * noise_reduction_q15.c — 全Q15 FFT (正向+逆向) + 定点OLA
 * ==========================================================
 * 改动 vs noise_reduction.c (混合版):
 *   - KissFFT 正向 → fft_q15_forward (Q15→float 输出给 DENOISE)
 *   - KissFFT 逆向 → fft_q15_inverse (已是 Q15, 不变)
 *   - OLA: float → int32 (消除 float 累加)
 *
 * 编译: 直接替换 noise_reduction.c
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
#define FRAME_IN    200
#define FRAME_16K   400
#define FIFO_SZ     (WIN_LEN * 4)
#define WARMUP_MUTE 5
#define WARMUP_FADE 3

/* Q15 stft_window from MATLAB stft_window.mat */
static const int16_t stft_win_q15[512] = {
       0,     1,     5,    11,    20,    31,    44,    60,    79,   100,   123,   149,   177,   208,   241,   277,
     315,   355,   398,   443,   491,   541,   593,   648,   705,   765,   827,   891,   958,  1027,  1098,  1171,
    1247,  1325,  1406,  1488,  1573,  1660,  1749,  1841,  1935,  2030,  2128,  2229,  2331,  2435,  2542,  2650,
    2761,  2874,  2989,  3105,  3224,  3345,  3468,  3592,  3719,  3847,  3978,  4110,  4244,  4380,  4518,  4657,
    4799,  4942,  5086,  5233,  5381,  5531,  5682,  5835,  5990,  6146,  6304,  6463,  6624,  6786,  6950,  7115,
    7281,  7449,  7618,  7789,  7961,  8134,  8308,  8484,  8660,  8838,  9017,  9197,  9379,  9561,  9744,  9929,
   10114, 10300, 10487, 10675, 10864, 11054, 11244, 11436, 11628, 11820, 12014, 12208, 12403, 12598, 12794, 12990,
   13187, 13385, 13583, 13781, 13980, 14179, 14378, 14578, 14778, 14978, 15178, 15379, 15580, 15780, 15981, 16182,
   16384, 16585, 16786, 16987, 17187, 17388, 17589, 17789, 17989, 18189, 18389, 18588, 18787, 18986, 19184, 19382,
   19580, 19777, 19973, 20169, 20364, 20559, 20753, 20947, 21139, 21331, 21523, 21713, 21903, 22092, 22280, 22467,
   22653, 22838, 23023, 23206, 23388, 23570, 23750, 23929, 24107, 24283, 24459, 24633, 24806, 24978, 25149, 25318,
   25486, 25652, 25817, 25981, 26143, 26304, 26463, 26621, 26777, 26932, 27085, 27236, 27386, 27534, 27681, 27825,
   27968, 28110, 28249, 28387, 28523, 28657, 28789, 28920, 29048, 29175, 29299, 29422, 29543, 29662, 29778, 29893,
   30006, 30117, 30225, 30332, 30436, 30538, 30639, 30737, 30832, 30926, 31018, 31107, 31194, 31279, 31361, 31442,
   31520, 31596, 31669, 31740, 31809, 31876, 31940, 32002, 32062, 32119, 32174, 32226, 32276, 32324, 32369, 32412,
   32452, 32490, 32526, 32559, 32590, 32618, 32644, 32667, 32688, 32707, 32723, 32736, 32747, 32756, 32762, 32766,
   32767, 32766, 32762, 32756, 32747, 32736, 32723, 32707, 32688, 32667, 32644, 32618, 32590, 32559, 32526, 32490,
   32452, 32412, 32369, 32324, 32276, 32226, 32174, 32119, 32062, 32002, 31940, 31876, 31809, 31740, 31669, 31596,
   31520, 31442, 31361, 31279, 31194, 31107, 31018, 30926, 30832, 30737, 30639, 30538, 30436, 30332, 30225, 30117,
   30006, 29893, 29778, 29662, 29543, 29422, 29299, 29175, 29048, 28920, 28789, 28657, 28523, 28387, 28249, 28110,
   27968, 27825, 27681, 27534, 27386, 27236, 27085, 26932, 26777, 26621, 26463, 26304, 26143, 25981, 25817, 25652,
   25486, 25318, 25149, 24978, 24806, 24633, 24459, 24283, 24107, 23929, 23750, 23570, 23388, 23206, 23023, 22838,
   22653, 22467, 22280, 22092, 21903, 21713, 21523, 21331, 21139, 20947, 20753, 20559, 20364, 20169, 19973, 19777,
   19580, 19382, 19184, 18986, 18787, 18588, 18389, 18189, 17989, 17789, 17589, 17388, 17187, 16987, 16786, 16585,
   16384, 16182, 15981, 15780, 15580, 15379, 15178, 14978, 14778, 14578, 14378, 14179, 13980, 13781, 13583, 13385,
   13187, 12990, 12794, 12598, 12403, 12208, 12014, 11820, 11628, 11436, 11244, 11054, 10864, 10675, 10487, 10300,
   10114,  9929,  9744,  9561,  9379,  9197,  9017,  8838,  8660,  8484,  8308,  8134,  7961,  7789,  7618,  7449,
    7281,  7115,  6950,  6786,  6624,  6463,  6304,  6146,  5990,  5835,  5682,  5531,  5381,  5233,  5086,  4942,
    4799,  4657,  4518,  4380,  4244,  4110,  3978,  3847,  3719,  3592,  3468,  3345,  3224,  3105,  2989,  2874,
    2761,  2650,  2542,  2435,  2331,  2229,  2128,  2030,  1935,  1841,  1749,  1660,  1573,  1488,  1406,  1325,
    1247,  1171,  1098,  1027,   958,   891,   827,   765,   705,   648,   593,   541,   491,   443,   398,   355,
     315,   277,   241,   208,   177,   149,   123,   100,    79,    60,    44,    31,    20,    11,     5,     1
};

void noise_init(void) {
    ulunas_state_init(&g_state);
    g_fifo_wpos = g_fifo_count = 0; g_ola_pos = 0;
    g_out_rpos = g_out_count = 0; g_frame_count = 0; g_last_in_8k = 0;
    memset(g_fifo, 0, sizeof(g_fifo));
    memset(g_ola, 0, sizeof(g_ola));
    memset(g_wola_sum, 0, sizeof(g_wola_sum));
    memset(g_out_fifo, 0, sizeof(g_out_fifo));
}

void noise_deinit(void) { }

void noise_reduction(short *voiceIn, short *voiceOut) {
    /* ── 1. 8kHz → 16kHz 上采样 (纯整数 Q15) ── */
    for (int i = 0; i < FRAME_IN; i++) {
        int16_t s_curr = voiceIn[i];
        int16_t s_prev = g_last_in_8k;
        g_fifo[g_fifo_wpos] = s_curr;
        g_fifo_wpos = (g_fifo_wpos + 1) % FIFO_SZ;
        g_fifo[g_fifo_wpos] = (int16_t)(((int)s_prev + (int)s_curr) >> 1);
        g_fifo_wpos = (g_fifo_wpos + 1) % FIFO_SZ;
        g_fifo_count += 2;
        g_last_in_8k = voiceIn[i];
    }

    /* ── 2. STFT→DENOISE→ISTFT ── */
    {
        int read_wpos = (g_fifo_wpos - g_fifo_count + WIN_LEN + FIFO_SZ) % FIFO_SZ;
        while (g_fifo_count >= WIN_LEN) {
            g_fifo_count -= WIN_INC;

            /* 2a. MATLAB stft_window: Q15×Q15 → Q15 */
            int32_t fft_in[WIN_LEN];
            int start = (read_wpos - WIN_LEN + FIFO_SZ) % FIFO_SZ;
            for (int i = 0; i < WIN_LEN; i++) {
                int32_t v = g_fifo[(start + i) % FIFO_SZ];
                fft_in[i] = ((int64_t)v * stft_win_q15[i] + 16384) >> 15;
            }

            /* 2b. Q15 Forward FFT: Q15 → Q15 output */
            int32_t fwd_r[N_BINS], fwd_i[N_BINS];
            fft_q15_forward(fft_in, fwd_r, fwd_i);

            /* 2c. Q15 → float → UL-UNAS infer → Q20 CRM */
            float spec_r[N_BINS], spec_i[N_BINS];
            for (int j = 0; j < N_BINS; j++) {
                spec_r[j] = (float)fwd_r[j] / 32768.0f;
                spec_i[j] = (float)fwd_i[j] / 32768.0f;
            }
            int32_t crm[2 * N_BINS];
            ulunas_infer_frame(spec_r, spec_i, &g_state, crm);

            /* 2e. CRM Q20 → Q15 IFFT input: >>5 */
            int32_t inv_r[N_BINS], inv_i[N_BINS];
            for (int i = 0; i < N_BINS; i++) {
                inv_r[i] = (crm[i] + 16) >> 5;
                inv_i[i] = (crm[N_BINS + i] + 16) >> 5;
            }

            /* 2f. Q15 Inverse FFT */
            int32_t ifft_out[WIN_LEN];
            fft_q15_inverse(inv_r, inv_i, ifft_out);

            /* 2g. Synthesis: ifft_out × window / N → Q15
             * fft_q15_inverse has NO 1/N (matches KissFFT) but numpy IFFT DOES.
             * >>9 compensates /N=512; >>15 for Q15×Q15→Q15; total >>24.
             * Also accumulate window² for WOLA normalization. */
            for (int i = 0; i < WIN_LEN; i++) {
                int32_t s = (int32_t)(((int64_t)ifft_out[i] * stft_win_q15[i] + 8388608) >> 24);
                int pos = (g_ola_pos + i) % (WIN_LEN + WIN_INC);
                g_ola[pos] += s;
                /* Window² Q15: win² >>15, scaled to match OLA magnitude */
                int32_t w2 = (int32_t)(((int64_t)stft_win_q15[i] * stft_win_q15[i] + 16384) >> 15);
                g_wola_sum[pos] += w2;
            }

            /* 2h. OLA output with WOLA normalization → clamp Q15 → output FIFO */
            for (int i = 0; i < WIN_INC; i++) {
                int32_t v = g_ola[g_ola_pos];
                int32_t ws = g_wola_sum[g_ola_pos];
                g_ola[g_ola_pos] = 0;
                g_wola_sum[g_ola_pos] = 0;
                g_ola_pos = (g_ola_pos + 1) % (WIN_LEN + WIN_INC);
                /* WOLA normalization: v = v * 32768 / ws (Q15 division) */
                if (ws > 0) {
                    int64_t norm = ((int64_t)v * 32768) / ws;
                    v = (int32_t)norm;
                }
                if (v >  32767) v =  32767;
                if (v < -32768) v = -32768;
                g_out_fifo[(g_out_rpos + g_out_count) % FIFO_SZ] = (int16_t)v;
                g_out_count++;
            }
            g_frame_count++;
            read_wpos = (read_wpos + WIN_INC) % FIFO_SZ;
        }
    }

    /* ── 3. 输出 8kHz PCM ── */
    if (g_out_count >= FRAME_16K) {
        int32_t gain_q15 = 32768;
        if (g_frame_count < WARMUP_MUTE) gain_q15 = 0;
        else if (g_frame_count < WARMUP_MUTE + WARMUP_FADE)
            gain_q15 = ((g_frame_count - WARMUP_MUTE) * 32768) / WARMUP_FADE;

        for (int i = 0; i < FRAME_IN; i++) {
            int32_t a = ((int32_t)g_out_fifo[g_out_rpos] * gain_q15 + 16384) >> 15;
            g_out_rpos = (g_out_rpos + 1) % FIFO_SZ;
            int32_t b = ((int32_t)g_out_fifo[g_out_rpos] * gain_q15 + 16384) >> 15;
            g_out_rpos = (g_out_rpos + 1) % FIFO_SZ;
            int32_t out = (a + b) >> 1; /* downmix 16k→8k */
            if (out >  32767) out =  32767;
            if (out < -32768) out = -32768;
            voiceOut[i] = (int16_t)out;
        }
        g_out_count -= FRAME_16K;
    } else {
        memset(voiceOut, 0, FRAME_IN * sizeof(short));
    }
}
