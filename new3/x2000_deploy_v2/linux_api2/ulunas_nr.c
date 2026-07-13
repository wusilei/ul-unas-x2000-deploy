/**
 * ulunas_nr.c — X2000 全定点降噪: Q15 FFT + UL-UNAS + WOLA
 * ==========================================================
 * v6: WIN_INC=200 (25ms@8kHz), 每call 400=2×200 精确对齐, 3/2窗混合WOLA.
 *
 * 定点路径:
 *   PCM Q15 → stft_window → fft_q15_forward → Q15→Q20(<<5)
 *   → log_gen→BM→Enc→GDPRNN→Dec→Sigmoid→BS→MASK (全定点)
 *   → CRM Q20→Q15(>>5) → fft_q15_inverse → ×window >>24 → OLA
 *   → WOLA归一化 (×inv_table >>15) → 增益校准 → 输出 Q15
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
#define WIN_INC     200   /* 25ms@8kHz, 每call 400=2×200 精确对齐 */
#define N_BINS      257
#define FIFO_SZ     (WIN_LEN * 4)
#define WARMUP_MUTE 5
#define WARMUP_FADE 3

/* ──── MATLAB stft_window.mat quantized to Q15 ──── */
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

/* ──── WOLA 预计算倒数表 (Q30), WIN_INC=200 ────
 * 0..111: 3窗重叠 win²_sum[i]=win²[i]+win²[i+200]+win²[i+400] (Q15)
 * 112..199: 2窗重叠 win²_sum[i]=win²[i]+win²[i+200] (Q15)
 * wola_inv[i] = round(2^30 / win²_sum[i])
 * 归一化: v_out = (v_ola * wola_inv[ola_pos%%200] + 2^14) >> 15
 */
static const uint32_t wola_inv_q30[200] = {
     34574u,  34499u,  34424u,  34349u,  34273u,  34198u,  34126u,  34051u,  33977u,  33903u,  33832u,  33760u,  33691u,  33621u,  33549u,  33481u,
     33415u,  33348u,  33282u,  33218u,  33155u,  33093u,  33033u,  32973u,  32915u,  32858u,  32804u,  32749u,  32696u,  32645u,  32598u,  32550u,
     32503u,  32460u,  32416u,  32377u,  32337u,  32299u,  32265u,  32231u,  32199u,  32169u,  32140u,  32114u,  32090u,  32068u,  32046u,  32028u,
     32012u,  31996u,  31983u,  31973u,  31964u,  31957u,  31952u,  31949u,  31947u,  31949u,  31952u,  31957u,  31964u,  31973u,  31983u,  31996u,
     32012u,  32028u,  32046u,  32068u,  32090u,  32114u,  32140u,  32169u,  32199u,  32231u,  32265u,  32299u,  32337u,  32377u,  32416u,  32460u,
     32503u,  32550u,  32598u,  32645u,  32696u,  32749u,  32804u,  32858u,  32915u,  32973u,  33033u,  33093u,  33155u,  33218u,  33282u,  33348u,
     33415u,  33481u,  33549u,  33621u,  33691u,  33760u,  33832u,  33903u,  33977u,  34051u,  34126u,  34198u,  34273u,  34349u,  34424u,  34499u,
     34574u,  34648u,  34723u,  34798u,  34872u,  34946u,  35020u,  35091u,  35164u,  35236u,  35305u,  35375u,  35444u,  35512u,  35578u,  35642u,
     35705u,  35766u,  35828u,  35886u,  35946u,  36001u,  36055u,  36107u,  36157u,  36205u,  36251u,  36296u,  36338u,  36377u,  36414u,  36449u,
     36480u,  36509u,  36538u,  36563u,  36587u,  36606u,  36623u,  36638u,  36650u,  36660u,  36667u,  36670u,  36672u,  36670u,  36667u,  36660u,
     36650u,  36638u,  36623u,  36606u,  36587u,  36563u,  36538u,  36509u,  36480u,  36449u,  36414u,  36377u,  36338u,  36296u,  36251u,  36205u,
     36157u,  36107u,  36055u,  36001u,  35946u,  35886u,  35828u,  35766u,  35705u,  35642u,  35578u,  35512u,  35444u,  35375u,  35305u,  35236u,
     35164u,  35091u,  35020u,  34946u,  34872u,  34798u,  34723u,  34648u
};

/* 全局输出增益 (Q15, 1.0=32768). 校准时调整此值使 RMS 匹配 Python 参考. */
#define OUTPUT_GAIN_Q15  32768

static ulunas_state_t g_state;
static int16_t g_fifo[FIFO_SZ];
static int     g_fifo_wpos, g_fifo_count;
static int32_t g_ola[WIN_LEN + WIN_INC];
static int     g_ola_pos;
static int16_t g_out_fifo[FIFO_SZ];
static int     g_out_rpos, g_out_count;
static int     g_frame_count;

void noise_init(void) {
    ulunas_state_init(&g_state);
    g_fifo_wpos = g_fifo_count = 0; g_ola_pos = 0;
    g_out_rpos = g_out_count = 0; g_frame_count = 0;
    memset(g_fifo, 0, sizeof(g_fifo));
    memset(g_ola, 0, sizeof(g_ola));
    memset(g_out_fifo, 0, sizeof(g_out_fifo));
}

void noise_deinit(void) { }

void noise_reduction(short *voiceIn, short *voiceOut, int n_samples) {
    /* ── 1. 8kHz PCM 直入 FIFO (无需上采样, ERB映射与采样率无关) ── */
    for (int i = 0; i < n_samples; i++) {
        g_fifo[g_fifo_wpos] = voiceIn[i];
        g_fifo_wpos = (g_fifo_wpos + 1) % FIFO_SZ;
        g_fifo_count++;
    }

    /* ── 2. STFT→DENOISE→ISTFT (全定点, 零 float, 零除法) ── */
    {
        int read_wpos = (g_fifo_wpos - g_fifo_count + WIN_LEN + FIFO_SZ) % FIFO_SZ;
        while (g_fifo_count >= WIN_LEN) {
            g_fifo_count -= WIN_INC;

            /* 2a. Analysis: PCM Q15 × stft_window Q15 → Q15 */
            int32_t fft_in[WIN_LEN];
            int start = (read_wpos - WIN_LEN + FIFO_SZ) % FIFO_SZ;
            for (int i = 0; i < WIN_LEN; i++) {
                int32_t v = g_fifo[(start + i) % FIFO_SZ];
                fft_in[i] = (int32_t)(((int64_t)v * stft_win_q15[i] + 16384) >> 15);
            }

            /* 2b. Q15 Forward FFT → Q15 spectrum */
            int32_t fwd_r[N_BINS], fwd_i[N_BINS];
            fft_q15_forward(fft_in, fwd_r, fwd_i);

            /* 2c. Q15→Q20 直接移位 (跳过 float! Q15<<5 = Q20)
             *     ulunas_infer_frame 内部: real_q20 = round(float * 2^20)
             *     = round(Q15/32768 * 1048576) = round(Q15 * 32) = Q15 << 5
             */
            int32_t real_q20[N_BINS], imag_q20[N_BINS];
            for (int j = 0; j < N_BINS; j++) {
                real_q20[j] = fwd_r[j] << 5;
                imag_q20[j] = fwd_i[j] << 5;
            }

            /* 2d. Full UL-UNAS inference (all fixed-point) */
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

            int32_t crm[2 * N_BINS];
            mask_fixed(y_bs, real_q20, imag_q20, N_BINS, crm);

            /* 2e. CRM Q20 → Q15 IFFT input */
            int32_t inv_r[N_BINS], inv_i[N_BINS];
            for (int i = 0; i < N_BINS; i++) {
                inv_r[i] = (crm[i] + 16) >> 5;
                inv_i[i] = (crm[N_BINS + i] + 16) >> 5;
            }

            /* 2f. Q15 Inverse FFT (no 1/N — matches KissFFT convention) */
            int32_t ifft_out[WIN_LEN];
            fft_q15_inverse(inv_r, inv_i, ifft_out);

            /* 2g. Synthesis + OLA
             *     ifft_out (Q15×N, no 1/N) × window (Q15) >> 24 → Q15
             *     >>9 补偿 numpy IFFT 的 1/N=1/512
             *     >>15 补偿 Q15×Q15→Q15
             */
            for (int i = 0; i < WIN_LEN; i++) {
                int32_t s = (int32_t)(((int64_t)ifft_out[i] * stft_win_q15[i] + 8388608) >> 24);
                int pos = (g_ola_pos + i) % (WIN_LEN + WIN_INC);
                g_ola[pos] += s;
            }

            /* 2h. WOLA 归一化输出 (预计算倒数表, 无除法!)
             *     v_out = (v_ola * wola_inv[pos%%256] + 2^14) >> 15
             *     wola_inv[i] = 2^30 / (win²[i] + win²[i+256]), Q30
             *     结果自动在 Q15.
             */
            for (int i = 0; i < WIN_INC; i++) {
                int32_t v = g_ola[g_ola_pos];
                g_ola[g_ola_pos] = 0;
                int idx = g_ola_pos % WIN_INC;
                g_ola_pos = (g_ola_pos + 1) % (WIN_LEN + WIN_INC);

                /* WOLA normalization: multiply by precomputed inverse */
                int64_t norm = ((int64_t)v * (int64_t)wola_inv_q30[idx] + 16384) >> 15;
                v = (int32_t)norm;

                /* Global gain calibration (adjust OUTPUT_GAIN_Q15 to match Python RMS) */
                v = (int32_t)(((int64_t)v * OUTPUT_GAIN_Q15 + 16384) >> 15);

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
    if (g_out_count >= n_samples) {
        int32_t gain_q15 = 32768;
        if (g_frame_count < WARMUP_MUTE) gain_q15 = 0;
        else if (g_frame_count < WARMUP_MUTE + WARMUP_FADE)
            gain_q15 = ((g_frame_count - WARMUP_MUTE) * 32768) / WARMUP_FADE;

        for (int i = 0; i < n_samples; i++) {
            int32_t out = ((int32_t)g_out_fifo[g_out_rpos] * gain_q15 + 16384) >> 15;
            g_out_rpos = (g_out_rpos + 1) % FIFO_SZ;
            if (out >  32767) out =  32767;
            if (out < -32768) out = -32768;
            voiceOut[i] = (int16_t)out;
        }
        g_out_count -= n_samples;
    } else {
        /* 缓冲不足 (启动期): 输出已有数据 + 静音填充 */
        int available = g_out_count;
        int32_t gain_q15 = 32768;
        if (g_frame_count < WARMUP_MUTE) gain_q15 = 0;
        else if (g_frame_count < WARMUP_MUTE + WARMUP_FADE)
            gain_q15 = ((g_frame_count - WARMUP_MUTE) * 32768) / WARMUP_FADE;

        for (int i = 0; i < available; i++) {
            int32_t out = ((int32_t)g_out_fifo[g_out_rpos] * gain_q15 + 16384) >> 15;
            g_out_rpos = (g_out_rpos + 1) % FIFO_SZ;
            if (out >  32767) out =  32767;
            if (out < -32768) out = -32768;
            voiceOut[i] = (int16_t)out;
        }
        for (int i = available; i < n_samples; i++)
            voiceOut[i] = 0;
        g_out_count = 0;
    }
}
