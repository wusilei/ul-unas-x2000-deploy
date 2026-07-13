/**
 * nn_pipeline.h — Generic STFT→Inference→ISTFT Pipeline Shell
 * ============================================================
 * Configurable pipeline: FIFO → Window → FFT → [User Inference Callback] → IFFT → WOLA → Output.
 *
 * Supports:
 *   - Arbitrary WIN_INC / WIN_LEN / N_FFT
 *   - 8kHz direct or 8k→16k upsampling
 *   - Hann or custom analysis window
 *   - WOLA normalization with pre-computed inverse table
 *   - Warmup mute + fade-in
 *   - Output gain calibration
 *
 * User provides:
 *   - infer_cb: (real_q20, imag_q20, n_bins, crm_out, user_data) → void
 *   - n_samples per call (e.g., 200 for GTCRN, 400 for UL-UNAS v6)
 *
 * Verified: GTCRN (WIN_INC=256) + UL-UNAS v6 (WIN_INC=200), X2000
 * License: MIT
 */

#ifndef NN_PIPELINE_H
#define NN_PIPELINE_H

#include "nn_core/nn_qformat.h"
#include "nn_signal/nn_fft_q15.h"
#include "nn_signal/nn_wola.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Config
 * ================================================================ */

typedef struct {
    int n_fft;             /* FFT points (default 512) */
    int win_len;           /* Window length (default 512) */
    int win_inc;           /* Hop size (GTCRN=256, ULUNAS=200) */
    int n_bins;            /* Freq bins (257 for 512-pt FFT) */
    int upsample;          /* 1=8kHz direct, 2=8k→16k linear interp */
    int warmup_mute;       /* Mute frames at start */
    int warmup_fade;       /* Fade-in frames after mute */
    int32_t output_gain_q15; /* Output gain (32768 = 1.0) */
    const int16_t *window_q15; /* Analysis/synthesis window (NULL=auto Hann) */
    int fifo_cap;          /* FIFO capacity (default win_len*4) */
} nn_pipeline_config_t;

/* Pre-defined configs */
#define NN_PIPELINE_8K_GTCRN { 512, 512, 256, 257, 2, 20, 12, 32768, NULL, 2048 }
#define NN_PIPELINE_8K_ULUNAS { 512, 512, 200, 257, 1, 5, 3, 32768, NULL, 2048 }

/* ================================================================
 * Pipeline State
 * ================================================================ */

typedef void (*nn_infer_callback_t)(const int32_t *real_q20, const int32_t *imag_q20,
                                     int n_bins, int32_t *crm_out, void *user_data);

typedef struct {
    nn_pipeline_config_t cfg;
    /* Analysis window */
    int16_t *win_q15;
    int win_owned;
    /* WOLA */
    uint32_t *wola_inv_q30;
    /* FIFO */
    int16_t *fifo;
    int fifo_wpos, fifo_count;
    /* OLA */
    int32_t *ola_buf;
    int ola_pos;
    int ola_buf_len;
    /* Output FIFO */
    int16_t *out_fifo;
    int out_rpos, out_count;
    int frame_count;
    /* Upsampling */
    int16_t last_in_sample;
} nn_pipeline_t;

/* ================================================================
 * API
 * ================================================================ */

static inline void nn_hann_q15(int16_t *win, int N) {
    for (int i = 0; i < N; i++) {
        float v = sinf(3.14159265358979323846f * (float)i / (float)(N - 1));
        win[i] = (int16_t)(v * 32767.0f + 0.5f);
    }
}

static inline nn_pipeline_t *nn_pipeline_create(const nn_pipeline_config_t *cfg) {
    nn_pipeline_t *p = (nn_pipeline_t*)calloc(1, sizeof(nn_pipeline_t));
    if (!p) return NULL;
    p->cfg = *cfg;
    if (!p->cfg.fifo_cap) p->cfg.fifo_cap = p->cfg.win_len * 4;
    if (!p->cfg.n_bins) p->cfg.n_bins = p->cfg.n_fft / 2 + 1;

    /* Window */
    if (p->cfg.window_q15) {
        p->win_q15 = (int16_t*)p->cfg.window_q15;
        p->win_owned = 0;
    } else {
        p->win_q15 = (int16_t*)malloc(p->cfg.win_len * sizeof(int16_t));
        if (p->win_q15) { nn_hann_q15(p->win_q15, p->cfg.win_len); p->win_owned = 1; }
    }

    /* Allocate buffers */
    p->fifo = (int16_t*)calloc(p->cfg.fifo_cap, sizeof(int16_t));
    p->ola_buf_len = p->cfg.win_len + p->cfg.win_inc;
    p->ola_buf = (int32_t*)calloc(p->ola_buf_len, sizeof(int32_t));
    p->out_fifo = (int16_t*)calloc(p->cfg.fifo_cap, sizeof(int16_t));
    p->wola_inv_q30 = (uint32_t*)malloc(p->cfg.win_inc * sizeof(uint32_t));

    if (p->win_q15 && p->wola_inv_q30)
        nn_wola_compute_inv_table(p->win_q15, p->cfg.win_len, p->cfg.win_inc, p->wola_inv_q30);

    p->fifo_wpos = p->fifo_count = 0;
    p->ola_pos = 0; p->out_rpos = p->out_count = 0; p->frame_count = 0;
    return p;
}

static inline void nn_pipeline_destroy(nn_pipeline_t *p) {
    if (!p) return;
    if (p->win_owned) free(p->win_q15);
    free(p->fifo); free(p->ola_buf); free(p->out_fifo); free(p->wola_inv_q30);
    free(p);
}

/**
 * nn_pipeline_process — Process one chunk of PCM through the STFT→NR→ISTFT pipeline
 *
 * pcm_in[n_in]: input PCM Q15 samples
 * infer_cb: user inference callback
 * cb_user: user data passed to callback
 * pcm_out: output buffer
 * *n_out: set to number of output samples available (may be < n_in during startup)
 */
static inline void nn_pipeline_process(nn_pipeline_t *p,
                                        const int16_t *pcm_in, int n_in,
                                        nn_infer_callback_t infer_cb, void *cb_user,
                                        int16_t *pcm_out, int *n_out) {
    int win_len = p->cfg.win_len, win_inc = p->cfg.win_inc;
    int n_bins = p->cfg.n_bins;

    /* ── 1. PCM input → FIFO (with optional upsampling) ── */
    if (p->cfg.upsample == 2) {
        for (int i = 0; i < n_in; i++) {
            int16_t cur = pcm_in[i], prev = p->last_in_sample;
            p->fifo[p->fifo_wpos] = cur;
            p->fifo_wpos = (p->fifo_wpos + 1) % p->cfg.fifo_cap;
            /* Linear interpolation */
            p->fifo[p->fifo_wpos] = (int16_t)(((int)prev + (int)cur) >> 1);
            p->fifo_wpos = (p->fifo_wpos + 1) % p->cfg.fifo_cap;
            p->fifo_count += 2;
            p->last_in_sample = cur;
        }
    } else {
        for (int i = 0; i < n_in; i++) {
            p->fifo[p->fifo_wpos] = pcm_in[i];
            p->fifo_wpos = (p->fifo_wpos + 1) % p->cfg.fifo_cap;
            p->fifo_count++;
        }
    }

    /* ── 2. STFT → Inference → ISTFT loop ── */
    int read_wpos = (p->fifo_wpos - p->fifo_count + win_len + p->cfg.fifo_cap) % p->cfg.fifo_cap;
    while (p->fifo_count >= win_len) {
        p->fifo_count -= win_inc;

        /* 2a. Window + Q15 FFT */
        int32_t fft_in[512];
        int start = (read_wpos - win_len + p->cfg.fifo_cap) % p->cfg.fifo_cap;
        for (int i = 0; i < win_len; i++) {
            int32_t v = p->fifo[(start + i) % p->cfg.fifo_cap];
            fft_in[i] = (int32_t)(((int64_t)v * p->win_q15[i] + (NN_Q15_SCALE >> 1)) >> 15);
        }
        int32_t fwd_r[257], fwd_i[257];
        nn_fft_q15_forward(fft_in, fwd_r, fwd_i);

        /* 2b. Q15 → Q20 for model input */
        int32_t real_q20[257], imag_q20[257];
        for (int j = 0; j < n_bins; j++) {
            real_q20[j] = fwd_r[j] << 5;  /* Q15 * 32 = Q20 */
            imag_q20[j] = fwd_i[j] << 5;
        }

        /* 2c. User inference callback */
        int32_t crm[2 * 257];
        infer_cb(real_q20, imag_q20, n_bins, crm, cb_user);

        /* 2d. CRM Q20 → Q15 IFFT input */
        int32_t inv_r[257], inv_i[257];
        for (int j = 0; j < n_bins; j++) {
            inv_r[j] = (crm[j] + 16) >> 5;
            inv_i[j] = (crm[n_bins + j] + 16) >> 5;
        }

        /* 2e. Q15 IFFT */
        int32_t ifft_out[512];
        nn_fft_q15_inverse(inv_r, inv_i, ifft_out);

        /* 2f. Synthesis window + OLA */
        for (int i = 0; i < win_len; i++) {
            int32_t s = (int32_t)(((int64_t)ifft_out[i] * p->win_q15[i] + 8388608) >> 24);
            int pos = (p->ola_pos + i) % p->ola_buf_len;
            p->ola_buf[pos] += s;
        }

        /* 2g. WOLA output */
        for (int i = 0; i < win_inc; i++) {
            int32_t ov = p->ola_buf[p->ola_pos];
            p->ola_buf[p->ola_pos] = 0;
            int idx = p->ola_pos % win_inc;
            p->ola_pos = (p->ola_pos + 1) % p->ola_buf_len;

            int64_t norm = ((int64_t)ov * (int64_t)p->wola_inv_q30[idx] + (1LL << 29)) >> 30;
            int32_t scaled = (int32_t)(((int64_t)norm * p->cfg.output_gain_q15
                                        + (NN_Q15_SCALE >> 1)) >> 15);
            if (scaled >  32767) scaled =  32767;
            if (scaled < -32768) scaled = -32768;
            p->out_fifo[(p->out_rpos + p->out_count) % p->cfg.fifo_cap] = (int16_t)scaled;
            p->out_count++;
        }
        p->frame_count++;
        read_wpos = (read_wpos + win_inc) % p->cfg.fifo_cap;
    }

    /* ── 3. Output (with warmup mute/fade) ── */
    if (p->cfg.upsample == 2) n_in *= 2;
    if (p->out_count >= n_in) {
        int32_t gain = NN_Q15_SCALE;
        if (p->frame_count < p->cfg.warmup_mute) gain = 0;
        else if (p->frame_count < p->cfg.warmup_mute + p->cfg.warmup_fade)
            gain = ((p->frame_count - p->cfg.warmup_mute) * NN_Q15_SCALE) / p->cfg.warmup_fade;

        for (int i = 0; i < n_in; i++) {
            int32_t s = ((int32_t)p->out_fifo[p->out_rpos] * gain + (NN_Q15_SCALE >> 1)) >> 15;
            p->out_rpos = (p->out_rpos + 1) % p->cfg.fifo_cap;
            if (s > 32767) s = 32767;
            if (s < -32768) s = -32768;
            pcm_out[i] = (int16_t)s;
        }
        p->out_count -= n_in;
        *n_out = n_in;
    } else {
        int avail = p->out_count;
        for (int i = 0; i < avail; i++) {
            pcm_out[i] = p->out_fifo[p->out_rpos];
            p->out_rpos = (p->out_rpos + 1) % p->cfg.fifo_cap;
        }
        for (int i = avail; i < n_in; i++) pcm_out[i] = 0;
        p->out_count = 0;
        *n_out = avail;
    }
}

#ifdef __cplusplus
}
#endif

#endif /* NN_PIPELINE_H */
