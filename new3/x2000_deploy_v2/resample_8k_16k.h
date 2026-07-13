/**
 * resample_8k_16k.h — Streaming 8kHz↔16kHz Half-Band Resampler
 * =============================================================
 * Design:
 *   - Independent FIFO, decoupled from model frame processing
 *   - Upsampler: accumulates 8kHz input → outputs 16kHz when ready
 *   - Downsampler: accumulates 16kHz model output → outputs 8kHz when ready
 *
 * Half-band filter: 15-tap, 7 non-zero taps, ~60dB stopband attenuation
 *   h[0]=0.5, h[±1]=0.2937, h[±3]=-0.0506, h[±5]=0.0193, h[±7]=-0.0057
 */

#ifndef RESAMPLE_8K_16K_H
#define RESAMPLE_8K_16K_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* Half-band coefficients (Q15) */
static const int16_t hb_coeff[4] = { -187, 632, -1658, 9624 };
/*                                 h[±7] h[±5] h[±3] h[±1] */

/* ──── Upsampler State ──── */
typedef struct {
    int16_t hist[8];    /* last 8 input samples for filter continuity */
    int     hist_len;
    int16_t out_buf[1024]; /* 16kHz output buffer */
    int     out_count;
} Upsampler8k16k;

static void upsampler_init(Upsampler8k16k *u) {
    memset(u, 0, sizeof(*u));
}

/* Feed N 8kHz samples. Call upsampler_pop() to retrieve 16kHz output. */
static void upsampler_feed(Upsampler8k16k *u, const int16_t *in8k, int n) {
    /* Build padded buffer: [history | new_input | 4 zeros for tail] */
    int total = u->hist_len + n + 4;
    int16_t *buf = (int16_t *)malloc(total * sizeof(int16_t));
    memcpy(buf, u->hist, u->hist_len * sizeof(int16_t));
    memcpy(buf + u->hist_len, in8k, n * sizeof(int16_t));
    memset(buf + u->hist_len + n, 0, 4 * sizeof(int16_t));

    /* Process each input sample to produce 2 output samples */
    for (int i = u->hist_len; i < u->hist_len + n; i++) {
        /* Phase 0 (even): y[2n] = x[n] / 2 */
        u->out_buf[u->out_count++] = (int16_t)(buf[i] >> 1);

        /* Phase 1 (odd): interpolated from neighbors */
        int64_t acc = 0;
        acc += (int64_t)hb_coeff[3] * (int32_t)(buf[i] + buf[i+1]);     /* k=±1 */
        acc += (int64_t)hb_coeff[2] * (int32_t)(buf[i-1] + buf[i+2]);   /* k=±3 */
        acc += (int64_t)hb_coeff[1] * (int32_t)(buf[i-2] + buf[i+3]);   /* k=±5 */
        acc += (int64_t)hb_coeff[0] * (int32_t)(buf[i-3] + buf[i+4]);   /* k=±7 */

        int32_t v = (int32_t)((acc + 16384) >> 15);
        if (v > 32767) v = 32767; if (v < -32768) v = -32768;
        u->out_buf[u->out_count++] = (int16_t)v;
    }

    /* Save last 8 samples for next call */
    int save = (n < 8) ? n : 8;
    memcpy(u->hist, in8k + n - save, save * sizeof(int16_t));
    u->hist_len = save;

    free(buf);
}

/* Pop available 16kHz samples. Returns number of samples copied. */
static int upsampler_pop(Upsampler8k16k *u, int16_t *out16k, int max_out) {
    int n = (u->out_count < max_out) ? u->out_count : max_out;
    memcpy(out16k, u->out_buf, n * sizeof(int16_t));
    u->out_count -= n;
    if (u->out_count > 0)
        memmove(u->out_buf, u->out_buf + n, u->out_count * sizeof(int16_t));
    return n;
}

/* ──── Downsampler State ──── */
typedef struct {
    int16_t hist[16];   /* filter history (16kHz samples) */
    int     hist_len;
    int16_t out_buf[512]; /* 8kHz output buffer */
    int     out_count;
} Downsampler16k8k;

static void downsampler_init(Downsampler16k8k *d) {
    memset(d, 0, sizeof(*d));
}

/* Feed N 16kHz samples. Call downsampler_pop() to get 8kHz output. */
static void downsampler_feed(Downsampler16k8k *d, const int16_t *in16k, int n) {
    int total = d->hist_len + n;
    int16_t *buf = (int16_t *)malloc(total * sizeof(int16_t));
    memcpy(buf, d->hist, d->hist_len * sizeof(int16_t));
    memcpy(buf + d->hist_len, in16k, n * sizeof(int16_t));

    /* Half-band filter at each even-indexed sample, decimate by 2 */
    for (int m = 0; 2*m + 7 < total; m++) {
        int idx = 2 * m;
        int64_t acc = (int64_t)buf[idx] * 16384LL;                        /* h[0]=0.5 */
        acc += (int64_t)hb_coeff[3] * (int32_t)(buf[idx-1] + buf[idx+1]); /* k=±1 */
        acc += (int64_t)hb_coeff[2] * (int32_t)(buf[idx-3] + buf[idx+3]); /* k=±3 */
        acc += (int64_t)hb_coeff[1] * (int32_t)(buf[idx-5] + buf[idx+5]); /* k=±5 */
        acc += (int64_t)hb_coeff[0] * (int32_t)(buf[idx-7] + buf[idx+7]); /* k=±7 */

        int32_t v = (int32_t)((acc + 16384) >> 15);
        if (v > 32767) v = 32767; if (v < -32768) v = -32768;
        d->out_buf[d->out_count++] = (int16_t)v;
    }

    /* Save tail samples for next call */
    int consumed = d->out_count * 2; /* 2 16kHz samples per 8kHz output */
    int remaining = total - consumed;
    if (remaining > 16) remaining = 16;
    if (remaining > 0) {
        memcpy(d->hist, buf + total - remaining, remaining * sizeof(int16_t));
        d->hist_len = remaining;
    } else {
        d->hist_len = 0;
    }

    free(buf);
}

/* Pop available 8kHz samples */
static int downsampler_pop(Downsampler16k8k *d, int16_t *out8k, int max_out) {
    int n = (d->out_count < max_out) ? d->out_count : max_out;
    memcpy(out8k, d->out_buf, n * sizeof(int16_t));
    d->out_count -= n;
    if (d->out_count > 0)
        memmove(d->out_buf, d->out_buf + n, d->out_count * sizeof(int16_t));
    return n;
}

#endif /* RESAMPLE_8K_16K_H */
