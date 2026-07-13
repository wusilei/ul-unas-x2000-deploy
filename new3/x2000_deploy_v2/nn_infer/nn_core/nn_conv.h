/**
 * nn_conv.h — Convolution Operators (Fixed-Point)
 * ================================================
 * 9 convolution variants extracted and unified from GTCRN + UL-UNAS:
 *
 *   nn_conv2d      — Standard 2D conv (GTCRN + ULUNAS)
 *   nn_tconv2d     — Transposed 2D conv (GTCRN + ULUNAS)
 *   nn_pconv2d     — Point-wise 1×1 conv (GTCRN + ULUNAS)
 *   nn_gconv2d     — Grouped temporal conv with cache (ULUNAS)
 *   nn_gtconv2d    — Grouped temporal transposed conv with cache (ULUNAS)
 *   nn_nongconv2d  — Grouped non-temporal freq-dim conv (ULUNAS)
 *   nn_nongtconv2d — Grouped non-temporal freq-dim transposed conv (ULUNAS)
 *   nn_ddconv2d    — Depth-wise dilated conv with history (GTCRN)
 *   nn_ddtconv2d   — Depth-wise dilated transposed conv with history (GTCRN)
 *
 * All convs:
 *   - Input x: int32_t Q20
 *   - Weight:   int16_t (Q specified by caller via qr parameter)
 *   - Bias:     int32_t Q20
 *   - Output y: int32_t Q20
 *   - Accumulate in int64, round_shr at end, saturate to int32
 *
 * Data layout (row-major):
 *   2D (C,W):  element[c][w] = data[c*W + w]
 *   Weight 4D (Cout,Cin,Kh,Kw):
 *     weight[cout][cin][kh][kw] = weight[cout*Cin*Kh*Kw + cin*Kh*Kw + kh*Kw + kw]
 *
 * ⚠️ CHECKLIST (from 18 bugs across 2 projects):
 *   1. Weight layout MUST match MATLAB col-major → C row-major conversion
 *   2. Accumulate in int64 to avoid overflow (Q20 × Q14 = Q34)
 *   3. round_shr after FULL accumulation, not per-element
 *   4. Group conv: weight_stride = Cout_per_group * Cin_per_group * Kh * Kw
 *   5. Cache for gconv/gtconv: updated in-place, caller allocates zero-init buffer
 *
 * Verified: GTCRN (denoise_v19_q15) + UL-UNAS (linux_api9), X2000 MIPS32R2
 * License: MIT
 */

#ifndef NN_CONV_H
#define NN_CONV_H

#include "nn_qformat.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * nn_conv2d — Standard 2D Convolution
 * ================================================================
 * x[Cin][Win], weight[Cout][Cin][Kh][Kw], bias[Cout]
 * y[Cout][Wout] in Q20
 *
 * Matching MATLAB:
 *   temp = round(x_kernel .* kernel_chan * 2^Qr);
 *   conv_result = round(sum(temp, 'all') + bias);
 */
static inline void nn_conv2d(const int32_t *x, int Cin, int Win,
                              int Cout, int Wout, int Kh, int Kw,
                              int stride_w, int pad_w,
                              const int16_t *weight, const int32_t *bias,
                              int qr, int32_t *y) {
    for (int co = 0; co < Cout; co++) {
        int32_t b = bias ? bias[co] : 0;
        for (int wo = 0; wo < Wout; wo++) {
            int64_t sum = 0;
            for (int ci = 0; ci < Cin; ci++) {
                for (int kh = 0; kh < Kh; kh++) {
                    for (int kw = 0; kw < Kw; kw++) {
                        int wi = wo * stride_w + kw - pad_w;
                        if (wi >= 0 && wi < Win) {
                            int32_t xv = x[ci * Win + wi];
                            int16_t wv = weight[NN_IDX_W4D(co, ci, kh, kw, Cin, Kh, Kw)];
                            sum += (int64_t)xv * wv;
                        }
                    }
                }
            }
            y[co * Wout + wo] = nn_sat_i32(nn_round_shr(sum, -qr) + b);
        }
    }
}

/* ================================================================
 * nn_tconv2d — Transposed 2D Convolution
 * ================================================================
 * x[Cin][Win], weight[Cin][Cout][Kh][Kw], bias[Cout]
 * Zero-insertion + rot180(kernel).
 * y[Cout][Wout] in Q20
 */
static inline void nn_tconv2d(const int32_t *x, int Cin, int Win,
                               int Cout, int Wout, int Kh, int Kw,
                               int stride_w, int pad_w,
                               const int16_t *weight, const int32_t *bias,
                               int qr, int32_t *y) {
    for (int co = 0; co < Cout; co++) {
        int32_t b = bias ? bias[co] : 0;
        for (int wo = 0; wo < Wout; wo++) {
            int64_t sum = 0;
            for (int ci = 0; ci < Cin; ci++) {
                for (int kh = 0; kh < Kh; kh++) {
                    for (int kw = 0; kw < Kw; kw++) {
                        int wi = (wo + pad_w - kw) / stride_w;
                        if (wi >= 0 && wi < Win && (wo + pad_w - kw) % stride_w == 0) {
                            int32_t xv = x[ci * Win + wi];
                            /* rot180: weight[ci][co][Kh-1-kh][Kw-1-kw] */
                            int wh = Kh - 1 - kh;
                            int ww = Kw - 1 - kw;
                            int16_t wv = weight[NN_IDX_W4D(ci, co, wh, ww, Cout, Kh, Kw)];
                            sum += (int64_t)xv * wv;
                        }
                    }
                }
            }
            y[co * Wout + wo] = nn_sat_i32(nn_round_shr(sum, -qr) + b);
        }
    }
}

/* ================================================================
 * nn_pconv2d — Point-wise 1×1 Convolution
 * ================================================================
 * x[Cin][W], weight[Cout][Cin][1][1], bias[Cout]
 * y[Cout][W] in Q20
 *
 * weight_stride: stride between output channel weight blocks.
 *   When groups > 1, weight_stride = Cout_per_group * Cin_per_group.
 *   When groups == 1, weight_stride = Cout * Cin.
 *   Caller should pass 0 for default (Cout * Cin).
 */
static inline void nn_pconv2d(const int32_t *x, int Cin, int Cout,
                               int W, const int16_t *weight,
                               const int32_t *bias, int qr,
                               int weight_stride, int32_t *y) {
    int wstride = weight_stride ? weight_stride : Cin;
    for (int co = 0; co < Cout; co++) {
        int32_t b = bias ? bias[co] : 0;
        for (int w = 0; w < W; w++) {
            int64_t sum = 0;
            for (int ci = 0; ci < Cin; ci++) {
                sum += (int64_t)x[ci * W + w] * weight[co * wstride + ci];
            }
            y[co * W + w] = nn_sat_i32(nn_round_shr(sum, -qr) + b);
        }
    }
}

/* ================================================================
 * nn_gconv2d — Grouped Temporal Convolution (with cache)
 * ================================================================
 * Input = [cache; x] concatenated along time dimension.
 * weight[Cout][1][Kh][Kw] — group-wise (each output channel has own kernel).
 * Updates cache = x for next frame.
 *
 * x[Cout][Wx], cache[Cout][Wcache]
 * y[Cout][Wout] in Q20
 */
static inline void nn_gconv2d(const int32_t *x, int Cout, int Wout,
                               int Wx, int Wcache,
                               int Kh, int Kw, int stride_w,
                               const int16_t *weight, const int32_t *bias,
                               int qr, int32_t *cache, int32_t *y) {
    for (int co = 0; co < Cout; co++) {
        int32_t b = bias ? bias[co] : 0;
        for (int wo = 0; wo < Wout; wo++) {
            int64_t sum = 0;
            for (int kh = 0; kh < Kh; kh++) {
                for (int kw = 0; kw < Kw; kw++) {
                    int wi = wo * stride_w + kw;
                    int32_t xv;
                    if (wi < Wcache) {
                        xv = cache[co * Wcache + wi];
                    } else {
                        int xi = wi - Wcache;
                        if (xi >= 0 && xi < Wx)
                            xv = x[co * Wx + xi];
                        else
                            xv = 0;
                    }
                    int16_t wv = weight[NN_IDX_W4D(co, 0, kh, kw, 1, Kh, Kw)];
                    sum += (int64_t)xv * wv;
                }
            }
            y[co * Wout + wo] = nn_sat_i32(nn_round_shr(sum, -qr) + b);
        }
        /* Update cache: shift old out, copy x in */
        if (Wcache > Wx) {
            memmove(&cache[co * Wcache], &cache[co * Wcache + Wx],
                    (Wcache - Wx) * sizeof(int32_t));
        }
        memcpy(&cache[co * Wcache + Wcache - Wx], &x[co * Wx], Wx * sizeof(int32_t));
    }
}

/* ================================================================
 * nn_gtconv2d — Grouped Temporal Transposed Convolution (with cache)
 * ================================================================
 * Transposed version of nn_gconv2d.
 * Zero-insertion + rot180(kernel) + [cache; x] input.
 */
static inline void nn_gtconv2d(const int32_t *x, int Cout, int Wout,
                                int Wx, int Wcache,
                                int Kh, int Kw, int stride_w,
                                const int16_t *weight, const int32_t *bias,
                                int qr, int32_t *cache, int32_t *y) {
    int total_Win = Wcache + Wx;
    for (int co = 0; co < Cout; co++) {
        int32_t b = bias ? bias[co] : 0;
        for (int wo = 0; wo < Wout; wo++) {
            int64_t sum = 0;
            for (int kh = 0; kh < Kh; kh++) {
                for (int kw = 0; kw < Kw; kw++) {
                    int wi = (wo - kw) / stride_w;
                    if (wi >= 0 && wi < total_Win && (wo - kw) % stride_w == 0) {
                        int32_t xv;
                        if (wi < Wcache)
                            xv = cache[co * Wcache + wi];
                        else
                            xv = x[co * Wx + (wi - Wcache)];
                        /* rot180 */
                        int wh = Kh - 1 - kh, ww = Kw - 1 - kw;
                        int16_t wv = weight[NN_IDX_W4D(co, 0, wh, ww, 1, Kh, Kw)];
                        sum += (int64_t)xv * wv;
                    }
                }
            }
            y[co * Wout + wo] = nn_sat_i32(nn_round_shr(sum, -qr) + b);
        }
        /* Update cache */
        if (Wcache > Wx) {
            memmove(&cache[co * Wcache], &cache[co * Wcache + Wx],
                    (Wcache - Wx) * sizeof(int32_t));
        }
        memcpy(&cache[co * Wcache + Wcache - Wx], &x[co * Wx], Wx * sizeof(int32_t));
    }
}

/* ================================================================
 * nn_nongconv2d — Grouped Non-Temporal Convolution
 * ================================================================
 * Like nn_gconv2d but kernel only in frequency dimension (no time cache).
 * x[Cout][Wx], weight[Cout][1][1][Kw]
 */
static inline void nn_nongconv2d(const int32_t *x, int Cout, int Wout,
                                  int Wx, int Kw, int stride_w,
                                  const int16_t *weight, const int32_t *bias,
                                  int qr, int32_t *y) {
    for (int co = 0; co < Cout; co++) {
        int32_t b = bias ? bias[co] : 0;
        for (int wo = 0; wo < Wout; wo++) {
            int64_t sum = 0;
            for (int kw = 0; kw < Kw; kw++) {
                int wi = wo * stride_w + kw;
                if (wi >= 0 && wi < Wx) {
                    sum += (int64_t)x[co * Wx + wi]
                         * weight[NN_IDX_W4D(co, 0, 0, kw, 1, 1, Kw)];
                }
            }
            y[co * Wout + wo] = nn_sat_i32(nn_round_shr(sum, -qr) + b);
        }
    }
}

/* ================================================================
 * nn_nongtconv2d — Grouped Non-Temporal Transposed Convolution
 * ================================================================
 * Transposed version of nn_nongconv2d.
 */
static inline void nn_nongtconv2d(const int32_t *x, int Cout, int Wout,
                                   int Wx, int Kw, int stride_w,
                                   const int16_t *weight, const int32_t *bias,
                                   int qr, int32_t *y) {
    for (int co = 0; co < Cout; co++) {
        int32_t b = bias ? bias[co] : 0;
        for (int wo = 0; wo < Wout; wo++) {
            int64_t sum = 0;
            for (int kw = 0; kw < Kw; kw++) {
                int wi = (wo - kw) / stride_w;
                if (wi >= 0 && wi < Wx && (wo - kw) % stride_w == 0) {
                    int ww = Kw - 1 - kw;  /* rot90 for 1D kernel */
                    sum += (int64_t)x[co * Wx + wi]
                         * weight[NN_IDX_W4D(co, 0, 0, ww, 1, 1, Kw)];
                }
            }
            y[co * Wout + wo] = nn_sat_i32(nn_round_shr(sum, -qr) + b);
        }
    }
}

/* ================================================================
 * nn_ddconv2d — Depth-wise Dilated Convolution (GTCRN)
 * ================================================================
 * x[C][H][W] 3D input with history buffer.
 * weight[C][1][Hk][Wk] per-channel kernel.
 * dilation: temporal dilation factor.
 * hist: history buffer [C * hist_time * W], updated in-place.
 */
static inline void nn_ddconv2d(const int32_t *x, int C, int H, int W,
                                int Wout, int Hk, int Wk,
                                int pad_h, int pad_w, int dilation, int qr,
                                const int16_t *weight, const int32_t *bias,
                                int32_t *y,
                                int32_t *hist, int hist_time) {
    int hist_H = hist_time;
    /* Build working buffer: [hist; current] along time */
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < hist_H; h++) {
            for (int w = 0; w < W; w++) {
                int idx = (h + 1) * W + w;
                int prev_idx = h * W + w;
                if (h == hist_H - 1) {
                    /* Current frame */
                    hist[c * hist_H * W + idx] = x[c * H * W + w];
                } else {
                    hist[c * hist_H * W + idx] = hist[c * hist_H * W + prev_idx];
                }
            }
        }
    }
    /* Convolve over history */
    for (int c = 0; c < C; c++) {
        int32_t b = bias ? bias[c] : 0;
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < Wout; w++) {
                int64_t sum = 0;
                for (int hk = 0; hk < Hk; hk++) {
                    for (int wk = 0; wk < Wk; wk++) {
                        int hi = h + hk * dilation - pad_h;
                        int wi = w + wk - pad_w;
                        if (hi >= 0 && hi < hist_H + 1 && wi >= 0 && wi < W) {
                            sum += (int64_t)hist[c * hist_H * W + hi * W + wi]
                                 * weight[NN_IDX_W4D(c, 0, hk, wk, 1, Hk, Wk)];
                        }
                    }
                }
                y[c * H * Wout + h * Wout + w] = nn_sat_i32(nn_round_shr(sum, -qr) + b);
            }
        }
    }
}

/* ================================================================
 * nn_ddtconv2d — Depth-wise Dilated Transposed Convolution (GTCRN)
 * ================================================================
 * Transposed version of nn_ddconv2d.
 */
static inline void nn_ddtconv2d(const int32_t *x, int C, int H, int W,
                                 int Wout, int Hk, int Wk,
                                 int pad_w, int dilation, int qr,
                                 const int16_t *weight, const int32_t *bias,
                                 int32_t *y,
                                 int32_t *hist, int hist_time) {
    /* hist management same as ddconv2d */
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < hist_time; h++) {
            for (int w = 0; w < W; w++) {
                int idx = (h + 1) * W + w;
                int prev_idx = h * W + w;
                if (h == hist_time - 1)
                    hist[c * hist_time * W + idx] = x[c * H * W + w];
                else
                    hist[c * hist_time * W + idx] = hist[c * hist_time * W + prev_idx];
            }
        }
    }
    for (int c = 0; c < C; c++) {
        int32_t b = bias ? bias[c] : 0;
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < Wout; w++) {
                int64_t sum = 0;
                for (int hk = 0; hk < Hk; hk++) {
                    for (int wk = 0; wk < Wk; wk++) {
                        int hi = h - hk * dilation + pad_w;  /* transposed */
                        int wi = w - wk + pad_w;
                        if (hi >= 0 && hi < hist_time + 1 && wi >= 0 && wi < W) {
                            int wh = Hk - 1 - hk, ww = Wk - 1 - wk;  /* rot180 */
                            sum += (int64_t)hist[c * hist_time * W + hi * W + wi]
                                 * weight[NN_IDX_W4D(c, 0, wh, ww, 1, Hk, Wk)];
                        }
                    }
                }
                y[c * H * Wout + h * Wout + w] = nn_sat_i32(nn_round_shr(sum, -qr) + b);
            }
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif /* NN_CONV_H */
