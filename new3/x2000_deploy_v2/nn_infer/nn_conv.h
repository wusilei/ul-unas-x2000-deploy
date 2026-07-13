/* nn_conv.h — Unified convolution operators (v2 backend: ulunas_fp.c) */
#ifndef NN_CONV_H
#define NN_CONV_H
#include "nn_qformat.h"
#include "../ulunas_fp.h"
#ifdef __cplusplus
extern "C" {
#endif

static inline void nn_conv2d(const int32_t *x, int Cin, int Cout, int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    const int16_t *weight, const int32_t *bias, int Qr, int32_t *y)
{ conv2d_func(x,Cin,Cout,Hout,Wout,Kh,Kw,stride_h,stride_w,weight,bias,Qr,y); }

static inline void nn_pconv2d(const int32_t *x, int Cin, int Cout, int Hout, int Wout,
    const int16_t *weight, const int32_t *bias, int Qr, int wstride, int32_t *y)
{ pconv2d_func(x,Cin,Cout,Hout,Wout,weight,bias,Qr,wstride,y); }

static inline void nn_gconv2d(const int32_t *x, int Cout, int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    const int16_t *weight, const int32_t *bias, int Qr, int32_t *cache, int32_t *y)
{ gconv2d_func(x,Cout,Hout,Wout,Kh,Kw,stride_h,stride_w,weight,bias,Qr,cache,y); }

static inline void nn_nongconv2d(const int32_t *x, int Cout, int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    const int16_t *weight, const int32_t *bias, int Qr, int32_t *y)
{ non_gconv2d_func(x,Cout,Hout,Wout,Kh,Kw,stride_h,stride_w,weight,bias,Qr,y); }

static inline void nn_tconv2d(const int32_t *x, int Cin, int Cout, int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    const int16_t *weight, const int32_t *bias, int Qr, int32_t *y)
{ tconv2d_func(x,Cin,Cout,Hout,Wout,Kh,Kw,stride_h,stride_w,weight,bias,Qr,y); }

static inline void nn_gtconv2d(const int32_t *x, int Cout, int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    const int16_t *weight, const int32_t *bias, int Qr, int32_t *cache, int32_t *y)
{ gtconv2d_func(x,Cout,Hout,Wout,Kh,Kw,stride_h,stride_w,weight,bias,Qr,cache,y); }

static inline void nn_nongtconv2d(const int32_t *x, int Cout, int Hout, int Wout,
    int Kh, int Kw, int stride_h, int stride_w,
    const int16_t *weight, const int32_t *bias, int Qr, int32_t *y)
{ non_gtconv2d_func(x,Cout,Hout,Wout,Kh,Kw,stride_h,stride_w,weight,bias,Qr,y); }

#ifdef __cplusplus
}
#endif
#endif
