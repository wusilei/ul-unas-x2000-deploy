/* nn_erb.h — ERB BM/BS/MASK + Shuffle (v2 backend) */
#ifndef NN_ERB_H
#define NN_ERB_H
#include "nn_qformat.h"
#include "../ulunas_fp.h"
#ifdef __cplusplus
extern "C" {
#endif

static inline void nn_bm(const int32_t *x, const uint16_t *w, int W_in, int W_out, int32_t *y)
{ bm_fixed(x,w,W_in,W_out,y); }

static inline void nn_bs(const uint16_t *x, const uint16_t *w, int W_in, int W_out, int16_t *y)
{ bs_fixed(x,w,W_in,W_out,y); }

static inline void nn_mask(const int16_t *mask, const int32_t *xr, const int32_t *xi,
    int N, int32_t *y)
{ mask_fixed(mask,xr,xi,N,y); }

static inline void nn_shuffle_interleave(const int32_t *src, int C, int W, int32_t *dst)
{ shuffle_interleave(src,C,W,dst); }

static inline void nn_shuffle_deinterleave(const int32_t *src, int C, int W, int32_t *dst)
{ shuffle_deinterleave(src,C,W,dst); }

#ifdef __cplusplus
}
#endif
#endif
