/* nn_act.h — Activations (v2 backend) */
#ifndef NN_ACT_H
#define NN_ACT_H
#include "nn_qformat.h"
#include "../ulunas_fp.h"
#ifdef __cplusplus
extern "C" {
#endif

static inline void nn_affine_prelu(const int32_t *x, const int16_t *weight, const int32_t *bias,
    const int16_t *slope, int Qr1, int Qr2, int C, int W, int32_t *y)
{ affineprelu_func(x,weight,bias,slope,Qr1,Qr2,C,W,y); }

static inline uint16_t nn_sigmoid_q15(int32_t x_q20)
{ return sigmoid_q20_to_q15(x_q20); }

static inline int16_t nn_tanh_q15(int32_t x_q20)
{ return tanh_q20_to_q15(x_q20); }

static inline void nn_log_gen(const int32_t *real, const int32_t *imag, int W, int32_t *out)
{ log_gen_fixed(real,imag,W,out); }

#ifdef __cplusplus
}
#endif
#endif
