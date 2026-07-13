/* nn_norm.h — BN + LN (v2 backend: ulunas_fp.c) */
#ifndef NN_NORM_H
#define NN_NORM_H
#include "nn_qformat.h"
#include "../ulunas_fp.h"
#ifdef __cplusplus
extern "C" {
#endif

static inline void nn_bn(const int32_t *x, const int16_t *weight, const int32_t *bias,
    const int32_t *running_mean, const uint16_t *running_var,
    int Qr1, int Qr2, int C, int N, int32_t *y)
{ bn_func(x,weight,bias,running_mean,running_var,Qr1,Qr2,C,N,y); }

static inline void nn_bn_uw(const int32_t *x, const uint16_t *weight, const int32_t *bias,
    const int32_t *running_mean, const uint16_t *running_var,
    int Qr1, int Qr2, int C, int N, int32_t *y)
{ bn_func_uw(x,weight,bias,running_mean,running_var,Qr1,Qr2,C,N,y); }

static inline void nn_ln(const int32_t *x, const int16_t *weight, const int32_t *bias,
    int Qr, int C, int N, int32_t *y)
{ ln_func(x,weight,bias,Qr,C,N,y); }

#ifdef __cplusplus
}
#endif
#endif
