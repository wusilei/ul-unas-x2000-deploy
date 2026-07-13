/* nn_rnn.h — GRU/BiGRU/cTFA (v2 backend) */
#ifndef NN_RNN_H
#define NN_RNN_H
#include "nn_qformat.h"
#include "../ulunas_fp.h"
#ifdef __cplusplus
extern "C" {
#endif

/* GRU: single-timestep, h_cache = int16_t Q15 */
static inline void nn_gru(const int32_t *x_t, int nHidden, int in_dim,
    int16_t *h_cache, const int32_t *ih_weight, const int32_t *ih_bias,
    const int32_t *hh_weight, const int32_t *hh_bias,
    int Qr1, int Qr2, int16_t *y_out)
{ gru_module(x_t,nHidden,in_dim,h_cache,ih_weight,ih_bias,hh_weight,hh_bias,Qr1,Qr2,y_out); }

/* BiGRU: T timesteps */
static inline void nn_bigru(const int32_t *x, int T, int nHidden, int in_dim,
    const int32_t *ih_w, const int32_t *ih_b, const int32_t *hh_w, const int32_t *hh_b,
    const int16_t *re_ih_w, const int32_t *re_ih_b,
    const int16_t *re_hh_w, const int32_t *re_hh_b,
    int Qr1, int Qr2, int16_t *y_out)
{ bigru_module(x,T,nHidden,in_dim,ih_w,ih_b,hh_w,hh_b,re_ih_w,re_ih_b,re_hh_w,re_hh_b,Qr1,Qr2,y_out); }

/* cTFA Time Attention */
static inline void nn_ctfa_ta(const int32_t *x, int C, int W, int nHidden,
    int16_t *h_cache, const int32_t *ih_w, const int32_t *ih_b,
    const int32_t *hh_w, const int32_t *hh_b,
    const int16_t *fc_w, const int32_t *fc_b,
    int Qr1, int Qr2, int fc_qr, uint16_t *ta_out)
{ ctfa_ta_module(x,C,W,nHidden,h_cache,ih_w,ih_b,hh_w,hh_b,fc_w,fc_b,Qr1,Qr2,fc_qr,ta_out); }

/* cTFA Frequency Attention */
static inline void nn_ctfa_fa(const int32_t *x, int C, int W, int nHidden,
    int group, int seg, int pad,
    const int32_t *ih_w, const int32_t *ih_b,
    const int32_t *hh_w, const int32_t *hh_b,
    const int16_t *re_ih_w, const int32_t *re_ih_b,
    const int16_t *re_hh_w, const int32_t *re_hh_b,
    const int16_t *fc_w, const int32_t *fc_b,
    int Qr1, int Qr2, int fc_qr, uint16_t *fa_out)
{ ctfa_fa_module(x,C,W,nHidden,group,seg,pad,ih_w,ih_b,hh_w,hh_b,re_ih_w,re_ih_b,re_hh_w,re_hh_b,fc_w,fc_b,Qr1,Qr2,fc_qr,fa_out); }

#ifdef __cplusplus
}
#endif
#endif
