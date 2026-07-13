/* ERB Band Merge/Split weights — extracted from para_in_mat_FP/*.mat, Q15 uint16_t */
#ifndef ERB_WEIGHTS_H
#define ERB_WEIGHTS_H
#include <stdint.h>

/* BM weight: 192×64 column-major, u16f15 */
static const uint16_t erb_erb_fc_weight[12288] = {
#include "erb_bm_weight.inc"
};

/* BS weight: 64×192 column-major, u16f15 */
static const uint16_t erb_ierb_fc_weight[12288] = {
#include "erb_bs_weight.inc"
};

#endif
