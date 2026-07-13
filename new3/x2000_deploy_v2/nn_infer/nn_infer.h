/* nn_infer.h — Reusable fixed-point neural network inference framework
 * Main entry point. Include this single header to get all operators.
 *
 * ============================================================================
 * Architecture:
 *   nn_qformat.h  — Q-format type system + saturation/rounding helpers
 *   nn_conv.h     — Conv2D family (conv, pconv, gconv, tconv, gtconv, ...)
 *   nn_norm.h     — BatchNorm + LayerNorm
 *   nn_act.h      — PReLU/AffinePReLU + Sigmoid/Tanh/LUT
 *   nn_rnn.h      — GRU step/sequence + BiGRU + cTFA(TA/FA)
 *   nn_erb.h      — ERB Band Merge/Split + CRM Mask + Shuffle
 *
 * ============================================================================
 * Design principles:
 *   1. Header-only where possible — zero .c dependency for core ops
 *   2. Currently wraps ulunas_fp.c implementations with nn_ prefixed names
 *   3. Future: extract actual implementations as static inline in each header
 *   4. Q-format constants are overridable via #define before #include
 *
 * ============================================================================
 * Data Layout Convention:
 *   ALL arrays are row-major (C convention):
 *     2D (C, W):  element[c][w] = data[c * W + w]
 *     3D (C,H,W): element[c][h][w] = data[c * H * W + h * W + w]
 *   MATLAB column-major conversion is handled at weight-export time.
 *
 * ============================================================================
 * Quick Start (new project):
 *   1. #include "nn_infer/nn_infer.h"
 *   2. Provide layer_dims.h with network dimensions
 *   3. Provide model_weights.h from MATLAB export
 *   4. Write model assembly (~50 lines) using nn_* operators
 *   5. Use nn_qformat.h helpers for Q-format conversions
 */

#ifndef NN_INFER_H
#define NN_INFER_H

#include "nn_qformat.h"
#include "nn_conv.h"
#include "nn_norm.h"
#include "nn_act.h"
#include "nn_rnn.h"
#include "nn_erb.h"

/* During transition: include original headers for ulunas_state_t type.
 * Remove once nn_infer has its own state definition. */
#include "../ulunas_fp.h"
#include "../ulunas_lut.h"

/* Version */
#define NN_INFER_VERSION_MAJOR 0
#define NN_INFER_VERSION_MINOR 1
#define NN_INFER_VERSION_PATCH 0

/* ========================================================================
 * State structure (mirrors ulunas_state_t for compatibility)
 * ======================================================================== */

/* Re-export ulunas_state_t as nn_state_t */
typedef ulunas_state_t nn_state_t;

/* Re-export init as inline */
static inline void nn_state_init(nn_state_t *s) { ulunas_state_init(s); }

#endif /* NN_INFER_H */
