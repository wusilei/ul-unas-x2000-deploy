# nn_infer — Reusable Fixed-Point Neural Network Inference Framework

C99 header-only framework for deploying MATLAB-trained audio denoise models
to MIPS/ARM embedded platforms (X2000, etc.) with zero floating-point operations.

## Quick Start

```c
#include "nn_infer.h"

// 1. Your model weights (exported from MATLAB)
#include "model_weights.h"

// 2. Your model inference (compose nn_* operators)
typedef struct { int16_t gru_h[16]; } my_state_t;

static void my_infer(const int32_t *real_q20, const int32_t *imag_q20,
                     int n_bins, int32_t *crm_out, void *user) {
    my_state_t *st = (my_state_t*)user;
    // Log-mag → BM → Encoder → GRU → Decoder → BS → MASK
    int32_t x_log[257], x_bm[129];
    nn_log_gen_q20(real_q20, imag_q20, 257, x_log);
    nn_bm(x_log, erb_weight, 257, 129, x_bm);
    // ... assemble your model ...
    nn_mask(y_mask, real_q20, imag_q20, 257, crm_out);
}

// 3. Main with X2000 iccom pipeline
int main(void) {
    nn_pipeline_config_t cfg = NN_PIPELINE_8K_ULUNAS;
    nn_pipeline_t *p = nn_pipeline_create(&cfg);
    my_state_t st = {0};
    // Optional AGC
    nn_agc_state_t agc;
    nn_agc_init(&agc, &NN_AGC_GATE_DEFAULT);

    return nn_iccom_main(400, my_infer, &st, &agc, p);
}
```

Build:
```bash
mips-linux-gnu-gcc -O3 -std=c99 -march=mips32r2 -msoft-float \
    -I. main.c nn_core/nn_lut.c -lm -lrt -static -o denoise_q15
```

## Directory Structure

```
nn_infer/
├── nn_infer.h                  # Master include
├── README.md
├── nn_core/                    # Platform-independent core operators
│   ├── nn_qformat.h            # Q-format system, sat/round macros
│   ├── nn_lut.h                # LUT declarations
│   ├── nn_lut.c                # LUT data tables (compile once)
│   ├── nn_conv.h               # conv2d/tconv2d/pconv2d/gconv2d/...
│   ├── nn_norm.h               # BN (sw/uw) + LN
│   ├── nn_act.h                # PReLU + AffinePReLU
│   ├── nn_rnn.h                # GRU + BiGRU
│   └── nn_shuffle.h            # Channel shuffle
├── nn_signal/                  # Signal processing (FFT, ERB, MASK, WOLA)
│   ├── nn_fft_q15.h            # 512-pt Q15 FFT (fw + inv)
│   ├── nn_erb.h                # ERB Band Merge/Split
│   ├── nn_mask.h               # CRM Mask application
│   └── nn_wola.h               # WOLA synthesis
└── nn_pipeline/                # Pipeline shells (X2000, AGC)
    ├── nn_pipeline.h           # Generic STFT→Infer→ISTFT pipeline
    ├── nn_agc.h                # Voice AGC (LMS + 3-level gate)
    └── nn_iccom.h              # X2000 iccom main loop template
```

## Q-Format Quick Reference

| Format | C Type | Scale | Used For |
|--------|--------|-------|----------|
| s32f20 | int32_t | ×1,048,576 | Activations, Conv I/O, BN intermediates |
| s16f15 | int16_t | ×32,768 | GRU hidden state, tanh output |
| u16f15 | uint16_t | ×32,768 | Sigmoid output, attention masks |
| s16f14 | int16_t | ×16,384 | Conv/DeConv weights, BN weights |
| s16f13 | int16_t | ×8,192 | PConv/FC weights |
| s16f12 | int16_t | ×4,096 | GRU/LN weights |
| s16f10 | int16_t | ×1,024 | GRU bias |

## Operators Summary

### Convolution (nn_conv.h)
| Operator | Description | Source |
|----------|-------------|--------|
| nn_conv2d | Standard 2D conv | Both |
| nn_tconv2d | Transposed 2D conv | Both |
| nn_pconv2d | Point-wise 1×1 conv | Both |
| nn_gconv2d | Grouped temporal conv + cache | ULUNAS |
| nn_gtconv2d | Grouped temporal transposed conv + cache | ULUNAS |
| nn_nongconv2d | Grouped non-temporal freq conv | ULUNAS |
| nn_nongtconv2d | Grouped non-temporal freq tconv | ULUNAS |
| nn_ddconv2d | Depth-wise dilated conv + history | GTCRN |
| nn_ddtconv2d | Depth-wise dilated tconv + history | GTCRN |

### Normalization (nn_norm.h)
| Operator | Description |
|----------|-------------|
| nn_bn_sw | BatchNorm with signed weight |
| nn_bn_uw | BatchNorm with unsigned weight |
| nn_ln | LayerNorm (online mean/var) |

### Activation (nn_act.h)
| Operator | Description |
|----------|-------------|
| nn_prelu | Parametric ReLU |
| nn_affine_prelu | AffinePReLU (PReLU + affine + residual) |
| nn_sigmoid_q15 | Sigmoid Q20→Q15 (1024-pt LUT) |
| nn_tanh_q15 | Tanh Q20→Q15 (1024-pt LUT) |
| nn_log10_q20 | Log10 Q20→Q20 (512-pt LUT) |
| nn_sqrt_q40_to_q20 | Integer sqrt Q40→Q20 |

### RNN (nn_rnn.h)
| Operator | Description |
|----------|-------------|
| nn_gru_step | Single-timestep GRU |
| nn_gru_seq | Multi-timestep GRU |
| nn_bigru | Bidirectional GRU |

### Signal Processing
| Module | Description |
|--------|-------------|
| nn_fft_q15 | 512-pt Q15 FFT (fw + inv) |
| nn_erb | ERB Band Merge/Split |
| nn_mask | CRM Mask application |
| nn_wola | WOLA synthesis |
| nn_log_gen_q20 | Log-magnitude compression |

## Bug Prevention Checklist

☐ All sigmoid gate variables use `uint16_t` (not `int16_t`!)
☐ GRU: ih and hh paths have independent `qr1`/`qr2`
☐ GRU hidden state: Q20→Q15 via `round_shr(h, 5)`
☐ LN: channel index = `i % C` (not `i / T`)
☐ BN: `running_var` Q-format matches MATLAB export per block
☐ Conv weights: row-major C layout equivalent to MATLAB col-major
☐ State initialization OUTSIDE frame loop
☐ All calibration uses **final output SNR** (never intermediate golden)

## Verified Deployments

- **GTCRN denoise_v19_q15**: X2000, 831KB, 576KB RSS, RTF 0.73×
- **UL-UNAS linux_api9**: X2000, 1.06MB, 764KB RSS, RTF 0.73×

## License

MIT
