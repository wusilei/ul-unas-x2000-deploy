# MATLAB → C 定点 sigmoid / BS / MASK 转换经验

## sigmoid_func

### 核心强制约束

| 参数 | MATLAB | C (PC验证) | C (X2000) |
|------|--------|-----------|-----------|
| 输入 | Decoder 输出 Q20, `/2^20` dequant | `double x = dec_q20 / 1048576.0` | Q20 int32 |
| 运算 | `1/(1+exp(-x))` float64 | `1/(1+exp(-x))` double | LUT `sigmoid_q20_to_q15()` |
| 输出 | `Fix_point(y, 'u16f15')` | `round(y*32768)` | LUT 直接输出 u16f15 |

### Frame 0 验证结果

```
整体 SNR: inf (bit-exact, PC float 路径)

┌─────┬────────────┬────────────┬──────────┬──────────┐
│ Bin │ C u16f15   │ MATLAB     │ 误差(LSB)│ Float 值  │
├─────┼────────────┼────────────┼──────────┼──────────┤
│  0  │     10324  │     10324  │     0    │  0.31506  │
│  4  │      1677  │      1677  │     0    │  0.05118  │
│  8  │      3776  │      3776  │     0    │  0.11523  │
│ 16  │      3676  │      3676  │     0    │  0.11218  │
│ 32  │       293  │       293  │     0    │  0.00894  │
│ 48  │       443  │       443  │     0    │  0.01352  │
│ 64  │       541  │       541  │     0    │  0.01651  │
│ 72  │       460  │       460  │     0    │  0.01404  │
│ 80  │       840  │       840  │     0    │  0.02563  │
│ 96  │      2121  │      2121  │     0    │  0.06473  │
│112  │      3229  │      3229  │     0    │  0.09854  │
│128  │     10503  │     10503  │     0    │  0.32053  │
└─────┴────────────┴────────────┴──────────┴──────────┘
```

> 注: 使用 golden decoder 输出 (frame0_dec.bin) 作为输入。PC float 路径 bit-exact。
> X2000 LUT 版本: 2048-entry, 预期 ±1-2 LSB u16f15。

### 代码模板 (PC验证)
```c
static uint16_t sigmoid_float_to_q15(double x) {
    double s = 1.0 / (1.0 + exp(-x));
    int32_t v = (int32_t)round(s * 32768.0);
    if (v < 0) v = 0; if (v > 65535) v = 65535;
    return (uint16_t)v;
}
```

### 代码模板 (X2000 LUT)
```c
uint16_t sigmoid_q20_to_q15(int32_t x_q20);  // 2048-entry LUT + 线性插值
```

---

## BS_module (ERB Band Splitting)

### 核心强制约束

| 参数 | MATLAB | C | 影响 |
|------|--------|---|------|
| 输入 | sigmoid 输出 u16f15, 129 bins | uint16_t Q15 | 一致 |
| 权重 | `ierbfc_weight` u16f15, [64×192] | `erb_ierb_fc_weight[12288]` | 逐值一致 |
| 低频 | bins 0-64 直通 | `y[i] = (int16_t)x[i]` | bit-exact |
| 高频 | `round(x(66:129)*W*2^(-15))` | Q15×Q15→>>15→clamp s16f15 | ±1 LSB |

### Frame 0 验证结果

```
整体 SNR: 84.07 dB
bins 0-64: bit-exact

┌─────┬────────────┬────────────┬──────────┬──────────┐
│ Bin │ C s16f15   │ MATLAB     │ 误差(LSB)│ Float 值  │
├─────┼────────────┼────────────┼──────────┼──────────┤
│  0  │     10324  │     10324  │     0    │  0.31506  │
│  8  │      3776  │      3776  │     0    │  0.11523  │
│ 16  │      3676  │      3676  │     0    │  0.11218  │
│ 32  │       293  │       293  │     0    │  0.00894  │
│ 48  │       443  │       443  │     0    │  0.01352  │
│ 64  │       541  │       541  │     0    │  0.01651  │
│ 65  │       889  │       889  │     0    │  0.02713  │
│ 72  │       718  │       718  │     0    │  0.02191  │
│ 80  │       696  │       696  │     0    │  0.02124  │
│ 96  │      2335  │      2335  │     0    │  0.07126  │
│128  │      2950  │      2950  │     0    │  0.09003  │
│160  │      1860  │      1860  │     0    │  0.05676  │
│200  │      7792  │      7792  │     0    │  0.23779  │
│240  │      8481  │      8481  │     0    │  0.25882  │
│256  │     10503  │     10503  │     0    │  0.32053  │
└─────┴────────────┴────────────┴──────────┴──────────┘
```

> 所有代表性 bin 0 LSB error。84 dB 来自少数 bin 的 ±1 LSB 差异（Q15×Q15→>>15 取整）。

### 代码模板
```c
void bs_fixed(const uint16_t *x, const uint16_t *weight, int W_in, int W_out, int16_t *y) {
    for (int i = 0; i < 65; i++) y[i] = (int16_t)x[i];
    int64_t r = 16384;
    for (int o = 0; o < 192; o++) {
        int64_t acc = 0;
        for (int i = 0; i < 64; i++) {
            int64_t prod = (int64_t)x[65+i] * weight[i + 64*o];
            if (prod >= 0) acc += (prod + r) >> 15;
            else           acc += (prod - r) >> 15;
        }
        if (acc > 32767) acc = 32767; if (acc < -32768) acc = -32768;
        y[65+o] = (int16_t)acc;
    }
}
```

---

## MASK_module — 复数频谱掩蔽

### 核心强制约束

| 参数 | MATLAB | C | 影响 |
|------|--------|---|------|
| mask | BS 输出 s16f15, 257 bins | int16_t Q15 | 一致 |
| x_real/x_imag | STFT Q20, 257 bins | int32_t Q20 | 一致 |
| 运算 | `round(x .* mask * 2^(-15))` | Q20×Q15 → >>15 → Q20 | ±0.5 LSB |

### Frame 0 验证结果

```
整体 SNR: 94.36 dB
Golden 格式: interleaved [R0,I0,R1,I1,...], 514 int32 values

┌─────┬──────────────┬──────────────┬──────────┬──────────┬───────────┐
│ Bin │ C Real Q20   │ MATLAB Real  │ 误差(LSB)│ C Imag   │ 有效位宽   │
├─────┼──────────────┼──────────────┼──────────┼──────────┼───────────┤
│  0  │      515894  │      515894  │     0    │     0    │    39     │
│  4  │      310120  │      310120  │     0    │     0    │    39     │
│  8  │      -23357  │      -23356  │    -1    │     0    │    35     │
│ 16  │        5345  │        5345  │     0    │     0    │    33     │
│ 32  │       -8264  │       -8263  │    -1    │     0    │    34     │
│ 46  │        2366  │        2366  │     0    │     0    │    32     │
│ 64  │        3818  │        3818  │     0    │     0    │    32     │
│ 80  │        1760  │        1760  │     0    │     0    │    31     │
│ 96  │        7351  │        7351  │     0    │     0    │    33     │
│128  │         911  │         911  │     0    │     0    │    30     │
│200  │       -1365  │       -1364  │    -1    │     0    │    31     │
│256  │         323  │         323  │     0    │     0    │    29     │
└─────┴──────────────┴──────────────┴──────────┴──────────┴───────────┘
```

> 93/257 bins (36%) 有 ±1 LSB 误差，来自 double vs int64 取整差异。无 bins 误差 >1 LSB。
> 虚部全部为 0（输入音频实数，STFT 后虚部全零）。

### ⚠️ Golden 文件关键细节

MATLAB `cat(1, y_real, y_imag)` 写出时 column-major flatten 产生 **interleaved** 格式：
```
[R0, I0, R1, I1, ..., R256, I256]  ← 不是 [R0..R256, I0..I256]!
```
C 侧读取时必须 deinterleave:
```c
for (int i = 0; i < 257; i++) {
    golden_real[i] = raw[i * 2];
    golden_imag[i] = raw[i * 2 + 1];
}
```

### 代码模板
```c
void mask_fixed(const int16_t *mask, const int32_t *x_real, const int32_t *x_imag,
                int W, int32_t *y) {
    int64_t r = 16384;
    for (int i = 0; i < W; i++) {
        int64_t pr = (int64_t)x_real[i] * mask[i];
        int64_t pi = (int64_t)x_imag[i] * mask[i];
        y[i]     = (pr >= 0) ? (int32_t)((pr + r) >> 15) : (int32_t)((pr - r) >> 15);
        y[W + i] = (pi >= 0) ? (int32_t)((pi + r) >> 15) : (int32_t)((pi - r) >> 15);
    }
}
```

---

## ISTFT_func

### 核心流程

```
MASK Q20 [2×257] → Q15 (>>5) → 共轭对称填充 512 → IFFT (9级 >>15 + /512)
  → 加窗 (Q15×Q15→>>15) → OLA (int32 累加) → WOSA 归一化 → PCM Q15
```

### 定点实现要点

- IFFT = `conj(FFT(conj(X))) / N`: 每级 >>15 后 >>9 (/512)
- 加窗: `(ifft[i] × window[i] + 16384) >> 15`
- OLA: int32 累加缓冲区, 50% overlap (256 hop)
- WOSA: `audio / sum(window²)`, Hann² 50% overlap 中间区域 ≈ 1.0

### 验证状态

⚠️ ISTFT 需要 T≥3 帧的 OLA 累积才能得到正确的时域输出。Frame 0 单帧 IFFT 可验证，但需要与 MATLAB 单帧 ifft 输出比较。当前无 ISTFT golden dump 文件。

验证方法:
```bash
gcc -O2 -std=c99 -o istft_test istft_test.c -lm
./istft_test <golden_dir>
# 输出: 512 点时域采样 (IFFT + 加窗)
# 对比 MATLAB: ifft(full_spec_frame0, 512) .* hann_window
```
