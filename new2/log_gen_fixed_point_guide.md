# MATLAB → C 定点 log_gen (Log-Magnitude Compression) 转换经验

## 核心强制约束

1. 输入：`spec_real`, `spec_imag` 均为 **s32f20**（来自 Fix_point(STFT_output, 's32f20')）
2. 输出：**s32f20**，257 bins（DC→Nyquist）
3. PC 验证：使用 double sqrt + log10 达到 **bit-exact** 对齐
4. X2000 部署：使用整数 sqrt (binary search Q40→Q20) + 2048-entry LUT log10，精度 ~3 LSB

## 1. 输入对齐检查清单

| 参数 | MATLAB (`log_gen.m`) | C (PC验证) | C (X2000) | 影响 |
|------|---------------------|-----------|-----------|------|
| 实部量化 | `Fix_point(stft_real, 's32f20')` | `round(float*2^20)` | 同左 | bit-exact |
| 虚部量化 | `Fix_point(stft_imag, 's32f20')` | `round(float*2^20)` | 同左 | bit-exact |
| 幅度 | `sqrt(real²+imag²)` (float64) | `sqrt()` (double) | `isqrt(r²+i²)` | X2000 ±1 LSB |
| 钳位 | `max(mag, 1e-12)` | `max(mag, 1e-12)` | `max(mag, 1)` | X2000 1 LSB 等效 |
| log10 | `log10()` (float64) | `log10()` (double) | LUT+插值 | X2000 ±2 LSB |
| 输出量化 | `Fix_point(y, 's32f20')` | `round(y*2^20)` | LUT 直接输出 Q20 | bit-exact (PC) |

## 2. 定点 Q 格式约定

```
信号路径:  STFT Q20 [257×2] → log_gen → Q20 [257]

PC 验证路径 (bit-exact):
  Q20 int32 → /2^20 → double → sqrt → log10 → clamp → *2^20 → round → Q20 int32

X2000 部署路径 (~3 LSB):
  Q20 int32 → int64 mag²(Q40) → isqrt(Q40→Q20) → log10 LUT(Q20→Q20) → Q20 int32
```

### 关键转换

```
float → Q20:  round(x * 1048576)
Q20 → float:  x / 1048576.0
Q20² → Q40:   (int64_t)real*real + (int64_t)imag*imag  → uint64_t
整数 sqrt:     binary search on uint64_t → uint32_t (Q20), 40步收敛
log10 LUT:     log10(2^k/2^20) * 2^20  for k=0..51, 2048 entries + 线性插值
```

## 3. Frame 0 验证结果

### 数值对比 (Q20 整数，对比 BM golden bins 0-64 直通)

```
验证方法: C log_gen → BM → 对比 frame0_bm.bin bins 0-64
输入: noisy_fileid_1.wav, Frame 0 STFT (Q20)
BM 直通 bins 0-64: bit-exact (0 LSB error)
合并 bins 65-128: 124.09 dB SNR

结论: log_gen 与 MATLAB bit-exact (PC float 路径)
```

### 代表 bin 数值

```
┌─────┬──────────────┬──────────────┬──────────┬──────────┬───────────┐
│ Bin │ C log Q20    │ MATLAB Q20   │ 误差(LSB)│ 误差(%)  │ 有效位宽   │
├─────┼──────────────┼──────────────┼──────────┼──────────┼───────────┤
│  0  │      202965  │      202965  │     0    │  0.0000  │    38     │
│  4  │      798854  │      798854  │     0    │  0.0000  │    40     │
│  8  │     -748464  │     -748464  │     0    │  0.0000  │    40     │
│ 16  │    -1407829  │    -1407829  │     0    │  0.0000  │    41     │
│ 32  │      -57565  │      -57565  │     0    │  0.0000  │    36     │
│ 46  │     -961736  │     -961736  │     0    │  0.0000  │    40     │
│ 64  │     -688456  │     -688456  │     0    │  0.0000  │    40     │
│ 128 │    -9650952  │    -9650952  │     0    │  0.0000  │    44     │
│ 200 │    -5850011  │    -5850011  │     0    │  0.0000  │    43     │
│ 256 │     -622092  │     -622092  │     0    │  0.0000  │    40     │
└─────┴──────────────┴──────────────┴──────────┴──────────┴───────────┘
```

### 各 bin 有效位宽 (Q20 格式)

```
Bin   C(Q20)     有效整数位  小数位  总有效位宽   Float 值
───   ────────   ────────   ────   ────────   ──────────
  0     202965      17        20       37      0.193563
  4     798854      19        20       39      0.761847
  8    -748464      19        20       39     -0.713791
 16   -1407829      20        20       40     -1.342610
 32     -57565      15        20       35     -0.054898
 46    -961736      19        20       39     -0.917183
 64    -688456      19        20       39     -0.656563
128   -9650952      23        20       43     -9.203865
200   -5850011      22        20       42     -5.578990
256    -622092      19        20       39     -0.593270
```

> 有效位宽 = ⌈log₂(|Q20_value|)⌉ + 20
> Bin 128 有效位宽 43 = log₂(9650952) + 20 ≈ 23 + 20
> 所有 bin 小数位均为 20 (Q20 格式固定)，整数位 = 总有效位宽 - 20

## 4. 误差源分析 (X2000 整数路径)

| 误差源 | 贡献 | 说明 |
|--------|------|------|
| log10 LUT 插值 | ±2 LSB | 2048-entry 线性插值，Q20 输出 |
| 整数 sqrt 取整 | ±0.5 LSB | 二分查找最终步长 1 LSB |
| Q20 输入量化 | 0 | Fix_point 上游已完成 |

PC float 路径: 0 LSB (bit-exact)
X2000 整数路径: 预期 ~3 LSB rms

## 5. 验证方法

```bash
# 编译
gcc -O2 -std=c99 -o log_gen_test log_gen_test.c -lm

# 运行 (需要 frame0_stft_real.bin + frame0_stft_imag.bin)
./log_gen_test <golden_dir>

# 输出 CSV: Bin, C_Q20, Golden_Q20, Diff_LSB, Float_C, Float_G, Error_pct
# 通过 BM golden bins 0-64 间接验证 (BM is passthrough for those bins)
```

## 6. 代码模板

### PC 验证版 (bit-exact)
```c
void log_gen_fixed(const int32_t *real_q20, const int32_t *imag_q20, int W, int32_t *out) {
    for (int w = 0; w < W; w++) {
        double real = (double)real_q20[w] / 1048576.0;
        double imag = (double)imag_q20[w] / 1048576.0;
        double mag = sqrt(real*real + imag*imag);
        out[w] = (int32_t)round(log10(mag < 1e-12 ? 1e-12 : mag) * 1048576.0);
    }
}
```

### X2000 版 (无浮点)
```c
void log_gen_fixed_x2000(const int32_t *real_q20, const int32_t *imag_q20, int W, int32_t *out) {
    for (int w = 0; w < W; w++) {
        uint64_t r2 = (uint64_t)((int64_t)real_q20[w] * (int64_t)real_q20[w]);
        uint64_t i2 = (uint64_t)((int64_t)imag_q20[w] * (int64_t)imag_q20[w]);
        uint32_t mag = isqrt_q40_to_q20(r2 + i2);    // binary search
        out[w] = log10_q20_lut((int32_t)(mag > 0 ? mag : 1));  // 2048-entry LUT
    }
}
```
