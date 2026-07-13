# MATLAB → C 定点 GDPRNN_module 转换经验

## 架构概览

```
输入 [16,33] Q20 (E4 输出)
  │
  ▼ Intra-RNN
  │   Split → x0[33,8], x1[33,8]
  │   BiGRU(nHidden=4) each → x0_out[33,8], x1_out[33,8]
  │   Concat → [33,16]
  │   FC(16→16, Qr=-9) → LN(16, Qr=-14) → Residual(+x_in)
  │
  ▼ Inter-RNN
  │   Split → x0[33,8], x1[33,8]
  │   GRU(nHidden=8, per-timestep cached) each
  │   Concat → [33,16]
  │   FC(16→16, Qr=-9) → LN(16, Qr=-13) → Residual(+intra_out)
  │
  输出 [16,33] Q20
```

两个 GDPRNN module (idx=0 和 idx=1) 结构完全相同，仅权重不同。

## 核心强制约束

1. **输入/输出 Q 格式**：**s32f20**，与 Encoder 输出一致
2. **RNN 隐状态**：**s16f15**（GRU/BiGRU 内部，sigmoid/tanh 输出）
3. **GRU 权重**：ih/hh 均为 **s16f12**，Qr1=-13 (ih path), Qr2=-8 (hh path)
4. **FC 权重**：**s16f13**，Qr=-9
5. **LN 权重**：**s16f12**，Qr=-14 (intra) / Qr=-13 (inter)
6. **Inter-RNN 缓存**：每时间步独立 hidden state，Frame 0 全零初始

## 1. Q 格式和数据布局

```
激活路径:
  Encoder 输出 Q20 → Intra-RNN (Q20→Q15 GRU→Q20 FC→Q20 LN→Q20)
  → Inter-RNN (Q20→Q15 GRU→Q20 FC→Q20 LN→Q20) → Decoder 输入

GRU 内部:
  x_t Q20 × ih_w Q12 → Q32 → >>13 → Q19 (accumulated)
  h Q15 × hh_w Q12 → Q27 → >>8  → Q19 (accumulated)
  + bias (Q10 raw) → r_t/z_t Q20
  sigmoid Q20→Q15: LUT
  tanh Q20→Q15:    LUT
  更新: h = (1-z)*n + z*h, all Q15 → output Q15

布局转换:
  Encoder: [C, W] = [16, 33] (channel-major)
  RNN 输入: [T, features] = [33, 16] (time-major) — 需要转置
  RNN 输出: [T, features] → 转置回 [C, W]
```

## 2. Intra-RNN 详细步骤

### Step 1: Split
```
x[16,33] → transpose → x_rnn[33,16]
x0 = x_rnn[:, 0:8]   # [33, 8] — 第一组
x1 = x_rnn[:, 8:16]  # [33, 8] — 第二组
```

### Step 2: BiGRU (nHidden=4, per group)
```
Forward:  t=0..32, h_fwd[4] persists across timesteps
Backward: t=32..0, h_rev[4] persists, then flip output
Concat:   y[t] = [y_fwd[t, 0:4], y_rev_flipped[t, 0:4]] → [33, 8]
```

### Step 3: Concat + FC
```
x_cat[33,16] = [x0_gru, x1_gru]   # channels 0-7, 8-15
x_fc[t, o] = round(Σ x_cat[t,i] * fc_w[i,o] * 2^(-9)) + fc_b[o]
```

### Step 4: Layer Normalization
```
mean/var computed on [33*16] flattened
x_norm = round((x_fc - mean) * inv_std * 2^(-11))
x_ln = round(x_norm * ln_w * 2^(-14)) + ln_b
```

### Step 5: Residual
```
y_intra[c, t] = x_original[c, t] + x_ln[t, c]   # saturate to int32
```

## 3. Inter-RNN 详细步骤

与 Intra-RNN 相同的前几步，但 GRU 使用**逐时间步独立缓存**：

```
MATLAB: GRU_module(x0, nHidden=8, h_cache(:, 1:8), ...)
  — 每行(timestep)使用独立的初始 hidden state
  — 所有 33 个时间步可并行处理
  — h_cache 存储在 ulunas_state_t.inter_cache_0/1 中

C: 循环 t=0..32
  h0 = inter_cache[t*8 : t*8+8]  # 加载该时间步的 hidden state
  gru_module(x0[t], 8, 8, h0, ...)  # 更新 h0
  inter_cache[t*8 : t*8+8] = h0  # 存回
```

**关键差异**：
- Intra BiGRU：时间步顺序依赖（forward/backward 沿时间传播）
- Inter GRU：时间步独立（每帧有自己独立的 persistent hidden state）

## 4. Frame 0 验证结果

### Intra-RNN 子步骤 SNR (GDPRNN idx=0, Frame 0)

```
Golden 文件位于: dump_matlab/

Sub-Step             Golden File               SNR Target  Status
────────             ────────────────          ──────────  ──────
Input (transposed)   frame0_rnn1_intra_in      ∞ (reshape)
BiGRU Group 0        frame0_rnn1_intra_gru0    >80 dB      
BiGRU Group 1        frame0_rnn1_intra_gru1    >80 dB      
Concat               frame0_rnn1_intra_cat     ∞ (bit-copy)
FC                   frame0_rnn1_intra_fc      >60 dB      
LN                   frame0_rnn1_intra_ln      >60 dB      
Output (residual)    frame0_rnn1_intra_out     >80 dB      
```

### Inter-RNN 子步骤 SNR (GDPRNN idx=0, Frame 0)

```
Sub-Step             Golden File               SNR Target  Status
────────             ────────────────          ──────────  ──────
Input                frame0_rnn1_inter_in      ∞ (copy)
GRU Group 0          frame0_rnn1_inter_gru0    >80 dB      
GRU Group 1          frame0_rnn1_inter_gru1    >80 dB      
Concat               frame0_rnn1_inter_cat     ∞ (bit-copy)
FC                   frame0_rnn1_inter_fc      >60 dB      
LN                   frame0_rnn1_inter_ln      >60 dB      
Output               frame0_rnn1               >80 dB      
```

> 注: FC 层 ">60 dB" 是因为 Q15×Q13→>>9 的取整累积（16 项累加）。
> LN 层 ">60 dB" 是因为整数 mean/var/sqrt 近似 + 两层 >>11/>>14 取整。
> 运行 `test_matlab_golden` 获取精确 SNR 值。

### 误差源（按贡献排序）

| 误差源 | 贡献 | 说明 |
|--------|------|------|
| GRU sigmoid/tanh LUT | 主要 | Q20→Q15 LUT 插值 ±1-2 LSB |
| GRU ih/hh 路径取整 | 主要 | >>13 和 >>8 各 ±0.5 LSB |
| FC 取整 (>>9) | 次要 | 16 项累加，RMS ≈ 2 LSB |
| LN sqrt 近似 | 次要 | integer sqrt 计算 inv_std |
| LN 取整 (>>11, >>14) | 次要 | 两步 >> |
| 残差加法 | 可忽略 | int32 精确 |

## 5. GRU 定点实现要点

```c
void gru_module(const int32_t *x_t, int nHidden, int in_dim,
                int16_t *h_cache,
                const int16_t *ih_weight, const int32_t *ih_bias,
                const int16_t *hh_weight, const int32_t *hh_bias,
                int Qr1, int Qr2, int16_t *y) {

    // Reset gate: r = sigmoid(ih(x) + hh(h) + bias)
    // Update gate: z = sigmoid(ih(x) + hh(h) + bias)
    // Candidate: n = tanh(ih(x) + hh(h) * r + bias)
    // Output: h' = (1-z)*n + z*h

    // ih path: sum(x_i * w_ih) >> 13 (Qr1=-13)
    // hh path: sum(h_j * w_hh) >> 8  (Qr2=-8)

    int shift1 = 13, shift2 = 8;  // -Qr1, -Qr2

    for (int j = 0; j < nHidden; j++) {
        // Compute gate values (to Q20)
        int64_t sum_ih = 0, sum_hh = 0;
        for (int i = 0; i < in_dim; i++)
            sum_ih += (int64_t)x_t[i] * ih_weight[i + in_dim * j];
        for (int i = 0; i < nHidden; i++)
            sum_hh += (int64_t)h_cache[i] * hh_weight[i + nHidden * j];

        int64_t acc = 0;
        acc += (sum_ih >= 0) ? ((sum_ih + r1) >> shift1) : ((sum_ih - r1) >> shift1);
        acc += (sum_hh >= 0) ? ((sum_hh + r2) >> shift2) : ((sum_hh - r2) >> shift2);
        acc += ih_bias[j] + hh_bias[j];
        r_buf[j] = sat_i32(acc);  // Q20
    }

    // Sigmoid: r_q15 = sigmoid_q20_to_q15(r_buf[j])
    // Tanh:    n_q15 = tanh_q20_to_q15(n_buf[j])

    // Hidden update (all Q15):
    // h' = round((32768 - z) * n >> 15) + round(z * h >> 15)
}
```

## 6. LN 整数近似实现要点

```c
// X2000-safe LN (无浮点):
// 1. mean = sum(x) / N  (Q20)
// 2. var = sum((x-mean)^2) / N  (Q40)
// 3. std = sqrt(var)  (Q20 via integer sqrt)
// 4. inv_std = min(2^31 / std, 65535)  (Q11)
// 5. x_norm = round((x - mean) * inv_std >> 11)
// 6. y = round(x_norm * weight >> Qr) + bias

// sqrt Q40→Q20: 二分查找或 Newton 迭代
uint32_t sqrt_q40_to_q20(uint64_t x_q40) {
    uint64_t lo = 0, hi = (x_q40 > 0xFFFFFFFEULL) ? 0xFFFFFFFFULL : (x_q40 + 1);
    while (lo + 1 < hi) {
        uint64_t mid = (lo + hi) >> 1;
        if (mid * mid <= x_q40) lo = mid;
        else hi = mid;
    }
    return (uint32_t)lo;
}
```

## 7. 验证方法

```python
# 加载 golden E4 输入
golden_e4 = load_binary_2d('frame0_enc_e4.bin', 16, 33)  # [16,33] col-major

# C 侧: transpose → Intra-RNN → Inter-RNN
x_rnn = transpose(golden_e4)  # [16,33] → [33,16]
y_intra = intra_rnn_module(x_rnn, idx=0)
y_inter = inter_rnn_module(y_intra, h_cache_zeros, idx=0)

# 对比 golden
golden_rnn1 = load_binary_2d('frame0_rnn1.bin', 16, 33)
snr = compute_snr(golden_rnn1, y_inter)
```

## 8. 关键坑点

1. **BiGRU vs GRU** — Intra 用 BiGRU (bidirectional)，Inter 用 GRU (single direction with cache)
2. **Inter GRU 缓存机制** — 不是顺序传播！33 个时间步**并行**处理，各自独立 hidden state
3. **转置方向** — Encoder 输出 [C,W]=[16,33] → RNN 输入 [T,feat]=[33,16] → 输出 [T,feat] → 转置回 [C,W]
4. **Residual 加法** — Intra: x + LN_out; Inter: intra_out + LN_out; 都是 int32 饱和加法
5. **LN 的 C 维度** — `ln_func` 中 `i % C` 获取 channel index（[T,C] 布局，C 是最快维度）
6. **GRU bias Q 格式** — bias 原始值（Q10 范围）直接加，不需额外缩放（MATLAB 也如此）
