# MATLAB → C 定点 Decoder_module 转换经验

## 架构概览

```
输入 x[16,33] Q20 (GDPRNN 输出) + skip connections from Encoder
  │
  ▼ D0: De_XDWS0
  │   x + e4 → PConv(16→32,g=2) → BN → AP → Shuffle → nonGTConv(1×5) → BN → AP → cTFA → Fusion
  │   [16,33]+[16,33] → [32,33]
  │
  ▼ D1: De_XMB0
  │   x + e3 → PConv0(32→24,g=2) → BN → AP → Shuffle → nonGTConv(1×5) → BN → AP → PConv1(g=2) → BN → cTFA → Fusion → Shuffle
  │   [32,33]+[32,33] → [24,33]
  │
  ▼ D2: De_XDWS1
  │   x + e2 → PConv(24→24,g=2) → BN → AP → Shuffle → GTConv(2×3,s=[1,1]) → BN → AP → cTFA → Fusion
  │   [24,33]+[24,33] → [24,33]
  │
  ▼ D3: De_XMB1
  │   x + e1 → PConv0(24→12,g=2) → BN → AP → Shuffle → GTConv(2×3,s=[1,2]) → BN → AP → PConv1(g=2) → BN → cTFA → Fusion → Shuffle
  │   [24,33]+[24,33] → [12,65]
  │
  ▼ D4: De_XConv
  │   x + e0 → TConv(12→1, 3×3, s=[1,2]) → BN → cTFA → Fusion
  │   [12,65]+[12,65] → [1,129]
  │
  输出 [1,129] Q20 → sigmoid
```

Decoder 是 Encoder 的镜像结构，通过 skip connections 逐层恢复频率分辨率。

## 核心强制约束

1. **Q 格式统一**：所有层间激活值为 **s32f20**
2. **Skip connections**：Decoder 输入 + Encoder skip = 逐元素 int32 饱和加法
3. **TConv/GTConv**：Decoder 使用**转置卷积**（zero-insertion + rot90 kernel），与 Encoder 的普通卷积方向相反
4. **nonGTConv**：D0/D1 使用非时序转置卷积（沿频率维 1×5 kernel, rot90(kernel, 90)）
5. **缓存管理**：D2/D3/D4 的 GTConv/TConv 使用跨帧缓存

## 各层详细参数

### D0: De_XDWS0

| 操作 | 规格 | Qr | 输入 | 输出 |
|------|------|-----|------|------|
| Skip add | x + e4, saturate | — | [16,33]×2 | [16,33] |
| PConv g0/g1 | 1×1, 8→16 | -14 | [8/8,33] | [16/16,33] |
| BN | 32 ch | Qr1=-5, Qr2=-15 | [32,33] | [32,33] |
| AffinePReLU | 32 ch | Qr1=-16, Qr2=-14 | [32,33] | [32,33] |
| Shuffle | interleave(32) | — | [32,33] | [32,33] |
| nonGTConv | 1×5, g=32, no cache | -14 | [32,33] | [32,33] |
| BN | 32 ch | Qr1=-5, Qr2=-15 | [32,33] | [32,33] |
| AffinePReLU | 32 ch | Qr1=-16, Qr2=-14 | [32,33] | [32,33] |
| cTFA TA | GRU(32→64→32) | GRU(-13,-8), FC(-8) | — | u16f15[32] |
| cTFA FA | BiGRU(32→4→4; g=4,s=9,p=3) | GRU(-13,-8), FC(-9) | — | u16f15[33] |
| Fusion | x × ta × fa >>15 each | — | [32,33] | [32,33] |

### D1: De_XMB0

| 操作 | 规格 | Qr | 输入 | 输出 |
|------|------|-----|------|------|
| Skip add | x + e3 | — | [32,33]×2 | [32,33] |
| PConv0 g0/g1 | 1×1, 16→12 | -15 | [16/16,33] | [12/12,33] |
| BN | 24 ch unsigned | Qr1=-5, Qr2=-15 | [24,33] | [24,33] |
| AffinePReLU | 24 ch | Qr1=-16, Qr2=-14 | [24,33] | [24,33] |
| Shuffle | interleave(24) | — | [24,33] | [24,33] |
| nonGTConv | 1×5, g=24, no cache | -14 | [24,33] | [24,33] |
| BN | 24 ch | Qr1=-5, Qr2=-15 | [24,33] | [24,33] |
| AffinePReLU | 24 ch | Qr1=-16, Qr2=-14 | [24,33] | [24,33] |
| PConv1 g0/g1 | 1×1, 12→12 | -14 | [12/12,33] | [12/12,33] |
| BN (in-place) | 24 ch | Qr1=-5, Qr2=-15 | [24,33] | [24,33] |
| cTFA TA+FA+Fusion+Shuffle | — | — | [24,33] | [24,33] |

### D2: De_XDWS1 — 关键：带缓存的转置时序卷积

| 操作 | 规格 | Qr | 输入 | 输出 |
|------|------|-----|------|------|
| Skip add | x + e2 | — | [24,33]×2 | [24,33] |
| PConv g0/g1 | 1×1, 12→12 | -14 | [12/12,33] | [12/12,33] |
| BN | 24 ch | Qr1=-5, Qr2=-15 | [24,33] | [24,33] |
| AffinePReLU | 24 ch | Qr1=-16, Qr2=-14 | [24,33] | [24,33] |
| Shuffle | interleave(24) | — | [24,33] | [24,33] |
| **GTConv** | **2×3, g=24, s=[1,1], cache** | **-14** | **[24,33]+cache** | **[24,33]** |
| BN | 24 ch | Qr1=-5, Qr2=-15 | [24,33] | [24,33] |
| AffinePReLU | 24 ch | Qr1=-16, Qr2=-14 | [24,33] | [24,33] |
| cTFA TA | GRU(24→48→24) | — | — | u16f15[24] |
| cTFA FA | BiGRU(...) | — | — | u16f15[33] |
| Fusion | — | — | [24,33] | [24,33] |

### D3: De_XMB1 — 上采样 (stride=2 恢复宽度)

| 操作 | 规格 | Qr | 输入 | 输出 |
|------|------|-----|------|------|
| Skip add | x + e1 | — | [24,33]×2 | [24,33] |
| PConv0 g0/g1 | 1×1, 12→6 | -14 | [12/12,33] | [6/6,33] |
| BN | 12 ch | Qr1=-7, Qr2=-15 | [12,33] | [12,33] |
| AffinePReLU | 12 ch | Qr1=-16, Qr2=-14 | [12,33] | [12,33] |
| Shuffle | interleave(12) | — | [12,33] | [12,33] |
| **GTConv** | **2×3, g=12, s=[1,2], cache** | **-14** | **[12,33]+cache** | **[12,65]** |
| BN | 12 ch | Qr1=-7, Qr2=-15 | [12,65] | [12,65] |
| AffinePReLU | 12 ch | Qr1=-16, Qr2=-14 | [12,65] | [12,65] |
| PConv1 g0/g1 | 1×1, 6→6 | -14 | [6/6,65] | [6/6,65] |
| BN (in-place) | 12 ch | Qr1=-7, Qr2=-15 | [12,65] | [12,65] |
| cTFA TA+FA+Fusion+Shuffle | — | — | [12,65] | [12,65] |

### D4: De_XConv — 最终层，回到频谱域

| 操作 | 规格 | Qr | 输入 | 输出 |
|------|------|-----|------|------|
| Skip add | x + e0 | — | [12,65]×2 | [12,65] |
| **TConv** | **3×3, 12→1, s=[1,2], cache** | **-14** | **[12,3,65]** | **[1,129]** |
| BN | 1 ch | Qr1=-7, Qr2=-13 | [1,129] | [1,129] |
| cTFA TA | GRU(1→2→1) | GRU(-13,-8), FC(-8) | — | u16f15[1] |
| cTFA FA | BiGRU(1→2→2; g=1,s=129,p=0) | GRU(-13,-8), FC(-8) | — | u16f15[129] |
| Fusion | x × ta[0] × fa[w] >>15 | — | [1,129] | [1,129] |

## 转置卷积关键差异

### GTConv (Grouped Transposed Conv with cache)
```c
// Step 1: 拼接缓存 [cache; x]
// Step 2: 零插入 stride_w=1 (D2) 或 2 (D3)
// Step 3: kernel 做 rot90(kernel, 2) — 180° 旋转
// Step 4: 标准卷积滑动
// Step 5: 更新缓存 = x
```

### nonGTConv (Non-Temporal Transposed Conv, 无缓存)
```c
// Step 1: 零插入 (stride=1, 沿频率维)
// Step 2: kernel 做 rot90(kernel, 90) — 90° 旋转 (1×5 → 5×1, no change)
// Step 3: 标准 1D 卷积滑动
// 无缓存更新
```

### TConv D4 (Multi-Channel Transposed Conv)
```c
// Step 1: 构建 x_cache[12, 3, 65] = [cache_d2[12,2,65]; x_con[12,1,65]]
// Step 2: 零插入 stride_w=2
// Step 3: kernel[1,12,3,3] 做 rot90(kernel, 2)
// Step 4: 12→1 通道求和 + 卷积
// Step 5: 更新 cache = x_cache[:, 1:3, :]
```

## Frame 0 验证结果

### 端到端 Decoder 层 SNR

```
Golden 文件位于: dump_matlab/

Layer   Dim        Golden File           SNR Target  Status
──────  ────────   ────────────────────  ──────────  ──────
D0      [32,33]    frame0_dec_d0.bin     >80 dB      
D1      [24,33]    frame0_dec_d1.bin     >60 dB      
D2      [24,33]    frame0_dec_d2.bin     >80 dB      
D3      [12,65]    frame0_dec_d3.bin     >60 dB      
D4      [1,129]    frame0_dec.bin        >80 dB      
```

### D0 子步骤 SNR (Frame 0)

```
Sub-Step           Golden File              SNR Target  Status
────────           ────────────────         ──────────  ──────
PConv0             frame0_d0_pconv0         >80 dB      
Shuffle            frame0_d0_shuf           ∞ (bit-exact)
TConv+BN+AP        frame0_dec_d0_tconv      >80 dB      
cTFA input         frame0_dec_d0_ctfa_in    >80 dB      
cTFA TA            frame0_dec_d0_ctfa_ta    >80 dB      
cTFA FA            frame0_dec_d0_ctfa_fa    >80 dB      
cTFA Fusion        frame0_dec_d0_ctfa_out   >80 dB      
```

### D3 子步骤 SNR (Frame 0)

```
Sub-Step           Golden File              SNR Target  Status
────────           ────────────────         ──────────  ──────
PConv0             frame0_d3_pconv0         >80 dB      
Shuffle            frame0_d3_shuf           ∞ (bit-exact)
TConv+BN+AP        frame0_dec_d3_tconv      >60 dB      
cTFA input         frame0_dec_d3_ctfa_in    >60 dB      
cTFA TA            frame0_dec_d3_ctfa_ta    >80 dB      
cTFA FA            frame0_dec_d3_ctfa_fa    >80 dB      
cTFA Fusion+Shuf   frame0_dec_d3_ctfa_out   >60 dB      
```

> 运行 `test_matlab_golden` 获取精确 SNR 值。

### 误差源（按贡献排序）

| 误差源 | 贡献 | 说明 |
|--------|------|------|
| GTConv/TConv 取整 | 主要 | 多通道累加 + >>14 取整 |
| cTFA GRU sigmoid/tanh LUT | 次要 | Q20→Q15 LUT |
| cTFA Fusion >>15 ×2 | 次要 | TA 和 FA 各一次 >>15 |
| Skip add 饱和 | 可忽略 | 仅极端值触发 |
| Shuffle | 无 | bit-exact memcpy |

## 验证方法

```python
# 加载 golden GDPRNN 输出和 Encoder skip connections
golden_rnn2 = load_binary_2d('frame0_rnn1.bin', 16, 33)  # RNN2 output
golden_e4 = load_binary_2d('frame0_enc_e4.bin', 16, 33)  # E4 skip
golden_e3 = load_binary_2d('frame0_enc_e3.bin', 32, 33)
# ... etc.

# C 侧: Decoder 全链路
fresh_state = ulunas_state_init()
d0 = decoder_layer0_de_xdws0(golden_rnn2, golden_e4, fresh_state)
d1 = decoder_layer1_de_xmb0(d0, golden_e3, fresh_state)
# ... etc.

# 对比 golden
for layer, c_out, golden in layers:
    snr = compute_snr(golden, c_out)
```

## 关键坑点

1. **Skip connection 维度匹配** — D0 skip=e4([16,33]), D1 skip=e3([32,33]), D2 skip=e2([24,33]), D3 skip=e1([24,33]), D4 skip=e0([12,65])
2. **GTConv vs nonGTConv** — D2/D3/D4 用 GTConv (带缓存+rot90 kernel)，D0/D1 用 nonGTConv (无缓存+rot90 kernel)
3. **rot90 旋转方向** — GTConv/TConv: rot90(kernel, 2) = 180°; nonGTConv: rot90(kernel, 90) = 90°
4. **D4 TConv 缓存维度** — cache 是 [12, 2, 65] 的 3D 结构，非 [Cout, W] 的 2D
5. **D4 cTFA 单通道** — TA 只有 1 个注意力权重，FA 覆盖 129 个频率 bin
6. **BN Qr 参数变化** — D0/D1/D2 用 Qr1=-5，D3/D4 用 Qr1=-7
