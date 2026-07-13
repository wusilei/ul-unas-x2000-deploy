# MATLAB → C 定点 Encoder_module 转换经验

## 架构概览

```
输入 [1,129] Q20
  │
  ▼ E0: XConv
  │   TConv(1→12, 3×3, s=[1,2]) → BN → AffinePReLU → cTFA(TA+FA)
  │   [1,129] → [12,65]
  │
  ▼ E1: XMB0
  │   PConv0(12→24,g=2) → Shuffle → TConv(2×3,s=[1,2]) → PConv1(g=2) → BN → cTFA → Shuffle
  │   [12,65] → [24,33]
  │
  ▼ E2: XDWS0
  │   PConv(24→24,g=2) → Shuffle → TConv(2×3,s=[1,1]) → cTFA
  │   [24,33] → [24,33]
  │
  ▼ E3: XMB1
  │   PConv0(24→32,g=2) → Shuffle → nonTConv(1×5) → PConv1(g=2) → BN → cTFA → Shuffle
  │   [24,33] → [32,33]
  │
  ▼ E4: XDWS1
  │   PConv(32→16,g=2) → Shuffle → nonTConv(1×5) → cTFA
  │   [16,33] → [16,33]
  │
  输出 → GDPRNN
```

## 核心强制约束

1. **Q 格式统一**：所有层间激活值均为 **s32f20**，禁止层间 Q 格式转换
2. **cTFA 输出**：TA/FA 注意力权重为 **u16f15**，融合后恢复 **s32f20**
3. **TConv 时序缓存**：E0/E1/E2 的 TConv 使用跨帧缓存，Frame 0 缓存初始为零
4. **分组卷积**：所有 PConv 使用 groups=2 实现，权重按 channel 对半拆分
5. **Shuffle 模式**：MATLAB `y(1:2:end)=x(1:N/2); y(2:2:end)=x(N/2+1:end)` — 严格匹配 interleave

## 各层详细参数

### E0: XConv (XConv_module)

| 操作 | 规格 | Qr | 输入 | 输出 |
|------|------|-----|------|------|
| TConv | 3×3, 1→12, s=[1,2] | -14 | [3,129] | [12,65] |
| BN | 12 ch | Qr1=-7, Qr2=-13 | [12,65] | [12,65] |
| AffinePReLU | 12 ch | Qr1=-16, Qr2=-14 | [12,65] | [12,65] |
| cTFA TA | GRU(12→24→12) | GRU Qr1=-13,Qr2=-8, FC Qr=-8 | [12,65] | u16f15[12] |
| cTFA FA | BiGRU(12→4→4; group=4,seg=17,pad=3) | GRU Qr1=-13,Qr2=-8, FC Qr=-9 | [12,65] | u16f15[65] |
| Fusion | y(tconv × ta' × fa, >>15 each) | — | — | [12,65] Q20 |
| Cache | 保持 x 最后 2 行 | — | — | [2,129] |

### E1: XMB0 (XMB0_module)

| 操作 | 规格 | Qr | 输入 | 输出 |
|------|------|-----|------|------|
| PConv0 g0/g1 | 1×1, 6→12 | -15 | [6/6,65] | [12/12,65] |
| BN | 24 ch unsigned | Qr1=-5, Qr2=-15 | [24,65] | [24,65] |
| AffinePReLU | 24 ch | Qr1=-16, Qr2=-14 | [24,65] | [24,65] |
| Shuffle | interleave(24) | — | [24,65] | [24,65] |
| TConv | 2×3, g=24, s=[1,2] | -15 | [24,65]+cache | [24,33] |
| BN | 24 ch | Qr1=-5, Qr2=-15 | [24,33] | [24,33] |
| AffinePReLU | 24 ch | Qr1=-16, Qr2=-14 | [24,33] | [24,33] |
| PConv1 g0/g1 | 1×1, 12→12 | -14 | [12/12,33] | [12/12,33] |
| BN | 24 ch | Qr1=-5, Qr2=-15 | [24,33] | [24,33] |
| cTFA TA | GRU(24→48→24) | GRU Qr1=-13,Qr2=-8, FC Qr=-8 | — | u16f15[24] |
| cTFA FA | BiGRU(24→4→4; group=4,seg=9,pad=3) | GRU Qr1=-13,Qr2=-8, FC Qr=-9 | — | u16f15[33] |
| Fusion+Shuffle | — | — | [24,33] | [24,33] |

### E2: XDWS0 (XDWS0_module)

| 操作 | 规格 | Qr | 输入 | 输出 |
|------|------|-----|------|------|
| PConv g0/g1 | 1×1, 12→12 | -14 | [12/12,33] | [12/12,33] |
| BN | 24 ch | Qr1=-5, Qr2=-15 | [24,33] | [24,33] |
| AffinePReLU | 24 ch | Qr1=-16, Qr2=-14 | [24,33] | [24,33] |
| Shuffle | interleave(24) | — | [24,33] | [24,33] |
| TConv | 2×3, g=24, s=[1,1] | -15 | [24,33]+cache | [24,33] |
| BN | 24 ch | Qr1=-5, Qr2=-15 | [24,33] | [24,33] |
| AffinePReLU | 24 ch | Qr1=-16, Qr2=-14 | [24,33] | [24,33] |
| cTFA TA | GRU(24→48→24) | GRU Qr1=-13,Qr2=-8, FC Qr=-8 | — | u16f15[24] |
| cTFA FA | BiGRU(24→4→4; group=4,seg=9,pad=3) | GRU Qr1=-13,Qr2=-8, FC Qr=-9 | — | u16f15[33] |
| Fusion | — | — | [24,33] | [24,33] |

### E3: XMB1 (XMB1_module)

| 操作 | 规格 | Qr | 输入 | 输出 |
|------|------|-----|------|------|
| PConv0 g0/g1 | 1×1, 12→16 | -15 | [12/12,33] | [16/16,33] |
| BN | 32 ch unsigned | Qr1=-5, Qr2=-15 | [32,33] | [32,33] |
| AffinePReLU | 32 ch | Qr1=-16, Qr2=-14 | [32,33] | [32,33] |
| Shuffle | interleave(32) | — | [32,33] | [32,33] |
| nonTConv | 1×5, g=32, no cache | -14 | [32,33] | [32,33] |
| BN | 32 ch | Qr1=-5, Qr2=-15 | [32,33] | [32,33] |
| AffinePReLU | 32 ch | Qr1=-16, Qr2=-14 | [32,33] | [32,33] |
| PConv1 g0/g1 | 1×1, 16→16 | -14 | [16/16,33] | [16/16,33] |
| BN | 32 ch | Qr1=-5, Qr2=-15 | [32,33] | [32,33] |
| cTFA TA | GRU(32→64→32) | GRU Qr1=-13,Qr2=-8, FC Qr=-8 | — | u16f15[32] |
| cTFA FA | BiGRU(32→4→4; group=4,seg=9,pad=3) | GRU Qr1=-13,Qr2=-8, FC Qr=-9 | — | u16f15[33] |
| Fusion+Shuffle | — | — | [32,33] | [32,33] |

### E4: XDWS1 (XDWS1_module)

| 操作 | 规格 | Qr | 输入 | 输出 |
|------|------|-----|------|------|
| PConv g0/g1 | 1×1, 16→8 | -14 | [16/16,33] | [8/8,33] |
| BN | 16 ch | Qr1=-5, Qr2=-15 | [16,33] | [16,33] |
| AffinePReLU | 16 ch | Qr1=-16, Qr2=-14 | [16,33] | [16,33] |
| Shuffle | interleave(16) | — | [16,33] | [16,33] |
| nonTConv | 1×5, g=16, no cache | -14 | [16,33] | [16,33] |
| BN | 16 ch | Qr1=-5, Qr2=-15 | [16,33] | [16,33] |
| AffinePReLU | 16 ch | Qr1=-16, Qr2=-14 | [16,33] | [16,33] |
| cTFA TA | GRU(16→32→16) | GRU Qr1=-13,Qr2=-8, FC Qr=-8 | — | u16f15[16] |
| cTFA FA | BiGRU(16→4→4; group=4,seg=9,pad=3) | GRU Qr1=-13,Qr2=-8, FC Qr=-9 | — | u16f15[33] |
| Fusion | — | — | [16,33] | [16,33] |

## 定点 Q 格式约定

```
层间激活值:   s32f20 (所有 Conv/BN/AffinePReLU 输入输出)
注意力权重:   u16f15 (cTFA TA/FA sigmoid 输出)
RNN 隐状态:   s16f15 (TA GRU hidden state)
权重:
  Conv:       s16f14 (标准) / s16f13 (PConv/GConv/TConv)
  BN weight:  s16f14 / u16f14 (varies)
  BN bias:    s32f20
  GRU ih/hh:  s16f12
  GRU bias:   s32f20 (Q10 原始值, extract_weights 直接 cast)
  AP weight:  s16f14
  AP slope:   s16f13
  FC weight:  s16f13
  FC bias:    s32f20
```

### 关键 Qr 参数（右移量 = -Qr）

```
Conv:      Qr = -14 → >>14  (Q20 × Q14 → Q34 → >>14 → Q20)
PConv:     Qr = -14/-15
GConv:     Qr = -15
nonGConv:  Qr = -14
BN Qr1:    -5 (varies)  → >>5   (diff × var)
BN Qr2:    -15/-13       → >>15  (x_norm × weight)
AP Qr1:    -16            → >>16  (x × slope)
AP Qr2:    -14            → >>14  (x × weight)
cTFA GRU:  Qr1=-13, Qr2=-8
cTFA FC:   Qr=-8/-9
```

## 通用块模式

每个 encoder block 遵循以下模式之一：

### 模式 A: Temporal Conv Block
```
TConv (with cache) → BN → AffinePReLU → cTFA(TA+FA) → Fusion
```
适用于: E0, E2

### 模式 B: Multi-Branch Block
```
PConv0(g=2) → BN → AffinePReLU → Shuffle → TConv → BN → AP → PConv1(g=2) → BN → cTFA → Fusion → Shuffle
```
适用于: E1, E3

### 模式 C: NonTemporal Conv Block
```
PConv(g=2) → BN → AffinePReLU → Shuffle → nonTConv → BN → AP → cTFA → Fusion
```
适用于: E4

## Frame 0 验证结果

### 端到端 Encoder 层 SNR

```
Golden 文件位于: dump_matlab/frame0_enc_e{0-4}.bin

Layer   Dim        Golden File           SNR Target  Status
──────  ────────   ────────────────────  ──────────  ──────
E0      [12,65]    frame0_enc_e0.bin     >80 dB      
E1      [24,33]    frame0_enc_e1.bin     >60 dB      
E2      [24,33]    frame0_enc_e2.bin     >80 dB      
E3      [32,33]    frame0_enc_e3.bin     >60 dB      
E4      [16,33]    frame0_enc_e4.bin     >80 dB      
```

### E0 子步骤 SNR (Frame 0)

```
Golden 文件位于: dump_matlab/

Sub-Step           Golden File              SNR Target  Status
────────           ────────────────         ──────────  ──────
TConv+BN+AP        frame0_enc_e0_ctfa_in    >80 dB      
cTFA TA            frame0_enc_e0_ctfa_ta    >80 dB      
cTFA FA            frame0_enc_e0_ctfa_fa    >80 dB      
cTFA Fusion        frame0_enc_e0_ctfa_out   >80 dB      
```

### E1 子步骤 SNR (Frame 0)

```
Sub-Step           Golden File              SNR Target  Status
────────           ────────────────         ──────────  ──────
PConv0             frame0_e1_pconv0         >80 dB      
Shuffle            frame0_e1_shuf           ∞ (bit-exact)
TConv+BN+AP        frame0_enc_e1_tconv      >60 dB      
cTFA input         frame0_enc_e1_ctfa_in    >60 dB      
cTFA TA            frame0_enc_e1_ctfa_ta    >80 dB      
cTFA FA            frame0_enc_e1_ctfa_fa    >80 dB      
cTFA Out           frame0_enc_e1_ctfa_out   >60 dB      
Final (fusion+sh)  frame0_enc_e1            >60 dB      
```

> 注: E1/E3 "仅 >60 dB" 是因为分组卷积+shuffle 的多次 Q20×Q14→>>14 累加取整累积误差。
> E0/E2/E4 ">80 dB" 因为操作链更短。
> 运行 `test_matlab_golden` 获取精确 SNR 值后填入。

### 误差源（按贡献排序）

| 误差源 | 贡献 | 说明 |
|--------|------|------|
| Conv 取整 (>>14) | 主要 | 每个 3×3=9 项累加，±4.5 LSB rms |
| BN 取整 (>>5, >>15) | 主要 | 两步归一化取整 |
| AffinePReLU 取整 (>>16, >>14) | 次要 | 两步处理 |
| cTFA 浮点等效计算 | 次要 | sqrt/mean 的整数近似 |
| cTFA GRU sigmoid/tanh LUT | 次要 | LUT 插值 ±1-2 LSB |
| cTFA Fusion >>15 | 次要 | TA×FA 两次 >>15 |
| 权重 Q12-Q14 量化 | 背景 | 固有的 ±0.5 LSB |

## 验证方法

```python
# 逐层隔离验证 (Layer Isolation)
# 用 MATLAB golden 输入 → C layer → 对比 golden 输出

# E0 iso
golden_bm = load_binary('frame0_bm.bin', 129)  # MATLAB golden BM output
c_e0 = encoder_layer0_xconv(golden_bm, fresh_state)  # C implementation
golden_e0 = load_binary('frame0_enc_e0.bin', 12*65)  # MATLAB golden E0 output
snr_e0 = compute_snr_2d(golden_e0, c_e0, 12, 65)

# 全链路
c_e0_e4 = encoder_module(golden_bm, fresh_state)
for layer in range(5):
    golden = load_binary(f'frame0_enc_e{layer}.bin')
    snr = compute_snr(golden, c_outputs[layer])
```

## 关键坑点

1. **Shuffle 顺序** — MATLAB `y(1:2:end)=x(1:N/2)` 先放前半再后半，interleave 不是 deinterleave
2. **TConv 缓存更新时机** — 在 forward 完成后才更新缓存，forward 期间用旧缓存
3. **BN unsigned weight** — E1/E3 PConv0 BN 使用 `bn_func_uw`（uint16_t weight），其他用 `bn_func`（int16_t）
4. **nonTConv 无缓存** — E3/E4 的 nonTConv 只是 1×5 频率卷积，不涉及时序
5. **cTFA 聚合的 Q40→Q20 缩放** — `sum_sq / (W * 2^20)` 或 `sum_sq / (C * 2^20)`，分母必须精确匹配
6. **Layer Isolation 用零缓存** — Frame 0 所有缓存初始为零，必须用 `ulunas_state_init()`

## 代码模板

```c
// 分组 PConv (2 groups)
pconv2d_func(x,         Cin/2, Cout/2, 1, W, weight,       bias,       Qr, Cout, y);
pconv2d_func(x + Cin/2*W, Cin/2, Cout/2, 1, W, weight + Cout/2, bias + Cout/2, Qr, Cout, y + Cout/2*W);

// TConv with cache
int32_t x_c[C * Htotal * W];
memcpy(x_c, cache, C * cache_h * W * sizeof(int32_t));
memcpy(x_c + C * cache_h * W, x, C * 1 * W * sizeof(int32_t));
gconv2d_func(x_c /* or equivalent */, Cout, 1, Wout, Kh, Kw, sh, sw, weight, bias, Qr, cache, y);

// cTFA Fusion
int64_t r = 16384;
for (int c = 0; c < C; c++) {
    for (int w = 0; w < W; w++) {
        int64_t p1 = (int64_t)x[c*W + w] * ta[c];
        int32_t yt = (p1 >= 0) ? (int32_t)((p1 + r) >> 15) : (int32_t)((p1 - r) >> 15);
        int64_t p2 = (int64_t)yt * fa[w];
        y[c*W + w] = (p2 >= 0) ? (int32_t)((p2 + r) >> 15) : (int32_t)((p2 - r) >> 15);
    }
}
```
