# new2 — UL-UNAS C Fixed-Point Verification Workspace

> 2026-07-13, 基于 new/ 已验证代码，在 new2 中继续全管线集成

---

## 当前状态

### ✅ 已验证通过（独立模块测试）

| 模块 | SNR | 状态 | 测试程序 |
|------|-----|------|---------|
| log_gen | bit-exact | PERFECT | log_gen_test |
| BM | 124 dB | PERFECT | bm_test |
| Sigmoid | bit-exact | PERFECT | sigmoid_bs_mask_test |
| BS | 84 dB | PASS | sigmoid_bs_mask_test |
| MASK | 94 dB | PASS | sigmoid_bs_mask_test |

### 🔶 全管线（成熟代码集成中）

| 层 | SNR | 状态 |
|----|-----|------|
| log_gen + BM | 84 dB | PASS |
| E0 TConv+BN+AP | 81 dB | PASS |
| E0 cTFA TA/FA | 37-51 dB | FAIL |
| E1-E4 Encoder | 0.5-4 dB | FAIL |
| RNN1 intra | 38-42 dB | FAIL |
| RNN1 inter | 7-18 dB | FAIL |
| Decoder D0-D4 | 44-105 dB | MIXED |
| Decoder (final) | 1 dB | FAIL |
| MASK | 0.5 dB | FAIL |

**根因**：成熟项目 `x2000_deploy_v2/` 的 encoder/RNN/decoder 算子（ulunas_fp.c + ulunas_modules.c）和当前 MATLAB golden 在权重/精度上不对齐。独立模块测试通过是因为它们自带了已验证的行内实现。

---

## 目录结构

```
new2/
├── README.md                    ← 本文件
├── export_frame0_golden_v3.m    ← MATLAB golden 导出（从 C STFT）
├── stft/
│   ├── STFT_func.m
│   ├── stft_window.mat
│   └── test_wavs/noisy_fileid_1.wav
├── dump_matlab/                 ← MATLAB golden 文件
├── dump_c/                      ← C dump 文件
└── ulunas_c/
    ├── Makefile
    ├── log_gen_test.c           ← ✅ log_gen 独立测试
    ├── bm_test.c                ← ✅ BM 独立测试
    ├── sigmoid_bs_mask_test.c   ← ✅ Sigmoid/BS/MASK 测试
    ├── istft_test.c             ← ISTFT 测试
    ├── erb_weights.h            ← ERB 权重（固定滤波器组）
    ├── erb_bm_weight.inc        ← BM 权重
    ├── erb_bs_weight.inc        ← BS 权重
    ├── test_matlab_golden.c     ← 🔶 全管线验证（集成中）
    └── dump_matlab/             ← golden 文件副本（供测试读取）
```

---

## 快速开始

### 1. MATLAB 导出 golden

```matlab
cd('D:\haidesi\haidesi\ul-unas-x2000-deploy\UL-UNAS_SE_FPversion_v2');
export_all_layers_v2
```
输出到 `x2000_deploy_v2/dump_matlab/`。

### 2. 拷贝 golden 到 new2

```bash
SRC=../../UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/dump_matlab
cp $SRC/frame*_*.bin dump_matlab/
```

### 3. 编译并运行（VM: 192.168.56.101）

```bash
cd /media/sf_haidesi/haidesi/ul-unas-x2000-deploy/new2/ulunas_c
make all                    # 编译所有测试
./log_gen_test dump_matlab  # 独立模块测试
./bm_test dump_matlab
./sigmoid_bs_mask_test dump_matlab
./test_matlab_golden dump_matlab  # 全管线验证
```

---

## Q-Format 约定（严格遵循 Fix_point.m）

```
激活值:
  s32f20: 主激活 (Q_ACT=20)
  u32f20: log_gen 输出, cTFA 聚合
  s16f15: GRU hidden state, tanh 输出
  u16f15: sigmoid 输出, attention masks
  u16f11: LN 1/sqrt(var)

关键转换:
  float → Q15: round(x * 32768)
  float → Q20: round(x * 1048576)
  Q15 → Q20:  x << 5 (exact)
  Q20 → Q15:  (x + 16) >> 5
```

---

## 下一步

1. **对齐 encoder/RNN/decoder 算子精度** — 当前独立模块（log_gen/BM/sigmoid/BS/mask）通过，但 encoder cTFA TA/FA、GRU、LN 等子模块有 ~40-60 dB SNR
2. **排查 GRU 定点精度** — intra_rnn GRU 和 FA GRU 的 C vs MATLAB 差异较大（~38 dB），可能的原因是：
   - GRU 权重 Q 格式不一致
   - sigmoid/tanh 定点近似精度
   - 缓存状态传播误差
3. **Decoder 串联** — D0 iso 测试 47 dB，但 D1 降到 -6 dB，需排查 D1 的 skip connection 权重索引

---

## VM 信息

- IP: 192.168.56.101 (VirtualBox Host-Only)
- 用户/密码: a / a
- 共享目录: /media/sf_haidesi/ → D:\haidesi\
