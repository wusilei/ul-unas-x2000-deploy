# new2 — UL-UNAS Full Pipeline C vs MATLAB 验证工作区

> 2026-07-13 基于 Main_infer.m 逐层对齐验证
> VM: 192.168.56.101 (a/a), WSL/交叉编译, 权重: linux_api13

---

## 项目概述

```
new2/
├── README.md                           ← 本文件
├── ulunas_full_pipeline_alignment.md   ← ★ 全管线逐层 C vs MATLAB 对比文档
├── export_frame0_golden_v3.m           ← MATLAB golden 导出 (从 C STFT dump)
├── dump_per_bin.py                     ← 自动生成逐 bin C vs M 对比表
│
├── stft/                               ← STFT 定点转换参考
│   ├── Main_infer.m                    ← MATLAB 推理入口 (函数调用顺序)
│   ├── STFT_func.m                     ← STFT 浮点实现
│   └── stft_window.mat                 ← Hann² 窗函数
│
├── ulunas_c/                           ← C 测试代码
│   ├── Makefile                        ← 统一编译
│   ├── log_gen_test.c / bm_test.c      ← 独立模块测试 (bit-exact / 124 dB)
│   ├── sigmoid_bs_mask_test.c          ← Sigmoid+BS+MASK 测试 (84-94 dB)
│   ├── dump_layers.c                   ← 全管线 12 层 C dump 程序
│   └── erb_weights.h / erb_*_weight.inc← ERB 滤波器组权重
│
├── dump_matlab/                        ← MATLAB golden 文件
└── dump_c/                             ← C dump 文件
```

---

## 全管线 SNR (Frame 0, 按 Main_infer.m 顺序)

| # | 函数 | SNR | C格式 | 关键 |
|---|------|-----|-------|------|
| 1 | STFT_func | bit-exact | s32f20 | V3: C dump→M加载, 输入对齐 |
| 2 | log_gen | bit-exact | s32f20 | int64平方和→isqrt64→LUT |
| 3 | BM_module | 84.2 dB | s32f20 | ERB矩阵乘, col-major权重 |
| 4 | Encoder E0 | 67.5 dB | s32f20 | cTFA TA/FA GRU 主导损失 |
| 5 | Encoder E1 | 64.7 dB | s32f20 | PConv0 96dB, cTFA TA 94dB |
| 6 | Encoder E2 | 65.3 dB | s32f20 | skip connection 稳定 |
| 7 | Encoder E3 | 64.5 dB | s32f20 | |
| 8 | Encoder E4 | 64.1 dB | s32f20 | |
| 9 | RNN1 | 67.5 dB | s32f20 | Intra GRU 72dB, Inter FC 80dB |
| 10 | RNN2 | 64.6 dB | s32f20 | |
| 11 | Decoder | 62.0 dB | s32f20 | D0-D4 级联, D3 TA 91dB bit-level |
| 12 | sigmoid_func | bit-exact | u16f15 | LUT |
| 13 | BS_module | 84.1 dB | s16f15 | ERB矩阵乘 |
| 14 | MASK_module | 94.4 dB | s32f20 | 独立测试 |

---

## 快速开始

### 1. MATLAB 导出 golden
```matlab
cd('D:\haidesi\haidesi\ul-unas-x2000-deploy\UL-UNAS_SE_FPversion_v2');
export_all_layers_v2
```

### 2. C 编译 (VM: 192.168.56.101)
```bash
ssh a@192.168.56.101
cd /media/sf_haidesi/haidesi/ul-unas-x2000-deploy/new2/ulunas_c
make all
```

### 3. 运行测试
```bash
# 独立模块
./log_gen_test dump_matlab       # bit-exact
./bm_test dump_matlab             # 124 dB
./sigmoid_bs_mask_test dump_matlab # 84-94 dB

# 全管线 (需要 x2000_deploy_v2 的 dump_full 二进制)
cd /media/sf_haidesi/haidesi/ul-unas-x2000-deploy/UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2
./dump_full dump_matlab           # dump 全部 12 层 C 输出

# 逐 bin 对比
cd /media/sf_haidesi/haidesi/ul-unas-x2000-deploy/new2
python3 dump_per_bin.py ../UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/dump_matlab ../UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/dump_c
```

---

## 关键发现

### 权重对齐
- 10:30 编译的二进制 (linux_api13 权重) 与 MATLAB golden 完全对齐 (62-67 dB)
- extract_weights 多次运行产生相同值 (para_in_mat_FP/ 未变)
- `ulunas_fp.h` 和 `ulunas_modules.c` 必须与编译时版本一致

### 精度瓶颈
1. **GRU hidden state s16f15** → 子步骤 ~72 dB, 限制全链路
2. **cTFA Fusion TA×FA 两次 >>15** → 每层 ~3.5 dB 损失
3. **Decoder 5层级联** → 各层 iso 85-89 dB, 级联后 62 dB

### MASK 端到端 ~0 dB
根因: Decoder 62 dB 误差被 sigmoid 非线性 + BS 矩阵乘放大
优化: GRU s16f15→s32f20 可提升至 ~75 dB

---

## Q-Format 速查

| 格式 | 位宽 | 范围 | 精度 | 用途 |
|------|------|------|------|------|
| s32f20 | 32 | [-2048, ~2048] | 2^-20 | 层间激活值 |
| s16f15 | 16 | [-1, ~1] | 2^-15 | PCM, GRU隐状态 |
| u16f15 | 16 | [0, ~2] | 2^-15 | sigmoid输出, ERB权重 |
| u16f11 | 16 | [0, ~64] | 2^-11 | LN 1/sqrt(var) |

```
Q15 → Q20: x_q20 = x_q15 << 5 (exact)
Q20 → Q15: x_q15 = (x_q20 + 16) >> 5 (四舍五入)
Q15 × Q15: int64乘积 → (prod+16384)>>15 → Q15
```

---

## 下一步

- [ ] linux_api14: 实现 8k↔16k 上下采样 + LPF 抗混叠滤波器
- [ ] GRU s16f15→s32f20 精度提升 (Decoder 62→75 dB)
- [ ] 多帧 RNN 退化修复 (Frame 1+ 从 67→12 dB)
- [ ] X2000 平台交叉编译部署测试
