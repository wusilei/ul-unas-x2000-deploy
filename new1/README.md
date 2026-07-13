# UL-UNAS MATLAB → C Fixed-Point Conversion — new1 Workspace

> **目标**: 基于 Main_infer.m 函数调用顺序，完成所有 MATLAB 浮点函数到纯 C 定点代码的精准转换
> **验证基准**: noisy_fileid_1.wav Frame 0
> **参考经验**: stft_fixed_point_guide.md (STFT 定点转换已验证 55.9 dB SNR)

---

## 一、目录结构

```
new1/
├── README.md                    ← 本文件
├── stft_fixed_point_guide.md    ← STFT 定点转换经验 (强制约束)
├── export_frame0_golden.m       ← MATLAB golden 数据导出脚本
├── verify_frame0.py             ← Python 验证脚本
├── stft/
│   ├── Main_infer.m             ← STFT 测试入口
│   ├── STFT_func.m              ← STFT 浮点参考
│   ├── stft_window.mat          ← Hann² 窗函数
│   └── test_wavs/
│       └── noisy_fileid_1.wav   ← 测试音频 (Frame 0)
└── ulunas_c/
    ├── ulunas_nr.c              ← ★ 主实现文件 (所有函数)
    ├── Makefile                 ← 编译脚本
    ├── ulunas_nr_ops.c          ← 核心算子 (已合并到 ulunas_nr.c)
    └── ulunas_nr_pipeline.c     ← 管线函数 (已合并到 ulunas_nr.c)
```

## 二、已完成 & 待完成

### ✅ 已完成 (STFT 部分)

| 函数 | 状态 | SNR | 说明 |
|------|------|-----|------|
| **STFT_func** | ✅ 完成 | 55.9 dB | Q15 FFT + Hann² 窗 + 镜像填充 |
| **FFT_q15_512** | ✅ 完成 | — | 9 级蝶形, >>15 per stage |
| **mirror_pad** | ✅ 完成 | — | 256 点镜像填充 |

### 🔶 算子已编写 (待权重数据验证)

| 函数 | 代码位置 | 行号 | 状态 |
|------|---------|------|------|
| **log_gen_fixed** | ulunas_nr.c | ~870 | ✅ 算子完成 |
| **bm_fixed** | ulunas_nr.c | ~890 | ✅ 算子完成 |
| **bs_fixed** | ulunas_nr.c | ~908 | ✅ 算子完成 |
| **mask_fixed** | ulunas_nr.c | ~929 | ✅ 算子完成 |
| **bn_func_sw/uw** | ulunas_nr.c | ~950 | ✅ 算子完成 |
| **affineprelu_func** | ulunas_nr.c | ~1000 | ✅ 算子完成 |
| **ln_func** | ulunas_nr.c | ~1061 | ✅ 算子完成 |
| **gru_module** | ulunas_nr.c | ~1110 | ✅ 算子完成 |
| **bigru_module** | ulunas_nr.c | ~1220 | ✅ 算子完成 |
| **ctfa_ta_module** | ulunas_nr.c | ~1260 | ✅ 算子完成 |
| **ctfa_fa_module** | ulunas_nr.c | ~1290 | ✅ 算子完成 |
| **shuffle_interleave** | ulunas_nr.c | ~1355 | ✅ 算子完成 |
| **shuffle_deinterleave** | ulunas_nr.c | ~1364 | ✅ 算子完成 |
| **pconv2d_func** | ulunas_nr.c | ~1379 | ✅ 算子完成 |
| **gconv2d_func** | ulunas_nr.c | ~1402 | ✅ 算子完成 |
| **conv2d_func** | ulunas_nr.c | ~1440 | ✅ 算子完成 |
| **non_gconv2d_func** | ulunas_nr.c | ~1476 | ✅ 算子完成 |
| **ctfa_fusion** | ulunas_nr.c | ~1500 | ✅ 算子完成 |

### 🔶 管线函数 (需完整权重数据)

| 函数 | 代码位置 | 状态 |
|------|---------|------|
| **encoder_layer0_xconv** | ulunas_nr.c | ✅ 完整实现 (需权重) |
| **encoder_layer1_xmb0** | — | ⬜ 需实现 |
| **encoder_layer2_xdws0** | — | ⬜ 需实现 |
| **encoder_layer3_xmb1** | — | ⬜ 需实现 |
| **encoder_layer4_xdws1** | — | ⬜ 需实现 |
| **intra_rnn_module** | ulunas_nr.c | 🔶 框架 (需权重) |
| **inter_rnn_module** | ulunas_nr.c | 🔶 框架 (需权重) |
| **gdprnn_module** | ulunas_nr.c | 🔶 框架 (需权重) |
| **decoder_layer0-4** | ulunas_nr.c | ⬜ 需实现 |

### ⬜ 待完成

| 函数 | MATLAB 文件 | 优先级 |
|------|------------|--------|
| **XMB0_PConv_block_0/1** | XMB0_PConv_block_*.m | P0 |
| **XMB0_TConv_block** | XMB0_TConv_block.m | P0 |
| **XDWS0_PConv_block** | XDWS0_PConv_block.m | P0 |
| **XDWS0_TConv_block** | XDWS0_TConv_block.m | P0 |
| **XMB1_PConv_block_0/1** | XMB1_PConv_block_*.m | P0 |
| **XMB1_nonTConv_block** | XMB1_nonTConv_block.m | P0 |
| **XDWS1_PConv_block** | XDWS1_PConv_block.m | P0 |
| **XDWS1_nonTConv_block** | XDWS1_nonTConv_block.m | P0 |
| **All Decoder sub-blocks** | De_*_block.m (15 files) | P0 |
| **ISTFT_func** | ISTFT_func.m | P1 |
| **Sigmoid 定点** | sigmoid_func.m | ✅ LUT 已有 |

## 三、快速开始

### Step 1: 导出 MATLAB Golden 数据

```matlab
% 在 MATLAB 中运行 (从 UL-UNAS_SE_FPversion_v2/ 目录)
cd /media/sf_haidesi/haidesi/ul-unas-x2000-deploy/new1/
run('export_frame0_golden.m')
```

输出: `dump_matlab/frame0_*.bin` (16 个 golden 文件)

### Step 2: 编译 C 代码

```bash
cd /media/sf_haidesi/haidesi/ul-unas-x2000-deploy/new1/ulunas_c/
make test
```

### Step 3: 运行验证

```bash
# 运行 C 测试 (STFT 部分已验证)
./ulunas_nr_test ../stft/test_wavs/noisy_fileid_1.wav

# Python 验证脚本
cd ..
python3 verify_frame0.py dump_matlab/
```

### Step 4: 完整管线编译 (需要权重数据)

```bash
# 链接成熟项目的权重数据
cd ulunas_c/
make full
```

## 四、Q-Format 约定 (严格遵循 Fix_point.m)

```
激活值:
  s32f20: 主激活 (Q_ACT=20) — conv I/O, BN 中间值
  u32f20: log_gen 输出, cTFA 聚合
  s16f15: GRU hidden state, tanh 输出
  u16f15: sigmoid 输出, attention masks
  u16f11: LN 1/sqrt(var)

权重:
  s16f14: Conv/DeConv/PConv/BN/AffinePReLU 权重
  s16f13: GConv/nonGConv/TConv/FC/TA_FC/DPRNN_FC 权重
  s16f12: GRU/LN 权重
  s16f10: GRU bias
  s32f20: FC bias, BN bias, BN running_mean
  u16f14: BN running_var
  u16f15: ERB merge/split weights

关键转换:
  float → Q15: round(x * 32768)
  float → Q20: round(x * 1048576)
  Q15 → Q20:  x << 5 (exact)
  Q20 → Q15:  (x + 16) >> 5
```

## 五、验证标准

| 层类型 | 目标 SNR | GTCRN 经验 |
|--------|---------|-----------|
| STFT (Q15 FFT) | > 55 dB | 55.9 dB ✅ |
| BM/BS (矩阵乘) | > 130 dB | bit-exact |
| Conv2D + BN + PReLU | > 80 dB | GTCRN: 100+ dB |
| GRU/BiGRU | > 70 dB | soft-float 限制 |
| LN | > 60 dB | int64 定点化后 |
| MASK | > 80 dB | 纯逐元素乘法 |
| 全管线 (单帧) | > 55 dB | GTCRN: 65.9 dB |

## 六、与成熟项目的对接

成熟实现位于:
```
../UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/
├── ulunas_fp.c/h          ← 核心算子 (参考实现)
├── ulunas_modules.c       ← 模块组装 (参考实现)
├── ulunas_matlab_weights.c/h ← 409 权重数组 (可直接引用)
├── ulunas_lut.c/h         ← LUT 表 (已内嵌到 ulunas_nr.c)
├── ulunas_infer.c         ← 顶层推理
├── layer_dims.h           ← 网络维度 (已内嵌)
├── qr_config.h            ← Qr 参数 (已内嵌)
└── ulunas_nr.c            ← STFT/WOLA 管线
```

当前 new1 工作区的 ulunas_nr.c 是完全独立的重写版本，但可以通过 `#include` 复用成熟项目的权重数据以完成全管线验证。

## 七、下一步

1. **运行 MATLAB golden export** → 生成 Frame 0 各级 golden 数据
2. **完成 Encoder Layer 1-4** → 参考 XMB0_module.m 等 MATLAB 文件
3. **完成 Decoder Layer 0-4** → 参考 De_*_module.m
4. **完成 GDPRNN** → 参考 Intra/Inter_RNN_module.m
5. **逐层验证** → 每完成一层, C vs MATLAB SNR 对比
6. **全管线验证** → 单帧端到端 SNR
7. **ISTFT** → 输出增强音频
