# UL-UNAS X2000 Fixed-Point Deploy — 接续上下文 (CONTEXT.md)

## 快速启动

```bash
cd /media/sf_haidesi/haidesi/ul-unas-x2000-deploy/UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2
make clean && make pc
./test_matlab_golden dump_matlab/
```

## 工程结构

```
c_version/x2000_deploy_v2/
├── ulunas_fp.h/c            # 全部定点算子 (conv/bn/ln/gru/bigru/ctfa/shuffle)
├── ulunas_modules.c          # Encoder 5层 + Decoder 5层 + GDPRNN (2个RNN)
├── ulunas_infer.c            # 顶层推理管线
├── ulunas_lut.h/c            # sigmoid(1024点)/tanh(1024点)/log10(512点) LUT
├── ulunas_matlab_weights.h/c # 409个权重/偏置 (944KB)
├── qr_config.h               # 每层Qr移位参数
├── layer_dims.h              # 全部维度宏
├── test_matlab_golden.c      # Golden比对测试 (含E0子块诊断)
├── dump_matlab/              # MATLAB golden二进制 (1710文件, gitignored)
├── Makefile                  # gcc + MIPS交叉编译
└── CONTEXT.md                # 本文件
```

## 网络结构速览

```
STFT[257] → log_gen → BM[129] → Encoder(5层) → GDPRNN×2 → Decoder(5层) → sigmoid → BS[257] → MASK
```

- Encoder: XConv → XMB0 → XDWS0 → XMB1 → XDWS1
- Decoder: De_XDWS0 → De_XMB0 → De_XDWS1 → De_XMB1 → De_XConv (每层有skip connection)
- GDPRNN: Intra-RNN(BiGRU) → Inter-RNN(GRU), 每个×2组

## 当前 SNR (Frame 0, 2026-07-08)

| 子块 | SNR | 状态 |
|------|-----|------|
| BM | 124 dB | ✅ PERFECT |
| E0.tconv (conv+BN+AffinePReLU) | 114.6 dB | ✅ PASS |
| E0.ta (cTFA时间注意力) | 51.4 dB | ⚠️ WARN |
| E0.fa (cTFA频率注意力) | 38.9 dB | ❌ FAIL |
| E0.ctfa | 37.2 dB | ❌ FAIL |
| enc_e1~e4 | -12~-16 dB | ❌ |
| MASK | -21 dB | ❌ |

## 已修复 Bug (10个)

1. 所有权重索引 row→col-major (12个算子)
2. BN/LN per-channel 索引 (weight[i]→weight[c])
3. AffinePReLU weight/bias col-major
4. cTFA FA reshape 转置 bug (输入+输出)
5. conv2d_func 完全重写 (原版无H维度, pad=3硬编码)
6. gconv2d_func cache_w=Wout→Wx
7. gtconv2d/tconv/non_gtconv Wx 公式修正
8. Decoder Qr 值6处修正
9. Decoder 全5层实现
10. 5个ASAN内存溢出修复 (BN bias/FC bias/GRU x_gru/tconv x_insert)

## 下一步 (优先级排序)

1. 🔴 Frame 1+ 数据依赖 bug — E0.tconv从114.6→1.4dB (状态已排除缓存, 疑似conv2d负输入问题)
2. 🔴 E0 FA 38.9dB — 需MATLAB导出FA子步骤golden定位 (agg/gru_fwd/gru_rev/fc)
3. 🟡 E0 TA 51.4dB — GRU bias/sigmoid LUT精度
4. 🟡 E1-E4 ~= — E0修复后自动改善

## Git

- Remote: https://github.com/wusilei/ul-unas-x2000-deploy (branch: main)
- 本地: `/media/sf_haidesi/haidesi/ul-unas-x2000-deploy/UL-UNAS_SE_FPversion_v2`
- push 需在 VM 终端手动执行 (git push origin main)
