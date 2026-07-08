# UL-UNAS C定点部署 — 会话上下文恢复

## 项目概述
将 UL-UNAS MATLAB 语音增强模型转为纯定点 C 推理，目标君正 X2000 (MIPS32R2, 无 FPU)。
MATLAB 源码在 `UL-UNAS_SE_FPversion_v2/`，C 代码在 `c_version/x2000_deploy_v2/`。

## 推理管线
STFT → log_gen → BM → Encoder(5层) → GDPRNN×2 → Decoder(5层+skip) → Sigmoid → BS → MASK

## 根因: MATLAB column-major vs C row-major 权重索引
extract_weights.m 用 data(:) 展平 .mat，保持 column-major。C 最初用 row-major 索引 → 所有权重被转置读取。

## 已修复 (git log --oneline)
- a728bf5: 所有权重索引 row→col-major (12个算子) + Decoder Qr 6处修正
- 6b91279: AffinePReLU weight/bias [C,W] 逐元素 col-major 索引
- 25e1691: cTFA FA reshape 转置修复
- 45a77ed: E0 cTFA FA/TA 子步骤 golden dumps

## 诊断状态 (Frame 0, E0层)
TConv: 114.6 dB PERFECT | TA: 51.4 dB WARN | FA: 38.9 dB FAIL

## 可用 golden
dump_matlab/frame0_e0_fa_{agg,gru0,gru1,fc,out}.bin (FA子步骤)
dump_matlab/frame0_e0_ta_{gru,out}.bin (TA子步骤)
dump_matlab/frame{0..4}_enc_e{0..4}.bin (Encoder层输出)
dump_matlab/frame{0..4}_{bm,dec,rnn1,rnn2,mask}.bin

## Q格式
Q20=int32×1M, Q15=int16×32768, Q14=int16×16384, Q13=int16×8192, Q12=int16×4096

## 关键文件
ulunas_fp.c — 所有算子 | ulunas_modules.c — Enc/Dec/GDPRNN | test_matlab_golden.c — 测试
qr_config.h — Qr值 | layer_dims.h — 维度 | diag_e0_fa.m — 诊断脚本

## Git: https://github.com/wusilei/ul-unas-x2000-deploy (main, 45a77ed)

## 下一步
1. VM pull + 用 FA golden 做 C 子步骤比对，定位 38.9dB
2. 检查 bn_func_sw 中 running_var 转型 (应为 (int64_t)running_var[c])
