# linux_api11 — 接续上下文 (2026-07-10)

## 当前 X2000 状态
- IP: 192.168.42.159
- 运行: linux_api11, PID 6987
- 模式: NOISE=OFF, AGC=ON (降噪关，AGC开 → 直通+AGC)
- GTCRN 备用: `/data/tianhai/denoise_v19_q15`

## 版本演进
| 版本 | 关键特征 | 状态 |
|------|---------|------|
| linux_api | v5 baseline, WIN_INC=256 | 存档 |
| linux_api1 | v6, WIN_INC=200 | 存档 |
| linux_api9 | 三级门控 AGC (200/30), 混合前后置 | 生产基准 |
| linux_api10 | = linux_api9 + 诊断工具 | 验证用 |
| **linux_api11** | **纯后置 AGC, agc_gate_decision() API** | **当前** |

## linux_api11 AGC 架构
- `energy_calculate_and_smooth_s16()` — 能量检测 (LongXueKun)
- `agc_gate_decision(raw_energy, &goal, &gain_num, &gain_den)` — 三级门控 (200/30)
- `voice_NoiseReductionAndAGC(in, out, goal, flag, n_samples)`
  - state_flag=0: bypass
  - state_flag=1: NR only
  - state_flag=2: AGC only
  - state_flag=3: NR→AGC (纯后置)
- 前置 1.5× 增益在 ulunas_linux.c 外部，仅近场/远场启用

## nn_infer 框架
- 位置: `/home/a/work/nn_infer/` (完整版)
- 架构: thin wrapper 封装 ulunas_fp.c + ulunas_lut.c
- 验证: log_gen/BM/LN/sigmoid/gru 全部 bit-exact (999 dB)
- 单元测试: 50/50 PASS

## Golden 逐层 SNR (已知天花板)
| 层 | SNR | 根因 |
|----|-----|------|
| BM | 84 dB | bit-exact |
| Encoder E0-E4 | 29-37 dB | GRU int16 Q12 权重量化 |
| GDPRNN | 7-8 dB | GRU → LN 误差累积 |
| Decoder | 5-12 dB | 上游累积 |
| GRU 天花板 | 38 dB | int16 权重物理极限 |
| LN 天花板 | ~0 dB | uint16 clamp |

## 关键决策
1. GRU 38 dB 天花板已接受（int16 权重量化限制）
2. nn_infer 改为 wrapper 模式（不重写，封装已验证实现）
3. 逐层 golden 验证用于算法正确性证明，不追求 80+ dB
4. WIN_INC=200 管线对 DSP 时序敏感，不同 X2000 可能需要适配

## 项目路径
- v2 生产: `UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/`
- v3 开发: `UL-UNAS_SE_FPversion_v3/UL-UNAS_SE_FPversion_v2/x2000_deploy_v2/`
- golden: `dump_matlab/` (v2)
- 权重: `para_in_mat_FP/` → `extract_weights.m` → `ulunas_matlab_weights.c/h`
- nn_infer: `/home/a/work/nn_infer/`

## 常用命令
```bash
# 编译部署
make -C linux_api11 CC=/home/a/work/mips-gcc720-glibc229/bin/mips-linux-gnu-gcc
scp linux_api11/ulunas_q15 root@192.168.42.159:/data/tianhai/

# X2000 信号控制
kill -USR1 <pid>  # 切换降噪
kill -USR2 <pid>  # 切换 AGC
killall ulunas_q15; cd /data/tianhai && nohup ./ulunas_q15 > /dev/null 2>&1 &  # 重启
./ulunas_q15 bypass &  # 旁路直通

# 切换 GTCRN
killall ulunas_q15; cd /data/tianhai && nohup ./denoise_v19_q15 > /dev/null 2>&1 &

# Golden 对比
cd x2000_deploy_v2/dump_matlab
../linux_api10/test_snr_v2 ../dump_matlab
```
