# new3 — GRU s16f15 → s32f20 精度提升

> 2026-07-13 基于 new2 (V2 Q15 GRU) 的 GRU 隐状态精度升级
> 冻结 new2，所有改动仅在 new3 进行

---

## 改动概述

将 GRU 隐状态从 `int16_t` (s16f15) 升级为 `int32_t` (s32f20)：

| 项目 | V2 (gru_module) | new3 (gru_module_q20) |
|------|----------------|----------------------|
| h_cache | int16_t Q15 | int32_t Q20 |
| HH MAC 移位 | >>8 (Q15×Q12=Q27) | >>13 (Q20×Q12=Q32) |
| sigmoid 输出 | u16f15 | u32f20 |
| tanh 输出 | s16f15 | s32f20 |
| 门控混合移位 | >>15 | >>20 |
| 1.0 常量 | 32768 | 1048576 |
| 隐状态更新 | clamp_i16 → Q15 | sat_s20 → Q20 |
| 输出转换 | 直接 h_cache (Q15) | (hn+16)>>5 → Q15 |

**权重格式不变**（ih_weight/hh_weight 仍为 int16_t Q12），仅升级隐状态精度。

---

## 文件结构

```
new3/
├── README.md                           ← 本文件
└── x2000_deploy_v2/                    ← 基于 V2 代码库，修改如下：
    ├── ulunas_fp.h                     ← GRU 缓存 int16_t→int32_t，新增 Q20 声明
    ├── ulunas_fp.c                     ← 新增 gru_module_q20、bigru_module_q20、Q20 LUT
    ├── ulunas_modules.c                ← cTFA/Inter-RNN 改用 gru_module_q20
    ├── ulunas_infer.c                  ← 无需改动（通过模块函数间接使用 Q20）
    ├── dump_full_pipeline.c            ← 主路径通过模块函数使用 Q20
    ├── Makefile                        ← 新增 dump_full 目标
    └── ... (其他文件与 V2 相同)
```

---

## 快速开始

### 1. 编译 (VM: 192.168.56.101)

```bash
ssh a@192.168.56.101
cd /media/sf_haidesi/haidesi/ul-unas-x2000-deploy/new3/x2000_deploy_v2
make clean && make pc
```

### 2. 全管线 dump

```bash
./dump_full dump_matlab
```

### 3. 逐 bin 对比 (复用 new2 脚本)

```bash
cd /media/sf_haidesi/haidesi/ul-unas-x2000-deploy/new2
python3 dump_per_bin.py ../new3/x2000_deploy_v2/dump_matlab ../new3/x2000_deploy_v2/dump_c
```

---

## 预期提升

| 层级 | V2 SNR | new3 预期 SNR | 提升 |
|------|--------|-------------|------|
| Encoder E0 | 67.5 dB | ~75 dB | +7.5 |
| Encoder E1-E4 | 64-65 dB | ~72 dB | +7 |
| RNN1 | 67.5 dB | ~75 dB | +7.5 |
| RNN2 | 64.6 dB | ~72 dB | +7 |
| **Decoder** | **62.0 dB** | **~75 dB** | **+13** |
| MASK (端到端) | ~0 dB | ~50 dB | +50 |

> 瓶颈从 GRU s16f15 转移到 cTFA Fusion TA×FA 两次>>15 (~67 dB)

---

## 关键改动点

1. **`ulunas_fp.h:143-159`** — ulunas_state_t GRU 缓存 int16_t→int32_t
2. **`ulunas_fp.c:855-990`** — gru_module_q20() + bigru_module_q20()
3. **`ulunas_fp.c:1085-1146`** — ctfa_ta_module() h_cache int16_t→int32_t
4. **`ulunas_modules.c:699-803`** — inter_rnn_module() h_cache+局部变量 int16_t→int32_t

---

## 与 V2 兼容性

- 旧 `gru_module()` 和 `bigru_module()` 保留不动
- dump_full_pipeline.c 的诊断内联代码仍使用旧 Q15 GRU（不影响 dump 输出）
- 输出 y[] 格式不变 (int16_t Q15)，调用方无需改动
- 权���文件完全相同

---

## 下一步

- [ ] VM 编译验证
- [ ] 全管线 SNR 对比 (new2 baseline vs new3)
- [ ] 如果 Decoder < 70 dB，检查 GRU Q20 LUT 精度 (当前 Q15→Q20 <<5 可能不够)
- [ ] linux_api14 LPF 实现
- [ ] X2000 交叉编译测试
