# linux_api12 — 接续上下文 (2026-07-10)

## 当前 X2000 状态
- IP: 192.168.42.159
- 生产: linux_api11 (frozen)
- 开发: **linux_api12** (8k→16k 上采样 + WIN_INC=256, 待部署验证)

## 核心变更 vs linux_api11

```
Pipeline:  8k in → [2×上采样] → 16k STFT(WIN_INC=256) → NR → ISTFT → [2×降采样] → 8k out
              ↑ 线性插值                                          ↑ 均值
```

**动机**: WIN_INC=200 + 8k 直通 → DSP 时序敏感, 跨设备一致性差
**方案**: 移植 GTCRN 已验证的上下采样管线

## 关键参数
- WIN_INC=256, WIN_LEN=512, FRAME_IN=400
- 8k→16k: 线性插值 2× (prev+curr)/2
- 16k→8k: 均值降采样 (a+b)/2
- WOLA: 无需归一化 (Hann 窗 power-complementary)
- 开销: +0.2ms/frame

## 交叉编译 & 部署

```bash
make -C /media/sf_haidesi/haidesi/ul-unas-x2000-deploy/UL-UNAS_SE_FPversion_v2/c_version/x2000_deploy_v2/linux_api12 \
  CC=/home/a/work/mips-gcc720-glibc229/bin/mips-linux-gnu-gcc

ssh root@192.168.42.159 "killall ulunas_q15; sleep 1"
scp linux_api12/ulunas_q15 root@192.168.42.159:/data/tianhai/
ssh root@192.168.42.159 "cd /data/tianhai && nohup ./ulunas_q15 > /dev/null 2>&1 &"
```

## 信号控制 (不变)
```bash
kill -USR1 <pid>   # 切换降噪 ON/OFF
kill -USR2 <pid>   # 切换 AGC ON/OFF
```

## 待验证
- [ ] X2000 编译通过
- [ ] 实时运行 (RTF < 1.5)
- [ ] 降噪效果 vs linux_api11
- [ ] 跨设备稳定性
- [ ] 输出 RMS 校准
- [ ] Golden SNR
