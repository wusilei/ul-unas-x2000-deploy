/**
 * ulunascfg.h — DSP/BIOS 5.31 Configuration (UL-UNAS / DM3730)
 * ==============================================================
 * 手工生成 (CCS3.3 tconf 不生成独立输出).
 *
 * 配置:
 *   TSK_Audio  prio=5  stack=36KB  (McBSP → ULUNAS NR → McBSP)
 *   TSK_Ctrl   prio=1  stack=4KB   (DSPLink 轮询, 100ms)
 *   HWI_INT6   EDMA McBSP RX 完成中断
 *   HWI_INT8   DSPLink 邮箱中断
 *   SEM        AUDIO_READY 音频帧就绪信号量
 *   L2         32KB SRAM + 32KB Cache
 */

#ifndef ULUNASCFG_H
#define ULUNASCFG_H

#include <std.h>
#include <tsk.h>
#include <sem.h>
#include <hwi.h>
#include <log.h>
#include <mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── 任务对象 ───────────────────────────────────────────────── */
extern far TSK_Obj TSK_Audio;
extern far TSK_Obj TSK_Ctrl;
extern far TSK_Obj TSK_idle;

/* ── 信号量 ─────────────────────────────────────────────────── */
extern far SEM_Obj AUDIO_READY;

/* ── BIOS 系统对象 ──────────────────────────────────────────── */
extern far LOG_Obj LOG_system;

#ifdef __cplusplus
}
#endif

#endif /* ULUNASCFG_H */
