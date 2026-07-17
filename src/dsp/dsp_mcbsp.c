/**
 * dsp_mcbsp.c — DM3730 DSP McBSP + EDMA 音频驱动
 * ===============================================
 * McBSP0: 8kHz 16-bit I2S 输入 (DSP 独占, ARM 不参与)
 * McBSP1: 8kHz 16-bit I2S 输出
 * EDMA:  Ping-Pong 双缓冲, 每满 FRAME_SIZE 点触发中断
 *
 * 使用 CSL v2 (CCS3.3 内置 C6000 CSL):
 *   - MCBSP_configArgs() 裸寄存器配置
 *   - EDMA_open() / EDMA_config()
 *
 * TODO: 实机调试时替换为 DM3730 EDMA3 LLD + CSL v3
 *
 * 约束: 音频 PCM 绝对不走 DSPLink (保护控制信令实时性)
 * 零浮点, 纯整数, CGT v6.0.8 兼容
 */

#include "dsp_mcbsp.h"
#include <std.h>
#include <tsk.h>
#include <sem.h>
#include <hwi.h>
#include <csl.h>
#include <csl_mcbsp.h>
#include <csl_edma.h>
#include <string.h>

/* ── 环形缓存 ──────────────────────────────────────────────── */
static int16_t g_rx_ring[RING_BUF_SZ];
static volatile int g_rx_wr;
static int g_rx_rd;
static volatile int g_rx_count;

static int16_t g_tx_ping[PING_PONG_SZ];
static int16_t g_tx_pong[PING_PONG_SZ];
static volatile int g_tx_bank;

static SEM_Handle g_sem_audio_ready;

static MCBSP_Handle hMcbsp0;
static MCBSP_Handle hMcbsp1;
static EDMA_Handle  hEdmaRx;
static EDMA_Handle  hEdmaTx;

/* ================================================================
 * McBSP 寄存器裸值 (CSL v2 MCBSP_configArgs)
 *
 * McBSP0 (RX): I2S slave, 16-bit, 单相帧, 双 slot, 8kHz
 * McBSP1 (TX): I2S slave, 16-bit, 单相帧, 双 slot, 8kHz
 *
 * SPCR: Serial Port Control Register
 * RCR:  Receive Control Register
 * XCR:  Transmit Control Register
 * SRGR: Sample Rate Generator Register
 * PCR:  Pin Control Register
 * ================================================================ */

/* SPCR: 复位状态, 使能 RX/TX */
#define SPCR_RX_ENABLE  0x00000001  /* RRST=1, 接收就绪 */
#define SPCR_TX_ENABLE  0x00010000  /* XRST=1, 发送就绪 */

/* RCR/XCR: 单相帧, 16-bit word, 2 slot, I2S 格式 (延迟 1 bit) */
#define RCR_CONFIG      0x00010040  /* RPHASE=0(单相), RFRLEN1=0(1word), RWDLEN1=2(16bit), RCOMPAND=0, RFIG=0, RDATDLY=1 */
#define XCR_CONFIG      0x00010040  /* 同上 TX 侧 */

/* SRGR: Sample Rate Generator, CLKGDV=0 (外部时钟), FSGM=0 */
#define SRGR_CONFIG     0x00000000  /* 由外部 Codec 提供时钟 */

/* PCR: CLKXM=0, CLKRM=0 (外部时钟), FSXM=0, FSRM=0 (外部帧同步) */
#define PCR_CONFIG      0x00000000  /* I2S slave mode */

static void mcbsp_init_rx(void) {
    hMcbsp0 = MCBSP_open(0, MCBSP_OPEN_RESET);
    MCBSP_configArgs(hMcbsp0,
        SPCR_RX_ENABLE,     /* spcr */
        RCR_CONFIG,         /* rcr */
        XCR_CONFIG,         /* xcr */
        SRGR_CONFIG,        /* srgr */
        0,                  /* mcr */
        0,0,0,0,            /* rcere0..3 (multi-channel, unused) */
        0,0,0,0,            /* xcere0..3 (multi-channel, unused) */
        PCR_CONFIG);        /* pcr */
    MCBSP_start(hMcbsp0,
        MCBSP_XMIT_START | MCBSP_RCV_START,
        2);  /* 2 cycle sample rate delay (I2S standard) */
}

static void mcbsp_init_tx(void) {
    hMcbsp1 = MCBSP_open(1, MCBSP_OPEN_RESET);
    MCBSP_configArgs(hMcbsp1,
        SPCR_TX_ENABLE,     /* spcr */
        0,                  /* rcr (TX only) */
        XCR_CONFIG,         /* xcr */
        SRGR_CONFIG,        /* srgr */
        0,                  /* mcr */
        0,0,0,0,            /* rcere0..3 (multi-channel, unused) */
        0,0,0,0,            /* xcere0..3 (multi-channel, unused) */
        PCR_CONFIG);        /* pcr */
    MCBSP_start(hMcbsp1,
        MCBSP_XMIT_START | MCBSP_RCV_START,
        2);
}

/* ================================================================
 * EDMA (CSL v2)
 * TODO: 替换为 DM3730 EDMA3 LLD PaRAM 配置
 * ================================================================ */
static void edma_init(void) {
    hEdmaRx = EDMA_open(EDMA_CHA_ANY, EDMA_OPEN_RESET);
    hEdmaTx = EDMA_open(EDMA_CHA_ANY, EDMA_OPEN_RESET);
    /* PaRAM entry 配置: SRC=McBSP DRR, DST=g_rx_ring, etc. */
    /* 实机调试时用 EDMA3 LLD 的 EDMA3_DRV_requestChannel + PaRAM */
}

/* ── EDMA 完成中断 ─────────────────────────────────────────── */
interrupt void edma_rx_isr(void) {
    g_rx_wr = (g_rx_wr + PING_PONG_SZ) % RING_BUF_SZ;
    g_rx_count += PING_PONG_SZ;
    SEM_post(g_sem_audio_ready);
    EDMA_intClear(0);
}

/* ================================================================
 * 公共接口
 * ================================================================ */

void dsp_mcbsp_init(void) {
    g_rx_wr = g_rx_rd = g_rx_count = 0;
    g_tx_bank = 0;
    memset(g_rx_ring, 0, sizeof(g_rx_ring));
    memset(g_tx_ping, 0, sizeof(g_tx_ping));
    memset(g_tx_pong, 0, sizeof(g_tx_pong));
    g_sem_audio_ready = SEM_create(0, NULL);
    mcbsp_init_rx();
    mcbsp_init_tx();
    edma_init();
}

void dsp_mcbsp_read(int16_t *buf) {
    while (g_rx_count < FRAME_SIZE)
        SEM_pend(g_sem_audio_ready, SYS_FOREVER);
    for (int i = 0; i < FRAME_SIZE; i++) {
        buf[i] = g_rx_ring[g_rx_rd];
        g_rx_rd = (g_rx_rd + 1) % RING_BUF_SZ;
    }
    g_rx_count -= FRAME_SIZE;
}

void dsp_mcbsp_write(const int16_t *buf) {
    int16_t *dst = g_tx_bank ? g_tx_pong : g_tx_ping;
    memcpy(dst, buf, FRAME_SIZE * sizeof(int16_t));
    g_tx_bank = 1 - g_tx_bank;
}
