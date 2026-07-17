/**
 * dsp_link.c — DM3730 DSPLink 控制协议实现
 * ==========================================
 * 依赖: DSPLink 1.65 for DM3730 (需单独安装, CCS3.3 不含)
 *   - DSPLink 库路径: $(DSPLINK)/packages/ti/dsplink/inc/
 *   - Makefile 需追加: -I$(DSPLINK)/inc
 *
 * 协议: 8 字节定长帧
 *   [SYNC0(0xAA)] [SYNC1(0x55)] [CMD] [PAYLOAD 4B LE] [CRC8]
 *
 * 约束: 音频 PCM 绝对不走 DSPLink (保护控制信令实时性)
 *       仅承载: 降噪开关 / AGC 参数 / 状态查询
 */

#include "dsp_link.h"
#include <std.h>
#include <tsk.h>
#include <string.h>

/* ── DSPLink 1.65 头文件 (需单独安装 DSPLink 包) ── */
/* 标准安装路径: C:/dvsdk_3_01_00_10/dsplink_1_65_00_03/packages/ti/dsplink/inc/ */
#include <proc.h>       /* PROC_setup, PROC_destroy, PROC_getAttrs     */
#include <msgq.h>       /* MSGQ_transportOpen, MSGQ_locate, MSGQ_get  */
#include <pool.h>       /* POOL_open, POOL_alloc, POOL_free            */
#include <mpcs.h>       /* MPCS_init (多核临界区保护)                   */
#include <notify.h>     /* NOTIFY_register (可选: 事件通知)             */

/* ================================================================
 * 全局状态
 * ================================================================ */
volatile DspState g_dsp_state;

/* ── DSPLink 对象 ──────────────────────────────────────────────── */
static MSGQ_TransportObj g_mqt;
static MSGQ_Queue        g_local_queue;
static MSGQ_Queue        g_remote_queue;
static POOL_Obj          g_pool;
static Uint16            g_local_queue_id;
static Uint16            g_pool_id;

/* ── 接收缓冲 (在 DSPLink 共享内存池分配) ──────────────────────── */
#define RX_MSG_SIZE   (sizeof(MSGQ_MsgHeader) + LINK_FRAME_LEN)
#define MSG_POOL_SIZE 4

/* ── CRC8 表 (多项式 0x07) ──────────────────────────────────────── */
static const uint8_t crc8_table[256] = {
    0x00,0x07,0x0E,0x09,0x1C,0x1B,0x12,0x15,0x38,0x3F,0x36,0x31,0x24,0x23,0x2A,0x2D,
    0x70,0x77,0x7E,0x79,0x6C,0x6B,0x62,0x65,0x48,0x4F,0x46,0x41,0x54,0x53,0x5A,0x5D,
    0xE0,0xE7,0xEE,0xE9,0xFC,0xFB,0xF2,0xF5,0xD8,0xDF,0xD6,0xD1,0xC4,0xC3,0xCA,0xCD,
    0x90,0x97,0x9E,0x99,0x8C,0x8B,0x82,0x85,0xA8,0xAF,0xA6,0xA1,0xB4,0xB3,0xBA,0xBD,
    0xC7,0xC0,0xC9,0xCE,0xDB,0xDC,0xD5,0xD2,0xFF,0xF8,0xF1,0xF6,0xE3,0xE4,0xED,0xEA,
    0xB7,0xB0,0xB9,0xBE,0xAB,0xAC,0xA5,0xA2,0x8F,0x88,0x81,0x86,0x93,0x94,0x9D,0x9A,
    0x27,0x20,0x29,0x2E,0x3B,0x3C,0x35,0x32,0x1F,0x18,0x11,0x16,0x03,0x04,0x0D,0x0A,
    0x57,0x50,0x59,0x5E,0x4B,0x4C,0x45,0x42,0x6F,0x68,0x61,0x66,0x73,0x74,0x7D,0x7A,
    0x89,0x8E,0x87,0x80,0x95,0x92,0x9B,0x9C,0xB1,0xB6,0xBF,0xB8,0xAD,0xAA,0xA3,0xA4,
    0xF9,0xFE,0xF7,0xF0,0xE5,0xE2,0xEB,0xEC,0xC1,0xC6,0xCF,0xC8,0xDD,0xDA,0xD3,0xD4,
    0x69,0x6E,0x67,0x60,0x75,0x72,0x7B,0x7C,0x51,0x56,0x5F,0x58,0x4D,0x4A,0x43,0x44,
    0x19,0x1E,0x17,0x10,0x05,0x02,0x0B,0x0C,0x21,0x26,0x2F,0x28,0x3D,0x3A,0x33,0x34,
    0x4E,0x49,0x40,0x47,0x52,0x55,0x5C,0x5B,0x76,0x71,0x78,0x7F,0x6A,0x6D,0x64,0x63,
    0x3E,0x39,0x30,0x37,0x22,0x25,0x2C,0x2B,0x06,0x01,0x08,0x0F,0x1A,0x1D,0x14,0x13,
    0xAE,0xA9,0xA0,0xA7,0xB2,0xB5,0xBC,0xBB,0x96,0x91,0x98,0x9F,0x8A,0x8D,0x84,0x83,
    0xDE,0xD9,0xD0,0xD7,0xC2,0xC5,0xCC,0xCB,0xE6,0xE1,0xE8,0xEF,0xFA,0xFD,0xF4,0xF3
};

static uint8_t crc8(const uint8_t *data, int len) {
    uint8_t crc = 0x00;
    while (len--) crc = crc8_table[crc ^ *data++];
    return crc;
}

/* ================================================================
 * 命令处理
 * ================================================================ */

static void link_handle_cmd(uint8_t cmd, const uint8_t *payload) {
    switch (cmd) {
    case CMD_DENOISE_ONOFF:
        g_dsp_state.denoise_enable = (payload[0] != 0) ? 1 : 0;
        break;

    case CMD_AGC_ONOFF:
        g_dsp_state.agc_enable = (payload[0] != 0) ? 1 : 0;
        break;

    case CMD_AGC_THRESHOLD:
        /* payload[0..1] LE → 新阈值, CGT v6.0.8 safe */
        break;

    case CMD_GET_STATUS:
        /* 由 link_send_status() 发送回复 */
        break;

    default:
        break;
    }
}

/* ── 解析一帧 (8 字节) ─────────────────────────────────────────── */
static int link_parse_frame(const uint8_t *frame) {
    if (frame[0] != LINK_SYNC0 || frame[1] != LINK_SYNC1)
        return -1;
    if (crc8(frame, LINK_FRAME_LEN - 1) != frame[LINK_FRAME_LEN - 1])
        return -1;
    link_handle_cmd(frame[2], &frame[3]);
    return 0;
}

/* ── 发送回复帧 ────────────────────────────────────────────────── */
static void link_send_reply(uint8_t cmd, const uint8_t *payload) {
    MSGQ_Msg msg;
    uint8_t *frame;

    if (MSGQ_alloc(g_pool_id, (MSGQ_Msg *)&msg, RX_MSG_SIZE) != MSGQ_S_SUCCESS)
        return;

    frame = ((uint8_t *)msg) + sizeof(MSGQ_MsgHeader);
    frame[0] = LINK_SYNC0;
    frame[1] = LINK_SYNC1;
    frame[2] = cmd;
    frame[3] = payload[0];
    frame[4] = payload[1];
    frame[5] = payload[2];
    frame[6] = payload[3];
    frame[7] = crc8(frame, LINK_FRAME_LEN - 1);

    MSGQ_put(g_remote_queue, msg);
}

/* ── 发送状态报告 (CMD_GET_STATUS 回复) ──────────────────────────── */
static void link_send_status(void) {
    uint8_t payload[4];
    payload[0] = g_dsp_state.denoise_enable;
    payload[1] = g_dsp_state.agc_enable;
    payload[2] = (uint8_t)(g_dsp_state.frame_count & 0xFF);
    payload[3] = (uint8_t)((g_dsp_state.frame_count >> 8) & 0xFF);
    link_send_reply(CMD_GET_STATUS, payload);
}

/* ================================================================
 * DSPLink ISR (HWI_INT8: ARM 邮箱中断)
 * ================================================================ */

/**
 * DSPLink 邮箱中断服务例程
 * 仅在 ARM 发来消息时触发, 不参与音频数据搬运
 * 标记消息就绪, 由 TSK_Ctrl → dsp_link_poll() 处理
 */
interrupt void dsp_link_isr(void) {
    /* DSPLink 传输层自动处理中断,
     * 此处仅清除硬件中断标志即可.
     * 实际消息收取在 dsp_link_poll() 中完成 */
    NOTIFY_notify(0, 0);  /* 通知 ARM 侧事件 */
}

/* ================================================================
 * 公共接口
 * ================================================================ */

/**
 * dsp_link_init — 初始化 DSPLink MSGQ 通道
 *
 * 步骤:
 *   1. PROC_setup: 将 DSP 挂接到 DSPLink 处理器阵列
 *   2. POOL_open: 打开共享内存池 (SMAPOOL, ID=0)
 *   3. MSGQ_transportOpen: 打开 MSGQ 传输层 (SHM + 中断)
 *   4. MSGQ_locate: 定位 ARM 侧 "ARM2DSP" 队列
 *   5. MSGQ_open: 创建本地 "DSP2ARM" 队列供 ARM 定位
 *   6. NOTIFY_register: 注册邮箱中断回调
 */
void dsp_link_init(void) {
    PROC_Attrs        proc_attrs;
    MSGQ_Attrs        msgq_attrs;
    MSGQ_LocateAttrs  locate_attrs;
    NOTIFY_Attrs      notify_attrs;
    Int               status;

    memset(&g_dsp_state, 0, sizeof(g_dsp_state));

    /* 1. 挂接 DSP 到 DSPLink 处理器阵列 (DSP = proc 1) */
    PROC_getAttrs(PROC_GETDEFAULTS, &proc_attrs);
    status = PROC_setup(&proc_attrs);
    if (status != PROC_S_SUCCESS)
        return;

    /* 2. 打开共享内存池 */
    POOL_getAttrs(POOL_GETDEFAULTS, NULL);
    g_pool_id = POOL_makePoolId(POOL_INTERNALHEAP, 0);
    status = POOL_open(&g_pool, g_pool_id);
    if (status != POOL_S_SUCCESS)
        return;

    /* 3. 打开 MSGQ 传输层 */
    MSGQ_getAttrs(MSGQ_GETDEFAULTS, &msgq_attrs);
    msgq_attrs.poolId = g_pool_id;
    status = MSGQ_transportOpen(&g_mqt, &msgq_attrs);
    if (status != MSGQ_S_SUCCESS)
        return;

    /* 4. 定位 ARM 侧队列 (ARM 端创建 "ARM2DSP") */
    locate_attrs.timeout = (Uns)-1;  /* 无限等待 */
    status = MSGQ_locate("ARM2DSP", &g_remote_queue, &locate_attrs);
    if (status != MSGQ_S_SUCCESS)
        return;

    /* 5. 创建本地队列 (ARM 端通过 MSGQ_locate("DSP2ARM") 获取) */
    status = MSGQ_open("DSP2ARM", &g_local_queue, &msgq_attrs);
    if (status != MSGQ_S_SUCCESS)
        return;
    g_local_queue_id = (Uint16)g_local_queue;

    /* 6. 注册通知回调 (可选: 用于 DSP→ARM 主动通知) */
    NOTIFY_getAttrs(NOTIFY_GETDEFAULTS, &notify_attrs);
    NOTIFY_register(0, 0, 0, &notify_attrs);
}

/**
 * dsp_link_poll — 非阻塞收取并处理 DSPLink 消息
 *
 * 由 TSK_Ctrl 每 100ms 调用一次
 * 超时=0: 立即返回, 不阻塞音频任务
 * 禁止在此函数内执行耗时操作 (CRC8 O(8) 可忽略)
 */
void dsp_link_poll(void) {
    MSGQ_Msg msg;
    Int      status;

    /* 非阻塞收取 (timeout=0) */
    status = MSGQ_get(g_local_queue, (MSGQ_Msg *)&msg, 0);
    if (status != MSGQ_S_SUCCESS)
        return;

    /* 跳过 MSGQ 消息头, 定位到 8 字节协议帧 */
    uint8_t *frame = ((uint8_t *)msg) + sizeof(MSGQ_MsgHeader);

    if (link_parse_frame(frame) == 0) {
        /* CMD_GET_STATUS 需回复当前状态 */
        if (frame[2] == CMD_GET_STATUS)
            link_send_status();
    }

    /* 释放消息回共享内存池 */
    MSGQ_free(msg);
}
