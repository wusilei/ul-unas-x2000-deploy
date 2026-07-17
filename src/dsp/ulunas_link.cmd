/* ================================================================
 * ulunas_link.cmd — DM3730 DSP Linker Script (UL-UNAS, DDR spill)
 * ================================================================
 * CGT v6.0.8, COFF output
 * L2: 64KB 全 SRAM (UL-UNAS static+stack ≈ 55KB, 必须全 SRAM 模式)
 * DDR: shared ARM-DSP, 64MB, 存放权重(965KB) + 溢出状态
 *
 * 与 GTCRN 版差异:
 *   - .ulunas_state (24.3KB) → DDR  (g_state: conv+tfa+inter caches)
 *   - .ulunas_ola   (3.0KB)  → DDR  (OLA 768×int32)
 *   - .ulunas_outfifo (4.0KB) → DDR  (Output FIFO 2048×int16)
 *   - L2 配置 64KB 全 SRAM (非 32+32 分拆)
 */

-c
-stack 0x4000
-heap  0x0800

MEMORY
{
    L2_SRAM  : origin = 0x00800000, len = 0x00010000   /* 64KB 全 SRAM */
    DDR      : origin = 0x80000000, len = 0x04000000   /* 64MB */
}

SECTIONS
{
    /* ── L2 SRAM (热数据, 每帧频繁访问) ── */
    .stack       > L2_SRAM
    .sysmem      > L2_SRAM
    .cio         > L2_SRAM
    .bss         > L2_SRAM   /* 不含 .ulunas_state/.ulunas_ola/.ulunas_outfifo 的剩余 .bss */

    /* ── DDR spill (冷数据, 每帧首尾访问一次) ── */
    .ulunas_state    > DDR   /* g_state: conv_cache(21.4KB) + tfa_cache(0.8KB) + inter_cache(2.1KB) */
    .ulunas_ola      > DDR   /* OLA buffer: 768 × int32 = 3.0KB */
    .ulunas_outfifo  > DDR   /* Output FIFO: 2048 × int16 = 4.0KB */

    /* ── DDR (权重 / 代码 / 常量) ── */
    .text        > DDR
    .cinit       > DDR
    .const       > DDR
    .switch      > DDR
    .data        > DDR
    .far         > DDR
}
