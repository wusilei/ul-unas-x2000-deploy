/* ================================================================
 * sim_link.cmd — CCS C64x+ Simulator Linker Script
 * ================================================================
 * VECTORS 专用于复位向量 (0x00800000-0x008001FF, 512B)
 * L2_SRAM 用于栈 (0x00800200+, rest of 64KB)
 */

-c
-stack 0x4000
-heap  0x2000

MEMORY
{
    VECTORS  : origin = 0x00800000, len = 0x00000200   /* 512B — 复位向量 */
    L2_SRAM  : origin = 0x00800200, len = 0x0000FE00   /* ~63KB — 栈 */
    DDR      : origin = 0x80000000, len = 0x10000000   /* 256MB */
}

SECTIONS
{
    .vectors     > VECTORS
    .stack       > L2_SRAM
    .text        > DDR
    .cinit       > DDR
    .const       > DDR
    .switch      > DDR
    .data        > DDR
    .bss         > DDR
    .far         > DDR
    .sysmem      > DDR
    .cio         > DDR
}
