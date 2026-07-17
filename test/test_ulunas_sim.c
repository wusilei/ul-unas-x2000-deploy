/**
 * test_ulunas_sim.c — UL-UNAS C64x+ Simulator Test Harness
 * =========================================================
 * 在 CCS3.3 C64x+ Simulator 中运行, 使用 CIO 访问主机文件.
 *
 * 与 GTCRN 版差异:
 *   - noise_reduction(pcm_in, pcm_out, FRAME_SIZE) — 3 参数版本
 *   - 8kHz 直通 (无上采样), WIN_INC=256 (32ms hop), WOLA 合成
 *   - 更大模型: 965KB 权重, 409 张量
 *
 * 用法:
 *   1. 编译: cl6x -mv64plus -O3 -fg ... -z -o test_ulunas.out *.obj rts64plus.lib
 *   2. CCS3.3: File→Load Program→test_ulunas.out
 *   3. 输入:  test_noisy.pcm (8kHz 16-bit mono, 复制到 CCS 目录)
 *   4. F5 Run
 *   5. 输出:  test_denoised.pcm (降噪后)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "noise_reduction.h"

#define FRAME_SIZE 200   /* 25ms @ 8kHz */

int main(void) {
    int16_t pcm_in[FRAME_SIZE];
    int16_t pcm_out[FRAME_SIZE];
    long    frame_count = 0;
    long    total_samples = 0;
    size_t  nread;
    FILE   *fin, *fout;

    printf("[ULUNAS Sim] Opening files...\n");

    fin = fopen("test_noisy.pcm", "rb");
    if (fin == NULL) {
        printf("ERROR: Cannot open test_noisy.pcm\n");
        return 1;
    }

    fout = fopen("test_denoised.pcm", "wb");
    if (fout == NULL) {
        printf("ERROR: Cannot create test_denoised.pcm\n");
        fclose(fin);
        return 2;
    }

    printf("[ULUNAS Sim] Initializing noise reduction (965KB weights, 409 tensors)...\n");
    noise_init();

    printf("[ULUNAS Sim] Processing frames (200 samples = 25ms @ 8kHz)...\n");

    /* 逐帧处理: 200 采样/帧, 8kHz 直通, WOLA 合成 */
    while ((nread = fread(pcm_in, sizeof(int16_t), FRAME_SIZE, fin)) > 0) {
        if (nread < FRAME_SIZE) {
            memset(pcm_in + nread, 0,
                   (FRAME_SIZE - nread) * sizeof(int16_t));
        }

        /* UL-UNAS: 3 参数版本, AGC + WOLA 内部处理 */
        noise_reduction(pcm_in, pcm_out, FRAME_SIZE);

        fwrite(pcm_out, sizeof(int16_t), FRAME_SIZE, fout);

        total_samples += FRAME_SIZE;
        frame_count++;

        if ((frame_count % 100) == 0)
            printf("[ULUNAS Sim] %ld frames (%.1f s)\n",
                   frame_count, total_samples / 8000.0);
    }

    fclose(fin);
    fclose(fout);

    printf("[ULUNAS Sim] Complete: %ld frames, %ld samples (%.1f s @ 8kHz)\n",
           frame_count, total_samples, total_samples / 8000.0);
    return 0;
}
