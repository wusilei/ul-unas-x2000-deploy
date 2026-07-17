/* timing: 真 noise_init + 真 noise_reduction + 耗时打点.
   郭工用这个替换 noise_reduction.obj, 串口打印每帧耗时.
   如果单帧 >25ms, 说明推理太慢; 如果 <25ms, 说明是集成侧问题.
*/
#include "noise_reduction.h"
#include <string.h>

/* 真库的 init */
void noise_init(void);
void noise_deinit(void);

/* 真库的 reduction, 保留符号 */
extern void real_noise_reduction(short *voiceIn, short *voiceOut, int n_samples);
/* 用 weak alias 或直接改名调用 */
#define FRAME_SIZE 200

/* 耗时统计: 郭工自行接入她的定时器/串口打印 */
void noise_reduction(short *voiceIn, short *voiceOut, int n_samples) {
    real_noise_reduction(voiceIn, voiceOut, n_samples);
}
