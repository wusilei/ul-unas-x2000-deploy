#include "noise_reduction.h"
#include <string.h>
void noise_init(void) {}
void noise_deinit(void) {}
void noise_reduction(short *voiceIn, short *voiceOut, int n_samples) {
    memcpy(voiceOut, voiceIn, n_samples * sizeof(short));
}
