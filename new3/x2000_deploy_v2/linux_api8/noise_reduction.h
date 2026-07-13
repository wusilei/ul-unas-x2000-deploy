/* noise_reduction.h — UL-UNAS noise reduction interface */
#ifndef NOISE_REDUCTION_H
#define NOISE_REDUCTION_H

void noise_init(void);
void noise_deinit(void);
void noise_reduction(short *voiceIn, short *voiceOut, int n_samples);

#endif
