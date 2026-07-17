#ifndef NOISE_REDUCTION_H
#define NOISE_REDUCTION_H
#ifdef __cplusplus
extern "C" {
#endif
void noise_init(void);
void noise_deinit(void);
void noise_reduction(short *voiceIn, short *voiceOut, int n_samples);
#ifdef __cplusplus
}
#endif
#endif
