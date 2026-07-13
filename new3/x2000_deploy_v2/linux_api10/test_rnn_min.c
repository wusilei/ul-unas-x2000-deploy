#include <stdio.h>
#include <stdlib.h>
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"
int main() {
    printf("Start...\n");
    int32_t x[33*8] = {0};
    int16_t y[33*8] = {0};
    printf("Running bigru_module...\n");
    bigru_module(x, 33, 4, 8,
        dpgrnn_0_intra_rnn_rnn1_weight_ih_l0, dpgrnn_0_intra_rnn_rnn1_bias_ih_l0,
        dpgrnn_0_intra_rnn_rnn1_weight_hh_l0, dpgrnn_0_intra_rnn_rnn1_bias_hh_l0,
        dpgrnn_0_intra_rnn_rnn1_weight_ih_l0_reverse, dpgrnn_0_intra_rnn_rnn1_bias_ih_l0_reverse,
        dpgrnn_0_intra_rnn_rnn1_weight_hh_l0_reverse, dpgrnn_0_intra_rnn_rnn1_bias_hh_l0_reverse,
        -13, -8, y);
    printf("Done!\n");
    return 0;
}
