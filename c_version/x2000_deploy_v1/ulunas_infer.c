/**
 * ulunas_infer.c — Full UL-UNAS Inference Pipeline
 * ================================================
 * Single-frame processing: log_gen → BM → Encoder → GDPRNN×2 → Decoder → Sigmoid → BS → MASK
 */

#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"

/* ================================================================
 * ulunas_state_init — Initialize state to zero
 * ================================================================ */
void ulunas_state_init(ulunas_state_t *state) {
    memset(state, 0, sizeof(ulunas_state_t));
}

/* ================================================================
 * ulunas_infer_frame — Single-frame UL-UNAS inference
 * ================================================================
 * Input:
 *   real_in, imag_in: float arrays of size N_BINS (257) — one STFT frame
 *   state: persistent model state (allocated by caller)
 *   erbfc_w: ERB forward weights (64×192) u16f15
 *   ierbfc_w: ERB inverse weights (192×64) u16f15
 * Output:
 *   crm_out: (2, 257) CRM output in s32f20
 */
void ulunas_infer_frame(const float *real_in, const float *imag_in,
                        ulunas_state_t *state,
                        const uint16_t *erbfc_w, const uint16_t *ierbfc_w,
                        int32_t *crm_out) {
    /* ---- Log-Magnitude Compression ---- */
    int32_t x_log[1 * N_BINS];
    log_gen_fixed(real_in, imag_in, N_BINS, x_log);

    /* ---- BM: Band Merging (1,257)→(1,129) ---- */
    int32_t x_bm[1 * N_BINS_BM];
    BM_fixed(x_log, erbfc_w, x_bm);

    /* ---- Encoder ---- */
    int32_t y_e0[CH_XCONV * N_BINS_MID];      /* (12, 65) */
    int32_t y_e1[CH_XMB0 * N_BINS_SMALL];      /* (24, 33) */
    int32_t y_e2[CH_XMB0 * N_BINS_SMALL];      /* (24, 33) */
    int32_t y_e3[CH_XMB1 * N_BINS_SMALL];      /* (32, 33) */
    int32_t y_e4[CH_XDWS1 * N_BINS_SMALL];     /* (16, 33) */
    Encoder_module(x_bm, state, y_e0, y_e1, y_e2, y_e3, y_e4);

    /* ---- GDPRNN ×2 ---- */
    int32_t y_rnn1[CH_XDWS1 * N_BINS_SMALL];   /* (16, 33) */
    int32_t y_rnn2[CH_XDWS1 * N_BINS_SMALL];   /* (16, 33) */
    GDPRNN_module(y_e4, state->inter_prev0, 0, y_rnn1);
    GDPRNN_module(y_rnn1, state->inter_prev1, 1, y_rnn2);

    /* ---- Decoder ---- */
    /* De_XConv outputs (1, 129) — single-channel mask shared between I/Q */
    int32_t y_dec[1 * N_BINS_BM];              /* (1, 129) */
    Decoder_module(y_rnn2, state, y_e0, y_e1, y_e2, y_e3, y_e4, y_dec);

    /* ---- Sigmoid: s32f20 → u16f15 ---- */
    uint16_t y_sig[1 * N_BINS_BM];
    sigmoid_fixed(y_dec, 1 * N_BINS_BM, y_sig);
    int16_t y_sig_s16[1 * N_BINS_BM];
    for (int i = 0; i < 1 * N_BINS_BM; i++) y_sig_s16[i] = (int16_t)y_sig[i];

    /* ---- BS: Band Splitting (1,129)→(1,257) ---- */
    int16_t y_bs[1 * N_BINS];
    BS_fixed(y_sig_s16, ierbfc_w, y_bs);

    /* ---- MASK: CRM application (same mask for I and Q) ---- */
    int32_t real_q[257], imag_q[257];
    for (int i = 0; i < N_BINS; i++) {
        real_q[i] = F2Q20(real_in[i]);
        imag_q[i] = F2Q20(imag_in[i]);
    }
    MASK_fixed(y_bs, real_q, imag_q, crm_out);
}
