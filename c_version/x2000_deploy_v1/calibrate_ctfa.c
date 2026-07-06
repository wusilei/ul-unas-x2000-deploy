/**
 * calibrate_ctfa.c — cTFA Q-format Calibration
 * =============================================
 * Calibrates cTFA_ta_module / cTFA_fa_module QR independently
 * for each encoder (e0-e4) and decoder (d0-d4) module.
 *
 * Golden files required in dump_matlab/:
 *   frame0_enc_e{0-4}_ctfa_in.bin  - cTFA input (PConv1 output)
 *   frame0_enc_e{0-4}_ctfa_ta.bin  - ta gate golden (uint16)
 *   frame0_enc_e{0-4}_ctfa_fa.bin  - fa gate golden (int32)
 *   frame0_enc_e{0-4}_ctfa_out.bin - cTFA output golden (before shuffle)
 *
 * Build: gcc -O2 -std=c99 -o calibrate_ctfa calibrate_ctfa.c
 *        ulunas_fp.c ulunas_matlab_weights.c -lm
 * Usage: ./calibrate_ctfa
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ulunas_fp.h"
#include "ulunas_matlab_weights.h"

static double snr_db(const int32_t *g, const int32_t *t, int n) {
    double s = 0, e = 0;
    for (int i = 0; i < n; i++) {
        double gv = g[i], d = gv - t[i];
        s += gv * gv; e += d * d;
    }
    return e < 1e-30 ? 999 : 10 * log10(s / e);
}

static double snr_db_u16(const uint16_t *g, const uint16_t *t, int n) {
    double s = 0, e = 0;
    for (int i = 0; i < n; i++) {
        double gv = g[i], d = gv - t[i];
        s += gv * gv; e += d * d;
    }
    return e < 1e-30 ? 999 : 10 * log10(s / e);
}

/* ================================================================
 * Encoder cTFA calibration data
 * ================================================================ */
typedef struct {
    const char *name;
    int C, W, hidden_dim;
    /* ta weights */
    const int16_t *ta_ih_w; const int32_t *ta_ih_b;
    const int16_t *ta_hh_w; const int32_t *ta_hh_b;
    const int16_t *ta_fc_w; const int32_t *ta_fc_b;
    /* fa weights */
    const int16_t *fa_ih_w; const int32_t *fa_ih_b;
    const int16_t *fa_hh_w; const int32_t *fa_hh_b;
    const int16_t *fa_re_ih_w; const int32_t *fa_re_ih_b;
    const int16_t *fa_re_hh_w; const int32_t *fa_re_hh_b;
    const int16_t *fa_fc_w; const int32_t *fa_fc_b;
    /* default QR */
    int ta_qr1, ta_qr2, ta_fc;
    int fa_qr1, fa_qr2, fa_fc;
} ctfa_module_t;

static const ctfa_module_t enc_modules[] = {
    {"enc_e0", 12, 65, CTA_XCONV_HID,
     encoder_en_convs_0_ops_4_ta_gru_weight_ih_l0, encoder_en_convs_0_ops_4_ta_gru_bias_ih_l0,
     encoder_en_convs_0_ops_4_ta_gru_weight_hh_l0, encoder_en_convs_0_ops_4_ta_gru_bias_hh_l0,
     encoder_en_convs_0_ops_4_ta_fc_weight, encoder_en_convs_0_ops_4_ta_fc_bias,
     encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0, encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0,
     encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0, encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0,
     encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0_reverse, encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0_reverse,
     encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0_reverse, encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0_reverse,
     encoder_en_convs_0_ops_4_fa_fc_weight, encoder_en_convs_0_ops_4_fa_fc_bias,
     -13, -8, -8,  -13, -8, -9},

    {"enc_e1", 24, 33, CTA_XMB0_HID,
     encoder_en_convs_1_pconv2_2_ta_gru_weight_ih_l0, encoder_en_convs_1_pconv2_2_ta_gru_bias_ih_l0,
     encoder_en_convs_1_pconv2_2_ta_gru_weight_hh_l0, encoder_en_convs_1_pconv2_2_ta_gru_bias_hh_l0,
     encoder_en_convs_1_pconv2_2_ta_fc_weight, encoder_en_convs_1_pconv2_2_ta_fc_bias,
     encoder_en_convs_1_pconv2_2_fa_gru_weight_ih_l0, encoder_en_convs_1_pconv2_2_fa_gru_bias_ih_l0,
     encoder_en_convs_1_pconv2_2_fa_gru_weight_hh_l0, encoder_en_convs_1_pconv2_2_fa_gru_bias_hh_l0,
     encoder_en_convs_1_pconv2_2_fa_gru_weight_ih_l0_reverse, encoder_en_convs_1_pconv2_2_fa_gru_bias_ih_l0_reverse,
     encoder_en_convs_1_pconv2_2_fa_gru_weight_hh_l0_reverse, encoder_en_convs_1_pconv2_2_fa_gru_bias_hh_l0_reverse,
     encoder_en_convs_1_pconv2_2_fa_fc_weight, encoder_en_convs_1_pconv2_2_fa_fc_bias,
     -13, -8, -8,  -13, -8, -9},

    {"enc_e2", 24, 33, CTA_XMB0_HID,
     encoder_en_convs_2_dconv_4_ta_gru_weight_ih_l0, encoder_en_convs_2_dconv_4_ta_gru_bias_ih_l0,
     encoder_en_convs_2_dconv_4_ta_gru_weight_hh_l0, encoder_en_convs_2_dconv_4_ta_gru_bias_hh_l0,
     encoder_en_convs_2_dconv_4_ta_fc_weight, encoder_en_convs_2_dconv_4_ta_fc_bias,
     encoder_en_convs_2_dconv_4_fa_gru_weight_ih_l0, encoder_en_convs_2_dconv_4_fa_gru_bias_ih_l0,
     encoder_en_convs_2_dconv_4_fa_gru_weight_hh_l0, encoder_en_convs_2_dconv_4_fa_gru_bias_hh_l0,
     encoder_en_convs_2_dconv_4_fa_gru_weight_ih_l0_reverse, encoder_en_convs_2_dconv_4_fa_gru_bias_ih_l0_reverse,
     encoder_en_convs_2_dconv_4_fa_gru_weight_hh_l0_reverse, encoder_en_convs_2_dconv_4_fa_gru_bias_hh_l0_reverse,
     encoder_en_convs_2_dconv_4_fa_fc_weight, encoder_en_convs_2_dconv_4_fa_fc_bias,
     -13, -8, -8,  -13, -8, -9},

    {"enc_e3", 32, 33, CTA_XMB1_HID,
     encoder_en_convs_3_pconv2_2_ta_gru_weight_ih_l0, encoder_en_convs_3_pconv2_2_ta_gru_bias_ih_l0,
     encoder_en_convs_3_pconv2_2_ta_gru_weight_hh_l0, encoder_en_convs_3_pconv2_2_ta_gru_bias_hh_l0,
     encoder_en_convs_3_pconv2_2_ta_fc_weight, encoder_en_convs_3_pconv2_2_ta_fc_bias,
     encoder_en_convs_3_pconv2_2_fa_gru_weight_ih_l0, encoder_en_convs_3_pconv2_2_fa_gru_bias_ih_l0,
     encoder_en_convs_3_pconv2_2_fa_gru_weight_hh_l0, encoder_en_convs_3_pconv2_2_fa_gru_bias_hh_l0,
     encoder_en_convs_3_pconv2_2_fa_gru_weight_ih_l0_reverse, encoder_en_convs_3_pconv2_2_fa_gru_bias_ih_l0_reverse,
     encoder_en_convs_3_pconv2_2_fa_gru_weight_hh_l0_reverse, encoder_en_convs_3_pconv2_2_fa_gru_bias_hh_l0_reverse,
     encoder_en_convs_3_pconv2_2_fa_fc_weight, encoder_en_convs_3_pconv2_2_fa_fc_bias,
     -13, -8, -8,  -13, -8, -9},

    {"enc_e4", 16, 33, CTA_XDWS1_HID,
     encoder_en_convs_4_dconv_4_ta_gru_weight_ih_l0, encoder_en_convs_4_dconv_4_ta_gru_bias_ih_l0,
     encoder_en_convs_4_dconv_4_ta_gru_weight_hh_l0, encoder_en_convs_4_dconv_4_ta_gru_bias_hh_l0,
     encoder_en_convs_4_dconv_4_ta_fc_weight, encoder_en_convs_4_dconv_4_ta_fc_bias,
     encoder_en_convs_4_dconv_4_fa_gru_weight_ih_l0, encoder_en_convs_4_dconv_4_fa_gru_bias_ih_l0,
     encoder_en_convs_4_dconv_4_fa_gru_weight_hh_l0, encoder_en_convs_4_dconv_4_fa_gru_bias_hh_l0,
     encoder_en_convs_4_dconv_4_fa_gru_weight_ih_l0_reverse, encoder_en_convs_4_dconv_4_fa_gru_bias_ih_l0_reverse,
     encoder_en_convs_4_dconv_4_fa_gru_weight_hh_l0_reverse, encoder_en_convs_4_dconv_4_fa_gru_bias_hh_l0_reverse,
     encoder_en_convs_4_dconv_4_fa_fc_weight, encoder_en_convs_4_dconv_4_fa_fc_bias,
     -13, -8, -8,  -13, -8, -9},
};

static const ctfa_module_t dec_modules[] = {
    {"dec_d0", 32, 33, CH_DEC_XDWS0,
     decoder_de_convs_0_dconv_4_ta_gru_weight_ih_l0, decoder_de_convs_0_dconv_4_ta_gru_bias_ih_l0,
     decoder_de_convs_0_dconv_4_ta_gru_weight_hh_l0, decoder_de_convs_0_dconv_4_ta_gru_bias_hh_l0,
     decoder_de_convs_0_dconv_4_ta_fc_weight, decoder_de_convs_0_dconv_4_ta_fc_bias,
     decoder_de_convs_0_dconv_4_fa_gru_weight_ih_l0, decoder_de_convs_0_dconv_4_fa_gru_bias_ih_l0,
     decoder_de_convs_0_dconv_4_fa_gru_weight_hh_l0, decoder_de_convs_0_dconv_4_fa_gru_bias_hh_l0,
     decoder_de_convs_0_dconv_4_fa_gru_weight_ih_l0_reverse, decoder_de_convs_0_dconv_4_fa_gru_bias_ih_l0_reverse,
     decoder_de_convs_0_dconv_4_fa_gru_weight_hh_l0_reverse, decoder_de_convs_0_dconv_4_fa_gru_bias_hh_l0_reverse,
     decoder_de_convs_0_dconv_4_fa_fc_weight, decoder_de_convs_0_dconv_4_fa_fc_bias,
     -13, -8, -9,  -12, -7, -9},

    {"dec_d1", 24, 33, CH_DEC_XMB0,
     decoder_de_convs_1_pconv2_2_ta_gru_weight_ih_l0, decoder_de_convs_1_pconv2_2_ta_gru_bias_ih_l0,
     decoder_de_convs_1_pconv2_2_ta_gru_weight_hh_l0, decoder_de_convs_1_pconv2_2_ta_gru_bias_hh_l0,
     decoder_de_convs_1_pconv2_2_ta_fc_weight, decoder_de_convs_1_pconv2_2_ta_fc_bias,
     decoder_de_convs_1_pconv2_2_fa_gru_weight_ih_l0, decoder_de_convs_1_pconv2_2_fa_gru_bias_ih_l0,
     decoder_de_convs_1_pconv2_2_fa_gru_weight_hh_l0, decoder_de_convs_1_pconv2_2_fa_gru_bias_hh_l0,
     decoder_de_convs_1_pconv2_2_fa_gru_weight_ih_l0_reverse, decoder_de_convs_1_pconv2_2_fa_gru_bias_ih_l0_reverse,
     decoder_de_convs_1_pconv2_2_fa_gru_weight_hh_l0_reverse, decoder_de_convs_1_pconv2_2_fa_gru_bias_hh_l0_reverse,
     decoder_de_convs_1_pconv2_2_fa_fc_weight, decoder_de_convs_1_pconv2_2_fa_fc_bias,
     -13, -8, -9,  -12, -7, -9},

    {"dec_d2", 24, 33, CH_DEC_XMB0,
     decoder_de_convs_2_dconv_4_ta_gru_weight_ih_l0, decoder_de_convs_2_dconv_4_ta_gru_bias_ih_l0,
     decoder_de_convs_2_dconv_4_ta_gru_weight_hh_l0, decoder_de_convs_2_dconv_4_ta_gru_bias_hh_l0,
     decoder_de_convs_2_dconv_4_ta_fc_weight, decoder_de_convs_2_dconv_4_ta_fc_bias,
     decoder_de_convs_2_dconv_4_fa_gru_weight_ih_l0, decoder_de_convs_2_dconv_4_fa_gru_bias_ih_l0,
     decoder_de_convs_2_dconv_4_fa_gru_weight_hh_l0, decoder_de_convs_2_dconv_4_fa_gru_bias_hh_l0,
     decoder_de_convs_2_dconv_4_fa_gru_weight_ih_l0_reverse, decoder_de_convs_2_dconv_4_fa_gru_bias_ih_l0_reverse,
     decoder_de_convs_2_dconv_4_fa_gru_weight_hh_l0_reverse, decoder_de_convs_2_dconv_4_fa_gru_bias_hh_l0_reverse,
     decoder_de_convs_2_dconv_4_fa_fc_weight, decoder_de_convs_2_dconv_4_fa_fc_bias,
     -13, -8, -8,  -13, -8, -9},

    {"dec_d3", 12, 65, CH_DEC_XMB1,
     decoder_de_convs_3_pconv2_2_ta_gru_weight_ih_l0, decoder_de_convs_3_pconv2_2_ta_gru_bias_ih_l0,
     decoder_de_convs_3_pconv2_2_ta_gru_weight_hh_l0, decoder_de_convs_3_pconv2_2_ta_gru_bias_hh_l0,
     decoder_de_convs_3_pconv2_2_ta_fc_weight, decoder_de_convs_3_pconv2_2_ta_fc_bias,
     decoder_de_convs_3_pconv2_2_fa_gru_weight_ih_l0, decoder_de_convs_3_pconv2_2_fa_gru_bias_ih_l0,
     decoder_de_convs_3_pconv2_2_fa_gru_weight_hh_l0, decoder_de_convs_3_pconv2_2_fa_gru_bias_hh_l0,
     decoder_de_convs_3_pconv2_2_fa_gru_weight_ih_l0_reverse, decoder_de_convs_3_pconv2_2_fa_gru_bias_ih_l0_reverse,
     decoder_de_convs_3_pconv2_2_fa_gru_weight_hh_l0_reverse, decoder_de_convs_3_pconv2_2_fa_gru_bias_hh_l0_reverse,
     decoder_de_convs_3_pconv2_2_fa_fc_weight, decoder_de_convs_3_pconv2_2_fa_fc_bias,
     -13, -8, -9,  -11, -6, -9},

    {"dec_d4", 1, 129, CH_DEC_XCONV,
     decoder_de_convs_4_ops_4_ta_gru_weight_ih_l0, decoder_de_convs_4_ops_4_ta_gru_bias_ih_l0,
     decoder_de_convs_4_ops_4_ta_gru_weight_hh_l0, decoder_de_convs_4_ops_4_ta_gru_bias_hh_l0,
     decoder_de_convs_4_ops_4_ta_fc_weight, decoder_de_convs_4_ops_4_ta_fc_bias,
     decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0, decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0,
     decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0, decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0,
     decoder_de_convs_4_ops_4_fa_gru_weight_ih_l0_reverse, decoder_de_convs_4_ops_4_fa_gru_bias_ih_l0_reverse,
     decoder_de_convs_4_ops_4_fa_gru_weight_hh_l0_reverse, decoder_de_convs_4_ops_4_fa_gru_bias_hh_l0_reverse,
     decoder_de_convs_4_ops_4_fa_fc_weight, decoder_de_convs_4_ops_4_fa_fc_bias,
     -13, -8, -8,  -13, -8, -9},
};

/* ================================================================
 * Parameterized cTFA block: ta → fa → apply
 * ================================================================ */
static void ctfa_block(const ctfa_module_t *m,
                       int ta_qr1, int ta_qr2, int ta_fc,
                       int fa_qr1, int fa_qr2, int fa_fc,
                       const int32_t *x_in, int32_t *y_out,
                       uint16_t *ta_gate, int32_t *fa_gate,
                       int16_t *ta_h) {
    cTFA_ta_module(x_in, m->C, m->W, m->hidden_dim,
                   m->ta_ih_w, m->ta_ih_b,
                   m->ta_hh_w, m->ta_hh_b,
                   m->ta_fc_w, m->ta_fc_b,
                   ta_h, ta_gate,
                   ta_qr1, ta_qr2, ta_fc);

    cTFA_fa_module(x_in, m->C, m->W,
                   m->fa_ih_w, m->fa_ih_b,
                   m->fa_hh_w, m->fa_hh_b,
                   m->fa_re_ih_w, m->fa_re_ih_b,
                   m->fa_re_hh_w, m->fa_re_hh_b,
                   m->fa_fc_w, m->fa_fc_b,
                   fa_gate,
                   fa_qr1, fa_qr2, fa_fc);

    cTFA_apply(x_in, ta_gate, fa_gate, m->C, m->W, y_out);
}

int main() {
    printf("=== cTFA QR Calibration ===\n\n");

    /* ================================================================
     * Encoder calibration
     * ================================================================ */
    printf("--- ENCODER cTFA ---\n");
    printf("%-10s %4s %3s %3s %3s | %4s %3s %3s %3s | %8s %8s %8s\n",
           "Module", "ta1","ta2","tfc","fa1","fa2","ffc","ta_SNR","fa_SNR","out_SNR");

    for (int mi = 0; mi < 5; mi++) {
        const ctfa_module_t *m = &enc_modules[mi];
        int total = m->C * m->W;

        /* Load golden files */
        int32_t *g_in = calloc(total, sizeof(int32_t));
        uint16_t *g_ta = calloc(m->C, sizeof(uint16_t));
        uint16_t tmp_fa[4096];  /* placeholder for fa golden read */
        int32_t *g_fa = calloc(total, sizeof(int32_t));
        int32_t *g_out = calloc(total, sizeof(int32_t));
        char path[256];

        snprintf(path, sizeof(path), "dump_matlab/frame0_%s_ctfa_in.bin", m->name);
        { FILE *f = fopen(path, "rb"); if(f) { fread(g_in, 4, total, f); fclose(f); } }
        snprintf(path, sizeof(path), "dump_matlab/frame0_%s_ctfa_ta.bin", m->name);
        { FILE *f = fopen(path, "rb"); if(f) { fread(g_ta, 2, m->C, f); fclose(f); } }
        snprintf(path, sizeof(path), "dump_matlab/frame0_%s_ctfa_fa.bin", m->name);
        { FILE *f = fopen(path, "rb"); if(f) { fread(g_fa, 4, total, f); fclose(f); } }
        snprintf(path, sizeof(path), "dump_matlab/frame0_%s_ctfa_out.bin", m->name);
        { FILE *f = fopen(path, "rb"); if(f) { fread(g_out, 4, total, f); fclose(f); } }

        /* Baseline */
        uint16_t ta_gate[m->C]; int32_t fa_gate[total];
        int16_t ta_h[m->hidden_dim]; memset(ta_h, 0, m->hidden_dim * sizeof(int16_t));
        int32_t y_out[total];
        ctfa_block(m, m->ta_qr1, m->ta_qr2, m->ta_fc, m->fa_qr1, m->fa_qr2, m->fa_fc,
                   g_in, y_out, ta_gate, fa_gate, ta_h);
        double base_ta = snr_db_u16(ta_gate, g_ta, m->C);
        double base_fa = snr_db(fa_gate, g_fa, total);
        double base_out = snr_db(y_out, g_out, total);

        /* Grid search: ta qr1∈[-18,-10], ta qr2∈[-12,-4], ta fc∈[-12,-4]
                        fa qr1∈[-18,-10], fa qr2∈[-12,-4], fa fc∈[-12,-4] */
        double best_ta = -999, best_fa = -999, best_out = -999;
        int bta1, bta2, btfc, bfa1, bfa2, bffc;

        /* Coarse search: step=2 first, then refine near best */
        for (int ta1 = -18; ta1 <= -10; ta1 += 2)
        for (int ta2 = -12; ta2 <= -4; ta2 += 2)
        for (int tfc = -12; tfc <= -4; tfc += 2)
        for (int fa1 = -18; fa1 <= -10; fa1 += 2)
        for (int fa2 = -12; fa2 <= -4; fa2 += 2)
        for (int ffc = -12; ffc <= -4; ffc += 2) {
            memset(ta_h, 0, m->hidden_dim * sizeof(int16_t));
            int32_t y[total]; uint16_t tg[m->C]; int32_t fg[total];
            ctfa_block(m, ta1, ta2, tfc, fa1, fa2, ffc, g_in, y, tg, fg, ta_h);
            double s_out = snr_db(y, g_out, total);
            if (s_out > best_out) {
                best_out = s_out;
                bta1 = ta1; bta2 = ta2; btfc = tfc;
                bfa1 = fa1; bfa2 = fa2; bffc = ffc;
                best_ta = snr_db_u16(tg, g_ta, m->C);
                best_fa = snr_db(fg, g_fa, total);
            }
        }

        /* Fine search: step=1 near best */
        for (int ta1 = bta1-1; ta1 <= bta1+1; ta1++)
        for (int ta2 = bta2-1; ta2 <= bta2+1; ta2++)
        for (int tfc = btfc-1; tfc <= btfc+1; tfc++)
        for (int fa1 = bfa1-1; fa1 <= bfa1+1; fa1++)
        for (int fa2 = bfa2-1; fa2 <= bfa2+1; fa2++)
        for (int ffc = bffc-1; ffc <= bffc+1; ffc++) {
            memset(ta_h, 0, m->hidden_dim * sizeof(int16_t));
            int32_t y[total]; uint16_t tg[m->C]; int32_t fg[total];
            ctfa_block(m, ta1, ta2, tfc, fa1, fa2, ffc, g_in, y, tg, fg, ta_h);
            double s_out = snr_db(y, g_out, total);
            if (s_out > best_out) {
                best_out = s_out;
                bta1 = ta1; bta2 = ta2; btfc = tfc;
                bfa1 = fa1; bfa2 = fa2; bffc = ffc;
                best_ta = snr_db_u16(tg, g_ta, m->C);
                best_fa = snr_db(fg, g_fa, total);
            }
        }

        printf("%-10s base: ta=%5.1f fa=%5.1f out=%5.1f | best: ta=%5.1f fa=%5.1f out=%5.1f Δ=%+.1f\n",
               m->name, base_ta, base_fa, base_out, best_ta, best_fa, best_out, best_out - base_out);
        printf("           (ta_qr=%d,%d,%d fa_qr=%d,%d,%d) → (ta_qr=%d,%d,%d fa_qr=%d,%d,%d)\n",
               m->ta_qr1, m->ta_qr2, m->ta_fc, m->fa_qr1, m->fa_qr2, m->fa_fc,
               bta1, bta2, btfc, bfa1, bfa2, bffc);

        free(g_in); free(g_ta); free(g_fa); free(g_out);
    }

    /* ================================================================
     * Decoder calibration (golden not yet exported for decoder cTFA)
     * ================================================================ */
    printf("\n--- DECODER cTFA ---\n");
    printf("(Golden not yet exported for decoder cTFA intermediates; skipping)\n");
    printf("To enable: add cTFA intermediate exports in export_all_layers.m ");
    printf("for decoder d0-d4, then re-run export_all_layers.m\n");

    return 0;
}
