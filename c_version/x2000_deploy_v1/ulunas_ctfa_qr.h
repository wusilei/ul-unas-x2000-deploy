/**
 * ulunas_ctfa_qr.h — Global cTFA QR Configuration
 * ===============================================
 * Allows calibration tools to override cTFA QRs without
 * duplicating decoder module code.
 *
 * When QR_CALIBRATION_MODE is defined, decoder modules read
 * QRs from these globals instead of hardcoded values.
 */

#ifndef ULUNAS_CTFA_QR_H
#define ULUNAS_CTFA_QR_H

#include <stdint.h>

typedef struct {
    int ta_qr1, ta_qr2, ta_fc;
    int fa_qr1, fa_qr2, fa_fc;
} ctfa_qr_t;

#ifdef QR_CALIBRATION_MODE
extern ctfa_qr_t g_qr_d0, g_qr_d1, g_qr_d2, g_qr_d3, g_qr_d4;
#endif

#endif /* ULUNAS_CTFA_QR_H */
