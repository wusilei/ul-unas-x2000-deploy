#include <stdio.h>
int main() {
#ifdef QR_CALIBRATION_MODE
    printf("QR_CALIBRATION_MODE defined\n");
#else
    printf("QR_CALIBRATION_MODE NOT defined\n");
#endif
#ifdef JOINT_CALIBRATION_MODE
    printf("JOINT_CALIBRATION_MODE defined\n");
#else
    printf("JOINT_CALIBRATION_MODE NOT defined\n");
#endif
    printf("D4_TCONV_CQR=%d D4_TCONV_BN1=%d D4_TCONV_BN2=%d\n", D4_TCONV_CQR, D4_TCONV_BN1, D4_TCONV_BN2);
    printf("D4_TA: %d,%d,%d\n", D4_TA);
    printf("D4_FA: %d,%d,%d\n", D4_FA);
    return 0;
}
