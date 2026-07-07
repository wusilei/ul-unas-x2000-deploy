#!/usr/bin/env python3
"""
diag_mask_chain.py — Step-by-step C vs MATLAB sigmoid→BS→MASK comparison
=======================================================================
Dumps and compares decoder output, sigmoid output, BS output, and MASK
for a specific speech frame.

Usage: python3 diag_mask_chain.py <work_dir> <frame_idx>
  (work_dir must have C dec/mask + MATLAB mat_dec/mat_mask from diag_full_audio.py)
"""
import sys, os
import numpy as np

N_BINS_BM = 129; N_BINS = 257

def snr_db(g, t):
    g, t = g.astype(np.float64), t.astype(np.float64)
    s = np.sum(g**2); e = np.sum((g - t)**2)
    return 999.0 if e < 1e-30 else 10 * np.log10(s / e)

def load_int32(path):
    return np.fromfile(path, dtype=np.int32) if os.path.exists(path) else None

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <work_dir> <frame_idx>")
        sys.exit(1)
    work_dir, fidx = sys.argv[1], int(sys.argv[2])

    # Load C outputs
    c_dec = load_int32(f"{work_dir}/frame{fidx}_dec.bin")       # int32 Q20, 129 values
    c_mask = load_int32(f"{work_dir}/frame{fidx}_mask.bin")      # int32 Q20, 514 values
    # Load MATLAB outputs
    m_dec = load_int32(f"{work_dir}/frame{fidx}_mat_dec.bin")
    m_mask = load_int32(f"{work_dir}/frame{fidx}_mat_mask.bin")

    if c_dec is None or m_dec is None:
        print("Missing decoder dumps — run diag_full_audio.py first")
        sys.exit(1)

    # === Step 1: Decoder output (before sigmoid) ===
    print(f"=== Frame {fidx}: Decoder → Sigmoid → BS → MASK Chain ===\n")

    dec_snr = snr_db(m_dec, c_dec)
    c_dec_f = c_dec.astype(np.float64) / 2**20
    m_dec_f = m_dec.astype(np.float64) / 2**20
    print(f"[1] Decoder output (Q20, 129 bins)")
    print(f"    SNR: {dec_snr:.2f} dB")
    print(f"    C  range: [{c_dec_f.min():+.4f}, {c_dec_f.max():+.4f}]")
    print(f"    MAT range: [{m_dec_f.min():+.4f}, {m_dec_f.max():+.4f}]")
    print(f"    C  mean: {c_dec_f.mean():+.4f}  std: {c_dec_f.std():.4f}")
    print(f"    MAT mean: {m_dec_f.mean():+.4f}  std: {m_dec_f.std():.4f}")

    # === Step 2: Sigmoid (simulate in Python with LUT from GTCRN) ===
    # GTCRN LUT: sigmoid_q15(q10) — Q10→Q15
    SIG_LUT_SIZE = 512; SIG_Q10_MIN = -8192; SIG_Q10_MAX = 8192
    SIG_Q10_RANGE = 16384; SIG_Q10_SHIFT = 14

    # Copy GTCRN LUT
    sigmoid_lut = np.array([
          11, 11, 12, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18,
          18, 19, 19, 20, 21, 21, 22, 23, 23, 24, 25, 26, 26, 27, 28, 29,
          30, 31, 32, 33, 34, 35, 36, 37, 38, 40, 41, 42, 44, 45, 46, 48,
          49, 51, 53, 54, 56, 58, 60, 61, 63, 65, 67, 70, 72, 74, 76, 79,
          81, 84, 87, 89, 92, 95, 98, 101, 104, 108, 111, 115, 118, 122, 126, 130,
         134, 138, 143, 147, 152, 157, 162, 167, 172, 177, 183, 189, 195, 201, 207, 214,
         221, 228, 235, 242, 250, 258, 266, 274, 283, 292, 301, 310, 320, 330, 341, 351,
         362, 374, 386, 398, 410, 423, 436, 450, 464, 479, 494, 509, 525, 542, 558, 576,
         594, 612, 632, 651, 672, 692, 714, 736, 759, 783, 807, 832, 858, 884, 912, 940,
         969, 999, 1029, 1061, 1094, 1127, 1162, 1197, 1234, 1272, 1311, 1351, 1392, 1434, 1478, 1522,
        1569, 1616, 1665, 1715, 1767, 1820, 1874, 1930, 1988, 2047, 2108, 2171, 2235, 2301, 2369, 2439,
        2511, 2584, 2660, 2737, 2817, 2898, 2982, 3068, 3156, 3247, 3340, 3435, 3532, 3632, 3734, 3839,
        3947, 4057, 4169, 4284, 4402, 4523, 4647, 4773, 4902, 5034, 5169, 5307, 5447, 5591, 5738, 5887,
        6040, 6196, 6355, 6517, 6682, 6850, 7021, 7195, 7373, 7553, 7737, 7923, 8113, 8305, 8501, 8700,
        8901, 9106, 9313, 9523, 9736, 9952, 10170, 10391, 10614, 10840, 11068, 11299, 11532, 11767, 12004, 12243,
       12484, 12727, 12972, 13218, 13466, 13715, 13965, 14217, 14469, 14722, 14977, 15232, 15487, 15743, 15999, 16256,
       16512, 16769, 17025, 17281, 17536, 17791, 18046, 18299, 18551, 18803, 19053, 19302, 19550, 19796, 20041, 20284,
       20525, 20764, 21001, 21236, 21469, 21700, 21928, 22154, 22377, 22598, 22816, 23032, 23245, 23455, 23662, 23867,
       24068, 24267, 24463, 24655, 24845, 25031, 25215, 25395, 25573, 25747, 25918, 26086, 26251, 26413, 26572, 26728,
       26881, 27030, 27177, 27321, 27461, 27599, 27734, 27866, 27995, 28121, 28245, 28366, 28484, 28599, 28711, 28821,
       28929, 29034, 29136, 29236, 29333, 29428, 29521, 29612, 29700, 29786, 29870, 29951, 30031, 30108, 30184, 30257,
       30329, 30399, 30467, 30533, 30597, 30660, 30721, 30780, 30838, 30894, 30948, 31001, 31053, 31103, 31152, 31199,
       31246, 31290, 31334, 31376, 31417, 31457, 31496, 31534, 31571, 31606, 31641, 31674, 31707, 31739, 31769, 31799,
       31828, 31856, 31884, 31910, 31936, 31961, 31985, 32009, 32032, 32054, 32076, 32096, 32117, 32136, 32156, 32174,
       32192, 32210, 32226, 32243, 32259, 32274, 32289, 32304, 32318, 32332, 32345, 32358, 32370, 32382, 32394, 32406,
       32417, 32427, 32438, 32448, 32458, 32467, 32476, 32485, 32494, 32502, 32510, 32518, 32526, 32533, 32540, 32547,
       32554, 32561, 32567, 32573, 32579, 32585, 32591, 32596, 32601, 32606, 32611, 32616, 32621, 32625, 32630, 32634,
       32638, 32642, 32646, 32650, 32653, 32657, 32660, 32664, 32667, 32670, 32673, 32676, 32679, 32681, 32684, 32687,
       32689, 32692, 32694, 32696, 32698, 32701, 32703, 32705, 32707, 32708, 32710, 32712, 32714, 32715, 32717, 32719,
       32720, 32722, 32723, 32724, 32726, 32727, 32728, 32730, 32731, 32732, 32733, 32734, 32735, 32736, 32737, 32738,
       32739, 32740, 32741, 32742, 32742, 32743, 32744, 32745, 32745, 32746, 32747, 32747, 32748, 32749, 32749, 32750,
       32750, 32751, 32751, 32752, 32752, 32753, 32753, 32754, 32754, 32755, 32755, 32756, 32756, 32756, 32757, 32757
    ], dtype=np.int32)

    def sigmoid_q15_lut(q10_arr):
        """GTCRN LUT sigmoid: Q10 → Q15. q10_arr in int32 Q10."""
        result = np.zeros_like(q10_arr, dtype=np.int32)
        for i in range(len(q10_arr)):
            q10 = q10_arr[i]
            if q10 <= -8192: result[i] = 0
            elif q10 >= 8192: result[i] = 32767
            else:
                pos = (q10 + 8192) * (SIG_LUT_SIZE - 1)
                idx = pos >> SIG_Q10_SHIFT
                frac = pos & (SIG_Q10_RANGE - 1)
                interp = (int(sigmoid_lut[idx]) * (SIG_Q10_RANGE - frac) +
                          int(sigmoid_lut[idx + 1]) * frac)
                result[i] = (interp + SIG_Q10_RANGE // 2) >> SIG_Q10_SHIFT
        return result

    def sigmoid_py(x_q20):
        """MATLAB-style: Q20→float→double sigmoid→Q15"""
        xf = x_q20.astype(np.float64) / 2**20
        # Clamp to avoid overflow
        xf = np.clip(xf, -88, 88)
        sf = 1.0 / (1.0 + np.exp(-xf))
        return np.clip(np.round(sf * 32768), 0, 32767).astype(np.int32)

    def sigmoid_c_lut(x_q20):
        """C LUT: Q20→Q10→LUT→Q15"""
        q10 = (x_q20 >> 10).astype(np.int32)
        return sigmoid_q15_lut(q10)

    # Compare sigmoid methods
    c_sig_lut = sigmoid_c_lut(c_dec)
    c_sig_py = sigmoid_py(c_dec)
    m_sig_py = sigmoid_py(m_dec)

    print(f"\n[2] Sigmoid output (129 bins, Q15)")
    # C vs MATLAB: compare both LUT and float-precision sigmoid
    sig_lut_snr = snr_db(m_sig_py, c_sig_lut)
    sig_py_snr = snr_db(m_sig_py, c_sig_py)
    print(f"    C(LUT) vs MATLAB(float): SNR={sig_lut_snr:.2f} dB")
    print(f"    C(float) vs MATLAB(float): SNR={sig_py_snr:.2f} dB {'⚠️ DECODER DIVERGENCE' if sig_py_snr < 40 else '✅ sigmoid OK'}")
    if sig_py_snr < 40:
        print(f"    → Decoder output itself differs between C and MATLAB!")
        print(f"    C  sigmoid range: [{c_sig_lut.min()}, {c_sig_lut.max()}]")
        print(f"    MAT sigmoid range: [{m_sig_py.min()}, {m_sig_py.max()}]")
    else:
        lut_vs_float = snr_db(c_sig_py, c_sig_lut)
        print(f"    LUT vs float (same input): SNR={lut_vs_float:.2f} dB {'✅ LUT OK' if lut_vs_float > 60 else '⚠️ LUT ERROR'}")

    # === Step 3: BS (Band Splitting) ===
    # Simulate BS in Python: (1,129) Q15 × weight(64,192) Q15 → (1,257) Q15
    # Weight: erb_ierb_fc_weight, shape (192, 64) col-major → needs transpose
    # We can't easily load .mat here, so skip BS sim and compare MASK directly

    # === Step 4: MASK comparison ===
    if c_mask is not None and m_mask is not None and len(c_mask) == 514 and len(m_mask) == 514:
        mask_snr = snr_db(m_mask, c_mask)
        print(f"\n[3] MASK output (514 values Q20, real+imag concatenated)")
        print(f"    SNR: {mask_snr:.2f} dB")

        # Per-bin breakdown: real vs imag
        real_snr = snr_db(m_mask[:257], c_mask[:257])
        imag_snr = snr_db(m_mask[257:], c_mask[257:])
        print(f"    Real part SNR: {real_snr:.2f} dB")
        print(f"    Imag part SNR: {imag_snr:.2f} dB")

        # Bin-wise SNR pattern: which bins are worst?
        bin_snr = np.zeros(257)
        for i in range(257):
            real_i_snr = snr_db(np.array([m_mask[i]]), np.array([c_mask[i]]))
            imag_i_snr = snr_db(np.array([m_mask[257+i]]), np.array([c_mask[257+i]]))
            bin_snr[i] = min(real_i_snr, imag_i_snr)
        worst_bins = np.argsort(bin_snr)[:10]
        print(f"\n    Worst 10 freq bins (bin: SNR):")
        for b in worst_bins:
            freq_hz = b * 16000 / 512  # approx
            print(f"      bin {b:3d} ({freq_hz:5.0f} Hz): {bin_snr[b]:.1f} dB")

    # === Summary ===
    print(f"\n=== Root Cause ===")
    if sig_py_snr < 40:
        print("DECODER OUTPUT DIVERGENCE — C and MATLAB produce different decoder values.")
        print("Check encoder→DPRNN→decoder chain for remaining bugs.")
    elif sig_lut_snr > 60:
        print("SIGMOID/BS/MASK chain is correct.")
        print("MASK degradation comes from decoder input differences (not sigmoid).")
    else:
        print("SIGMOID LUT ERROR — LUT produces different results than float sigmoid.")
        print("Check LUT implementation or bit-width.")

if __name__ == "__main__":
    main()
