% gen_lut_tables.m — Generate Fixed-Point LUT Tables for C
% ==========================================================
% Generates sigmoid (1024pt), tanh (1024pt), log10 (512pt) LUTs
% and exports them to ulunas_lut.c
%
% Usage: cd to UL-UNAS_SE_FPversion_v2/, run this script.
% Output: c_version/x2000_deploy_v2/ulunas_lut.c

clear; clc;

out_path = 'c_version/x2000_deploy_v2/ulunas_lut.c';
fid = fopen(out_path, 'w');

fprintf(fid, '/**\n');
fprintf(fid, ' * ulunas_lut.c — MATLAB-Generated LUT Tables\n');
fprintf(fid, ' * Generated: %s\n', datestr(now));
fprintf(fid, ' */\n\n');
fprintf(fid, '#include "ulunas_lut.h"\n');
fprintf(fid, '#include <stdint.h>\n\n');

%% Sigmoid LUT: 1024 entries, [-8, 8) → Q15 [0, 32768]
fprintf('Generating sigmoid LUT (1024 entries)...\n');
N = 1024;
x_min = -8.0;
x_range = 16.0;
x = linspace(x_min, x_min + x_range, N);
y = 1 ./ (1 + exp(-x));
y_q = round(y * 32768);  % Q15
y_q(y_q > 32768) = 32768;
y_q(y_q < 0) = 0;

fprintf(fid, '// Sigmoid LUT: 1024 entries, input [-8, 8), output Q15 [0, 32768]\n');
fprintf(fid, 'const uint16_t sigmoid_lut_q15[SIGMOID_LUT_SIZE] = {\n');
for j = 1:12:N
    fprintf(fid, '    ');
    for k = 0:11
        if j+k <= N
            fprintf(fid, '%d', y_q(j+k));
            if j+k < N; fprintf(fid, ', '); end
        end
    end
    fprintf(fid, '\n');
end
fprintf(fid, '};\n\n');

%% Tanh LUT: 1024 entries, [-4, 4) → Q15 [-32768, 32767]
fprintf('Generating tanh LUT (1024 entries)...\n');
x = linspace(-4.0, 4.0, N);
y = tanh(x);
y_q = round(y * 32768);
y_q(y_q > 32767) = 32767;
y_q(y_q < -32768) = -32768;

fprintf(fid, '// Tanh LUT: 1024 entries, input [-4, 4), output Q15\n');
fprintf(fid, 'const int16_t tanh_lut_q15[TANH_LUT_SIZE] = {\n');
for j = 1:12:N
    fprintf(fid, '    ');
    for k = 0:11
        if j+k <= N
            fprintf(fid, '%d', y_q(j+k));
            if j+k < N; fprintf(fid, ', '); end
        end
    end
    fprintf(fid, '\n');
end
fprintf(fid, '};\n\n');

%% Log10 LUT: 512 entries, normalized mantissa [0.5, 1.0) → Q20
fprintf('Generating log10 LUT (512 entries)...\n');
M = 512;
frac = linspace(0.5, 1.0, M);
y = log10(frac);
y_q = round(y * 1048576);  % Q20

fprintf(fid, '// Log10 LUT: 512 entries, mantissa [0.5, 1.0), output Q20\n');
fprintf(fid, 'const int32_t log10_lut_q20[LOG10_LUT_SIZE] = {\n');
for j = 1:8:M
    fprintf(fid, '    ');
    for k = 0:7
        if j+k <= M
            fprintf(fid, '%d', y_q(j+k));
            if j+k < M; fprintf(fid, ', '); end
        end
    end
    fprintf(fid, '\n');
end
fprintf(fid, '};\n\n');

%% Lookup function: sigmoid_q20_to_q15
fprintf(fid, 'uint16_t sigmoid_q20_to_q15(int32_t x_q20) {\n');
fprintf(fid, '    static const int32_t x_min = -8388608;\n');  % -8 * 2^20
fprintf(fid, '    if (x_q20 <= x_min) return 0;\n');
fprintf(fid, '    if (x_q20 >= 8388608) return 32768;\n');
fprintf(fid, '    int32_t offset = x_q20 - x_min;\n');
fprintf(fid, '    int32_t idx = offset >> 14;\n');          % / 16384
fprintf(fid, '    int32_t frac = offset & 0x3FFF;\n');      % 14-bit fraction
fprintf(fid, '    if (idx >= 1023) return sigmoid_lut_q15[1023];\n');
fprintf(fid, '    int32_t v0 = sigmoid_lut_q15[idx];\n');
fprintf(fid, '    int32_t v1 = sigmoid_lut_q15[idx + 1];\n');
fprintf(fid, '    return (uint16_t)(v0 + ((v1 - v0) * frac + 8192) / 16384);\n');
fprintf(fid, '}\n\n');

%% Lookup function: tanh_q20_to_q15
fprintf(fid, 'int16_t tanh_q20_to_q15(int32_t x_q20) {\n');
fprintf(fid, '    static const int32_t x_min = -4194304;\n');  % -4 * 2^20
fprintf(fid, '    if (x_q20 <= x_min) return -32768;\n');
fprintf(fid, '    if (x_q20 >= 4194304) return 32767;\n');
fprintf(fid, '    int32_t offset = x_q20 - x_min;\n');
fprintf(fid, '    int32_t idx = offset >> 13;\n');          % / 8192
fprintf(fid, '    int32_t frac = offset & 0x1FFF;\n');      % 13-bit fraction
fprintf(fid, '    if (idx >= 1023) return tanh_lut_q15[1023];\n');
fprintf(fid, '    int32_t v0 = tanh_lut_q15[idx];\n');
fprintf(fid, '    int32_t v1 = tanh_lut_q15[idx + 1];\n');
fprintf(fid, '    return (int16_t)(v0 + ((v1 - v0) * frac + 4096) / 8192);\n');
fprintf(fid, '}\n\n');

%% Lookup function: log10_q20
fprintf(fid, 'int32_t log10_q20(int32_t x_q20) {\n');
fprintf(fid, '    if (x_q20 <= 0) return -12582912;\n');  % log10(1e-12) ≈ -12 in Q20
fprintf(fid, '    uint32_t v = (uint32_t)x_q20;\n');
fprintf(fid, '    int32_t log2_int = 0;\n');
fprintf(fid, '    uint32_t t = v;\n');
fprintf(fid, '    while (t >>= 1) log2_int++;\n');
fprintf(fid, '    int32_t exponent = log2_int - 20;\n');
fprintf(fid, '    int32_t shift = log2_int - 19;\n');
fprintf(fid, '    uint32_t mantissa;\n');
fprintf(fid, '    if (shift >= 0) mantissa = v >> shift;\n');
fprintf(fid, '    else mantissa = v << (-shift);\n');
fprintf(fid, '    int32_t lut_idx = (int32_t)(mantissa - 524288) >> 10;\n');
fprintf(fid, '    if (lut_idx < 0) lut_idx = 0;\n');
fprintf(fid, '    if (lut_idx >= 512) lut_idx = 511;\n');
fprintf(fid, '    int32_t log10_mantissa = log10_lut_q20[lut_idx];\n');
fprintf(fid, '    int64_t log10_exp = (int64_t)exponent * 315653LL;\n');
fprintf(fid, '    return (int32_t)((log10_exp + 524288) / 1048576) + log10_mantissa;\n');
fprintf(fid, '}\n\n');

%% Sqrt function: integer binary search
fprintf(fid, 'uint32_t sqrt_q40_to_q20(uint64_t x_q40) {\n');
fprintf(fid, '    if (x_q40 == 0) return 0;\n');
fprintf(fid, '    if (x_q40 == 1) return 1;\n');
fprintf(fid, '    uint64_t lo = 1;\n');
fprintf(fid, '    uint64_t hi = x_q40;\n');
fprintf(fid, '    if (hi > 0xFFFFFFFFULL) hi = 0xFFFFFFFFULL;\n');
fprintf(fid, '    while (lo < hi) {\n');
fprintf(fid, '        uint64_t mid = (lo + hi + 1) >> 1;\n');
fprintf(fid, '        uint64_t mid_sq = mid * mid;\n');
fprintf(fid, '        if (mid_sq <= x_q40) lo = mid;\n');
fprintf(fid, '        else hi = mid - 1;\n');
fprintf(fid, '    }\n');
fprintf(fid, '    return (uint32_t)lo;\n');
fprintf(fid, '}\n');

fclose(fid);
fprintf('LUT tables written to %s\n', out_path);
