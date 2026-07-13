%% export_frame0_golden_v3.m — Export Frame 0 golden data at every pipeline stage
%  V3: Loads C STFT output directly (skip MATLAB STFT) → model comparison only
%  Run this from new1/ directory after C has dumped STFT to dump_c/
%  Output: dump_matlab/ directory with per-stage .bin files
%
%  Usage:
%    cd('D:\haidesi\haidesi\ul-unas-x2000-deploy\new1');
%    run('export_frame0_golden_v3.m');

close all; clear; clc;

%% Paths
addpath('stft');                                          % for Fix_point and helper functions
addpath('../UL-UNAS_SE_FPversion_v2');                    % model functions (Encoder_module, etc.)
addpath('../UL-UNAS_SE_FPversion_v2/para_in_mat_FP');     % weights
mkdir('dump_matlab');

%% Load C STFT dump (skip MATLAB STFT entirely)
fprintf('=== Loading C STFT dump ===\n');
C_DUMP_DIR = 'ulunas_c/dump_c';

spec_real_q20 = read_bin_int32(fullfile(C_DUMP_DIR, 'frame0_stft_real_c.bin'), 257);
spec_imag_q20 = read_bin_int32(fullfile(C_DUMP_DIR, 'frame0_stft_imag_c.bin'), 257);
fprintf('  C STFT real[0]=%d, imag[0]=%d\n', spec_real_q20(1), spec_imag_q20(1));
fprintf('  C STFT real non-zero: %d, imag non-zero: %d\n', ...
    sum(spec_real_q20 ~= 0), sum(spec_imag_q20 ~= 0));

%% Load ERB weights
erbfc_weight = importdata('../UL-UNAS_SE_FPversion_v2/para_in_mat_FP/erb_erb_fc_weight.mat');
ierbfc_weight = importdata('../UL-UNAS_SE_FPversion_v2/para_in_mat_FP/erb_ierb_fc_weight.mat');

%% Write STFT (C dump) as golden for reference
write_bin('dump_matlab/frame0_stft_real.bin', spec_real_q20);
write_bin('dump_matlab/frame0_stft_imag.bin', spec_imag_q20);

%% --- Pipeline (identical to export_frame0_golden.m, but input from C dump) ---

fprintf('\n=== Running MATLAB fixed-point pipeline ===\n');

%% log_gen
x_log = log_gen(spec_real_q20, spec_imag_q20);
write_bin('dump_matlab/frame0_log_gen.bin', x_log);
fprintf('  log_gen: [%d] min=%d max=%d\n', length(x_log), min(x_log), max(x_log));

%% BM
x_bm = BM_module(x_log, erbfc_weight);
write_bin('dump_matlab/frame0_bm.bin', x_bm);
fprintf('  BM: [%d]\n', length(x_bm));

%% Temporal cache (frame 0 → all zeros)
conv_cache_e0 = zeros(2,129);
conv_cache_e1 = zeros(24,65);
conv_cache_e2 = zeros(24,33);
conv_cache_d0 = zeros(24,33);
conv_cache_d1 = zeros(12,33);
conv_cache_d2 = zeros(12,2,65);

tfa_cache_e0 = zeros(1,24);
tfa_cache_e1 = zeros(1,48);
tfa_cache_e2 = zeros(1,48);
tfa_cache_e3 = zeros(1,64);
tfa_cache_e4 = zeros(1,32);
tfa_cache_d0 = zeros(1,64);
tfa_cache_d1 = zeros(1,48);
tfa_cache_d2 = zeros(1,48);
tfa_cache_d3 = zeros(1,24);
tfa_cache_d4 = zeros(1,2);

inter_cache_0 = zeros(33,16);
inter_cache_1 = zeros(33,16);

%% Encoder
[y_e0, conv_cache_e0, tfa_cache_e0, ...
 y_e1, conv_cache_e1, tfa_cache_e1, ...
 y_e2, conv_cache_e2, tfa_cache_e2, ...
 y_e3, tfa_cache_e3, ...
 y_e4, tfa_cache_e4] = Encoder_module(x_bm, ...
    conv_cache_e0, tfa_cache_e0, ...
    conv_cache_e1, tfa_cache_e1, ...
    conv_cache_e2, tfa_cache_e2, ...
    tfa_cache_e3, ...
    tfa_cache_e4);

write_bin('dump_matlab/frame0_e0_xconv.bin', y_e0);
write_bin('dump_matlab/frame0_e1_xmb0.bin',  y_e1);
write_bin('dump_matlab/frame0_e2_xdws0.bin', y_e2);
write_bin('dump_matlab/frame0_e3_xmb1.bin',  y_e3);
write_bin('dump_matlab/frame0_e4_xdws1.bin', y_e4);
fprintf('  Encoder: E0[%d,%d] E1[%d,%d] E2[%d,%d] E3[%d,%d] E4[%d,%d]\n', ...
    size(y_e0,1),size(y_e0,2), size(y_e1,1),size(y_e1,2), ...
    size(y_e2,1),size(y_e2,2), size(y_e3,1),size(y_e3,2), size(y_e4,1),size(y_e4,2));

%% GDPRNN
[y_rnn1, inter_cache_0] = GDPRNN_module(y_e4, inter_cache_0, 0);
write_bin('dump_matlab/frame0_gdprnn1.bin', y_rnn1);
fprintf('  RNN1: [%d,%d]\n', size(y_rnn1,1), size(y_rnn1,2));

[y_rnn2, inter_cache_1] = GDPRNN_module(y_rnn1, inter_cache_1, 1);
write_bin('dump_matlab/frame0_gdprnn2.bin', y_rnn2);
fprintf('  RNN2: [%d,%d]\n', size(y_rnn2,1), size(y_rnn2,2));

%% Decoder
[y_dec, tfa_cache_d0, tfa_cache_d1, ...
 conv_cache_d0, tfa_cache_d2, ...
 conv_cache_d1, tfa_cache_d3, ...
 conv_cache_d2, tfa_cache_d4] = Decoder_module(y_rnn2, y_e4, tfa_cache_d0, ...
    y_e3, tfa_cache_d1, ...
    y_e2, conv_cache_d0, tfa_cache_d2, ...
    y_e1, conv_cache_d1, tfa_cache_d3, ...
    y_e0, conv_cache_d2, tfa_cache_d4);

write_bin('dump_matlab/frame0_decoder.bin', y_dec);
fprintf('  Decoder: [%d]\n', length(y_dec));

%% Sigmoid
y_dec_dq = y_dec * 2^(-20);
y_sig_dq = sigmoid_func(y_dec_dq);
y_sig = Fix_point(y_sig_dq, 'u16f15');
write_bin('dump_matlab/frame0_sigmoid.bin', double(y_sig));
fprintf('  Sigmoid: [%d] u16f15\n', length(y_sig));

%% BS
y_bs = BS_module(y_sig, ierbfc_weight);
write_bin('dump_matlab/frame0_bs.bin', y_bs);
fprintf('  BS: [%d]\n', length(y_bs));

%% MASK
y_mask = MASK_module(y_bs, spec_real_q20, spec_imag_q20);
write_bin('dump_matlab/frame0_mask_real.bin', y_mask(1,:));
write_bin('dump_matlab/frame0_mask_imag.bin', y_mask(2,:));
fprintf('  MASK: real [%d] imag [%d]\n', length(y_mask(1,:)), length(y_mask(2,:)));

%% Write metadata
fid = fopen('dump_matlab/frame0_meta.txt', 'w');
fprintf(fid, 'source=C_STFT_dump\n');
fprintf(fid, 'T=1\n');
fprintf(fid, 'N_fft=512\n');
fprintf(fid, 'win_len=512\n');
fprintf(fid, 'win_inc=256\n');
fprintf(fid, 'STFT_real_shape=[1,257]\n');
fprintf(fid, 'log_gen_shape=[1,129]\n');
fprintf(fid, 'e0_xconv_shape=[12,65]\n');
fprintf(fid, 'e1_xmb0_shape=[24,33]\n');
fprintf(fid, 'e2_xdws0_shape=[24,33]\n');
fprintf(fid, 'e3_xmb1_shape=[32,33]\n');
fprintf(fid, 'e4_xdws1_shape=[16,33]\n');
fprintf(fid, 'gdprnn1_shape=[16,33]\n');
fprintf(fid, 'gdprnn2_shape=[16,33]\n');
fprintf(fid, 'decoder_shape=[1,129]\n');
fprintf(fid, 'sigmoid_shape=[1,129]\n');
fprintf(fid, 'bs_shape=[1,257]\n');
fprintf(fid, 'mask_real_shape=[1,257]\n');
fprintf(fid, 'mask_imag_shape=[1,257]\n');
fclose(fid);

fprintf('\n=== Golden export complete (V3: from C STFT) ===\n');
fprintf('  Output: dump_matlab/*.bin (16 files)\n');

%% ===== Helper Functions =====

function write_bin(filename, data)
    data_i32 = int32(data(:));
    fid = fopen(filename, 'wb');
    fwrite(fid, data_i32, 'int32');
    fclose(fid);
    fprintf('  Wrote %s (%d int32)\n', filename, numel(data_i32));
end

function data = read_bin_int32(filename, n)
    fid = fopen(filename, 'rb');
    if fid < 0
        error('Cannot open: %s', filename);
    end
    data = fread(fid, n, 'int32');
    fclose(fid);
    if length(data) ~= n
        warning('Expected %d elements, got %d from %s', n, length(data), filename);
    end
    data = data(:)';  % row vector
end
