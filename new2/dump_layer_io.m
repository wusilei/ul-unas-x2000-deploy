%% dump_layer_io.m — Dump every function's input/output for Frame 0 as float
%  Run from new2/ directory
close all; clear; clc;

addpath('stft');
addpath('../UL-UNAS_SE_FPversion_v2');
addpath('../UL-UNAS_SE_FPversion_v2/para_in_mat_FP');
addpath('../UL-UNAS_SE_FPversion_v2/test_wavs');

out_file = 'frame0_layer_io.txt';
diary(out_file);
diary on;

%% Load
[noisy_audio, fs] = audioread('stft/test_wavs/noisy_fileid_1.wav');
erbfc_weight = importdata('../UL-UNAS_SE_FPversion_v2/para_in_mat_FP/erb_erb_fc_weight.mat');
ierbfc_weight = importdata('../UL-UNAS_SE_FPversion_v2/para_in_mat_FP/erb_ierb_fc_weight.mat');
hann_window = importdata('stft/stft_window.mat');

fprintf('=== Frame 0 Layer I/O Dump (float) ===\n');
fprintf('Audio: %d samples, fs=%d Hz\n', length(noisy_audio), fs);
fprintf('N_fft=512, win_len=512, win_inc=256\n\n');

%% [1] STFT_func
fprintf('=== [1] STFT_func ===\n');
[cmp_real, cmp_imag] = STFT_func(noisy_audio.', 512, 512, 256, hann_window);
t = 1;  % Frame 0
fprintf('INPUT: noisy_audio[1:10] = '); fprintf('%.6f ', noisy_audio(1:10)); fprintf('\n');
fprintf('INPUT: hann_window[1:5] = '); fprintf('%.6f ', hann_window(1:5)); fprintf('...\n');
fprintf('OUTPUT cmp_real[%d,1:10] = ', t); fprintf('%.6f ', cmp_real(t,1:10)); fprintf('\n');
fprintf('OUTPUT cmp_imag[%d,1:10] = ', t); fprintf('%.6f ', cmp_imag(t,1:10)); fprintf('\n');
fprintf('OUTPUT shape: [%d,%d]\n\n', size(cmp_real,1), size(cmp_real,2));

%% Q20 quantize
spec_real_q20 = Fix_point(cmp_real(t,:), 's32f20');
spec_imag_q20 = Fix_point(cmp_imag(t,:), 's32f20');
fprintf('Q20 spec_real[1:10] = '); fprintf('%d ', spec_real_q20(1:10)); fprintf('\n');
fprintf('Q20 spec_imag[1:10] = '); fprintf('%d ', spec_imag_q20(1:10)); fprintf('\n\n');

%% [2] log_gen
fprintf('=== [2] log_gen ===\n');
x_log = log_gen(spec_real_q20, spec_imag_q20);
fprintf('INPUT spec_real_q20[1:10] = '); fprintf('%d ', spec_real_q20(1:10)); fprintf('\n');
fprintf('OUTPUT x_log[1:10] = '); fprintf('%d ', x_log(1:10)); fprintf('\n\n');

%% Dequant for float display
x_log_f = double(x_log) / 1048576.0;
fprintf('OUTPUT x_log float[1:10] = '); fprintf('%.6f ', x_log_f(1:10)); fprintf('\n\n');

%% [3] BM_module
fprintf('=== [3] BM_module ===\n');
x_bm = BM_module(x_log, erbfc_weight);
fprintf('INPUT x_log[1:10] = '); fprintf('%d ', x_log(1:10)); fprintf('\n');
fprintf('OUTPUT x_bm[1:10] = '); fprintf('%d ', x_bm(1:10)); fprintf('\n');
x_bm_f = double(x_bm) / 1048576.0;
fprintf('OUTPUT x_bm float[1:10] = '); fprintf('%.6f ', x_bm_f(1:10)); fprintf('\n\n');

%% Temporal cache
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

%% [4] Encoder_module
fprintf('=== [4] Encoder_module ===\n');
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
fprintf('INPUT x_bm[1:10] = '); fprintf('%d ', x_bm(1:10)); fprintf('\n');
fprintf('OUTPUT E0[1:8] = '); fprintf('%d ', y_e0(1:8)); fprintf('\n');
fprintf('OUTPUT E0 float[1:8] = '); fprintf('%.6f ', double(y_e0(1:8))/1048576.0); fprintf('\n');
fprintf('OUTPUT E1[1:8] = '); fprintf('%d ', y_e1(1:8)); fprintf('\n');
fprintf('OUTPUT E2[1:8] = '); fprintf('%d ', y_e2(1:8)); fprintf('\n');
fprintf('OUTPUT E3[1:8] = '); fprintf('%d ', y_e3(1:8)); fprintf('\n');
fprintf('OUTPUT E4[1:8] = '); fprintf('%d ', y_e4(1:8)); fprintf('\n\n');

%% [5] GDPRNN
fprintf('=== [5] GDPRNN ===\n');
[y_rnn1, inter_cache_0] = GDPRNN_module(y_e4, inter_cache_0, 0);
[y_rnn2, inter_cache_1] = GDPRNN_module(y_rnn1, inter_cache_1, 1);
fprintf('OUTPUT RNN1[1:8] = '); fprintf('%d ', y_rnn1(1:8)); fprintf('\n');
fprintf('OUTPUT RNN1 float[1:8] = '); fprintf('%.6f ', double(y_rnn1(1:8))/1048576.0); fprintf('\n');
fprintf('OUTPUT RNN2[1:8] = '); fprintf('%d ', y_rnn2(1:8)); fprintf('\n\n');

%% [6] Decoder_module
fprintf('=== [6] Decoder_module ===\n');
[y_dec, tfa_cache_d0, tfa_cache_d1, ...
 conv_cache_d0, tfa_cache_d2, ...
 conv_cache_d1, tfa_cache_d3, ...
 conv_cache_d2, tfa_cache_d4] = Decoder_module(y_rnn2, y_e4, tfa_cache_d0, ...
    y_e3, tfa_cache_d1, ...
    y_e2, conv_cache_d0, tfa_cache_d2, ...
    y_e1, conv_cache_d1, tfa_cache_d3, ...
    y_e0, conv_cache_d2, tfa_cache_d4);
fprintf('OUTPUT y_dec[1:10] = '); fprintf('%d ', y_dec(1:10)); fprintf('\n');
fprintf('OUTPUT y_dec float[1:10] = '); fprintf('%.6f ', double(y_dec(1:10))/1048576.0); fprintf('\n\n');

%% [7] sigmoid_func
fprintf('=== [7] sigmoid_func ===\n');
y_dec_dq = y_dec * 2^(-20);
y_sig_dq = sigmoid_func(y_dec_dq);
y_sig = Fix_point(y_sig_dq, 'u16f15');
fprintf('OUTPUT y_sig[1:10] = '); fprintf('%d ', y_sig(1:10)); fprintf('\n');
fprintf('OUTPUT y_sig float[1:10] = '); fprintf('%.6f ', double(y_sig)/32768.0); fprintf('\n\n');

%% [8] BS_module
fprintf('=== [8] BS_module ===\n');
y_bs = BS_module(y_sig, ierbfc_weight);
fprintf('OUTPUT y_bs[1:10] = '); fprintf('%d ', y_bs(1:10)); fprintf('\n');
fprintf('OUTPUT y_bs float[1:10] = '); fprintf('%.6f ', double(y_bs)/32768.0); fprintf('\n\n');

%% [9] MASK_module
fprintf('=== [9] MASK_module ===\n');
y_mask = MASK_module(y_bs, spec_real_q20, spec_imag_q20);
fprintf('OUTPUT MASK real[1:10] = '); fprintf('%d ', y_mask(1,1:10)); fprintf('\n');
fprintf('OUTPUT MASK real float[1:10] = '); fprintf('%.6f ', double(y_mask(1,1:10))/1048576.0); fprintf('\n');
fprintf('OUTPUT MASK imag[1:10] = '); fprintf('%d ', y_mask(2,1:10)); fprintf('\n');
fprintf('OUTPUT MASK imag float[1:10] = '); fprintf('%.6f ', double(y_mask(2,1:10))/1048576.0); fprintf('\n\n');

%% === Full dump: all values as float ===
fprintf('\n=== FULL FLOAT DUMP (Frame 0) ===\n');

fprintf('STFT_real = ['); fprintf('%.8f, ', cmp_real(t,:)); fprintf('];\n');
fprintf('STFT_imag = ['); fprintf('%.8f, ', cmp_imag(t,:)); fprintf('];\n');
fprintf('log_gen = ['); fprintf('%.8f, ', double(x_log)/1048576.0); fprintf('];\n');
fprintf('BM = ['); fprintf('%.8f, ', double(x_bm)/1048576.0); fprintf('];\n');
fprintf('E0 = ['); fprintf('%.8f, ', double(y_e0)/1048576.0); fprintf('];\n');
fprintf('E1 = ['); fprintf('%.8f, ', double(y_e1)/1048576.0); fprintf('];\n');
fprintf('E2 = ['); fprintf('%.8f, ', double(y_e2)/1048576.0); fprintf('];\n');
fprintf('E3 = ['); fprintf('%.8f, ', double(y_e3)/1048576.0); fprintf('];\n');
fprintf('E4 = ['); fprintf('%.8f, ', double(y_e4)/1048576.0); fprintf('];\n');
fprintf('RNN1 = ['); fprintf('%.8f, ', double(y_rnn1)/1048576.0); fprintf('];\n');
fprintf('RNN2 = ['); fprintf('%.8f, ', double(y_rnn2)/1048576.0); fprintf('];\n');
fprintf('Decoder = ['); fprintf('%.8f, ', double(y_dec)/1048576.0); fprintf('];\n');
fprintf('Sigmoid = ['); fprintf('%.8f, ', double(y_sig)/32768.0); fprintf('];\n');
fprintf('BS = ['); fprintf('%.8f, ', double(y_bs)/32768.0); fprintf('];\n');
fprintf('MASK_real = ['); fprintf('%.8f, ', double(y_mask(1,:))/1048576.0); fprintf('];\n');
fprintf('MASK_imag = ['); fprintf('%.8f, ', double(y_mask(2,:))/1048576.0); fprintf('];\n');

diary off;
fprintf('\nOutput saved to %s\n', out_file);
