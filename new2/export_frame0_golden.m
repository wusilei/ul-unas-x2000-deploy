%% export_frame0_golden.m — Export Frame 0 golden data at every pipeline stage
%  Run this from UL-UNAS_SE_FPversion_v2/ directory
%  Output: dump_matlab/ directory with per-stage .bin files

close all; clear; clc;
addpath('para_in_mat_FP');
addpath('test_wavs');
mkdir('dump_matlab');

%% Load ERB weights
erbfc_weight = importdata('erb_erb_fc_weight.mat');
ierbfc_weight = importdata('erb_ierb_fc_weight.mat');

%% Load noisy audio
[noisy_audio, fs] = audioread('noisy_fileid_1.wav');

%% Temporal cache (frame 0)
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

%% STFT
N_fft = 512; win_len = 512; win_inc = 256;
hann_window = importdata('stft_window.mat');
[cmp_real, cmp_imag] = STFT_func(noisy_audio.', N_fft, win_len, win_inc, hann_window);

%% Frame 0
t = 1;
fprintf('=== Exporting Frame %d Golden Data ===\n', t);

%% STFT output (Q20)
spec_real_q20 = Fix_point(cmp_real(t,:), 's32f20');
spec_imag_q20 = Fix_point(cmp_imag(t,:), 's32f20');
write_bin('dump_matlab/frame0_stft_real.bin', spec_real_q20);
write_bin('dump_matlab/frame0_stft_imag.bin', spec_imag_q20);

%% log_gen
x_log = log_gen(spec_real_q20, spec_imag_q20);
write_bin('dump_matlab/frame0_log_gen.bin', x_log);

%% BM
x_bm = BM_module(x_log, erbfc_weight);
write_bin('dump_matlab/frame0_bm.bin', x_bm);

%% Encoder — export each layer output
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

%% GDPRNN
[y_rnn1, inter_cache_0] = GDPRNN_module(y_e4, inter_cache_0, 0);
write_bin('dump_matlab/frame0_gdprnn1.bin', y_rnn1);

[y_rnn2, inter_cache_1] = GDPRNN_module(y_rnn1, inter_cache_1, 1);
write_bin('dump_matlab/frame0_gdprnn2.bin', y_rnn2);

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

%% Sigmoid
y_dec_dq = y_dec * 2^(-20);
y_sig_dq = sigmoid_func(y_dec_dq);
y_sig = Fix_point(y_sig_dq, 'u16f15');
write_bin('dump_matlab/frame0_sigmoid.bin', double(y_sig));

%% BS
y_bs = BS_module(y_sig, ierbfc_weight);
write_bin('dump_matlab/frame0_bs.bin', y_bs);

%% MASK
y_mask = MASK_module(y_bs, spec_real_q20, spec_imag_q20);
write_bin('dump_matlab/frame0_mask_real.bin', y_mask(1,:));
write_bin('dump_matlab/frame0_mask_imag.bin', y_mask(2,:));

%% Write metadata
fid = fopen('dump_matlab/frame0_meta.txt', 'w');
fprintf(fid, 'T=%d\n', t);
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

fprintf('=== Golden export complete ===\n');

%% Helper: write int32 data as binary
function write_bin(filename, data)
    data_i32 = int32(data(:));
    fid = fopen(filename, 'wb');
    fwrite(fid, data_i32, 'int32');
    fclose(fid);
    fprintf('  Wrote %s (%d elements)\n', filename, numel(data_i32));
end
