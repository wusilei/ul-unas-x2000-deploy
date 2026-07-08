% diag_d0.m — Export D0 (De_XDWS0) intermediate golden for C debugging
% ===================================================================
% Usage: run from UL-UNAS_SE_FPversion_v2/ directory
% Output: c_version/x2000_deploy_v2/dump_matlab/

close all; clear; clc;
addpath('para_in_mat_FP');
addpath('test_wavs');

out_dir = 'D:\haidesi\haidesi\ul-unas-x2000-deploy\UL-UNAS_SE_FPversion_v2\c_version\x2000_deploy_v2\dump_matlab\';

%% Load ERB and audio, run frame 0 through full pipeline
erbfc_weight = importdata('erb_erb_fc_weight.mat');
[noisy_audio, fs] = audioread('noisy_fileid_1.wav');
N_fft = 512; win_len = 512; win_inc = 256;
hann_window = importdata('stft_window.mat');
[cmp_real, cmp_imag] = STFT_func(noisy_audio.', N_fft, win_len, win_inc, hann_window);

t = 1;
fprintf('=== D0 De_XDWS0 Diagnostic — Frame %d ===\n', t-1);

spec_real = Fix_point(cmp_real(t, :), 's32f20');
spec_imag = Fix_point(cmp_imag(t, :), 's32f20');
x_log = log_gen(spec_real, spec_imag);
x_bm = BM_module(x_log, erbfc_weight);

% Encoder E0-E4 (inline, matching export_all_layers_v2.m)
conv_cache_e0 = zeros(2, 129); conv_cache_e1 = zeros(24, 65);
conv_cache_e2 = zeros(24, 33);
tfa_cache_e0 = zeros(1, 24); tfa_cache_e1 = zeros(1, 48);
tfa_cache_e2 = zeros(1, 48); tfa_cache_e3 = zeros(1, 64);
tfa_cache_e4 = zeros(1, 32);

[y_e0, conv_cache_e0, tfa_cache_e0] = XConv_module(x_bm, conv_cache_e0, tfa_cache_e0);
[y_e1, conv_cache_e1, tfa_cache_e1] = XMB0_module(y_e0, conv_cache_e1, tfa_cache_e1);
[y_e2, conv_cache_e2, tfa_cache_e2] = XDWS0_module(y_e1, conv_cache_e2, tfa_cache_e2);
y_e3 = XMB1_module(y_e2, tfa_cache_e3);
y_e4 = XDWS1_module(y_e3, tfa_cache_e4);

fprintf('E4: [%d x %d]\n', size(y_e4,1), size(y_e4,2));

% RNN block 0 (inline, matching export_all_layers_v2.m lines 201-281)
% Input: y_e4 is [16, 33], transpose to [33, 16]
x_dprnn = y_e4.';

% Intra-RNN
ir0_ih_w  = importdata('dpgrnn_0_intra_rnn_rnn1_weight_ih_l0.mat');
ir0_ih_b  = importdata('dpgrnn_0_intra_rnn_rnn1_bias_ih_l0.mat');
ir0_hh_w  = importdata('dpgrnn_0_intra_rnn_rnn1_weight_hh_l0.mat');
ir0_hh_b  = importdata('dpgrnn_0_intra_rnn_rnn1_bias_hh_l0.mat');
ir0_re_ih_w = importdata('dpgrnn_0_intra_rnn_rnn1_weight_ih_l0_reverse.mat');
ir0_re_ih_b = importdata('dpgrnn_0_intra_rnn_rnn1_bias_ih_l0_reverse.mat');
ir0_re_hh_w = importdata('dpgrnn_0_intra_rnn_rnn1_weight_hh_l0_reverse.mat');
ir0_re_hh_b = importdata('dpgrnn_0_intra_rnn_rnn1_bias_hh_l0_reverse.mat');
ir1_ih_w  = importdata('dpgrnn_0_intra_rnn_rnn2_weight_ih_l0.mat');
ir1_ih_b  = importdata('dpgrnn_0_intra_rnn_rnn2_bias_ih_l0.mat');
ir1_hh_w  = importdata('dpgrnn_0_intra_rnn_rnn2_weight_hh_l0.mat');
ir1_hh_b  = importdata('dpgrnn_0_intra_rnn_rnn2_bias_hh_l0.mat');
ir1_re_ih_w = importdata('dpgrnn_0_intra_rnn_rnn2_weight_ih_l0_reverse.mat');
ir1_re_ih_b = importdata('dpgrnn_0_intra_rnn_rnn2_bias_ih_l0_reverse.mat');
ir1_re_hh_w = importdata('dpgrnn_0_intra_rnn_rnn2_weight_hh_l0_reverse.mat');
ir1_re_hh_b = importdata('dpgrnn_0_intra_rnn_rnn2_bias_hh_l0_reverse.mat');
intra_fc_w = importdata('dpgrnn_0_intra_fc_weight.mat');
intra_fc_b = importdata('dpgrnn_0_intra_fc_bias.mat');
intra_ln_w = importdata('dpgrnn_0_intra_ln_weight.mat');
intra_ln_b = importdata('dpgrnn_0_intra_ln_bias.mat');

x0 = x_dprnn(:, 1:8); x1 = x_dprnn(:, 9:16);
nH = 4;
x0_gru = BiGRU_module(x0, nH, ir0_ih_w, ir0_ih_b, ir0_hh_w, ir0_hh_b, ir0_re_ih_w, ir0_re_ih_b, ir0_re_hh_w, ir0_re_hh_b, -13, -8);
x1_gru = BiGRU_module(x1, nH, ir1_ih_w, ir1_ih_b, ir1_hh_w, ir1_hh_b, ir1_re_ih_w, ir1_re_ih_b, ir1_re_hh_w, ir1_re_hh_b, -13, -8);
x_cat = cat(2, x0_gru, x1_gru);
x_fc = round(x_cat * intra_fc_w * 2^(-9)) + intra_fc_b;
x_ln = ln_func(x_fc, intra_ln_w, intra_ln_b, -14);
y_intra = x_dprnn + x_ln;

% Inter-RNN block 0
er0_ih_w = importdata('dpgrnn_0_inter_rnn_rnn1_weight_ih_l0.mat');
er0_ih_b = importdata('dpgrnn_0_inter_rnn_rnn1_bias_ih_l0.mat');
er0_hh_w = importdata('dpgrnn_0_inter_rnn_rnn1_weight_hh_l0.mat');
er0_hh_b = importdata('dpgrnn_0_inter_rnn_rnn1_bias_hh_l0.mat');
er1_ih_w = importdata('dpgrnn_0_inter_rnn_rnn2_weight_ih_l0.mat');
er1_ih_b = importdata('dpgrnn_0_inter_rnn_rnn2_bias_ih_l0.mat');
er1_hh_w = importdata('dpgrnn_0_inter_rnn_rnn2_weight_hh_l0.mat');
er1_hh_b = importdata('dpgrnn_0_inter_rnn_rnn2_bias_hh_l0.mat');
inter_fc_w = importdata('dpgrnn_0_inter_fc_weight.mat');
inter_fc_b = importdata('dpgrnn_0_inter_fc_bias.mat');
inter_ln_w = importdata('dpgrnn_0_inter_ln_weight.mat');
inter_ln_b = importdata('dpgrnn_0_inter_ln_bias.mat');

inter_cache_0 = zeros(33, 16);
i0 = y_intra(:, 1:8); i1 = y_intra(:, 9:16);
[i0_gru, inter_cache_0(:,1:8)] = GRU_module(i0, 8, inter_cache_0(:,1:8), er0_ih_w, er0_ih_b, er0_hh_w, er0_hh_b, -13, -8);
[i1_gru, inter_cache_0(:,9:16)] = GRU_module(i1, 8, inter_cache_0(:,9:16), er1_ih_w, er1_ih_b, er1_hh_w, er1_hh_b, -13, -8);
i_cat = cat(2, i0_gru, i1_gru);
i_fc = round(i_cat * inter_fc_w * 2^(-9)) + inter_fc_b;
i_ln = ln_func(i_fc, inter_ln_w, inter_ln_b, -13);
y_rnn1 = (y_intra + i_ln).';

% RNN block 1
inter_cache_1 = zeros(33, 16);
[y_rnn2, inter_cache_1] = GDPRNN_module(y_rnn1, inter_cache_1, 1);

fprintf('RNN2: [%d x %d], range [%.2f, %.2f]\n', ...
    size(y_rnn2,1), size(y_rnn2,2), min(y_rnn2(:)), max(y_rnn2(:)));

%% ====================================================================
%% D0: De_XDWS0 — skip(y_e4) → PConv → Shuffle → nonGTConv → cTFA
%% ====================================================================
x_cat_d0 = y_rnn2 + y_e4;

y_d0_pconv = De_XDWS0_PConv_block(x_cat_d0);
fprintf('D0 PConv+BN+AP: [%d x %d], range [%.2f, %.2f]\n', ...
    size(y_d0_pconv,1), size(y_d0_pconv,2), min(y_d0_pconv(:)), max(y_d0_pconv(:)));
write_bin([out_dir 'frame0_d0_pconv0.bin'], int32(y_d0_pconv));
fprintf('  -> wrote frame0_d0_pconv0.bin\n');

y_d0_s = zeros(32, 33);
y_d0_s(1:2:end, :) = y_d0_pconv(1:16, :);
y_d0_s(2:2:end, :) = y_d0_pconv(17:32, :);
fprintf('D0 Shuffle: [%d x %d], range [%.2f, %.2f]\n', ...
    size(y_d0_s,1), size(y_d0_s,2), min(y_d0_s(:)), max(y_d0_s(:)));
write_bin([out_dir 'frame0_d0_shuf.bin'], int32(y_d0_s));
fprintf('  -> wrote frame0_d0_shuf.bin\n');

fprintf('\n=== Done ===\n');

function write_bin(filename, data)
    fid = fopen(filename, 'wb');
    fwrite(fid, data(:), class(data));
    fclose(fid);
end
