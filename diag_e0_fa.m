% diag_e0_fa.m — Export E0 cTFA FA sub-step golden for C debugging
% ===================================================================
% Usage: run from UL-UNAS_SE_FPversion_v2/ directory
% Output: c_version/x2000_deploy_v2/dump_matlab/

close all; clear; clc;
addpath('para_in_mat_FP');
addpath('test_wavs');

out_dir = 'D:\haidesi\haidesi\ul-unas-x2000-deploy\UL-UNAS_SE_FPversion_v2\c_version\x2000_deploy_v2\dump_matlab\';
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

%% Load ERB and audio
erbfc_weight = importdata('erb_erb_fc_weight.mat');
[noisy_audio, fs] = audioread('noisy_fileid_1.wav');
N_fft = 512; win_len = 512; win_inc = 256;
hann_window = importdata('stft_window.mat');
[cmp_real, cmp_imag] = STFT_func(noisy_audio.', N_fft, win_len, win_inc, hann_window);

%% Frame 0
t = 1;
fprintf('=== E0 cTFA FA Diagnostic — Frame %d ===\n', t-1);

% log_gen → BM
spec_real = Fix_point(cmp_real(t, :), 's32f20');
spec_imag = Fix_point(cmp_imag(t, :), 's32f20');
x_log = log_gen(spec_real, spec_imag);
x_bm = BM_module(x_log, erbfc_weight);

% XConv TConv (to get y_tconv as cTFA input)
conv_cache_e0 = zeros(2, 129);
tfa_cache_e0 = zeros(1, 24);
[y_tconv, conv_cache_e0] = XConv_TConv_block(x_bm, conv_cache_e0);

fprintf('TConv output: [%d x %d], range [%.2f, %.2f]\n', ...
    size(y_tconv,1), size(y_tconv,2), min(y_tconv(:)), max(y_tconv(:)));

%% ====================================================================
%% FA Diagnostic: inline XConv_cTFA_fa_module with sub-step dumps
%% ====================================================================

% Load FA weights
fa_ih_weight = importdata('encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0.mat');
fa_ih_bias   = importdata('encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0.mat');
fa_hh_weight = importdata('encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0.mat');
fa_hh_bias   = importdata('encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0.mat');

fa_re_ih_weight = importdata('encoder_en_convs_0_ops_4_fa_gru_weight_ih_l0_reverse.mat');
fa_re_ih_bias   = importdata('encoder_en_convs_0_ops_4_fa_gru_bias_ih_l0_reverse.mat');
fa_re_hh_weight = importdata('encoder_en_convs_0_ops_4_fa_gru_weight_hh_l0_reverse.mat');
fa_re_hh_bias   = importdata('encoder_en_convs_0_ops_4_fa_gru_bias_hh_l0_reverse.mat');

fa_fc_weight = importdata('encoder_en_convs_0_ops_4_fa_fc_weight.mat');
fa_fc_bias   = importdata('encoder_en_convs_0_ops_4_fa_fc_bias.mat');

%% Step 1: Aggregation (square + mean over channels)
x = y_tconv;
x_dq = x * 2^(-20);
x_squared = x_dq.^2;
x_agg = mean(x_squared, 1);           % [1, 65] float
x_agg_q = Fix_point(x_agg, 'u32f20'); % [1, 65] uint32 Q20

write_bin([out_dir 'frame0_e0_fa_agg.bin'], uint32(x_agg_q));
fprintf('FA agg: [%d], range [%.2f, %.2f]\n', ...
    length(x_agg_q), min(x_agg_q(:)), max(x_agg_q(:)));

%% Step 2: Pad + Reshape for BiGRU
pad_len = 3;
x_pad = [x_agg_q zeros(1, pad_len)];  % [1, 68]
x_t = reshape(x_pad, [4, 17])';       % [17, 4]

fprintf('FA reshaped: [%d x %d]\n', size(x_t,1), size(x_t,2));

%% Step 3: FA BiGRU (nHidden=4, forward + reverse)
nHidden = 4;

% Forward GRU
h_cache0 = zeros(1, nHidden);
x_gru0 = zeros(17, nHidden);
for f = 1:17
    [x_gru0(f,:), h_cache0] = GRU_module(x_t(f,:), nHidden, h_cache0, ...
        fa_ih_weight, fa_ih_bias, fa_hh_weight, fa_hh_bias, -13, -8);
end
write_bin([out_dir 'frame0_e0_fa_gru0.bin'], int16(x_gru0));
fprintf('FA BiGRU fwd: [%d x %d], range [%.0f, %.0f]\n', ...
    size(x_gru0,1), size(x_gru0,2), min(x_gru0(:)), max(x_gru0(:)));

% Reverse GRU
x_t_re = x_t(end:-1:1, :);
h_cache1 = zeros(1, nHidden);
x_gru1 = zeros(17, nHidden);
for f = 1:17
    [x_gru1(17-f+1,:), h_cache1] = GRU_module(x_t_re(f,:), nHidden, h_cache1, ...
        fa_re_ih_weight, fa_re_ih_bias, fa_re_hh_weight, fa_re_hh_bias, -13, -8);
end
write_bin([out_dir 'frame0_e0_fa_gru1.bin'], int16(x_gru1));
fprintf('FA BiGRU rev: [%d x %d], range [%.0f, %.0f]\n', ...
    size(x_gru1,1), size(x_gru1,2), min(x_gru1(:)), max(x_gru1(:)));

% Concat
x_gru = cat(2, x_gru0, x_gru1);  % [17, 8]

%% Step 4: FA FC
x_fc = round(x_gru * fa_fc_weight * 2^(-9)) + fa_fc_bias;  % [17, 4]
x_shape = reshape(x_fc.', 1, []);  % [1, 68]
write_bin([out_dir 'frame0_e0_fa_fc.bin'], int32(x_shape));
fprintf('FA FC: [%d], range [%.0f, %.0f]\n', ...
    length(x_shape), min(x_shape(:)), max(x_shape(:)));

%% Step 5: Sigmoid (final)
x_shape_dq = x_shape * 2^(-20);
y_dq = sigmoid_func(x_shape_dq(1:end-pad_len));  % remove pad
y = Fix_point(y_dq, 'u16f15');
write_bin([out_dir 'frame0_e0_fa_out.bin'], uint16(y));
fprintf('FA sigmoid: [%d], range [%d, %d]\n', ...
    length(y), min(y(:)), max(y(:)));

%% Also export TA for comparison
fprintf('\n=== TA Diagnostic ===\n');
ta_ih_weight = importdata('encoder_en_convs_0_ops_4_ta_gru_weight_ih_l0.mat');
ta_ih_bias   = importdata('encoder_en_convs_0_ops_4_ta_gru_bias_ih_l0.mat');
ta_hh_weight = importdata('encoder_en_convs_0_ops_4_ta_gru_weight_hh_l0.mat');
ta_hh_bias   = importdata('encoder_en_convs_0_ops_4_ta_gru_bias_hh_l0.mat');
ta_fc_weight = importdata('encoder_en_convs_0_ops_4_ta_fc_weight.mat');
ta_fc_bias   = importdata('encoder_en_convs_0_ops_4_ta_fc_bias.mat');

% Aggregation
x_dq = y_tconv * 2^(-20);
x_sq = x_dq.^2;
x_agg_ta = mean(x_sq, 2);  % [12, 1]
x_t_ta = Fix_point(x_agg_ta', 'u32f20');  % [1, 12]

% GRU
nHidden_ta = 24;
h_cache_ta = zeros(1, nHidden_ta);
[x_gru_ta, ~] = GRU_module(x_t_ta, nHidden_ta, h_cache_ta, ...
    ta_ih_weight, ta_ih_bias, ta_hh_weight, ta_hh_bias, -13, -8);
write_bin([out_dir 'frame0_e0_ta_gru.bin'], int16(x_gru_ta));

% FC
x_fc_ta = round(x_gru_ta * ta_fc_weight * 2^(-8)) + ta_fc_bias;
% Sigmoid
x_fc_ta_dq = x_fc_ta * 2^(-20);
y_ta_dq = sigmoid_func(x_fc_ta_dq);
y_ta = Fix_point(y_ta_dq, 'u16f15');
write_bin([out_dir 'frame0_e0_ta_out.bin'], uint16(y_ta));
fprintf('TA GRU: [%d], FC: [%d], sigmoid range [%d, %d]\n', ...
    length(x_gru_ta), length(y_ta), min(y_ta(:)), max(y_ta(:)));

fprintf('\nAll golden dumps written to %s\n', out_dir);

%% Helper
function write_bin(filename, data)
    fid = fopen(filename, 'wb');
    fwrite(fid, data(:), class(data));
    fclose(fid);
end
