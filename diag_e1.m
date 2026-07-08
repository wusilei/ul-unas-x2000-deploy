% diag_e1.m — Export E1 (XMB0) intermediate golden for C debugging
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
fprintf('=== E1 XMB0 Diagnostic — Frame %d ===\n', t-1);

% log_gen -> BM
spec_real = Fix_point(cmp_real(t, :), 's32f20');
spec_imag = Fix_point(cmp_imag(t, :), 's32f20');
x_log = log_gen(spec_real, spec_imag);
x_bm = BM_module(x_log, erbfc_weight);

% E0: XConv (full)
conv_cache_e0 = zeros(2, 129);
tfa_cache_e0 = zeros(1, 24);
[y_e0, conv_cache_e0, tfa_cache_e0] = XConv_module(x_bm, conv_cache_e0, tfa_cache_e0);
fprintf('E0 output: [%d x %d], range [%.2f, %.2f]\n', ...
    size(y_e0,1), size(y_e0,2), min(y_e0(:)), max(y_e0(:)));

%% ====================================================================
%% E1 Step 1: PConv0 + BN + AffinePReLU (XMB0_PConv_block_0)
%% ====================================================================
y_pconv0 = XMB0_PConv_block_0(y_e0);
fprintf('E1 PConv0+BN+AP: [%d x %d], range [%.2f, %.2f]\n', ...
    size(y_pconv0,1), size(y_pconv0,2), min(y_pconv0(:)), max(y_pconv0(:)));
write_bin([out_dir 'frame0_e1_pconv0.bin'], int32(y_pconv0));
fprintf('  -> wrote frame0_e1_pconv0.bin\n');

%% ====================================================================
%% E1 Step 2: Shuffle (interleave)
%% ====================================================================
y_s = zeros(24, 65);
y_s(1:2:end, :) = y_pconv0(1:12, :);
y_s(2:2:end, :) = y_pconv0(13:24, :);
fprintf('E1 Shuffle: [%d x %d], range [%.2f, %.2f]\n', ...
    size(y_s,1), size(y_s,2), min(y_s(:)), max(y_s(:)));
write_bin([out_dir 'frame0_e1_shuf.bin'], int32(y_s));
fprintf('  -> wrote frame0_e1_shuf.bin\n');

fprintf('\n=== Done. Copy .bin files to dump_matlab/ and run C test ===\n');

function write_bin(filename, data)
    fid = fopen(filename, 'wb');
    fwrite(fid, data(:), class(data));
    fclose(fid);
end
