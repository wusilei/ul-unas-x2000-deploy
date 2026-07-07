% export_all_layers.m — Export golden layer outputs from MATLAB
% ==============================================================
% Runs the UL-UNAS MATLAB inference and dumps intermediate tensors
% to binary files for C layer-by-layer verification.
%
% v2: Dumps TConv intermediate outputs (before cTFA) at each encoder
%     stage, to isolate conv/BN/AffinePReLU Q-format from cTFA.
%
% Usage: run from UL-UNAS_SE_FPversion_v2/ directory

close all; clear; clc;
addpath('para_in_mat_FP');
addpath('test_wavs');

% Output directory (absolute path)
out_dir = 'D:\haidesi\haidesi\ul-unas-x2000-deploy\UL-UNAS_SE_FPversion_v2\c_version\x2000_deploy_v1\dump_matlab\';
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

% Load ERB weights
erbfc_weight = importdata('erb_erb_fc_weight.mat');
ierbfc_weight = importdata('erb_ierb_fc_weight.mat');

% Load test audio
[noisy_audio, fs] = audioread('noisy_fileid_1.wav');
disp(['Audio: ', num2str(length(noisy_audio)), ' samples, fs=', num2str(fs)]);

% STFT parameters
N_fft = 512;
win_len = 512;
win_inc = 256;
hann_window = importdata('stft_window.mat');

[cmp_real, cmp_imag] = STFT_func(noisy_audio.', N_fft, win_len, win_inc, hann_window);

T = size(cmp_real, 1);
fprintf('Total frames: %d\n', T);

% Initialize caches
conv_cache_e0 = zeros(2, 129);
conv_cache_e1 = zeros(24, 65);
conv_cache_e2 = zeros(24, 33);
conv_cache_d0 = zeros(24, 33);
conv_cache_d1 = zeros(12, 33);
conv_cache_d2 = zeros(12, 2, 65);

tfa_cache_e0 = zeros(1, 24);
tfa_cache_e1 = zeros(1, 48);
tfa_cache_e2 = zeros(1, 48);
tfa_cache_e3 = zeros(1, 64);
tfa_cache_e4 = zeros(1, 32);
tfa_cache_d0 = zeros(1, 64);
tfa_cache_d1 = zeros(1, 48);
tfa_cache_d2 = zeros(1, 48);
tfa_cache_d3 = zeros(1, 24);
tfa_cache_d4 = zeros(1, 2);

inter_cache_0 = zeros(33, 16);
inter_cache_1 = zeros(33, 16);

% Process first few frames and dump intermediate tensors
max_frames = min(T, 5);
for t = 1:max_frames
    fprintf('Frame %d/%d\n', t, max_frames);

    % Log-magnitude
    spec_real = Fix_point(cmp_real(t, :), 's32f20');
    spec_imag = Fix_point(cmp_imag(t, :), 's32f20');
    x = log_gen(spec_real, spec_imag);

    % BM
    x_bm = BM_module(x, erbfc_weight);

    % Save BM output
    write_bin([out_dir 'frame' num2str(t-1) '_bm.bin'], int32(x_bm));

    % ====================================================================
    % Encoder — expanded inline to dump TConv intermediates
    % ====================================================================

    % --- XConv: TConv(1→12, k=3×3, s=2) → y_tconv (12,65) → cTFA → y_e0 ---
    [y_e0_tconv, conv_cache_e0] = XConv_TConv_block(x_bm, conv_cache_e0);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e0_tconv.bin'], int32(y_e0_tconv));

    write_bin([out_dir 'frame' num2str(t-1) '_enc_e0_ctfa_in.bin'], int32(y_e0_tconv));
    [y_e0_ta, tfa_cache_e0] = XConv_cTFA_ta_module(y_e0_tconv, tfa_cache_e0);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e0_ctfa_ta.bin'], uint16(y_e0_ta));
    y_e0_fa = XConv_cTFA_fa_module(y_e0_tconv);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e0_ctfa_fa.bin'], uint16(y_e0_fa));
    y_e0_t = round(y_e0_tconv .* (y_e0_ta.') * 2^(-15));
    y_e0 = round(y_e0_t .* y_e0_fa * 2^(-15));
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e0_ctfa_out.bin'], int32(y_e0));
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e0.bin'], int32(y_e0));

    % --- XMB0: PConv0(12→24) → shuffle → TConv(24→24,s=2) → y_tconv (24,33) → PConv1(BN) → cTFA → y_e1 ---
    y_e1_pconv0 = XMB0_PConv_block_0(y_e0);
    y_e1_s = zeros(24, 65);
    y_e1_s(1:2:end, :) = y_e1_pconv0(1:12, :);
    y_e1_s(2:2:end, :) = y_e1_pconv0(13:24, :);
    [y_e1_tconv, conv_cache_e1] = XMB0_TConv_block(y_e1_s, conv_cache_e1);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e1_tconv.bin'], int32(y_e1_tconv));

    y_e1_pconv1 = XMB0_PConv_block_1(y_e1_tconv);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e1_ctfa_in.bin'], int32(y_e1_pconv1));  % cTFA input
    [y_e1_ta, tfa_cache_e1] = XMB0_cTFA_ta_module(y_e1_pconv1, tfa_cache_e1);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e1_ctfa_ta.bin'], uint16(y_e1_ta));     % ta gate
    y_e1_fa = XMB0_cTFA_fa_module(y_e1_pconv1);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e1_ctfa_fa.bin'], uint16(y_e1_fa));     % fa gate
    y_e1_t = round(y_e1_pconv1 .* (y_e1_ta.') * 2^(-15));
    y_e1_t = round(y_e1_t .* y_e1_fa * 2^(-15));
    y_e1 = zeros(24, 33);
    y_e1(1:2:end, :) = y_e1_t(1:12, :);
    y_e1(2:2:end, :) = y_e1_t(13:24, :);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e1_ctfa_out.bin'], int32(y_e1_t));      % after cTFA, before shuffle
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e1.bin'], int32(y_e1));

    % --- XDWS0: PConv(24→24) → shuffle → TConv(24→24,s=1) → y_tconv (24,33) → cTFA → y_e2 ---
    y_e2_pconv = XDWS0_PConv_block(y_e1);
    y_e2_s = zeros(24, 33);
    y_e2_s(1:2:end, :) = y_e2_pconv(1:12, :);
    y_e2_s(2:2:end, :) = y_e2_pconv(13:24, :);
    [y_e2_tconv, conv_cache_e2] = XDWS0_TConv_block(y_e2_s, conv_cache_e2);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e2_tconv.bin'], int32(y_e2_tconv));

    write_bin([out_dir 'frame' num2str(t-1) '_enc_e2_ctfa_in.bin'], int32(y_e2_tconv));
    [y_e2_ta, tfa_cache_e2] = XDWS0_cTFA_ta_module(y_e2_tconv, tfa_cache_e2);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e2_ctfa_ta.bin'], uint16(y_e2_ta));
    y_e2_fa = XDWS0_cTFA_fa_module(y_e2_tconv);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e2_ctfa_fa.bin'], uint16(y_e2_fa));
    y_e2_t = round(y_e2_tconv .* (y_e2_ta.') * 2^(-15));
    y_e2 = round(y_e2_t .* y_e2_fa * 2^(-15));
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e2_ctfa_out.bin'], int32(y_e2));
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e2.bin'], int32(y_e2));

    % --- XMB1: PConv0(24→32 grouped) → shuffle → nonTConv(32→32,k=1×5) → y_tconv (32,33) → PConv1(BN) → cTFA → y_e3 ---
    % PConv0 expanded: pconv2d(2 groups) → BN → AffinePReLU
    e3_p0_conv_w = importdata('encoder_en_convs_3_pconv1_0_weight.mat');
    e3_p0_conv_b = importdata('encoder_en_convs_3_pconv1_0_bias.mat');
    e3_p0_bn_w   = importdata('encoder_en_convs_3_pconv1_1_weight.mat');
    e3_p0_bn_b   = importdata('encoder_en_convs_3_pconv1_1_bias.mat');
    e3_p0_bn_m   = importdata('encoder_en_convs_3_pconv1_1_running_mean.mat');
    e3_p0_bn_v   = importdata('encoder_en_convs_3_pconv1_1_running_var.mat');
    e3_p0_ap_w   = importdata('encoder_en_convs_3_pconv1_2_affine_weight.mat');
    e3_p0_ap_b   = importdata('encoder_en_convs_3_pconv1_2_affine_bias.mat');
    e3_p0_ap_s   = importdata('encoder_en_convs_3_pconv1_2_slope_weight.mat');

    e3_p0_c0 = pconv2d_func(y_e2(1:12,:), 12, 16, 1, 33, e3_p0_conv_w(1:16,:),  e3_p0_conv_b(1:16),  -13);
    e3_p0_c1 = pconv2d_func(y_e2(13:24,:),12, 16, 1, 33, e3_p0_conv_w(17:32,:), e3_p0_conv_b(17:32), -13);
    e3_p0_conv = cat(1, e3_p0_c0, e3_p0_c1);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e3_p0_conv.bin'], int32(e3_p0_conv));  % after pconv2d+bias

    e3_p0_bn = bn_func(e3_p0_conv, e3_p0_bn_w, e3_p0_bn_b, e3_p0_bn_m, e3_p0_bn_v, -11, -14);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e3_p0_bn.bin'], int32(e3_p0_bn));      % after BN

    y_e3_pconv0 = affineprelu_func(e3_p0_bn, e3_p0_ap_w, e3_p0_ap_b, e3_p0_ap_s, -13, -13);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e3_pconv0.bin'], int32(y_e3_pconv0));  % after AffinePReLU

    y_e3_s = zeros(32, 33);
    y_e3_s(1:2:end, :) = y_e3_pconv0(1:16, :);
    y_e3_s(2:2:end, :) = y_e3_pconv0(17:32, :);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e3_shuf.bin'], int32(y_e3_s));

    y_e3_tconv = XMB1_nonTConv_block(y_e3_s);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e3_tconv.bin'], int32(y_e3_tconv));

    y_e3_pconv1 = XMB1_PConv_block_1(y_e3_tconv);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e3_pconv1.bin'], int32(y_e3_pconv1));
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e3_ctfa_in.bin'], int32(y_e3_pconv1));  % cTFA input
    [y_e3_ta, tfa_cache_e3] = XMB1_cTFA_ta_module(y_e3_pconv1, tfa_cache_e3);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e3_ctfa_ta.bin'], uint16(y_e3_ta));     % ta gate
    y_e3_fa = XMB1_cTFA_fa_module(y_e3_pconv1);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e3_ctfa_fa.bin'], uint16(y_e3_fa));     % fa gate
    y_e3_t = round(y_e3_pconv1 .* (y_e3_ta.') * 2^(-15));
    y_e3_t = round(y_e3_t .* y_e3_fa * 2^(-15));
    y_e3 = zeros(32, 33);
    y_e3(1:2:end, :) = y_e3_t(1:16, :);
    y_e3(2:2:end, :) = y_e3_t(17:32, :);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e3_ctfa_out.bin'], int32(y_e3_t));      % after cTFA, before shuffle
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e3.bin'], int32(y_e3));

    % --- XDWS1: PConv(32→16 grouped) → shuffle → nonTConv(16→16,k=1×5) → y_tconv (16,33) → cTFA → y_e4 ---
    y_e4_pconv = XDWS1_PConv_block(y_e3);
    y_e4_s = zeros(16, 33);
    y_e4_s(1:2:end, :) = y_e4_pconv(1:8, :);
    y_e4_s(2:2:end, :) = y_e4_pconv(9:16, :);
    y_e4_tconv = XDWS1_nonTConv_block(y_e4_s);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e4_tconv.bin'], int32(y_e4_tconv));

    write_bin([out_dir 'frame' num2str(t-1) '_enc_e4_ctfa_in.bin'], int32(y_e4_tconv));
    [y_e4_ta, tfa_cache_e4] = XDWS1_cTFA_ta_module(y_e4_tconv, tfa_cache_e4);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e4_ctfa_ta.bin'], uint16(y_e4_ta));
    y_e4_fa = XDWS1_cTFA_fa_module(y_e4_tconv);
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e4_ctfa_fa.bin'], uint16(y_e4_fa));
    y_e4_t = round(y_e4_tconv .* (y_e4_ta.') * 2^(-15));
    y_e4 = round(y_e4_t .* y_e4_fa * 2^(-15));
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e4_ctfa_out.bin'], int32(y_e4));
    write_bin([out_dir 'frame' num2str(t-1) '_enc_e4.bin'], int32(y_e4));

    % ====================================================================
    % GDPRNN Block 0 (rnn1) — expanded for sub-step diagnostics
    % ====================================================================
    % Load intra weights
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

    % Transpose: (16,33)→(33,16)
    x_dprnn = y_e4.';
    write_bin([out_dir 'frame' num2str(t-1) '_rnn1_intra_in.bin'], int32(x_dprnn));

    % Intra: split → BiGRU ×2 → concat → FC → LN → residual
    x0 = x_dprnn(:, 1:8); x1 = x_dprnn(:, 9:16);
    nH = 4;
    x0_gru = BiGRU_module(x0, nH, ir0_ih_w, ir0_ih_b, ir0_hh_w, ir0_hh_b, ir0_re_ih_w, ir0_re_ih_b, ir0_re_hh_w, ir0_re_hh_b, -13, -8);
    x1_gru = BiGRU_module(x1, nH, ir1_ih_w, ir1_ih_b, ir1_hh_w, ir1_hh_b, ir1_re_ih_w, ir1_re_ih_b, ir1_re_hh_w, ir1_re_hh_b, -13, -8);
    write_bin([out_dir 'frame' num2str(t-1) '_rnn1_intra_gru0.bin'], int16(x0_gru));
    write_bin([out_dir 'frame' num2str(t-1) '_rnn1_intra_gru1.bin'], int16(x1_gru));

    x_cat = cat(2, x0_gru, x1_gru);
    write_bin([out_dir 'frame' num2str(t-1) '_rnn1_intra_cat.bin'], int16(x_cat));

    x_fc = round(x_cat * intra_fc_w * 2^(-9)) + intra_fc_b;
    write_bin([out_dir 'frame' num2str(t-1) '_rnn1_intra_fc.bin'], int32(x_fc));

    x_ln = ln_func(x_fc, intra_ln_w, intra_ln_b, -14);
    write_bin([out_dir 'frame' num2str(t-1) '_rnn1_intra_ln.bin'], int32(x_ln));

    y_intra = x_dprnn + x_ln;
    write_bin([out_dir 'frame' num2str(t-1) '_rnn1_intra_out.bin'], int32(y_intra));

    % Inter: split → GRU ×2 (per-frame) → concat → FC → LN → residual
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

    write_bin([out_dir 'frame' num2str(t-1) '_rnn1_inter_in.bin'], int32(y_intra));

    i0 = y_intra(:, 1:8); i1 = y_intra(:, 9:16);
    i0_gru = GRU_module(i0, 8, inter_cache_0(:,1:8), er0_ih_w, er0_ih_b, er0_hh_w, er0_hh_b, -13, -8);
    i1_gru = GRU_module(i1, 8, inter_cache_0(:,9:16), er1_ih_w, er1_ih_b, er1_hh_w, er1_hh_b, -13, -8);
    write_bin([out_dir 'frame' num2str(t-1) '_rnn1_inter_gru0.bin'], int16(i0_gru));
    write_bin([out_dir 'frame' num2str(t-1) '_rnn1_inter_gru1.bin'], int16(i1_gru));

    i_cat = cat(2, i0_gru, i1_gru);
    write_bin([out_dir 'frame' num2str(t-1) '_rnn1_inter_cat.bin'], int16(i_cat));

    i_fc = round(i_cat * inter_fc_w * 2^(-9)) + inter_fc_b;
    write_bin([out_dir 'frame' num2str(t-1) '_rnn1_inter_fc.bin'], int32(i_fc));

    i_ln = ln_func(i_fc, inter_ln_w, inter_ln_b, -13);
    write_bin([out_dir 'frame' num2str(t-1) '_rnn1_inter_ln.bin'], int32(i_ln));

    y_rnn1 = (y_intra + i_ln).';
    write_bin([out_dir 'frame' num2str(t-1) '_rnn1_inter_out.bin'], int32(y_intra + i_ln));
    write_bin([out_dir 'frame' num2str(t-1) '_rnn1.bin'], int32(y_rnn1));

    % ====================================================================
    % GDPRNN Block 1 (rnn2) — use module call (rnn1 is input)
    % ====================================================================
    [y_rnn2, inter_cache_1] = GDPRNN_module(y_rnn1, inter_cache_1, 1);
    write_bin([out_dir 'frame' num2str(t-1) '_rnn2.bin'], int32(y_rnn2));

    % ====================================================================
    % Decoder — expanded inline to dump TConv intermediates
    % ====================================================================

    % --- De_XDWS0: add(y_rnn2, y_e4) → PConv → shuffle → nonTConv(32→32,k=1×5,s=1) → cTFA → y_d0 ---
    x_cat_d0 = y_rnn2 + y_e4;
    y_d0_pconv = De_XDWS0_PConv_block(x_cat_d0);
    y_d0_s = zeros(32, 33);
    y_d0_s(1:2:end, :) = y_d0_pconv(1:16, :);
    y_d0_s(2:2:end, :) = y_d0_pconv(17:32, :);
    y_d0_tconv = De_XDWS0_nonTConv_block(y_d0_s);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d0_tconv.bin'], int32(y_d0_tconv));

    write_bin([out_dir 'frame' num2str(t-1) '_dec_d0_ctfa_in.bin'], int32(y_d0_tconv));
    [y_d0_ta, tfa_cache_d0] = De_XDWS0_cTFA_ta_module(y_d0_tconv, tfa_cache_d0);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d0_ctfa_ta.bin'], uint16(y_d0_ta));
    y_d0_fa = De_XDWS0_cTFA_fa_module(y_d0_tconv);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d0_ctfa_fa.bin'], uint16(y_d0_fa));
    y_d0_t = round(y_d0_tconv .* (y_d0_ta.') * 2^(-15));
    y_d0 = round(y_d0_t .* y_d0_fa * 2^(-15));
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d0_ctfa_out.bin'], int32(y_d0));
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d0.bin'], int32(y_d0));

    % --- De_XMB0: add(y_d0, y_e3) → PConv0 → shuffle → nonTConv(24→24,k=1×5,s=1) → PConv1 → cTFA → y_d1 ---
    x_cat_d1 = y_d0 + y_e3;
    y_d1_pconv0 = De_XMB0_PConv_block_0(x_cat_d1);
    y_d1_s = zeros(24, 33);
    y_d1_s(1:2:end, :) = y_d1_pconv0(1:12, :);
    y_d1_s(2:2:end, :) = y_d1_pconv0(13:24, :);
    y_d1_tconv = De_XMB0_nonTConv_block(y_d1_s);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d1_tconv.bin'], int32(y_d1_tconv));

    y_d1_pconv1 = De_XMB0_PConv_block_1(y_d1_tconv);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d1_ctfa_in.bin'], int32(y_d1_pconv1));
    [y_d1_ta, tfa_cache_d1] = De_XMB0_cTFA_ta_module(y_d1_pconv1, tfa_cache_d1);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d1_ctfa_ta.bin'], uint16(y_d1_ta));
    y_d1_fa = De_XMB0_cTFA_fa_module(y_d1_pconv1);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d1_ctfa_fa.bin'], uint16(y_d1_fa));
    y_d1_t = round(y_d1_pconv1 .* (y_d1_ta.') * 2^(-15));
    y_d1_t = round(y_d1_t .* y_d1_fa * 2^(-15));
    y_d1 = zeros(24, 33);
    y_d1(1:2:end, :) = y_d1_t(1:12, :);
    y_d1(2:2:end, :) = y_d1_t(13:24, :);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d1_ctfa_out.bin'], int32(y_d1));
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d1.bin'], int32(y_d1));

    % --- De_XDWS1: add(y_d1, y_e2) → PConv → shuffle → GTConv(24→24,k=1×3,s=1,cache) → cTFA → y_d2 ---
    x_cat_d2 = y_d1 + y_e2;
    y_d2_pconv = De_XDWS1_PConv_block(x_cat_d2);
    y_d2_s = zeros(24, 33);
    y_d2_s(1:2:end, :) = y_d2_pconv(1:12, :);
    y_d2_s(2:2:end, :) = y_d2_pconv(13:24, :);
    [y_d2_tconv, conv_cache_d0] = De_XDWS1_TConv_block(y_d2_s, conv_cache_d0);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d2_tconv.bin'], int32(y_d2_tconv));

    write_bin([out_dir 'frame' num2str(t-1) '_dec_d2_ctfa_in.bin'], int32(y_d2_tconv));
    [y_d2_ta, tfa_cache_d2] = De_XDWS1_cTFA_ta_module(y_d2_tconv, tfa_cache_d2);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d2_ctfa_ta.bin'], uint16(y_d2_ta));
    y_d2_fa = De_XDWS1_cTFA_fa_module(y_d2_tconv);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d2_ctfa_fa.bin'], uint16(y_d2_fa));
    y_d2_t = round(y_d2_tconv .* (y_d2_ta.') * 2^(-15));
    y_d2 = round(y_d2_t .* y_d2_fa * 2^(-15));
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d2_ctfa_out.bin'], int32(y_d2));
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d2.bin'], int32(y_d2));

    % --- De_XMB1: add(y_d2, y_e1) → PConv0 → shuffle → GTConv(12→12,k=1×3,s=2,cache) → PConv1 → cTFA → y_d3 ---
    x_cat_d3 = y_d2 + y_e1;
    y_d3_pconv0 = De_XMB1_PConv_block_0(x_cat_d3);
    y_d3_s = zeros(12, 33);
    y_d3_s(1:2:end, :) = y_d3_pconv0(1:6, :);
    y_d3_s(2:2:end, :) = y_d3_pconv0(7:12, :);
    [y_d3_tconv, conv_cache_d1] = De_XMB1_TConv_block(y_d3_s, conv_cache_d1);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d3_tconv.bin'], int32(y_d3_tconv));

    y_d3_pconv1 = De_XMB1_PConv_block_1(y_d3_tconv);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d3_ctfa_in.bin'], int32(y_d3_pconv1));
    [y_d3_ta, tfa_cache_d3] = De_XMB1_cTFA_ta_module(y_d3_pconv1, tfa_cache_d3);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d3_ctfa_ta.bin'], uint16(y_d3_ta));
    y_d3_fa = De_XMB1_cTFA_fa_module(y_d3_pconv1);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d3_ctfa_fa.bin'], uint16(y_d3_fa));
    y_d3_t = round(y_d3_pconv1 .* (y_d3_ta.') * 2^(-15));
    y_d3_t = round(y_d3_t .* y_d3_fa * 2^(-15));
    y_d3 = zeros(12, 65);
    y_d3(1:2:end, :) = y_d3_t(1:6, :);
    y_d3(2:2:end, :) = y_d3_t(7:12, :);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d3_ctfa_out.bin'], int32(y_d3));
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d3.bin'], int32(y_d3));

    % --- De_XConv: add(y_d3, y_e0) → GTConv(12→1,k=3×3,s=2,cache,3D) → cTFA → y_dec ---
    x_cat_d4 = y_d3 + y_e0;
    [y_d4_tconv, conv_cache_d2] = De_XConv_TConv_block(x_cat_d4, conv_cache_d2);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d4_tconv.bin'], int32(y_d4_tconv));

    write_bin([out_dir 'frame' num2str(t-1) '_dec_d4_ctfa_in.bin'], int32(y_d4_tconv));
    [y_d4_ta, tfa_cache_d4] = De_XConv_cTFA_ta_module(y_d4_tconv, tfa_cache_d4);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d4_ctfa_ta.bin'], uint16(y_d4_ta));
    y_d4_fa = De_XConv_cTFA_fa_module(y_d4_tconv);
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d4_ctfa_fa.bin'], uint16(y_d4_fa));
    y_d4_t = round(y_d4_tconv .* (y_d4_ta.') * 2^(-15));
    y_dec = round(y_d4_t .* y_d4_fa * 2^(-15));
    write_bin([out_dir 'frame' num2str(t-1) '_dec_d4_ctfa_out.bin'], int32(y_dec));

    % Sigmoid
    y_dec_dq = y_dec * 2^(-20);
    y_sig_dq = sigmoid_func(y_dec_dq);
    y_sig = Fix_point(y_sig_dq, 'u16f15');

    write_bin([out_dir 'frame' num2str(t-1) '_dec.bin'], int32(y_dec));
    write_bin([out_dir 'frame' num2str(t-1) '_sig.bin'], uint16(y_sig));

    % BS
    y_bs = BS_module(y_sig, ierbfc_weight);
    write_bin([out_dir 'frame' num2str(t-1) '_bs.bin'], int16(y_bs));

    % MASK (MASK_module returns float y_q*2^(-20); re-quantize to s32f20 for C)
    y_mask_q = cat(1, ...
        round(spec_real .* y_bs * 2^(-15)), ...
        round(spec_imag .* y_bs * 2^(-15)));
    write_bin([out_dir 'frame' num2str(t-1) '_mask.bin'], int32(y_mask_q));
end

fprintf('Golden outputs saved to %s\n', out_dir);

%% Helper: write binary file
function write_bin(filename, data)
    fid = fopen(filename, 'wb');
    fwrite(fid, data(:), class(data));
    fclose(fid);
end
