close all;clear;clc;

%% Enable multi-threaded BLAS/FFT for faster matrix ops
maxNumCompThreads('automatic');
fprintf('CPU threads: %d\n', maxNumCompThreads);

addpath 'para_in_mat_FP';

%% Load ERB
erbfc_weight  = importdata('erb_erb_fc_weight.mat');
ierbfc_weight = importdata('erb_ierb_fc_weight.mat');

%% STFT parameters
N_fft    = 512;
win_len  = 512;
win_inc  = 256;
hann_window = importdata('stft_window.mat');

%% Load noisy audio (8 kHz)
wav_file = 'D:\haidesi\haidesi\session\session_data\session10.wav';
[noisy_audio_raw, fs] = audioread(wav_file);
fprintf('Original: %d Hz, %.2f s\n', fs, length(noisy_audio_raw)/fs);

%% Resample 8k -> 16k (2x)
target_fs = 16000;
N_in = length(noisy_audio_raw);
noisy_audio = spline(1:N_in, noisy_audio_raw, linspace(1, N_in, 2*N_in));
noisy_audio = noisy_audio(:);
fprintf('Resampled to %d Hz, %.2f s\n', target_fs, length(noisy_audio)/target_fs);

%% Temporal caches
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
[cmp_real, cmp_imag] = STFT_func(noisy_audio.', N_fft, win_len, win_inc, hann_window);

%% Frame-by-frame inference
T = size(cmp_real,1);
fprintf('Total frames: %d\n', T);
y_mask_infer = zeros(T,2,257);

t_start = tic;
report_interval = 100;
for t = 1:T
    if mod(t, report_interval) == 0 || t == 1
        fprintf('Frame %d/%d (%.0f%%), elapsed: %.0fs\n', t, T, t/T*100, toc(t_start));
    end

    spec_real = Fix_point(cmp_real(t,:), 's32f20');
    spec_imag = Fix_point(cmp_imag(t,:), 's32f20');

    x = log_gen(spec_real, spec_imag);

    %% BM
    x_bm = BM_module(x, erbfc_weight);

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

    %% G-DPRNN
    [y_rnn1, inter_cache_0] = GDPRNN_module(y_e4, inter_cache_0, 0);
    [y_rnn2, inter_cache_1] = GDPRNN_module(y_rnn1, inter_cache_1, 1);

    %% Decoder
    [y_dec, ...
        tfa_cache_d0, ...
        tfa_cache_d1, ...
        conv_cache_d0, tfa_cache_d2, ...
        conv_cache_d1, tfa_cache_d3, ...
        conv_cache_d2, tfa_cache_d4] = Decoder_module(y_rnn2, y_e4, tfa_cache_d0, ...
        y_e3, tfa_cache_d1, ...
        y_e2, conv_cache_d0, tfa_cache_d2, ...
        y_e1, conv_cache_d1, tfa_cache_d3, ...
        y_e0, conv_cache_d2, tfa_cache_d4);

    %% Sigmoid
    y_dec_dq = y_dec*2^(-20);
    y_sig_dq = sigmoid_func(y_dec_dq);
    y_sig = Fix_point(y_sig_dq, 'u16f15');

    %% BS
    y_bs = BS_module(y_sig, ierbfc_weight);

    %% MASK
    y_mask = MASK_module(y_bs, spec_real, spec_imag);
    y_mask_infer(t,:,:) = y_mask;
end
fprintf('Inference done. Elapsed: %.1f s\n', toc(t_start));

%% ISTFT
I_spec   = squeeze(y_mask_infer(:,1,:));
Q_spec   = squeeze(y_mask_infer(:,2,:));
cmp_spec = I_spec + 1i*Q_spec;

enh_audio = ISTFT_func(cmp_spec, N_fft, win_len, win_inc, hann_window);

%% Save output
out_file = 'test_wavs\enh_session\session10_enh.wav';
audiowrite(out_file, enh_audio, 16000);
fprintf('Saved: %s\n', out_file);
fprintf('Total elapsed: %.1f s\n', toc(t_start));
