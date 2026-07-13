close all;clear;clc;

addpath 'test_wavs';

[noisy_audio, fs] = audioread('noisy_fileid_1.wav');

%% STFT ----------------------------------------------------------------- %
N_fft = 512;
win_len = 512;
win_inc = 256;
hann_window = importdata('stft_window.mat');

[cmp_real,cmp_imag] = STFT_func(noisy_audio.', N_fft, win_len, win_inc, hann_window);
