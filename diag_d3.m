% diag_d3.m — Export D3 (De_XMB1) intermediate golden for C debugging
close all; clear; clc;
addpath('para_in_mat_FP'); addpath('test_wavs');
out_dir = 'D:\haidesi\haidesi\ul-unas-x2000-deploy\UL-UNAS_SE_FPversion_v2\c_version\x2000_deploy_v2\dump_matlab\';

erbfc_weight = importdata('erb_erb_fc_weight.mat');
[noisy_audio, fs] = audioread('noisy_fileid_1.wav');
N_fft = 512; win_len = 512; win_inc = 256;
hann_window = importdata('stft_window.mat');
[cmp_real, cmp_imag] = STFT_func(noisy_audio.', N_fft, win_len, win_inc, hann_window);
t = 1;
fprintf('=== D3 De_XMB1 Diagnostic — Frame %d ===\n', t-1);

spec_real = Fix_point(cmp_real(t, :), 's32f20');
spec_imag = Fix_point(cmp_imag(t, :), 's32f20');
x_log = log_gen(spec_real, spec_imag);
x_bm = BM_module(x_log, erbfc_weight);

% Full encoder
conv_cache_e0 = zeros(2, 129); conv_cache_e1 = zeros(24, 65); conv_cache_e2 = zeros(24, 33);
tfa_cache_e0 = zeros(1, 24); tfa_cache_e1 = zeros(1, 48); tfa_cache_e2 = zeros(1, 48);
tfa_cache_e3 = zeros(1, 64); tfa_cache_e4 = zeros(1, 32);
[y_e0, conv_cache_e0, tfa_cache_e0] = XConv_module(x_bm, conv_cache_e0, tfa_cache_e0);
[y_e1, conv_cache_e1, tfa_cache_e1] = XMB0_module(y_e0, conv_cache_e1, tfa_cache_e1);
[y_e2, conv_cache_e2, tfa_cache_e2] = XDWS0_module(y_e1, conv_cache_e2, tfa_cache_e2);
y_e3 = XMB1_module(y_e2, tfa_cache_e3);
y_e4 = XDWS1_module(y_e3, tfa_cache_e4);

% RNN (inline, matching export_all_layers_v2.m)
x_dprnn = y_e4.';
ir0_ih_w=importdata('dpgrnn_0_intra_rnn_rnn1_weight_ih_l0.mat');
ir0_ih_b=importdata('dpgrnn_0_intra_rnn_rnn1_bias_ih_l0.mat');
ir0_hh_w=importdata('dpgrnn_0_intra_rnn_rnn1_weight_hh_l0.mat');
ir0_hh_b=importdata('dpgrnn_0_intra_rnn_rnn1_bias_hh_l0.mat');
ir0_re_ih_w=importdata('dpgrnn_0_intra_rnn_rnn1_weight_ih_l0_reverse.mat');
ir0_re_ih_b=importdata('dpgrnn_0_intra_rnn_rnn1_bias_ih_l0_reverse.mat');
ir0_re_hh_w=importdata('dpgrnn_0_intra_rnn_rnn1_weight_hh_l0_reverse.mat');
ir0_re_hh_b=importdata('dpgrnn_0_intra_rnn_rnn1_bias_hh_l0_reverse.mat');
ir1_ih_w=importdata('dpgrnn_0_intra_rnn_rnn2_weight_ih_l0.mat');
ir1_ih_b=importdata('dpgrnn_0_intra_rnn_rnn2_bias_ih_l0.mat');
ir1_hh_w=importdata('dpgrnn_0_intra_rnn_rnn2_weight_hh_l0.mat');
ir1_hh_b=importdata('dpgrnn_0_intra_rnn_rnn2_bias_hh_l0.mat');
ir1_re_ih_w=importdata('dpgrnn_0_intra_rnn_rnn2_weight_ih_l0_reverse.mat');
ir1_re_ih_b=importdata('dpgrnn_0_intra_rnn_rnn2_bias_ih_l0_reverse.mat');
ir1_re_hh_w=importdata('dpgrnn_0_intra_rnn_rnn2_weight_hh_l0_reverse.mat');
ir1_re_hh_b=importdata('dpgrnn_0_intra_rnn_rnn2_bias_hh_l0_reverse.mat');
intra_fc_w=importdata('dpgrnn_0_intra_fc_weight.mat');
intra_fc_b=importdata('dpgrnn_0_intra_fc_bias.mat');
intra_ln_w=importdata('dpgrnn_0_intra_ln_weight.mat');
intra_ln_b=importdata('dpgrnn_0_intra_ln_bias.mat');
x0=x_dprnn(:,1:8); x1=x_dprnn(:,9:16);
x0_gru=BiGRU_module(x0,4,ir0_ih_w,ir0_ih_b,ir0_hh_w,ir0_hh_b,ir0_re_ih_w,ir0_re_ih_b,ir0_re_hh_w,ir0_re_hh_b,-13,-8);
x1_gru=BiGRU_module(x1,4,ir1_ih_w,ir1_ih_b,ir1_hh_w,ir1_hh_b,ir1_re_ih_w,ir1_re_ih_b,ir1_re_hh_w,ir1_re_hh_b,-13,-8);
x_cat=cat(2,x0_gru,x1_gru);
x_fc=round(x_cat*intra_fc_w*2^(-9))+intra_fc_b;
x_ln=ln_func(x_fc,intra_ln_w,intra_ln_b,-14);
y_intra=x_dprnn+x_ln;
er0_ih_w=importdata('dpgrnn_0_inter_rnn_rnn1_weight_ih_l0.mat');
er0_ih_b=importdata('dpgrnn_0_inter_rnn_rnn1_bias_ih_l0.mat');
er0_hh_w=importdata('dpgrnn_0_inter_rnn_rnn1_weight_hh_l0.mat');
er0_hh_b=importdata('dpgrnn_0_inter_rnn_rnn1_bias_hh_l0.mat');
er1_ih_w=importdata('dpgrnn_0_inter_rnn_rnn2_weight_ih_l0.mat');
er1_ih_b=importdata('dpgrnn_0_inter_rnn_rnn2_bias_ih_l0.mat');
er1_hh_w=importdata('dpgrnn_0_inter_rnn_rnn2_weight_hh_l0.mat');
er1_hh_b=importdata('dpgrnn_0_inter_rnn_rnn2_bias_hh_l0.mat');
inter_fc_w=importdata('dpgrnn_0_inter_fc_weight.mat');
inter_fc_b=importdata('dpgrnn_0_inter_fc_bias.mat');
inter_ln_w=importdata('dpgrnn_0_inter_ln_weight.mat');
inter_ln_b=importdata('dpgrnn_0_inter_ln_bias.mat');
inter_cache_0=zeros(33,16);
i0=y_intra(:,1:8); i1=y_intra(:,9:16);
[i0_gru,inter_cache_0(:,1:8)]=GRU_module(i0,8,inter_cache_0(:,1:8),er0_ih_w,er0_ih_b,er0_hh_w,er0_hh_b,-13,-8);
[i1_gru,inter_cache_0(:,9:16)]=GRU_module(i1,8,inter_cache_0(:,9:16),er1_ih_w,er1_ih_b,er1_hh_w,er1_hh_b,-13,-8);
i_cat=cat(2,i0_gru,i1_gru);
i_fc=round(i_cat*inter_fc_w*2^(-9))+inter_fc_b;
i_ln=ln_func(i_fc,inter_ln_w,inter_ln_b,-13);
y_rnn1=(y_intra+i_ln).';
inter_cache_1=zeros(33,16);
[y_rnn2,inter_cache_1]=GDPRNN_module(y_rnn1,inter_cache_1,1);

% Decoder D0-D2 (to get golden D2 output as D3 input)
conv_cache_d0=zeros(24,33); conv_cache_d1=zeros(12,33); conv_cache_d2=zeros(12,2,65);
tfa_cache_d0=zeros(1,64); tfa_cache_d1=zeros(1,48); tfa_cache_d2=zeros(1,48);
tfa_cache_d3=zeros(1,24);
[y_d0,tfa_cache_d0]=De_XDWS0_module(y_rnn2,y_e4,tfa_cache_d0);
[y_d1,tfa_cache_d1]=De_XMB0_module(y_d0,y_e3,tfa_cache_d1);
[y_d2,conv_cache_d0,tfa_cache_d2]=De_XDWS1_module(y_d1,y_e2,conv_cache_d0,tfa_cache_d2);

%% D3: De_XMB1 — skip(y_e1) → PConv0 → Shuffle → GTConv(stride=2) → PConv1 → cTFA
x_cat_d3 = y_d2 + y_e1;

% PConv0: groups=2, Cin=12→Cout=6 each
d3_p0_w=importdata('decoder_de_convs_3_pconv1_0_weight.mat');
d3_p0_b=importdata('decoder_de_convs_3_pconv1_0_bias.mat');
d3_p0_bn_w=importdata('decoder_de_convs_3_pconv1_1_weight.mat');
d3_p0_bn_b=importdata('decoder_de_convs_3_pconv1_1_bias.mat');
d3_p0_bn_m=importdata('decoder_de_convs_3_pconv1_1_running_mean.mat');
d3_p0_bn_v=importdata('decoder_de_convs_3_pconv1_1_running_var.mat');
d3_p0_ap_w=importdata('decoder_de_convs_3_pconv1_2_affine_weight.mat');
d3_p0_ap_b=importdata('decoder_de_convs_3_pconv1_2_affine_bias.mat');
d3_p0_ap_s=importdata('decoder_de_convs_3_pconv1_2_slope_weight.mat');

d3_p0_c0=pconv2d_func(x_cat_d3(1:12,:),12,6,1,33,d3_p0_w(1:6,:),d3_p0_b(1:6),-14);
d3_p0_c1=pconv2d_func(x_cat_d3(13:24,:),12,6,1,33,d3_p0_w(7:12,:),d3_p0_b(7:12),-14);
d3_p0_conv=cat(1,d3_p0_c0,d3_p0_c1);
d3_p0_bn=bn_func(d3_p0_conv,d3_p0_bn_w,d3_p0_bn_b,d3_p0_bn_m,d3_p0_bn_v,-11,-14);
y_d3_pconv0=affineprelu_func(d3_p0_bn,d3_p0_ap_w,d3_p0_ap_b,d3_p0_ap_s,-13,-13);
fprintf('D3 PConv0+BN+AP: [%d x %d]\n',size(y_d3_pconv0,1),size(y_d3_pconv0,2));
write_bin([out_dir 'frame0_d3_pconv0.bin'],int32(y_d3_pconv0));

% Shuffle
y_d3_s=zeros(12,33);
y_d3_s(1:2:end,:)=y_d3_pconv0(1:6,:);
y_d3_s(2:2:end,:)=y_d3_pconv0(7:12,:);
write_bin([out_dir 'frame0_d3_shuf.bin'],int32(y_d3_s));
fprintf('  -> wrote frame0_d3_pconv0.bin, frame0_d3_shuf.bin\n');

fprintf('=== Done ===\n');

function write_bin(filename,data)
fid=fopen(filename,'wb'); fwrite(fid,data(:),class(data)); fclose(fid);
end
