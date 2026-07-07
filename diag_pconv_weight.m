addpath('para_in_mat_FP');
s = load('para_in_mat_FP/encoder_en_convs_3_pconv1_0_weight.mat');
fn = fieldnames(s); w = s.(fn{1});
fprintf('Weight [1:4]=%d %d %d %d\n', w(1), w(2), w(3), w(4));

s = load('para_in_mat_FP/encoder_en_convs_3_pconv1_0_bias.mat');
fn = fieldnames(s); b = s.(fn{1});
fprintf('Bias [1:4]=%d %d %d %d\n', b(1), b(2), b(3), b(4));
