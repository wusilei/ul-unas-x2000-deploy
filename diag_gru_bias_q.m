addpath('para_in_mat_FP'); addpath('para_in_mat');

fprintf('=== DPRNN GRU Bias Q-Format Check ===\n\n');

files = {'dpgrnn_0_intra_rnn_rnn1_bias_ih_l0', ...
         'dpgrnn_0_inter_rnn_rnn1_bias_ih_l0', ...
         'encoder_en_convs_1_pconv2_2_ta_gru_bias_ih_l0'};

for i = 1:length(files)
    nm = files{i};
    s = load(['para_in_mat_FP/' nm '.mat']);
    fn = fieldnames(s); v = double(s.(fn{1}));
    fprintf('%s:\n  range=[%d,%d]  size=%s\n', nm, min(v(:)), max(v(:)), mat2str(size(v)));

    try
        s2 = load(['para_in_mat/' nm '.mat']);
        fn2 = fieldnames(s2); v2 = double(s2.(fn2{1}));
        r = v(1) / v2(1);
        fprintf('  FP=%d Float=%.4f ratio=%.0f Q=%.1f\n', v(1), v2(1), r, log2(abs(r)));
        if abs(r - 1024) < 100, fprintf('  → Q10!\n');
        elseif abs(r - 8192) < 1000, fprintf('  → Q13!\n');
        elseif abs(r - 1048576) < 100000, fprintf('  → Q20!\n');
        else, fprintf('  → unknown Q=%.0f\n', r); end
    catch
        fprintf('  (float not available)\n');
    end
    fprintf('\n');
end
