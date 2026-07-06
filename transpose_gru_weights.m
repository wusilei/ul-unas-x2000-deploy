% transpose_gru_weights.m — Transpose GRU/RNN weight .mat files back to MATLAB-native shape
% para_in_mat_FP has transposed shapes (C-compatible, e.g., [24,144]).
% Restore to original float shape (e.g., [144,24]) so that column-major
% binary export matches C row-major indexing: weight[i * stride + h].
%
% Usage: cd to UL-UNAS_SE_FPversion_v2/, run this script.

addpath('para_in_mat_FP');

files = dir('para_in_mat_FP/*gru_weight*');
files = [files; dir('para_in_mat_FP/*rnn*rnn*weight*')];
fprintf('Found %d GRU/RNN weight files\n', length(files));

for i = 1:length(files)
    fname = files(i).name;
    nm = fname(1:end-4);
    s = load(['para_in_mat_FP/' fname]);
    w = s.(nm);
    sz = size(w);
    if sz(1) < sz(2)
        % Transposed shape (input_dim, hidden*3) → restore (hidden*3, input_dim)
        w_t = w.';
        s.(nm) = w_t;
        save(['para_in_mat_FP/' fname], '-struct', 's', nm);
        fprintf('  Transposed: %s (%s → %s)\n', nm, mat2str(sz), mat2str(size(w_t)));
    else
        fprintf('  Skip (already MATLAB-native): %s (%s)\n', nm, mat2str(sz));
    end
end

fprintf('Done!\n');
