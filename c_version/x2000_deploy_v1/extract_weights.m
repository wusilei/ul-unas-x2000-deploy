% extract_weights.m — Export ALL UL-UNAS weights to C header
% ==============================================================
% Auto-discovers .mat files in para_in_mat_FP/, infers Q-format,
% quantizes to fixed-point, writes C arrays to ulunas_matlab_weights.h
%
% Usage: cd to UL-UNAS_SE_FPversion_v2/, run this script.

clear; clc;
addpath('para_in_mat_FP');

out_path = 'c_version/x2000_deploy_v1/ulunas_matlab_weights.h';
fid = fopen(out_path, 'w');
fprintf(fid, '/**\n * ulunas_matlab_weights.h — Auto-generated %s\n */\n\n', datestr(now));
fprintf(fid, '#ifndef ULUNAS_MATLAB_WEIGHTS_H\n#define ULUNAS_MATLAB_WEIGHTS_H\n\n#include <stdint.h>\n\n');

% Auto-discover all .mat files
files = dir('para_in_mat_FP/*.mat');
fprintf('Found %d .mat files\n', length(files));

count = 0;
for i = 1:length(files)
    name = files(i).name(1:end-4);  % strip .mat
    try
        s = load(['para_in_mat_FP/' files(i).name]);
        data = double(s.(name));
    catch
        fprintf('  SKIP (load error): %s\n', name);
        continue;
    end

    % Infer Q-format from filename suffix patterns
    q_fmt = infer_qformat(name, data);

    % Quantize and write
    switch q_fmt
        case 's16f14'; arr = q_int16(data, 14); write_int16(fid, name, arr);
        case 's16f13'; arr = q_int16(data, 13); write_int16(fid, name, arr);
        case 's16f12'; arr = q_int16(data, 12); write_int16(fid, name, arr);
        case 's16f10'; arr = q_int16(data, 10); write_int16(fid, name, arr);
        case 'u16f14'; arr = q_uint16(data, 14); write_uint16(fid, name, arr);
        case 'u16f13'; arr = q_uint16(data, 13); write_uint16(fid, name, arr);
        case 'u16f12'; arr = q_uint16(data, 12); write_uint16(fid, name, arr);
        case 'u16f15'; arr = q_uint16(data, 15); write_uint16(fid, name, arr);
        case 's32f20'; arr = q_int32(data, 20); write_int32(fid, name, arr);
        case 'skip'
            fprintf('  SKIP: %s (unknown format, range [%.2g,%.2g])\n', name, min(data(:)), max(data(:)));
            continue;
    end
    count = count + 1;
    if mod(count, 20) == 0; fprintf('  %d exported...\n', count); end
end

fprintf(fid, '#endif\n');
fclose(fid);
fprintf('Done! %d weights exported to %s\n', count, out_path);

% ===============================================================
% Q-format inference function
% ===============================================================
function q_fmt = infer_qformat(name, data)
    rng = max(abs(data(:)));

    % Check if unsigned (all values >= 0)
    is_unsigned = all(data(:) >= 0);

    % Heuristic rules based on filename pattern and value range
    % GRU bias: C code expects int32_t (s32f20)
    if contains(name, 'bias') && (contains(name, 'gru') || contains(name, 'rnn'))
        q_fmt = 's32f20'; return;
    end

    % GRU/RNN weights: s16f14
    if (contains(name, 'gru') || contains(name, 'rnn')) && contains(name, 'weight')
        q_fmt = 's16f14'; return;
    end

    % BN running_var: unsigned, small
    if contains(name, 'running_var')
        if rng < 0.01; q_fmt = 'u16f14';
        elseif rng < 0.1; q_fmt = 'u16f13';
        else; q_fmt = 'u16f12';
        end
        return;
    end

    % BN running_mean: s32f20
    if contains(name, 'running_mean')
        q_fmt = 's32f20'; return;
    end

    % BN weight: unsigned
    if contains(name, 'bn_') && contains(name, 'weight')
        q_fmt = 'u16f14'; return;
    end

    % ERB filterbank: u16f15 (must come BEFORE fc_weight/LN/affine checks)
    if contains(name, 'erb_')
        q_fmt = 'u16f15'; return;
    end

    % BN/LN/FC bias: s32f20
    if contains(name, 'bias') && (contains(name, 'bn_') || contains(name, 'ln_') || contains(name, 'fc_') || contains(name, 'affine_'))
        q_fmt = 's32f20'; return;
    end

    % FC weights: s16f13
    if contains(name, 'fc_weight')
        q_fmt = 's16f13'; return;
    end

    % LN weights: s16f12
    if contains(name, 'ln_weight')
        q_fmt = 's16f12'; return;
    end

    % Affine/slope weights (exclude bias): s16f13
    if ~contains(name, 'bias') && (contains(name, 'affine') || contains(name, 'slope'))
        q_fmt = 's16f13'; return;
    end

    % Conv weight (exclude bias): s16f14
    if ~contains(name, 'bias') && (contains(name, 'conv') || contains(name, 'dconv') || contains(name, 'pconv') || contains(name, 'ops_1'))
        q_fmt = 's16f14'; return;
    end

    % Conv bias: s32f20
    if contains(name, 'bias')
        q_fmt = 's32f20'; return;
    end

    % Default: skip unknown
    q_fmt = 'skip';
end

% ===============================================================
% Quantization helpers
% Detect already-quantized FP values (integers in range) to
% avoid double-quantization from para_in_mat_FP/
% ===============================================================
function arr = q_int16(mat, fb)
    if all(mod(mat(:), 1) == 0) && max(abs(mat(:))) <= 32767
        arr = int16(mat);  % already quantized, just cast
    else
        s = 2^fb; arr = round(mat * s); arr = max(min(arr, 32767), -32768); arr = int16(arr);
    end
end

function arr = q_uint16(mat, fb)
    if all(mod(mat(:), 1) == 0) && max(mat(:)) <= 65535
        arr = uint16(mat);  % already quantized, just cast
    else
        s = 2^fb; arr = round(mat * s); arr = max(min(arr, 65535), 0); arr = uint16(arr);
    end
end

function arr = q_int32(mat, fb)
    if all(mod(mat(:), 1) == 0) && max(abs(mat(:))) <= 2147483647
        arr = int32(mat);  % already quantized, just cast
    else
        s = 2^fb; arr = round(mat * s); arr = max(min(arr, 2147483647), -2147483648); arr = int32(arr);
    end
end

function write_int16(fid, name, arr)
    n = numel(arr);
    fprintf(fid, 'static const int16_t %s[%d] = {\n', name, n);
    for j = 1:n
        if mod(j,12)==1; fprintf(fid, '    '); end
        fprintf(fid, '%d', arr(j));
        if j<n; fprintf(fid, ','); end
        if mod(j,12)==0; fprintf(fid, '\n'); end
    end
    if mod(n,12)~=0; fprintf(fid, '\n'); end
    fprintf(fid, '};\n\n');
end

function write_uint16(fid, name, arr)
    n = numel(arr);
    fprintf(fid, 'static const uint16_t %s[%d] = {\n', name, n);
    for j = 1:n
        if mod(j,12)==1; fprintf(fid, '    '); end
        fprintf(fid, '%d', arr(j));
        if j<n; fprintf(fid, ','); end
        if mod(j,12)==0; fprintf(fid, '\n'); end
    end
    if mod(n,12)~=0; fprintf(fid, '\n'); end
    fprintf(fid, '};\n\n');
end

function write_int32(fid, name, arr)
    n = numel(arr);
    fprintf(fid, 'static const int32_t %s[%d] = {\n', name, n);
    for j = 1:n
        if mod(j,8)==1; fprintf(fid, '    '); end
        fprintf(fid, '%d', arr(j));
        if j<n; fprintf(fid, ','); end
        if mod(j,8)==0; fprintf(fid, '\n'); end
    end
    if mod(n,8)~=0; fprintf(fid, '\n'); end
    fprintf(fid, '};\n\n');
end
