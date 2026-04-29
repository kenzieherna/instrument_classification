function [audio_out, fs_out] = preprocess_audio(filepath, target_fs)
% PREPROCESS_AUDIO  Load an audio file, convert to mono, and resample.
%                   Requires only base MATLAB (no Signal Processing Toolbox).
%                   Uses interp1 for resampling instead of resample().
%
%   [audio_out, fs_out] = preprocess_audio(filepath, target_fs)

    if nargin < 2
        target_fs = 44100;
    end

    if ~isfile(filepath)
        error('preprocess_audio: file not found: %s', filepath);
    end

    % ---- Load ---------------------------------------------------------------
    [audio_raw, fs_raw] = audioread(filepath);
    fprintf('  Loaded:   %s\n', filepath);
    fprintf('  Original: %d Hz, %d ch, %.2f s\n', ...
        fs_raw, size(audio_raw, 2), size(audio_raw, 1) / fs_raw);

    % ---- Mono ---------------------------------------------------------------
    if size(audio_raw, 2) > 1
        audio_mono = mean(audio_raw, 2);
        fprintf('  Converted %d channels -> mono.\n', size(audio_raw, 2));
    else
        audio_mono = audio_raw;
    end

    % ---- Resample via linear interpolation (no SPT needed) ------------------
    if fs_raw ~= target_fs
        n_orig   = length(audio_mono);
        n_target = round(n_orig * target_fs / fs_raw);
        t_orig   = linspace(0, 1, n_orig);
        t_new    = linspace(0, 1, n_target);
        audio_rs = interp1(t_orig, audio_mono, t_new, 'linear')';
        fprintf('  Resampled: %d Hz -> %d Hz\n', fs_raw, target_fs);
    else
        audio_rs = audio_mono;
    end

    % ---- Normalize ----------------------------------------------------------
    peak = max(abs(audio_rs));
    if peak < 1e-8
        warning('preprocess_audio: signal is near-silent (peak=%.2e).', peak);
        peak = 1;
    end
    audio_out = audio_rs / peak;
    fs_out    = target_fs;

    fprintf('  Output:   %d Hz, mono, %.2f s\n', fs_out, length(audio_out)/fs_out);
end