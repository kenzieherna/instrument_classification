function [S, f, t] = my_stft(x, window, noverlap, nfft, fs)
% MY_STFT  Short-Time Fourier Transform — replaces spectrogram().
%          Requires only base MATLAB (no Signal Processing Toolbox).
%
%   [S, f, t] = my_stft(x, window, noverlap, nfft, fs)
%
%   Inputs:
%     x        - input signal (column or row vector)
%     window   - window vector (e.g. from make_window())
%     noverlap - number of overlapping samples between frames
%     nfft     - FFT size
%     fs       - sampling rate (Hz)
%
%   Outputs:
%     S - complex STFT matrix [nfft/2+1 x n_frames]
%     f - frequency vector (Hz)
%     t - time vector (s) at frame centres

    x = x(:);                          % ensure column vector
    win_len  = length(window);
    hop      = win_len - noverlap;
    n_frames = floor((length(x) - win_len) / hop) + 1;
    n_bins   = nfft / 2 + 1;

    S = zeros(n_bins, n_frames);
    for i = 1:n_frames
        idx   = (i-1)*hop + 1 : (i-1)*hop + win_len;
        frame = x(idx) .* window(:);
        X     = fft(frame, nfft);
        S(:, i) = X(1:n_bins);
    end

    f = (0 : n_bins-1)' * fs / nfft;
    t = ((0:n_frames-1) * hop + win_len/2) / fs;
end
