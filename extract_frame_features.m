function frame_features = extract_frame_features(S, f)
% EXTRACT_FRAME_FEATURES  Compute per-frame spectral centroid, rolloff, and flux.
%
%   frame_features = extract_frame_features(S, f)
%
%   Inputs:
%     S  - magnitude spectrogram [n_bins x n_frames]
%     f  - frequency vector [n_bins x 1]
%
%   Output:
%     frame_features - [n_frames x 3]: columns are [centroid, rolloff, flux]

    n_frames = size(S, 2);
    frame_features = zeros(n_frames, 3);
    P = S .^ 2;

    for m = 1:n_frames
        P_m      = P(:, m);
        P_norm   = P_m / (sum(P_m) + 1e-10);

        % Spectral centroid
        sc = sum(f .* P_norm);

        % Spectral rolloff (85% energy threshold)
        cdf = cumsum(P_norm);
        idx = find(cdf >= 0.85, 1, 'first');
        if isempty(idx), idx = length(f); end
        sr = f(idx);

        % Spectral flux
        if m > 1
            sf = sqrt(sum((P_m - P(:,m-1)).^2));
        else
            sf = 0;
        end

        frame_features(m, :) = [sc, sr, sf];
    end
end
