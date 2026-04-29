function [X_normalized, scaling_params] = normalize_features(X, method)
% NORMALIZE_FEATURES  Scale feature matrix using the specified method.
%
%   [X_norm, params] = normalize_features(X, method)
%
%   Methods:
%     'standardize' (default) - zero mean, unit variance (z-score)
%     'minmax'                - scale each feature to [0, 1]
%     'robust'                - median centering, IQR scaling

    if nargin < 2
        method = 'standardize';
    end

    scaling_params = struct('method', method);

    switch lower(method)
        case 'standardize'
            scaling_params.mean = mean(X);
            scaling_params.std  = std(X);
            scaling_params.std(scaling_params.std == 0) = 1;
            X_normalized = (X - scaling_params.mean) ./ scaling_params.std;

        case 'minmax'
            scaling_params.min   = min(X);
            scaling_params.max   = max(X);
            scaling_params.range = scaling_params.max - scaling_params.min;
            scaling_params.range(scaling_params.range == 0) = 1;
            X_normalized = (X - scaling_params.min) ./ scaling_params.range;

        case 'robust'
            scaling_params.median = median(X);
            scaling_params.iqr    = iqr(X);
            scaling_params.iqr(scaling_params.iqr == 0) = 1;
            X_normalized = (X - scaling_params.median) ./ scaling_params.iqr;

        otherwise
            error('normalize_features: unknown method "%s". Choose standardize, minmax, or robust.', method);
    end
end