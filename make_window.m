function w = make_window(N, type)
% MAKE_WINDOW  Generate a window vector — replaces hann(), hamming(), blackman().
%              Requires only base MATLAB (no Signal Processing Toolbox).
%
%   w = make_window(N)           % Hann window (default)
%   w = make_window(N, 'hann')
%   w = make_window(N, 'hamming')
%   w = make_window(N, 'blackman')
%   w = make_window(N, 'rect')

    if nargin < 2
        type = 'hann';
    end

    n = (0:N-1)';

    switch lower(type)
        case {'hann', 'hanning'}
            w = 0.5 * (1 - cos(2*pi*n / (N-1)));
        case 'hamming'
            w = 0.54 - 0.46 * cos(2*pi*n / (N-1));
        case 'blackman'
            w = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1));
        case {'rect', 'rectangular'}
            w = ones(N, 1);
        otherwise
            error('make_window: unknown type "%s". Use hann, hamming, blackman, or rect.', type);
    end
end
