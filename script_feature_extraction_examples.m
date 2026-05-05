function script_feature_extraction_examples()
% SCRIPT_FEATURE_EXTRACTION_EXAMPLES
%   Demonstrates detailed spectral feature extraction on synthetic
%   Trumpet and Guitar signals.
%   (Flute removed — replaced with Guitar for direct comparison.)

    fprintf('=== Feature Extraction Examples ===\n\n');
    fs       = 44100;
    nfft     = 2048;
    hop      = 512;
    duration = 2;
    t        = linspace(0, duration, duration * fs);

    % ---- Trumpet: bright, many harmonics (rolloff exponent = 0.85) --------
    f0_trumpet = 440;
    trumpet    = zeros(size(t));
    for h = 1:15
        trumpet = trumpet + (1/h^0.85) * sin(2*pi*h*f0_trumpet*t);
    end
    env_trumpet = [linspace(0,1,round(fs*0.04)), ...
                   ones(1, duration*fs - round(fs*0.04) - round(fs*0.05)), ...
                   linspace(1,0,round(fs*0.05))];
    env_trumpet = env_trumpet(1:length(t));
    trumpet     = trumpet .* env_trumpet;
    trumpet     = trumpet / max(abs(trumpet));

    % ---- Guitar: plucked, fast decay (rolloff exponent = 1.60) ------------
    f0_guitar = 220;    % A3 — typical open string
    guitar    = zeros(size(t));
    for h = 1:8
        guitar = guitar + (1/h^1.60) * sin(2*pi*h*f0_guitar*t + 2*pi*rand());
    end
    % Exponential decay envelope (fast, like a plucked string)
    decay_rate  = 7.0;
    env_guitar  = exp(-decay_rate * t);
    guitar      = guitar .* env_guitar;
    guitar      = guitar / max(abs(guitar));

    window = make_window(nfft, 'hann');
    [S_trumpet, f, t_spec] = my_stft(trumpet, window, nfft-hop, nfft, fs);
    [S_guitar,  ~, ~      ] = my_stft(guitar,  window, nfft-hop, nfft, fs);
    S_trumpet = abs(S_trumpet);
    S_guitar  = abs(S_guitar);

    % ---- Spectrograms -------------------------------------------------------
    figure('Position', [100 100 1200 400]);

    subplot(1,2,1);
    imagesc(t_spec, f, 20*log10(S_trumpet + 1e-6));
    set(gca,'YDir','normal'); colorbar; caxis([-80 -20]);
    xlabel('Time (s)'); ylabel('Frequency (Hz)');
    title('Trumpet Spectrogram (Bright, Sustained, Many Harmonics)');
    axis([0 duration 0 5000]);

    subplot(1,2,2);
    imagesc(t_spec, f, 20*log10(S_guitar + 1e-6));
    set(gca,'YDir','normal'); colorbar; caxis([-80 -20]);
    xlabel('Time (s)'); ylabel('Frequency (Hz)');
    title('Guitar Spectrogram (Plucked, Fast Decay, Fewer High Harmonics)');
    axis([0 duration 0 5000]);

    sgtitle('Spectrogram Comparison: Trumpet vs Guitar');

    % ---- Frame-level features -----------------------------------------------
    ff_t = extract_frame_features(S_trumpet, f);
    ff_g = extract_frame_features(S_guitar,  f);

    figure('Position', [100 100 1200 600]);
    feature_labels = {'Spectral Centroid', 'Spectral Rolloff', 'Spectral Flux'};
    ylabels        = {'Frequency (Hz)', 'Frequency (Hz)', 'Flux Magnitude'};

    for k = 1:3
        subplot(2,2,k);
        plot(t_spec, ff_t(:,k), 'LineWidth', 2, 'DisplayName', 'Trumpet'); hold on;
        plot(t_spec, ff_g(:,k), 'LineWidth', 2, 'DisplayName', 'Guitar');
        xlabel('Time (s)'); ylabel(ylabels{k});
        title([feature_labels{k} ' Over Time']);
        legend('Location','best'); grid on;
    end

    subplot(2,2,4);
    mean_t = mean(ff_t(:,1:3));
    mean_g = mean(ff_g(:,1:3));
    norm_factor = max([mean_t, mean_g]);
    x = 1:3;
    bar(x - 0.2, mean_t / norm_factor, 0.4, 'DisplayName', 'Trumpet'); hold on;
    bar(x + 0.2, mean_g / norm_factor, 0.4, 'DisplayName', 'Guitar');
    set(gca, 'XTick', x, 'XTickLabel', {'Centroid','Rolloff','Flux'});
    ylabel('Normalized Value'); title('Mean Feature Comparison');
    legend('Location','best'); grid on;

    sgtitle('Feature Dynamics: Trumpet vs Guitar');

    fprintf('Feature extraction visualization complete.\n\n');
end