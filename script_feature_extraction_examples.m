function script_feature_extraction_examples()
% SCRIPT_FEATURE_EXTRACTION_EXAMPLES
%   Demonstrates detailed spectral feature extraction on synthetic trumpet/flute signals.

    fprintf('=== Feature Extraction Examples ===\n\n');

    fs   = 44100;
    nfft = 2048;
    hop  = 512;

    duration = 2;
    t = linspace(0, duration, duration * fs);

    % Trumpet: bright, many harmonics (rolloff exponent = 1.2)
    f0 = 440;
    trumpet = zeros(size(t));
    for h = 1:15
        trumpet = trumpet + (1/h^1.2) * sin(2*pi*h*f0*t);
    end
    env = [linspace(0,1,fs/2), ones(1,fs), linspace(1,0,fs/2)];
    trumpet = trumpet .* env;
    trumpet = trumpet / max(abs(trumpet));

    % Flute: mellow, fewer harmonics (rolloff exponent = 1.8)
    flute = zeros(size(t));
    for h = 1:6
        flute = flute + (1/h^1.8) * sin(2*pi*h*f0*t);
    end
    flute = flute .* env;
    flute = flute / max(abs(flute));

    window = hann(nfft);
    [S_trumpet, f, t_spec] = spectrogram(trumpet, window, nfft-hop, nfft, fs);
    [S_flute,  ~, ~       ] = spectrogram(flute,   window, nfft-hop, nfft, fs);
    S_trumpet = abs(S_trumpet);
    S_flute   = abs(S_flute);

    % ---- Spectrograms -------------------------------------------------------
    figure('Position', [100 100 1200 400]);

    subplot(1,2,1);
    imagesc(t_spec, f, 20*log10(S_trumpet + 1e-6));
    set(gca,'YDir','normal'); colorbar; caxis([-80 -20]);
    xlabel('Time (s)'); ylabel('Frequency (Hz)');
    title('Trumpet Spectrogram (Bright, Many Harmonics)');
    axis([0 duration 0 5000]);

    subplot(1,2,2);
    imagesc(t_spec, f, 20*log10(S_flute + 1e-6));
    set(gca,'YDir','normal'); colorbar; caxis([-80 -20]);
    xlabel('Time (s)'); ylabel('Frequency (Hz)');
    title('Flute Spectrogram (Mellow, Fewer High Harmonics)');
    axis([0 duration 0 5000]);

    sgtitle('Spectrogram Comparison: Trumpet vs Flute');

    % ---- Frame-level features -----------------------------------------------
    ff_t = extract_frame_features(S_trumpet, f);
    ff_f = extract_frame_features(S_flute,   f);

    figure('Position', [100 100 1200 600]);

    feature_labels = {'Spectral Centroid', 'Spectral Rolloff', 'Spectral Flux'};
    ylabels = {'Frequency (Hz)', 'Frequency (Hz)', 'Flux Magnitude'};

    for k = 1:3
        subplot(2,2,k);
        plot(t_spec, ff_t(:,k), 'LineWidth', 2, 'DisplayName', 'Trumpet'); hold on;
        plot(t_spec, ff_f(:,k), 'LineWidth', 2, 'DisplayName', 'Flute');
        xlabel('Time (s)'); ylabel(ylabels{k});
        title([feature_labels{k} ' Over Time']);
        legend('Location','best'); grid on;
    end

    subplot(2,2,4);
    mean_t = mean(ff_t(:,1:3));
    mean_f = mean(ff_f(:,1:3));
    norm_factor = max([mean_t, mean_f]);
    x = 1:3;
    bar(x - 0.2, mean_t / norm_factor, 0.4, 'DisplayName', 'Trumpet'); hold on;
    bar(x + 0.2, mean_f / norm_factor, 0.4, 'DisplayName', 'Flute');
    set(gca, 'XTick', x, 'XTickLabel', {'Centroid','Rolloff','Flux'});
    ylabel('Normalized Value'); title('Mean Feature Comparison');
    legend('Location','best'); grid on;

    sgtitle('Feature Dynamics: Trumpet vs Flute');
    fprintf('Feature extraction visualization complete.\n\n');
end