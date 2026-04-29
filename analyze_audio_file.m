function result = analyze_audio_file(audio_filepath, model_path)
% ANALYZE_AUDIO_FILE  Segment-level majority vote instrument classifier.

    if nargin < 2 || isempty(model_path)
        model_path = 'instrument_model.mat';
    end

    fprintf('=== Instrument Identification ===\n\n');

    if ~isfile(model_path)
        error('Model not found: "%s"\nRun train_and_save_model() first.', model_path);
    end

    fprintf('Loading model: %s\n', model_path);
    M = load(model_path, 'model_svm', 'model_rf', 'model_knn', ...
             'scaling_params', 'instrument_classes', 'config');
    cfg       = M.config;
    n_classes = length(M.instrument_classes);
    fprintf('Classes: %s\n\n', strjoin(M.instrument_classes, ', '));

    fprintf('--- Preprocessing audio ---\n');
    [audio, ~] = preprocess_audio(audio_filepath, cfg.fs);
    duration   = length(audio) / cfg.fs;
    fprintf('  Duration: %.1f seconds\n\n', duration);

    % ---- Segmentation -------------------------------------------------------
    seg_len  = 2.0;
    seg_hop  = 1.0;
    seg_samp = round(seg_len * cfg.fs);
    hop_samp = round(seg_hop * cfg.fs);
    starts   = 1 : hop_samp : length(audio) - seg_samp + 1;

    fprintf('--- Classifying %d segments (%.1fs each, %.1fs hop) ---\n', ...
        length(starts), seg_len, seg_hop);

    window   = make_window(cfg.nfft, 'hann');
    noverlap = cfg.nfft - cfg.hop;

    seg_labels_svm = zeros(1, length(starts));
    seg_labels_rf  = zeros(1, length(starts));
    seg_labels_knn = zeros(1, length(starts));
    seg_scores_svm = zeros(length(starts), n_classes);
    valid_segs     = false(1, length(starts));

    for i = 1:length(starts)
        seg = audio(starts(i) : starts(i) + seg_samp - 1);

        if local_rms(seg) < 0.01
            continue;
        end
        valid_segs(i) = true;

        [S, f_vec] = my_stft(seg, window, noverlap, cfg.nfft, cfg.fs);
        S_mag = abs(S);
        feat  = extract_features_local(S_mag, f_vec, seg, cfg.fs, cfg.nfft, cfg.hop, cfg.n_mfcc);

        feat_norm = (feat - M.scaling_params.mean) ./ M.scaling_params.std;

        [seg_labels_svm(i), sc] = predict(M.model_svm, feat_norm);
        seg_labels_rf(i)        = predict(M.model_rf,  feat_norm);
        seg_labels_knn(i)       = predict(M.model_knn, feat_norm);
        seg_scores_svm(i,:)     = sc;
    end

    valid_idx = find(valid_segs);
    n_valid   = length(valid_idx);

    if n_valid == 0
        error('All segments were silent. Check your audio file.');
    end

    fprintf('  Valid segments: %d / %d\n\n', n_valid, length(starts));

    labels_svm = seg_labels_svm(valid_idx);
    labels_rf  = seg_labels_rf(valid_idx);
    labels_knn = seg_labels_knn(valid_idx);
    scores_svm = seg_scores_svm(valid_idx, :);

    % ---- Weighted majority vote ---------------------------------------------
    % k-NN gets 2x weight: SVM and RF share a kernel/tree bias that makes
    % them jointly overconfident on synthetic-to-real domain gaps.
    % k-NN is instance-based and tends to be more honest on real audio.
    vote_svm = histcounts(labels_svm, 1:n_classes+1) / n_valid * 100;
    vote_rf  = histcounts(labels_rf,  1:n_classes+1) / n_valid * 100;
    vote_knn = histcounts(labels_knn, 1:n_classes+1) / n_valid * 100;
    combined = vote_svm + vote_rf + 2*vote_knn;   % k-NN weighted 2x

    [~, final_label] = max(combined);
    predicted_class  = M.instrument_classes{final_label};

    % Individual classifier top predictions
    [~, svm_top_idx] = max(vote_svm);
    [~, rf_top_idx]  = max(vote_rf);
    [~, knn_top_idx] = max(vote_knn);

    mean_scores = mean(scores_svm, 1);

    % ---- Print results ------------------------------------------------------
    fprintf('========================================\n');
    fprintf('  Ensemble prediction: %s\n', predicted_class);
    fprintf('========================================\n\n');

    fprintf('Segment vote breakdown (%% of %d segments):\n', n_valid);
    fprintf('  %-12s  %8s  %8s  %8s  %8s\n', 'Instrument','SVM','RF','k-NN','Combined');
    fprintf('  %s\n', repmat('-',1,55));

    [~, order] = sort(combined, 'descend');
    for k = 1:n_classes
        idx    = order(k);
        marker = '';
        if idx == final_label, marker = '  <--'; end
        fprintf('  %-12s  %7.1f%%  %7.1f%%  %7.1f%%  %7.1f%%%s\n', ...
            M.instrument_classes{idx}, vote_svm(idx), vote_rf(idx), ...
            vote_knn(idx), combined(idx)/3, marker);
    end
    fprintf('\n');

    % ---- Plot ---------------------------------------------------------------
    plot_analysis(audio, cfg, predicted_class, mean_scores, ...
        M.instrument_classes, labels_svm, labels_rf, labels_knn, ...
        starts(valid_idx), seg_len, n_classes);

    % ---- Output struct (all fields that run_instrument_pipeline needs) ------
    result = struct( ...
        'predicted_class',    predicted_class, ...
        'svm_top',            M.instrument_classes{svm_top_idx}, ...
        'rf_top',             M.instrument_classes{rf_top_idx}, ...
        'knn_top',            M.instrument_classes{knn_top_idx}, ...
        'vote_pct_svm',       vote_svm, ...
        'vote_pct_rf',        vote_rf, ...
        'vote_pct_knn',       vote_knn, ...
        'mean_svm_scores',    mean_scores, ...
        'n_valid_segments',   n_valid, ...
        'instrument_classes', {M.instrument_classes});
end


% =========================================================================
%  LOCAL FEATURE EXTRACTION — must exactly mirror create_synthetic_dataset
% =========================================================================

function feat = extract_features_local(S_mag, f_vec, sig, fs, nfft, hop, n_mfcc)
    n_frames = size(S_mag, 2);
    P        = S_mag .^ 2;

    sc_all   = zeros(1, n_frames);
    sr_all   = zeros(1, n_frames);
    sf_all   = zeros(1, n_frames);
    zcr_all  = zeros(1, n_frames);
    flat_all = zeros(1, n_frames);

    for m = 1:n_frames
        P_m    = P(:, m);
        P_norm = P_m / (sum(P_m) + 1e-10);

        sc_all(m) = sum(f_vec .* P_norm);

        cdf = cumsum(P_norm);
        idx = find(cdf >= 0.85, 1, 'first');
        if isempty(idx), idx = length(f_vec); end
        sr_all(m) = f_vec(idx);

        if m > 1
            sf_all(m) = sum((S_mag(:,m) - S_mag(:,m-1)).^2);
        end

        frame_start = (m-1)*hop + 1;
        frame_end   = min(frame_start + nfft - 1, length(sig));
        frame       = sig(frame_start:frame_end);
        if length(frame) > 1
            zcr_all(m) = sum(frame(1:end-1) .* frame(2:end) < 0) / length(frame);
        end

        gm = exp(mean(log(P_m + 1e-10)));
        am = mean(P_m) + 1e-10;
        flat_all(m) = gm / am;
    end

    rms_frames = sqrt(mean(P, 1));
    edr   = compute_energy_decay_rate_local(rms_frames);
    lat   = compute_log_attack_time_local(sig, fs);
    burst = std(rms_frames) / (mean(rms_frames) + 1e-10);

    % Spectral band energy ratios (must match create_synthetic_dataset exactly)
    P_mean      = mean(P, 2);
    total_power = sum(P_mean) + 1e-10;
    bands = [0, 250, 500, 1000, 2000, 4000, f_vec(end)];
    band_ratios = zeros(1, length(bands)-1);
    for b = 1:length(bands)-1
        idx_b = f_vec >= bands(b) & f_vec < bands(b+1);
        band_ratios(b) = sum(P_mean(idx_b)) / total_power;
    end
    low_band_ratio  = sum(P_mean(f_vec < 700))  / total_power;
    high_band_ratio = sum(P_mean(f_vec > 2000)) / total_power;

    mfcc_mat = compute_mfcc_local(S_mag, f_vec, fs, nfft, n_mfcc, 40);

    agg = @(x) [mean(x), std(x), median(x)];

    feat = [agg(sc_all), agg(sr_all), agg(sf_all), agg(zcr_all), ...
            agg(flat_all), edr, lat, burst, ...
            band_ratios, low_band_ratio, high_band_ratio, ...
            agg_rows_local(mfcc_mat)];
end


function edr = compute_energy_decay_rate_local(rms_frames)
    log_rms  = log(rms_frames + 1e-10);
    [~, peak_idx] = max(log_rms);
    tail   = log_rms(peak_idx:end);
    n_tail = length(tail);
    if n_tail < 3, edr = 0; return; end
    x     = (1:n_tail)';
    slope = (n_tail*sum(x.*tail(:)) - sum(x)*sum(tail(:))) / ...
            (n_tail*sum(x.^2) - sum(x)^2);
    edr   = slope;
end


function lat = compute_log_attack_time_local(sig, fs)
    env        = abs(sig);
    win        = round(0.020 * fs);
    env_smooth = conv(env, ones(1,win)/win, 'same');
    peak_val   = max(env_smooth);
    if peak_val < 1e-8, lat = -3; return; end
    onset_idx = find(env_smooth >= 0.1*peak_val, 1, 'first');
    peak_idx  = find(env_smooth >= 0.9*peak_val, 1, 'first');
    if isempty(onset_idx) || isempty(peak_idx) || peak_idx <= onset_idx
        lat = -3; return;
    end
    lat = log10((peak_idx - onset_idx)/fs + 1e-5);
end


function out = agg_rows_local(M)
    out = [];
    for r = 1:size(M,1)
        row = M(r,:);
        out = [out, mean(row), std(row), median(row)]; %#ok<AGROW>
    end
end


function mfcc_matrix = compute_mfcc_local(S_mag, f_vec, fs, nfft, n_mfcc, n_filters)
    if nargin < 6, n_filters = 40; end
    mel  = @(f) 2595 * log10(1 + f / 700);
    imel = @(m) 700 * (10.^(m / 2595) - 1);
    mel_pts = linspace(mel(0), mel(fs/2), n_filters + 2);
    hz_pts  = imel(mel_pts);
    bin_pts = max(1, min(round((nfft+1) * hz_pts / fs), size(S_mag,1)));
    H = zeros(n_filters, size(S_mag,1));
    for m = 1:n_filters
        lo = bin_pts(m); ctr = bin_pts(m+1); hi = bin_pts(m+2);
        for k = lo:ctr
            if ctr > lo, H(m,k) = (k-lo)/(ctr-lo); end
        end
        for k = ctr:hi
            if hi > ctr, H(m,k) = (hi-k)/(hi-ctr); end
        end
    end
    log_mel     = log(H * (S_mag.^2) + 1e-10);
    mfcc_matrix = zeros(n_mfcc, size(S_mag,2));
    for i = 1:n_mfcc
        mfcc_matrix(i,:) = sum(log_mel .* cos(pi*(i-1)/n_filters * ((1:n_filters)'-0.5)), 1);
    end
end


function v = local_rms(x)
    v = sqrt(mean(x.^2));
end


% =========================================================================
function plot_analysis(audio, cfg, predicted_class, mean_scores, ...
                       classes, labels_svm, labels_rf, labels_knn, ...
                       seg_starts, seg_len, n_classes)

    t_audio   = (0:length(audio)-1) / cfg.fs;
    n_valid   = length(labels_svm);
    seg_times = (seg_starts - 1) / cfg.fs + seg_len/2;

    figure('Name','Instrument Analysis','Position',[60 60 1400 750]);

    % Waveform
    subplot(3,1,1);
    plot(t_audio, audio, 'Color',[0.7 0.7 0.7], 'LineWidth', 0.5);
    hold on;
    cmap = lines(n_classes);
    h = zeros(n_classes,1);
    for c = 1:n_classes
        h(c) = plot(nan, nan, 's', 'MarkerFaceColor', cmap(c,:), ...
            'MarkerEdgeColor', cmap(c,:), 'DisplayName', classes{c});
    end
    for i = 1:n_valid
        lbl = labels_svm(i);
        plot(seg_times(i), 0.85, 's', 'MarkerFaceColor', cmap(lbl,:), ...
            'MarkerEdgeColor', cmap(lbl,:), 'MarkerSize', 7);
    end
    xlabel('Time (s)'); ylabel('Amplitude');
    title(sprintf('Waveform — per-segment SVM labels  (ensemble: %s)', predicted_class));
    legend(h, classes, 'Location','northeast', 'FontSize', 8);
    grid on; xlim([0, t_audio(end)]);

    % Vote bars
    subplot(3,2,3);
    vote_svm = histcounts(labels_svm, 1:n_classes+1) / n_valid * 100;
    vote_rf  = histcounts(labels_rf,  1:n_classes+1) / n_valid * 100;
    vote_knn = histcounts(labels_knn, 1:n_classes+1) / n_valid * 100;
    x = 1:n_classes;
    bar(x-0.25, vote_svm, 0.25, 'DisplayName','SVM');  hold on;
    bar(x,      vote_rf,  0.25, 'DisplayName','RF');
    bar(x+0.25, vote_knn, 0.25, 'DisplayName','k-NN');
    set(gca,'XTick',x,'XTickLabel',classes); xtickangle(25);
    ylabel('% of segments'); ylim([0 105]);
    title('Segment Vote Distribution'); legend; grid on;

    % Mean SVM confidence
    subplot(3,2,4);
    [scores_sorted, sort_idx] = sort(mean_scores, 'descend');
    class_labels = classes(sort_idx);
    bar_colors   = repmat([0.6 0.8 0.95], n_classes, 1);
    bar_colors(1,:) = [0.2 0.65 0.3];
    b = bar(scores_sorted * 100, 'FaceColor','flat');
    b.CData = bar_colors;
    set(gca,'XTick',1:n_classes,'XTickLabel',class_labels); xtickangle(25);
    ylabel('Mean SVM Confidence (%)'); ylim([0 105]);
    title('Mean SVM Confidence'); grid on;

    % Timeline
    subplot(3,1,3);
    for i = 1:n_valid
        lbl = labels_svm(i);
        patch([seg_times(i)-seg_len/2, seg_times(i)+seg_len/2, ...
               seg_times(i)+seg_len/2, seg_times(i)-seg_len/2], ...
              [0 0 1 1], cmap(lbl,:), 'EdgeColor','none', 'FaceAlpha',0.7);
        hold on;
    end
    set(gca,'YTick',0.5,'YTickLabel','SVM');
    xlabel('Time (s)'); xlim([0, t_audio(end)]); ylim([0 1]);
    title('SVM Classification Timeline');
    legend(h, classes, 'Location','northeast', 'FontSize', 8);

    sgtitle(sprintf('Segment Analysis  —  Prediction: %s  (%d segments)', ...
        predicted_class, n_valid), 'FontSize',14, 'FontWeight','bold');
end