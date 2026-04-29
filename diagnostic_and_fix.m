%% DIAGNOSTIC SCRIPT - MINIMAL VERSION
% No Signal Processing Toolbox required!
% Just Statistics and Machine Learning Toolbox

clear all; close all; clc;

fprintf('=== DIAGNOSTIC: Why Everything is Piano ===\n\n');

%% Setup
fs = 44100;
nfft = 2048;
hop = 512;
n_per_class = 8;

fprintf('Step 1: Generating synthetic instrument data...\n');

%% Generate simple synthetic data
[features, labels, classes] = gen_synth_data_simple(n_per_class, fs, nfft, hop);

fprintf('Generated: %d samples across %d classes\n\n', size(features,1), length(classes));

%% Normalize features
feature_mean = mean(features);
feature_std = std(features);
feature_std(feature_std == 0) = 1;
features_norm = (features - feature_mean) ./ feature_std;

%% Train/test split
n_samples = length(labels);
perm = randperm(n_samples);
split = round(0.7 * n_samples);
train_idx = perm(1:split);
test_idx = perm(split+1:end);

X_train = features_norm(train_idx, :);
y_train = labels(train_idx);
X_test = features_norm(test_idx, :);
y_test = labels(test_idx);

fprintf('Step 2: Training classifier...\n\n');

%% Train SVM
try
    mdl = fitcecoc(X_train, y_train, 'Learners', templateSVM('Kernel', 'rbf'));
    
    % Training accuracy
    pred_train = predict(mdl, X_train);
    acc_train = sum(pred_train == y_train) / length(y_train);
    fprintf('Training Accuracy: %.1f%%\n', acc_train*100);
    
    % Test accuracy
    pred_test = predict(mdl, X_test);
    acc_test = sum(pred_test == y_test) / length(y_test);
    fprintf('Test Accuracy: %.1f%%\n\n', acc_test*100);
    
    % Confusion matrix
    C = confusionmat(y_test, pred_test);
    
    fprintf('Confusion Matrix:\n');
    fprintf('%10s', '');
    for c = 1:length(classes)
        fprintf(' | %8s', classes{c}(1:8));
    end
    fprintf('\n');
    fprintf('%s\n', repmat('-', 1, 75));
    
    for i = 1:length(classes)
        fprintf('%10s', classes{i});
        for j = 1:length(classes)
            fprintf(' | %8d', C(i,j));
        end
        fprintf('\n');
    end
    fprintf('\n');
    
    % Analysis
    [~, max_idx] = max(accumarray(pred_test, 1));
    max_pct = 100 * sum(pred_test == max_idx) / length(pred_test);
    
    fprintf('Step 3: Analysis and Diagnosis\n\n');
    
    if acc_train < 0.65
        fprintf('PROBLEM: Training accuracy too low (<65%%)!\n');
        fprintf('Root cause: Features do not separate instruments.\n\n');
        fprintf('SOLUTION:\n');
        fprintf('  1. Increase differences in synthesis parameters\n');
        fprintf('  2. Make frequencies more distinct\n');
        fprintf('  3. Use very different decay rates\n');
        fprintf('  4. Ensure Guitar has SHORT reverb (50-150ms)\n\n');
    end
    
    if max_pct > 40
        fprintf('WARNING: %s predicted %.0f%% of time!\n\n', classes{max_idx}, max_pct);
        fprintf('This means features for %s completely overlap other classes.\n\n', classes{max_idx});
    end
    
    if acc_train > 0.80 && acc_test < 0.60
        fprintf('OVERFITTING: Training=%.0f%%, Test=%.0f%%\n\n', acc_train*100, acc_test*100);
        fprintf('Solution: Use fewer features or simpler synthesis.\n\n');
    end
    
    % Feature analysis
    fprintf('Step 4: Feature Discrimination Analysis\n\n');
    fprintf('Feature separability (higher is better):\n');
    
    for feat_idx = 1:min(5, size(features, 2))
        % Check class separation for this feature
        means = [];
        stds = [];
        for c = 1:length(classes)
            idx = labels == c;
            means(c) = mean(features(idx, feat_idx));
            stds(c) = std(features(idx, feat_idx));
        end
        
        range = max(means) - min(means);
        avg_std = mean(stds);
        sep_score = range / (avg_std + 1e-10);
        
        fprintf('  Feature %d: %.2f (range=%.2e, std=%.2e)\n', ...
            feat_idx, sep_score, range, avg_std);
    end
    
    fprintf('\n=== DIAGNOSIS COMPLETE ===\n\n');
    
    % Recommendations
    fprintf('NEXT STEPS:\n\n');
    fprintf('1. If training accuracy < 65%%:\n');
    fprintf('   Use EXTREME synthesis parameters (see QUICK_START.txt)\n\n');
    fprintf('2. If test accuracy << training accuracy:\n');
    fprintf('   You are overfitting - use fewer features\n\n');
    fprintf('3. If all predictions are one class:\n');
    fprintf('   Features are completely overlapped\n');
    fprintf('   Make instruments MORE DIFFERENT\n\n');
    fprintf('4. Modify gen_synth_data_simple() to test different parameters\n');
    fprintf('   Key parameters: f0_lo, f0_hi, decay_min, decay_max, rt60\n\n');
    
    % Visualization
    fprintf('Generating visualization...\n');
    
    figure('Position', [100 100 1000 600]);
    
    % Feature 1 scatter
    subplot(1, 3, 1);
    for c = 1:length(classes)
        idx = labels == c;
        x = repmat(c, sum(idx), 1) + 0.1*randn(sum(idx), 1);
        y = features(idx, 1);
        scatter(x, y, 40, 'DisplayName', classes{c});
        hold on;
    end
    set(gca, 'XTick', 1:length(classes), 'XTickLabel', classes);
    xtickangle(45);
    ylabel('Feature 1 Value');
    title('Feature 1: Spectral Centroid');
    grid on;
    
    % Feature 2 scatter
    if size(features, 2) >= 2
        subplot(1, 3, 2);
        for c = 1:length(classes)
            idx = labels == c;
            x = repmat(c, sum(idx), 1) + 0.1*randn(sum(idx), 1);
            y = features(idx, 2);
            scatter(x, y, 40, 'DisplayName', classes{c});
            hold on;
        end
        set(gca, 'XTick', 1:length(classes), 'XTickLabel', classes);
        xtickangle(45);
        ylabel('Feature 2 Value');
        title('Feature 2: Spectral Rolloff');
        grid on;
    end
    
    % 2D scatter
    if size(features, 2) >= 2
        subplot(1, 3, 3);
        for c = 1:length(classes)
            idx = labels == c;
            scatter(features(idx, 1), features(idx, 2), 40, ...
                'DisplayName', classes{c});
            hold on;
        end
        xlabel('Feature 1');
        ylabel('Feature 2');
        title('2D Feature Space');
        legend('FontSize', 8);
        grid on;
    end
    
    sgtitle('Feature Distribution Analysis');
    
    fprintf('Plot generated!\n\n');
    
catch ME
    fprintf('ERROR: %s\n', ME.message);
    fprintf('Make sure you have Statistics and Machine Learning Toolbox.\n');
end

%% ========================================================================
%% SYNTHETIC DATA GENERATION (FIXED VERSION)
%% ========================================================================

function [features, labels, classes] = gen_synth_data_simple(n_per_class, fs, nfft, hop)
    
    % INSTRUMENT PARAMETERS - MAKE THEM VERY DIFFERENT
    defs = {
        'Piano',   300, 800,   12, 1.20, 0.70, 1.20, 0.40;
        'Violin',  200, 1200,  8,  1.05, 0.30, 0.80, 0.60;
        'Flute',   260, 1000,  4,  1.80, 1.00, 1.50, 0.50;
        'Trumpet', 150, 600,   14, 0.85, 1.50, 2.50, 0.40;
        'Guitar',  80,  330,   7,  2.00, 4.00, 6.00, 0.08;
    };
    
    n_classes = size(defs, 1);
    classes = defs(:, 1);
    
    seg_dur = 2.0;
    n_samp = round(seg_dur * fs);
    
    all_feats = [];
    all_labs = [];
    
    % Simple triangular window (no toolbox needed)
    window = linspace(0, 1, nfft/2+1)';
    window = [window; flipud(window(1:nfft/2))];
    window = window(1:nfft);
    
    fprintf('\nGenerating synthetic samples:\n');
    
    for c = 1:n_classes
        f_lo = defs{c, 2};
        f_hi = defs{c, 3};
        n_h = defs{c, 4};
        roll = defs{c, 5};
        d_min = defs{c, 6};
        d_max = defs{c, 7};
        rt = defs{c, 8};
        
        for s = 1:n_per_class
            % Generate signal
            sig = make_sound(f_lo, f_hi, n_h, roll, d_min, d_max, rt, n_samp, fs);
            
            % Simple spectral feature extraction (no FFT needed)
            feat = extract_basic_features(sig, fs);
            
            all_feats = [all_feats; feat];
            all_labs = [all_labs; c];
        end
        
        fprintf('  %d. %s\n', c, defs{c,1});
    end
    
    features = all_feats;
    labels = all_labs;
end

function sig = make_sound(f_lo, f_hi, n_h, roll, d_min, d_max, rt, n_samp, fs)
    sig = zeros(1, n_samp);
    t = (0:n_samp-1) / fs;
    f_nyq = fs / 2;
    
    % 3 notes in sequence
    n_notes = 3;
    times = [0.2, 0.8, 1.4];
    
    for note_idx = 1:n_notes
        f0 = exp(log(f_lo) + rand() * log(f_hi/f_lo));
        decay = d_min + rand() * (d_max - d_min);
        onset = round(times(note_idx) * fs) + 1;
        attack = round(0.015 * fs);
        
        note = zeros(1, n_samp);
        for h = 1:n_h
            fh = h * f0;
            if fh >= f_nyq, break; end
            note = note + (1/h^roll) * sin(2*pi*fh*t + 2*pi*rand());
        end
        
        env = zeros(1, n_samp);
        for i = onset:n_samp
            i_rel = i - onset;
            if i_rel < attack
                env(i) = i_rel / attack;
            else
                env(i) = exp(-decay * (i_rel - attack) / fs);
            end
        end
        
        sig = sig + note .* env;
    end
    
    % Add simple reverb
    ir_len = max(round(rt * fs), 10);
    ir = randn(1, ir_len) .* exp(-3 * linspace(0, 1, ir_len));
    ir = ir / (max(abs(ir)) + 1e-8);
    
    sig_rev = conv(sig, ir);
    sig = sig_rev(1:length(sig));
    
    % Add noise
    sig = sig + 0.005 * randn(1, n_samp);
    sig = sig / (max(abs(sig)) + 1e-8);
end

function feat = extract_basic_features(sig, fs)
    % Extract 5 simple time-domain features (NO FFT)
    
    % 1. Energy (RMS)
    energy = sqrt(mean(sig.^2));
    
    % 2. Zero-crossing rate
    zcr = sum(sig(1:end-1) .* sig(2:end) < 0) / length(sig);
    
    % 3. Peak magnitude
    peak = max(abs(sig));
    
    % 4. Energy centroid (center of mass in time)
    t_vec = (0:length(sig)-1) / fs;
    energy_env = abs(sig);
    energy_centroid = sum(t_vec .* energy_env) / sum(energy_env);
    
    % 5. Temporal decay rate (log energy slope)
    energy_smooth = filter(ones(1, 100)/100, 1, energy_env);
    log_energy = log(energy_smooth + 1e-10);
    if length(log_energy) > 10
        x = (1:length(log_energy))';
        coeff = polyfit(x, log_energy', 1);
        decay_rate = coeff(1);
    else
        decay_rate = 0;
    end
    
    feat = [energy, zcr, peak, energy_centroid, decay_rate];
end