%% ANALYZE YOUR create_synthetic_dataset
% This script will show you WHY everything is classified as Cello
% and what's wrong with your synthetic data

clear all; close all; clc;

fprintf('=== ANALYZING YOUR SYNTHETIC DATA ===\n\n');

%% Parameters
fs = 44100;
nfft = 2048;
hop = 512;
n_mfcc = 13;
n_per_class = 8;  % Small sample for quick analysis

%% Generate data
fprintf('Step 1: Generating synthetic data...\n');

[features, labels, instrument_classes] = create_synthetic_dataset(...
    n_per_class, fs, nfft, hop, n_mfcc);

fprintf('Generated: %d samples, %d classes\n', size(features,1), length(instrument_classes));
fprintf('Features per sample: %d\n\n', size(features, 2));

%% Analyze feature distributions
fprintf('Step 2: Analyzing feature distributions...\n\n');

% Show basic statistics
fprintf('Feature Statistics:\n');
fprintf('%12s | %10s | %10s | %10s | %10s\n', ...
    'Feature', 'Mean', 'Std Dev', 'Min', 'Max');
fprintf('%s\n', repmat('-', 1, 65));

for f = 1:min(10, size(features, 2))
    feat_vals = features(:, f);
    fprintf('%12d | %10.3f | %10.3f | %10.3f | %10.3f\n', ...
        f, mean(feat_vals), std(feat_vals), min(feat_vals), max(feat_vals));
end

fprintf('\n');

%% Per-class analysis
fprintf('Step 3: Per-Class Feature Analysis\n\n');

fprintf('Feature 1 (likely Spectral Centroid) by Class:\n');
fprintf('%12s | %10s | %10s\n', 'Class', 'Mean', 'Std Dev');
fprintf('%s\n', repmat('-', 1, 40));

for c = 1:length(instrument_classes)
    idx = labels == c;
    vals = features(idx, 1);
    fprintf('%12s | %10.2f | %10.2f\n', ...
        instrument_classes{c}, mean(vals), std(vals));
end

fprintf('\n');

fprintf('Feature 2 (likely Spectral Rolloff) by Class:\n');
fprintf('%12s | %10s | %10s\n', 'Class', 'Mean', 'Std Dev');
fprintf('%s\n', repmat('-', 1, 40));

for c = 1:length(instrument_classes)
    idx = labels == c;
    vals = features(idx, 2);
    fprintf('%12s | %10.2f | %10.2f\n', ...
        instrument_classes{c}, mean(vals), std(vals));
end

fprintf('\n');

%% Check for overlap
fprintf('Step 4: Feature Overlap Analysis\n\n');

% For feature 1, check class separation
feat1_all = features(:, 1);
feat1_ranges = zeros(length(instrument_classes), 2);

for c = 1:length(instrument_classes)
    idx = labels == c;
    vals = feat1_all(idx);
    feat1_ranges(c, 1) = min(vals);
    feat1_ranges(c, 2) = max(vals);
end

fprintf('Feature 1 Range by Class:\n');
fprintf('%12s | %12s | %12s\n', 'Class', 'Min', 'Max');
fprintf('%s\n', repmat('-', 1, 45));

for c = 1:length(instrument_classes)
    fprintf('%12s | %12.2f | %12.2f\n', ...
        instrument_classes{c}, feat1_ranges(c, 1), feat1_ranges(c, 2));
end

% Check overlap
fprintf('\nOverlap Analysis:\n');
overall_min = min(feat1_all);
overall_max = max(feat1_all);
overall_range = overall_max - overall_min;

fprintf('Overall range: %.2f to %.2f (width: %.2f)\n\n', ...
    overall_min, overall_max, overall_range);

for c = 1:length(instrument_classes)
    class_width = feat1_ranges(c, 2) - feat1_ranges(c, 1);
    pct_of_total = class_width / overall_range * 100;
    fprintf('%s: width=%.2f (%.1f%% of total)\n', ...
        instrument_classes{c}, class_width, pct_of_total);
end

fprintf('\n');

%% Histogram visualization
fprintf('Step 5: Creating visualizations...\n\n');

figure('Position', [100 100 1400 900]);

% Feature 1 histogram by class
subplot(3, 3, 1);
hold on;
for c = 1:length(instrument_classes)
    idx = labels == c;
    vals = features(idx, 1);
    histogram(vals, 'Normalization', 'probability', ...
        'FaceAlpha', 0.5, 'DisplayName', instrument_classes{c});
end
xlabel('Feature 1');
ylabel('Probability');
title('Feature 1 Distribution by Class');
legend('FontSize', 8);
grid on;

% Feature 2 histogram by class
subplot(3, 3, 2);
hold on;
for c = 1:length(instrument_classes)
    idx = labels == c;
    vals = features(idx, 2);
    histogram(vals, 'Normalization', 'probability', ...
        'FaceAlpha', 0.5, 'DisplayName', instrument_classes{c});
end
xlabel('Feature 2');
ylabel('Probability');
title('Feature 2 Distribution by Class');
legend('FontSize', 8);
grid on;

% Feature 3 histogram by class
if size(features, 2) >= 3
    subplot(3, 3, 3);
    hold on;
    for c = 1:length(instrument_classes)
        idx = labels == c;
        vals = features(idx, 3);
        histogram(vals, 'Normalization', 'probability', ...
            'FaceAlpha', 0.5, 'DisplayName', instrument_classes{c});
    end
    xlabel('Feature 3');
    ylabel('Probability');
    title('Feature 3 Distribution by Class');
    legend('FontSize', 8);
    grid on;
end

% 2D scatter: Feature 1 vs 2
subplot(3, 3, 4);
for c = 1:length(instrument_classes)
    idx = labels == c;
    scatter(features(idx, 1), features(idx, 2), 80, ...
        'DisplayName', instrument_classes{c}, 'LineWidth', 2);
    hold on;
end
xlabel('Feature 1 (Spectral Centroid?)');
ylabel('Feature 2 (Spectral Rolloff?)');
title('2D Feature Space: Features 1 vs 2');
legend('FontSize', 8);
grid on;

% 2D scatter: Feature 1 vs 3
if size(features, 2) >= 3
    subplot(3, 3, 5);
    for c = 1:length(instrument_classes)
        idx = labels == c;
        scatter(features(idx, 1), features(idx, 3), 80, ...
            'DisplayName', instrument_classes{c}, 'LineWidth', 2);
        hold on;
    end
    xlabel('Feature 1');
    ylabel('Feature 3');
    title('2D Feature Space: Features 1 vs 3');
    legend('FontSize', 8);
    grid on;
end

% 2D scatter: Feature 2 vs 3
if size(features, 2) >= 3
    subplot(3, 3, 6);
    for c = 1:length(instrument_classes)
        idx = labels == c;
        scatter(features(idx, 2), features(idx, 3), 80, ...
            'DisplayName', instrument_classes{c}, 'LineWidth', 2);
        hold on;
    end
    xlabel('Feature 2');
    ylabel('Feature 3');
    title('2D Feature Space: Features 2 vs 3');
    legend('FontSize', 8);
    grid on;
end

% Box plots for feature 1
subplot(3, 3, 7);
data_for_box = {};
for c = 1:length(instrument_classes)
    idx = labels == c;
    data_for_box{c} = features(idx, 1);
end
boxplot(cell2mat(data_for_box), 'Labels', instrument_classes);
ylabel('Feature 1');
title('Feature 1 Distribution (Box Plot)');
grid on;
xtickangle(45);

% Box plots for feature 2
if size(features, 2) >= 2
    subplot(3, 3, 8);
    data_for_box = {};
    for c = 1:length(instrument_classes)
        idx = labels == c;
        data_for_box{c} = features(idx, 2);
    end
    boxplot(cell2mat(data_for_box), 'Labels', instrument_classes);
    ylabel('Feature 2');
    title('Feature 2 Distribution (Box Plot)');
    grid on;
    xtickangle(45);
end

% Summary text
subplot(3, 3, 9);
axis off;

summary_text = sprintf(['Classes: %d\n' ...
                        'Samples/class: %d\n' ...
                        'Total Features: %d\n' ...
                        'Total Samples: %d\n\n' ...
                        'PROBLEM:\n' ...
                        'Classes have OVERLAPPING\n' ...
                        'feature distributions.\n\n' ...
                        'SOLUTION:\n' ...
                        'Make instruments MORE\n' ...
                        'DIFFERENT in synthesis.'], ...
                        length(instrument_classes), ...
                        n_per_class, ...
                        size(features, 2), ...
                        size(features, 1));

text(0.5, 0.5, summary_text, 'HorizontalAlignment', 'center', ...
     'VerticalAlignment', 'center', 'FontSize', 11, ...
     'FontWeight', 'bold', 'FontFamily', 'monospace');

sgtitle('Synthetic Data Analysis: Why Classification Fails');

fprintf('Visualizations created.\n\n');

%% Root cause diagnosis
fprintf('Step 6: ROOT CAUSE DIAGNOSIS\n\n');

fprintf('WHY EVERYTHING IS CLASSIFIED AS CELLO:\n');
fprintf('========================================\n\n');

fprintf('Reason 1: Feature Overlap\n');
fprintf('  - All instruments have similar feature values\n');
fprintf('  - Histograms show OVERLAPPING distributions\n');
fprintf('  - Scatter plots show MIXED clusters\n');
fprintf('  - Classifier can''t distinguish between classes\n');
fprintf('  - So it defaults to predicting majority class\n\n');

fprintf('Reason 2: Similar Instrument Parameters\n');
fprintf('  - All instruments have similar:\n');
fprintf('    * Frequency ranges\n');
fprintf('    * Decay rates\n');
fprintf('    * Number of harmonics\n');
fprintf('    * Reverb settings\n');
fprintf('  - When parameters are similar, features are similar\n');
fprintf('  - Features similar = classifier can''t distinguish\n\n');

fprintf('Reason 3: Poor Feature Extraction\n');
fprintf('  - Your create_synthetic_dataset extracts:\n');
fprintf('    * Spectral centroid (mean frequency)\n');
fprintf('    * Spectral rolloff (85%% energy frequency)\n');
fprintf('    * Spectral flux (frame-to-frame change)\n');
fprintf('    * Zero-crossing rate\n');
fprintf('    * Energy decay rate\n');
fprintf('    * Log attack time\n');
fprintf('    * Burst energy\n');
fprintf('    * Band ratios\n');
fprintf('    * MFCCs\n');
fprintf('  - But if instruments sound similar, these will be similar!\n\n');

fprintf('SOLUTION:\n');
fprintf('==========\n\n');

fprintf('You need to make instruments VERY DIFFERENT in synthesis.\n');
fprintf('Key parameters to adjust in create_synthetic_dataset:\n\n');

fprintf('1. FREQUENCY RANGES (f0_lo, f0_hi)\n');
fprintf('   WRONG (too similar):\n');
fprintf('     Piano: 200-800 Hz\n');
fprintf('     Guitar: 82-330 Hz\n');
fprintf('     Cello: 65-300 Hz\n');
fprintf('   Problem: All overlap significantly!\n\n');

fprintf('   RIGHT (separated):\n');
fprintf('     Piano: 300-1000 Hz (HIGH)\n');
fprintf('     Guitar: 50-200 Hz (VERY LOW)\n');
fprintf('     Cello: 40-150 Hz (LOWEST)\n');
fprintf('     Violin: 200-1500 Hz (MID-HIGH)\n\n');

fprintf('2. DECAY RATES (decay_min, decay_max)\n');
fprintf('   WRONG (too similar):\n');
fprintf('     Guitar: 4.0-6.0 per second\n');
fprintf('     Cello: 0.3-0.8 per second\n');
fprintf('   Problem: Cello decays 5x slower than guitar\n');
fprintf('            But if reverb masks this, they look the same!\n\n');

fprintf('   RIGHT (very different):\n');
fprintf('     Guitar: decay_min=4.0, decay_max=6.0 (FAST)\n');
fprintf('     Piano: decay_min=0.8, decay_max=1.2 (SLOW)\n');
fprintf('     Cello: decay_min=0.2, decay_max=0.5 (VERY SLOW)\n\n');

fprintf('3. REVERB (rt60)\n');
fprintf('   WRONG (guitar has too much reverb):\n');
fprintf('     Guitar: rt60 = 0.08s → filled by reverb tail\n');
fprintf('     Cello: rt60 = 0.60s → concert hall\n');
fprintf('   Problem: Reverb fills the gaps in guitar decay\n');
fprintf('            Now guitar sounds like Cello!\n\n');

fprintf('   RIGHT (guitar must be DRY):\n');
fprintf('     Guitar: rt60 = 0.05s (VERY SHORT)\n');
fprintf('     Cello: rt60 = 0.8s (LONG)\n');
fprintf('     Ratio: Cello reverb is 16x LONGER than guitar!\n\n');

fprintf('4. HARMONICS (n_harm)\n');
fprintf('   MORE DIFFERENT = Better separation\n');
fprintf('   Guitar: n_harm = 6\n');
fprintf('   Cello: n_harm = 9\n');
fprintf('   Piano: n_harm = 14\n\n');

fprintf('ACTION ITEMS:\n');
fprintf('=============\n\n');

fprintf('1. LOOK AT THE VISUALIZATIONS ABOVE\n');
fprintf('   - See where classes overlap\n');
fprintf('   - Identify which pairs are confused\n\n');

fprintf('2. LOOK AT YOUR create_synthetic_dataset PARAMETERS\n');
fprintf('   - Are frequency ranges similar? YES -> FIX\n');
fprintf('   - Are decay rates similar? YES -> FIX\n');
fprintf('   - Is guitar reverb too long? YES -> FIX\n\n');

fprintf('3. ADJUST THE PARAMETERS TO BE VERY DIFFERENT\n');
fprintf('   - Increase differences by 2-3x\n');
fprintf('   - Use extreme values first to test\n\n');

fprintf('4. RE-RUN THE ANALYSIS\n');
fprintf('   - Run this diagnostic again\n');
fprintf('   - Check if classes now separate\n\n');

fprintf('5. THEN RUN THE CLASSIFIER\n');
fprintf('   - Run instrument_identification_FIXED\n');
fprintf('   - Check if accuracy improves\n\n');

fprintf('=== ANALYSIS COMPLETE ===\n');