%% INSTRUMENT IDENTIFICATION - CORRECTED MAIN SCRIPT
% This script properly uses your create_synthetic_dataset function
% with CORRECT classifier training syntax

clear all; close all; clc;

fprintf('=== Instrument Classification System ===\n\n');

%% Parameters
fs = 44100;
nfft = 2048;
hop = 512;
n_mfcc = 13;
n_per_class = 15;

%% Step 1: Generate synthetic data using YOUR function
fprintf('Step 1: Generating synthetic data...\n');

[features, labels, instrument_classes] = create_synthetic_dataset(...
    n_per_class, fs, nfft, hop, n_mfcc);

fprintf('Generated: %d samples across %d classes\n\n', ...
    size(features, 1), length(instrument_classes));

%% Step 2: Standardize features
fprintf('Step 2: Standardizing features...\n');

feature_mean = mean(features);
feature_std = std(features);
feature_std(feature_std == 0) = 1;
features_norm = (features - feature_mean) ./ feature_std;

fprintf('Features normalized to zero mean, unit variance\n\n');

%% Step 3: Train/test split
fprintf('Step 3: Splitting into train/test...\n');

train_ratio = 0.7;
n_samples = length(labels);
n_train = round(n_samples * train_ratio);
perm = randperm(n_samples);
train_idx = perm(1:n_train);
test_idx = perm(n_train+1:end);

X_train = features_norm(train_idx, :);
y_train = labels(train_idx);
X_test = features_norm(test_idx, :);
y_test = labels(test_idx);

fprintf('Train: %d samples, Test: %d samples\n\n', ...
    size(X_train, 1), size(X_test, 1));

%% Step 4: Train classifiers
fprintf('Step 4: Training classifiers...\n\n');

% === METHOD 1: Random Forest (RECOMMENDED - most reliable) ===
fprintf('  Training Random Forest...\n');
try
    model_rf = fitensemble(X_train, y_train, 'Bag', 200, ...
        'Tree', 'Type', 'classification');
    
    pred_rf_train = predict(model_rf, X_train);
    acc_rf_train = sum(pred_rf_train == y_train) / length(y_train);
    
    pred_rf_test = predict(model_rf, X_test);
    acc_rf_test = sum(pred_rf_test == y_test) / length(y_test);
    
    fprintf('    Training accuracy: %.1f%%\n', acc_rf_train * 100);
    fprintf('    Test accuracy: %.1f%%\n\n', acc_rf_test * 100);
    
    rf_works = true;
except
    fprintf('    Random Forest training failed\n\n');
    rf_works = false;
end

% === METHOD 2: k-NN (simple alternative) ===
fprintf('  Training k-NN classifier...\n');
try
    model_knn = fitcknn(X_train, y_train, 'NumNeighbors', 5, ...
        'Distance', 'euclidean');
    
    pred_knn_train = predict(model_knn, X_train);
    acc_knn_train = sum(pred_knn_train == y_train) / length(y_train);
    
    pred_knn_test = predict(model_knn, X_test);
    acc_knn_test = sum(pred_knn_test == y_test) / length(y_test);
    
    fprintf('    Training accuracy: %.1f%%\n', acc_knn_train * 100);
    fprintf('    Test accuracy: %.1f%%\n\n', acc_knn_test * 100);
    
    knn_works = true;
catch
    fprintf('    k-NN training failed\n\n');
    knn_works = false;
end

% === METHOD 3: SVM (FIXED SYNTAX) ===
fprintf('  Training SVM classifier...\n');
try
    % CORRECT WAY to pass parameters to fitcecoc
    t = templateSVM('Kernel', 'rbf', 'BoxConstraint', 1.0);
    model_svm = fitcecoc(X_train, y_train, 'Learners', t);
    
    pred_svm_train = predict(model_svm, X_train);
    acc_svm_train = sum(pred_svm_train == y_train) / length(y_train);
    
    pred_svm_test = predict(model_svm, X_test);
    acc_svm_test = sum(pred_svm_test == y_test) / length(y_test);
    
    fprintf('    Training accuracy: %.1f%%\n', acc_svm_train * 100);
    fprintf('    Test accuracy: %.1f%%\n\n', acc_svm_test * 100);
    
    svm_works = true;
catch ME
    fprintf('    SVM training failed: %s\n\n', ME.message);
    svm_works = false;
end

%% Step 5: Select best model
fprintf('Step 5: Model Comparison\n\n');

accuracies = [];
model_names = {};
models = {};

if rf_works
    accuracies = [accuracies, acc_rf_test];
    model_names = [model_names, {'Random Forest'}];
    models = [models, {model_rf}];
end

if knn_works
    accuracies = [accuracies, acc_knn_test];
    model_names = [model_names, {'k-NN'}];
    models = [models, {model_knn}];
end

if svm_works
    accuracies = [accuracies, acc_svm_test];
    model_names = [model_names, {'SVM'}];
    models = [models, {model_svm}];
end

[best_acc, best_idx] = max(accuracies);
best_model_name = model_names{best_idx};
best_model = models{best_idx};

fprintf('Best Model: %s (%.1f%% test accuracy)\n\n', ...
    best_model_name, best_acc * 100);

%% Step 6: Detailed evaluation
fprintf('Step 6: Detailed Evaluation\n\n');

% Get predictions from best model
if strcmp(best_model_name, 'Random Forest')
    pred_best = predict(model_rf, X_test);
elseif strcmp(best_model_name, 'k-NN')
    pred_best = predict(model_knn, X_test);
else % SVM
    pred_best = predict(model_svm, X_test);
end

% Confusion matrix
C = confusionmat(y_test, pred_best);

fprintf('Confusion Matrix:\n');
fprintf('%12s', '');
for c = 1:length(instrument_classes)
    fprintf(' | %8s', instrument_classes{c}(1:min(8, length(instrument_classes{c}))));
end
fprintf('\n');
fprintf('%s\n', repmat('-', 1, 12 + 11*length(instrument_classes)));

for i = 1:length(instrument_classes)
    fprintf('%12s', instrument_classes{i});
    for j = 1:length(instrument_classes)
        fprintf(' | %8d', C(i,j));
    end
    fprintf('\n');
end

fprintf('\n');

% Per-class metrics
fprintf('Per-Class Metrics:\n');
fprintf('%12s | %10s | %10s | %10s\n', 'Class', 'Precision', 'Recall', 'F1-Score');
fprintf('%s\n', repmat('-', 1, 50));

f1_scores = [];
for i = 1:length(instrument_classes)
    tp = C(i, i);
    fp = sum(C(:, i)) - tp;
    fn = sum(C(i, :)) - tp;
    
    precision = tp / (tp + fp + 1e-10);
    recall = tp / (tp + fn + 1e-10);
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10);
    
    fprintf('%12s | %10.2f%% | %10.2f%% | %10.2f%%\n', ...
        instrument_classes{i}, precision * 100, recall * 100, f1 * 100);
    
    f1_scores(i) = f1;
end

fprintf('\nMacro-averaged F1-Score: %.2f%%\n\n', mean(f1_scores) * 100);

%% Step 7: Diagnosis
fprintf('Step 7: Root Cause Analysis\n\n');

% Check if one class dominates
pred_counts = accumarray(pred_best, 1, [length(instrument_classes), 1]);
[max_count, max_class_idx] = max(pred_counts);
max_pct = max_count / length(y_test) * 100;

if max_pct > 60
    fprintf('PROBLEM: %s predicted %.1f%% of the time!\n\n', ...
        instrument_classes{max_class_idx}, max_pct);
    fprintf('Likely causes:\n');
    fprintf('  1. Overlapping frequency ranges\n');
    fprintf('  2. Similar decay rates (see ROOT_CAUSE_ANALYSIS.txt)\n');
    fprintf('  3. Too much reverb on plucked instruments\n');
    fprintf('  4. Similar harmonic content\n\n');
    fprintf('Solutions:\n');
    fprintf('  - Read QUICK_START.txt\n');
    fprintf('  - Try extreme parameter values first\n');
    fprintf('  - Make instruments MORE DIFFERENT\n\n');
else
    fprintf('GOOD: Classes are being predicted more evenly.\n\n');
end

if best_acc > 0.80
    fprintf('SUCCESS: Classifier is working well!\n');
    fprintf('  Training data is well-separated\n');
    fprintf('  Test accuracy is good (%.1f%%)\n\n', best_acc * 100);
else
    fprintf('CAUTION: Accuracy could be better (%.1f%%)\n', best_acc * 100);
    fprintf('  Consider: More distinct instrument parameters\n');
    fprintf('  Or: More training samples per class\n\n');
end

%% Step 8: Visualizations
fprintf('Step 8: Creating visualizations...\n\n');

figure('Position', [100 100 1400 800]);

% 2D Feature space
if size(features, 2) >= 2
    subplot(2, 3, 1);
    for c = 1:length(instrument_classes)
        idx = labels == c;
        scatter(features(idx, 1), features(idx, 2), 50, ...
            'DisplayName', instrument_classes{c});
        hold on;
    end
    xlabel('Feature 1 (Spectral Centroid)');
    ylabel('Feature 2 (Spectral Rolloff)');
    title('2D Feature Space');
    legend('FontSize', 8);
    grid on;
end

% Feature distributions
if size(features, 2) >= 1
    subplot(2, 3, 2);
    hold on;
    for c = 1:length(instrument_classes)
        idx = labels == c;
        histogram(features(idx, 1), 'Normalization', 'probability', ...
            'FaceAlpha', 0.5, 'DisplayName', instrument_classes{c});
    end
    xlabel('Feature 1');
    ylabel('Probability');
    title('Feature 1 Distribution');
    legend('FontSize', 8);
    grid on;
end

% Confusion matrix heatmap
subplot(2, 3, 3);
C_norm = C ./ (sum(C, 2) + 1e-10);
imagesc(C_norm);
colorbar;
caxis([0 1]);
colormap('hot');
set(gca, 'XTick', 1:length(instrument_classes));
set(gca, 'YTick', 1:length(instrument_classes));
set(gca, 'XTickLabel', instrument_classes);
set(gca, 'YTickLabel', instrument_classes);
xlabel('Predicted');
ylabel('True');
title('Confusion Matrix');
xtickangle(45);

% Per-class accuracy
subplot(2, 3, 4);
class_acc = diag(C) ./ sum(C, 2);
bar(class_acc * 100);
set(gca, 'XTick', 1:length(instrument_classes));
set(gca, 'XTickLabel', instrument_classes);
ylabel('Accuracy (%)');
title('Per-Class Accuracy');
ylim([0 105]);
xtickangle(45);
grid on;

% Model comparison
if length(model_names) > 1
    subplot(2, 3, 5);
    bar(accuracies * 100);
    set(gca, 'XTick', 1:length(model_names));
    set(gca, 'XTickLabel', model_names);
    ylabel('Test Accuracy (%)');
    title('Classifier Comparison');
    ylim([0 105]);
    xtickangle(45);
    grid on;
end

% Summary text
subplot(2, 3, 6);
axis off;
summary_text = sprintf(['Best Model: %s\n' ...
                        'Test Accuracy: %.1f%%\n' ...
                        'Training Accuracy: %.1f%%\n' ...
                        'Total Samples: %d\n' ...
                        'Classes: %d\n' ...
                        'Features: %d'], ...
                        best_model_name, best_acc * 100, ...
                        acc_rf_train * 100, length(y_test) + length(y_train), ...
                        length(instrument_classes), size(features, 2));

text(0.5, 0.5, summary_text, 'HorizontalAlignment', 'center', ...
     'VerticalAlignment', 'center', 'FontSize', 12, ...
     'FontWeight', 'bold', 'FontFamily', 'monospace');

sgtitle('Instrument Classification Results');

fprintf('Analysis complete!\n\n');
fprintf('=== END ===\n');