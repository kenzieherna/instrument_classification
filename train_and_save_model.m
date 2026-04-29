function train_and_save_model(save_path, n_per_class)
% TRAIN_AND_SAVE_MODEL  Train SVM, Random Forest, and k-NN classifiers
%                       and save to disk. Matches theoretical document.
%
%   train_and_save_model()
%   train_and_save_model(save_path, n_per_class)

    if nargin < 1 || isempty(save_path),    save_path   = 'instrument_model'; end
    if nargin < 2 || isempty(n_per_class),  n_per_class = 200; end

    % ---- Configuration (matches doc exactly) --------------------------------
    config.fs     = 44100;
    config.nfft   = 2048;
    config.hop    = 1024;    % 50% overlap (was wrong: 512 = 75%)
    config.n_mfcc = 13;
    % Feature vector: 4 features x 3 stats + 13 MFCCs x 3 stats = 51

    fprintf('=== Instrument Classification — Training ===\n\n');
    fprintf('Config: fs=%d Hz | nfft=%d | hop=%d (50%% overlap) | n_mfcc=%d\n\n', ...
        config.fs, config.nfft, config.hop, config.n_mfcc);

    % ---- Generate synthetic training data -----------------------------------
    fprintf('--- Generating synthetic dataset (%d samples/class) ---\n', n_per_class);
    [features, labels, instrument_classes] = create_synthetic_dataset( ...
        n_per_class, config.fs, config.nfft, config.hop, config.n_mfcc);
    n_feat = size(features, 2);
    fprintf('\nDataset: %d samples | %d features | %d classes\n\n', ...
        size(features,1), n_feat, length(instrument_classes));

    % ---- Normalize features (z-score) ---------------------------------------
    fprintf('--- Normalizing features (z-score) ---\n');
    [features_norm, scaling_params] = normalize_features(features, 'standardize');

    % ---- Build learner templates --------------------------------------------
    svm_template = templateSVM( ...
        'KernelFunction', 'rbf', ...
        'KernelScale',    'auto', ...
        'Standardize',    false);

    % ---- 5-fold cross-validation (all 3 classifiers) -----------------------
    fprintf('\n--- 5-fold cross-validation ---\n');
    cv       = cvpartition(labels, 'KFold', 5);
    acc_svm  = zeros(5,1);
    acc_rf   = zeros(5,1);
    acc_knn  = zeros(5,1);

    for k = 1:5
        tr = training(cv, k);
        te = test(cv, k);
        Xtr = features_norm(tr,:);  ytr = labels(tr);
        Xte = features_norm(te,:);  yte = labels(te);

        % SVM
        m_svm = fitcecoc(Xtr, ytr, 'Learners', svm_template);
        acc_svm(k) = mean(predict(m_svm, Xte) == yte);

        % Random Forest (500 trees, doc spec)
        m_rf = fitcensemble(Xtr, ytr, 'Method', 'Bag', 'NumLearningCycles', 500, ...
            'Learners', templateTree('MaxNumSplits', 30));
        acc_rf(k) = mean(predict(m_rf, Xte) == yte);

        % k-NN (k=5, doc spec)
        m_knn = fitcknn(Xtr, ytr, 'NumNeighbors', 5, 'Distance', 'euclidean');
        acc_knn(k) = mean(predict(m_knn, Xte) == yte);

        fprintf('  Fold %d: SVM=%.1f%%  RF=%.1f%%  k-NN=%.1f%%\n', k, ...
            acc_svm(k)*100, acc_rf(k)*100, acc_knn(k)*100);
    end

    fprintf('\n  CV Summary:\n');
    fprintf('  SVM:   %.2f%% (+/- %.2f%%)\n', mean(acc_svm)*100, std(acc_svm)*100);
    fprintf('  RF:    %.2f%% (+/- %.2f%%)\n', mean(acc_rf)*100,  std(acc_rf)*100);
    fprintf('  k-NN:  %.2f%% (+/- %.2f%%)\n\n', mean(acc_knn)*100, std(acc_knn)*100);

    % ---- Train final models on all data -------------------------------------
    fprintf('--- Training final models on full dataset ---\n');

    model_svm = fitcecoc(features_norm, labels, ...
        'Learners',     svm_template, ...
        'ClassNames',   (1:length(instrument_classes))', ...
        'FitPosterior', true);   % returns true [0,1] probabilities, not raw decision values

    model_rf = fitcensemble(features_norm, labels, 'Method', 'Bag', ...
        'NumLearningCycles', 500, ...
        'Learners', templateTree('MaxNumSplits', 30));

    model_knn = fitcknn(features_norm, labels, ...
        'NumNeighbors', 5, 'Distance', 'euclidean');

    fprintf('Training complete.\n\n');

    % ---- Save ---------------------------------------------------------------
    mat_path = [save_path '.mat'];
    save(mat_path, 'model_svm', 'model_rf', 'model_knn', ...
        'scaling_params', 'instrument_classes', 'config');
    fprintf('Model saved to: %s\n', mat_path);
    fprintf('  Classes: %s\n\n', strjoin(instrument_classes, ', '));
end