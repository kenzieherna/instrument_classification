%% RUN_INSTRUMENT_PIPELINE.m
%  End-to-end instrument identification pipeline.
%  Matches theoretical document (MATH 310) exactly.

clc; clear; close all;

%% ---- USER SETTINGS -------------------------------------------------------

AUDIO_FILE  = '/Users/mackenziehernandez/Documents/MATLAB/MATH310/audiofiles/audiotest2.mp3';
MODEL_PATH  = 'instrument_model';
N_PER_CLASS = 200;

%% ---- STEP 1: Train or load model ----------------------------------------

model_file = [MODEL_PATH '.mat'];

if isfile(model_file)
    fprintf('Found existing model: %s\n', model_file);
    fprintf('Delete instrument_model.mat to retrain.\n\n');
else
    fprintf('No saved model found. Training now...\n\n');
    train_and_save_model(MODEL_PATH, N_PER_CLASS);
end

%% ---- STEP 2: Analyze audio file -----------------------------------------

if ~isfile(AUDIO_FILE)
    error('Audio file not found: "%s"\nUpdate AUDIO_FILE above.', AUDIO_FILE);
end

result = analyze_audio_file(AUDIO_FILE, model_file);

%% ---- STEP 3: Summary  (uses correct result field names) -----------------

fprintf('=== FINAL RESULT ===\n');
fprintf('  File:             %s\n', AUDIO_FILE);
fprintf('  SVM prediction:   %s\n', result.svm_top);
fprintf('  RF  prediction:   %s\n', result.rf_top);
fprintf('  k-NN prediction:  %s\n', result.knn_top);
fprintf('  Ensemble vote:    %s\n', result.predicted_class);
fprintf('  Valid segments:   %d\n', result.n_valid_segments);
fprintf('====================\n\n');
