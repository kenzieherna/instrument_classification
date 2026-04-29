function f1_scores = compute_per_class_f1(conf_matrix)
% COMPUTE_PER_CLASS_F1  Compute per-class F1 scores from a confusion matrix.
%
%   f1_scores = compute_per_class_f1(conf_matrix)
%
%   Input:
%     conf_matrix - [n x n] confusion matrix (rows = true, cols = predicted)
%
%   Output:
%     f1_scores - [n x 1] F1 score for each class

    n_classes = size(conf_matrix, 1);
    f1_scores = zeros(n_classes, 1);

    for i = 1:n_classes
        tp = conf_matrix(i, i);
        fp = sum(conf_matrix(:, i)) - tp;
        fn = sum(conf_matrix(i, :)) - tp;

        precision = tp / (tp + fp + 1e-10);
        recall    = tp / (tp + fn + 1e-10);
        f1_scores(i) = 2 * precision * recall / (precision + recall + 1e-10);
    end
end
