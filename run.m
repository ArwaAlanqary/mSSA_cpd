
addpath(fullfile(pwd,'algorithms', 'density_ratio'));
addpath(fullfile(pwd,'algorithms', 'utils_matlab'));
addpath(fullfile(pwd,'algorithms'));
addpath(fullfile(pwd,'optimizing'));
addpath(fullfile(pwd,'evaluation'));

clear
% Specify experiment
load('config.mat')
algorithm_name = 'density_ratio';
dataset = 'yahoo';
metric = 'compute_f1_score';
data_names = config.DATASETS.(dataset);

% Specify paths 
search_results_path = fullfile(pwd,'results', 'search', algorithm_name, dataset);
test_results_path = fullfile(pwd,'results', 'test', algorithm_name, dataset);
data_path = fullfile(pwd,'data');

grid = allcomb(config.PARAMS.n, config.PARAMS.k, config.PARAMS.alpha, ...
    config.PARAMS.thr, config.PARAMS.peak_dist, config.PARAMS.fold);

% data = csvread(fullfile(data_path, dataset, sprintf('%s_ts.csv',data_names(1))));
% labels = csvread(fullfile(data_path, dataset, sprintf('%s_labels.csv',data_names(1))));
% ts = data(:, 2:end);
% param = grid(2, :);
% [~, cp] = density_ratio(ts, param(1), param(2), param(3), param(4), ...
%             param(5), param(6));
data_sets_length = length(data_names);
parfor data_name_i = 1:data_sets_length
    data_name = data_names(data_name_i);
    experiment = struct('dataset', dataset, 'data_name', data_name, ...
                        'algorithm_name', algorithm_name, 'metric', metric);
    % Load data 
    data = csvread(fullfile(data_path, dataset, sprintf('%s_ts.csv',data_name)));
    labels = csvread(fullfile(data_path, dataset, sprintf('%s_labels.csv',data_name)));
    ts = data(:, 2:end);
    % Search for best parameters
    param_comb = size(grid);
    param_comb = param_comb(1);
    f1_scores = zeros(1, param_comb); 
    for i = 1:param_comb
        %n, k, alpha, thr, peak_dist, fold
        param = grid(i, :);
        param = struct('n', param(1), 'k', param(2), 'alpha', param(3), ...
                       'thr', param(4), 'peak_dist', param(5), ...
                       'fold', param(6));
        [~, cp] = density_ratio(ts, param.n, param.k, param.alpha, ...
                                param.thr, param.peak_dist, param.fold);
        score = compute_f1_score(labels,cp, config.MARGIN);
        f1_scores(i) = score;
        save_results_json(experiment, param, cp, score, search_results_path, ...
                          'success', '')
    end
    [~, best_ind] = max(f1_scores);
    param = grid(best_ind, :);
    param = struct('n', param(1), 'k', param(2), 'alpha', param(3), ...
                   'thr', param(4), 'peak_dist', param(5), ...
                   'fold', param(6));
    [~, cp] = density_ratio(ts, param.n, param.k, param.alpha, ...
                                param.thr, param.peak_dist, param.fold);
    score = compute_f1_score(labels,cp, config.MARGIN);
    f1_scores(i) = score;
    save_results_json(experiment, param, cp, score, test_results_path, ...
                          'success', '')
%     save_results_table(experiment, score, test_results_path, 'success')
end

