addpath(fullfile(pwd,'algorithms', 'density_ratio'));
addpath(fullfile(pwd,'algorithms', 'utils_matlab'));
addpath(fullfile(pwd,'algorithms'));
addpath(fullfile(pwd,'optimizing'));
addpath(fullfile(pwd,'evaluation'));

clear

% Specify data
load('config.mat')
dataset = 'struct';
metric = 'compute_f1_score';
data_names = config.DATASETS.(dataset);

% Specify paths 
data_path = fullfile(pwd,'data');

%Generate grid
grid = allcomb(config.PARAMS.n, config.PARAMS.k, config.PARAMS.alpha, ...
    config.PARAMS.thr, config.PARAMS.peak_dist, config.PARAMS.fold);

%Load data
data = csvread(fullfile(data_path, dataset, sprintf('%s_ts.csv',data_names(3))));
labels = csvread(fullfile(data_path, dataset, sprintf('%s_labels.csv',data_names(3))));
ts = data(:, 2:end);
ts = transpose(ts);
% ts = vecnorm(transpose(ts));
% data = load("logwell.mat");
% ts = data.y;
ts = ts(1:3000);
subplot(2,1,1);
plot(ts, 'linewidth',2);
axis([-inf,size(ts,2),-inf,inf])
title('Original Signal')

%%
% n=50, k = 10 (compare time with n=100, k=20)
% increase folds 10
% best n and k with alpha 0.2
n = 100;
k = 30; 
alpha = 0.1;
thr = 2;
peak_dist = 0.9;
fold = 5;
tic;
[score, cp] = density_ratio(ts, n, k, alpha, ...
                            thr, peak_dist, fold);
toc
f1_score = compute_f1_score(labels,cp, config.MARGIN);

subplot(2,1,2);
plot(score, 'r-', 'linewidth',2);
axis([-inf,size(ts,2),-inf,inf])
title('Change-Point Score')