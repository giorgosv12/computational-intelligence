%Vellios Georgios Serafeim AEM:9471

% Clear workspace and close figures
clc;
close all;
clear;

% Load and normalize data
data = csvread('train.csv', 1, 0);
[n_rows, n_cols] = size(data);
data_norm = normalize(data(:, 1:end-1), 'range');

for i = 1:n_rows
  data_norm(i, n_cols) = data(i, n_cols);
end

% Grid search parameters
num_features_tests = 5;
num_ra_tests = 4;
features_tests = [8 12 14 16 20];
ra_tests = [0.2 0.4 0.6 0.8];

% Feature selection using relieff algorithm
[idx, weights] = relieff(data_norm(:, 1:end-1), data_norm(:, end), 5);

% Dummy variable for test tracking
testa = 1;

% Initialize variables for storing results
mean_error = zeros(num_features_tests, num_ra_tests);
n_rules = zeros(num_features_tests, num_ra_tests);
n_features = zeros(num_features_tests, num_ra_tests);

% Grid search for number of features and rule activation range
for i = 1:num_features_tests
  for j = 1:num_ra_tests
    disp("Test " + testa)
    testa = testa + 1;
    num_features = features_tests(i);
    num_ra = ra_tests(j);

    n_folds = 5;
    c = cvpartition(n_rows, 'KFold', n_folds);

    % Initialize temporary array for fold errors
    error = zeros(n_folds);

    % Loop through each fold
    for fold = 1:n_folds

      % Split data into training, validation, and testing sets
      train_val_data = data_norm(training(c, fold), :);
      train_data = train_val_data(1:round(0.6*n_rows), :);
      val_data = train_val_data(round(0.6*n_rows)+1:end, :);
      test_data = data_norm(test(c, fold), :);

      % Select features based on current test parameters
      train_data_fis = train_data(:, idx(1:num_features));
      train_data_fis = [train_data_fis train_data(:, end)];
      val_data_fis = val_data(:, idx(1:num_features));
      val_data_fis = [val_data_fis val_data(:, end)];
      test_data_fis = test_data(:, idx(1:num_features));
      test_data_fis = [test_data_fis test_data(:, end)];

      % Extract features and target values for ANFIS
      X_train_fis = train_data_fis(:, 1:end-1);
      y_train_fis = train_data_fis(:, end);

      % Train ANFIS for each fold and parameter combination for 100 epochs
      opt = genfisOptions('SubtractiveClustering', 'ClusterInfluenceRange', num_ra);
      fis = genfis(X_train_fis, y_train_fis, opt);
      opt2 = anfisOptions('InitialFis', fis, 'EpochNumber', 100, 'DisplayANFISInformation', 0, 'ValidationData', val_data_fis);
      [trn_fis, train_error, step_size, val_fis, val_error] = anfis(train_data_fis, opt2);

      % Calculate RMSE on testing set using best validation loss epoch
      y_pred = evalfis(test_data_fis(:, 1:end-1), val_fis);
      MSE = mse(y_pred, test_data_fis(:, end));
      RMSE = sqrt(MSE);
      error(fold) = RMSE;

    end

    % Store results for current parameter combination
    mean_error(i, j) = sum(error(:)) / n_folds;
    n_rules(i, j) = size


