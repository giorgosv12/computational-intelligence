%Vellios Georgios Serafeim AEM:9471

% Clear workspace and close figures
clc;
close all;
clear;

% Load data and normalize features
data = load('airfoil_self_noise.dat');
[n_rows, n_cols] = size(data);
data_norm = normalize(data(:, 1:end-1), 'range');

for i = 1:n_rows
  data_norm(i, n_cols) = data(i, n_cols);
end

% Shuffle data and split into sets
id = randperm(n_rows);
train_data = data_norm(id(1:round(n_rows*0.6)), :);
val_data = data_norm(id(round(n_rows*0.6)+1:round(n_rows*0.8)), :);
test_data = data_norm(id(round(n_rows*0.8)+1:end), :);

% Extract features and target values
X_train = train_data(:, 1:end-1);
y_train = train_data(:, end);
X_test = test_data(:, 1:end-1);
y_test = test_data(:, end);

% Define ANFIS models with different options
opt1 = genfisOptions('GridPartition', 'NumMembershipFunctions', 2, ...
                     'InputMembershipFunctionType', 'gbellmf', ...
                     'OutputMembershipFunctionType', 'constant');
fis(1) = genfis(X_train, y_train, opt1);

opt2 = genfisOptions('GridPartition', 'NumMembershipFunctions', 3, ...
                     'InputMembershipFunctionType', 'gbellmf', ...
                     'OutputMembershipFunctionType', 'constant');
fis(2) = genfis(X_train, y_train, opt2);

opt3 = genfisOptions('GridPartition', 'NumMembershipFunctions', 2, ...
                     'InputMembershipFunctionType', 'gbellmf', ...
                     'OutputMembershipFunctionType', 'linear');
fis(3) = genfis(X_train, y_train, opt3);

opt4 = genfisOptions('GridPartition', 'NumMembershipFunctions', 3, ...
                     'InputMembershipFunctionType', 'gbellmf', ...
                     'OutputMembershipFunctionType', 'linear');
fis(4) = genfis(X_train, y_train, opt4);

% Initialize metrics table
metrics = zeros(4, 4);

% Train and evaluate each ANFIS model
for i = 1:4
  % Train ANFIS for 100 epochs with validation
  opt(i) = anfisOptions('InitialFis', fis(i), 'EpochNumber', 100, ...
                        'DisplayANFISInformation', 0, 'ValidationData', val_data);
  [trn_fis, train_error, step_size, val_fis, val_error] = anfis(train_data, opt(i));

  % Plot membership functions after training (for validation data)
  for j = 1:n_cols-1
    figure();
    plotmf(val_fis, 'input', j);
    title("Feature: " + j + " for Model: " + i);
  end

  % Plot learning curves
  figure();
  plot([train_error val_error]);
  legend('Training Error', 'Validation Error');
  title("Learning Curve for Model " + i);
  xlabel('Epoch');
  ylabel('Error');

  % Plot prediction error
  [a, b] = size(y_test);
  A = 1:1:a;
  y_pred = evalfis(X_test, val_fis);
  prediction_error = y_test - y_pred;
  figure();
  scatter(A, prediction_error);
  title("Prediction Error for Model " + i);
  xlabel('Samples');
  ylabel('Error');

  % Calculate and store metrics
  MSE = mse(y_pred, y_test);
  RMSE = sqrt(MSE);
  R2 = 1 - sum((y_test - y_pred).^2) / sum((y_test - mean(y_pred)).^2);
  NMSE = 1 - R2;
  NDEI = sqrt(NMSE);
  metrics(i, :) = [RMSE NMSE NDEI R2];
end

% Display metrics table
metrics


