%Vellios Georgios Serafeim AEM:9471

clc;
close all;
clear;

% Load and normalize the data
data = csvread('train.csv',1,0);
[n_rows,n_cols] = size(data);
data_norm = normalize(data(:,1:end-1),'range');

for i = 1:n_rows
    data_norm(i,n_cols) = data(i,n_cols);
end

% Randomly shuffle and split the data into training, validation, and testing sets
id = randperm(n_rows);
train_data = data_norm(id(1:round(n_rows*0.6)),:);
val_data = data_norm(id(round(n_rows*0.6)+1:round(n_rows*0.8)),:);
test_data = data_norm(id(round(n_rows*0.8)+1:end),:);

% Feature selection using the ReliefF algorithm
[idx,weights] = relieff(data_norm(:,1:end-1),data_norm(:,end),5);

num_features=14;
ra=0.4;

% Keep only the specified number of features and create subsets for training, validation, and testing
train_data_fis = train_data(:, idx(1:num_features));
train_data_fis = [train_data_fis train_data(:,end)];
val_data_fis = val_data(:, idx(1:num_features));
val_data_fis = [val_data_fis val_data(:,end)];
test_data_fis = test_data(:, idx(1:num_features));
test_data_fis = [test_data_fis test_data(:,end)];

X_train_fis = train_data_fis(:,1:end-1);
y_train_fis = train_data_fis(:,end);
X_test = test_data_fis(:,1:end-1);
y_test = test_data_fis(:,end);

% Define initial parameters for the FIS
opt = genfisOptions('SubtractiveClustering','ClusterInfluenceRange',ra);
fis = genfis(X_train_fis, y_train_fis,opt);

% Plot membership functions for features 2 and 5 before training
figure()
plotmf(fis,'input',2)
title("Mfs for feature 2 before training")

figure()
plotmf(fis,'input',5)
title("Mfs for feature 5 before training")

% Train the ANFIS model
opt2 = anfisOptions('InitialFis',fis,'EpochNumber',100,'DisplayANFISInformation',0,'ValidationData',val_data_fis);
[trn_fis,train_error,step_size,val_fis,val_error] = anfis(train_data_fis,opt2);

% Plot membership functions for features 2 and 5 after training
figure()
plotmf(val_fis,'input',2)
title("Mfs for feature 2 After training")

figure()
plotmf(val_fis,'input',5)
title("Mfs for feature 5 After training")

% Learning curves
figure();
plot([train_error val_error]);
legend('Training Error','Validation Error');
title("Learning Curve");
xlabel('Epoch'); 
ylabel('Error');

% Prediction error and scatter plots for actual vs. predicted values
[a,b]=size(y_test);
A = 1:1:a;
y_pred = evalfis(X_test, val_fis);
prediction_error = y_test - y_pred; 

figure();
scatter(A,prediction_error)
title("Prediction Error");
xlabel('Samples');
ylabel('Error');

figure()
scatter(A,y_pred)
xlabel('Sample number');
title('Model predictions');

figure()
scatter(A,y_test)
xlabel('Sample Number');
title('Real Value');

% Calculating metrics and displaying the number of rules for the ANFIS model
MSE = mse(y_pred, y_test);
RMSE = sqrt(MSE);
R2 = 1 - sum((y_test - y_pred).^2)/sum((y_test - mean(y_pred)).^2);
NMSE = 1 - R2;
NDEI = sqrt(NMSE);

% Displaying the metrics and the number of rules for the ANFIS model
metrics = [RMSE NMSE NDEI R2];
n_rules=size(showrule(val_fis),1);


