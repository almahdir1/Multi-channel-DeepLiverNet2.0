% Run 10 K-folds regression for liver data
% Author: Redha Ali, PhD
% Data Modified: 04/28/2025
%% Initialize Environment
clear; close all; clc;

% Add necessary folders to the path
addpath('..\Swin_features_11_slices_Seg\');
addpath('..\Clinical_fea_Tabels\');
addpath('.\Excel sheet data\');
addpath('.\util\');

% Set GPU device
gpuIndex = 2;
gpuDevice(gpuIndex);
disp(['Using GPU: ' gpuDevice().Name]);

%% Load Data for Each Site
% Site 1 Data Loading
[x_site1, y_site1, clinical_site1] = loadSiteData('Site1', 'Batch1', 'Batch2');

% Site 2 Data Loading
[x_site2, y_site2, clinical_site2] = loadSiteData('Site2', 'Batch1', 'Batch2');

% Site 3 Data Loading
[x_site3, y_site3, clinical_site3] = loadSiteData('Site3', 'Batch1', 'Batch2');

%% Prepare Combined Data
X_all = [x_site1; x_site2; x_site3];
Y_all = [y_site1; y_site2; y_site3];

% Store number of clinical features
numClinicalFeatures = size(clinical_site1, 2);

%% Setup Cross Validation
K_folds = 10;
cv_site1 = cvpartition(y_site1, 'KFold', K_folds);
cv_site2 = cvpartition(y_site2, 'KFold', K_folds);
cv_site3 = cvpartition(y_site3, 'KFold', K_folds);

%% Initialize result containers
out_all_folds = struct();
idx_model_run = 1;  % Only one run for now

for foldIdx = 1:K_folds
    fprintf('Processing Fold %d of %d...\n', foldIdx, K_folds);
    
    %% Split Data for Training and Testing (Outer Fold)
    [train1, test1] = splitData(x_site1, y_site1, cv_site1, foldIdx);
    [train2, test2] = splitData(x_site2, y_site2, cv_site2, foldIdx);
    [train3, test3] = splitData(x_site3, y_site3, cv_site3, foldIdx);

    X_test = [test1.data; test2.data; test3.data];
    Y_test = [test1.labels; test2.labels; test3.labels];

    %% Further split Training into Train/Validation (Inner Holdout)
    [train1_inner, val1] = innerHoldout(train1);
    [train2_inner, val2] = innerHoldout(train2);
    [train3_inner, val3] = innerHoldout(train3);

    % Final training and validation sets
    X_train = [train1_inner.data; train2_inner.data; train3_inner.data];
    Y_train = [train1_inner.labels; train2_inner.labels; train3_inner.labels];

    X_val = [val1.data; val2.data; val3.data];
    Y_val = [val1.labels; val2.labels; val3.labels];

    %% Separate Imaging and Clinical Features
    [X_train_img, X_train_clin] = splitFeatures(X_train, numClinicalFeatures);
    [X_val_img, X_val_clin] = splitFeatures(X_val, numClinicalFeatures);
    [X_test_img, X_test_clin] = splitFeatures(X_test, numClinicalFeatures);

    %% Create Datastores
    dsTrain = createCombinedDatastore(X_train_img, X_train_clin, Y_train);
    dsValid = createCombinedDatastore(X_val_img, X_val_clin, Y_val);
    dsTest = createTestDatastore(X_test_img, X_test_clin);

    %% Define Network
    net = defineNetwork(size(X_train_img,2), numClinicalFeatures);

    %% Train Network
    options = defineTrainingOptions(dsValid);
    seed = 42;
    rng(seed);
    trainedNet = trainNetwork(dsTrain, net, options);

    %% Predict
    preds_test = predict(trainedNet, dsTest);

    %% Store per-fold metrics
    fold_metrics = regressions_metrics(Y_test, preds_test);

    out_10fold.RMSE(foldIdx) = fold_metrics.RMSE;
    out_10fold.MAE(foldIdx) = fold_metrics.MAE;
    out_10fold.R_sq(foldIdx) = fold_metrics.R_sq;
    out_10fold.CC(foldIdx) = fold_metrics.CC;
end

%% Aggregate Results Across Folds
out_all_folds.RMSE(idx_model_run,:) = out_10fold.RMSE;
out_all_folds.MAE(idx_model_run,:) = out_10fold.MAE;
out_all_folds.R_sq(idx_model_run,:) = out_10fold.R_sq;
out_all_folds.CC(idx_model_run,:) = out_10fold.CC;

out_mean.RMSE = mean(out_all_folds.RMSE,2);
out_mean.MAE = mean(out_all_folds.MAE,2);
out_mean.R_sq = mean(out_all_folds.R_sq,2);
out_mean.CC = mean(out_all_folds.CC,2);

out_std.RMSE = std(out_all_folds.RMSE,0,2);
out_std.MAE = std(out_all_folds.MAE,0,2);
out_std.R_sq = std(out_all_folds.R_sq,0,2);
out_std.CC = std(out_all_folds.CC,0,2);

%% Display Results
fprintf('Cross-Validation Results:\n');
fprintf('  CC = %.3f ± %.2f\n', mean(out_mean.CC), mean(out_std.CC));
fprintf('  R_sq = %.2f ± %.2f\n', mean(out_mean.R_sq), mean(out_std.R_sq));
fprintf('  MAE = %.2f ± %.2f\n', mean(out_mean.MAE), mean(out_std.MAE));
fprintf('  RMSE = %.2f ± %.2f\n', mean(out_mean.RMSE), mean(out_std.RMSE));

%% --------------------------- Helper Functions ---------------------------
function [X_data, Y_labels, clinical_data] = loadSiteData(siteName, batch1, batch2)
    % Load Features and Labels for Site
    liverdata1 = readtable(sprintf('T2_T1_DWI_pair_%s_%s.xlsx', siteName, batch1));
    liverdata2 = readtable(sprintf('T2_T1_DWI_pair_%s_%s.xlsx', siteName, batch2));

    X1 = [loadFeatures(siteName, batch1), loadClinical(siteName, batch1)];
    X2 = [loadFeatures(siteName, batch2), loadClinical(siteName, batch2)];
    
    X_data = [X1; X2];
    Y_labels = [liverdata1.Stiffness; liverdata2.Stiffness];
    clinical_data = loadClinical(siteName, batch1); % One clinical set
end

function X = loadFeatures(siteName, batch)
    % Load T2, T1, and DWI features and concatenate
    path = ['..\fea_MR_all_new_v1\' siteName '_' batch '\'];
    load([path 'T2\All_features.mat']); T2 = All_features;
    load([path 'T1\All_features.mat']); T1 = All_features;
    load([path 'DWI\All_features.mat']); DWI = All_features;
    X = [T2, T1, DWI];
end

function clinicalData = loadClinical(siteName, batch)
    % Load clinical features
    file = sprintf('%s_%s_clinical_fea.xlsx', siteName, batch);
    if contains(siteName,'Site3') % Specific naming
        file = sprintf('T2_T1_DWI_pair_%s_%s_clinical_fea.xlsx', siteName, batch);
    end
    clinicalData = readtable(file);
    clinicalData(isnan(clinicalData)) = 0;
end

function [train, test] = splitData(X, Y, cv, idx)
    train.data = X(cv.training(idx),:);
    train.labels = Y(cv.training(idx));
    test.data = X(cv.test(idx),:);
    test.labels = Y(cv.test(idx));
end

function [train, val] = innerHoldout(data)
    cv = cvpartition(data.labels, 'HoldOut', 0.1);
    train.data = data.data(cv.training,:);
    train.labels = data.labels(cv.training);
    val.data = data.data(cv.test,:);
    val.labels = data.labels(cv.test);
end

function [imgFeatures, clinicalFeatures] = splitFeatures(X, numClinical)
    imgFeatures = X(:,1:end-numClinical);
    clinicalFeatures = X(:,end-numClinical+1:end);
end

function ds = createCombinedDatastore(imgData, clinData, labels)
    dsX1 = arrayDatastore(imgData', IterationDimension=2);
    dsX2 = arrayDatastore(clinData', IterationDimension=2);
    dsT  = arrayDatastore(labels);
    ds = combine(dsX1, dsX2, dsT);
end

function ds = createTestDatastore(imgData, clinData)
    dsX1 = arrayDatastore(imgData', IterationDimension=2);
    dsX2 = arrayDatastore(clinData', IterationDimension=2);
    ds = combine(dsX1, dsX2);
end

function net = defineNetwork(numImgFeatures, numClinicalFeatures)
    % Define Dual-branch Neural Network
    imgBranch = [
        featureInputLayer(numImgFeatures,"Normalization","zscore","Name","img_input")
        fullyConnectedLayer(4096,"WeightsInitializer","narrow-normal")
        leakyReluLayer
        fullyConnectedLayer(4096,"WeightsInitializer","narrow-normal")
        leakyReluLayer
        fullyConnectedLayer(128,"WeightsInitializer","narrow-normal")
        leakyReluLayer
    ];

    clinicalBranch = [
        featureInputLayer(numClinicalFeatures,"Normalization","zscore","Name","clinical_input")
        fullyConnectedLayer(4096,"WeightsInitializer","narrow-normal")
        leakyReluLayer
        fullyConnectedLayer(128,"WeightsInitializer","narrow-normal")
        leakyReluLayer
    ];

    combinedLayers = [
        concatenationLayer(1,2,"Name","concat")
        fullyConnectedLayer(1,"Name","fc")
        regressionLayer("Name","regression")
    ];

    lgraph = layerGraph(imgBranch);
    lgraph = addLayers(lgraph, clinicalBranch);
    lgraph = addLayers(lgraph, combinedLayers);
    lgraph = connectLayers(lgraph,"leakyrelu_3","concat/in1");
    lgraph = connectLayers(lgraph,"leakyrelu_5","concat/in2");
    net = lgraph;
end

function options = defineTrainingOptions(dsValid)
    options = trainingOptions('rmsprop', ...
        'InitialLearnRate',1e-4, ...
        'MaxEpochs',140, ...
        'MiniBatchSize',128, ...
        'ValidationData',dsValid, ...
        'ValidationFrequency',50, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropPeriod',10, ...
        'L2Regularization',1e-4, ...
        'OutputNetwork','best-validation-loss', ...
        'ValidationPatience',35, ...
        'ExecutionEnvironment','gpu', ...
        'Shuffle','every-epoch', ...
        'Verbose',false);
end
