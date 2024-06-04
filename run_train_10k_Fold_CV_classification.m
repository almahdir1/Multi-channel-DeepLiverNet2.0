% Run 10 K-folds classification for liver data
% Author: Redha Ali, PhD
% Data Modified: 03/10/2023
%% Load Training Data

clear; close all; clc;

% add path
addpath '.\Excel sheet data\'
addpath '.\util\'

% Load Training Data Site1
data_dir_Site1 = 'Swin features\path';
liverdata_test_Philips = readtable("liverdata_Site1.xlsx");
[x_data_Site1,y_data_Site1,y_Stiffness_Site1] = load_features_set(data_dir_Site1,liverdata_test_Philips,index_1);
[a,~] = histc(y_data_Site1,unique(y_data_Site1));
y_data_Site1 = categorical(y_data_Site1)';
truth_labels = y_data_Site1;

% Load Training Data Site2
data_dir_Site2 = 'Swin features\path';
liverdata_test_Philips = readtable("liverdata_Site2.xlsx");
[x_data_Site2,y_data_Site2,y_Stiffness_Site2] = load_features_set(data_dir_Site2,liverdata_test_Philips,index_1);
[a,~] = histc(y_data_Site2,unique(y_data_Site2));
y_data_Site2 = categorical(y_data_Site2)';
truth_labels = y_data_Site2;

% Load Training Data Site3
data_dir_Site3 = 'Swin features\path';
liverdata_test_Philips = readtable("liverdata_Site3.xlsx");
[x_data_Site3,y_data_Site3,y_Stiffness_Site3] = load_features_set(data_dir_Site3,liverdata_test_Philips,index_1);
[a,~] = histc(y_data_Site3,unique(y_data_Site3));
y_data_Site3 = categorical(y_data_Site3)';
truth_labels = y_data_Site3;

% Load Training Data Site4
data_dir_Site4 = 'Swin features\path';
liverdata_test_Philips = readtable("liverdata_Site4.xlsx");
[x_data_Site4,y_data_Site4,y_Stiffness_Site4] = load_features_set(data_dir_Site4,liverdata_test_Philips,index_1);
[a,~] = histc(y_data_Site4,unique(y_data_Site4));
y_data_Site4 = categorical(y_data_Site4)';
truth_labels = y_data_Site4;

truth_labels = [y_data_site1;y_data_site2;y_data_site3igan;y_data_site4];

sites_count = countlabels(categorical(truth_labels));
sites_count.Total =[0 ; sum(sites_count.Count)]

%% Loop over for 5 K_folds for 11 times experiments with different seeds

for idx_1 =1:10
    fprintf('running seed %d of %d seeds \n\n',idx_1,10 )

    K_folds = 10;
    CV_site1 = cvpartition(y_data_site1,'KFold',K_folds,'Stratify',false);
    CV_site2 = cvpartition(y_data_site2,'KFold',K_folds,'Stratify',false);
    CV_site3 = cvpartition(y_data_site3igan,'KFold',K_folds,'Stratify',false);
    CV_site4 = cvpartition(y_data_site4,'KFold',K_folds,'Stratify',false);

    for idx = 1:K_folds

        fprintf('Processing %d of %d folds \n\n',idx,K_folds )

        % Training and Testing Index
        train_site1_index = CV_site1.training(idx);
        test_site1_index = CV_site1.test(idx);

        train_site2_index = CV_site2.training(idx);
        test_site2_index = CV_site2.test(idx);

        train_site3_index = CV_site3.training(idx);
        test_site3_index = CV_site3.test(idx);

        train_site4_index = CV_site4.training(idx);
        test_site4_index = CV_site4.test(idx);

        site1_train_data = x_data_site1(train_site1_index,:);
        site1_test_data = x_data_site1(test_site1_index,:);
        site1_train_label = y_data_site1(train_site1_index);
        site1_test_label = y_data_site1(test_site1_index);

        site2_train_data = x_data_nyu(train_site2_index,:);
        site2_test_data = x_data_nyu(test_site2_index,:);
        site2_train_label = y_data_site2(train_site2_index);
        site2_test_label = y_data_site2(test_site2_index);

        site3_train_data = x_data_site3igan(train_site3_index,:);
        site3_test_data = x_data_site3igan(test_site3_index,:);
        site3_train_label = y_data_site3igan(train_site3_index);
        site3_test_label = y_data_site3igan(test_site3_index);

        site4_train_data = x_data_site4(train_site4_index,:);
        site4_test_data = x_data_site4(test_site4_index,:);
        site4_train_label = y_data_site4(train_site4_index);
        site4_test_label = y_data_site4(test_site4_index);

        x_test = [site1_test_data;site2_test_data;site3_test_data;site4_test_data];
        y_test = [site1_test_label;site2_test_label;site3_test_label;site4_test_label];

        CV1 = cvpartition(site1_train_label,'HoldOut',0.1,'Stratify',false);
        T_site1_index = gather(CV1.training);
        V_site1_index =   gather(CV1.test);
        site1_train_data_P = site1_train_data(T_site1_index,:);
        site1_test_data_P = site1_train_data(V_site1_index,:);
        site1_train_label_P = site1_train_label(T_site1_index);
        site1_test_label_P = site1_train_label(V_site1_index);

        CV2 = cvpartition(site2_train_label,'HoldOut',0.1,'Stratify',false);
        T_site2_index = gather(CV2.training);
        V_site2_index =   gather(CV2.test);
        site2_train_data_P = site2_train_data(T_site2_index,:);
        site2_test_data_P = site2_train_data(V_site2_index,:);
        site2_train_label_P = site2_train_label(T_site2_index);
        site2_test_label_P = site2_train_label(V_site2_index);

        CV3 = cvpartition(site3_train_label,'HoldOut',0.1,'Stratify',false);
        T_site3igan_index = gather(CV3.training);
        V_site3igan_index =   gather(CV3.test);
        site3_train_data_P = site3_train_data(T_site3igan_index,:);
        site3_test_data_P = site3_train_data(V_site3igan_index,:);
        site3_train_label_P = site3_train_label(T_site3igan_index);
        site3_test_label_P = site3_train_label(V_site3igan_index);

        CV4 = cvpartition(site4_train_label,'HoldOut',0.1,'Stratify',false);
        T_site4_index = gather(CV4.training);
        V_site4_index =   gather(CV4.test);
        site4_train_data_P = site4_train_data(T_site4_index,:);
        site4_test_data_P = site4_train_data(V_site4_index,:);
        site4_train_label_P = site4_train_label(T_site4_index);
        site4_test_label_P = site4_train_label(V_site4_index);

        x_train = [site1_train_data_P;site2_train_data_P;site3_train_data_P;site4_train_data_P];
        y_train = [site1_train_label_P;site2_train_label_P;site3_train_label_P;site4_train_label_P];

        x_Valid = [site1_test_data_P;site2_test_data_P;site3_test_data_P;site4_test_data_P];
        y_Valid = [site1_test_label_P;site2_test_label_P;site3_test_label_P;site4_test_label_P];

        totalNumberOfsample = length(y_train);
        frequency = countcats(categorical(y_train)) / totalNumberOfsample;
        invFreqClassWeights = 1./frequency;
        ClassWeights = invFreqClassWeights';

        % Define Deep Learning Model
        % Define the main block of layers as a layer graph.
        % Create DNN
        classes = categorical([1,0]);
        layers = [
            featureInputLayer(numFeatures_1,"Name","featureinput","Normalization","none")
            fullyConnectedLayer(4096,"Name","fc1_1","WeightsInitializer","narrow-normal")
            leakyReluLayer("Name","relu5_1")
            fullyConnectedLayer(4096,"Name","fc1_2","WeightsInitializer","narrow-normal")
            leakyReluLayer("Name","relu5_2")
            fullyConnectedLayer(128,"Name","fc2","WeightsInitializer","narrow-normal")
            leakyReluLayer("Name","EmbeddingBatch")
            fullyConnectedLayer(2,"Name","fc")
            softmaxLayer("Name","softmax")
            classificationLayer('Classes',classes,'ClassWeights',ClassWeights,"Name","Classi")];

        rng(seed)
        lgraph = layerGraph(layers);

        options = trainingOptions('rmsprop', ...
            'InitialLearnRate',1e-04,...
            'MaxEpochs',140, ...%66
            'MiniBatchSize',64, ...
            'LearnRateSchedule', 'piecewise', ...
            'L2Regularization',1.0000e-04,...
            'ValidationData',{x_Valid,y_Valid},...6
            'LearnRateDropPeriod', 10,...
            'OutputNetwork','best-validation-loss',...
            'ValidationPatience',35,...
            'ExecutionEnvironment','gpu',...
            'Shuffle','every-epoch',...
            'Verbose', false,...
            'ValidationFrequency',50);

        rng(seed)
        Teacher_net = trainNetwork(x_train,y_train,lgraph,options);


        [Pred,S] = classify(Teacher_net,x_test);

        out_per_fold = binary_metrics_new( y_test, Pred,S);


        out_10_fold.acc(idx) = out_per_fold.accuracy;
        out_10_fold.spe(idx) = out_per_fold.specificity;
        out_10_fold.sen(idx) = out_per_fold.sensitivity;
        out_10_fold.auc(idx) = out_per_fold.AUC;

        [site1_Pred,site1_S] = classify(Teacher_net,site1_test_data);
        [site2_Pred,site2_S] = classify(Teacher_net,site2_test_data);
        [site3_Pred,site3_S] = classify(Teacher_net,site3igan_test_data);
        [site4_Pred,site4_S] = classify(Teacher_net,site4_test_data);


        %predictions Labels and scores
        model_pred_site1_Labels(test_site1_index) = site1_Pred;
        model_pred__site1_score(test_site1_index,:) = site1_S;

        model_pred_site2_Labels(test_site2_index) = site2_Pred;
        model_pred__site2_score(test_site2_index,:) = site2_S;

        model_pred_site3_Labels(test_site3_index) = site3_Pred;
        model_pred__site3_score(test_site3_index,:) = site3_S;

        model_pred_site4_Labels(test_site4_index) = site4_Pred;
        model_pred__site4_score(test_site4_index,:) = site4_S;
    end

    total_pred_Labels = [model_pred_site1_Labels';model_pred_site2_Labels';model_pred_site3_Labels';model_pred_site4_Labels'];
    total_pred_score = [model_pred__site1_score;model_pred__site2_score;model_pred__site3_score;model_pred__site4_score];

    out = binary_metrics_new( truth_labels, total_pred_Labels,total_pred_score);

    OutSwin.acc(idx_1)= out.accuracy;
    OutSwin.sen(idx_1)= out.sensitivity;
    OutSwin.spe(idx_1)= out.specificity;
    OutSwin.bacc(idx_1)= out.balanced_accuracy;
    OutSwin.auc(idx_1)= out.AUC;

    out_all_fold.acc(idx_1,:) = out_10_fold.acc;
    out_all_fold.spe(idx_1,:) = out_10_fold.spe;
    out_all_fold.sen(idx_1,:) = out_10_fold.sen;
    out_all_fold.auc(idx_1,:) = out_10_fold.auc;
end

%%
out_m_all_fold.auc = mean(out_all_fold.auc,2);
out_m_all_fold.acc = mean(out_all_fold.acc,2);
out_m_all_fold.sen = mean(out_all_fold.sen,2);
out_m_all_fold.spe = mean(out_all_fold.spe,2);

out_std_all_fold.auc = std(out_all_fold.auc);
out_std_all_fold.acc = std(out_all_fold.acc);
out_std_all_fold.sen = std(out_all_fold.sen);
out_std_all_fold.spe = std(out_all_fold.spe);


acc_m =mean(out_m_all_fold.acc);
acc_std = mean(out_std_all_fold.acc);

spe_m =mean(out_m_all_fold.spe);
spe_std =mean(out_std_all_fold.spe);

sen_m =mean(out_m_all_fold.sen);
sen_std =mean(out_std_all_fold.sen);

auc_m =mean(out_m_all_fold.auc);
auc_std = mean(out_std_all_fold.auc);


fprintf('M_AUC = %0.3f ± %0.2f \n',auc_m, auc_std )
fprintf('M_SEN = %0.2f ± %0.2f \n',sen_m*100, sen_std )
fprintf('M_SPE = %0.2f ± %0.2f \n',spe_m*100, spe_std )
fprintf('M_ACC = %0.2f ± %0.2f \n',acc_m*100, acc_std )