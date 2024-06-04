% Run 10 K-folds classification for liver data
% Author: Redha Ali, PhD
% Data Modified: 03/10/2022
%% Load Training Data

clear; close all; clc;

% add path
addpath '.\Excel sheet data\'
addpath '.\util\'

% Load Training Data
data_dir_test_Philips = 'Swin features\path';
liverdata_test_Philips = readtable("liverdata.xlsx");
[x_data_cchmc,y_data_cchmc,y_Stiffness_cchmc] = load_features_set(data_dir_test_Philips,liverdata_test_Philips,index_1);
[a,~] = histc(y_data_cchmc,unique(y_data_cchmc));
y_data_cchmc = categorical(y_data_cchmc)';
truth_labels = y_data_cchmc;

%% Loop over for 5 K_folds for 11 times experiments with different seeds

for idx_1 =1:10
    fprintf('running seed %d of %d seeds \n\n',idx_1,10 )
    
    K_folds = 10;
    CV_cchmc = cvpartition(y_data_cchmc,'KFold',K_folds,'Stratify',true);

    for idx = 1:K_folds

        fprintf('Processing %d of %d folds \n\n',idx,K_folds )

        % Training and Testing Index
        train_cchmc_index = CV_cchmc.training(idx);
        test_cchmc_index = CV_cchmc.test(idx);

        cchmc_train_data = x_data_cchmc(train_cchmc_index,:);
        cchmc_test_data = x_data_cchmc(test_cchmc_index,:);
        cchmc_train_label = y_data_cchmc(train_cchmc_index);
        cchmc_test_label = y_data_cchmc(test_cchmc_index);


        x_test = cchmc_test_data;
        y_test = cchmc_test_label;

        CV1 = cvpartition(cchmc_train_label,'HoldOut',0.1,'Stratify',true);
        T_cchmc_index = gather(CV1.training);
        V_cchmc_index =   gather(CV1.test);
        cchmc_train_data_P = cchmc_train_data(T_cchmc_index,:);
        cchmc_test_data_P = cchmc_train_data(V_cchmc_index,:);
        cchmc_train_label_P = cchmc_train_label(T_cchmc_index);
        cchmc_test_label_P = cchmc_train_label(V_cchmc_index);

       
        x_train = cchmc_train_data_P;
        y_train = cchmc_train_label_P;

        x_Valid = cchmc_test_data_P;
        y_Valid = cchmc_test_label_P;

        y_train = categorical(y_train);
        y_Valid = categorical(y_Valid);
        y_test = categorical(y_test);

        
        totalNumberOfsample = length(y_train);
        frequency = countcats(categorical(y_train)) / totalNumberOfsample;
        invFreqClassWeights = 1./frequency;
        ClassWeights = invFreqClassWeights';
        ClassWeights(1) = ClassWeights(1)+0.6;

        %% Define Deep Learning Model
        % Define the main block of layers as a layer graph.
        

        % Create DNN
        classes = categorical([1,0]);
        layers = [
            featureInputLayer(size(x_train,2),"Name","featureinput","Normalization","zscore")
            fullyConnectedLayer(512,"Name","fc1")
            reluLayer("Name","relu1")
            fullyConnectedLayer(256,"Name","fc2")
            reluLayer("Name","relu2")
            fullyConnectedLayer(2,"Name","fc")
            softmaxLayer("Name","softmax")
            classificationLayer('Classes',classes,'ClassWeights',ClassWeights,"Name","Classi")
            ];
        
        lgraph = layerGraph(layers);

        options = trainingOptions('rmsprop', ...
            'InitialLearnRate',1e-04,...
            'MaxEpochs',140, ...
            'MiniBatchSize',64,...
            'LearnRateSchedule', 'piecewise', ...
            'L2Regularization',0.0001,...
            'ValidationData',{x_Valid,y_Valid},...
            'ValidationPatience',20,...
            'ExecutionEnvironment','gpu',...
            'Shuffle','every-epoch',...
            'Verbose', true,...
            'ValidationFrequency',50);

        
        Teacher_net = trainNetwork(x_train,y_train,layers,options);
        close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_UIFIGURE'))

        % % Test
        [Pred,S] = classify(Teacher_net,x_test);

        out_per_fold = binary_metrics_v4( y_test, Pred,S(:,2))
        

        out_10_fold.acc(idx) = out_per_fold.accuracy;
        out_10_fold.spe(idx) = out_per_fold.specificity;
        out_10_fold.sen(idx) = out_per_fold.sensitivity;
        out_10_fold.auc(idx) = out_per_fold.AUC;


        [cchmc_Pred,cchmc_S] = classify(Teacher_net,cchmc_test_data);

        %predictions Labels and scores
        model_pred_cchmc_Labels(test_cchmc_index) = cchmc_Pred;
        model_pred__cchmc_score(test_cchmc_index,:) = cchmc_S;



    end
    

    total_pred_Labels = model_pred_cchmc_Labels';
    total_pred_score = model_pred__cchmc_score;

    out = binary_metrics_v4( truth_labels, total_pred_Labels,total_pred_score(:,2))

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

%% Mean for 10 times
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
