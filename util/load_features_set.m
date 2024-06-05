function [x_data_cchmc,y_data_cchmc,y_Stiffness_cchmc] = load_features_set(data_dir,liverdata)

% Author: Redha Ali, PhD
% Data Modified: 03/10/2023

Fea_set = [];
slice_name = ["Med_1", "Med_2","Med","Med__1","Med__2"];

for j = 1:5

    baseFileName = sprintf('swin_base_224_MAT_T2_fea_%s_train_fea_Rot_%d',slice_name(j),0);
    load([data_dir num2str(baseFileName) '.mat']);
    Fea_set = [Fea_set,train_features];
    clearvars train_features

end

x_data_cchmc = Fea_set;

y_Stiffness_cchmc = double(string(liverdata.Stiffness));

% convert scores to label with cutoff of <= 3kPa
for idx = 1:length(y_Stiffness_cchmc)

    if y_Stiffness_cchmc(idx) < 3
        y_data_cchmc(idx) =1;
    else
        y_data_cchmc(idx) =0;
    end
end

[a,~] = histc(y_data_cchmc,unique(y_data_cchmc))

end


