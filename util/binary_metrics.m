function out = binary_metrics( actual, predicted,prop )
%
% Multiclass performance metrics
%
% REQUIRED INPUTS
% actual    Actual numeric labels
% predicted Predicted numeric labels
%
% OUTPUTS
% out.confusion_matrix (rows are true, cols are predicted)
% out.sensitivity
% out.specificity
% out.balanced_accuracy
% out.accuracy
%
% References:
% http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
% https://en.wikipedia.org/wiki/Confusion_matrix
% Author: Redha Ali, PhD
% Data Modified: 03/10/2023


% confusion matrix
out.confusion_matrix = confusionmat(actual,predicted);
[~,~,~,AUC,~] = perfcurve(actual, prop,'1');

out.AUC = AUC;

% Class k versus all others
TP = out.confusion_matrix(1,1); % correctly classified class k samples

% Class k samples that are labeled other than k
FN=  sum( out.confusion_matrix(1,:) ) - out.confusion_matrix(1,1);

% Other than class k samples labeled as other
cm = out.confusion_matrix;
cm(1,:) = [];
cm(:,1) = [];
TN = sum( cm(:) );

% Non class k samples that are labeled as k
FP= sum( out.confusion_matrix(:,1) ) - out.confusion_matrix(1,1);

out.accuracy = ( TP+TN )/( TP+FP+TN+FN );
out.error_rate = ( FP+FN)/( TP+FP+TN+FN );
out.sensitivity = TP/(TP+FN);
out.specificity = TN/(TN+FP);
% out.precision = TP/(TP+FP);
% out.recall = TP/(TP+FN);
out.balanced_accuracy = mean([(TP/(TP+FN)),(TN/(TN+FP))]);


