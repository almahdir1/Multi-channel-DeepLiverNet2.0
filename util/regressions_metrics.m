function metrics = regressions_metrics(y_true, y_pred)
% REGRESSIONS_METRICS Computes evaluation metrics for regression models.
%
% INPUTS:
%   y_true : True labels (ground truth)
%   y_pred : Predicted labels
%
% OUTPUT:
%   metrics : Struct containing Errors, MSE, RMSE, MAE, R-squared, Adjusted R-squared, and Correlation Coefficient

%% Compute Basic Error Terms
errors = y_true - y_pred;                   % Raw errors
squaredErrors = errors.^2;                   % Squared errors

%% Compute Primary Metrics
MSE = mean(squaredErrors);                   % Mean Squared Error
RMSE = sqrt(MSE);                            % Root Mean Squared Error
MAE = mean(abs(errors));                     % Mean Absolute Error

%% Compute R-squared (Coefficient of Determination)
residualSumSquares = sum(errors.^2);          % Sum of squared residuals
totalSumSquares = (length(y_true) - 1) * var(y_true); % Total sum of squares

R_squared = 1 - (residualSumSquares / totalSumSquares);  % Unadjusted R^2

% For adjusted R-squared
numPredictors = 1; % You have only one output predictor
adj_R_squared = 1 - (residualSumSquares / totalSumSquares) * ...
                (length(y_true)-1) / (length(y_true) - numPredictors);

%% Compute Correlation Coefficient
correlationCoefficient = corr2(y_true, y_pred);    % Pearson correlation
correlationSquared = correlationCoefficient^2;     % r^2

%% Package Output
metrics.Errors = errors;
metrics.SquaredErrors = squaredErrors;
metrics.MSE = MSE;
metrics.RMSE = RMSE;
metrics.MAE = MAE;
metrics.CC = correlationCoefficient;
metrics.R_sq = correlationSquared;
metrics.Rsq_adj = adj_R_squared;

end
