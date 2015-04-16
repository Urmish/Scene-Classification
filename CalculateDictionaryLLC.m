function [B] = CalculateDictionaryLLC(B_init, X, sigma, lambda)
% Implements Iterative codebook optimization, Algorithm 4.1 of LLC paper (Wang et al 2010)
% Inputs
%     `B_init` Initial dictionary M x D
%     `X` Features N x D
%     `sigma` Distance decay, higher = faster decay per unit distance
%     `lambda` Regularization parameter, higher = more regularization
% Outputs
%     `B` M x D updated dictionary
% 

%% Perform incremental codebook optimization

M = size(B_init, 1);  % No. of entries in codebook
[N, D] = size(X);  % No. of instances, dimensionality of code
% sigma = 1;  % Distance decay, higher = faster decay per unit distance
% lambda = 1;  % Regularization parameter, higher = more regularization

B = B_init;
for x_ind = 1 : N
    
    x_i = X(x_ind, :);
    
    % Get locality constraint d of this descriptor w.r.t the dictionary
    dist = sqrt(sum((B - repmat(x_i, M, 1)).^2, 2));  % Euclidean distance of x_i from each codebook entry
    dist = dist - max(dist);  % So that d is between zero and 1
    d = exp(dist / sigma);  % M x 1 matrix
    
    % Encode descriptor, constrained by d
    z = B - repmat(x_i, M, 1);
    C = z*z';  % Data covariance matrix
    C = C + lambda*diag(d);
    c_constrained = C \ ones(M,1);  % M x 1 matrix
    c_constrained = c_constrained / sum(c_constrained); % Enforce sum(w)=1
    
    % Get codebook with entries significant to this descriptor
    inds = find(abs(c_constrained) > 0.01);
    B_i = B(inds, :);
    numEntries = length(inds);
    
    % Encode descriptor with smaller dictionary, without locality
    % constraint
    z = B_i - repmat(x_i, numEntries, 1);
    C = z*z';
    C = C + lambda * eye(numEntries) *trace(C);
    c_unc = C \ ones(numEntries,1);
    c_unc = c_unc / sum(c_unc);
    
%     error = norm(x_i - c_unc' * B_i);
%     errors(x_ind) = error;
    
    % Update smaller dictionary
    learning_rate = sqrt(1 / x_ind);  % decays with time
    delta_B = -2 * c_unc * (x_i - c_unc' * B_i);
    B_i = B_i - learning_rate * delta_B / norm(c_unc);
    
    % Replace rows of larger dictionary with new updated entries
    B(inds, :) = B_i;
    
%     fprintf('instance #%i, error = %f\n', x_ind, error);
end
