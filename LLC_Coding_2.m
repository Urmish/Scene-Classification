function [c_hat] = LLC_Coding_2(dictionary, features, neighbors, regularization, beta)
%LLC_CODING_2 Summary of this function goes here
%   Detailed explanation goes here
f = features.data;
d = dictionary;
f_size=size(f,1);%Number of descriptors in an image
d_size=size(d,1);%Number of dictionary elements

if ~exist('neighbors', 'var') || isempty(neighbors),
    knn = 5;
end
if ~exist('regularization', 'var') || isempty(regularization),
    regularization = 0;
end

% Calculate the distance matrix
% Using the method used in sp_dist2 to do so.
n2 = (ones(d_size, 1) * sum((f.^2)', 1))' + ones(f_size, 1) * sum((d.^2)',1) - 2.*(f*(d'));
%n2 has the distance of each point in one matrix with that in another
%matrix

% Calculation of nearest neighbors is just a sorting algorithm now
nn_index = zeros(f_size, neighbors);
for i = 1:f_size,
	dist = n2(i,:);
    %Each row represents the distance of i with every other point in the
    %second matrix, we need to pick the best n among them.
	[~, temp_index] = sort(dist, 'ascend'); 
	nn_index(i, :) = temp_index(1:neighbors);
end

c_hat = zeros(f_size, d_size); %Initialize c_hat variable
%This could be changed to return 2 things, c values and indexes of
%neighbors. Will look something like this
% c_hat = zeros(f_size, neigbors, neigbors);
if (~regularization)
    for i=1:f_size
          %Taken from demo.m as provided with the assignment
%         B = randn( neighbors, size(dictionary,2));
%         % create truth code
%         c = randn(neighbors, 1);
%         c = c /sum(c);
%         % compute feature
%         x = B'*c;
        n_temp = nn_index(i,:);
        one = ones(neighbors, 1);
        x = f(i,:);
        B_1x = d(n_temp,:) - one*x;
        C = B_1x * B_1x';
        w = C \ one;
        w = w /sum(w);
        c_hat(i,n_temp) = w';
    end
else
    %Used only for experimental purposes. This part of the code is only
    %used to evaluate our implementation performance with their
    %implementation performance. The intention is to understand the effect
    %of beta on the final output.
    t = ones(neighbors,1);
    t = diag(t);
    if ~exist('beta', 'var') || isempty(beta),
        beta = 1e-4;
    end
    for i=1:f_size
        n_temp = nn_index(i,:);
        z = d(n_temp,:) - repmat(f(i,:), neighbors, 1);           % shift ith pt to origin
        C = z*z';                                        % local covariance
        C = C + t*beta*trace(C);                        % regularlization (K>D)
        w = C\ones(neighbors,1);
        w = w/sum(w);                                    % enforce sum(w)=1
        c_hat(i,n_temp) = w';
    end
end
c_hat = c_hat';
end