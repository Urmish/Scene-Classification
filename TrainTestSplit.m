function [ strTrain, strTest, yTrain, yTest ] = TrainTestSplit( strInstances, labels, numTrainPerClass )
%TRAINTESTSPLIT Stratified train/test split where instances are filenames
% Inputs:
%     `strInstances` M x 1 cell array of instances
%     `labels` M x 1 array of integer labels
%     `numTrainPerClass` Number of training examples per class. Takes all
%     instances in the class as training examples if number of instances in
%     that class is less than this.
% Outputs:
%     `fTrain` Filenames of training instances
%     `fTest` Filenames of test instances
%     `yTrain` Labels of training instances
%     `yTest` Labels of test instances

strTrain = {};
strTest = {};
yTrain = [];
yTest = [];

labeledInstances = [strInstances' num2cell(labels)'];
for i = unique(labels)
    
    % Get indices of a random sample of labeled instances that match this label
    indAll = find(cell2mat(labeledInstances(:, 2)) == i);
    
    % Split indices into training and test
    indAll = indAll(randperm(numel(indAll)));  % shuffle indices
    if (numel(indAll) >= numTrainPerClass)
        indTrain = indAll(1 : numTrainPerClass);
        indTest = indAll(numTrainPerClass + 1: end);
    else
        indTrain = indAll;
        indTest = [];
    end
    
    %labeledInstancesTrain = labeledInstances(indTrain, 2);
    %labeledInstancesTest = labeledInstances(indTest, :);
    
    strTrain = [strTrain labeledInstances(indTrain, 1)'];
    strTest = [strTest labeledInstances(indTest, 1)'];
    yTrain = [yTrain; cell2mat(labeledInstances(indTrain, 2))];
    yTest = [yTest; cell2mat(labeledInstances(indTest, 2))];
    
end

end

