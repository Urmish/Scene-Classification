warning('off', 'MATLAB:hg:EraseModeIgnored')
dataBaseDir = 'data';

rng(0);  % Seed RNG so that randomization is deterministic

%% Get all filenames from imageBaseDir

%rdir http://www.mathworks.com/matlabcentral/fileexchange/19550-recursive-directory-listing
imageBaseDir = '../scene_categories';
structList = rdir('../scene_categories/*/*.jpg');
imageFileList = {structList.name};  % Get filenames from struct
numExamples = length(imageFileList);

%% Subsample from image filenames to test whether code functions

if (exist('doSubsample', 'var') && doSubsample)
    % Shuffle image filenames
    imageFileList = imageFileList(randperm(numExamples));

    % Retain a subset
    subsample_size = 100;  % number of filenames to retain
    imageFileList = imageFileList(1:subsample_size);
end


%% Get labels of every image filename

% Get subdirectory of each image filename
subdirs = cellfun(@fileparts, imageFileList, 'UniformOutput', 0);

% Subdirectories are string labels for each class
unique_subdirs = unique(subdirs);

numExamples = length(imageFileList);

labels = [];
for i = 1 : numExamples
    subdir = subdirs{i};
    label = find(ismember(unique_subdirs, subdir));
    
    labels(i) = label;
end

%% Split data into train and test sets

numTrainPerClass = 10;  % Number of training examples per class;
[fTrain, fTest, yTrain, yTest] = TrainTestSplit( imageFileList, labels, numTrainPerClass )


%% Print histogram of label distribution

fprintf('Distribution of labels in entire dataset of %i instances:\n', numExamples);
PrintLabelDistribution(labels);
fprintf('Distribution of labels in training set of %i instances:\n', length(yTrain));
PrintLabelDistribution(yTrain);
fprintf('Distribution of labels in test set of %i instances:\n', length(yTest));
PrintLabelDistribution(yTest);


%% Define parameters of feature extraction

params.maxImageSize = 1000;
params.gridSpacing = 8;
params.patchSize = 16;
params.dictionarySize = 200;
params.numTextonImages = 50;
params.pyramidLevels = 3;

canSkip = 1;
pfig = figure;

% Default suffix where SIFT features are stored. One mat file is generated per image.
featureSuffix = '_sift.mat';

% Default dictionary created by CalculateDictionary. We need to delete this
% if we want to create a new dictionary.
dictFilename = sprintf('dictionary_%d.mat', params.dictionarySize);

% Default suffix of files created by BuildHistograms
textonSuffix = sprintf('_texton_ind_%d.mat',params.dictionarySize);
histSuffix = sprintf('_hist_%d.mat', params.dictionarySize);

% Default suffix of files created by CompilePyramid
pyramidSuffix = sprintf('_pyramid_%d_%d.mat', params.dictionarySize, params.pyramidLevels);


%% Extract features of both training and test images

xTrain = [];  % feature vectors for training set
xTest = [];  % feature vectors for test set

% Generate sift descriptors from both training and test images
imageFileList = [fTrain fTest];
GenerateSiftDescriptors( imageFileList, imageBaseDir, dataBaseDir, params, canSkip, pfig );

% Calculate dictionary only from training images. IMPORTANT!!
imageFileList = fTrain;
CalculateDictionary( imageFileList, imageBaseDir, dataBaseDir, featureSuffix, params, 0, pfig );

% Build histograms from both training and test images
imageFileList = [fTrain fTest];
H_all = BuildHistograms( imageFileList,imageBaseDir, dataBaseDir, featureSuffix, params, 0, pfig );

% Calculate feature vectors for training and test images separately
xTrain = CompilePyramid( fTrain, dataBaseDir, textonSuffix, params, 0, pfig );
xTest = CompilePyramid( fTest, dataBaseDir, textonSuffix, params, 0, pfig );


%% Train SVM

% Compute kernel matrix so that we can train a kernel SVM
% From the libSVM README:
% To use precomputed kernel, you must include sample serial number as
% the first column of the training and testing data
kernelMatrixTrain = hist_isect(xTrain, xTrain);
kernelMatrixTrain = [[1:size(kernelMatrixTrain, 1)]' kernelMatrixTrain];

yTrain = double(yTrain);  % liblinear requires labels to be double
svmModel = svmtrain(yTrain, kernelMatrixTrain);


%% Predict labels for test images

kernelMatrixTest = hist_isect(xTest, xTrain);
kernelMatrixTest = [[1:size(kernelMatrixTest, 1)]' kernelMatrixTest];

% Labels are required to compute accuracy. Just a convenient featuretrain .
[predicted_label, accuracy, ~] = svmpredict(yTest, kernelMatrixTest, svmModel);
%[predicted_label, accuracy, ~] = predict(yTest, sparse(xTest), model);


%% Train and predict with linear SVM
fprintf('Train and predict with linear SVM');
linearModel = train(yTrain, sparse(xTrain));
predictions = predict(yTest, sparse(xTest), linearModel);