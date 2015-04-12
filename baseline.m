imageFileList = { 'p1010847.jpg', 'p1010846.jpg','p1010845.jpg','p1010844.jpg','p1010843.jpg'};
imageBaseDir = 'images';
dataBaseDir = 'data';

%% Get all filenames from imageBaseDir
imageBaseDir = '../scene_categories_together';
structList = dir(imageBaseDir);
imageFileList = {structList.name};  % Get filenames from struct
imageFileList = imageFileList(3:end);  % Remove . and ..

%% Get labels of every image filename

numExamples = length(imageFileList);
valid_prefixes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ...
    '10', '11', '12', '13', '14'};

labels = [];
for i = 1 : numExamples
    filename = imageFileList{i};
    underscoreLocations = strfind(filename, '_');
    label = str2num(filename(1 : underscoreLocations(1) - 1));
    
    labels(i) = label;
end


%% Split data into train and test sets

numTrainPerClass = 3;  % Number of training examples per class;
fTrain = {};  % Filenames of training instances
fTest = {};  % Filenames of test instances
yTrain = [];  % Labels of training instances
yTest = [];  % Labels of test instances

labeledInstances = [imageFileList' num2cell(labels)'];
for i = 0 : 14
    
    % Get indices of a random sample of labeled instances that match this label
    indAll = find(cell2mat(labeledInstances(:, 2)) == i);
    
    % Split indices into training and test
    indAll = indAll(randperm(numel(indAll)));  % shuffle indices
    indTrain = indAll(1 : numTrainPerClass);
    indTest = indAll(numTrainPerClass + 1: end);
    
    %labeledInstancesTrain = labeledInstances(indTrain, 2);
    %labeledInstancesTest = labeledInstances(indTest, :);
    
    fTrain = [fTrain labeledInstances(indTrain, 1)'];
    fTest = [fTest labeledInstances(indTest, 1)'];
    yTrain = [yTrain; cell2mat(labeledInstances(indTrain, 2))];
    yTest = [yTest; cell2mat(labeledInstances(indTest, 2))];
    
end

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
CalculateDictionary( imageFileList, imageBaseDir, dataBaseDir, featureSuffix, params, canSkip, pfig );

% Build histograms from both training and test images
imageFileList = [fTrain fTest];
H_all = BuildHistograms( imageFileList,imageBaseDir, dataBaseDir, featureSuffix, params, canSkip, pfig );

% Calculate feature vectors for training and test images separately
xTrain = CompilePyramid( fTrain, dataBaseDir, textonSuffix, params, canSkip, pfig );
xTest = CompilePyramid( fTest, dataBaseDir, textonSuffix, params, canSkip, pfig );


%% Train SVM

yTrain = double(yTrain);  % liblinear requires labels to be double
xTrain = pyramid_all;  % x is examples, y is labels
numFeatures = size(xTrain, 1);
model = train(yTrain, sparse(xTrain));  % liblinear requires xTrain to be sparse


%% Predict labels for test images

% Labels are required to compute accuracy. Just a convenient feature.
[predicted_label, accuracy, ~] = predict(yTest, sparse(xTest), model);