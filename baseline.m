imageFileList = { 'p1010847.jpg', 'p1010846.jpg','p1010845.jpg','p1010844.jpg','p1010843.jpg'};
imageBaseDir = 'images';
dataBaseDir = 'data';

%% Get all filenames from imageBaseDir
imageBaseDir = '../scene_categories_together';
structList = dir(imageBaseDir);
imageFileList = {structList.name};  % Get filenames from struct
imageFileList = imageFileList(3:end);  % Remove . and ..

%% Get labels of each 

numExamples = length(imageFileList);
valid_prefixes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ...
    '10', '11', '12', '13', '14'};

yTrain = zeros(numExamples, 1);  % liblinear requires column vector of labels
for i = 1 : numExamples
    filename = imageFileList{i};
    underscoreLocations = strfind(filename, '_');
    label = str2num(filename(1 : underscoreLocations(1) - 1));
    
    yTrain(i) = label;
end

%% Extract features of images

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

    
GenerateSiftDescriptors( imageFileList, imageBaseDir, dataBaseDir, params, canSkip, pfig );
CalculateDictionary( imageFileList, imageBaseDir, dataBaseDir, featureSuffix, params, canSkip, pfig );
H_all = BuildHistograms( imageFileList,imageBaseDir, dataBaseDir, featureSuffix, params, canSkip, pfig );
pyramid_all = CompilePyramid( imageFileList, dataBaseDir, textonSuffix, params, canSkip, pfig );

%% Train SVM

yTrain = double(yTrain);  % liblinear requires labels to be double
xTrain = pyramid_all;  % x is examples, y is labels
numFeatures = size(xTrain, 1);
model = train(yTrain, sparse(xTrain));  % liblinear requires xTrain to be sparse