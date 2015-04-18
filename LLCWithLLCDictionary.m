% Implements Algorithm 4.1 of LLC paper (Wang et al 2010)
% Iterative codebook optimization

warning('off', 'MATLAB:hg:EraseModeIgnored')
dataBaseDir = 'LLC_2/data';

rng(0);  % Seed RNG so that randomization is deterministic

%% Get all filenames from imageBaseDir

%rdir http://www.mathworks.com/matlabcentral/fileexchange/19550-recursive-directory-listing
imageBaseDir = 'CS766-3';
structList = rdir('CS766-3/*/*.jpg');
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


%% Extract features

GenerateSiftDescriptors( imageFileList, imageBaseDir, dataBaseDir, params, canSkip, pfig );

initialDictFilename = 'LLC_2/data/dictionary_256.mat';
load(initialDictFilename);
B_init = dictionary;
fprintf('Loaded initial dictionary from %s\n', initialDictFilename);


%% Parameters to build dictionary

reduce_flag = 1;
ndata_max = 100000; %use 4% avalible memory if its greater than the default


if(~exist('params','var'))
    params.maxImageSize = 1000;
    params.gridSpacing = 8;
    params.patchSize = 16;
    params.dictionarySize = 256;
    params.numTextonImages = 50;
    params.pyramidLevels = 3;
end
if(~isfield(params,'maxImageSize'))
    params.maxImageSize = 1000;
end
if(~isfield(params,'gridSpacing'))
    params.gridSpacing = 8;
end
if(~isfield(params,'patchSize'))
    params.patchSize = 16;
end
if(~isfield(params,'dictionarySize'))
    params.dictionarySize = 200;
end
if(~isfield(params,'numTextonImages'))
    params.numTextonImages = 50;
end
if(~isfield(params,'pyramidLevels'))
    params.pyramidLevels = 3;
end
if(~exist('canSkip','var'))
    canSkip = 1;
end

if(params.numTextonImages > length(imageFileList))
    params.numTextonImages = length(imageFileList);
end

outFName = fullfile(dataBaseDir, sprintf('dictionary_%d.mat', params.dictionarySize));
    

%% load file list and determine indices of training images

inFName = fullfile(dataBaseDir, 'f_order.txt');
if ~isempty(dir(inFName))
    R = load(inFName, '-ascii');
    if(size(R,2)~=length(imageFileList))
        R = randperm(length(imageFileList));
        sp_make_dir(inFName);
        save(inFName, 'R', '-ascii');
    end
else
    R = randperm(length(imageFileList));
    sp_make_dir(inFName);
    save(inFName, 'R', '-ascii');
end

training_indices = R(1:params.numTextonImages);

%% load all SIFT descriptors

sift_all = [];

if(exist('pfig','var'))
    tic;
end

for f = 1:params.numTextonImages    
    
    imageFName = imageFileList{training_indices(f)};
    [dirN base] = fileparts(imageFName);
    baseFName = fullfile(dirN, base);
    inFName = fullfile(dataBaseDir, sprintf('%s%s', baseFName, featureSuffix));
    if(exist(inFName,'file'))
        load(inFName, 'features');
    else
        features = sp_gen_sift(fullfile(imageBaseDir, imageFName),params);
    end
    ndata = size(features.data,1);

    data2add = features.data;
    if(size(data2add,1)>ndata_max/params.numTextonImages )
        p = randperm(size(data2add,1));
        data2add = data2add(p(1:floor(ndata_max/params.numTextonImages)),:);
    end
    sift_all = [sift_all; data2add];
    %fprintf('Loaded %s, %d descriptors, %d so far\n', inFName, ndata, size(sift_all,1));
    if(mod(f,10)==0 && exist('pfig','var'))
        sp_progress_bar(pfig,2,4,f,params.numTextonImages,'Computing Dictionary: ');
    end
end

fprintf('\nTotal descriptors loaded: %d\n', size(sift_all,1));

ndata = size(sift_all,1);    
if (reduce_flag > 0) & (ndata > ndata_max)
    fprintf('Reducing to %d descriptors\n', ndata_max);
    p = randperm(ndata);
    sift_all = sift_all(p(1:ndata_max),:);
end

%% Perform incremental codebook optimization

B_init = B_init;  % Initial dictionary M x D
X = sift_all;  % Features N x D
lambda = 10^-4;
sigma = 1;

fprintf('Performing incremental codebook optimization... '); tic;
B = CalculateDictionaryLLC(B_init, X, sigma, lambda);

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

%% Print histogram of label distribution

labelSet = unique(labels);
counts = hist(labels, labelSet);
fprintf('Distribution of labels in entire dataset of %i instances:\n', numExamples);
fprintf('Labels: %s\n', sprintf('%4i ', labelSet));
fprintf('Counts: %s\n', sprintf('%4i ', counts));

%% Split data into train and test sets

numTrainPerClass = 100;  % Number of training examples per class;
fTrain = {};  % Filenames of training instances
fTest = {};  % Filenames of test instances
yTrain = [];  % Labels of training instances
yTest = [];  % Labels of test instances

labeledInstances = [imageFileList' num2cell(labels)'];
numTestImagesPerClass = zeros(length(unique(labels)),1);
for i = unique(labels)
    
    % Get indices of a random sample of labeled instances that match this label
    indAll = find(cell2mat(labeledInstances(:, 2)) == i);
    
    % Split indices into training and test
    indAll = indAll(randperm(numel(indAll)));  % shuffle indices
    if (numel(indAll) >= numTrainPerClass)
        indTrain = indAll(1 : numTrainPerClass);
        indTest = indAll(numTrainPerClass + 1: end);
        numTestImagesPerClass(i) = length(indTest);
    else
        indTrain = indAll;
        indTest = [];
    end
    
    %labeledInstancesTrain = labeledInstances(indTrain, 2);
    %labeledInstancesTest = labeledInstances(indTest, :);
    
    fTrain = [fTrain labeledInstances(indTrain, 1)'];
    fTest = [fTest labeledInstances(indTest, 1)'];
    yTrain = [yTrain; cell2mat(labeledInstances(indTrain, 2))];
    yTest = [yTest; cell2mat(labeledInstances(indTest, 2))];
    
end

imageFileList = [fTrain fTest];

% Calculate feature vectors for training and test images separately
xTrain = CompilePyramid_LLC( fTrain, dataBaseDir, featureSuffix, params, pfig );
xTest = CompilePyramid_LLC( fTest, dataBaseDir, featureSuffix, params, pfig );


%% Train SVM

yTrain = double(yTrain);  % liblinear requires labels to be double
%xTrain = pyramid_all;  % x is examples, y is labels
numFeatures = size(xTrain, 1);
model = train(yTrain, sparse(xTrain));  % liblinear requires xTrain to be sparse


%% Predict labels for test images

% Labels are required to compute accuracy. Just a convenient feature.
[predicted_label, accuracy, ~] = predict(yTest, sparse(xTest), model);
no_of_image_classes = length(unique(labels));
per_class_accuracy = zeros(no_of_image_classes,1);
confusion_matrix = zeros(no_of_image_classes,no_of_image_classes);
start = 1;
for i=1:no_of_image_classes
        count = 0;
        m_factor = 0;
        predictions = predicted_label(start:start+numTestImagesPerClass(i)-1);
%     if (i > 1)
%         m_factor = numTestImagesPerClass(i-1);
%     end
    for j=1:numTestImagesPerClass(i)
%         if (predicted_label((i-1)*m_factor + j) == i-1)
        if (predictions(j) == i)
            count = count+1;
        end
%         confusion_matrix(i,predicted_label((i-1)*m_factor + j)) = confusion_matrix(i,predicted_label((i-1)*m_factor + j))+1;
          confusion_matrix(i,predictions(j)) = confusion_matrix(i,predictions(j))+1;
    end
    per_class_accuracy(i) = (count/numTestImagesPerClass(i));
    confusion_matrix(i,:) = confusion_matrix(i,:)./numTestImagesPerClass(i);
    start = numTestImagesPerClass(i)+start;
end
mean_accuracy = sum(per_class_accuracy)*100/no_of_image_classes;
sprintf('Mean accuracy is %f',mean_accuracy)
toc;