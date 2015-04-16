% Implements Algorithm 4.1 of LLC paper (Wang et al 2010)
% Iterative codebook optimization

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

initialDictFilename = 'data/dictionary_200.mat';
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
    params.dictionarySize = 200;
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
toc;