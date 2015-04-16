function [hist] = LLC_pool(features, dictionary, pyramidLevels, neighbors)
%LLC_POOL Summary of this function goes here
%   Detailed explanation goes here
if ~exist('neighbors', 'var') || isempty(neighbors),
    neighbors = 5;
end
%Setting up pyramid structures
pyramid = (2.^(0:(pyramidLevels-1))); %Needed to identify in which region does a particular point
%lies on each pyramid level
sub_Regions = pyramid.^2; %Number of regions in each pyramid level
total_Regions = sum(sub_Regions);
d_size = size(dictionary, 1);%Dictionary is in transposed condition, see LLC_coding_appr call, B' is passed
hist = zeros(d_size*total_Regions,1);
%The output of this function should be a nx1 matrix as expected by the
%CompilePyramid script

img_width = features.wid;
img_height = features.hgt;

% llc coding
llc_codes = LLC_Coding_2(dictionary, features, neighbors,1);

start = 1;
for current_Level = 1:pyramidLevels
    %We need to find out in which region does each feature point lies.
    %The way this is done here is that, for each level find out on which
    %side do x and y coordinates lie. This is the region coordinate. Then
    %use the region coordinate to pinpoint the region number
    scaled_width = img_width / pyramid(current_Level);
    scaled_height = img_height / pyramid(current_Level);
    %The scaled values are used to divide x and y coordinates into
    %pyramid(current_level) regions
    x_subRegion = ceil(features.x / scaled_width);
    y_subRegion = ceil(features.y / scaled_height);
    %Now I have the region coordinates for each feature point
    
    index_Region = (y_subRegion - 1)*pyramid(current_Level) + x_subRegion;
    %Now I have the region number for each coordinate
    %Region Number is numbered from left to right and then moving down
    % 1 2 3 4
    % 5 6 7 8 etc
    for current_Region = 1:sub_Regions(current_Level)
        %For each region, find out which features lie in it, so that they
        %could be maxed!!
        %NOTE: SPM used sum pooling instead of max
        target_indexes = find(index_Region == current_Region);
        %There might be certain regions without any feature points
        if isempty(target_indexes),
            continue;
        end      
        hist(start+(current_Region-1)*d_size:start+current_Region*d_size-1,1) = max(llc_codes(:, target_indexes), [], 2);
    end
    start = start + d_size*sub_Regions(current_Level);
end

hist = hist./sqrt(sum(hist.^2));
end
