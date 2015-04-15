function [hist] = LLC_pool(features, dictionary, pyramidLevels, neighbors)
%LLC_POOL Summary of this function goes here
%   Detailed explanation goes here
if ~exist('neighbors', 'var') || isempty(neighbors),
    neighbors = 5;
end
%Setting up pyramid structures
pyramid = (2.^(0:(pyramidLevels-1)));
sub_Regions = pyramid.^2;
total_Regions = sum(sub_Regions);
d_size = size(dictionary, 1);%Dictionary is in transposed condition, see LLC_coding_appr call, B' is passed
hist = zeros(d_size*total_Regions,1);

img_width = features.wid;
img_height = features.hgt;

% llc coding
llc_codes = LLC_Coding_2(dictionary, features, neighbors,1);


%Setting up pyramid structure


start = 1;
for current_Level = 1:pyramidLevels
    
    scaled_width = img_width / pyramid(current_Level);
    scaled_height = img_height / pyramid(current_Level);
    t_subRegions = sub_Regions(current_Level);
    
    % find to which spatial bin each local descriptor belongs
    x_subRegion = ceil(features.x / scaled_width);
    y_subRegion = ceil(features.y / scaled_height);
    index_Region = (y_subRegion - 1)*pyramid(current_Level) + x_subRegion;
    for current_Region = 1:t_subRegions
        %For each bin, find out which features lie in it, so that they
        %could be maxed!!
        %NOTE: SPM used sum pooling instead of max
        target_indexes = find(index_Region == current_Region);
        if isempty(target_indexes),
            continue;
        end      
        hist(start+(current_Region-1)*d_size:start+current_Region*d_size-1,1) = max(llc_codes(:, target_indexes), [], 2);
    end
    start = start + d_size*t_subRegions;
end

hist = hist./sqrt(sum(hist.^2));
end
