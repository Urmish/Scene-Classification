function [beta] = LLC_pool(features, dictionary, pyramidLevels, neighbors)
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


img_width = features.wid;
img_height = features.hgt;

% llc coding
llc_codes = LLC_Coding_2(dictionary, features, neighbors,1);
llc_codes = llc_codes';

%Setting up pyramid structure

beta = zeros(d_size, total_Regions);
loopCount = 0;

for current_Level = 1:pyramidLevels
    
    scaled_width = img_width / pyramid(current_Level);
    scaled_height = img_height / pyramid(current_Level);
    t_subRegions = sub_Regions(current_Level);
    
    % find to which spatial bin each local descriptor belongs
    x_subRegion = ceil(features.x / scaled_width);
    y_subRegion = ceil(features.y / scaled_height);
    index_Region = (y_subRegion - 1)*pyramid(current_Level) + x_subRegion;
    for current_Region = 1:t_subRegions
        loopCount = loopCount + 1;
        %For each bin, find out which features lie in it, so that they
        %could be maxed!!
        %NOTE: SPM used sum pooling instead of max
        target_indexes = find(index_Region == current_Region);
        if isempty(target_indexes),
            continue;
        end      
        beta(:, loopCount) = max(llc_codes(:, target_indexes), [], 2);
    end
    
end

if loopCount ~= total_Regions,
    error('Index number error!');
end

beta = beta(:);
beta = beta./sqrt(sum(beta.^2));
end

