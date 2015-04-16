function PrintLabelDistribution( labels )
%PRINTLABELDISTRIBUTION Print distribution of labels
%   Detailed explanation goes here


labelSet = unique(labels);
counts = hist(labels, labelSet);
fprintf('Labels: %s\n', sprintf('%4i ', labelSet));
fprintf('Counts: %s\n', sprintf('%4i ', counts));

end

