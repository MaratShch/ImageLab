close all;
fclose all;
clearvars;
clc;

% select image file (noised and normal)
[name, path] = uigetfile("E:\WORK\MARAT\DATASETS\*.png");
% read BW R8G8B8 image
full_original_file_name = strcat(path, name);

% INFO about image
INFO = imfinfo(full_original_file_name);
data = imread(full_original_file_name);

minR = min(data(:,:,1), [], 'all');
maxR = max(data(:,:,1), [], 'all');

minG = min(data(:,:,2), [], 'all');
maxG = max(data(:,:,2), [], 'all');

minB = min(data(:,:,3), [], 'all');
maxB = max(data(:,:,3), [], 'all');

gamma = 0.4545;
imshow(imadjust(data, [], [], gamma));
