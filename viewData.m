width = 320;
height = 240;

fileName = 'D://colorMap.raw';

fid = fopen(fileName, 'rb');
rawData = fread(fid, width * height * 3, 'float32');
fclose(fid);

img = reshape(rawData, [3, width, height]);
img = permute(img, [3, 2, 1]);  % Reorder to [height, width, channels]
imshow(img);