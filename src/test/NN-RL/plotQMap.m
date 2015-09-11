clear all
close all

for i = 0 : 2
    filename = strcat('QMap',num2str(i),'.dat');
    data{i+1} = load(filename);
    figure(i + 1)
    imagesc(data{i+1});
    colorbar;
end

 data{4} = load('actionMapNN.dat');
 figure(5)
 imagesc(data{4});
 colorbar;