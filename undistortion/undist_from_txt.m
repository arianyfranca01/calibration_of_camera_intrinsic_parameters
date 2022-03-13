clear
close all
clc
%%
dist_folder = 'folder_to_save_undistorted_images'
if ~exist(dist_folder, 'dir')
  mkdir(dist_folder);
end

fileID = fopen('path_txt_file','r');
file_content = textscan(fileID,'%s');
fclose(fileID);


pathsaux = file_content{1};
pathsaux2 = pathsaux(1:3:end);
paths = cellstr(pathsaux2);
focalaux =  pathsaux(2:3:end);
focal = str2double(focalaux );
distortionaux =  pathsaux(3:3:end);
distortion =  str2double(distortionaux);

hfig=figure;


for i=1:length(paths)

    Idis = imread(paths{i});
    f = 0;
    dist = 0;
    % xi = 1.08;
    xi = distortion(i); % distortion
    dist = dist + xi;
    [ImH,ImW,~] = size(Idis);
    % f_dist = 320 * (ImW/ImH) * (ImH/299); 
    f_dist = focal(i) * (ImW/ImH) * (ImH/299); % focal length
    f = f + f_dist;
    u0_dist = ImW/2;
    v0_dist = ImH/2;
    Paramsd.f = f_dist;
    Paramsd.W = u0_dist*2;  
    Paramsd.H = v0_dist*2;
    Paramsd.xi = xi;

    Paramsund.f = f_dist;
    Paramsund.W = u0_dist*2;  
    Paramsund.H = v0_dist*2;
    
    tic
    Image_und = undistSphIm(Idis, Paramsd, Paramsund);
    toc

    if (size(Image_und,1)~=0)
        paths_list = strsplit(paths{i}, '/');
        res1 = strsplit(paths_list{4}, '_');
        res2 = strsplit(res1{6}, '.')

        out = str2double(res2{2});
        
        filename = [sprintf('%d', out),'.jpg'];
        fullname = fullfile(dist_folder,filename);
        imwrite(Image_und,fullname);
    end
end