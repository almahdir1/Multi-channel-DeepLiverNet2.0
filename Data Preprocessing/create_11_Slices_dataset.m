% Run create 11 MRI slices dataset
% Author: Redha Ali, PhD
% Data Modified: 10/3/2022

addpath("..\Util\")
addpath("..\Excel sheet data\")

% Import the data
liverdata = readtable("liverdata.xlsx");
Image_ID = string(liverdata.Image_ID);

% add path
liver_MRI_Path = '\Dataset\';

for idx = 1:length(Image_ID)

    File_path2 = string(sprintf('%s%s',liver_MRI_Path,Image_ID(idx)));
    Image_ID3 = string(GetSubDirsFirstLevelOnly(File_path2));
    File_path3(idx,:) = string(sprintf('%s%s%s',File_path2,Image_ID3));


end

% 11 Slices dataset
for idx = 1:length(Image_ID)

    disp(sprintf('cases name:%s ------ indx:%d',Image_ID(idx),idx))


    [raw_vol, slice_data, image_meta_data] = dicom23D(raw_nifti_path);

    normalized_volum = normalize_mean_std(raw_vol);

    [sx, sy, sz] = size(normalized_volum);

    mid = floor(sz/2);

    volume_image1 = (normalized_volum(:,:,mid));
    volume_image2 = (normalized_volum(:,:,mid+1));
    volume_image3 = (normalized_volum(:,:,mid-1));
    volume_image4 = (normalized_volum(:,:,mid+2));
    volume_image5 = (normalized_volum(:,:,mid-2));
    volume_image6 = (normalized_volum(:,:,mid+3));
    volume_image7 = (normalized_volum(:,:,mid-3));
    volume_image8 = (normalized_volum(:,:,mid+4));
    volume_image9 = (normalized_volum(:,:,mid-4));
    volume_image10 = (normalized_volum(:,:,mid+5));
    volume_image11 = (normalized_volum(:,:,mid-5));

    output_fov1 = unified_field_of_view(volume_image1, image_meta_data, 256, 256);

    out = cat(3,output_fov1,output_fov1,output_fov1);

    folder = 'Pre_processed Dataset\Med\';
    if ~exist(folder, 'dir')
    mkdir(folder)
    end

    baseFileName = sprintf('%s_%s',Image_ID(idx),'Med');
    fullFileName = fullfile(folder, baseFileName);
    save([folder num2str(baseFileName) '.mat'],'out')
    clearvars out output_fov1 seg_fov1 out_1


    output_fov2 = unified_field_of_view(volume_image2, image_meta_data, 256, 256);

    out_2 = output_fov2;
    out = cat(3,out_2,out_2,out_2);
    

    folder = 'Pre_processed Dataset\Med_1\';
    
    if ~exist(folder, 'dir')
    mkdir(folder)
    end

    baseFileName = sprintf('%s_%s',Image_ID(idx),'Med_1');
    fullFileName = fullfile(folder, baseFileName);
    save([folder num2str(baseFileName) '.mat'],'out')
    clearvars out output_fov2 seg_fov2 out_2

    output_fov3 = unified_field_of_view(volume_image3, image_meta_data, 256, 256);


    out_3 = output_fov3;
    out = cat(3,out_3,out_3,out_3);
    

    folder = 'Pre_processed Dataset\Med__1\';
    if ~exist(folder, 'dir')
    mkdir(folder)
    end

    baseFileName = sprintf('%s_%s',Image_ID(idx),'Med__1');
    fullFileName = fullfile(folder, baseFileName);
    save([folder num2str(baseFileName) '.mat'],'out')
    clearvars out output_fov3 seg_fov3 out_3

    output_fov4 = unified_field_of_view(volume_image4, image_meta_data, 256, 256);


    out_4 = output_fov4;
    out = cat(3,out_4,out_4,out_4);
    

    folder = 'Pre_processed Dataset\Med_2\';
    if ~exist(folder, 'dir')
    mkdir(folder)
    end

    baseFileName = sprintf('%s_%s',Image_ID(idx),'Med_2');
    fullFileName = fullfile(folder, baseFileName);
    save([folder num2str(baseFileName) '.mat'],'out')
    clearvars out output_fov4 seg_fov4 out_4

    output_fov5 = unified_field_of_view(volume_image5, image_meta_data, 256, 256);


    out_5 = output_fov5;
    out = cat(3,out_5,out_5,out_5);
    

    folder = 'Pre_processed Dataset\Med__2\';
    if ~exist(folder, 'dir')
    mkdir(folder)
    end

    baseFileName = sprintf('%s_%s',Image_ID(idx),'Med__2');
    fullFileName = fullfile(folder, baseFileName);
    save([folder num2str(baseFileName) '.mat'],'out')
    clearvars out output_fov5 seg_fov5 out_5


    output_fov6 = unified_field_of_view(volume_image6, image_meta_data, 256, 256);

    out_6 = output_fov6;
    out = cat(3,out_6,out_6,out_6);
    

    folder = 'Pre_processed Dataset\Med_3\';
    if ~exist(folder, 'dir')
    mkdir(folder)
    end

    baseFileName = sprintf('%s_%s',Image_ID(idx),'Med_3');
    fullFileName = fullfile(folder, baseFileName);
    save([folder num2str(baseFileName) '.mat'],'out')
    clearvars out output_fov6 seg_fov6 out_6


    output_fov7 = unified_field_of_view(volume_image7, image_meta_data, 256, 256);

    out_7 = output_fov7;
    out = cat(3,out_7,out_7,out_7);
    

    folder = 'Pre_processed Dataset\Med__3\';
    if ~exist(folder, 'dir')
    mkdir(folder)
    end

    baseFileName = sprintf('%s_%s',Image_ID(idx),'Med__3');
    fullFileName = fullfile(folder, baseFileName);
    save([folder num2str(baseFileName) '.mat'],'out')
    clearvars out output_fov7 seg_fov7 out_7


    output_fov8 = unified_field_of_view(volume_image8, image_meta_data, 256, 256);

    out_8 = output_fov8;
    out = cat(3,out_8,out_8,out_8);
    

    folder = 'Pre_processed Dataset\Med_4\';
    if ~exist(folder, 'dir')
    mkdir(folder)
    end

    baseFileName = sprintf('%s_%s',Image_ID(idx),'Med_4');
    fullFileName = fullfile(folder, baseFileName);
    save([folder num2str(baseFileName) '.mat'],'out')
    clearvars out output_fov8 seg_fov8 out_8


    output_fov9 = unified_field_of_view(volume_image9, image_meta_data, 256, 256);

    out_9 = output_fov9;
    out = cat(3,out_9,out_9,out_9);
    

    folder = 'Pre_processed Dataset\Med__4\';
    if ~exist(folder, 'dir')
    mkdir(folder)
    end

    baseFileName = sprintf('%s_%s',Image_ID(idx),'Med__4');
    fullFileName = fullfile(folder, baseFileName);
    save([folder num2str(baseFileName) '.mat'],'out')
    clearvars out output_fov9 seg_fov9 out_9


    output_fov10 = unified_field_of_view(volume_image10, image_meta_data, 256, 256);

    out_10 = output_fov10;
    out = cat(3,out_10,out_10,out_10);
    

    folder = 'Pre_processed Dataset\Med_5\';
    if ~exist(folder, 'dir')
    mkdir(folder)
    end

    baseFileName = sprintf('%s_%s',Image_ID(idx),'Med_5');
    fullFileName = fullfile(folder, baseFileName);
    save([folder num2str(baseFileName) '.mat'],'out')
    clearvars out output_fov10 seg_fov10 out_10

    output_fov11 = unified_field_of_view(volume_image11, image_meta_data, 256, 256);

    out_11 = output_fov11;
    out = cat(3,out_11,out_11,out_11);
    

    folder = 'Pre_processed Dataset\Med__5\';
    if ~exist(folder, 'dir')
    mkdir(folder)
    end

    baseFileName = sprintf('%s_%s',Image_ID(idx),'Med__5');
    fullFileName = fullfile(folder, baseFileName);
    save([folder num2str(baseFileName) '.mat'],'out')
    clearvars out output_fov11 seg_fov11 out_11

    clearvars  subDirsNames volume_image image_meta_data normalized_volum_11 volume_image1 volume_image2 volume_image3

end
