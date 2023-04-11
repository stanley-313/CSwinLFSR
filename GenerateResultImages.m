clear all;
clc;
close all;

%% The upscaling factor must match to the super-resolved LFs in './Results/'
factor = 4;

%%
sourceDataPath = '../BasicLFSR/datasets/';
sourceDatasets = dir(sourceDataPath);
sourceDatasets(1:2) = [];
datasetsNum = length(sourceDatasets);

resultsFolder = './Results/';

for DatasetIndex = 1 : datasetsNum
    DatasetName = sourceDatasets(DatasetIndex).name;
    gtFolder = [sourceDataPath, sourceDatasets(DatasetIndex).name, '/test/'];
    scenefiles = dir(gtFolder);
    scenefiles(1:2) = [];
    sceneNum = length(scenefiles);
    
    resultsFolder = ['./Results/', 'CSWinLFSR/', DatasetName, '/'];
    
    for iScene = 1 : sceneNum
        sceneName = scenefiles(iScene).name;
        sceneName(end-3:end) = [];
        fprintf('Generating result images of Scene_%s in Dataset %s......\n', sceneName, sourceDatasets(DatasetIndex).name);
        
        data = load([resultsFolder, sceneName, '.mat']);
        LFsr_y = data.LF;
        [angRes, ~, H, W] = size(LFsr_y);  
        
        data = load([gtFolder, '/', sceneName, '.mat']);
        LFgt_rgb = data.LF;
        LFgt_rgb = LFgt_rgb(:, :, 1:H, 1:W, 1:3);
        LFsr = zeros(angRes, angRes, H, W, 3);
        
        for u = 1 : angRes
            for v = 1 : angRes    
                imgHR_rgb = squeeze(LFgt_rgb(u, v, :, :, :));
                imgHR_ycbcr = rgb2ycbcr(imgHR_rgb);
                imgHR_ycbcr(:, :, 1) = LFsr_y(u, v, :, :);
                imgSR_rgb = ycbcr2rgb(imgHR_ycbcr);
                LFsr(u, v, :, :, :) = imgSR_rgb;                
              
                SavePath = ['./SRimages/', DatasetName, '/', sceneName, '/'];
                if exist(SavePath, 'dir')==0
                    mkdir(SavePath);
                end
                imwrite(uint8(255*imgSR_rgb), [SavePath, 'View_', num2str(u-1,'%d'), '_', num2str(v-1,'%d'), '.bmp' ]);
            end
        end        
    end
end
