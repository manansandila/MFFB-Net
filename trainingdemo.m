
close all
clear all
clc
load '3FBBnetworkpolyp'
Folder = 'C:\Users\Malik\Desktop\4 disease paper new network';% Main directory to all images
train_img_dir = fullfile(Folder,'skin img');%Training image directory
imds = imageDatastore(train_img_dir); 
classes = ["Tumor","NoTumor"]; %% Class names
labelIDs   = [255 0]; % Class id
train_label_dir = fullfile(Folder,'S mask');  %% Training label directory
pxds = pixelLabelDatastore(train_label_dir,classes,labelIDs);
tbl = countEachLabel(pxds); % 
frequency = tbl.PixelCount/sum(tbl.PixelCount); % frequency of each class
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;    % frequency balancing median 
pxLayer = pixelClassificationLayer('Name','labels','ClassNames',tbl.Name,'ClassWeights',classWeights); % adding weights tp pixel classification layer

lgraph = removeLayers(lgraph_1,'labels'); % deleting previous layer
lgraph = addLayers(lgraph, pxLayer); % adding new layer with weights
lgraph = connectLayers(lgraph,'softmax','labels');% retreiving the connection


%%% Training options %%%%%

options = trainingOptions('sgdm', ...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-4, ...
    'L2Regularization',0.0005, ...
    'MaxEpochs',50, ...  
    'MiniBatchSize',8, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2);


augment_data = imageDataAugmenter('RandXReflection',true,...
    'RandXTranslation',[-5 5],'RandYTranslation',[-5 5]); % optional data augmentation


training_data = pixelLabelImageDatastore(imds,pxds,...
    'DataAugmentation',augment_data); %% complete image+label data


[net, info] = trainNetwork(training_data,lgraph,options);% Train the network