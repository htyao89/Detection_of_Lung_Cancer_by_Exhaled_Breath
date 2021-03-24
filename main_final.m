clc;
clear;

rand('seed',0);
train_struct = load('train.mat');
test_struct = load('test.mat');
train_data = sqrt(train_struct.train_data);
test_data = sqrt(test_struct.test_data);
train_label = train_struct.train_label;
test_label = test_struct.test_label;

addpath('./liblinear-2.20/matlab/');
fprintf('train process ... ... \n');
option=['-c 1 -q'];


model = train(train_label,sparse((train_data)),option);
[p1,c1,d1] = predict(train_label,sparse((train_data)),model);
stats1=confusionmatStats(train_label,p1);
[p2,c2,test_pred_score] = predict(test_label,sparse((test_data)),model);
stats2=confusionmatStats(test_label,p2);
cMat = stats2.confusionMat;
acc2 = (cMat(1,1)+cMat(2,2))/length(test_label);


figure;
hold on
plot(test_pred_score,'r-');
plot(test_label,'g-');
hold off

%figure;
prec_rec(test_pred_score,1-test_label,'plotPR',1,'plotROC',0);

figure;
auc = roc_curve(test_pred_score,-test_label,0);

%%
% How many trees do you want in the forest? 
nTrees = 50;
% Train the TreeBagger (Decision Forest).
B = TreeBagger(nTrees,train_data,train_label, 'Method', 'classification');
predChar = B.predict(test_data);
predictedClass = str2double(predChar);
stats=confusionmatStats(test_label,predictedClass);
cMat = stats.confusionMat;
acc = (cMat(1,1)+cMat(2,2))/length(test_label);
score=[acc2,stats2.sensitivity(1),stats2.specificity(1);acc,stats.sensitivity(1),stats.specificity(1)]

% figure;
% no_dims=2;
% initial_dims=8;
% perplexity=30;
% mappedX = tsne(test_data,'Standardize',false);
% gscatter(mappedX(:,1),mappedX(:,2),test_label);
% 
% figure;
% weight = model.w; 
% data = test_data.*weight;
% mappedX = tsne(data,'Standardize',false);
% gscatter(mappedX(:,1),mappedX(:,2),test_label);

