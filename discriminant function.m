% 0. Data Import
close all;clc;clear all;
load iris.mat
nFold=loadjson('SETTINGS.json');
N = str2num(nFold.N);

% 1. 3-class classification
disp('Question 2 starts...');
class = unique(label);
for i = 1:size(class,1)
    foldSize = []; 
    featTemp = feat(label==class(i),:);
    classIndex = find(label==class(i));
    classCount = numel(classIndex);
    rng('shuffle');
    featTemp = featTemp(randperm(classCount),:);
    quotient = floor(classCount / N);
    remainder = mod(classCount, N);
    rng('shuffle');
    extra = randsample(1:N,remainder);
    foldSize = ones(N,1) * quotient;
    foldSize(extra) = foldSize(extra) + 1;
    foldSizeCum = cumsum(foldSize);
    for j = 1:N
        if j == 1
            eval(['nFold.class', num2str(i), '.fold', num2str(j), ...
                '=featTemp(', num2str(1), ':', num2str(foldSizeCum(1)), ',:);']);
        else
            eval(['nFold.class', num2str(i), '.fold', num2str(j), ...
                '=featTemp(', num2str(foldSizeCum(j-1)+1), ':', num2str(foldSizeCum(j)), ',:);']);
        end
    end
end

q2AccuracyMatrix = [];
for i = 1:N    
    testFeat = []; testLabel = []; trainFeat = []; trainLabel = [];
    % Aggregating testing data
    for j = 1:size(class,1)
        eval(['testFeat = [testFeat; nFold.class', num2str(j), '.fold', num2str(i), '];']);
        eval(['testLabel = [testLabel; j * ones(size(nFold.class', num2str(j), '.fold', num2str(i), ',1),1)];']);
    end
   
    % Aggregating training data
    for j = 1:size(class,1)
        for k = 1:N
            if k ~= i
                eval(['trainFeat = [trainFeat; nFold.class', num2str(j), '.fold', num2str(k), '];']);
                eval(['trainLabel = [trainLabel; j * ones(size(nFold.class', num2str(j), '.fold', num2str(k), ',1),1)];']);;
            end
        end
    end
    
    % Bayesian
    testPredLabel = HW2_q2_Bayesian(trainFeat, trainLabel, testFeat);
%     class = unique(trainLabel);
%     for c = 1:length(class)
%         Prior = length(trainLabel(trainLabel==class(c))) / length(trainLabel);
%         c_id = find(trainLabel == c);
%         Posterior(:,c) = Prior * mvnpdf(testFeat, mean(trainFeat(c_id,:)), cov(trainFeat(c_id,:))); 
%     end
%     [num,testPredLabel]=max(Posterior,[],2);
    
    % Classification accuracy and Confusion matrix
    Accuracy = sum(testPredLabel==testLabel) / length(testLabel);
    eval(['q2confusionMatrix.fold', num2str(i), ' = confusionmat(testLabel, testPredLabel);']);
    if i == 1
        q2confusionMatrix.allFolds = q2confusionMatrix.fold1;
    else
        eval(['q2confusionMatrix.allFolds = q2confusionMatrix.allFolds + q2confusionMatrix.fold', num2str(i),';']);
    end    
    q2AccuracyMatrix = [q2AccuracyMatrix; Accuracy];
end
disp('Question 2 is finished...');
mean(q2AccuracyMatrix)
q2confusionMatrix.allFolds