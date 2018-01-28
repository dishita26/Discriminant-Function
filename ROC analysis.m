% 1. Data Import
close all;clc;clear all;
class = [1;-1];

disp('Question 5 Part 4 starts. There are four cases in this part.');
disp('1:  Features 1 and 2');
disp('2:  Features 2 and 3');
disp('3:  Features 3 and 4');
disp('4:  All four features');
caseNo = input('Please enter case number: ');

filename = 'iris';
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data';
outfilename = websave(filename,url);
fileID = fopen('iris.data');
data = textscan(fileID,'%f %f %f %f %s','Delimiter',',');
fclose(fileID);
for i = 1:size(data{5},1)
       if strcmp(data{5}(i),'Iris-versicolor')
           temp(i) = class(1);
       elseif strcmp(data{5}(i),'Iris-virginica')
           temp(i) = class(2);  
       end 
end
data = [data{1} data{2} data{3} data{4} temp'];
id = find(data(:,5)==class(1) | data(:,5)==class(2));
if caseNo == 1
    feat = data(id,1:2);
    graphTitle = 'ROC curve of flower classification using features 1 and 2';
elseif caseNo == 2
    feat = data(id,2:3);
    graphTitle = 'ROC curve of flower classification using features 2 and 3';
elseif caseNo == 3
    feat = data(id,3:4);
    graphTitle = 'ROC curve of flower classification using features 3 and 4';
elseif caseNo == 4
    feat = data(id,1:4);
    graphTitle = 'ROC curve of flower classification using all features';
end
label = data(id,5);

trainFeat = []; trainLabel = []; testFeat = []; testLabel = []; 
for i = 1:size(class,1)
	id = find(label == class(i));
    classFeat = feat(id,:);
    classLabel = label(id,:);
    trainFeat = [trainFeat; classFeat(1:40,:)];
    trainLabel = [trainLabel; classLabel(1:40,:)];
    testFeat = [testFeat; classFeat(41:50,:)];
    testLabel = [testLabel; classLabel(41:50,:)];
end

sensitivity = []; specificity = [];
for decisionBoundary = -75:1:75
    testPredLabel = HW2_q3c_discriminant(trainFeat, trainLabel, testFeat, decisionBoundary);
    [sen,spe]=HW2_q3c_cal_senspe(testPredLabel,testLabel);
    sensitivity = [sensitivity; sen];
    specificity = [specificity; spe];
end

specificity = 1 - specificity;
%AUC = sum((sensitivity(1:end-1)+sensitivity(2:end))/2.*(specificity(2:end)-specificity(1:end-1)));
AUC = trapz(specificity, sensitivity);
plot(specificity, sensitivity, 'r');
xlabel('1 - specificity');
ylabel('sensitivity');
title(graphTitle);
AUC

