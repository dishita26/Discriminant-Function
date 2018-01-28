function [testPredLabel]=HW2_q3_discriminant_v2(trainFeat,trainLabel,testFeat)

class = unique(trainLabel);
for c = 1:length(class)
    Prior                   = length(trainLabel(trainLabel==class(c))) / length(trainLabel);
    c_id                    = find(trainLabel == class(c));
    covM                    = cov(trainFeat(c_id,:));
    covInv                  = covM^-1;
    mu                      = mean(trainFeat(c_id,:))';
    bigW                    = -0.5 * covInv;
    smallW                  = covInv * mu;
    constantW               = -0.5*mu'*covInv*mu - 0.5*log(det(covM)) + log(Prior);
    discriminantFuncG(:,c)  = diag(bsxfun(@plus, testFeat * bigW * testFeat', smallW' * testFeat' + constantW));
end
[~,testPredLabel] = max(discriminantFuncG,[],2);
