function [sig, C, mu] = kelmtrain(img, img_gt, train_idx,test_idx ,sigs, Cs, mus, wind, rows, cols)
kerneloption = [1];
T = @(x) [(1:size(x,1))' x];
Train.idx = train_idx;
Train.dat = img(:, train_idx);
Train.lab = img_gt(train_idx)';
Test.idx = test_idx;
Test.dat = img(:, test_idx);
Test.lab = img_gt(test_idx)';
sigs_size = length(sigs); cs_size = length(Cs); mus_size = length(mus);
Ytrain = Train.lab';
Ytest=Test.lab';
k = 1;
sig_c = zeros(sigs_size*cs_size*mus_size,4);
for mu = mus,
    for ss = sigs,
        [Kgtrain Kgtest] = composite_kernel(img, rows, cols, Train.idx,Test.idx, wind, ss, mu, 10000,1);
        xapp=[Ytrain Kgtrain'];xtest=[Test.lab' Kgtest'];
        for cc = Cs,
            [TTrain,TTest,TrainAC,accur_ELM,TY,pred,actual] = elm_kernel(xapp,xtest,1,cc,'RBF_kernel',kerneloption);
            ac = class_eval(pred,actual);
            sig_c(k,:) = [ss cc mu ac.OA];
            k = k + 1;
        end
    end
end
[~, k] = max(sig_c(:,4));
sig = sig_c(k, 1);
C = sig_c(k, 2);
mu = sig_c(k, 3);
end
