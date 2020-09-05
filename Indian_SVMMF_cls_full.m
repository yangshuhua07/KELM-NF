clc
n=1;
addpath([cd '\privates\']);
load Indian_remove_index.mat;
[img, img_gt, rows, cols] = load_data(n);
L = size(img, 1);

switch n
    case 1,
        nts =5; its = 3;
    case 2,
        nts = 5; its = 2;
    case 3,
        nts = 3; its = 3;
end

T = @(x) [(1:size(x,1))' x];%@º¯Êý¾ä±ú

SVMMF_pred = cell(nts, its); SVMMF_acc = cell(nts, its); SVMMF_para = cell(nts,its);
time=[];
for nt = 5 : nts,
    for it = 3: its,
        disp(['======' num2str(nt) ',' num2str(it) '========']);
        [train_idx, test_idx] = load_train_test(n, 1, nt, it);
        [Train, Test] = set_train_test(train_idx, test_idx, img, img_gt);    
        
        index=find(img_gt==0);
        index=[index;remove_index];
        background=img(:,index);
        back_test=[zeros(length(index),1) background'];
        Ytrain = Train.lab';
        new_index=cat(2,Test.idx,index');
        
        switch n % parameters
            case 1,
                gams = exp(-3:12) ./ L; Cs = 10.^(0:3); wind =5;
                %gams = exp(3) ./ L; Cs = 10.^2; wind =7;
            case 2,
                gams = exp(-3:12) ./ L; Cs = 10.^(0:3); wind = 13;
            case 3,
                gams = exp(-3:12) ./ L; Cs = 10.^(0:3); wind = 11;
        end

        sigs = sqrt(0.5 ./ gams);
        
        %[sig, C] = svm_train(img, img_gt, train_idx, rows, cols, sigs, Cs, wind);
        %gam = 0.5 / sig^2;
        gam=299.3707;
        sig=sqrt(0.5 ./ gam);
        C=1000;
        
        %Ytrain = Train.lab';
        [Ktrain, Ktest] = ker_mm(img, rows, cols, Train.idx, new_index, wind, sig, [80 80], 1);
        model  = svmtrain(Ytrain, T(Ktrain), ['-q -t 4 -c ' num2str(C)]);
        %Ytest = Test.lab';
        lab=cat(2,Test.lab,zeros(1,length(index)));
        Ytest=lab';
        pred = svmpredict(Ytest, T(Ktest'), model);
        acc =  assessment(Ytest, pred, 'class');
        %ac=class_eval(pred,Ytest);
        disp(acc.OA);
        ac = class_eval(pred(1:length(Test.idx))',Ytest(1:length(Test.idx))');
        disp(ac.OA);
%         SVMMF_acc{nt, it} = acc;
%         SVMMF_pred{nt, it} = pred;
%         SVMMF_para{nt, it} = [gam C];
        %time=[time t];
        
%         switch n
%             case 1, save results\SVMMF\SVMMF_indian.mat SVMMF_acc SVMMF_pred SVMMF_para;
%             case 2, save results\SVMMF\pavia\SVMMF_pavia_40.mat SVMMF_acc SVMMF_pred SVMMF_para;
%             case 3, save results\SVMMF\ksc\SVMMF_ksc_10_11wind.mat SVMMF_acc SVMMF_pred SVMMF_para;
%         end
        
    end
end


% function [sig, C] = svm_train(img, img_gt, train_idx, rows, cols, sigs, Cs, wind)
% T = @(x) [(1:size(x,1))' x];
% Train.idx = train_idx;
% Train.dat = img(:, train_idx);
% Train.lab = img_gt(train_idx)';
% sigs_size = length(sigs); cs_size = length(Cs);
% Ytrain = Train.lab';
% k = 1;
% sig_c = zeros(sigs_size*cs_size,3);
% for ss = sigs,
%     [Kgtrain, ~] = ker_mm(img, rows, cols, Train.idx, [], wind, ss, [80 80], 1);
%     for cc = Cs,
%         model = svmtrain(Ytrain, T(Kgtrain), ['-q -t 4 -v 5 -c ' num2str(cc)]);
%         sig_c(k,:) = [ss cc model];
%         k = k + 1;
%     end
% end
% [~, k] = max(sig_c(:,3));
% sig = sig_c(k, 1);
% C = sig_c(k, 2);
% end