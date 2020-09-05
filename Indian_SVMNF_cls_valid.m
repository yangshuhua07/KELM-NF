clc
n=1;

addpath([cd '\privates\']);
t=cputime;
t2=clock;
[img, img_gt, rows, cols] = load_data(n);
L = size(img, 1);

switch n
    case 1,
        nts = 5; its = 3;
		load results\SVMMF\SVMMF_indian.mat;
    case 2,
        nts = 5; its = 2;
		load results\SVMMF\SVMMF_pavia.mat;
    case 3,
        nts = 3; its = 3;
		load results\SVMMF\ksc\SVMMF_ksc_10_11wind.mat;
end

T = @(x) [(1:size(x,1))' x];

SVMNF_pred = cell(nts, its); SVMNF_acc = cell(nts, its); SVMNF_para = cell(nts,its);
time1=[];
for nt = 5 : nts,
    for it = 3: its,
        disp(['======' num2str(nt) ',' num2str(it) '========']);
        [train_idx, test_idx] = load_train_test(n, 1, nt, it);
        [Train, Test] = set_train_test(train_idx, test_idx, img, img_gt);    
        Ytrain = Train.lab';
		tmp = SVMMF_para{nt, it};
        switch n % parameters
            case 1,
                gams = exp(-3:12) ./ L; Cs = 10.^(0:3); wind = 9; sig = sqrt(0.5 ./ tmp(1));
            case 2,
                gams = exp(-3:12) ./ L; Cs = 10.^(0:3); wind = 7; sig = sqrt(0.5 ./ tmp(1));
                %gams = exp(10) ./ L; Cs = 10.^2; wind = 7; sig = sqrt(0.5 ./ tmp(1));
            case 3,
                gams = exp(-3:12) ./ L; Cs = 10.^(0:3); wind = 11; sig = sqrt(0.5 ./ tmp(1));
        end
        sig0s = sqrt(0.5 ./ gams);
%         [sig0, C] = svm_train(img, img_gt, train_idx, rows, cols, sig, sig0s, Cs, wind);
%         gam = 0.5 / sig0^2;
        C=1000,
        gam=0.7421;
        sig0=sqrt(0.5 ./ gam);
        %gam = 0.5 / sig0^2;
        
%         Ytrain = Train.lab';
        [Ktrain, Ktest] = ker_lwm(img, rows, cols, Train.idx, Test.idx, wind, sig, sig0, [80 80], 1);
        model  = svmtrain(Ytrain, T(Ktrain), ['-q -t 4 -c ' num2str(C)]);
        Ytest = Test.lab';
        %lab=cat(2,Test.lab,zeros(1,length(index)));
        %Ytest=lab';
        pred = svmpredict(Ytest, T(Ktest'), model);
        acc =  assessment(Ytest, pred, 'class');
        disp(acc.OA);
%         ac = class_eval(pred(1:length(Test.idx))',Ytest(1:length(Test.idx))');
%         disp(ac.OA);
%         SVMNF_acc{nt, it} = acc;
%         SVMNF_pred{nt, it} = pred;
%         SVMNF_para{nt, it} = [gam C];
   
%         time1=[time1 t];
%         switch n
%             case 1, save results\SVMNF\SVMNF_indian.mat SVMNF_acc SVMNF_pred SVMNF_para;
%             case 2, save results\SVMNF\pavia\SVMNF_pavia_40.mat SVMNF_acc SVMNF_pred SVMNF_para;
%             case 3, save results\SVMNF\SVMNF_ksc_10_11wind.mat SVMNF_acc SVMNF_pred SVMNF_para;
%         end
        
    end
end
etime(clock,t2)
t1=cputime-t


% function [sig0, C] = svm_train(img, img_gt, train_idx, rows, cols, sig, sig0s, Cs, wind)
% T = @(x) [(1:size(x,1))' x];
% Train.idx = train_idx;
% Train.dat = img(:, train_idx);
% Train.lab = img_gt(train_idx)';
% sig0s_size = length(sig0s); cs_size = length(Cs);
% Ytrain = Train.lab';
% k = 1;
% sig0_c = zeros(sig0s_size*cs_size,3);
% for ss = sig0s,
%     [Kgtrain, ~] = ker_lwm(img, rows, cols, Train.idx, [], wind, sig, ss, [80 80], 1);
%     for cc = Cs,
%         model = svmtrain(Ytrain, T(Kgtrain), ['-q -t 4 -v 5 -c ' num2str(cc)]);
%         sig0_c(k,:) = [ss cc model];
%         k = k + 1;
%     end
% end
% [~, k] = max(sig0_c(:,3));
% sig0 = sig0_c(k, 1);
% C = sig0_c(k, 2);
% end