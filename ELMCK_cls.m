clc;
clear all
n=2;
t=cputime;
t2=clock;
addpath([cd '\privates\']);
[img, img_gt, rows, cols] = load_data(n);
L = size(img, 1);
gams = exp(-3:12) ./ L;
sigs = sqrt(0.5 ./ gams);
Cs = 10.^(0:3); nus = 0.1 : 0.1 : 1.0;
d=1;
its = 1; wins = 11; sgs = length(sigs);
switch n
	case 1, nt = 5;
	case 2, nt = 5;
	case 3, nt = 3;
end

kerneloption = [1];
c = 1024;

mu = 1e-3; lam = 1e-4;
T = @(x) [(1:size(x,1))' x];
ELM_oa = zeros((wins+1)/2, d,its); ELM_ka = zeros((wins+1)/2,d, its); ksrc_sig = zeros(wins, its); ksrc_nu = zeros(wins, its);
svm_oa = zeros(wins, its); svm_ka = zeros(wins, its); svm_sig = zeros(wins, its); svm_nu = zeros(wins, its);
for it = 1 : its,
    [train_idx, test_idx] = load_train_test(n, 1, nt, it);
    [Train, Test] = set_train_test(train_idx, test_idx, img, img_gt);
    Ytrain = Train.lab';
    for wind = 5,
        for d=1,
        disp('===========================================================');
        disp([it wind]);
        %[sig, nu] = ksrctrain(img, img_gt, rows, cols, Train, sigs, nus, mu, lam, wind, n, nt, its);
%         sig=sigs((wins+1)/2);
%         nu=nus((wins+1)/2);
        sig=sigs(3);
        nu=nus(3);
        [Ktrain, Ktest] = composite_kernel(img, rows, cols, Train.idx, Test.idx, wind,  sig, nu, 80,d);
%         AtX = Ktest; AtA = Ktrain;
%         p.mu = mu; p.lam = lam;
%         S = SpRegKL1(AtX, AtA, p);
%         pred = class_ker_pred(AtX, AtA, S, Test.lab, Train.lab);
%         test = Test.lab;
%         ac = class_eval(pred, test);
%         ksrc_oa(wind, it) = ac.OA; ksrc_ka(wind, it) = ac.Kappa;
% 		ksrc_sig(wind, it) = sig; ksrc_nu(wind, it) = nu;
        xapp=[Ytrain Ktrain'];xtest=[Test.lab' Ktest'];

            
            [TTrain,TTest,TrainAC,accur_ELM,TY,pred,actual] = elm_kernel(xapp,xtest,1,c,'RBF_kernel',kerneloption);
            disp(accur_ELM)
            %acc(wins,d, it)=accur_ELM;
            ac = class_eval(pred,actual);
			ELM_oa(wind, d,it) = ac.OA; ELM_ka(wind,d, it) = ac.Kappa;
            disp(ac.OA)

%         [sig, C, nu] = svm_train(img, img_gt, train_idx, sigs, Cs, nus, wind, rows, cols);
%         [Ktrain, Ktest] = composite_kernel(img, rows, cols, Train.idx, Test.idx, wind, sig, nu, 10000);
%         model  = svmtrain(Ytrain, T(Ktrain), ['-q -t 4 -c ' num2str(C)]);
%         Ytest = Test.lab';
%         Ypred = svmpredict(Ytest, T(Ktest'), model);
%         ACC =  assessment(Ytest, Ypred, 'class');
%         svm_oa(wind, it) = ACC.OA; svm_ka(wind, it) = ACC.Kappa;
% 		svm_sig(wind, it) = sig; svm_nu(wind, it) = nu;
        
%         switch n
%             case 1, save results\ELMCK\indian_40.mat ELM_oa ELM_ka;
%             case 2, save results\ELMCK\pavia_40.mat ELM_oa ELM_ka;
%             case 3, save results\CK\ksc.mat ksrc_oa ksrc_ka ksrc_sig ksrc_nu svm_oa svm_ka svm_sig svm_nu;
%         end
        end
    end
end
etime(clock,t2)
t1=cputime-t

% function [sig, C, mu] = svm_train(img, img_gt, train_idx, sigs, Cs, mus, wind, rows, cols)
% T = @(x) [(1:size(x,1))' x];
% Train.idx = train_idx;
% Train.dat = img(:, train_idx);
% Train.lab = img_gt(train_idx)';
% sigs_size = length(sigs); cs_size = length(Cs); mus_size = length(mus);
% Ytrain = Train.lab';
% k = 1;
% sig_c = zeros(sigs_size*cs_size*mus_size,4);
% for mu = mus,
%     for ss = sigs,
%         Kgtrain = composite_kernel(img, rows, cols, Train.idx, [], wind, ss, mu, 10000);
%         for cc = Cs,
%             model = svmtrain(Ytrain, T(Kgtrain), ['-q -t 4 -v 5 -c ' num2str(cc)]);
%             sig_c(k,:) = [ss cc mu model];
%             k = k + 1;
%         end
%     end
% end
% [~, k] = max(sig_c(:,4));
% sig = sig_c(k, 1);
% C = sig_c(k, 2);
% mu = sig_c(k, 3);
% end
% 
% function [sig, C] = ksrctrain(img, img_gt, rows, cols, Train, sigs, Cs, mu, lam, wind, n, nt, its)
% [train_idx, test_idx] = load_train_test(n, 1, nt, its+1);
% [Test, ~] = set_train_test(train_idx, test_idx, img, img_gt);
% 
% sigs_size = length(sigs); cs_size = length(Cs);
% sig_mu_lam = zeros(sigs_size*cs_size,3);
% k = 1;
% for cc = Cs,
%     for ss = sigs,
%         [Ktrain, Ktest] = composite_kernel(img, rows, cols, Train.idx, Test.idx, wind, ss, cc, 10000);
%         AtX = Ktest; AtA = Ktrain;
%         p.mu = mu; p.lam = lam;
%         S = SpRegKL1(AtX, AtA, p);
%         pred = class_ker_pred(AtX, AtA, S, Test.lab, Train.lab);
%         test = Test.lab;
%         ac = class_eval(pred, test);
%         sig_mu_lam(k,:) = [ss cc ac.OA];
%         k = k + 1;
%     end
% end
% [~, k] = max(sig_mu_lam(:,3));
% sig = sig_mu_lam(k, 1);
% C = sig_mu_lam(k, 2);
% end