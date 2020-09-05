function [sig,C,d] = elmtrain(img, img_gt, rows, cols, Train, sigs,Cs, wind, n, nt, its, ds)
[train_idx, test_idx] = load_train_test(n, 1, nt, its+1);
[Test, ~] = set_train_test(train_idx, test_idx, img, img_gt);
kerneloption = [1];
c = 1024;
sigs_size = length(sigs); cs_size = length(Cs); ds_size = length(ds);
sig_mu_lam = zeros(sigs_size*cs_size*ds_size,4);
k = 1;
Ytrain = Train.lab';
for cc = Cs,
    for ss = sigs,
        for d = ds,
            [Ktrain, Ktest] = composite_kernel(img, rows, cols, Train.idx, Test.idx, wind, ss, cc, 10000, d);
            xapp=[Ytrain Ktrain'];xtest=[Test.lab' Ktest'];

            [TTrain,TTest,TrainAC,accur_ELM,TY,pred,actual] = elm_kernel(xapp,xtest,1,c,'RBF_kernel',kerneloption);
			ac = class_eval(pred,actual);
% 			ELM_oa(wind, s, it) = ac.OA; ELM_ka(wind, s, it) = ac.Kappa;
%             disp(ac.OA)
%             AtX = Ktest; AtA = Ktrain;
%             p.mu = mu; p.lam = lam;
%             S = SpRegKL1(AtX, AtA, p);
%             pred = class_ker_pred(AtX, AtA, S, Test.lab, Train.lab);
%             test = Test.lab;
%             ac = class_eval(pred, test);
            
            sig_mu_lam(k,:) = [ss cc d ac.OA];
            k = k + 1;
        end
    end
end
[~, k] = max(sig_mu_lam(:,4));
sig = sig_mu_lam(k, 1);
C = sig_mu_lam(k, 2);
d = sig_mu_lam(k, 3);
end

