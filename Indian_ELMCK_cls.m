clear all; close all; clc
n=1;

addpath([cd '\privates\']);

[img, img_gt, rows, cols] = load_data(n);
L = size(img, 1);

switch n
    case 1,
        nts = 5; its = 9;
    case 2,
        nts = 8; its = 5;
    case 3,
        nts = 5; its = 5;
end
wins=13;
kerneloption = [1];
c = 1024;
%ELMCK_pred = zeros(wins, its); 
ELMCK_acc = zeros(wins, its); ELMCK_para = cell(wins,its);
for nt = 5 : nts,
    for it = 1 : its,
        for wind=3:2:wins;
        disp(wind);
        [train_idx, test_idx] = load_train_test(n, 1, nt, it);
        [Train, Test] = set_train_test(train_idx, test_idx, img, img_gt);    
        Ytrain = Train.lab';
        switch n % parameters
            case 1,
                mu = 1e-3; lam = 1e-4; Cs = 0.1 : 0.1 : 0.9;  ds = 1 : 10; sigs = 10.^(-1:3);
            case 2,
                mu = 1e-3; lam = 1e-4; Cs = 0.1 : 0.1 : 0.9; wind = 2; ds = 1 : 10; sigs = 10.^(-1:3);
            case 3,
                mu = 1e-3; lam = 1e-4; Cs = 0.1 : 0.1 : 0.9; wind = 4; ds = 1 : 10; sigs = 10.^(-1:3);
        end

        [sig,C,d] = elmtrain(img, img_gt, rows, cols, Train, sigs, Cs,wind, n, nt, its, ds);
        gam = 0.5 / sig^2;
        
%         p.mu = mu; p.lam = lam;     
        [Ktrain, Ktest] = composite_kernel(img, rows, cols, Train.idx, Test.idx, wind,  sig, C, 10000, d);
%         AtA = Ktrain; AtX = Ktest;
%         S = SpRegKL1(AtX, AtA, p);
%         pred = class_ker_pred(AtX, AtA, S, Test.lab, Train.lab);
%         test = Test.lab;
%         acc = class_eval(pred, test);
        xapp=[Ytrain Ktrain'];xtest=[Test.lab' Ktest'];

        [TTrain,TTest,TrainAC,accur_ELM,TY,pred,actual] = elm_kernel(xapp,xtest,1,c,'RBF_kernel',kerneloption);
        disp(accur_ELM);
        ac = class_eval(pred,actual);
        disp(ac.OA);
        ELMCK_acc(wind, it) = ac.OA;
        %ELMCK_pred{nt, it} = pred;
        ELMCK_para{wind, it} = [sig C d];
        
%         switch n
%             case 1, save results\ELMCK\ELMCK_indian_20171031_new_wind3-13.mat ELMCK_acc ELMCK_para;
%             case 2, save results\KSRCCK\KSRCCK_pavia.mat KSRCCK_acc KSRCCK_pred KSRCCK_para;
%             case 3, save results\KSRCCK\KSRCCK_ksc.mat KSRCCK_acc KSRCCK_pred KSRCCK_para;
%         end
        
    end
end
end

% function [sig, C, d] = ksrctrain(img, img_gt, rows, cols, Train, sigs, Cs, mu, lam, wind, n, nt, its, ds)
% [train_idx, test_idx] = load_train_test(n, 1, nt, its+1);
% [Test, ~] = set_train_test(train_idx, test_idx, img, img_gt);
% 
% sigs_size = length(sigs); cs_size = length(Cs); ds_size = length(ds);
% sig_mu_lam = zeros(sigs_size*cs_size*ds_size,4);
% k = 1;
% for cc = Cs,
%     for ss = sigs,
%         for d = ds,
%             [Ktrain, Ktest] = composite_kernel(img, rows, cols, Train.idx, Test.idx, wind, ss, cc, 10000, d);
%             AtX = Ktest; AtA = Ktrain;
%             p.mu = mu; p.lam = lam;
%             S = SpRegKL1(AtX, AtA, p);
%             pred = class_ker_pred(AtX, AtA, S, Test.lab, Train.lab);
%             test = Test.lab;
%             ac = class_eval(pred, test);
%             sig_mu_lam(k,:) = [ss cc d ac.OA];
%             k = k + 1;
%         end
%     end
% end
% [~, k] = max(sig_mu_lam(:,4));
% sig = sig_mu_lam(k, 1);
% C = sig_mu_lam(k, 2);
% d = sig_mu_lam(k, 3);
% end