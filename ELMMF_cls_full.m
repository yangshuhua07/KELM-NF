%function ELMMF_cls(n)
clear all;
clc
n=1;
tic
addpath([cd '\privates\']);
load Indian_remove_index.mat;
[img, img_gt, rows, cols] = load_data(n);
L = size(img, 1);
gams = exp(-3:12) ./ L;
%gams = exp(7:11) ./ L;
sigs = sqrt(0.5 ./ gams);
its = 4; wins = 3; sgs = length(sigs);
switch n
	case 1, nt = 5;
	case 2, nt = 5;
	case 3, nt = 4;
end
mu = 1e-3; lam = 1e-4;
T = @(x) [(1:size(x,1))' x];
kerneloption = [1];
c = 2^10;
ELM_oa = zeros(wins, sgs, its); ELM_ka = zeros(wins, sgs, its);
xapp=[];xtest=[];acc=zeros(wins, sgs, its);
% svm_oa = zeros(wins, sgs, its); svm_ka = zeros(wins, sgs, its);
for it = 4 : its,
	[train_idx, test_idx] = load_train_test(n, 1, nt, it);
	[Train, Test] = set_train_test(train_idx, test_idx, img, img_gt);
    index=find(img_gt==0);
    index=[index;remove_index];
    background=img(:,index);
    back_test=[zeros(length(index),1) background'];
	Ytrain = Train.lab';
    new_index=cat(2,Test.idx,index');
	for wind =3,
        s = 0;
		for sig = sigs(14),
            disp('===========================================================');
            disp([it wind s]);
            s = s + 1;
            %tic
			[Ktrain, Ktest] = ker_mm(img, rows, cols, Train.idx, new_index, wind, sig, [80 80], 1);		
            %t1=toc;
            lab=cat(2,Test.lab,zeros(1,length(index)));
            xapp=[Ytrain Ktrain'];xtest=[lab' Ktest'];
%			AtX = Ktest; AtA = Ktrain;
% 			p.mu = mu; p.lam = lam;
% 			S = SpRegKL1(AtX, AtA, p);
% 			pred = class_ker_pred(AtX, AtA, S, Test.lab, Train.lab);
            %tic
            [TTrain,TTest,TrainAC,accur_ELM,TY,pred,actual] = elm_kernel(xapp,xtest,1,c,'RBF_kernel',kerneloption);
            %[TTrain1,TTest1,TrainAC1,accur_ELM1,TY1,pred1,actual1] = elm_kernel(xapp,back_test,1,c,'RBF_kernel',kerneloption);
            
            disp(accur_ELM)
            acc(wins, s, it)=accur_ELM;
            %t2=toc;
			
			ac = class_eval(pred(1:length(Test.idx)),actual(1:length(Test.idx)));
			ELM_oa(wind, s, it) = ac.OA; ELM_ka(wind, s, it) = ac.Kappa;
            disp(ac.OA)
			
%             mds = zeros(3,1); Cs = 10.^(0:3); i = 1;
%             for C = Cs,
%                 model  = svmtrain(Ytrain, T(Ktrain), ['-q -t 4 -v 5 -c ' num2str(C)]);
%                 mds(i) = model;
%                 i = i + 1;
%             end
%             [~, i] = max(mds);
%             C = mds(i);
%             model  = svmtrain(Ytrain, T(Ktrain), ['-q -t 4 -c ' num2str(C)]);
% 			Ytest = Test.lab';
% 			Ypred = svmpredict(Ytest, T(Ktest'), model);
% 			ACC =  assessment(Ytest, Ypred, 'class');
% 			svm_oa(wind, s, it) = ACC.OA; svm_ka(wind, s, it) = ACC.Kappa;
			
% 			switch n
% 				case 1, save results\ELMMF\indian_80.mat ELM_oa ELM_ka ;
%                         save results\ELMMF\indian_MF_acc_80.mat acc;
% 				case 2, save results\ELMMF\pavia\pavia_3.mat ELM_oa ELM_ka ;
% 				case 3, save results\ELMMF\ksc\ksc_20.mat ELM_oa ELM_ka ;
% 			end
		end
	end
end
t2=toc;
%end