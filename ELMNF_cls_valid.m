n=2;
addpath([cd '\privates\']);
[img, img_gt, rows, cols] = load_data(n);
L = size(img, 1);
gams = exp(-3:12) ./ L;
sigs = sqrt(0.5 ./ gams);
nt = 5; its =3; wins = 19; sgs = length(sigs);
mu = 1e-3; lam = 1e-4;
T = @(x) [(1:size(x,1))' x];

ELM_oa= zeros(wins, sgs, its);ELM_ka = zeros(wins, sgs, its);
xapp=[];xtest=[];acc=zeros(wins, sgs, its);

kerneloption = [100];
c = 2^50;
switch n,
	case 1,
	sigs1 = sigs([10    14    13    13    13    13    13    13    13]);
	sigs2 = sigs([14    14    14    14    13    13    13    13    13]);
	case 2,
    sigs1 = sigs([10    14    13    13    13    13    13    13    13]);
	sigs2 = sigs([14    14    14    14    13    13    13    13    13]);
	case 3,
	sigs1 = sigs([14    14    13    14    14    13    13    14    14  14]);
	sigs2 = sigs([11    12    11    11    12    12    12    12    12]);
% 	load results\NF\ksc.mat;
end
for it = 3 : its,
	[train_idx, test_idx] = load_train_test(n, 1, nt, it);
	[Train, Test] = set_train_test(train_idx, test_idx, img, img_gt);
    
	Ytrain = Train.lab';
	for wind = 9,
    %for wind = 1 :2: wins,
        s = 0; 
		for sig0 = sigs(8),
        %for sig0 = sigs,
            disp('===========================================================');
            disp([it wind s]);
            s = s + 1;
            sig1 = sigs1((wind+1)/2);
			[Ktrain, Ktest] = ker_lwm(img, rows, cols, Train.idx, Test.idx, wind, sig1, sig0, [100 80], 1);
		    xapp=[Ytrain Ktrain'];xtest=[Test.lab' Ktest'];
            [TTrain,TTest,TrainAC,accur_ELM,TY,pred,actual] = elm_kernel(xapp,xtest,1,c,'RBF_kernel',kerneloption);
            disp(accur_ELM)
            acc(wins, s, it)=accur_ELM;
            ac = class_eval(pred,actual);
			ELM_oa(wind, s, it) = ac.OA; ELM_ka(wind, s, it) = ac.Kappa;
            disp(ac.OA)
			
%             sig2 = sigs2(wind);
%             if sig2 ~= sig1,
%                 [Ktrain, Ktest] = ker_lwm(img, rows, cols, Train.idx, Test.idx, wind, sig2, sig0, [100 80], 1);
%             end
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
% 				case 1, save results\ELMNF\indian_new_40_20171025.mat ksrc_oa ksrc_ka svm_oa svm_ka;
% 				case 2, save results\ELMNF\pavia.mat ksrc_oa ksrc_ka svm_oa svm_ka;
% 				case 3, save results\ELMNF\ksc_20.mat ELM_oa ELM_ka;
% 			end
		end
	end
end
