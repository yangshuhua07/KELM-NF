function MF_cls(n)
addpath([cd '\privates\']);
[img, img_gt, rows, cols] = load_data(n);
L = size(img, 1);
gams = exp(-3:12) ./ L;
sigs = sqrt(0.5 ./ gams);
its = 5; wins = 9; sgs = length(sigs);
switch n
	case 1, nt = 5;
	case 2, nt = 5;
	case 3, nt = 3;
end
mu = 1e-3; lam = 1e-4;
T = @(x) [(1:size(x,1))' x];
ksrc_oa = zeros(wins, sgs, its); ksrc_ka = zeros(wins, sgs, its);
svm_oa = zeros(wins, sgs, its); svm_ka = zeros(wins, sgs, its);
for it = 1 : its,
	[train_idx, test_idx] = load_train_test(n, 1, nt, it);
	[Train, Test] = set_train_test(train_idx, test_idx, img, img_gt);
	Ytrain = Train.lab';
	for wind = 1 : wins,
        s = 0;
		for sig = sigs,
            disp('===========================================================');
            disp([it wind s]);
            s = s + 1;
			[Ktrain, Ktest] = ker_mm(img, rows, cols, Train.idx, Test.idx, wind, sig, [80 80], 1);
			
			AtX = Ktest; AtA = Ktrain;
			p.mu = mu; p.lam = lam;
			S = SpRegKL1(AtX, AtA, p);
			pred = class_ker_pred(AtX, AtA, S, Test.lab, Train.lab);
			test = Test.lab;
			ac = class_eval(pred, test);
			ksrc_oa(wind, s, it) = ac.OA; ksrc_ka(wind, s, it) = ac.Kappa;
			
            mds = zeros(3,1); Cs = 10.^(0:3); i = 1;
            for C = Cs,
                model  = svmtrain(Ytrain, T(Ktrain), ['-q -t 4 -v 5 -c ' num2str(C)]);
                mds(i) = model;
                i = i + 1;
            end
            [~, i] = max(mds);
            C = mds(i);
            model  = svmtrain(Ytrain, T(Ktrain), ['-q -t 4 -c ' num2str(C)]);
			Ytest = Test.lab';
			Ypred = svmpredict(Ytest, T(Ktest'), model);
			ACC =  assessment(Ytest, Ypred, 'class');
			svm_oa(wind, s, it) = ACC.OA; svm_ka(wind, s, it) = ACC.Kappa;
			
			switch n
				case 1, save results\MF\indian.mat ksrc_oa ksrc_ka svm_oa svm_ka;
				case 2, save results\MF\pavia.mat ksrc_oa ksrc_ka svm_oa svm_ka;
				case 3, save results\MF\ksc.mat ksrc_oa ksrc_ka svm_oa svm_ka;
			end
		end
	end
end
end