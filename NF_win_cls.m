function NF_win_cls(n)
addpath([cd '\privates\']);
[img, img_gt, rows, cols] = load_data(n);
L = size(img, 1);
%gams = exp(-3:12) ./ L; 
gams = exp(-3:5) ./ L;
sig0s = sqrt(0.5 ./ gams); Cs = 10.^(0:3);
nt = 5; its = 5; wins = 9;
mu = 1e-3; lam = 1e-4;
T = @(x) [(1:size(x,1))' x];
load results\MF\pavia.mat;
sigs1 = ksrc_sig;
sigs2 = svm_sig;
ksrc_oa = zeros(wins, its); ksrc_ka = zeros(wins, its); ksrc_sig = zeros(wins, its);
svm_oa = zeros(wins, its); svm_ka = zeros(wins, its); svm_sig = zeros(wins, its);
for it = 1 : its,
    if it == 5, break; end
	[train_idx, test_idx] = load_train_test(n, 1, nt, it);
	[Train, Test] = set_train_test(train_idx, test_idx, img, img_gt);
	Ytrain = Train.lab';
	for wind = 1 : wins,
		disp('===========================================================');
		disp([it wind]);
		sig1 = sigs1(wind, it);
        sig0 = ksrctrain(img, img_gt, rows, cols, Train, sig1, sig0s, mu, lam, n, nt, its, wind);
		[Ktrain, Ktest] = ker_lwm(img, rows, cols, Train.idx, Test.idx, wind, sig1, sig0, [80 80], 1);
		AtX = Ktest; AtA = Ktrain;
		p.mu = mu; p.lam = lam;
		S = SpRegKL1(AtX, AtA, p);
		pred = class_ker_pred(AtX, AtA, S, Test.lab, Train.lab);
		test = Test.lab;
		ac = class_eval(pred, test);
		ksrc_oa(wind, it) = ac.OA; ksrc_ka(wind, it) = ac.Kappa; ksrc_sig(wind, it) = sig0;
		sig2 = sigs2(wind, it);
        [sig0, C] = svm_train(img, img_gt, train_idx, rows, cols, sig2, sig0s, Cs, wind);
        [Ktrain, Ktest] = ker_lwm(img, rows, cols, Train.idx, Test.idx, wind, sig2, sig0, [80 80], 1);
        model  = svmtrain(Ytrain, T(Ktrain), ['-q -t 4 -c ' num2str(C)]);
		Ytest = Test.lab';
		Ypred = svmpredict(Ytest, T(Ktest'), model);
		ACC =  assessment(Ytest, Ypred, 'class');
		svm_oa(wind, it) = ACC.OA; svm_ka(wind, it) = ACC.Kappa; svm_sig(wind, it) = sig0;
		
		switch n
			case 1, save results\NF\indian.mat ksrc_oa ksrc_ka ksrc_sig svm_oa svm_ka svm_sig;
			case 2, save results\NF\pavia.mat ksrc_oa ksrc_ka ksrc_sig svm_oa svm_ka svm_sig;
			case 3, save results\NF\ksc.mat ksrc_oa ksrc_ka ksrc_sig svm_oa svm_ka svm_sig;
		end
	end
end
end

function [sig0, C] = svm_train(img, img_gt, train_idx, rows, cols, sig, sig0s, Cs, wind)
T = @(x) [(1:size(x,1))' x];
Train.idx = train_idx;
Train.dat = img(:, train_idx);
Train.lab = img_gt(train_idx)';
sig0s_size = length(sig0s); cs_size = length(Cs);
Ytrain = Train.lab';
k = 1;
sig0_c = zeros(sig0s_size*cs_size,3);
for ss = sig0s,
    [Kgtrain, ~] = ker_lwm(img, rows, cols, Train.idx, [], wind, sig, ss, [80 80], 1);
    for cc = Cs,
        model = svmtrain(Ytrain, T(Kgtrain), ['-q -t 4 -v 5 -c ' num2str(cc)]);
        sig0_c(k,:) = [ss cc model];
        k = k + 1;
    end
end
[~, k] = max(sig0_c(:,3));
sig0 = sig0_c(k, 1);
C = sig0_c(k, 2);
end

function sig0 = ksrctrain(img, img_gt, rows, cols, Train, sig, sig0s, mu, lam, n, nt, its, wind)
[train_idx, test_idx] = load_train_test(n, 1, nt, its+1);
[Test, ~] = set_train_test(train_idx, test_idx, img, img_gt);

sig0s_size = length(sig0s);
sig0_mu_lam = zeros(sig0s_size,2);
k = 1;
for ss = sig0s,
    [Ktrain, Ktest] = ker_lwm(img, rows, cols, Train.idx, Test.idx, wind, sig, ss, [80 80], 1);
    AtX = Ktest; AtA = Ktrain;
    p.mu = mu; p.lam = lam;
    S = SpRegKL1(AtX, AtA, p);
    pred = class_ker_pred(AtX, AtA, S, Test.lab, Train.lab);
    test = Test.lab;
    ac = class_eval(pred, test);
    sig0_mu_lam(k,:) = [ss ac.OA];
    k = k + 1;
end
[~, k] = max(sig0_mu_lam(:,2));
sig0 = sig0_mu_lam(k, 1);
end