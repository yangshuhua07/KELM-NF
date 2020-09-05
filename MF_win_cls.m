function MF_win_cls(n)
addpath([cd '\privates\']);
[img, img_gt, rows, cols] = load_data(n);
L = size(img, 1);
gams = exp(-3:12) ./ L;
sigs = sqrt(0.5 ./ gams);
Cs = 10.^(0:3);
nt = 5; its = 5; wins = 9; sgs = length(sigs);
mu = 1e-3; lam = 1e-4;
T = @(x) [(1:size(x,1))' x];
ksrc_oa = zeros(wins, its); ksrc_ka = zeros(wins, its); ksrc_sig = zeros(wins, its);
svm_oa = zeros(wins, its); svm_ka = zeros(wins, its); svm_sig = zeros(wins, its);
for it = 1 : its,
    [train_idx, test_idx] = load_train_test(n, 1, nt, it);
    [Train, Test] = set_train_test(train_idx, test_idx, img, img_gt);
    Ytrain = Train.lab';
    for wind = 1 : wins,
        disp('===========================================================');
        disp([it wind]);
        [svmsig, ksrcsig, C] = trainning(img, img_gt, Train, rows, cols, sigs, Cs, mu, lam, n, nt, its, wind);
        %sig = ksrctrain(img, img_gt, rows, cols, Train, sigs, mu, lam, n, nt, its, wind);
        sig = ksrcsig;
        [Ktrain, Ktest] = ker_mm(img, rows, cols, Train.idx, Test.idx, wind, sig, [80 80], 1);
        AtX = Ktest; AtA = Ktrain;
        p.mu = mu; p.lam = lam;
        S = SpRegKL1(AtX, AtA, p);
        pred = class_ker_pred(AtX, AtA, S, Test.lab, Train.lab);
        test = Test.lab;
        ac = class_eval(pred, test);
        ksrc_oa(wind, it) = ac.OA; ksrc_ka(wind, it) = ac.Kappa; ksrc_sig(wind, it) = sig;
        
        %[sig, C] = svm_train(img, img_gt, train_idx, rows, cols, sigs, Cs, wind);
        sig = svmsig;
        [Ktrain, Ktest] = ker_mm(img, rows, cols, Train.idx, Test.idx, wind, sig, [80 80], 1);
        model  = svmtrain(Ytrain, T(Ktrain), ['-q -t 4 -c ' num2str(C)]);
        Ytest = Test.lab';
        Ypred = svmpredict(Ytest, T(Ktest'), model);
        ACC =  assessment(Ytest, Ypred, 'class');
        svm_oa(wind, it) = ACC.OA; svm_ka(wind, it) = ACC.Kappa; svm_sig(wind, it) = sig;
        
        switch n
            case 1, save results\MF\indian.mat ksrc_oa ksrc_ka ksrc_sig svm_oa svm_ka svm_sig;
            case 2, save results\MF\pavia.mat ksrc_oa ksrc_ka ksrc_sig svm_oa svm_ka svm_sig;
            case 3, save results\MF\ksc.mat ksrc_oa ksrc_ka ksrc_sig svm_oa svm_ka svm_sig;
        end
    end
end
end

function [sig, C] = svm_train(img, img_gt, train_idx, rows, cols, sigs, Cs, wind)
T = @(x) [(1:size(x,1))' x];
Train.idx = train_idx;
Train.dat = img(:, train_idx);
Train.lab = img_gt(train_idx)';
sigs_size = length(sigs); cs_size = length(Cs);
Ytrain = Train.lab';
k = 1;
sig_c = zeros(sigs_size*cs_size,3);
for ss = sigs,
    [Kgtrain, ~] = ker_mm(img, rows, cols, Train.idx, [], wind, ss, [80 80], 1);
    for cc = Cs,
        model = svmtrain(Ytrain, T(Kgtrain), ['-q -t 4 -v 5 -c ' num2str(cc)]);
        sig_c(k,:) = [ss cc model];
        k = k + 1;
    end
end
[~, k] = max(sig_c(:,3));
sig = sig_c(k, 1);
C = sig_c(k, 2);
end

function sig = ksrctrain(img, img_gt, rows, cols, Train, sigs, mu, lam, n, nt, its, wind)
[train_idx, test_idx] = load_train_test(n, 1, nt, its+1);
[Test, ~] = set_train_test(train_idx, test_idx, img, img_gt);

sigs_size = length(sigs);
sig_mu_lam = zeros(sigs_size,2);
k = 1;
for ss = sigs,
    [Ktrain, Ktest] = ker_mm(img, rows, cols, Train.idx, Test.idx, wind, ss, [80 80], 1);
    AtX = Ktest; AtA = Ktrain;
    p.mu = mu; p.lam = lam;
    S = SpRegKL1(AtX, AtA, p);
    pred = class_ker_pred(AtX, AtA, S, Test.lab, Train.lab);
    test = Test.lab;
    ac = class_eval(pred, test);
    sig_mu_lam(k,:) = [ss ac.OA];
    k = k + 1;
end
[~, k] = max(sig_mu_lam(:,2));
sig = sig_mu_lam(k, 1);
end

function [svmsig, ksrcsig, C] = trainning(img, img_gt, Train, rows, cols, sigs, Cs, mu, lam, n, nt, its, wind)
[train_idx, test_idx] = load_train_test(n, 1, nt, its+1);
[Test, ~] = set_train_test(train_idx, test_idx, img, img_gt);
T = @(x) [(1:size(x,1))' x];
sigs_size = length(sigs); cs_size = length(Cs);
Ytrain = Train.lab';
sig_c = zeros(sigs_size*cs_size,3); sig_mu_lam = zeros(sigs_size,2);
for i = 1 : sigs_size,
    ss = sigs(i);
    [Ktrain, Ktest] = ker_mm(img, rows, cols, Train.idx, Test.idx, wind, ss, [80 80], 1);
    AtX = Ktest; AtA = Ktrain;
    p.mu = mu; p.lam = lam;
    S = SpRegKL1(AtX, AtA, p);
    pred = class_ker_pred(AtX, AtA, S, Test.lab, Train.lab);
    test = Test.lab;
    ac = class_eval(pred, test);
    sig_mu_lam(i,:) = [ss ac.OA];
    for j = 1 : cs_size,
        cc = Cs(j);
        model = svmtrain(Ytrain, T(Ktrain), ['-q -t 4 -v 5 -c ' num2str(cc)]);
        sig_c((i-1)*cs_size+j,:) = [ss cc model];
    end
end
[~, k] = max(sig_c(:,3));
svmsig = sig_c(k, 1);
C = sig_c(k, 2);
[~, k] = max(sig_mu_lam(:,2));
ksrcsig = sig_mu_lam(k, 1);
end