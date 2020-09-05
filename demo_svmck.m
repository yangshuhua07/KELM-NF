function demo_svmck(n)
addpath([cd '\privates\']);
[img, img_gt, rows, cols] = load_data(n);
switch n,
    case 1,
        load results\SVMCK\SVMCK_indian.mat;
        nt = 5; its = 9;
    case 2,
        load results\SVMCK\SVMCK_pavia.mat;
        nt = 5; its = 5;
    case 3,
        load results\SVMCK\SVMCK_ksc.mat;
        nt = 3; its = 5;
end
tic
it = search(SVMCK_acc, nt, its);
tmp = SVMCK_para{nt,it}; gam = tmp(1); C = tmp(2); nu = tmp(3); d = tmp(4);
test_pred = SVMCK_pred{nt,it};
tmp = SVMCK_acc{nt,it};
acc.OA = tmp.OA; acc.Kappa = tmp.Kappa; acc.ratio = diag(tmp.ConfusionMatrix)' ./ sum(tmp.ConfusionMatrix);

[train_idx, test_idx] = load_train_test(n, 1, nt, it);
[Train, Test] = set_train_test(train_idx, test_idx, img, img_gt);
Back = get_background(train_idx, test_idx, img);

T = @(x) [(1:size(x,1))' x];

sig = sqrt(0.5 ./ gam); wind = 5;
Ytrain = Train.lab';
[Ktrain, Ktest] = composite_kernel(img, rows, cols, Train.idx, Back.idx, wind, sig, nu, 10000, d);
model  = svmtrain(Ytrain, T(Ktrain), ['-q -t 4 -c ' num2str(C)]);
Ytest = Back.lab';
back_pred = svmpredict(Ytest, T(Ktest'), model);

pred = zeros(1, rows*cols);
pred(Train.idx) = Train.lab; pred(Test.idx) = test_pred; pred(Back.idx) = back_pred;
t=toc
% switch n
%     case 1, save results\MAP\SVMCK_indian.mat acc pred;
%     case 2, save results\MAP\SVMCK_pavia.mat acc pred;
%     case 3, save results\MAP\SVMCK_ksc.mat acc pred;
% end
end

function it = search(acc, nt, its)
oa = zeros(its, 1);
for it = 1 : its,
    oa(it)  = acc{nt, it}.OA;
end
[~, b] = sort(oa);
it = b(floor(its/2) + 1);
end

function Back = get_background(train_idx, test_idx, img)
img_size = size(img,2);
tmp = true(img_size,1); tmp(train_idx) = false; tmp(test_idx) = false;
map = 1 : img_size;
Back.idx = map(tmp);
Back.dat = img(:, Back.idx);
Back.lab = 0 .* Back.idx;
end