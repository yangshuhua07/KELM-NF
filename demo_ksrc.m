function demo_ksrc(n)
addpath([cd '\privates\']);
[img, img_gt, rows, cols] = load_data(n);
switch n,
    case 1,
        load results\KSRC\KSRC_indian.mat;
        nt = 5; its = 9;
    case 2,
        load results\KSRC\KSRC_pavia.mat;
        nt = 5; its = 5;
    case 3,
        load results\KSRC\KSRC_ksc.mat;
        nt = 3; its = 5;
end

it = search(KSRC_acc, nt, its);
tmp = KSRC_para{nt,it}; gam = tmp(1);
test_pred = KSRC_pred{nt,it};
acc = KSRC_acc{nt,it};

[train_idx, test_idx] = load_train_test(n, 1, nt, it);
[Train, Test] = set_train_test(train_idx, test_idx, img, img_gt);
Back = get_background(train_idx, test_idx, img);
        
mu = 1e-3; lam = 1e-4; sig = sqrt(0.5 ./ gam);
p.mu = mu; p.lam = lam;
AtA = RBF(Train.dat, Train.dat, sig, 10000);
AtX = RBF(Train.dat, Back.dat, sig, 10000);
S = SpRegKL1(AtX, AtA, p);
back_pred = class_ker_pred(AtX, AtA, S, Back.lab, Train.lab);

pred = zeros(1, rows*cols);
pred(Train.idx) = Train.lab; pred(Test.idx) = test_pred; pred(Back.idx) = back_pred;

switch n
    case 1, save results\MAP\KSRC_indian.mat acc pred;
    case 2, save results\MAP\KSRC_pavia.mat acc pred;
    case 3, save results\MAP\KSRC_ksc.mat acc pred;
end
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