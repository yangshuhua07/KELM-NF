clc;
clear;
n=1;nt=3;it=4;
addpath([cd '\data\']);
load data\indian\Indian_gt.mat;
[img, img_gt, rows, cols] = load_data(n);



 load indian_map.mat;
load results\KSRCMF\KSRCMF_indian.mat;


   [train_idx, test_idx] = load_train_test(n, 1, nt, it);
   [Train, Test] = set_train_test(train_idx, test_idx, img, img_gt);
gt = img_gt;
gt = gt ~= 0;

pred =zeros(rows*cols,1);
pred(Test.idx) = KSRCMF_pred{nt,it};
pred(Train.idx) = Train.lab;
%figure; imshow(uint8(reshape(indian_pines_gt(1:rows*cols),rows,cols)), map, 'border','tight');
figure; imshow(indian_pines_gt, map, 'border','tight');
         figure; imshow(uint8(reshape(pred(1:rows*cols),rows,cols)), map, 'border','tight');