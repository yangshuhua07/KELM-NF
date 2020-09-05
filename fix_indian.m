function img_gt = fix_indian( gt )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

img_gt = gt;
img_gt(gt==2) = 1;
img_gt(gt==3) = 2;
img_gt(gt==5) = 3;
img_gt(gt==6) = 4;
img_gt(gt==8) = 5;
img_gt(gt==10) = 6;
img_gt(gt==11) = 7;
img_gt(gt==12) = 8;
img_gt(gt==14) = 9;

end

