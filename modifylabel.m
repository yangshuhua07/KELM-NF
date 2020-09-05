review=pred;
for i=1:8985,
m=pred(1,i);
switch m
    case 1,review(1,i)=2;
    case 2,review(1,i)=3;
    case 3,review(1,i)=5;
    case 4,review(1,i)=6;
    case 5,review(1,i)=8;
    case 6,review(1,i)=10;
    case 7,review(1,i)=11;
    case 8,review(1,i)=12;
    case 9,review(1,i)=14;
end
end
accurancy=class_eval(pred,actual);