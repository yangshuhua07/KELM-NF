n=1;
addpath([cd '\privates\']);
t=cputime;
t2=clock;
[img, img_gt, rows, cols] = load_data(n);
L = size(img, 1);

[train_idx, test_idx] = load_train_test(n, 1, 5, 5);
[Train, Test] = set_train_test(train_idx, test_idx, img, img_gt);  
% 
% xapp=[Train.lab' Train.dat'];
% xtest=[Test.lab' Test.dat'];

% kerneloption = [1];
% c = 1024;
% 
% [TTrain,TTest,TrainAC,accur_ELM,TY,pred,actual] = elm_kernel(xapp,xtest,1,c,'RBF_kernel',kerneloption);

% [TrainingTime, TestingTime, TrainingAccuracy, TestingAccuracy,TY,pred,actual] = elm(xapp,xtest, 1,5000,  'sig');
% acc = class_eval(pred, actual);

model = svmtrain(Train.lab',Train.dat', ['-c ',num2str(100),' -g ',num2str(0.5)]);
pred_label = svmpredict(Test.lab', Test.dat', model);
ac=class_eval(pred_label, Test.lab);
etime(clock,t2)
t1=cputime-t