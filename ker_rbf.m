function K = ker_rbf(X, X2, sigma)
% RBF kernel
n1sq = sum(X.^2); n1 = size(X,2);%sum（A）默认列和，sum(A,2)行和，sum(A(:))矩阵和
n2sq = sum(X2.^2); n2 = size(X2,2);
D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq - 2 * X' * X2;
K = exp(-D/(2*sigma^2));
end

