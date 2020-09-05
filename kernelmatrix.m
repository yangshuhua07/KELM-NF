function K = kernelmatrix(X, X2, sig)
% RBF
n1sq = sum(X.^2); n1 = size(X,2);
n2sq = sum(X2.^2); n2 = size(X2,2);
D = n1sq'*ones(1,n2) + ones(n1,1)*n2sq - 2 * X' * X2;
K = exp(-1 / (2*sig^2) * D);
end

