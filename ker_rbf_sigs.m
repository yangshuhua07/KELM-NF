function K = ker_rbf_sigs(X, X2)
% RBF kernel
n1sq = sum(X.^2); n1 = size(X,2);
n2sq = sum(X2.^2); n2 = size(X2,2);
D = (ones(n2,1)*n1sq)' + ones(n1,1)*n2sq - 2 * X' * X2;
K = -D/2;
end
