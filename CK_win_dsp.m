function [m_ksrc_oa, m_ksrc_ka, m_svm_oa, m_svm_ka] = CK_win_dsp(n)
addpath([cd '\privates\']);

switch n,
    case 1,
        load results\CK\indian.mat;
        its = 4;
    case 2,
        load results\CK\pavia.mat;
		its = 4;
    case 3,
        load results\CK\ksc.mat;
		its = 4;
end

ksrc_oa = ksrc_oa(:,1:its); ksrc_ka = ksrc_ka(:,1:its);
svm_oa = svm_oa(:,1:its); svm_ka = svm_ka(:,1:its);
m_ksrc_oa = sum(ksrc_oa,2) / its; m_ksrc_ka = sum(ksrc_ka,2) / its;
m_svm_oa = sum(svm_oa,2) / its; m_svm_ka = sum(svm_ka,2) / its;

end