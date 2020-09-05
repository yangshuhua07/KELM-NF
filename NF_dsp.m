function [m_ksrc_oa, m_ksrc_ka, m_svm_oa, m_svm_ka] = NF_dsp(n)
addpath([cd '\privates\']);

switch n,
    case 1,
        load results\NF\indian.mat;
        its = 4;
    case 2,
        load results\NF\pavia.mat;
        its = 4;
    case 3,
        load results\NF\ksc.mat;
        its = 4;
end

m_ksrc_oa = 0; m_ksrc_ka = 0; m_svm_oa = 0; m_svm_ka = 0;
for i = 1 : its,
    m_ksrc_oa = m_ksrc_oa + ksrc_oa(:,:,i);
    m_ksrc_ka = m_ksrc_ka + ksrc_ka(:,:,i);
    m_svm_oa = m_svm_oa + svm_oa(:,:,i);
    m_svm_ka = m_svm_ka + svm_ka(:,:,i);
end

m_ksrc_oa = m_ksrc_oa / its; m_ksrc_ka = m_ksrc_ka / its;
m_svm_oa = m_svm_oa / its; m_svm_ka = m_svm_ka / its;

end