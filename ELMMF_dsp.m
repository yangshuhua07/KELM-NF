n=1;
addpath([cd '\privates\']);

switch n,
    case 1,
        load results\ELMMF\indian.mat;
        its = 4;
    case 2,
        load results\MF\pavia.mat;
    case 3,
        load results\MF\ksc.mat;
        its = 4;
end

m_ELM_oa = 0; m_ELM_ka = 0; 
for i = 1 : its,
    m_ELM_oa = m_ELM_oa + ELM_oa(:,:,i);
    m_ELM_ka = m_ELM_ka + ELM_ka(:,:,i);
%     m_svm_oa = m_svm_oa + svm_oa(:,:,i);
%     m_svm_ka = m_svm_ka + svm_ka(:,:,i);
end

m_ELM_oa = m_ELM_oa / its; m_ELM_ka = m_ELM_ka / its;
% m_svm_oa = m_svm_oa / its; m_svm_ka = m_svm_ka / its;
disp(m_ELM_oa)


