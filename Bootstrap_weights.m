clc
clear

load('data_matrix.mat');
dat = fmri_data('E:\HCP\WM-Getm-over\FunImg\group\cond_diff_diff\100206_cond_0001.nii','E:\HCP\WM-Getm-over\Schaefer_atlas\rschaefer400MNI.nii');
label = repmat([1:4]',[700,1]);

dat.Y = label;
dat.dat = data_matrix';
rng(20, 'twister'); 

% 生成被试ID（每个被试重复4次）
subjIDs = repelem(1:701, 4);  % 700个被试，每个被试4个样本

% 手动生成被试级别的5折交叉验证
unique_subjects = unique(subjIDs);  % 唯一被试列表
num_subjects = length(unique_subjects);  % 被试总数
num_folds = 10;

% 随机将被试分配到5个折叠中（确保每个被试属于唯一折叠）
rng(42);  % 固定随机种子以保证可重复性
subjects_folds = crossvalind('KFold', num_subjects, num_folds);  % 被试级别的折叠分配

% 将被试的折叠编号映射到所有样本
nfolds = zeros(size(subjIDs));
for i = 1:num_subjects
    subj = unique_subjects(i);
    idx = (subjIDs == subj);  % 当前被试的所有样本索引
    nfolds(idx) = subjects_folds(i);  % 样本继承被试的折叠编号
end
[cverr, stats_boot, optout] = predict(dat, 'algorithm_name', 'cv_svm','nfolds', 10, 'error_type', 'mse', 'useparallel', 1, 'bootweights', 'bootsamples', 10000);
