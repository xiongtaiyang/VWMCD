% Clear command window and workspace
clc
clear

% Load fMRI data matrix from a .mat file
load('data_matrix.mat');

% Create an fmri_data object using a template image ('template.nii') 
% and a mask image ('mask.nii') that defines the brain regions of interest
dat = fmri_data('template.nii', 'mask.nii');

% Create class labels for all samples (each class repeated 700 times)
% The dataset has 4 classes, creating a label vector of 2800 samples (700×4)
label = repmat([1:4]', [700, 1]);

% Assign labels to the fmri_data object
dat.Y = label;

% Assign fMRI data matrix to dat.dat (transpose needed for proper orientation)
% Rows become voxels, columns become samples
dat.dat = data_matrix';

% Initialize random number generator with seed 20 for reproducibility
rng(20, 'twister'); 

% Generate subject IDs: each subject has 4 consecutive samples
subjIDs = repelem(1:700, 4);  % Creating 700 subjects × 4 = 2800 samples

% Set up subject-level 10-fold cross-validation
unique_subjects = unique(subjIDs);  % Get list of unique subject IDs
num_subjects = length(unique_subjects);  % Total number of subjects (700)
num_folds = 10;  % Number of folds for cross-validation

% Set random seed for fold assignment reproducibility
rng(42);  
% Assign subjects to folds (subject-level assignment)
subjects_folds = crossvalind('KFold', num_subjects, num_folds);  

% Map subject-level fold assignments to individual samples
% Create fold index vector for all 2800 samples
nfolds = zeros(size(subjIDs));
for i = 1:num_subjects
    subj = unique_subjects(i);
    idx = (subjIDs == subj);  % Find indices for all samples from this subject
    nfolds(idx) = subjects_folds(i);  % Assign same fold to all samples of this subject
end

% Run SVM classification with cross-validation and bootstrap statistics
[cverr, stats_boot, optout] = predict(dat, ...
    'algorithm_name', 'cv_svm', ... % Use SVM classifier
    'nfolds', 10, ...              % 10-fold cross-validation
    'error_type', 'mse', ...        % Evaluate using mean squared error
    'useparallel', 1, ...          % Enable parallel processing
    'bootweights', ...              % Bootstrap weight maps
    'bootsamples', 10000);          % Use 10,000 bootstrap samples
