#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pickle
from scipy.io import savemat

# Import custom functions (loading, visualization, CV)
from load_mat_data import load_mat_data
from plot_confusion_matrix_new import plot_confusion_matrix
from k_fold_cv import k_fold_cv
from plot_multiclass_roc_curve import plot_multiclass_roc_curve

# ===== Configuration =====
# Paths to data files (replace with actual paths)
matfile_path = "path/to/fmri_data.mat"
mask_file = "path/to/brain_mask.mat"
output_basepath = "path/to/results"

# Load brain mask and networks definition
mask_data = load_mat_data(mask_file, 'mask_key').ravel()
Networks = range(1, 10)  # Brain networks to analyze
pattern = load_mat_data("path/to/pattern.mat", 'pattern_key')

# Load fMRI data matrix
feature_matrix_orig = load_mat_data(matfile_path, 'fmri_key')
feature_matrix_orig = feature_matrix_orig[0:44880, :]  # Use first 561 subjects

# Voxel counts to test
voxels = load_mat_data(mask_file, 'voxels_key').astype(np.int32).ravel()

# Subject and task parameters
subjects = list(range(0, 561))  # Subject IDs
labels = np.array([1]*20 + [2]*20 + [3]*20 + [4]*20)  # Class labels (4 categories)
class_name = ['Body', 'Face', 'Place', 'Tool']  # Class names
colors = ['#FFBC00','#92D050','#A076A1','#F58A83']  # Plot colors

# Initialize SVM classifier
svm = SVC(kernel='linear', probability=True)

# Result matrices
All_Network_acc = np.full((9, len(voxels)), np.nan)
All_Network_stdacc = np.full((9, len(voxels)), np.nan)

# ===== Main Processing Loop =====
for s, Network in tqdm(enumerate(Networks), desc="Brain Networks"):
    # Select brain regions based on network type
    if Network < 9:
        # For defined brain networks
        selected_network_index = mask_data == Network
        all_true_indices = np.flatnonzero(selected_network_index)
    else:
        # For whole-brain pattern analysis
        sorted_indices_asc = np.argsort(pattern.ravel())  
        selected_indices = np.concatenate([
            sorted_indices_asc[:66016], 
            sorted_indices_asc[-66016:]
        ])
        all_true_indices = selected_indices
        
    # Process different voxel counts
    for j, voxel in enumerate(voxels):
        output_path = os.path.join(output_basepath, f"Network-{Network}", f"voxel_{voxel}")
        os.makedirs(output_path, exist_ok=True)
        
        # Storage for 100 permutation results
        all_iter_results = {
            'mean_accs': [], 'std_accs': [], 'roc_aucs': [],
            'conf_matrices': [], 'y_true_labels': [], 
            'y_pred_labels': [], 'y_test_binarized': [], 'y_score': []
        }
        
        # 100 iterations with random voxel selection
        for iteration in tqdm(range(100), desc=f"Voxel {voxel}"):
            if voxel < len(all_true_indices):
                # Random voxel selection for this iteration
                np.random.seed(iteration)
                selected_indices = np.random.choice(all_true_indices, voxel, False)
                feature_matrix_Network = feature_matrix_orig[:, selected_indices]
                
                # Per-subject processing
                subj_results = {'y_true': [], 'y_pred': [], 'y_score': [],
                               'y_true_binarized': [], 'mean_accuracy': [], 
                               'std_accuracy': []}
                
                for subj_idx in range(len(subjects)):
                    # Extract subject data (80 samples per subject)
                    subject_data = feature_matrix_Network[subj_idx*80 : subj_idx*80+80, :]
                    
                    # Run 10-fold cross-validation
                    results = k_fold_cv(subject_data, labels, model=svm, n_jobs=10)
                    
                    # Store results
                    subj_results['y_true'].append(results['y_true'])
                    subj_results['y_pred'].append(results['y_pred'])
                    subj_results['y_score'].append(results['y_score'])
                    subj_results['y_true_binarized'].append(results['y_true_binarized'])
                    subj_results['mean_accuracy'].append(results['mean_accuracy'])
                    subj_results['std_accuracy'].append(results['std_accuracy'])
                
                # Process group results
                y_pred = np.hstack(subj_results['y_pred'])
                y_true = np.hstack(subj_results['y_true'])
                y_test_binarized = np.vstack(subj_results['y_true_binarized'])
                y_score = np.vstack(subj_results['y_score'])
                
                # Calculate ROC and confusion matrix
                roc_aucs = []
                for i in range(y_test_binarized.shape[1]):
                    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
                    roc_aucs.append(auc(fpr, tpr))
                
                conf_matrix = confusion_matrix(y_true, y_pred)
                
                # Store iteration results
                all_iter_results['mean_accs'].append(np.mean(subj_results['mean_accuracy']))
                all_iter_results['roc_aucs'].append(np.mean(roc_aucs))
                all_iter_results['conf_matrices'].append(conf_matrix)
                all_iter_results['y_true_labels'].append(y_true)
                all_iter_results['y_pred_labels'].append(y_pred)
                all_iter_results['y_test_binarized'].append(y_test_binarized)
                all_iter_results['y_score'].append(y_score)
        
        # ===== Save Results =====
        # Aggregate metrics
        mean_acc = np.mean(all_iter_results['mean_accs'])
        mean_auc = np.mean(all_iter_results['roc_aucs'])
        
        # Save CSV summary
        pd.DataFrame({
            'Mean_Accuracy': [mean_acc],
            'Mean_AUC': [mean_auc],
            'Iterations': [100]
        }).to_csv(os.path.join(output_path, 'summary.csv'))
        
        # Save confusion matrix plot
        avg_conf_matrix = np.mean(all_iter_results['conf_matrices'], axis=0)
        plot_confusion_matrix(avg_conf_matrix, 
                              os.path.join(output_path, 'confusion_matrix.png'),
                              class_labels=class_name)
        
        # Save raw results
        with open(os.path.join(output_path, "results.pkl"), "wb") as f:
            pickle.dump(all_iter_results, f)
            
        # Update network-level results
        All_Network_acc[s, j] = mean_acc

# Save final results
savemat(os.path.join(output_basepath, "network_results.mat"), 
        {"Accuracy": All_Network_acc, "Std_Accuracy": All_Network_stdacc})
