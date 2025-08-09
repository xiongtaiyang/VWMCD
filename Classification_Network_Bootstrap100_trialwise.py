#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 20:23:59 2025

@author: ubuntu
"""

import numpy as np
from tqdm import tqdm
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc
from load_mat_data import load_mat_data
import pickle
from scipy.io import savemat
from plot_confusion_matrix_new import plot_confusion_matrix
from k_fold_cv import k_fold_cv
from plot_multiclass_roc_curve import plot_multiclass_roc_curve
# 文件路径设置
matfile_path = r"/media/ubuntu/SD/ronglong.xiong/Feature_Matrix_Canlab/HCP_FeatureMatrix_Trial_GrayMask_Canlab.mat"
mask_file = r'/media/ubuntu/SD/ronglong.xiong/Feature_Matrix_Canlab/VWMCM/VWMCM_Network.mat'
mask_data = load_mat_data(mask_file ,'VWMCM_Network').ravel()
Networks = range(1,10)
pattern = load_mat_data(r'/media/ubuntu/SD/ronglong.xiong/Feature_Matrix_Canlab/VWMCM/VWMCM_dat.mat','VWMCM_FDR_05_SVC_Conda_linear_boot10000')

output_basepath = r"/media/ubuntu/SD/ronglong.xiong/Feature_Matrix_Canlab/HCP_classifyResult_Distributed/Trial/Classify_Linear_Networks"

feature_matrix_orig = load_mat_data(matfile_path,'data_matrix')
feature_matrix_orig = feature_matrix_orig[0:44880,:]

voxels = load_mat_data(mask_file ,'Voxels_new')
voxels = voxels.astype(np.int32).ravel()

# 生成被试分组标识（每4个样本属于同一被试）
subjects = list(range(0, 561))

labels = np.array([1] * 20 + [2] * 20 + [3] * 20 + [4] * 20)


# Train the SVM classifier
#svm = SVC(kernel='rbf',C=1,gamma='scale', probability=True)
svm = SVC(kernel='linear', probability=True)
All_Network_acc = np.full((9,len(voxels)),np.nan)
All_Network_stdacc = np.full((9,len(voxels)),np.nan)
colors = ['#FFBC00','#92D050','#A076A1','#F58A83']
class_name = ['Body', 'Face', 'Place', 'Tool']
for s,Network in tqdm(enumerate(Networks), desc="Processing Network"): 
    if Network < 9:
        # 创建新的掩膜：选择当前Network体素
        selected_network_index = mask_data == Network
        # 获取所有为True的线性索引（展平后的一维位置）
        all_true_indices = np.flatnonzero(selected_network_index)
        
    elif Network == 9:
        sorted_indices_asc = np.argsort(pattern.ravel())  
        # 选取前 k 个索引
        selected_indices_pos = sorted_indices_asc[:66016]
        selected_indices_neg = sorted_indices_asc[-66016:] 
        # 创建新的掩膜：选择大于当前阈值的体素
        all_true_indices = np.concatenate([selected_indices_pos, selected_indices_neg])

    
    # 修改后的代码
    for j, voxel in enumerate(voxels):
        # 为当前voxel数量创建输出目录
        output_path = os.path.join(output_basepath, f"Network-{Network}", f"voxel_{voxel}")
        os.makedirs(output_path, exist_ok=True)
        
        # 初始化储存所有迭代结果的列表
        all_iter_results = {
            'mean_accs': [],
            'std_accs': [],
            'roc_aucs': [],
            'conf_matrices': [],
            'y_true_labels': [],
            'y_pred_labels': [],
            'y_test_binarized': [],
            'y_score': []
        }
        
        # 添加100次迭代循环
        for iteration in tqdm(range(100), desc=f"Voxel {voxel} - 100 Iterations"):
            if voxel < len(all_true_indices):
                # 随机选择特征（每次迭代都重新选择）
                np.random.seed(iteration)  # 确保可重复性
                selected_linear_indices = np.random.choice(all_true_indices, voxel, replace=False)
                feature_matrix_Network = feature_matrix_orig[:, selected_linear_indices]
                
                # 每次迭代初始化结果存储
                iter_tprs = []
                iter_fprs = []
                iter_y_true_labels = []
                iter_roc_aucs = []
                iter_y_test_binarized = []
                iter_y_score = []
                iter_y_pred_labels = []
                iter_model_results = []
                iter_std_accs = []
                iter_mean_accs = []
                
                # 处理每个被试
                for subj_idx in range(len(subjects)):
                    feature_matrix_subj = feature_matrix_Network[subj_idx*80:subj_idx*80+80, :] 
                    
                    # 执行十折交叉验证
                    results = k_fold_cv(feature_matrix_subj, labels, groups=None, model=svm, 
                                       n_jobs=10, random_state=subj_idx)
                    
                    iter_y_true_labels.append(results['y_true'])
                    iter_y_pred_labels.append(results['y_pred'])
                    iter_y_test_binarized.append(results['y_true_binarized'])
                    iter_y_score.append(results['y_score'])
                    iter_model_results.append(results)
                    iter_std_accs.append(results['std_accuracy'])
                    iter_mean_accs.append(results['mean_accuracy'])
                
                # 合并所有被试的结果
                iter_y_pred_labels = np.hstack(iter_y_pred_labels)
                iter_y_true_labels = np.hstack(iter_y_true_labels)
                iter_y_test_binarized = np.vstack(iter_y_test_binarized)
                iter_y_score = np.vstack(iter_y_score)
                
                # 计算ROC曲线
                n_classes = iter_y_test_binarized.shape[1]
                class_roc_aucs = []
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(iter_y_test_binarized[:, i], iter_y_score[:, i])
                    roc_auc = auc(fpr, tpr)
                    class_roc_aucs.append(roc_auc)
                
                # 计算混淆矩阵
                conf_matrix = confusion_matrix(iter_y_true_labels, iter_y_pred_labels)
                
                # 保存本次迭代的结果
                iter_mean_acc = np.mean(iter_mean_accs)
                iter_std_acc = np.mean(iter_std_accs)
                
                # 储存到总结果中
                all_iter_results['mean_accs'].append(iter_mean_acc)
                all_iter_results['std_accs'].append(iter_std_acc)
                all_iter_results['roc_aucs'].append(np.mean(class_roc_aucs))
                all_iter_results['conf_matrices'].append(conf_matrix)
                all_iter_results['y_true_labels'].append(iter_y_true_labels)
                all_iter_results['y_pred_labels'].append(iter_y_pred_labels)
                all_iter_results['y_test_binarized'].append(iter_y_test_binarized)
                all_iter_results['y_score'].append(iter_y_score)
        
        # 计算100次迭代的平均性能
        mean_acc_over_iters = np.mean(all_iter_results['mean_accs'])
        std_acc_over_iters = np.mean(all_iter_results['std_accs'])
        mean_auc_over_iters = np.mean(all_iter_results['roc_aucs'])
        std_auc_over_iters = np.std(all_iter_results['roc_aucs'])
        
        # 保存总体结果
        final_results = pd.DataFrame({
            'Mean_Accuracy': [mean_acc_over_iters],
            'Std_Accuracy': [std_acc_over_iters],
            'Mean_AUC': [mean_auc_over_iters],
            'Std_AUC': [std_auc_over_iters],
            'Iterations': [100]
        })
        final_results.to_csv(os.path.join(output_path, 'overall_results_100_iterations.csv'), index=False)
        
        # 计算平均混淆矩阵
        avg_conf_matrix = np.mean(all_iter_results['conf_matrices'], axis=0)
        savefile_name_cm = os.path.join(output_path, 'Average_ConfusionMatrix_100iter.png')
        plot_confusion_matrix(avg_conf_matrix, savefile_name=savefile_name_cm, class_labels=class_name)
        
        # 保存所有迭代的原始数据
        with open(os.path.join(output_path, "all_iterations_results.pkl"), "wb") as f:
            pickle.dump(all_iter_results, f)
                 
                   
# 保存为 .mat 文件
All_Network_accs = np.array(All_Network_acc)
All_Network_stdacc = np.array(All_Network_stdacc)

output_path = os.path.join(output_basepath,r"All_Network_Mean_Accuracy_linear_Trial_trainnew_net9.mat")
savemat(output_path, {"All_Network_Mean_Accuracy": All_Network_accs,"All_Network_Std_Accuracy": All_Network_stdacc})
            
    


    

    
