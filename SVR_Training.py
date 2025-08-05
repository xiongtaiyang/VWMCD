# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:38:38 2025

@author: ronglong.xiong
"""

import numpy as np
import scipy.io as sio
from scipy import stats
from joblib import Parallel, delayed
import os
import h5py
from statsmodels.stats.multitest import multipletests
from plot_corr_scatter import plot_corr_scatter
import pickle
from cross_svr import repeated_kfold_final_corr
# 定义并行计算核数（根据CPU核心数调整）
N_JOBS = -1  

# 读取nifti文件
Feature_path = r'E:\HCP\WM-Getm-over\Results\machinelearning\FeatureMatrix_Canlab\data\HCP\FeatureMatrix'
VWMCM_dat_file = r'E:\HCP\WM-Getm-over\Results\machinelearning\FeatureMatrix_Canlab\data\VWMCM\VWMCM_dat.mat'
save_path_base = r'E:\HCP\WM-Getm-over\Results\machinelearning\FeatureMatrix_Canlab\data\HCP\RegressionResults\Training_allvoxels_top5'
beha_path = r'E:\HCP\WM-Getm-over\Behavior data\HCP_Behavior_701.mat'
pattern = 'Distributed'
beha_types = ['Acc','dp','RT']
data_type = 'Condition'
VoxelsType = 'PNC'
percentile =95
def load_mat_data(file_path, var_name):
    """加载MATLAB数据文件并提取指定变量，自动兼容v7.3和非v7.3版本"""
    try:
        # 尝试用scipy读取常规版本
        data = sio.loadmat(file_path)
    except NotImplementedError:
        # 捕获v7.3格式的异常，使用h5py读取
        try:
            with h5py.File(file_path, 'r') as f:
                dataset = f[var_name]
                # 转置以保持与MATLAB一致的维度排列
                data = np.array(dataset).astype(np.float32).T  
        except KeyError:
            raise KeyError(f"变量 '{var_name}' 在文件中不存在")
        except Exception as e:
            raise IOError(f"读取v7.3文件失败: {str(e)}")
    else:
        # 从常规mat文件中提取数据
        if var_name not in data:
            raise KeyError(f"变量 '{var_name}' 在文件中不存在")
        data = data[var_name].astype(np.float32)
    
    return data

def pearsonr_parallel(data, behavior):
    """并行计算Pearson相关系数和p值"""
    n_voxels = data.shape[1]
    results = Parallel(n_jobs=N_JOBS)(delayed(stats.pearsonr)(data[:, i], behavior) for i in range(n_voxels))
    r_values = np.array([r for r, _ in results])
    p_values = np.array([p for _, p in results])
    return r_values, p_values


# 主程序
if __name__ == "__main__":
    # 数据路径配置
    stim_types = ['Body','Face','Place','Tool','All']
    colors = ['#FFBC00','#92D050','#A076A1','#F58A83','#38459B']
    # 初始化结果存储
    results = {
        'r_values': [],
        'p_values': [],
        'rmse_values': [],
        'actual_behav':[],
        'predicted_behav':[],
        'pos_index':[],
        'neg_index':[],
        'data_types':[],
        'beha_types':[],
        'stim_types':[],
        'patterns':[],
        'q_values':[],
        'colors':[],
        'weights':[],
        'percentile':[]
    }

    for beha_type in beha_types:
        save_path = os.path.join(save_path_base, beha_type)
        save_dir = os.path.dirname(save_path) if "." in os.path.basename(save_path) else save_path
        os.makedirs(save_dir, exist_ok=True)
        # 加载行为数据
        beha_data = load_mat_data(beha_path, f'{beha_type}')

        # 获取统计值并排序
        VWMCM_dat_temp = load_mat_data(VWMCM_dat_file,'VWMCM_FDR_05_SVC_Conda_linear_boot10000')
        VWMCM_dat= VWMCM_dat_temp[~np.isnan(VWMCM_dat_temp).flatten()]

        threshold = np.percentile(VWMCM_dat[VWMCM_dat != 0], percentile)  # 前5%的阈值
        selected_voxels = (VWMCM_dat > threshold).flatten()  # 二值化掩膜
            
        # 处理每个刺激类型
        for i,stim in enumerate(stim_types):
            print(f"Processing {stim}... {beha_type}")

            # 1. 加载beta数据
            diff_data = load_mat_data(
                os.path.join(Feature_path,data_type, f"{stim}_response_2bk0bk_6mm.mat"),
                f"{stim}_response"
            )
            
            diff_data = diff_data[:,selected_voxels]
            # 3. 计算行为指标差异
            col_idx = stim_types.index(stim)

            behavior_diff = (beha_data[:, 4 + col_idx] - beha_data[:, col_idx])
            # 4. 计算相关性和显著性
            
            feature = diff_data
                
            # 7. 执行留一法回归
            #r_val, p_val, rmse,predicted_behav = loo_svr(feature.reshape(-1,1), behavior_diff)
            # 7. 执行十折交叉验证回归
            y_true, y_pred, r_val, p_val, rmse = repeated_kfold_final_corr(feature[0:561,:], behavior_diff[0:561], n_repeats=100, n_splits=10,n_jobs=-1)
            # 存储结果
            results['r_values'].append(r_val)
            results['p_values'].append(p_val)
            results['rmse_values'].append(rmse)
            results['actual_behav'].append(behavior_diff[0:561])
            results['predicted_behav'].append(y_pred)
            results['data_types'].append(data_type)
            results['beha_types'].append(beha_type)
            results['stim_types'].append(stim)
            results['patterns'].append(pattern)
            results['colors'].append(colors[i])
            results['percentile'].append(percentile)
        
    p_values = np.array(results['p_values'])
    # 执行 Benjamini-Hochberg 校正
    reject, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh')
    results['q_values'] = pvals_corrected
    
   
    # 保存数据
    with open(os.path.join(save_path,'All_HCP_Results_6mm_topbottom5_linear_combination_training.pkl'), 'wb') as f:
        pickle.dump(results, f)
    #打印结果
    print("\nFinal Results:")
    for index,(stim, r, p, q_value, rmse, x, y, beha_type,color) in enumerate(zip(results['stim_types'], 
                                                            results['r_values'], 
                                                            results['p_values'], 
                                                            results['q_values'],
                                                            results['rmse_values'], 
                                                            results['actual_behav'], 
                                                            results['predicted_behav'],
                                                            results['beha_types'],
                                                            results['colors'],
                                                            )):                                             
        print(f"{stim}: {beha_type} \n  r = {r:.4f}\n  p = {p:.4f}\n  RMSE = {rmse:.4f}\n")
        save_path = os.path.join(save_path_base,beha_type)
        save_dir = os.path.dirname(save_path) if "." in os.path.basename(save_path) else save_path
        os.makedirs(save_dir, exist_ok=True)
        # 拟合线性回归方程和画图          
        plot_corr_scatter(
            x=x,
            y=y,
            q_value=q_value,
            color=color,
            #title=f'{stim}: ',
            xlabel=f'Actual {beha_type}',
            ylabel=f'Predicted {beha_type}',
            save_path=os.path.join(save_path, f'{stim}_predicted_results_{beha_type}_topbottom_6mm_training.png'),
            bins=50,
            confidence_level=0.95
        )
