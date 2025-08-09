# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:38:38 2025

@author: ronglong.xiong
"""

import numpy as np
import scipy.io as sio
from scipy import stats
from sklearn.svm import SVR
from joblib import Parallel, delayed
import os
from sklearn.metrics import mean_squared_error
import h5py
from statsmodels.stats.multitest import multipletests
from plot_corr_scatter import plot_corr_scatter
import pickle
# 定义并行计算核数（根据CPU核心数调整）
N_JOBS = -1  

# 读取nifti文件
Feature_path = r'E:\HCP\WM-Getm-over\Results\machinelearning\FeatureMatrix_Canlab\data\HCP\FeatureMatrix'
VWMCM_dat_file = r'E:\HCP\WM-Getm-over\Results\machinelearning\FeatureMatrix_Canlab\data\VWMCM\VWMCM_dat.mat'
save_path_base = r'E:\HCP\WM-Getm-over\Results\machinelearning\FeatureMatrix_Canlab\data\HCP\RegressionResults\Validation'
beha_path = r'E:\HCP\WM-Getm-over\Behavior data\HCP_Behavior_701.mat'
patterns = ['Distributed','Localized']
beha_types = ['Acc','dp','RT']
data_types = ['Trial','Condition']
VoxelsTypes = ['BPV', 'NCV','PCV','PNC']
percentile=95
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


def validation_SVR(X_train,y_train,X_test,y_test):
    # 标准化处理（仅用训练集统计量）
    X_mean, X_std = X_train.mean(0), X_train.std(0)
    y_mean, y_std = y_train.mean(), y_train.std()
    
    X_train = (X_train - X_mean) / (X_std + 1e-6)
    X_test = (X_test - X_mean) / (X_std + 1e-6)
    y_train = (y_train - y_mean) / y_std
    
    # 模型训练与预测
    model = SVR(kernel='linear')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) * y_std + y_mean
    # 计算评估指标
    r, p = stats.pearsonr(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return r, p, rmse, y_pred

# 主程序
if __name__ == "__main__":
    # 数据路径配置
    stim_types = ['Body','Face','Place','Tool','All']
    colors = ['#FFBC00','#92D050','#A076A1','#F58A83','#38459B']

    for VoxelsType in VoxelsTypes:
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
            'colors':[]
        }
        save_path = os.path.join(save_path_base, VoxelsType)
        save_dir = os.path.dirname(save_path) if "." in os.path.basename(save_path) else save_path
        os.makedirs(save_dir, exist_ok=True)
        for beha_type in beha_types:
            # 加载行为数据
            beha_data = load_mat_data(beha_path, f'{beha_type}')
            for data_type in data_types:
                for pattern in patterns:
                    
                    if pattern == 'Distributed':
                        # 获取统计值并排序
                        VWMCM_dat_temp = load_mat_data(VWMCM_dat_file,'VWMCM_SVC_zscore_new')
                        VWMCM_dat= VWMCM_dat_temp[~np.isnan(VWMCM_dat_temp).flatten()]
                    if pattern == 'Localized':
                        VWMCM_dat = load_mat_data(VWMCM_dat_file,'Searchlight_result_fdrq005_zscore')
                        
                    threshold= np.percentile(VWMCM_dat[VWMCM_dat != 0], percentile)  # 前5%的阈值
                    selected_voxels = (VWMCM_dat > threshold).flatten()  # 二值化掩膜
                
            
                    # 处理每个刺激类型
                    for i,stim in enumerate(stim_types):
                        print(f"Processing {stim}...{pattern} {beha_type} {data_type}-wise")
    
                        # 1. 加载beta数据
                        diff_data = load_mat_data(
                            os.path.join(Feature_path,data_type, f"{stim}_response_2bk0bk_nosmooth.mat"),
                            f"{stim}_response"
                        )
                        
                        diff_data = diff_data[:,selected_voxels]
                        # 3. 计算行为指标差异
                        col_idx = stim_types.index(stim)
    
                        behavior_diff = (beha_data[:, 4 + col_idx] - beha_data[:, col_idx])
                        # 4. 计算相关性和显著性
                        r, p = pearsonr_parallel(diff_data, behavior_diff)
                        
                        # # 5. 筛选显著体素
                        pos_mask = (r > 0) 
                        neg_mask = (r < 0)
                        # 6. 生成特征向量
                        pos_data = np.nanmean(diff_data[:, pos_mask], axis=1)  # 对每行选中的列，剔除 NaN 后求平均
                        neg_data = np.nanmean(diff_data[:, neg_mask], axis=1)  # 同理
                        if VoxelsType=='BPV':
                            feature = pos_data + neg_data
                            feature = feature.reshape(-1,1)
                        elif VoxelsType=='PCV':
                            feature = pos_data
                            feature = feature.reshape(-1,1)
                        elif VoxelsType=='NCV':
                            feature = neg_data
                            feature = feature.reshape(-1,1)
                        elif VoxelsType=='PNC':
                            feature = [pos_data] + [neg_data]
                            feature = np.vstack(feature).T
                            
                        
                        # 7. 执行留一法回归
                        #r_val, p_val, rmse,predicted_behav = loo_svr(feature.reshape(-1,1), behavior_diff)
                        # 7. 执行十折交叉验证回归
                        #r_val, p_val, rmse,predicted_behav = kfold_svr(feature, behavior_diff, n_repeats=100, random_state=42)
                        X_train = feature[0:561,:]
                        y_train = behavior_diff[0:561]
                        X_test = feature[561:701,:]
                        y_test = behavior_diff[561:701]
                        r_val, p_val, rmse,predicted_behav = validation_SVR(X_train,y_train, X_test, y_test)
                        
                        # 存储结果
                        results['r_values'].append(r_val)
                        results['p_values'].append(p_val)
                        results['rmse_values'].append(rmse)
                        results['actual_behav'].append(y_test)
                        results['predicted_behav'].append(predicted_behav)
                        results['pos_index'].append(pos_mask)
                        results['neg_index'].append(neg_mask)
                        results['data_types'].append(data_type)
                        results['beha_types'].append(beha_type)
                        results['stim_types'].append(stim)
                        results['patterns'].append(pattern)
                        results['colors'].append(colors[i])
            
        p_values = np.array(results['p_values'])
        # 执行 Benjamini-Hochberg 校正
        reject, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        results['q_values'] = pvals_corrected
        
       
        # 保存数据
        with open(os.path.join(save_path,'All_HCP_Results_nosmooth_top5_validation.pkl'), 'wb') as f:
            pickle.dump(results, f)
        #打印结果
        print("\nFinal Results:")
        for index,(stim, r, p, q_value, rmse, x, y, data_type, beha_type,pattern,color) in enumerate(zip(results['stim_types'], 
                                                                results['r_values'], 
                                                                results['p_values'], 
                                                                results['q_values'],
                                                                results['rmse_values'], 
                                                                results['actual_behav'], 
                                                                results['predicted_behav'],
                                                                results['data_types'],
                                                                results['beha_types'],
                                                                results['patterns'],
                                                                results['colors']
                                                                )):
            # 确保目录存在                                                 
            # 创建目标目录
            file_path = os.path.join(save_path, beha_type)
            target_dir = os.path.dirname(file_path) if "." in os.path.basename(file_path) else file_path
            os.makedirs(target_dir, exist_ok=True)
            print(f"{stim}: {pattern} {beha_type} {data_type}-wise \n  r = {r:.4f}\n  p = {p:.4f}\n  RMSE = {rmse:.4f}\n")
            # 拟合线性回归方程和画图          
            plot_corr_scatter(
                x=x,
                y=y,
                q_value=q_value,
                color=color,
                title=f'{stim}: {pattern} Pattern ({data_type}-wise)',
                xlabel=f'Actual {beha_type}',
                ylabel=f'Predicted {beha_type}',
                save_path=os.path.join(save_path,beha_type, f'{stim}_predicted_results_{beha_type}_{data_type}_{pattern}_nosmooth_top5_validation.png'),
                bins=50,
                confidence_level=0.95
            )
