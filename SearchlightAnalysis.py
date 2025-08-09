#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 00:57:55 2025

@author: ubuntu
"""

import os
import numpy as np
import nibabel as nib
from nilearn import image
from nilearn.decoding import SearchLight
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from nilearn import plotting
from scipy.io import loadmat
from sklearn.model_selection import RepeatedKFold

# 文件路径设置
base_path = r"~"
output_path = r"~"
os.makedirs(output_path, exist_ok=True)  # 创建输出目录

# 参数设置
radius = 5  # 搜索半径（单位：体素）
n_jobs = -1  # 使用所有CPU核心
mask_filename = r"mask.nii" 

# 加载掩膜文件（如果提供了mask）
if mask_filename:
    mask_img = image.load_img(mask_filename)
    print(f"Loaded mask: {mask_img.shape}")
    
    # 获取掩膜数据并修改
    mask_data = mask_img.get_fdata()

    threshold = np.percentile(mask_data[mask_data > 0], 0)
    # 创建新的掩膜：选择大于当前阈值的体素
    mask_data = np.where(mask_data >= threshold, 1, 0)  # 使大于等于1的值设为1，其余设为0

    # 创建新mask图像
    mask_img = image.new_img_like(mask_img, mask_data)
else:
    mask_img = None
    print("No mask file provided, proceeding without mask.")
# 遍历所有被试
# 获取所有 {subj}_feature.mat 文件并提取subjects
subjects = [
    filename.replace("_feature.mat", "") 
    for filename in os.listdir(base_path) 
    if (
        filename.endswith("_feature.mat") 
        and os.path.isfile(os.path.join(base_path, filename))
    )
]
 
# 创建标签 (1=Body, 2=Face,3=Place, 4=Tool)
labels = np.array([1] * 20 + [2] * 20 +[3] * 20 + [4] * 20)
        

def load_mat_file(file_path):
    try:
        data = loadmat(file_path)
        return data['feature_matrix']  # 根据实际变量名调整
    except Exception as e:
        print(f"使用scipy加载时出错: {e}")
        raise
        
# searchlight分析
results = {}
#for subj in tqdm(subjects, asend="Processing Subjects"):
for subj in tqdm(sorted(subjects, reverse=False), desc="Processing Subjects"):
    feature_file = os.path.join(base_path, f"{subj}_feature.mat")
    output_file = os.path.join(output_path, f"{subj}_searchlight.nii.gz")
    
    if subj == r"123420":
        continue
    
    if os.path.exists(output_file):
        print(f"文件 {output_file} 已存在，跳过当前循环。")
        continue
    
    feature_matrix = load_mat_file(feature_file)
    
    # 将feature_matrix重新写入nifti图像中
    # 步骤1：验证维度一致性
    assert mask_data.sum() == feature_matrix.shape[0], "Mask体素数量与特征矩阵行数不匹配"
    
    # 步骤2：获取展平后的mask索引
    mask_flat = mask_data.ravel()  # 展平为1D数组 (91 * 109 * 91,)
    indices = np.where(mask_flat == 1)[0]  # 获取有效体素位置 (130026,)
    
    # 3. 创建 4D NIfTI 图像
    # 初始化4D矩阵 (91, 109, 91, 80)
    data_4d = np.zeros(mask_data.shape + (feature_matrix.shape[1],), 
                      dtype=feature_matrix.dtype)
    
    # 填充数据（使用向量化操作）
    for t in range(feature_matrix.shape[1]):
        data_4d[..., t][mask_data == 1] = feature_matrix[:, t]
    
    # 创建并保存NIfTI
    combined_img = nib.Nifti1Image(data_4d, mask_img.affine, mask_img.header)
    

    # 交叉验证设置
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
    #cv = StratifiedKFold(n_splits=10)  # 10折交叉验证
    # 初始化searchlight
    searchlight = SearchLight(
        mask_img=mask_img,
        process_mask_img=mask_img,
        radius=radius,
        estimator=SVC(kernel="linear"),  # 使用线性核支持向量机
        n_jobs=n_jobs,
        cv=cv,
        verbose=1
    )

    # 拟合模型
    searchlight.fit(combined_img, labels)

    # 保存结果
    results[subj] = searchlight.scores_

    # 保存searchlight结果到 .nii 文件
    
    result_img = nib.Nifti1Image(searchlight.scores_, affine=mask_img.affine)
    nib.save(result_img, output_file)
    
    # plotting.plot_glass_brain(
    #     result_img,
    #     title =f"Subj-{subj}",
    #     display_mode="lyrz",
    #     vmin=0.5,
    #     threshold=0.5,
    #     plot_abs=False,
    #     colorbar=True,
    #     cmap='RdBu_r'
    #     )
    # plotting.show()
    # print(f"Saved searchlight result for subject {subj} to {output_file}\n")
