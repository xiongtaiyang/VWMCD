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

# Set input/output directories
base_path = "/path/to/feature/files"  # Directory containing subject feature files
output_path = "/path/to/output"  # Results directory
os.makedirs(output_path, exist_ok=True)  # Create output folder if missing

# Analysis parameters
radius = 5  # Searchlight sphere radius (voxels)
n_jobs = -1  # Use all CPU cores
mask_filename = "/path/to/brain_mask.nii.gz"  # Brain mask file

# Load and process brain mask
mask_img = image.load_img(mask_filename)  # Load mask image
mask_data = mask_img.get_fdata()  # Get voxel data
# Create binary mask (1=active, 0=inactive)
mask_data = np.where(mask_data >= np.percentile(mask_data[mask_data > 0], 0), 1, 0)
mask_img = image.new_img_like(mask_img, mask_data)  # Create new mask image

# Identify subjects from feature filenames
subjects = [
    filename.replace("_feature.mat", "")  # Extract subject ID
    for filename in os.listdir(base_path)  # Scan directory
    if filename.endswith("_feature.mat") and os.path.isfile(os.path.join(base_path, filename))
]

# Create class labels: 20 samples per category (Body=1, Face=2, Place=3, Tool=4)
labels = np.array([1]*20 + [2]*20 + [3]*20 + [4]*20)

def load_mat_file(file_path):
    """Load MATLAB feature file and extract feature matrix"""
    try:
        data = loadmat(file_path)
        return data['feature_matrix']  # Return feature data
    except Exception as e:
        print(f"Error loading: {e}")
        raise

# Main processing loop
for subj in tqdm(sorted(subjects), desc="Processing Subjects"):
    # Set file paths
    feature_file = os.path.join(base_path, f"{subj}_feature.mat")
    output_file = os.path.join(output_path, f"{subj}_searchlight.nii.gz")
    
    # Skip subject if output exists
    if os.path.exists(output_file):
        continue
    
    # Load feature data
    feature_matrix = load_mat_file(feature_file)
    
    # Verify mask-feature dimensions match
    assert mask_data.sum() == feature_matrix.shape[0]
    
    # Create 4D brain image from features
    data_4d = np.zeros(mask_data.shape + (feature_matrix.shape[1],), dtype=feature_matrix.dtype)
    for t in range(feature_matrix.shape[1]):
        data_4d[..., t][mask_data == 1] = feature_matrix[:, t]
    combined_img = nib.Nifti1Image(data_4d, mask_img.affine, mask_img.header)
    
    # Configure cross-validation (10 stratified folds)
    #cv = StratifiedKFold(n_splits=10)
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
    # Initialize searchlight with linear SVM
    searchlight = SearchLight(
        mask_img=mask_img,
        process_mask_img=mask_img,
        radius=radius,
        estimator=SVC(kernel="linear"),
        n_jobs=n_jobs,
        cv=cv,
        verbose=1
    )
    
    # Run searchlight analysis
    searchlight.fit(combined_img, labels)
    
    # Save results as brain map
    result_img = nib.Nifti1Image(searchlight.scores_, affine=mask_img.affine)
    nib.save(result_img, output_file)
