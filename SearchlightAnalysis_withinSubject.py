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

# Set base and output directories
base_path = r"~"  # Input data directory
output_path = r"~"  # Results directory
os.makedirs(output_path, exist_ok=True)  # Create output folder

# Analysis parameters
radius = 5  # Searchlight sphere radius in voxels
n_jobs = -1  # Use all available CPU cores
mask_filename = r"mask.nii"  # Brain mask file path

# Load and process brain mask
if mask_filename:
    mask_img = image.load_img(mask_filename)  # Load mask image
    mask_data = mask_img.get_fdata()  # Get voxel data
    
    # Threshold mask to binary (0/1)
    threshold = np.percentile(mask_data[mask_data > 0], 0)  # Find minimum non-zero value
    mask_data = np.where(mask_data >= threshold, 1, 0)  # Create binary mask
    
    # Create new mask image object
    mask_img = image.new_img_like(mask_img, mask_data)
else:
    mask_img = None  # Proceed without mask

# Identify subjects from feature files
subjects = [
    filename.replace("_feature.mat", "")  # Extract subject ID
    for filename in os.listdir(base_path)  # Scan input directory
    if (
        filename.endswith("_feature.mat")  # Only feature files
        and os.path.isfile(os.path.join(base_path, filename))  # Verify file exists
    )
]

# Create class labels: 20 samples per category
labels = np.array([1]*20 + [2]*20 + [3]*20 + [4]*20)  # Body, Face, Place, Tool

def load_mat_file(file_path):
    """Load MATLAB feature file (implementation not shown)"""
    # Would typically contain code to load feature_matrix
    pass

# Main processing loop would go here
# For each subject:
#   Load feature file
#   Create 4D brain image from features
#   Run searchlight analysis
#   Save results as NIfTI file

# Example saving command (not in actual loop)
nib.save(result_img, output_file)  # Save NIfTI image
