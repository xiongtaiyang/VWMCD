#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:38:38 2025

@author: ronglong.xiong
"""

# Import required libraries
import numpy as np
import scipy.io as sio
from scipy import stats
from joblib import Parallel, delayed  # For parallel processing
import os
import h5py  # For handling HDF5 files (MATLAB v7.3 format)
from statsmodels.stats.multitest import multipletests  # For multiple comparison correction
from plot_corr_scatter import plot_corr_scatter  # Custom visualization function
import pickle  # For saving Python objects
from cross_svr import repeated_kfold_final_corr  # Custom cross-validation regression function

# Set number of jobs for parallel processing (-1 = use all available CPU cores)
N_JOBS = -1  

# Define file paths and parameters for analysis
Feature_path = '~'
VWMCM_dat_file = 'VWMCM_dat.mat'
save_path_base = '~'
beha_path = r'HCP_Behavior_701.mat'
pattern = 'Distributed'  # Analysis pattern type
beha_types = ['Acc','dp','RT']  # Behavioral metrics to analyze
data_type = 'Condition'  # Type of neuroimaging data
VoxelsType = 'PNC'  # Voxel selection method
percentile = 95  # Percentile threshold for voxel selection

def load_mat_data(file_path, var_name):
    """
    Load MATLAB data file and extract specified variable
    Handles both standard .mat files and HDF5-based v7.3 format
    
    Args:
        file_path (str): Path to MATLAB data file
        var_name (str): Name of variable to extract
    
    Returns:
        np.array: Extracted data as NumPy array
    """
    try:
        # Attempt to load using scipy for standard MAT files
        data = sio.loadmat(file_path)
    except NotImplementedError:
        # Fall back to h5py for v7.3 MAT files
        try:
            with h5py.File(file_path, 'r') as f:
                dataset = f[var_name]
                # Transpose to match MATLAB's dimension ordering
                data = np.array(dataset).astype(np.float32).T  
        except KeyError:
            raise KeyError(f"Variable '{var_name}' not found in file")
        except Exception as e:
            raise IOError(f"Error reading v7.3 file: {str(e)}")
    else:
        # Extract data from standard MAT file
        if var_name not in data:
            raise KeyError(f"Variable '{var_name}' not found in file")
        data = data[var_name].astype(np.float32)
    
    return data

def pearsonr_parallel(data, behavior):
    """
    Compute Pearson correlation coefficients in parallel
    across all voxels in brain data
    
    Args:
        data (np.array): Brain data matrix (samples × voxels)
        behavior (np.array): Behavioral data vector
    
    Returns:
        tuple: (correlation coefficients, p-values)
    """
    n_voxels = data.shape[1]
    # Distribute computation across cores
    results = Parallel(n_jobs=N_JOBS)(
        delayed(stats.pearsonr)(data[:, i], behavior) 
        for i in range(n_voxels)
    )
    # Separate correlation coefficients and p-values
    r_values = np.array([r for r, _ in results])
    p_values = np.array([p for _, p in results])
    return r_values, p_values


# Main execution block
if __name__ == "__main__":
    # Define experimental conditions and plotting colors
    stim_types = ['Body','Face','Place','Tool','All']  # Stimulus categories
    colors = ['#FFBC00','#92D050','#A076A1','#F58A83','#38459B']  # Color codes for each condition
    
    # Initialize results dictionary to store analysis outputs
    results = {
        'r_values': [],  # Pearson correlation coefficients
        'p_values': [],  # Uncorrected p-values
        'rmse_values': [],  # Root mean squared errors
        'actual_behav': [],  # Actual behavioral scores
        'predicted_behav': [],  # Predicted behavioral scores
        'pos_index': [],  # Positive effect indices (unused in current implementation)
        'neg_index': [],  # Negative effect indices (unused in current implementation)
        'data_types': [],  # Data type tags
        'beha_types': [],  # Behavioral metric tags
        'stim_types': [],  # Stimulus category tags
        'patterns': [],  # Analysis pattern tags
        'q_values': [],  # FDR-corrected q-values
        'colors': [],  # Color codes for plotting
        'weights': [],  # Model weights (unused in current implementation)
        'percentile': []  # Voxel selection percentile
    }

    # Loop through each behavioral metric (Accuracy, d-prime, Reaction Time)
    for beha_type in beha_types:
        # Create output directory for current behavioral metric
        save_path = os.path.join(save_path_base, beha_type)
        save_dir = os.path.dirname(save_path) if "." in os.path.basename(save_path) else save_path
        os.makedirs(save_dir, exist_ok=True)
        
        # Load behavioral data for current metric
        beha_data = load_mat_data(beha_path, f'{beha_type}')

        # Load thresholded neuroimaging map for voxel selection
        VWMCM_dat_temp = load_mat_data(VWMCM_dat_file, 'VWMCM_FDR_05_SVC_Conda_linear_boot10000')
        VWMCM_dat = VWMCM_dat_temp[~np.isnan(VWMCM_dat_temp).flatten()]  # Remove NaN values

        # Determine threshold for top voxel selection
        threshold = np.percentile(VWMCM_dat[VWMCM_dat != 0], percentile) 
        # Create binary mask of selected voxels
        selected_voxels = (VWMCM_dat > threshold).flatten()  
            
        # Process each stimulus category
        for i, stim in enumerate(stim_types):
            print(f"Processing {stim} for {beha_type}...")

            # 1. Load fMRI beta difference data for current stimulus category
            diff_data = load_mat_data(
                os.path.join(Feature_path, data_type, f"{stim}_response_2bk0bk_6mm.mat"),
                f"{stim}_response"
            )
            
            # Apply voxel selection mask
            diff_data = diff_data[:, selected_voxels]
            
            # 2. Calculate behavioral contrast
            # Behavior data structure: first 4 columns are baselines, next 4 are conditions
            col_idx = stim_types.index(stim)
            behavior_diff = (beha_data[:, 4 + col_idx] - beha_data[:, col_idx])
            
            # Set features and target
            feature = diff_data
                
            # 3. Perform repeated k-fold cross-validation regression
            # Using 100 repeats of 10-fold CV to predict behavior from brain data
            # Only using first 561 subjects (to match HCP dataset size)
            y_true, y_pred, r_val, p_val, rmse = repeated_kfold_final_corr(
                feature[0:561, :], 
                behavior_diff[0:561], 
                n_repeats=100, 
                n_splits=10,
                n_jobs=-1
            )
            
            # Store results for current stimulus-behavior combination
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
        
    # Perform Benjamini-Hochberg FDR correction on p-values
    p_values = np.array(results['p_values'])
    reject, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh')
    results['q_values'] = pvals_corrected
    
    # Save all results to a pickle file
    with open(os.path.join(save_path, 'All_HCP_Results_6mm_topbottom5_linear_combination_training.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    # Print and visualize results for each condition
    print("\nFinal Results Summary:")
    # Iterate through each result entry
    for index, (stim, r, p, q_value, rmse, x, y, beha_type, color) in enumerate(zip(
        results['stim_types'], 
        results['r_values'], 
        results['p_values'], 
        results['q_values'],
        results['rmse_values'], 
        results['actual_behav'], 
        results['predicted_behav'],
        results['beha_types'],
        results['colors']
    )):                                             
        # Print statistical results
        print(f"{stim} ({beha_type}): r = {r:.4f}, p = {p:.4f}, q = {q_value:.4f}, RMSE = {rmse:.4f}")
        
        # Ensure output directory exists
        save_path = os.path.join(save_path_base, beha_type)
        save_dir = os.path.dirname(save_path) if "." in os.path.basename(save_path) else save_path
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate correlation scatter plot
        plot_corr_scatter(
            x=x,  # Actual behavioral scores
            y=y,  # Predicted behavioral scores
            q_value=q_value,  # FDR-corrected p-value
            color=color,  # Category-specific color
            xlabel=f'Actual {beha_type}',  # X-axis label
            ylabel=f'Predicted {beha_type}',  # Y-axis label
            save_path=os.path.join(save_path, f'{stim}_predicted_results_{beha_type}_topbottom_6mm_training.png'),
            bins=50,  # Number of bins for density estimation
            confidence_level=0.95  # Confidence interval level
        )
