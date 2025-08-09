# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 19:05:19 2025

@author: ronglong.xiong
"""
# MATLAB Data Loader with Dual Format Support

import h5py  # For HDF5 file handling (MATLAB v7.3+)
import numpy as np
import scipy.io as sio  # For standard MATLAB file handling

def load_mat_data(file_path, var_name):
    """
    Load MATLAB data file and extract specified variable
    Supports both standard (.mat) and HDF5-based (v7.3) formats
    
    Args:
        file_path: Path to MATLAB data file
        var_name: Name of target variable to extract
    
    Returns:
        NumPy array containing variable data (float32)
    
    Raises:
        KeyError: If specified variable not found
        IOError: For file reading failures
    """
    try:
        # Attempt to load using scipy (for standard MAT files)
        data = sio.loadmat(file_path)
    except NotImplementedError:
        # Handle HDF5-based v7.3 format
        try:
            with h5py.File(file_path, 'r') as f:
                dataset = f[var_name]  # Access target dataset
                # Transpose to match MATLAB's column-major order
                data = np.array(dataset).astype(np.float32).T  
        except KeyError:
            raise KeyError(f"Variable '{var_name}' not found")
        except Exception as e:
            raise IOError(f"v7.3 read error: {str(e)}")
    else:
        # Handle standard MAT file data
        if var_name not in data:
            raise KeyError(f"Variable '{var_name}' not found")
        # Extract and convert to float32
        data = data[var_name].astype(np.float32)
    
    return data
