# Searchlight Analysis for condition-wise fMRI Data Classification

# Import required libraries
import numpy as np
import nibabel as nib  # Neuroimaging data handling
from nilearn.decoding import SearchLight  # Searchlight analysis
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.model_selection import StratifiedKFold  # Cross-validation
from nilearn import plotting  # Brain visualization
from sklearn.model_selection import RepeatedKFold
# Load fMRI data and brain mask
fmri_img = nib.load('fmri_data.nii')  # 4D fMRI data (space × time)
mask_img = nib.load('brain_mask.nii.gz')  # Brain parcellation map

# Output configuration
output_file = 'searchlight_results.nii.gz'  # Results file path
radius = 5  # Searchlight sphere radius (voxels)
n_jobs = -1  # Use all CPU cores

# Create class labels (4 conditions repeated for 561 subjects)
labels = np.array([1,2,3,4]*561)  # Conditions: Body, Face, Place, Tool

# Configure stratified 10-fold cross-validation
#cv = StratifiedKFold(n_splits=10)  # Balanced class distribution in folds
cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)


# Process brain mask to binary format
mask_data = mask_img.get_fdata()  # Load voxel data
binary_mask = (mask_data >= 1).astype(np.int8)  # Convert to binary (0/1)
binary_mask_img = nib.Nifti1Image(binary_mask, mask_img.affine, mask_img.header)

# Initialize searchlight analysis
searchlight = SearchLight(
    mask_img=binary_mask_img,  # Brain mask defining analysis region
    radius=radius,  # Spatial extent of analysis spheres
    estimator=SVC(kernel="linear"),  # Linear classifier
    n_jobs=n_jobs,  # Parallel processing
    cv=cv,  # Cross-validation scheme
    verbose=2  # Detailed progress reporting
)

# Run searchlight analysis
searchlight.fit(fmri_img, labels)  # Map brain regions predictive of conditions

# Save results as brain map
result_img = nib.Nifti1Image(searchlight.scores_, affine=mask_img.affine)
nib.save(result_img, output_file)  # Save classification accuracy map

# Visualize results on glass brain
plotting.plot_glass_brain(
    result_img,
    title="Condition Classification Accuracy",
    display_mode="lyrz",  # Multiple projection views
    vmin=0.25,  # Minimum display value
    threshold=0.25,  # Significance threshold
    plot_abs=False,  # Show negative values
    colorbar=True,  # Display color scale
    cmap='RdBu_r'  # Color map (red-blue reversed)
)
plotting.show()  # Display plot
