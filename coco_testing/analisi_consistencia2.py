import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

masks_dir = '/home/gcordoba/snapshots/kits/masks_train'
images_dir = '/data/users/gcordoba/kits/imagesTr'
labels_dir = '/data/users/gcordoba/kits/labelsTr'
"""
masks_dir = '/home/gcordoba/snapshots/dogs/masks'
images_dir = '/data/users/gcordoba/coco_data/imagesTr'
labels_dir = '/data/users/gcordoba/coco_data/labelsTr'
"""
def load_nifti_image(file_path):
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata()

def calculate_mask_properties(mask):
    non_zero_coords = np.argwhere(mask)
    if non_zero_coords.size == 0:
        return None, None  # Retornem None si la màscara és buida
    min_coords = non_zero_coords.min(axis=0)
    max_coords = non_zero_coords.max(axis=0)
    dimensions = max_coords - min_coords + 1
    center = (min_coords + max_coords) / 2
    return dimensions, center

def analyze_all_cases():
    dimensions_list = []
    centers_list = []
    cases = []
    for mask_file in os.listdir(masks_dir):
        if mask_file.endswith('.nii.gz'):
            case_name = mask_file.replace('mask_', '').replace('.nii.gz', '')
            mask_path = os.path.join(masks_dir, mask_file)
            mask = load_nifti_image(mask_path)
            dimensions, center = calculate_mask_properties(mask)
            if dimensions is not None and center is not None:
                dimensions_list.append(dimensions)
                centers_list.append(center)
                cases.append(case_name)
    return dimensions_list, centers_list, cases

def plot_variability(dimensions, centers, title, num_new_points=150):
    dimensions = np.array(dimensions)
    centers = np.array(centers)
    mean_dimensions, std_dimensions, mean_centers, std_centers = calculate_consistency(dimensions, centers)
    new_dimensions = np.random.normal(loc=mean_dimensions, scale=std_dimensions, size=(150, 3))
    new_centers = np.random.normal(loc=mean_centers, scale=std_centers, size=(150, 3))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Dimensions
    axes[0].scatter(dimensions[:, 0], dimensions[:, 1], c='r', label='Casos de test', edgecolor='black')
    axes[0].scatter(new_dimensions[:, 0], new_dimensions[:, 1], facecolors='none', label='Casos de train', edgecolors='r', alpha=0.7)
    axes[0].set_xlabel('Alçada')
    axes[0].set_ylabel('Llargada')
    axes[0].legend()
    axes[0].set_title('Dimensions')

    # Centres
    axes[1].scatter(centers[:, 0], centers[:, 1], c='r', label='Centre XY', edgecolor='black')
    axes[1].scatter(new_centers[:, 0], new_centers[:, 1], facecolors='none', edgecolors='r', alpha=0.7)
    axes[1].set_xlabel('Centre X')
    axes[1].set_ylabel('Centre Y')
    axes[1].legend()
    axes[1].set_title('Centres')

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    print("len(Train) = ", len(new_dimensions))
    print("len(Test) = ", len(dimensions))

def calculate_consistency(dimensions, centers):
    dimensions = np.array(dimensions)
    centers = np.array(centers)
    mean_dimensions = dimensions.mean(axis=0)
    std_dimensions = dimensions.std(axis=0)
    mean_centers = centers.mean(axis=0)
    std_centers = centers.std(axis=0)
    return mean_dimensions, std_dimensions, mean_centers, std_centers

all_dimensions, all_centers, all_cases = analyze_all_cases()

plot_variability(all_dimensions, all_centers, 'Variabilitat d\'escales - Tots els casos (test i train (color difuminat))')

all_mean_dim, all_std_dim, all_mean_center, all_std_center = calculate_consistency(all_dimensions, all_centers)

print("Tots els casos - Dimensions (mitjana ± desviació estàndard):", all_mean_dim, "±", all_std_dim)
print("Tots els casos - Centres (mitjana ± desviació estàndard):", all_mean_center, "±", all_std_center)