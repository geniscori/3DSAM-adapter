import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

masks_dir = '/home/gcordoba/snapshots/kits/masks_train'
images_dir = '/data/users/gcordoba/kits/imagesTr'
labels_dir = '/data/users/gcordoba/kits/labelsTr'

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

def plot_variability(dimensions, centers, title):
    dimensions = np.array(dimensions)
    centers = np.array(centers)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(dimensions[:, 0], dimensions[:, 1], c='r', label='Alçada vs Llargada')
    axes[0].scatter(dimensions[:, 0], dimensions[:, 2], c='g', label='Alçada vs Profunditat')
    axes[0].scatter(dimensions[:, 1], dimensions[:, 2], c='b', label='Llargada vs Profunditat')
    axes[0].set_xlabel('Dimensió 1')
    axes[0].set_ylabel('Dimensió 2')
    axes[0].legend()
    axes[0].set_title('Dimensions')

    axes[1].scatter(centers[:, 0], centers[:, 1], c='r', label='Centre XY')
    axes[1].scatter(centers[:, 0], centers[:, 2], c='g', label='Centre XZ')
    axes[1].scatter(centers[:, 1], centers[:, 2], c='b', label='Centre YZ')
    axes[1].set_xlabel('Centre 1')
    axes[1].set_ylabel('Centre 2')
    axes[1].legend()
    axes[1].set_title('Centres')

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def calculate_consistency(dimensions, centers):
    dimensions = np.array(dimensions)
    centers = np.array(centers)
    mean_dimensions = dimensions.mean(axis=0)
    std_dimensions = dimensions.std(axis=0)
    mean_centers = centers.mean(axis=0)
    std_centers = centers.std(axis=0)
    return mean_dimensions, std_dimensions, mean_centers, std_centers
"""
all_dimensions_train, all_centers_train, all_cases_train = analyze_all_cases()

plot_variability(all_dimensions_train, all_centers_train, 'Variabilitat d\'escales - Tots els casos de train')

train_mean_dim, train_std_dim, train_mean_center, train_std_center = calculate_consistency(all_dimensions_train, all_centers_train)

print("Tots els casos de train - Dimensions (mitjana ± desviació estàndard):", train_mean_dim, "±", train_std_dim)
print("Tots els casos de train - Centres (mitjana ± desviació estàndard):", train_mean_center, "±", train_std_center)

"""
all_dimensions, all_centers, all_cases = analyze_all_cases()

plot_variability(all_dimensions, all_centers, 'Variabilitat d\'escales casos de train: colon')

all_mean_dim, all_std_dim, all_mean_center, all_std_center = calculate_consistency(all_dimensions, all_centers)

print("Tots els casos de test - Dimensions (mitjana ± desviació estàndard):", all_mean_dim, "±", all_std_dim)
print("Tots els casos de test - Centres (mitjana ± desviació estàndard):", all_mean_center, "±", all_std_center)

