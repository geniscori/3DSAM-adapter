import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

"""
masks_dir = '/home/gcordoba/snapshots/coco_data/masks_test'
images_dir = '/data/users/gcordoba/coco_data/imagesTr'
labels_dir = '/data/users/gcordoba/coco_data/labelsTr'
"""
masks_dir = '/home/gcordoba/snapshots/colon/masks_train'
images_dir = '/data/users/gcordoba/colon/imagesTr'
labels_dir = '/data/users/gcordoba/colon/labelsTr'

def load_nifti_image(file_path):
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata()

def plot_comparison(img, ground_truth_mask, predicted_mask, title=None):
    non_empty_slices = np.where(ground_truth_mask.sum(axis=(0, 1)) > 0)[0]
    num_slices_to_show = min(len(non_empty_slices), 5)  # Limita el nombre de capes a mostrar

    fig, axes = plt.subplots(3, num_slices_to_show, figsize=(20, 10))
    for i, slice_idx in enumerate(non_empty_slices[:num_slices_to_show]):
        axes[0, i].imshow(np.rot90(img[:, :, slice_idx]), cmap="gray")
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Imatge capa {slice_idx}")
        axes[1, i].imshow(np.rot90(ground_truth_mask[:, :, slice_idx]), cmap="gray")
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Ground Truth capa {slice_idx}")
        axes[2, i].imshow(predicted_mask[slice_idx, :, :], cmap="gray")
        axes[2, i].axis('off')
        axes[2, i].set_title(f"Màscara predita capa {slice_idx}")
    if title:
        fig.suptitle(title)
    plt.show()

def visualize_cases(cases, title):
    for case in cases:
        img_path = os.path.normpath(os.path.join(images_dir, f'{case}.nii.gz')).replace('\\','/')
        label_path = os.path.normpath(os.path.join(labels_dir, f'{case}.nii.gz')).replace('\\','/')
        mask_path = os.path.normpath(os.path.join(masks_dir, f'{case}.nii.gz')).replace('\\','/')

        img = load_nifti_image(img_path)
        ground_truth_mask = load_nifti_image(label_path)
        predicted_mask = load_nifti_image(mask_path)

        plot_comparison(img, ground_truth_mask, predicted_mask, title=f'{title} - {case}')


best_cases = ['colon_136', 'colon_218', 'colon_142']
worst_cases = ['colon_104', 'colon_129', 'colon_181']
visualize_cases(best_cases, 'Millors resultats')
visualize_cases(worst_cases, 'Pitjors resultats')
