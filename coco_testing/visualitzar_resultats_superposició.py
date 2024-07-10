import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

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
    num_slices_to_show = min(len(non_empty_slices), 5)

    fig, axes = plt.subplots(2, num_slices_to_show, figsize=(20, 10))

    for i, slice_idx in enumerate(non_empty_slices[:num_slices_to_show]):
        gt_non_zero = np.argwhere(ground_truth_mask[:, :, slice_idx])
        minr, minc = gt_non_zero.min(axis=0)
        maxr, maxc = gt_non_zero.max(axis=0)
        margin = 10
        minr = max(minr - margin, 0)
        minc = max(minc - margin, 0)
        maxr = min(maxr + margin, ground_truth_mask.shape[0])
        maxc = min(maxc + margin, ground_truth_mask.shape[1])

        # Imatge original amb Ground Truth
        ax_gt = axes[0, i]
        ax_gt.imshow(np.rot90(img[:, :, slice_idx], 3), cmap="gray")
        ax_gt.imshow(np.rot90(ground_truth_mask[:, :, slice_idx], 0), cmap="jet", alpha=0.5)
        rect_gt = Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor='yellow', facecolor='none', linewidth=2)
        ax_gt.add_patch(rect_gt)
        ax_gt.axis('off')
        ax_gt.set_title(f"Ground Truth superposat capa {slice_idx}")

        # Zoom in la zona d'interès per Ground Truth
        axins_gt = zoomed_inset_axes(ax_gt, 3, loc='lower left')
        axins_gt.imshow(np.rot90(img[:, :, slice_idx], 3), cmap="gray")
        axins_gt.imshow(np.rot90(ground_truth_mask[:, :, slice_idx], 0), cmap="jet", alpha=0.5)
        axins_gt.set_xlim(minc, maxc)
        axins_gt.set_ylim(maxr, minr)
        axins_gt.axis('off')
        mark_inset(ax_gt, axins_gt, loc1=2, loc2=4, fc="none", ec="yellow", linewidth=2)
        axins_gt.add_patch(
            Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor='yellow', facecolor='none', linewidth=2))

        # Imatge original amb màscara predita
        ax_pred = axes[1, i]
        ax_pred.imshow(np.rot90(img[:, :, slice_idx], 3), cmap="gray")
        ax_pred.imshow(np.fliplr(np.rot90(predicted_mask[slice_idx, :, :], 3)), cmap="jet", alpha=0.5)
        rect_pred = Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor='yellow', facecolor='none', linewidth=2)
        ax_pred.add_patch(rect_pred)
        ax_pred.axis('off')
        ax_pred.set_title(f"Màscara predita superposada capa {slice_idx}")

        # Zoom in la zona d'interès per Màscara predita
        axins_pred = zoomed_inset_axes(ax_pred, 3, loc='lower left')
        axins_pred.imshow(np.rot90(img[:, :, slice_idx], 3), cmap="gray")
        axins_pred.imshow(np.fliplr(np.rot90(predicted_mask[slice_idx, :, :], 3)), cmap="jet", alpha=0.5)
        axins_pred.set_xlim(minc, maxc)
        axins_pred.set_ylim(maxr, minr)
        axins_pred.axis('off')
        mark_inset(ax_pred, axins_pred, loc1=2, loc2=4, fc="none", ec="yellow", linewidth=2)
        axins_pred.add_patch(
            Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor='yellow', facecolor='none', linewidth=2))

    if title:
        fig.suptitle(title)
    fig.tight_layout(pad=2.0)
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
visualize_cases(best_cases, 'Resultats per al volum:')
#visualize_cases(worst_cases, 'Pitjors resultats')
