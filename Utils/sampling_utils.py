# Importation of packages
import numpy as np

def sample(gt_points_lr, gt_points_hr, pool_indices, num_points_patch=8192, max_number_patches=32):
    list_patch_indices = []
    while (len(list_patch_indices) < max_number_patches) and (len(pool_indices) != 0):
        # Selecting a random pool index for label l
        i = pool_indices[np.random.choice(len(pool_indices))]
        # Getting the patch indices for that query points
        distances = np.linalg.norm(np.expand_dims(gt_points_lr[i], axis=0) - gt_points_hr, axis=1)
        patch_indices = np.argsort(distances)[:num_points_patch]
        list_patch_indices.append(patch_indices)
        patch_distances = np.sort(distances)[:num_points_patch]
        # Deleting the neighbours in the pool of indices
        distances = np.linalg.norm(np.expand_dims(gt_points_lr[i], axis=0) - gt_points_lr[pool_indices], axis=1)
        pool_indices_selected = np.where(distances <= np.max(patch_distances))[0]
        pool_indices = np.delete(pool_indices, pool_indices_selected)
    patch_indices = np.stack(list_patch_indices, axis=0)
    return patch_indices