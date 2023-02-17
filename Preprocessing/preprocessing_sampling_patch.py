# Importation of packages
import os
import h5py
import numba
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from joblib import Parallel, delayed

def get_small_primitives(gt_labels_hr, max_nb_points):
    unique_labels, unique_counts = np.unique(gt_labels_hr, return_counts=True)
    small_primitives_pool = np.where(unique_counts < max_nb_points)[0]
    small_primitives_id = unique_labels[small_primitives_pool]
    return small_primitives_id

def extract_pool_indices(gt_points_lr, gt_labels_lr, small_primitives_id):
    pool_indices = np.where(np.isin(gt_labels_lr, small_primitives_id))[0]
    pool_labels = gt_labels_lr[pool_indices]
    return pool_indices, pool_labels

def sample(gt_points_lr, gt_points_hr, pool_indices, pool_labels, num_points_patch=8192, max_number_patches=32):
    list_patch_indices = []
    while (len(list_patch_indices) < max_number_patches) and (len(pool_indices) != 0):
        # Selecting the remaining labels
        unique_pool_labels = np.unique(pool_labels)
        for label in unique_pool_labels:
            # Checking if the maximum number of patches have been reached
            if len(list_patch_indices) >= max_number_patches:
                break
            # Selecting a random pool index for label l
            ind_pool_indices = np.where(pool_labels==label)[0]
            if len(ind_pool_indices) == 0:
                continue
            i = pool_indices[np.random.choice(ind_pool_indices)]
            # Getting the patch indices for that query points
            distances = np.linalg.norm(np.expand_dims(gt_points_lr[i], axis=0) - gt_points_hr, axis=1)
            patch_indices = np.argsort(distances)[:num_points_patch]
            list_patch_indices.append(patch_indices)
            patch_distances = np.sort(distances)[:num_points_patch]
            # Deleting the neighbours in the pool of indices
            distances = np.linalg.norm(np.expand_dims(gt_points_lr[i], axis=0) - gt_points_lr[pool_indices], axis=1)
            pool_indices_selected = np.where(distances <= np.max(patch_distances))[0]
            pool_indices = np.delete(pool_indices, pool_indices_selected)
            pool_labels = np.delete(pool_labels, pool_indices_selected)
    patch_indices = np.stack(list_patch_indices, axis=0)
    return patch_indices

def multiprocessing(tuple):
    i, n, file_, max_number_patches, num_points_patch, scale, path_lowres, path_highres, path_patches = tuple
    if i%100==0: print('Processing File (%d / %d): '%(i, n), file_)
    with h5py.File(os.path.join(path_highres, file_), 'r') as f:
        gt_points_hr = f['gt_points'][()]
        gt_labels_hr = f['gt_labels'][()]
    with h5py.File(os.path.join(path_lowres, file_), 'r') as f:
        gt_points_lr = f['gt_points'][()]
        gt_labels_lr = f['gt_labels'][()]
    nb_points, _ = gt_points_hr.shape
    max_nb_points = int(scale * nb_points)
    small_primitives_id = get_small_primitives(gt_labels_hr, max_nb_points=max_nb_points)
    pool_indices, pool_labels = extract_pool_indices(gt_points_lr, gt_labels_lr, small_primitives_id)
    if len(pool_indices) == 0:
        return
    patch_indices = sample(gt_points_lr, gt_points_hr, pool_indices, pool_labels, num_points_patch=num_points_patch, max_number_patches=max_number_patches)
    np.save(os.path.join(path_patches, file_.replace('.h5', '_indices.npy')), patch_indices)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_highres', help='Path to Highres h5 files', type=str, default=os.path.expanduser('data/TraceParts_v2/'))
    parser.add_argument('--path_lowres', help='Path to Highres h5 files', type=str, default=os.path.expanduser('data/TraceParts_v2_lowres/'))
    parser.add_argument('--path_patches', help='Path to Sampled Patches h5 files', type=str, default=os.path.expanduser('data/TraceParts_v2_patches/'))
    parser.add_argument('--path_split_file', help='Path to the csv file for the corresponding split', type=str, default='Dataset/train_models.csv')
    parser.add_argument('--scale', help='Scale to select the smallest primitive', type=float, default=0.05)
    parser.add_argument('--max_number_patches', help='Maximum number of patch', type=int, default=32)
    parser.add_argument('--num_points_patch', help='Number of points per patch', type=int, default=8192)
    parser.add_argument('--ratio_cpu_touse', help='Ratio of the Total number of CPUs to use', type=float, default=0.70)
    args = parser.parse_args()

    path_patches = os.path.join(args.path_patches, str(round(args.scale,2)))
    if not os.path.isdir(path_patches):
        os.makedirs(path_patches, exist_ok=True)

    nb_cores = int(args.ratio_cpu_touse * mp.cpu_count())
    list_files = pd.read_csv(args.path_split_file, header=None).values[:,0]
    n_files = len(list_files)
    results = Parallel(n_jobs=nb_cores)(delayed(multiprocessing)((i, n_files, file_, args.max_number_patches, args.num_points_patch, args.scale, args.path_lowres, args.path_highres, path_patches)) for i, file_ in enumerate(list_files))