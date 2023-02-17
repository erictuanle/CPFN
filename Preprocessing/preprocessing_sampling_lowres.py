# Importation of packages
import os
import h5py
import time
import numba
import shutil
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from joblib import Parallel, delayed

# Furthest point sampling code
@numba.jit(numba.int32[:](numba.float32[:, :], numba.int32[:], numba.int32), nopython=True)
def furthest_point_sampling(input_points, index_query_points1, nb_query_points):
    num_points, _ = input_points.shape
    index_query_points2 = np.zeros(nb_query_points, dtype=numba.int32)
    min_distances = 10 ** 6 * np.ones(num_points, dtype=numba.float64)
    min_distances[index_query_points1] = 0
    index = np.argmax(min_distances)
    for i in range(nb_query_points):
        index_query_points2[i] = index
        additional_distances = np.sqrt(np.sum((input_points - input_points[index]) ** 2, axis=1))
        min_distances = np.minimum(min_distances, additional_distances)
        index = np.argmax(min_distances)
    return index_query_points2

@numba.jit(numba.int32[:](numba.float32[:, :], numba.int32[:]), nopython=True)
def furthest_point_sampling_per_label(input_points, labels):
    num_points, _ = input_points.shape
    unique_labels = np.unique(labels)
    index_query_points = np.zeros(len(unique_labels), dtype=numba.int32)
    min_distances = 10 ** 6 * np.ones(num_points, dtype=numba.float64)
    index = np.random.randint(0, num_points)
    for i in range(len(unique_labels)):
        label = labels[index]
        index_query_points[i] = index
        additional_distances = np.sqrt(np.sum((input_points - input_points[index]) ** 2, axis=1))
        min_distances = np.minimum(min_distances, additional_distances)
        min_distances[labels==label] = 0
        index = np.argmax(min_distances)
    return index_query_points

# Furthest point sampling per labels code
def multiprocessing_sampling(input_tuple):
    ind_file, file_, nb_query_points, path_lowres, path_highres = input_tuple
    object_filename = file_.replace('.h5', '')
    if ind_file%100==0: print('%d / %d' % (ind_file, nfiles))
    # Loading the GT data
    try:
    	with h5py.File(os.path.join(path_highres, object_filename + '.h5'), 'r') as f:
            gt_points = f['gt_points'][()].astype(np.float32)
            noisy_points = f['noisy_points'][()].astype(np.float32)
            gt_labels = f['gt_labels'][()].astype(np.int32)
            gt_normals = f['gt_normals'][()].astype(np.float32)
            primitives = {}
            nb_labels = gt_labels.max() + 1
            for i in range(nb_labels):
                key = object_filename + '_soup_' + str(i)
                primitives[key] = {'gt_points': f[key]['gt_points'][()],
                                    'gt_normals': f[key]['gt_normals'][()],
                                    'meta': f[key].attrs['meta'].copy()}
    except:
        return
    index_query_points1 = furthest_point_sampling_per_label(gt_points, gt_labels)
    index_query_points2 = furthest_point_sampling(gt_points, index_query_points1, nb_query_points)
    index_query_points = np.concatenate((index_query_points1, index_query_points2))
    assert(len(np.unique(gt_labels)) == len(np.unique(gt_labels[index_query_points])))
    with h5py.File(os.path.join(path_lowres, object_filename + '.h5'), 'w') as f:
        f.create_dataset('gt_points', data=gt_points[index_query_points])
        f.create_dataset('gt_normals', data=gt_normals[index_query_points])
        f.create_dataset('gt_labels', data=gt_labels[index_query_points])
        f.create_dataset('noisy_points', data=noisy_points[index_query_points])
        f.create_dataset('index_query_points', data=index_query_points)
        for key in primitives.keys():
            grp = f.create_group(key)
            grp['gt_points'] = primitives[key]['gt_points']
            grp['gt_normals'] = primitives[key]['gt_normals']
            grp.attrs['meta'] = primitives[key]['meta']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_highres', help='Path to Highres h5 files', type=str, default='data/TraceParts_v2/')
    parser.add_argument('--path_lowres', help='Path to Highres h5 files', type=str, default='data/TraceParts_v2_lowres/')
    parser.add_argument('--path_split_file', help='Path to the csv file for the corresponding split', type=str, default='Dataset/train_models.csv')
    parser.add_argument('--nb_query_points', help='Number of Query Points', type=int, default=8192)
    parser.add_argument('--ratio_cpu_touse', help='Ratio of the Total number of CPUs to use', type=float, default=0.70)
    args = parser.parse_args()

    # Path    
    files = pd.read_csv(args.path_split_file, header=None).values[:,0]
    nfiles = len(files)
    if not os.path.isdir(args.path_lowres):
        os.mkdir(args.path_lowres)

    # Multiprocessing
    num_cores = int(args.ratio_cpu_touse * mp.cpu_count())
    results = Parallel(n_jobs=num_cores)(delayed(multiprocessing_sampling)((i, file_, args.nb_query_points, args.path_lowres, args.path_highres)) for i, file_ in enumerate(files))