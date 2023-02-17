# Importatiom of packages
import os
import re
import sys
import h5py
import pickle
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from joblib import Parallel, delayed

def multiprocessing(tuple):
    ind_file, n_file, file_, path_lowres, path_highres, path_features, path_patches, num_points = tuple
    if ind_file%100==0: print('%d / %d'%(ind_file, n_file))
    if not os.path.isfile(os.path.join(path_patches, file_.replace('.h5', '_indices.npy'))):
        return
    patch_indices = np.load(os.path.join(path_patches, file_.replace('.h5', '_indices.npy')))
    nb_patches, _ = patch_indices.shape
    with h5py.File(os.path.join(path_highres, file_), 'r') as f:
        P = f['gt_points'][()].astype(np.float32)
        P_noisy = f['noisy_points'][()].astype(np.float32)
        normal_gt = f['gt_normals'][()].astype(np.float32)
        I_gt = f['gt_labels'][()].astype(np.int64)
    with h5py.File(os.path.join(path_lowres, file_), 'r') as f:
        index_query_points = f['index_query_points'][()]
        # Primitive keys
        found_soup_ids = []
        soup_id_to_key = {}
        soup_prog = re.compile('(.*)_soup_([0-9]+)$')
        for key in list(f.keys()):
            m = soup_prog.match(key)
            if m is not None:
                soup_id = int(m.group(2))
                found_soup_ids.append(soup_id)
                soup_id_to_key[soup_id] = key
        found_soup_ids.sort()
        n_instances = len(found_soup_ids)
        instances = []
        P_gt = []
        N_gt = []
        metas = []
        for i in range(n_instances):
            g = f[soup_id_to_key[i]]
            P_gt_cur = g['gt_points'][()]
            P_gt.append(P_gt_cur)
            N_gt_cur = g['gt_normals'][()]
            N_gt.append(N_gt_cur)
            meta = pickle.loads(g.attrs['meta'])
            metas.append(meta)
        P_gt = np.array(P_gt)
        N_gt = np.array(N_gt)
    # Patch Selection
    P = np.reshape(P[patch_indices.flatten()], [nb_patches, num_points, 3])
    P_noisy = np.reshape(P_noisy[patch_indices.flatten()], [nb_patches, num_points, 3])
    normal_gt = np.reshape(normal_gt[patch_indices.flatten()], [nb_patches, num_points, 3])
    I_gt = np.reshape(I_gt[patch_indices.flatten()], [nb_patches, num_points])
    # Normalisation
    mean = np.mean(P, axis=1, keepdims=True)
    P = P - mean
    norm = np.linalg.norm(P, axis=2, keepdims=True).max(axis=1, keepdims=True)
    P = P / norm
    P_noisy = P_noisy - mean
    P_noisy = P_noisy / norm
    P_gt = P_gt - np.expand_dims(mean, axis=1)
    P_gt = P_gt / np.expand_dims(norm, axis=1)
    flag = ~np.all(P_gt == - np.expand_dims(mean, axis=1) / np.expand_dims(norm, axis=1), axis=3, keepdims=True).all(axis=2, keepdims=True)
    P_gt = P_gt * flag.astype(np.float32)
    # SPFN Feature
    glob_features = np.load(os.path.join(path_features, file_.replace('.h5', ''), 'global_feat.npy'))
    loc_features = np.load(os.path.join(path_features, file_.replace('.h5', ''), 'local_feat.npy'))
    # Export
    if not os.path.isdir(os.path.join(path_patches, file_.replace('.h5', ''))):
        os.mkdir(os.path.join(path_patches, file_.replace('.h5', '')))
    for i in range(nb_patches):
        flag = -1 in I_gt[i]
        unique_values, inverse_values = np.unique(I_gt[i], return_inverse=True)
        unique_values = unique_values[unique_values != -1]
        if flag: inverse_values = inverse_values - 1

        with h5py.File(os.path.join(path_patches, file_.replace('.h5', ''), file_.replace('.h5','_patch%d.h5'%i)), 'w') as f:
            f.create_dataset('gt_points', data=P[i].astype(np.float32))
            f.create_dataset('gt_normals', data=normal_gt[i].astype(np.float32))
            f.create_dataset('gt_labels', data=inverse_values.astype(np.int64))
            f.create_dataset('noisy_points', data=P_noisy[i].astype(np.float32))
            f.create_dataset('glob_features', data=glob_features.astype(np.float32))
            f.create_dataset('loc_features', data=loc_features[:,i].astype(np.float32))
            for j, value in enumerate(unique_values):
                key = file_.replace('.h5','_soup_%d'%value)
                new_key = file_.replace('.h5','_soup_%d'%j)
                grp = f.create_group(new_key)
                grp['gt_points'] = P_gt[i][value].astype(np.float32)
                grp['gt_normals'] = N_gt[value].astype(np.float32)
                if metas[value]['type'] == 'plane':
                    metas[value]['location_x'] = str((float(metas[value]['location_x']) - mean[i,0,0]) / norm[i,0,0])
                    metas[value]['location_y'] = str((float(metas[value]['location_y']) - mean[i,0,1]) / norm[i,0,0])
                    metas[value]['location_z'] = str((float(metas[value]['location_z']) - mean[i,0,2]) / norm[i,0,0])
                elif metas[value]['type'] == 'sphere':
                    metas[value]['location_x'] = str((float(metas[value]['location_x']) - mean[i,0,0]) / norm[i,0,0])
                    metas[value]['location_y'] = str((float(metas[value]['location_y']) - mean[i,0,1]) / norm[i,0,0])
                    metas[value]['location_z'] = str((float(metas[value]['location_z']) - mean[i,0,2]) / norm[i,0,0])
                    metas[value]['radius'] = str(float(metas[value]['radius'])/ norm[i, 0, 0])
                elif metas[value]['type'] == 'cylinder':
                    metas[value]['location_x'] = str((float(metas[value]['location_x']) - mean[i,0,0]) / norm[i,0,0])
                    metas[value]['location_y'] = str((float(metas[value]['location_y']) - mean[i,0,1]) / norm[i,0,0])
                    metas[value]['location_z'] = str((float(metas[value]['location_z']) - mean[i,0,2]) / norm[i,0,0])
                    metas[value]['radius'] = str(float(metas[value]['radius'])/ norm[i, 0, 0])
                elif metas[value]['type'] == 'cone':
                    metas[value]['apex_x'] = str((float(metas[value]['apex_x']) - mean[i,0,0]) / norm[i,0,0])
                    metas[value]['apex_y'] = str((float(metas[value]['apex_y']) - mean[i,0,1]) / norm[i,0,0])
                    metas[value]['apex_z'] = str((float(metas[value]['apex_z']) - mean[i,0,2]) / norm[i,0,0])
                grp.attrs['meta'] = str(metas[value])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_highres', help='Path to Highres h5 files', type=str, default=os.path.expanduser('data/TraceParts_v2/'))
    parser.add_argument('--path_lowres', help='Path to Lowres h5 files', type=str, default=os.path.expanduser('data/TraceParts_v2_lowres/'))
    parser.add_argument('--path_features', help='Path to SPFN Local and Global Features', type=str, default=os.path.expanduser('data/TraceParts_v2_globalspfn/'))
    parser.add_argument('--path_patches', help='Path to Sampled Patches h5 files', type=str, default=os.path.expanduser('data/TraceParts_v2_patches/'))
    parser.add_argument('--path_split_file', help='Path to the csv file for the corresponding split', type=str, default='Dataset/train_models.csv')
    parser.add_argument('--scale', help='Scale to select the smallest primitive', type=float, default=0.05)
    parser.add_argument('--max_number_patches', help='Maximum number of patch', type=int, default=32)
    parser.add_argument('--num_points_patch', help='Number of points per patch', type=int, default=8192)
    parser.add_argument('--ratio_cpu_touse', help='Ratio of the Total number of CPUs to use', type=float, default=0.70)
    args = parser.parse_args()

    path_patches = os.path.join(args.path_patches, str(round(args.scale,2)))
    
    files = np.sort(pd.read_csv(args.path_split_file, delimiter=',', header=None)[0])
    nfiles = len(files)
    nb_cores = int(args.ratio_cpu_touse * mp.cpu_count())
    results = Parallel(n_jobs=nb_cores)(delayed(multiprocessing)((i, nfiles, file_, args.path_lowres, args.path_highres, args.path_features, path_patches, args.num_points_patch)) for i, file_ in enumerate(files))