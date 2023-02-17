# Importation of packages
import os
import re
import h5py
import torch
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.utils.data as data

# Importing Utils files
from Utils import dataset_utils

class Dataset_PatchSelection(data.Dataset):
    def __init__(self, csv_path, lowres_folder, highres_folder, scale, n_points=None, normalisation=True):
        self.lowres_folder = lowres_folder
        self.highres_folder = highres_folder
        self.scale = scale
        self.n_points = n_points
        self.normalisation = normalisation
        csv_raw = pd.read_csv(csv_path, delimiter=',', header=None)[0]
        self.hdf5_file_list = np.sort([file_ for file_ in csv_raw])
        self.hdf5_file_list_lowres = [os.path.join(self.lowres_folder, file_.split('.')[0] + '.h5') for file_ in self.hdf5_file_list]
        self.hdf5_file_list_highres = [os.path.join(self.highres_folder, file_.split('.')[0] + '.h5') for file_ in self.hdf5_file_list]
        self.n_data = len(self.hdf5_file_list)
        self.preload_dataset()
    def preload_dataset(self):
        self.list_points = []
        self.list_output_labels = []
        self.list_shuffled_indices = []
        print('Preloading Dataset:')
        for i in tqdm(range(self.n_data)):
            points, output_labels, shuffled_indices = dataset_utils.create_unit_data_from_hdf5_patch_selection(self.hdf5_file_list_lowres[i], self.hdf5_file_list_highres[i], normalisation=self.normalisation, scale=self.scale, n_points=self.n_points)
            self.list_points.append(points)
            self.list_output_labels.append(output_labels)
            self.list_shuffled_indices.append(shuffled_indices)
    def __getitem__(self, index):
        # find shape that contains the point with given global index
        points = self.list_points[index]
        points = torch.from_numpy(points).float()
        output_labels = self.list_output_labels[index]
        output_labels = torch.from_numpy(output_labels).long()
        shuffled_indices = self.list_shuffled_indices[index]
        shuffled_indices = torch.from_numpy(shuffled_indices).long()
        return points, output_labels, shuffled_indices
    def __len__(self):
        return self.n_data

class Dataset_GlobalSPFN(data.Dataset):
    def __init__(self, n_max_global_instances, csv_path, lowres_folder, highres_folder, path_patches, noisy, n_points=8192, test=False, first_n=-1, fixed_order=False):
        self.n_max_global_instances = n_max_global_instances
        self.lowres_folder = lowres_folder
        self.highres_folder = highres_folder
        if not test:
            self.dir_files = self.lowres_folder
            self.path_patches = None
        else:
            self.dir_files = self.highres_folder
            self.path_patches = path_patches
        self.noisy = noisy
        self.n_points = n_points
        self.test = test
        self.first_n = first_n
        self.fixed_order = fixed_order
        csv_raw = pd.read_csv(csv_path, delimiter=',', header=None)[0]
        self.hdf5_file_list = np.sort(csv_raw)
        if not fixed_order:
            random.shuffle(self.hdf5_file_list)
        if first_n != -1:
            self.hdf5_file_list = self.hdf5_file_list[:first_n]
        self.n_data = len(self.hdf5_file_list)
        if not self.test:
            self.preload_dataset()
    def preload_dataset(self):
        print(f'Preloading Dataset:')
        for index in tqdm(range(self.__len__())):
            data_elt = self.fetch_data_at_index(index)
            if not hasattr(self, 'data_matrix'):
                self.data_matrix = {}
                for key in data_elt.keys():
                    trailing_ones = np.full([len(data_elt[key].shape)], 1, dtype=int)
                    self.data_matrix[key] = np.tile(np.expand_dims(np.zeros_like(data_elt[key]), axis=0), [self.n_data, *trailing_ones])
            for key in data_elt.keys():
                self.data_matrix[key][index, ...] = data_elt[key]
    def fetch_data_at_index(self, i):
        file_ = self.hdf5_file_list[i]
        with h5py.File(os.path.join(self.dir_files, file_), 'r') as f:
            data = dataset_utils.create_unit_data_from_hdf5_spfn(f, self.n_max_global_instances, self.noisy, n_points=self.n_points, use_glob_features=False, use_loc_features=False, fixed_order=self.fixed_order, shuffle=not self.fixed_order)
            assert data is not None  # assume data are all clean
        if self.test:
            if os.path.isfile(os.path.join(self.path_patches, file_.replace('.h5','_indices.npy'))):
                data['patch_centers'] = np.load(os.path.join(self.path_patches, file_.replace('.h5','_indices.npy')))[:,0]
            else:
                data['patch_centers'] = np.array([])
        return data
    def __getitem__(self, index):
        # find shape that contains the point with given global index
        if not self.test:
            data = {}
            for key in self.data_matrix.keys():
                data[key] = self.data_matrix[key][index,...]
        else:
            data = self.fetch_data_at_index(index)
        P = torch.from_numpy(data['P'].astype(np.float32))
        normal_gt = torch.from_numpy(data['normal_gt'].astype(np.float32))
        P_gt = torch.from_numpy(data['P_gt'].astype(np.float32))
        I_gt = torch.from_numpy(data['I_gt'].astype(np.int64))
        T_gt = torch.from_numpy(data['T_gt'].astype(np.int64))
        plane_n_gt = torch.from_numpy(data['plane_n_gt'].astype(np.float32))
        cylinder_axis_gt = torch.from_numpy(data['cylinder_axis_gt'].astype(np.float32))
        cone_axis_gt = torch.from_numpy(data['cone_axis_gt'].astype(np.float32))
        if self.test:
            patch_centers = torch.from_numpy(data['patch_centers'].astype(np.int64))
            return P, normal_gt, P_gt, I_gt, T_gt, plane_n_gt, cylinder_axis_gt, cone_axis_gt, patch_centers
        else:
            return P, normal_gt, P_gt, I_gt, T_gt, plane_n_gt, cylinder_axis_gt, cone_axis_gt
    def __len__(self):
        return self.n_data

class Dataset_TrainLocalSPFN(data.Dataset):
    def __init__(self, n_max_local_instances, csv_path, patch_folder, noisy, first_n=-1, fixed_order=False, lean=False):
        self.n_max_local_instances = n_max_local_instances
        self.noisy = noisy
        self.first_n = first_n
        self.fixed_order = fixed_order
        self.lean = lean
        self.patch_folder = patch_folder
        csv_raw = pd.read_csv(csv_path, delimiter=',', header=None)[0]
        self.hdf5_file_list = np.sort(csv_raw)
        self.n_data = 0
        self.hdf5_file_list = np.sort([elt for elt in self.hdf5_file_list if self.check_dataset(elt)])
        if not fixed_order:
            random.shuffle(self.hdf5_file_list)
        if self.lean:
            nb_patch_file = np.zeros([len(self.hdf5_file_list)])
            for i, file_ in enumerate(self.hdf5_file_list):
                patch_files = [os.path.join(self.patch_folder, file_.split('.')[0], file_) for file_ in os.listdir(os.path.join(self.patch_folder, file_.split('.')[0])) if file_.split('.')[1] == 'h5']
                nb_patch_file[i] = len(patch_files)
            self.nb_patch_file = nb_patch_file
        if first_n != -1:
            self.hdf5_file_list = self.hdf5_file_list[:first_n]
        if not self.lean:
            self.preload_data()
    def check_dataset(self, file_):
        cond = os.path.isdir(os.path.join(self.patch_folder, file_.split('.')[0]))
        if not cond:
            return False
        patch_files = [os.path.join(self.patch_folder, file_.split('.')[0], file_) for file_ in os.listdir(os.path.join(self.patch_folder, file_.split('.')[0])) if file_.split('.')[1] == 'h5']
        self.n_data += len(patch_files)
        return True
    def preload_data(self):
        cpt = 0
        print('Preloading Dataset:')
        for i, file_ in tqdm(enumerate(self.hdf5_file_list)):
            if i%100==0: print('%d / %d'%(i, len(self.hdf5_file_list)))
            patch_files = [os.path.join(self.patch_folder, file_.split('.')[0], file_) for file_ in os.listdir(os.path.join(self.patch_folder, file_.split('.')[0])) if file_.split('.')[1] == 'h5']
            patch_files = np.sort(patch_files)
            for j in range(len(patch_files)):
                patch_file = os.path.join(self.patch_folder, file_.split('.')[0], file_.replace('.h5','_patch%d.h5'%j))
                data_elt = self.fetch_data_at_index(patch_file)
                if not hasattr(self, 'data_matrix'):
                    self.data_matrix = {}
                    for key in data_elt.keys():
                        trailing_ones = np.full([len(data_elt[key].shape)], 1, dtype=int)
                        self.data_matrix[key] = np.tile(np.expand_dims(np.zeros_like(data_elt[key]), axis=0), [self.n_data, *trailing_ones])
                for key in data_elt.keys():
                    self.data_matrix[key][cpt, ...] = data_elt[key]
                cpt += 1
    def fetch_data_at_index(self, patch_file):
        with h5py.File(patch_file, 'r') as f:
            data = dataset_utils.create_unit_data_from_hdf5_spfn(f, self.n_max_local_instances, noisy=self.noisy, n_points=None, use_glob_features=True, use_loc_features=True, fixed_order=self.fixed_order, shuffle=not self.fixed_order)
        assert data is not None  # assume data are all clean
        return data
    def __getitem__(self, index):
        # find shape that contains the point with given global index
        if not self.lean:
            data = {}
            for key in self.data_matrix.keys():
                data[key] = self.data_matrix[key][index, ...]
        else:
            cumsum = np.cumsum(self.nb_patch_file)
            index_ = np.where(index<cumsum)[0][0]
            file_ = self.hdf5_file_list[index_]
            if index_ == 0:
                j = index
            else:
                j = int(index - cumsum[index_-1])
            patch_file = os.path.join(self.patch_folder, file_.split('.')[0], file_.replace('.h5', '_patch%d.h5' % j))
            data = self.fetch_data_at_index(patch_file)
        P = torch.from_numpy(data['P'].astype(np.float32))
        normal_gt = torch.from_numpy(data['normal_gt'].astype(np.float32))
        P_gt = torch.from_numpy(data['P_gt'].astype(np.float32))
        I_gt = torch.from_numpy(data['I_gt'].astype(np.int64))
        T_gt = torch.from_numpy(data['T_gt'].astype(np.int64))
        plane_n_gt = torch.from_numpy(data['plane_n_gt'].astype(np.float32))
        cylinder_axis_gt = torch.from_numpy(data['cylinder_axis_gt'].astype(np.float32))
        cone_axis_gt = torch.from_numpy(data['cone_axis_gt'].astype(np.float32))
        glob_features = torch.from_numpy(data['glob_features'].astype(np.float32))
        loc_features = torch.from_numpy(data['loc_features'].astype(np.float32))
        output_tuple = (P, normal_gt, P_gt, I_gt, T_gt, plane_n_gt, cylinder_axis_gt, cone_axis_gt, glob_features, loc_features)
        return output_tuple
    def __len__(self):
        return self.n_data

class Dataset_TestLocalSPFN(data.Dataset):
    def __init__(self, n_max_global_instances, n_max_local_instances, csv_path, dir_spfn, dir_lowres, dir_highres, dir_indices, noisy, first_n=-1, fixed_order=False):
        self.n_max_global_instances = n_max_global_instances
        self.n_max_local_instances = n_max_local_instances
        self.dir_spfn = dir_spfn
        self.dir_lowres = dir_lowres
        self.dir_highres = dir_highres
        self.dir_indices = dir_indices
        self.noisy = noisy
        self.first_n = first_n
        self.fixed_order = fixed_order
        csv_raw = pd.read_csv(csv_path, delimiter=',', header=None)[0]
        self.hdf5_file_list = np.sort(csv_raw)
        self.n_data = len(self.hdf5_file_list)
        self.hdf5_file_list_improvement = [elt for elt in self.hdf5_file_list if self.check_dataset(elt)]
    def check_dataset(self, file_):
        cond = os.path.isfile(os.path.join(self.dir_indices, file_.split('.')[0] + '_indices.npy'))
        if not cond:
            return False
        return True
    def fetch_data_at_index(self, patch_file):
        with h5py.File(patch_file, 'r') as f:
            data = dataset_utils.create_unit_data_from_hdf5_spfn(f, self.n_max_global_instances, self.noisy, n_points=None, fixed_order=True, shuffle=False)
        assert data is not None  # assume data are all clean
        return data
    def __getitem__(self, index):
        # find shape that contains the point with given global index
        folder = self.hdf5_file_list[index]
        # Loading the highres file
        data_elt = self.fetch_data_at_index(os.path.join(self.dir_highres, folder))
        P_global = data_elt['P']
        normal_gt_global = data_elt['normal_gt']
        P_gt_global = data_elt['P_gt']
        I_gt_global = data_elt['I_gt']
        T_gt_global = data_elt['T_gt']
        plane_n_gt_global = data_elt['plane_n_gt']
        cylinder_axis_gt_global = data_elt['cylinder_axis_gt']
        cone_axis_gt_global = data_elt['cone_axis_gt']
        if (folder in self.hdf5_file_list_improvement):
            # Loading the patch indices
            patch_indices = np.load(os.path.join(self.dir_indices, folder.replace('.h5', '_indices.npy')))
            nb_patches, _ = patch_indices.shape
            P_unormalised = P_global[patch_indices]
            mean = np.mean(P_unormalised, axis=1, keepdims=True)
            P = P_unormalised - mean
            norm = np.linalg.norm(P, axis=2, keepdims=True).max(axis=1, keepdims=True)
            P = P / norm
            _, num_local_points, _ = P.shape
            normal_gt = normal_gt_global[patch_indices]
            I_gt = I_gt_global[patch_indices]
            P_gt = np.zeros((nb_patches,) + P_gt_global[:self.n_max_local_instances].shape)
            T_gt = np.zeros((nb_patches,) + T_gt_global[:self.n_max_local_instances].shape)
            plane_n_gt = np.zeros((nb_patches,) + plane_n_gt_global[:self.n_max_local_instances].shape)
            cylinder_axis_gt = np.zeros((nb_patches,) + cylinder_axis_gt_global[:self.n_max_local_instances].shape)
            cone_axis_gt = np.zeros((nb_patches,) + cone_axis_gt_global[:self.n_max_local_instances].shape)
            for i in range(nb_patches):
                flag = -1 in I_gt[i]
                unique_values, inverse_values = np.unique(I_gt[i], return_inverse=True)
                if flag: inverse_values = inverse_values - 1
                I_gt[i] = inverse_values
                P_gt[i,np.arange(len(unique_values))] = P_gt_global[unique_values]
                T_gt[i, np.arange(len(unique_values))] = T_gt_global[unique_values]
                plane_n_gt[i, np.arange(len(unique_values))] = plane_n_gt_global[unique_values]
                cylinder_axis_gt[i, np.arange(len(unique_values))] = cylinder_axis_gt_global[unique_values]
                cone_axis_gt[i, np.arange(len(unique_values))] = cone_axis_gt_global[unique_values]
            # Loading the features
            glob_features = np.load(os.path.join(self.dir_spfn, folder.replace('.h5',''), 'global_feat.npy'))
            loc_features = np.load(os.path.join(self.dir_spfn, folder.replace('.h5',''), 'local_feat_full.npy'))
            list_glob_features = []
            list_loc_features = []
            for patch_id in range(nb_patches):
                list_glob_features.append(glob_features)
                list_loc_features.append(loc_features[:,patch_id])
            glob_features = np.stack(list_glob_features, axis=0)
            loc_features = np.stack(list_loc_features, axis=0)
        else:
            nb_patches = 0
            P = np.zeros([0, 8192, 3]).astype(np.float32)
            normal_gt = np.zeros([0, 8192, 3]).astype(np.float32)
            I_gt = np.zeros([0, 8192]).astype(np.int64)
            glob_features = np.zeros([0, 1024]).astype(np.float32)
            loc_features = np.zeros([0, 128]).astype(np.float32)
            patch_indices = np.zeros([0, 8192]).astype(np.int64)
            P_unormalised = np.zeros([0, 8192, 3]).astype(np.float32)
            P_gt = np.zeros([0, 21, 512, 3]).astype(np.float32)
            T_gt = np.zeros([0, 21]).astype(np.int64)
            plane_n_gt = np.zeros([0, 21, 3]).astype(np.float32)
            cylinder_axis_gt = np.zeros([0, 21, 3]).astype(np.float32)
            cone_axis_gt = np.zeros([0, 21, 3]).astype(np.float32)
        # Loading the SPFN output
        spfn_labels = np.load(os.path.join(self.dir_spfn, folder.replace('.h5', ''), 'object_seg.npy'))
        spfn_normals = np.load(os.path.join(self.dir_spfn, folder.replace('.h5', ''), 'object_normals.npy'))
        spfn_type = np.load(os.path.join(self.dir_spfn, folder.replace('.h5', ''), 'object_type.npy'))
        # Shuffling the output
        for i in range(nb_patches):
            perm = np.random.permutation(num_local_points)
            P[i] = P[i, perm]
            P_unormalised[i] = P_unormalised[i, perm]
            normal_gt[i] = normal_gt[i, perm]
            I_gt[i] = I_gt[i, perm]
            patch_indices[i] = patch_indices[i, perm]
        # Exporting all the data
        P = torch.from_numpy(P.astype(np.float32))
        normal_gt = torch.from_numpy(normal_gt.astype(np.float32))
        P_gt = torch.from_numpy(P_gt.astype(np.float32))
        I_gt = torch.from_numpy(I_gt.astype(np.int64))
        T_gt = torch.from_numpy(T_gt.astype(np.int64))
        plane_n_gt = torch.from_numpy(plane_n_gt.astype(np.float32))
        cylinder_axis_gt = torch.from_numpy(cylinder_axis_gt.astype(np.float32))
        cone_axis_gt = torch.from_numpy(cone_axis_gt.astype(np.float32))
        patch_indices = torch.from_numpy(patch_indices.astype(np.float32))
        spfn_labels = torch.from_numpy(spfn_labels.astype(np.int64))
        spfn_normals = torch.from_numpy(spfn_normals.astype(np.float32))
        spfn_type = torch.from_numpy(spfn_type.astype(np.float32))
        glob_features = torch.from_numpy(glob_features.astype(np.float32))
        loc_features = torch.from_numpy(loc_features.astype(np.float32))
        I_gt_global = torch.from_numpy(I_gt_global.astype(np.int64))
        return P, normal_gt, P_gt_global, I_gt, T_gt_global, patch_indices, spfn_labels, spfn_normals, spfn_type, glob_features, loc_features, P_global, normal_gt_global, I_gt_global, plane_n_gt_global, cylinder_axis_gt_global, cone_axis_gt_global, P_unormalised, P_gt, T_gt, plane_n_gt, cylinder_axis_gt, cone_axis_gt
    def __len__(self):
        return self.n_data

class RandomSampler(data.sampler.Sampler):
    def __init__(self, data_source, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.seed = seed
        if self.seed is None:
            self.seed = np.random.randint(0, 2 ** 32 - 1, 1)[0]
        self.identical_epochs = identical_epochs
        self.rng = np.random.RandomState(self.seed)
        self.total_samples_count = len(self.data_source)
    def __iter__(self):
        if self.identical_epochs:
            self.rng.seed(self.seed)
        return iter(self.rng.choice(self.total_samples_count, size=self.total_samples_count, replace=False))
    def __len__(self):
        return self.total_samples_count

class Sampler(data.sampler.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.total_samples_count = len(self.data_source)
    def __iter__(self):
        return iter(np.arange(0, self.total_samples_count))
    def __len__(self):
        return self.total_samples_count