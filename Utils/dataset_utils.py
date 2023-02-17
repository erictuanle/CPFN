# Importation of packages
import os
import re
import h5py
import pickle
import numpy as np

from SPFN import cone_fitter, cylinder_fitter, fitter_factory, plane_fitter, sphere_fitter

def create_unit_data_from_hdf5_patch_selection(h5file_lowres, h5file_highres, normalisation, scale, n_points=None):
    with h5py.File(h5file_lowres, 'r') as f:
        points = f['noisy_points'][()].astype(np.float32)
        if n_points is not None:
            points = points[:n_points]
        else:
            n_points, _ = points.shape
        labels = f['gt_labels'][()].astype(np.int64)[:n_points]
    with h5py.File(h5file_highres, 'r') as f:
        highres_labels = f['gt_labels'][()].astype(np.int64)
        highres_npoints = highres_labels.shape[0]
    unique_labels, unique_counts = np.unique(highres_labels, return_counts=True)
    unique_labels = unique_labels[unique_counts>highres_npoints*scale]
    output_labels = 1 - np.isin(labels, unique_labels).astype(np.int64)
    highres_output_labels = 1 - np.isin(highres_labels, unique_labels).astype(np.int64)
    if normalisation:
        points = (points - np.mean(points, axis=0))
        points = points / np.linalg.norm(points, axis=1).max()
    shuffled_indices = np.random.choice(n_points, n_points, replace=False)
    points = points[shuffled_indices]
    output_labels = output_labels[shuffled_indices]
    shuffled_indices = np.argsort(shuffled_indices)
    return points, output_labels, shuffled_indices

def create_unit_data_from_hdf5_spfn(f, n_max_instances, noisy, n_points=None, use_glob_features=False, use_loc_features=False, fixed_order=False, shuffle=True):
    # Loading Point Features
    P = f['noisy_points'][()] if noisy else f['gt_points'][()]  # Nx3
    normal_gt = f['gt_normals'][()]
    I_gt = f['gt_labels'][()]
    # Loading the SPFN global and local features
    if use_glob_features:
        glob_features = f['glob_features'][()]
    if use_loc_features:
        loc_features = f['loc_features'][()]
    # Reducing the number of points if needed
    if n_points is not None:
        P = P[:n_points]
        normal_gt = normal_gt[:n_points]
        I_gt = I_gt[:n_points]
    n_total_points = P.shape[0]
    # Checking if soup_ids are consecutive
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
    if n_instances == 0:
        return None
    for i in range(n_instances):
        if i not in found_soup_ids:
            print('{} is not found in soup ids!'.format(i))
            return None
    # Adding Primitive Information
    P_gt = []
    instances = []
    for i in range(n_instances):
        g = f[soup_id_to_key[i]]
        P_gt_cur = g['gt_points'][()]
        P_gt.append(P_gt_cur)
        if type(g.attrs['meta']) == np.void:
            meta = pickle.loads(g.attrs['meta'])
        else:
            meta = eval(g.attrs['meta'])
        primitive = fitter_factory.create_primitive_from_dict(meta)
        if primitive is None:
            return None
        instances.append(primitive)
    if n_instances > n_max_instances:
        print('n_instances {} > n_max_instances {}'.format(n_instances, n_max_instances))
        return None
    if np.amax(I_gt) >= n_instances:
        print('max label {} > n_instances {}'.format(np.amax(I_gt), n_instances))
        return None
    T_gt = [fitter_factory.primitive_name_to_id(primitive.get_primitive_name()) for primitive in instances]
    T_gt.extend([0 for _ in range(n_max_instances - n_instances)])
    n_gt_points_per_instance = P_gt[0].shape[0]
    P_gt.extend([np.zeros(dtype=float, shape=[n_gt_points_per_instance, 3]) for _ in range(n_max_instances - n_instances)])
    # Converting everything to numpy array
    P_gt = np.array(P_gt)
    T_gt = np.array(T_gt)
    if shuffle and (not fixed_order):
        # shuffle per point information around
        perm = np.random.permutation(n_total_points)
        P = P[perm]
        normal_gt = normal_gt[perm]
        I_gt = I_gt[perm]
    result = {
        'P': P,
        'normal_gt': normal_gt,
        'P_gt': P_gt,
        'I_gt': I_gt,
        'T_gt': T_gt,
    }
    if use_glob_features: result['glob_features'] = glob_features
    if use_loc_features: result['loc_features'] = loc_features
    # Adding in primitive parameters
    for class_ in fitter_factory.primitive_name_to_id_dict.keys():
        if class_ == 'plane':
            result.update(plane_fitter.extract_parameter_data_as_dict(instances, n_max_instances))
        elif class_ == 'sphere':
            result.update(sphere_fitter.extract_parameter_data_as_dict(instances, n_max_instances))
        elif class_ == 'cylinder':
            result.update(cylinder_fitter.extract_parameter_data_as_dict(instances, n_max_instances))
        elif class_ == 'cone':
            result.update(cone_fitter.extract_parameter_data_as_dict(instances, n_max_instances))
        else:
            raise NotImplementedError
    return result