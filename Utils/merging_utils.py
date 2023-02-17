# Importation of packages
import torch
import numba
import numpy as np

def similarity_soft(spfn_labels, predicted_labels, point_indices):
    num_points_per_object, max_label_per_object = spfn_labels.size()
    nb_patches, num_points_per_patch, max_label_per_patch = predicted_labels.size()
    point2primitive_prediction = torch.zeros([num_points_per_object, nb_patches*max_label_per_patch+max_label_per_object]).to(predicted_labels.device)
    for b in range(nb_patches):
        predicted_labels_b = predicted_labels[b]
        point2primitive_prediction[point_indices[b],b*max_label_per_patch:(b+1)*max_label_per_patch] += predicted_labels_b
    point2primitive_prediction[:,(b+1)*max_label_per_patch:] = spfn_labels
    intersection_primitives = torch.mm(point2primitive_prediction.transpose(0,1),point2primitive_prediction)
    return intersection_primitives

@numba.jit(numba.int64[:](numba.int64[:,:], numba.int64[:], numba.float64[:]), nopython=True)
def heuristic_merging(pairs_id, patch_id, penalty_value):
    pairs_id1 = pairs_id[:,0]
    pairs_id2 = pairs_id[:,1]
    segment_id = np.arange(len(patch_id), dtype=numba.int64)
    patch_1hot = np.eye(patch_id.max()+1)[patch_id]
    while len(pairs_id1) > 0:
        pair_id1 = pairs_id1[np.argmax(penalty_value)]
        pair_id2 = pairs_id2[np.argmax(penalty_value)]
        segment_id[segment_id==segment_id[pair_id2]] = segment_id[pair_id1]
        selection_row = segment_id==segment_id[pair_id1]
        patch_1hot[selection_row] = np.sum(patch_1hot[selection_row], axis=0)
        intersection = np.sum(patch_1hot[pairs_id1] * patch_1hot[pairs_id2], axis=1)
        pairs_id1 = pairs_id1[intersection==0]
        pairs_id2 = pairs_id2[intersection==0]
        penalty_value = penalty_value[intersection==0]
    return segment_id

def run_heuristic_solver(similarity_matrix, nb_patches, max_label_per_object, max_label_per_patch, threshold=0):
    # Building the Gurobi optimisation problem
    indices = np.where(similarity_matrix>threshold)
    penalty_array = np.stack((indices[0], indices[1], similarity_matrix[indices[0], indices[1]]), axis=1)
    penalty_array = penalty_array[penalty_array[:,0]<penalty_array[:,1]]
    # Heuristic
    patch_id = np.concatenate((np.repeat(np.arange(nb_patches), repeats=max_label_per_patch, axis=0), nb_patches*np.ones([max_label_per_object], dtype=int)), axis=0)
    glob_output_labels_heuristic = heuristic_merging(penalty_array[:,:2].astype(int), patch_id, penalty_array[:,2])
    flag = np.diag(similarity_matrix)
    replacement_values = np.concatenate((np.tile(np.arange(-max_label_per_patch, 0), nb_patches), np.arange(-max_label_per_object, 0)), axis=0)
    glob_output_labels_heuristic[flag<threshold] = replacement_values[flag<threshold]
    _, glob_output_labels_heuristic = np.unique(glob_output_labels_heuristic, return_inverse=True)
    return glob_output_labels_heuristic

def get_point_final(point2primitive_prediction, output_labels_heuristic):
    output_labels_heuristic = torch.eye(output_labels_heuristic.max()+1).to(output_labels_heuristic.device)[output_labels_heuristic.long()]
    output_labels_heuristic = output_labels_heuristic / (torch.sum(output_labels_heuristic, dim=0, keepdim=True) + 1e-10)
    final_output_labels_heuristic = torch.mm(point2primitive_prediction, output_labels_heuristic)
    return final_output_labels_heuristic