# Importation of packages
import torch
import numpy as np
if __name__ == '__main__':
    import tensorflow as tf
from scipy.optimize import linear_sum_assignment

from SPFN import plane_fitter, sphere_fitter, cylinder_fitter, cone_fitter

# Segmentation Loss
def hungarian_matching(W_pred, I_gt):
    # This non-tf function does not backprob gradient, only output matching indices
    # W_pred - BxNxK
    # I_gt - BxN, may contain -1's
    # Output: matching_indices - BxK, where (b,k)th ground truth primitive is matched with (b, matching_indices[b, k])
    #   where only n_gt_labels entries on each row have meaning. The matching does not include gt background instance
    batch_size, n_points, n_max_labels = W_pred.size()
    matching_indices = torch.zeros([batch_size, n_max_labels], dtype=torch.long).to(W_pred.device)
    for b in range(batch_size):
        # assuming I_gt[b] does not have gap
        n_gt_labels = torch.max(I_gt[b]).item() + 1  # this is K'
        W_gt = torch.eye(n_gt_labels+1).to(I_gt.device)[I_gt[b]]
        dot = torch.mm(W_gt.transpose(0,1), W_pred[b])
        denominator = torch.sum(W_gt, dim=0).unsqueeze(1) + torch.sum(W_pred[b], dim=0).unsqueeze(0) - dot
        cost = dot / torch.clamp(denominator, min=1e-10, max=None)  # K'xK
        cost = cost[:n_gt_labels, :]  # remove last row, corresponding to matching gt background instance
        _, col_ind = linear_sum_assignment(-cost.detach().cpu().numpy())  # want max solution
        col_ind = torch.from_numpy(col_ind).long().to(matching_indices.device)
        matching_indices[b, :n_gt_labels] = col_ind
    return matching_indices

def hungarian_matching_tensorflow(W_pred, I_gt):
    # This non-tf function does not backprob gradient, only output matching indices
    # W_pred - BxNxK
    # I_gt - BxN, may contain -1's
    # Output: matching_indices - BxK, where (b,k)th ground truth primitive is matched with (b, matching_indices[b, k])
    #   where only n_gt_labels entries on each row have meaning. The matching does not include gt background instance
    batch_size = I_gt.shape[0]
    n_points = I_gt.shape[1]
    n_max_labels = W_pred.shape[2]
    matching_indices = np.zeros([batch_size, n_max_labels], dtype=np.int32)
    for b in range(batch_size):
        # assuming I_gt[b] does not have gap
        n_gt_labels = np.max(I_gt[b]) + 1  # this is K'
        W_gt = np.zeros([n_points, n_gt_labels + 1])  # HACK: add an extra column to contain -1's
        W_gt[np.arange(n_points), I_gt[b]] = 1.0  # NxK'
        dot = np.sum(np.expand_dims(W_gt, axis=2) * np.expand_dims(W_pred[b], axis=1), axis=0)  # K'xK
        denominator = np.expand_dims(np.sum(W_gt, axis=0), axis=1) + np.expand_dims(np.sum(W_pred[b], axis=0), axis=0) - dot
        cost = dot / np.maximum(denominator, 1e-10)  # K'xK
        cost = cost[:n_gt_labels, :]  # remove last row, corresponding to matching gt background instance
        _, col_ind = linear_sum_assignment(-cost)  # want max solution
        matching_indices[b, :n_gt_labels] = col_ind
    return matching_indices

if __name__ == '__main__':
    batch_size = 100
    num_points = 1024
    n_max_instances = 12
    device = torch.device('cuda:0')
    np.random.seed(0)
    W_pred = np.random.rand(batch_size, num_points, n_max_instances)
    I_gt = np.random.randint(-1, n_max_instances, (batch_size, num_points))
    W_pred = W_pred / np.linalg.norm(W_pred, axis=2, keepdims=True)
    W_pred_torch = torch.from_numpy(W_pred).float().to(device)
    I_gt_torch = torch.from_numpy(I_gt).long().to(device)
    matching_indices_torch = hungarian_matching(W_pred_torch, I_gt_torch)
    matching_indices_torch = matching_indices_torch.detach().cpu().numpy()
    print('matching_indices_torch', matching_indices_torch)
    # Debugging with Tensorflow
    W_pred_tensorflow = tf.constant(W_pred, dtype=tf.float32)
    I_gt_tensorflow = tf.constant(I_gt, dtype=tf.int32)
    matching_indices_tensorflow = tf.py_func(hungarian_matching_tensorflow, [W_pred_tensorflow, I_gt_tensorflow], Tout=tf.int32)
    sess = tf.Session()
    matching_indices_tensorflow = sess.run(matching_indices_tensorflow)
    print(np.abs(matching_indices_torch - matching_indices_tensorflow).max())

def compute_miou_loss(W, I_gt, matching_indices, div_eps=1e-10):
    # W - BxNxK
    # I_gt - BxN
    batch_size, n_points, n_max_labels = W.size()
    _, n_labels = matching_indices.size()
    W_reordered = torch.gather(W, 2, matching_indices.unsqueeze(1).expand(batch_size, n_points, n_labels)) # BxNxK
    # notice in tf.one_hot, -1 will result in a zero row, which is what we want
    W_gt = torch.eye(n_labels+2).to(I_gt.device)[I_gt]
    W_gt = W_gt[:,:,:n_labels]
    dot = torch.sum(W_gt * W_reordered, axis=1) # BxK
    denominator = torch.sum(W_gt, dim=1) + torch.sum(W_reordered, dim=1) - dot
    mIoU = dot / (denominator + div_eps) # BxK
    return 1.0 - mIoU, 1 - dot / n_points

def batched_gather_tensorflow(data, indices, axis):
    # data - Bx...xKx..., axis is where dimension K is
    # indices - BxK
    # output[b, ..., k, ...] = in[b, ..., indices[b, k], ...]
    assert axis >= 1
    ndims = data.get_shape().ndims # allow dynamic rank
    if axis > 1:
        # tranpose data to BxKx...
        perm = np.arange(ndims)
        perm[axis] = 1
        perm[1] = axis
        data = tf.transpose(data, perm=perm)
    batch_size = tf.shape(data)[0]
    batch_nums = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(batch_size), axis=1), axis=2), multiples=[1, tf.shape(indices)[1], 1]) # BxKx1
    indices = tf.concat([batch_nums, tf.expand_dims(indices, axis=2)], axis=2) # BxKx2
    gathered_data = tf.gather_nd(data, indices=indices)
    if axis > 1:
        gathered_data = tf.transpose(gathered_data, perm=perm)
    return gathered_data

def compute_miou_loss_tensorflow(W, I_gt, matching_indices):
    # W - BxNxK
    # I_gt - BxN
    W_reordered = batched_gather_tensorflow(W, indices=matching_indices, axis=2) # BxNxK
    depth = tf.shape(W)[2]
    # notice in tf.one_hot, -1 will result in a zero row, which is what we want
    W_gt = tf.one_hot(I_gt, depth=depth, dtype=tf.float32) # BxNxK
    dot = tf.reduce_sum(W_gt * W_reordered, axis=1) # BxK
    denominator = tf.reduce_sum(W_gt, axis=1) + tf.reduce_sum(W_reordered, axis=1) - dot
    mIoU = dot / (denominator + 1e-10) # BxK
    return 1.0 - mIoU

if __name__ == '__main__':
    batch_size = 100
    num_points = 1024
    n_max_instances = 12
    device = torch.device('cuda:0')
    np.random.seed(0)
    W = np.random.rand(batch_size, num_points, n_max_instances)
    I_gt = np.random.randint(-1, n_max_instances, (batch_size, num_points))
    W = W / np.linalg.norm(W, axis=2, keepdims=True)
    W_torch = torch.from_numpy(W).float().to(device)
    I_gt_torch = torch.from_numpy(I_gt).long().to(device)
    with torch.no_grad():
        matching_indices_torch = hungarian_matching(W_torch, I_gt_torch)
    loss_torch, _ = compute_miou_loss(W_torch, I_gt_torch, matching_indices_torch)
    loss_torch = loss_torch.detach().cpu().numpy()
    print('loss_torch', loss_torch)
    # Debugging with Tensorflow
    W_tensorflow = tf.constant(W, dtype=tf.float32)
    I_gt_tensorflow = tf.constant(I_gt, dtype=tf.int32)
    matching_indices_tensorflow = tf.stop_gradient(tf.py_func(hungarian_matching_tensorflow, [W_tensorflow, I_gt_tensorflow], Tout=tf.int32))
    loss_tensorflow = compute_miou_loss_tensorflow(W_tensorflow, I_gt_tensorflow, matching_indices_tensorflow)
    sess = tf.Session()
    loss_tensorflow = sess.run(loss_tensorflow)
    print(np.abs(loss_torch - loss_tensorflow).max())

# Normal Loss
def acos_safe(x):
    return torch.acos(torch.clamp(x, min=-1.0+1e-6, max=1.0-1e-6))

def compute_normal_loss(normal, normal_gt, angle_diff):
    # normal, normal_gt: BxNx3
    # Assume normals are unoriented
    dot_abs = torch.abs(torch.sum(normal * normal_gt, dim=2)) # BxN
    if angle_diff:
        return torch.mean(acos_safe(dot_abs), dim=1)
    else:
        return torch.mean(1.0 - dot_abs, dim=1)

def acos_safe_tensorflow(x):
    return tf.math.acos(tf.clip_by_value(x, -1.0+1e-6, 1.0-1e-6))

def compute_normal_loss_tensorflow(normal, normal_gt, angle_diff):
    # normal, normal_gt: BxNx3
    # Assume normals are unoriented
    dot_abs = tf.abs(tf.reduce_sum(normal * normal_gt, axis=2)) # BxN
    if angle_diff:
        return tf.reduce_mean(acos_safe_tensorflow(dot_abs), axis=1)
    else:
        return tf.reduce_mean(1.0 - dot_abs, axis=1)

if __name__ == '__main__':
    batch_size = 100
    num_points = 1024
    device = torch.device('cuda:0')
    np.random.seed(0)
    normal = np.random.randn(batch_size, num_points, 3)
    normal_gt = np.random.randn(batch_size, num_points, 3)
    angle_diff = True
    normal_torch = torch.from_numpy(normal).float().to(device)
    normal_gt_torch = torch.from_numpy(normal_gt).float().to(device)
    loss_torch = compute_normal_loss(normal_torch, normal_gt_torch, angle_diff)
    loss_torch = loss_torch.detach().cpu().numpy()
    print('loss_torch', loss_torch)
    # Debugging with Tensorflow
    normal_tensorflow = tf.constant(normal, dtype=tf.float32)
    normal_gt_tensorflow = tf.constant(normal_gt, dtype=tf.float32)
    loss_tensorflow = compute_normal_loss_tensorflow(normal_tensorflow, normal_gt_tensorflow, angle_diff)
    sess = tf.Session()
    loss_tensorflow = sess.run(loss_tensorflow)
    print(np.abs(loss_torch - loss_tensorflow).max())

# Type Loss
def compute_per_point_type_loss(per_point_type, I_gt, T_gt, is_eval):
    # For training, per_point_type is BxNxQ, where Q = n_registered_primitives
    # For test, per_point_type is BxN
    # I_gt - BxN, allow -1
    # T_gt - BxK
    batch_size, n_points = I_gt.size()
    per_point_type_gt = torch.gather(T_gt, 1, torch.clamp(I_gt, min=0, max=None))
    if is_eval:
        type_loss = 1.0 - (per_point_type == per_point_type_gt).float()
    else:
        type_loss = torch.nn.functional.cross_entropy(per_point_type.contiguous().view(batch_size*n_points, -1), per_point_type_gt.view(batch_size*n_points), reduction='none') # BxN
        type_loss = type_loss.view(batch_size, n_points)
    # do not add loss to background points in gt
    type_loss = torch.where(I_gt == -1, torch.zeros_like(type_loss), type_loss)
    return torch.sum(type_loss, dim=1) / (torch.sum((I_gt != -1).float(), dim=1).float()) # B

def compute_per_point_type_loss_tensorflow(per_point_type, I_gt, T_gt, is_eval):
    # For training, per_point_type is BxNxQ, where Q = n_registered_primitives
    # For test, per_point_type is BxN
    # I_gt - BxN, allow -1
    # T_gt - BxK
    batch_size = tf.shape(I_gt)[0]
    n_points = tf.shape(I_gt)[1]
    indices_0 = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, n_points]) # BxN
    indices = tf.stack([indices_0, tf.maximum(0, I_gt)], axis=2)
    per_point_type_gt = tf.gather_nd(T_gt, indices=indices) # BxN
    if is_eval:
        type_loss = 1.0 - tf.to_float(tf.equal(per_point_type, per_point_type_gt))
    else:
        type_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=per_point_type, labels=per_point_type_gt) # BxN
    # do not add loss to background points in gt
    type_loss = tf.where(tf.equal(I_gt, -1), tf.zeros_like(type_loss), type_loss)
    return tf.reduce_sum(type_loss, axis=1) / tf.to_float(tf.count_nonzero(tf.not_equal(I_gt, -1), axis=1)) # B

if __name__ == '__main__':
    batch_size = 100
    num_points = 1024
    Q = 4
    K = 10
    device = torch.device('cuda:0')
    np.random.seed(0)
    per_point_type = np.random.randn(batch_size, num_points, Q)
    I_gt = np.random.randint(-1, K, (batch_size, num_points))
    T_gt = np.random.randint(0, Q, (batch_size, K))
    is_eval = False
    per_point_type_torch = torch.from_numpy(per_point_type).float().to(device)
    I_gt_torch = torch.from_numpy(I_gt).long().to(device)
    T_gt_torch = torch.from_numpy(T_gt).long().to(device)
    loss_torch = compute_per_point_type_loss(per_point_type_torch, I_gt_torch, T_gt_torch, is_eval)
    loss_torch = loss_torch.detach().cpu().numpy()
    print('loss_torch', loss_torch)
    # Debugging with Tensorflow
    per_point_type_tensorflow = tf.constant(per_point_type, dtype=tf.float32)
    I_gt_tensorflow = tf.constant(I_gt, dtype=tf.int32)
    T_gt_tensorflow = tf.constant(T_gt, dtype=tf.int32)
    loss_tensorflow = compute_per_point_type_loss_tensorflow(per_point_type_tensorflow, I_gt_tensorflow, T_gt_tensorflow, is_eval)
    sess = tf.Session()
    loss_tensorflow = sess.run(loss_tensorflow)
    print(np.abs(loss_torch - loss_tensorflow).max())

def compute_parameters(P, W, X, classes=['plane','sphere','cylinder','cone']):
    parameters = {}
    for class_ in classes:
        if class_ == 'plane':
            plane_normal, plane_center = plane_fitter.compute_parameters(P, W)
            parameters['plane_normal'] = plane_normal
            parameters['plane_center'] = plane_center
        elif class_ == 'sphere':
            sphere_center, sphere_radius_squared = sphere_fitter.compute_parameters(P, W)
            parameters['sphere_center'] = sphere_center
            parameters['sphere_radius_squared'] = sphere_radius_squared
        elif class_ == 'cylinder':
            cylinder_axis, cylinder_center, cylinder_radius_squared = cylinder_fitter.compute_parameters(P, W, X)
            parameters['cylinder_axis'] = cylinder_axis
            parameters['cylinder_center'] = cylinder_center
            parameters['cylinder_radius_squared'] = cylinder_radius_squared
        elif class_ == 'cone':
            cone_apex, cone_axis, cone_half_angle = cone_fitter.compute_parameters(P, W, X)
            parameters['cone_apex'] = cone_apex
            parameters['cone_axis'] = cone_axis
            parameters['cone_half_angle'] = cone_half_angle
        else:
            raise NotImplementedError
    return parameters

def compute_parameters_tensorflow(P, W, X, classes=['plane','sphere','cylinder','cone']):
    parameters = {}
    for class_ in classes:
        if class_ == 'plane':
            plane_normal, plane_center = plane_fitter.compute_parameters_tensorflow(P, W)
            parameters['plane_normal'] = plane_normal
            parameters['plane_center'] = plane_center
        elif class_ == 'sphere':
            sphere_center, sphere_radius_squared = sphere_fitter.compute_parameters_tensorflow(P, W)
            parameters['sphere_center'] = sphere_center
            parameters['sphere_radius_squared'] = sphere_radius_squared
        elif class_ == 'cylinder':
            cylinder_axis, cylinder_center, cylinder_radius_squared = cylinder_fitter.compute_parameters_tensorflow(P, W, X)
            parameters['cylinder_axis'] = cylinder_axis
            parameters['cylinder_center'] = cylinder_center
            parameters['cylinder_radius_squared'] = cylinder_radius_squared
        elif class_ == 'cone':
            cone_apex, cone_axis, cone_half_angle = cone_fitter.compute_parameters_tensorflow(P, W, X)
            parameters['cone_apex'] = cone_apex
            parameters['cone_axis'] = cone_axis
            parameters['cone_half_angle'] = cone_half_angle
        else:
            raise NotImplementedError
    return parameters

if __name__ == '__main__':
    batch_size = 100
    num_points = 1024
    n_max_instances = 12
    device = torch.device('cuda:0')
    np.random.seed(0)
    P = np.random.randn(batch_size, num_points, 3)
    W = np.random.rand(batch_size, num_points, n_max_instances)
    X = np.random.randn(batch_size, num_points, 3)
    X = X / np.linalg.norm(X, axis=2, keepdims=True)
    P_torch = torch.from_numpy(P).float().to(device)
    W_torch = torch.from_numpy(W).float().to(device)
    X_torch = torch.from_numpy(X).float().to(device)
    parameters = compute_parameters(P_torch, W_torch, X_torch)
    plane_normal_torch, plane_center_torch, sphere_center_torch, sphere_radius_squared_torch, cylinder_axis_torch, cylinder_center_torch, cylinder_radius_square_torch, cone_apex_torch, cone_axis_torch, cone_half_angle_torch = \
        parameters['plane_normal'], parameters['plane_center'], parameters['sphere_center'], parameters['sphere_radius_squared'], parameters['cylinder_axis'], parameters['cylinder_center'], parameters['cylinder_radius_square'], parameters['cone_apex'] ,parameters['cone_axis'], parameters['cone_half_angle']
    plane_normal_torch = plane_normal_torch.detach().cpu().numpy()
    plane_center_torch = plane_center_torch.detach().cpu().numpy()
    sphere_center_torch = sphere_center_torch.detach().cpu().numpy()
    sphere_radius_squared_torch = sphere_radius_squared_torch.detach().cpu().numpy()
    cylinder_axis_torch = cylinder_axis_torch.detach().cpu().numpy()
    cylinder_center_torch = cylinder_center_torch.detach().cpu().numpy()
    cylinder_radius_square_torch = cylinder_radius_square_torch.detach().cpu().numpy()
    cone_apex_torch = cone_apex_torch.detach().cpu().numpy()
    cone_axis_torch = cone_axis_torch.detach().cpu().numpy()
    cone_half_angle_torch = cone_half_angle_torch.detach().cpu().numpy()
    # Debugging with Tensorflow
    P_tensorflow = tf.constant(P, dtype=tf.float32)
    W_tensorflow = tf.constant(W, dtype=tf.float32)
    X_tensorflow = tf.constant(X, dtype=tf.float32)
    parameters = compute_parameters_tensorflow(P_tensorflow, W_tensorflow, X_tensorflow)
    sess = tf.Session()
    plane_normal_tensorflow, plane_center_tensorflow, sphere_center_tensorflow, sphere_radius_squared_tensorflow, cylinder_axis_tensorflow, cylinder_center_tensorflow, cylinder_radius_square_tensorflow, cone_apex_tensorflow, cone_axis_tensorflow, cone_half_angle_tensorflow = \
        sess.run([parameters['plane_normal'], parameters['plane_center'], parameters['sphere_center'], parameters['sphere_radius_squared'], parameters['cylinder_axis'], parameters['cylinder_center'], parameters['cylinder_radius_square'], parameters['cone_apex'] ,parameters['cone_axis'], parameters['cone_half_angle']])
    print(np.minimum(np.abs(plane_normal_tensorflow - plane_normal_torch), np.abs(plane_normal_tensorflow + plane_normal_torch)).max())
    print(np.minimum(np.abs(plane_center_tensorflow - plane_center_torch), np.abs(plane_center_tensorflow + plane_center_torch)).max())
    print(np.abs(sphere_center_tensorflow - sphere_center_torch).max())
    print(np.abs(sphere_radius_squared_tensorflow - sphere_radius_squared_torch).max())
    print(np.minimum(np.abs(cylinder_axis_tensorflow - cylinder_axis_torch), np.abs(cylinder_axis_tensorflow + cylinder_axis_torch)).max())
    print(np.abs(cylinder_center_tensorflow - cylinder_center_torch).max())
    print(np.abs(cylinder_radius_square_tensorflow - cylinder_radius_square_torch).max())
    print(np.abs(cone_apex_tensorflow - cone_apex_torch).max())
    print(np.minimum(np.abs(cone_axis_tensorflow - cone_axis_torch), np.abs(cone_axis_tensorflow + cone_axis_torch)).max())
    print(np.abs(cone_half_angle_tensorflow - cone_half_angle_torch).max())

# Residue Loss
def compute_residue_loss(parameters, matching_indices, points_per_instance, T_gt, classes=['plane','sphere','cylinder','cone']):
    # parameters is a dictionary where each key represents a different parameter
    # points_per_instance of size BxKxN'x3
    residue_losses = []  # a length T array of BxK tensors
    residue_per_point_array = []  # a length T array of BxKxN' tensors
    #residue_per_class = []
    batch_size, n_labels = matching_indices.size()
    for class_ in classes:
        if class_ == 'plane':
            residue_per_point = plane_fitter.compute_residue_single(torch.gather(parameters['plane_normal'], 1, matching_indices.unsqueeze(2).expand(batch_size, n_labels, 3)).unsqueeze(2),
                                                                    torch.gather(parameters['plane_center'], 1, matching_indices).unsqueeze(2),
                                                                    points_per_instance)
        elif class_ == 'sphere':
            residue_per_point = sphere_fitter.compute_residue_single(torch.gather(parameters['sphere_center'], 1, matching_indices.unsqueeze(2).expand(batch_size, n_labels, 3)).unsqueeze(2),
                                                                  torch.gather(parameters['sphere_radius_squared'], 1, matching_indices.expand(batch_size, n_labels)).unsqueeze(2),
                                                                  points_per_instance)
        elif class_ == 'cylinder':
            residue_per_point = cylinder_fitter.compute_residue_single(torch.gather(parameters['cylinder_axis'], 1, matching_indices.unsqueeze(2).expand(batch_size, n_labels, 3)).unsqueeze(2),
                                                                     torch.gather(parameters['cylinder_center'], 1, matching_indices.unsqueeze(2).expand(batch_size, n_labels, 3)).unsqueeze(2),
                                                                     torch.gather(parameters['cylinder_radius_squared'], 1, matching_indices.expand(batch_size, n_labels)).unsqueeze(2),
                                                                     points_per_instance)
        elif class_ == 'cone':
            residue_per_point = cone_fitter.compute_residue_single(torch.gather(parameters['cone_apex'], 1, matching_indices.unsqueeze(2).expand(batch_size, n_labels, 3)).unsqueeze(2),
                                                                 torch.gather(parameters['cone_axis'], 1, matching_indices.unsqueeze(2).expand(batch_size, n_labels, 3)).unsqueeze(2),
                                                                 torch.gather(parameters['cone_half_angle'], 1, matching_indices.expand(batch_size, n_labels)).unsqueeze(2),
                                                                 points_per_instance)
        else:
            raise NotImplementedError

        #residue_per_class.append(residue_per_point)

        residue_per_point_array.append(residue_per_point)
        residue_losses.append(torch.mean(residue_per_point, dim=2))
    residue_losses = torch.stack(residue_losses, dim=2)
    residue_loss = torch.gather(residue_losses, 2, T_gt.unsqueeze(2)).squeeze(2)
    residue_per_point_array = torch.stack(residue_per_point_array, dim=3)  # BxKxN'xT
    return residue_loss, residue_per_point_array#, residue_per_class

def aggregate_loss_from_stacked_tensorflow(loss_stacked, T_gt):
    # loss_stacked - BxKxT, T_gt - BxK
    # out[b, k] = loss_stacked[b, k, T_gt[b, k]]
    B = tf.shape(loss_stacked)[0]
    K = tf.shape(loss_stacked)[1]
    indices_0 = tf.tile(tf.expand_dims(tf.range(B), axis=1), multiples=[1, K]) # BxK
    indices_1 = tf.tile(tf.expand_dims(tf.range(K), axis=0), multiples=[B, 1]) # BxK
    indices = tf.stack([indices_0, indices_1, T_gt], axis=2) # BxKx3
    return tf.gather_nd(loss_stacked, indices=indices)

def compute_residue_loss_tensorflow(parameters, matching_indices, points_per_instance, T_gt, classes=['plane','sphere','cylinder','cone']):
    residue_losses = []  # a length T array of BxK tensors
    residue_per_point_array = []  # a length T array of BxKxN' tensors
    #residue_per_class = []
    for class_ in classes:
        if class_ == 'plane':
            residue_per_point = plane_fitter.compute_residue_single_tensorflow(tf.expand_dims(batched_gather_tensorflow(parameters['plane_normal'], matching_indices, axis=1), axis=2),
                                                                               tf.expand_dims(batched_gather_tensorflow(parameters['plane_center'], matching_indices, axis=1), axis=2),
                                                                               points_per_instance)
        elif class_ == 'sphere':
            residue_per_point = sphere_fitter.compute_residue_single_tensorflow(tf.expand_dims(batched_gather_tensorflow(parameters['sphere_center'], matching_indices, axis=1), axis=2),
                                                                                tf.expand_dims(batched_gather_tensorflow(parameters['sphere_radius_squared'], matching_indices, axis=1), axis=2),
                                                                                points_per_instance)
        elif class_ == 'cylinder':
            residue_per_point = cylinder_fitter.compute_residue_single_tensorflow(tf.expand_dims(batched_gather_tensorflow(parameters['cylinder_axis'], matching_indices, axis=1), axis=2),
                                                                                  tf.expand_dims(batched_gather_tensorflow(parameters['cylinder_center'], matching_indices, axis=1), axis=2),
                                                                                  tf.expand_dims(batched_gather_tensorflow(parameters['cylinder_radius_squared'], matching_indices, axis=1), axis=2),
                                                                                  points_per_instance)
        elif class_ == 'cone':
            residue_per_point = cone_fitter.compute_residue_single_tensorflow(tf.expand_dims(batched_gather_tensorflow(parameters['cone_apex'], matching_indices, axis=1), axis=2),
                                                                              tf.expand_dims(batched_gather_tensorflow(parameters['cone_axis'], matching_indices, axis=1), axis=2),
                                                                              tf.expand_dims(batched_gather_tensorflow(parameters['cone_half_angle'], matching_indices, axis=1), axis=2),
                                                                              points_per_instance)
        else:
            raise NotImplementedError

        #residue_per_class.append(residue_per_point)

        residue_per_point_array.append(residue_per_point)
        residue_losses.append(tf.reduce_mean(residue_per_point, axis=2))
    residue_losses = tf.stack(residue_losses, axis=2)
    residue_per_point_array = tf.stack(residue_per_point_array, axis=3)  # BxKxN'xT
    # Aggregate losses across fitters
    residue_loss = aggregate_loss_from_stacked_tensorflow(residue_losses, T_gt)  # BxK
    return residue_loss, residue_per_point_array#, residue_per_class

if __name__ == '__main__':
    batch_size = 100
    num_points = 1024
    num_points_instance = 512
    n_max_instances = 12
    device = torch.device('cuda:0')
    np.random.seed(0)
    P = np.random.randn(batch_size, num_points, 3)
    W = np.random.rand(batch_size, num_points, n_max_instances)
    X = np.random.randn(batch_size, num_points, 3)
    X = X / np.linalg.norm(X, axis=2, keepdims=True)
    points_per_instance = np.random.randn(batch_size, n_max_instances, num_points_instance, 3)
    T_gt = np.random.randint(0, 4, (batch_size, n_max_instances))
    I_gt = np.random.randint(0, n_max_instances, (batch_size, num_points))
    P_torch = torch.from_numpy(P).float().to(device)
    W_torch = torch.from_numpy(W).float().to(device)
    X_torch = torch.from_numpy(X).float().to(device)
    points_per_instance_torch = torch.from_numpy(points_per_instance).float().to(device)
    T_gt_torch = torch.from_numpy(T_gt).long().to(device)
    I_gt_torch = torch.from_numpy(I_gt).long().to(device)
    parameters_torch = compute_parameters(P_torch, W_torch, X_torch)
    matching_indices_torch = hungarian_matching(W_torch, I_gt_torch)
    residue_loss_torch, residue_per_point_array_torch, residue_per_class_torch = compute_residue_loss(parameters_torch, matching_indices_torch, points_per_instance_torch, T_gt_torch, classes=['plane', 'sphere', 'cylinder', 'cone'])
    residue_loss_torch = residue_loss_torch.detach().cpu().numpy()
    residue_per_point_array_torch = residue_per_point_array_torch.detach().cpu().numpy()
    residue_per_class_torch = [elt.detach().cpu().numpy() for elt in residue_per_class_torch]
    print('residue_loss_torch', residue_loss_torch)
    print('residue_per_point_array_torch', residue_per_point_array_torch)
    # Debugging with Tensorflow
    P_tensorflow = tf.constant(P, dtype=tf.float32)
    W_tensorflow = tf.constant(W, dtype=tf.float32)
    X_tensorflow = tf.constant(X, dtype=tf.float32)
    points_per_instance_tensorflow = tf.constant(points_per_instance, dtype=tf.float32)
    T_gt_tensorflow = tf.constant(T_gt, dtype=tf.int32)
    I_gt_tensorflow = tf.constant(I_gt, dtype=tf.int32)
    parameters_tensorflow = compute_parameters_tensorflow(P_tensorflow, W_tensorflow, X_tensorflow)
    matching_indices_tensorflow = tf.stop_gradient(tf.py_func(hungarian_matching_tensorflow, [W_tensorflow, I_gt_tensorflow], Tout=tf.int32))
    residue_loss_tensorflow, residue_per_point_array_tensorflow, residue_per_class_tensorflow = compute_residue_loss_tensorflow(parameters_tensorflow, matching_indices_tensorflow, points_per_instance_tensorflow, T_gt_tensorflow, classes=['plane', 'sphere', 'cylinder', 'cone'])
    sess = tf.Session()
    residue_loss_tensorflow, residue_per_point_array_tensorflow, residue_per_class_tensorflow = sess.run([residue_loss_tensorflow, residue_per_point_array_tensorflow, residue_per_class_tensorflow])
    print(np.abs(residue_loss_tensorflow - residue_loss_torch).max())
    print(np.abs(residue_per_point_array_tensorflow - residue_per_point_array_torch).max())
    for i, class_ in enumerate(['plane', 'sphere', 'cylinder', 'cone']):
        print(class_, np.abs(residue_per_class_tensorflow[i] - residue_per_class_torch[i]).max())

def compute_parameter_loss(predicted_parameters, gt_parameters, matching_indices, T_gt, is_eval=False, classes=['plane','sphere','cylinder','cone']):
    parameter_losses = []  # a length T array of BxK tensors
    batch_size, n_max_instances = predicted_parameters[list(predicted_parameters.keys())[0]].size()[0:2]
    for class_ in classes:
        if class_ == 'plane':
            parameter_loss = plane_fitter.compute_parameter_loss(predicted_parameters['plane_normal'], gt_parameters['plane_normal'], matching_indices, angle_diff=is_eval)
        elif class_ == 'sphere':
            parameter_loss = torch.zeros([batch_size, n_max_instances], dtype=torch.float).to(T_gt.device)
        elif class_ == 'cylinder':
            parameter_loss = cylinder_fitter.compute_parameter_loss(predicted_parameters['cylinder_axis'], gt_parameters['cylinder_axis'], matching_indices, angle_diff=is_eval)
        elif class_ == 'cone':
            parameter_loss = cone_fitter.compute_parameter_loss(predicted_parameters['cone_axis'], gt_parameters['cone_axis'], matching_indices, angle_diff=is_eval)
        else:
            raise NotImplementedError
        parameter_losses.append(parameter_loss)
    parameter_losses = torch.stack(parameter_losses, dim=2)
    parameter_loss = torch.gather(parameter_losses, 2, T_gt.unsqueeze(2)).squeeze(2)  # BxK
    return parameter_loss

def compute_parameter_loss_tensorflow(predicted_parameters, gt_parameters, matching_indices, T_gt, is_eval=False, classes=['plane','sphere','cylinder','cone']):
    parameter_losses = []  # a length T array of BxK tensors
    for class_ in classes:
        if class_ == 'plane':
            parameter_loss = plane_fitter.compute_parameter_loss_tensorflow(predicted_parameters['plane_normal'], gt_parameters['plane_normal'], matching_indices, angle_diff=is_eval)
        elif class_ == 'sphere':
            parameter_loss = tf.zeros(dtype=tf.float32, shape=[batch_size, n_max_instances])
        elif class_ == 'cylinder':
            parameter_loss = cylinder_fitter.compute_parameter_loss_tensorflow(predicted_parameters['cylinder_axis'], gt_parameters['cylinder_axis'], matching_indices, angle_diff=is_eval)
        elif class_ == 'cone':
            parameter_loss = cone_fitter.compute_parameter_loss_tensorflow(predicted_parameters['cone_axis'], gt_parameters['cone_axis'], matching_indices, angle_diff=is_eval)
        else:
            raise NotImplementedError
        parameter_losses.append(parameter_loss)
    parameter_losses = tf.stack(parameter_losses, axis=2)
    parameter_loss = aggregate_loss_from_stacked_tensorflow(parameter_losses, T_gt)  # BxK
    return parameter_loss

if __name__ == '__main__':
    batch_size = 100
    num_points = 1024
    num_points_instance = 512
    n_max_instances = 12
    device = torch.device('cuda:0')
    np.random.seed(0)
    P = np.random.randn(batch_size, num_points, 3)
    W = np.random.rand(batch_size, num_points, n_max_instances)
    X = np.random.randn(batch_size, num_points, 3)
    X = X / np.linalg.norm(X, axis=2, keepdims=True)
    T_gt = np.random.randint(0, 4, (batch_size, n_max_instances))
    I_gt = np.random.randint(0, n_max_instances, (batch_size, num_points))
    plane_normal = np.random.randn(batch_size, n_max_instances, 3)
    plane_normal = plane_normal / np.linalg.norm(plane_normal, axis=2, keepdims=True)
    plane_center = np.random.randn(batch_size, n_max_instances)
    sphere_center = np.random.randn(batch_size, n_max_instances, 3)
    sphere_radius_squared = np.abs(np.random.randn(batch_size, n_max_instances))
    cylinder_axis = np.random.randn(batch_size, n_max_instances, 3)
    cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis, axis=2, keepdims=True)
    cylinder_center = np.random.randn(batch_size, n_max_instances, 3)
    cylinder_radius_square = np.abs(np.random.randn(batch_size, n_max_instances))
    cone_apex = np.random.randn(batch_size, n_max_instances, 3)
    cone_axis = np.random.randn(batch_size, n_max_instances, 3)
    cone_half_angle = np.abs(np.random.randn(batch_size, n_max_instances))
    gt_parameters = {'plane_normal': plane_normal,
                     'plane_center': plane_center,
                     'sphere_center': sphere_center,
                     'sphere_radius_squared': sphere_radius_squared,
                     'cylinder_axis': cylinder_axis,
                     'cylinder_center': cylinder_center,
                     'cylinder_radius_square': cylinder_radius_square,
                     'cone_apex': cone_apex,
                     'cone_axis': cone_axis,
                     'cone_half_angle': cone_half_angle}
    P_torch = torch.from_numpy(P).float().to(device)
    W_torch = torch.from_numpy(W).float().to(device)
    X_torch = torch.from_numpy(X).float().to(device)
    T_gt_torch = torch.from_numpy(T_gt).long().to(device)
    I_gt_torch = torch.from_numpy(I_gt).long().to(device)
    gt_parameters_torch = {'plane_normal': torch.from_numpy(gt_parameters['plane_normal']).float().to(device),
                           'plane_center': torch.from_numpy(gt_parameters['plane_center']).float().to(device),
                           'sphere_center': torch.from_numpy(gt_parameters['sphere_center']).float().to(device),
                           'sphere_radius_squared': torch.from_numpy(gt_parameters['sphere_radius_squared']).float().to(device),
                           'cylinder_axis': torch.from_numpy(gt_parameters['cylinder_axis']).float().to(device),
                           'cylinder_center': torch.from_numpy(gt_parameters['cylinder_center']).float().to(device),
                           'cylinder_radius_square': torch.from_numpy(gt_parameters['cylinder_radius_square']).float().to(device),
                           'cone_apex': torch.from_numpy(gt_parameters['cone_apex']).float().to(device),
                           'cone_axis': torch.from_numpy(gt_parameters['cone_axis']).float().to(device),
                           'cone_half_angle': torch.from_numpy(gt_parameters['cone_half_angle']).float().to(device)}
    predicted_parameters_torch = compute_parameters(P_torch, W_torch, X_torch)
    matching_indices_torch = hungarian_matching(W_torch, I_gt_torch)
    parameter_loss_torch = compute_parameter_loss(predicted_parameters_torch, gt_parameters_torch, matching_indices_torch, T_gt_torch, is_eval=False, classes=['plane','sphere','cylinder','cone'])
    parameter_loss_torch = parameter_loss_torch.detach().cpu().numpy()
    print('parameter_loss_torch', parameter_loss_torch)
    # Debugging with Tensorflow
    P_tensorflow = tf.constant(P, dtype=tf.float32)
    W_tensorflow = tf.constant(W, dtype=tf.float32)
    X_tensorflow = tf.constant(X, dtype=tf.float32)
    T_gt_tensorflow = tf.constant(T_gt, dtype=tf.int32)
    I_gt_tensorflow = tf.constant(I_gt, dtype=tf.int32)
    gt_parameters_tensorflow = {'plane_normal': tf.constant(gt_parameters['plane_normal'], dtype=tf.float32),
                                'plane_center': tf.constant(gt_parameters['plane_center'], dtype=tf.float32),
                                'sphere_center': tf.constant(gt_parameters['sphere_center'], dtype=tf.float32),
                                'sphere_radius_squared': tf.constant(gt_parameters['sphere_radius_squared'], dtype=tf.float32),
                                'cylinder_axis': tf.constant(gt_parameters['cylinder_axis'], dtype=tf.float32),
                                'cylinder_center': tf.constant(gt_parameters['cylinder_center'], dtype=tf.float32),
                                'cylinder_radius_square': tf.constant(gt_parameters['cylinder_radius_square'], dtype=tf.float32),
                                'cone_apex': tf.constant(gt_parameters['cone_apex'], dtype=tf.float32),
                                'cone_axis': tf.constant(gt_parameters['cone_axis'], dtype=tf.float32),
                                'cone_half_angle': tf.constant(gt_parameters['cone_half_angle'], dtype=tf.float32)}
    predicted_parameters_tensorflow = compute_parameters_tensorflow(P_tensorflow, W_tensorflow, X_tensorflow)
    matching_indices_tensorflow = tf.stop_gradient(tf.py_func(hungarian_matching_tensorflow, [W_tensorflow, I_gt_tensorflow], Tout=tf.int32))
    parameter_loss_tensorflow = compute_parameter_loss_tensorflow(predicted_parameters_tensorflow, gt_parameters_tensorflow, matching_indices_tensorflow, T_gt_tensorflow, is_eval=False, classes=['plane', 'sphere', 'cylinder', 'cone'])
    sess = tf.Session()
    parameter_loss_tensorflow = sess.run(parameter_loss_tensorflow)
    print(np.abs(parameter_loss_tensorflow - parameter_loss_torch).max())

def sequence_mask(lengths, maxlen=None):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = lengths.unsqueeze(dim=-1)
    mask = row_vector < matrix
    return mask

def get_mask_gt(I_gt, n_max_instances):
    n_instances_gt = torch.max(I_gt, dim=1)[0] + 1  # only count known primitive type instances, as -1 will be ignored
    mask_gt = sequence_mask(n_instances_gt, maxlen=n_max_instances)
    return mask_gt

def get_mask_gt_tensorflow(I_gt, n_max_instances):
    n_instances_gt = tf.reduce_max(I_gt, axis=1) + 1  # only count known primitive type instances, as -1 will be ignored
    mask_gt = tf.sequence_mask(n_instances_gt, maxlen=n_max_instances)
    return mask_gt

if __name__ == '__main__':
    batch_size = 100
    num_points = 1024
    n_max_instances = 12
    device = torch.device('cuda:0')
    np.random.seed(0)
    I_gt1 = np.random.randint(0, n_max_instances, (batch_size-batch_size//2, num_points))
    I_gt2 = np.random.randint(0, n_max_instances//2, (batch_size//2, num_points))
    I_gt = np.concatenate((I_gt1, I_gt2), axis=0)
    I_gt_torch = torch.from_numpy(I_gt).long().to(device)
    mask_gt_torch = get_mask_gt(I_gt_torch, n_max_instances)
    mask_gt_torch = mask_gt_torch.detach().cpu().numpy()
    print('mask_gt_torch', mask_gt_torch)
    # Debugging with Tensorflow
    I_gt_tensorflow = tf.constant(I_gt, dtype=tf.int32)
    mask_gt_tensorflow = get_mask_gt_tensorflow(I_gt_tensorflow, n_max_instances)
    sess = tf.Session()
    mask_gt_tensorflow = sess.run(mask_gt_tensorflow)
    print(np.all(mask_gt_torch == mask_gt_tensorflow))

def reduce_mean_masked_instance(loss, mask_gt):
    # loss: BxK
    loss = torch.where(mask_gt, loss, torch.zeros_like(loss))
    reduced_loss = torch.sum(loss, axis=1) # B
    denom = torch.sum(mask_gt.float(), dim=1) # B
    return torch.where(denom > 0, reduced_loss / denom, torch.zeros_like(reduced_loss)) # B

def collect_losses(normal_loss, normal_loss_multiplier, type_loss, type_loss_multiplier, avg_miou_loss, miou_loss, miou_loss_multiplier,
                   avg_residue_loss, residue_loss, residue_loss_multiplier, avg_parameter_loss, parameter_loss, parameter_loss_multiplier,
                   total_loss_multiplier):
    total_loss = 0
    # Normal Loss
    normal_loss_per_data = normal_loss
    total_normal_loss = torch.mean(normal_loss_per_data)
    if normal_loss_multiplier > 0:
        total_loss = total_loss + normal_loss_multiplier * total_normal_loss
    # Total loss
    type_loss_per_data = type_loss
    total_type_loss = torch.mean(type_loss_per_data)
    if type_loss_multiplier > 0:
        total_loss = total_loss + type_loss_multiplier * total_type_loss
    # mIoU Loss
    miou_loss_per_data = avg_miou_loss
    miou_loss_per_instance = miou_loss
    total_miou_loss = torch.mean(miou_loss_per_data)
    if miou_loss_multiplier > 0:
        total_loss = total_loss + miou_loss_multiplier * total_miou_loss
    # Residue Loss
    residue_loss_per_data = avg_residue_loss
    residue_loss_per_instance = residue_loss
    total_residue_loss = torch.mean(residue_loss_per_data)
    if residue_loss_multiplier > 0:
        total_loss = total_loss + residue_loss_multiplier * total_residue_loss
    # Paramerer Loss
    parameter_loss_per_data = avg_parameter_loss
    parameter_loss_per_instance = parameter_loss
    total_parameter_loss = torch.mean(parameter_loss_per_data)
    if parameter_loss_multiplier > 0:
        total_loss = total_loss + parameter_loss_multiplier * total_parameter_loss
    total_loss = total_loss * total_loss_multiplier
    return total_loss, total_normal_loss, total_type_loss, total_miou_loss, total_residue_loss, total_parameter_loss

def compute_all_losses(P, W, I_gt, X, X_gt, T, T_gt, gt_parameters, points_per_instance,
                       normal_loss_multiplier, type_loss_multiplier, miou_loss_multiplier, residue_loss_multiplier, parameter_loss_multiplier, total_loss_multiplier, is_eval,
                       mode_seg='mIoU', classes=['plane','sphere','cylinder','cone']):
    assert(mode_seg in ['mIoU', 'intersection'])
    batch_size, _, n_max_instances = W.size()
    matching_indices = hungarian_matching(W, I_gt)
    if (residue_loss_multiplier>0) or (parameter_loss_multiplier>0):
        predicted_parameters = compute_parameters(P, W, X)
    mask_gt = get_mask_gt(I_gt, n_max_instances)
    if normal_loss_multiplier>0:
        normal_loss = compute_normal_loss(X, X_gt, angle_diff=is_eval)
    else:
        normal_loss = torch.zeros([batch_size, n_max_instances]).to(P.device)
    if type_loss_multiplier>0:
        type_loss = compute_per_point_type_loss(T, I_gt, T_gt, is_eval)
    else:
        type_loss = torch.zeros([batch_size, n_max_instances]).to(P.device)
    if (mode_seg == 'mIoU') and (miou_loss_multiplier>0):
        miou_loss, _ = compute_miou_loss(W, I_gt, matching_indices)
        avg_miou_loss = reduce_mean_masked_instance(miou_loss, mask_gt)
    elif (mode_seg == 'intersection') and (miou_loss_multiplier>0):
        _, miou_loss = compute_miou_loss(W, I_gt, matching_indices)
        avg_miou_loss = reduce_mean_masked_instance(miou_loss, mask_gt)
    else:
        miou_loss = torch.zeros([batch_size, n_max_instances]).to(P.device)
        avg_miou_loss = torch.zeros([batch_size]).to(P.device)
    if residue_loss_multiplier>0:
        residue_loss, residue_per_point_array = compute_residue_loss(predicted_parameters, matching_indices, points_per_instance, T_gt, classes=classes)
        avg_residue_loss = reduce_mean_masked_instance(residue_loss, mask_gt)
    else:
        residue_loss = torch.zeros([batch_size, n_max_instances]).to(P.device)
        avg_residue_loss = torch.zeros([batch_size]).to(P.device)
    if parameter_loss_multiplier>0:
        parameter_loss = compute_parameter_loss(predicted_parameters, gt_parameters, matching_indices, T_gt, is_eval, classes=classes)
        avg_parameter_loss = reduce_mean_masked_instance(parameter_loss, mask_gt)
    else:
        parameter_loss = torch.zeros([batch_size, n_max_instances]).to(P.device)
        avg_parameter_loss = torch.zeros([batch_size]).to(P.device)
    total_loss, total_normal_loss, total_type_loss, total_miou_loss, total_residue_loss, total_parameter_loss = \
        collect_losses(normal_loss, normal_loss_multiplier, type_loss, type_loss_multiplier, avg_miou_loss, miou_loss, miou_loss_multiplier,
                       avg_residue_loss, residue_loss, residue_loss_multiplier, avg_parameter_loss, parameter_loss, parameter_loss_multiplier,
                       total_loss_multiplier)
    if (residue_loss_multiplier > 0) or (parameter_loss_multiplier > 0):
        return total_loss, total_normal_loss, total_type_loss, total_miou_loss, total_residue_loss, total_parameter_loss, predicted_parameters['plane_normal'], predicted_parameters['cylinder_axis'], predicted_parameters['cone_axis']
    else:
        return total_loss, total_normal_loss, total_type_loss, total_miou_loss, total_residue_loss, total_parameter_loss, None, None, None

def reduce_mean_masked_instance_tensorflow(loss, mask_gt):
    # loss: BxK
    loss = tf.where(mask_gt, loss, tf.zeros_like(loss))
    reduced_loss = tf.reduce_sum(loss, axis=1) # B
    denom = tf.reduce_sum(tf.to_float(mask_gt), axis=1) # B
    return tf.where(denom > 0, reduced_loss / denom, tf.zeros_like(reduced_loss)) # B

def collect_losses_tensorflow(normal_loss, normal_loss_multiplier, type_loss, type_loss_multiplier, avg_miou_loss, miou_loss, miou_loss_multiplier,
                              avg_residue_loss, residue_loss, residue_loss_multiplier, avg_parameter_loss, parameter_loss, parameter_loss_multiplier,
                              total_loss_multiplier):
    total_loss = tf.zeros(shape=[], dtype=tf.float32)
    normal_loss_per_data = normal_loss
    total_normal_loss = tf.reduce_mean(normal_loss_per_data)
    if normal_loss_multiplier > 0:
        total_loss += normal_loss_multiplier * total_normal_loss
    type_loss_per_data = type_loss
    total_type_loss = tf.reduce_mean(type_loss_per_data)
    if type_loss_multiplier > 0:
        total_loss += type_loss_multiplier * total_type_loss
    miou_loss_per_data = avg_miou_loss
    miou_loss_per_instance = miou_loss
    total_miou_loss = tf.reduce_mean(miou_loss_per_data)
    if miou_loss_multiplier > 0:
        total_loss += miou_loss_multiplier * total_miou_loss
    residue_loss_per_data = avg_residue_loss
    residue_loss_per_instance = residue_loss
    total_residue_loss = tf.reduce_mean(residue_loss_per_data)
    if residue_loss_multiplier > 0:
        total_loss += residue_loss_multiplier * total_residue_loss
    parameter_loss_per_data = avg_parameter_loss
    parameter_loss_per_instance = parameter_loss
    total_parameter_loss = tf.reduce_mean(parameter_loss_per_data)
    if parameter_loss_multiplier > 0:
        total_loss += parameter_loss_multiplier * total_parameter_loss
    total_loss *= total_loss_multiplier
    return total_loss

def compute_all_losses_tensorflow(P, W, I_gt, X, X_gt, T, T_gt, gt_parameters, points_per_instance,
                                  normal_loss_multiplier, type_loss_multiplier, miou_loss_multiplier, residue_loss_multiplier, parameter_loss_multiplier, total_loss_multiplier, is_eval,
                                  classes=['plane','sphere','cylinder','cone']):
    b_max_instances = W.shape[2]
    matching_indices = tf.stop_gradient(tf.py_func(hungarian_matching_tensorflow, [W, I_gt], Tout=tf.int32))
    predicted_parameters = compute_parameters_tensorflow(P, W, X)
    mask_gt = get_mask_gt_tensorflow(I_gt, n_max_instances)
    normal_loss = compute_normal_loss_tensorflow(X, X_gt, angle_diff=is_eval)
    type_loss = compute_per_point_type_loss_tensorflow(T, I_gt, T_gt, is_eval)
    miou_loss = compute_miou_loss_tensorflow(W, I_gt, matching_indices)
    avg_miou_loss = reduce_mean_masked_instance_tensorflow(miou_loss, mask_gt)
    residue_loss, residue_per_point_array = compute_residue_loss_tensorflow(predicted_parameters, matching_indices, points_per_instance, T_gt, classes=classes)
    avg_residue_loss = reduce_mean_masked_instance_tensorflow(residue_loss, mask_gt)
    parameter_loss = compute_parameter_loss_tensorflow(predicted_parameters, gt_parameters, matching_indices, T_gt, is_eval, classes=classes)
    avg_parameter_loss = reduce_mean_masked_instance_tensorflow(parameter_loss, mask_gt)
    total_loss = collect_losses_tensorflow(normal_loss, normal_loss_multiplier, type_loss, type_loss_multiplier, avg_miou_loss, miou_loss, miou_loss_multiplier,
                                avg_residue_loss, residue_loss, residue_loss_multiplier, avg_parameter_loss, parameter_loss, parameter_loss_multiplier,
                                total_loss_multiplier)
    return total_loss

if __name__ == '__main__':
    batch_size = 100
    num_points = 1024
    num_points_instance = 512
    n_max_instances = 12
    device = torch.device('cuda:0')
    np.random.seed(0)
    P = np.random.randn(batch_size, num_points, 3)
    W = np.random.rand(batch_size, num_points, n_max_instances)
    X = np.random.randn(batch_size, num_points, 3)
    X = X / np.linalg.norm(X, axis=2, keepdims=True)
    X_gt = np.random.randn(batch_size, num_points, 3)
    X_gt = X_gt / np.linalg.norm(X_gt, axis=2, keepdims=True)
    T = np.random.rand(batch_size, num_points, 4)
    T_gt = np.random.randint(0, 4, (batch_size, n_max_instances))
    I_gt = np.random.randint(0, n_max_instances, (batch_size, num_points))
    plane_normal = np.random.randn(batch_size, n_max_instances, 3)
    plane_normal = plane_normal / np.linalg.norm(plane_normal, axis=2, keepdims=True)
    plane_center = np.random.randn(batch_size, n_max_instances)
    sphere_center = np.random.randn(batch_size, n_max_instances, 3)
    sphere_radius_squared = np.abs(np.random.randn(batch_size, n_max_instances))
    cylinder_axis = np.random.randn(batch_size, n_max_instances, 3)
    cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis, axis=2, keepdims=True)
    cylinder_center = np.random.randn(batch_size, n_max_instances, 3)
    cylinder_radius_square = np.abs(np.random.randn(batch_size, n_max_instances))
    cone_apex = np.random.randn(batch_size, n_max_instances, 3)
    cone_axis = np.random.randn(batch_size, n_max_instances, 3)
    cone_half_angle = np.abs(np.random.randn(batch_size, n_max_instances))
    points_per_instance = np.random.randn(batch_size, n_max_instances, num_points_instance, 3)
    normal_loss_multiplier = 1.0
    type_loss_multiplier = 1.0
    miou_loss_multiplier = 1.0
    residue_loss_multiplier = 1.0
    parameter_loss_multiplier = 1.0
    total_loss_multiplier = 1.0
    is_eval = False
    gt_parameters = {'plane_normal': plane_normal,
                     'plane_center': plane_center,
                     'sphere_center': sphere_center,
                     'sphere_radius_squared': sphere_radius_squared,
                     'cylinder_axis': cylinder_axis,
                     'cylinder_center': cylinder_center,
                     'cylinder_radius_square': cylinder_radius_square,
                     'cone_apex': cone_apex,
                     'cone_axis': cone_axis,
                     'cone_half_angle': cone_half_angle}
    P_torch = torch.from_numpy(P).float().to(device)
    W_torch = torch.from_numpy(W).float().to(device)
    X_torch = torch.from_numpy(X).float().to(device)
    X_gt_torch = torch.from_numpy(X_gt).float().to(device)
    T_torch = torch.from_numpy(T).float().to(device)
    T_gt_torch = torch.from_numpy(T_gt).long().to(device)
    I_gt_torch = torch.from_numpy(I_gt).long().to(device)
    gt_parameters_torch = {'plane_normal': torch.from_numpy(gt_parameters['plane_normal']).float().to(device),
                           'plane_center': torch.from_numpy(gt_parameters['plane_center']).float().to(device),
                           'sphere_center': torch.from_numpy(gt_parameters['sphere_center']).float().to(device),
                           'sphere_radius_squared': torch.from_numpy(gt_parameters['sphere_radius_squared']).float().to(device),
                           'cylinder_axis': torch.from_numpy(gt_parameters['cylinder_axis']).float().to(device),
                           'cylinder_center': torch.from_numpy(gt_parameters['cylinder_center']).float().to(device),
                           'cylinder_radius_square': torch.from_numpy(gt_parameters['cylinder_radius_square']).float().to(device),
                           'cone_apex': torch.from_numpy(gt_parameters['cone_apex']).float().to(device),
                           'cone_axis': torch.from_numpy(gt_parameters['cone_axis']).float().to(device),
                           'cone_half_angle': torch.from_numpy(gt_parameters['cone_half_angle']).float().to(device)}
    points_per_instance_torch = torch.from_numpy(points_per_instance).long().to(device)
    total_loss_torch = compute_all_losses(P_torch, W_torch, I_gt_torch, X_torch, X_gt_torch, T_torch, T_gt_torch, gt_parameters_torch, points_per_instance_torch,
                                          normal_loss_multiplier, type_loss_multiplier, miou_loss_multiplier, residue_loss_multiplier, parameter_loss_multiplier,
                                          total_loss_multiplier, is_eval)[0]
    total_loss_torch = total_loss_torch.detach().cpu().numpy()
    print('total_loss_torch', total_loss_torch)
    # Debugging with Tensorflow
    P_tensorflow = tf.constant(P, dtype=tf.float32)
    W_tensorflow = tf.constant(W, dtype=tf.float32)
    X_tensorflow = tf.constant(X, dtype=tf.float32)
    X_gt_tensorflow = tf.constant(X_gt, dtype=tf.float32)
    T_tensorflow = tf.constant(T, dtype=tf.float32)
    T_gt_tensorflow = tf.constant(T_gt, dtype=tf.int32)
    I_gt_tensorflow =  tf.constant(I_gt, dtype=tf.int32)
    gt_parameters_tensorflow = {'plane_normal': tf.constant(gt_parameters['plane_normal'], dtype=tf.float32),
                                'plane_center': tf.constant(gt_parameters['plane_center'], dtype=tf.float32),
                                'sphere_center': tf.constant(gt_parameters['sphere_center'], dtype=tf.float32),
                                'sphere_radius_squared': tf.constant(gt_parameters['sphere_radius_squared'], dtype=tf.float32),
                                'cylinder_axis': tf.constant(gt_parameters['cylinder_axis'], dtype=tf.float32),
                                'cylinder_center': tf.constant(gt_parameters['cylinder_center'], dtype=tf.float32),
                                'cylinder_radius_square': tf.constant(gt_parameters['cylinder_radius_square'], dtype=tf.float32),
                                'cone_apex': tf.constant(gt_parameters['cone_apex'], dtype=tf.float32),
                                'cone_axis': tf.constant(gt_parameters['cone_axis'], dtype=tf.float32),
                                'cone_half_angle': tf.constant(gt_parameters['cone_half_angle'], dtype=tf.float32)}
    points_per_instance_tensorflow = tf.constant(points_per_instance, dtype=tf.float32)
    total_loss_tensorflow = compute_all_losses_tensorflow(P_tensorflow, W_tensorflow, I_gt_tensorflow, X_tensorflow, X_gt_tensorflow, T_tensorflow, T_gt_tensorflow, gt_parameters_tensorflow, points_per_instance_tensorflow,
                                                           normal_loss_multiplier, type_loss_multiplier, miou_loss_multiplier, residue_loss_multiplier, parameter_loss_multiplier,
                                                           total_loss_multiplier, is_eval)
    sess = tf.Session()
    total_loss_tensorflow = sess.run(total_loss_tensorflow)
    print(np.abs(total_loss_tensorflow - total_loss_torch).max())