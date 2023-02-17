# Importation of packages
import torch
import numpy as np
if __name__ == '__main__':
    import tensorflow as tf
from SPFN.primitives import Cone
from SPFN.geometry_utils import guarded_matrix_solve_ls, guarded_matrix_solve_ls_tensorflow, weighted_plane_fitting, weighted_plane_fitting_tensorflow

def acos_safe(x):
    return torch.acos(torch.clamp(x, min=-1.0+1e-6, max=1.0-1e-6))

def compute_parameters(P, W, X, div_eps=1e-10):
    batch_size, n_points, _ = P.size()
    _, _, n_max_instances = W.size()
    W_reshaped = W.transpose(1,2).contiguous().view(batch_size * n_max_instances, n_points)  # BKxN
    # A - BKxNx3
    A = X.unsqueeze(1).expand(batch_size, n_max_instances, n_points, 3).contiguous().view(batch_size * n_max_instances, n_points, 3)
    # b - BKxNx1
    b = torch.sum(P * X, dim=2).unsqueeze(1).expand(batch_size, n_max_instances, n_points).contiguous().view(batch_size * n_max_instances, n_points, 1)
    apex = guarded_matrix_solve_ls(A, b, W_reshaped).view(batch_size, n_max_instances, 3)  # BxKx3
    X_tiled = A
    # TODO: use P-apex instead of X for plane fitting
    plane_n, plane_c = weighted_plane_fitting(X_tiled, W_reshaped)
    axis = plane_n.view(batch_size, n_max_instances, 3)  # BxKx3
    P_minus_apex = P.unsqueeze(2) - apex.unsqueeze(1)  # BxNxKx3
    P_minus_apex_normalized = torch.nn.functional.normalize(P_minus_apex, p=2, dim=3, eps=1e-12)
    P_minus_apex_normalized_dot_axis = torch.sum(axis.unsqueeze(1) * P_minus_apex_normalized, dim=3)  # BxNxK
    # flip direction of axis if wrong
    sgn_axis = torch.sign(torch.sum(W * P_minus_apex_normalized_dot_axis, dim=1))  # BxK
    sgn_axis = sgn_axis + (sgn_axis==0.0).float()  # prevent sgn == 0
    axis = axis * sgn_axis.unsqueeze(2)  # BxKx3
    tmp = W * acos_safe(torch.abs(P_minus_apex_normalized_dot_axis))  # BxNxK
    W_sum = torch.sum(W, dim=1)  # BxK
    half_angle = torch.sum(tmp, dim=1) / (W_sum + div_eps)  # BxK
    half_angle = torch.clamp(half_angle, min=1e-3, max=np.pi/2-1e-3)  # angle cannot be too weird
    return apex, axis, half_angle

def acos_safe_tensorflow(x):
    return tf.math.acos(tf.clip_by_value(x, -1.0+1e-6, 1.0-1e-6))

def compute_parameters_tensorflow(P, W, X):
    batch_size = tf.shape(P)[0]
    n_points = tf.shape(P)[1]
    n_max_instances = W.get_shape()[2]
    W_reshaped = tf.reshape(tf.transpose(W, [0, 2, 1]), [batch_size * n_max_instances, n_points])  # BKxN
    # A - BKxNx3
    A = tf.reshape(tf.tile(tf.expand_dims(X, axis=1), [1, n_max_instances, 1, 1]), [batch_size * n_max_instances, n_points, 3])  # BKxNx3
    # b - BKxNx1
    b = tf.expand_dims(tf.reshape(tf.tile(tf.expand_dims(tf.reduce_sum(P * X, axis=2), axis=1), [1, n_max_instances, 1]), [batch_size * n_max_instances, n_points]), axis=2)
    apex = tf.reshape(guarded_matrix_solve_ls_tensorflow(A, b, W_reshaped), [batch_size, n_max_instances, 3])  # BxKx3
    X_tiled = A
    # TODO: use P-apex instead of X for plane fitting
    plane_n, plane_c = weighted_plane_fitting_tensorflow(X_tiled, W_reshaped)
    axis = tf.reshape(plane_n, [batch_size, n_max_instances, 3])  # BxKx3
    P_minus_apex_normalized = tf.nn.l2_normalize(tf.expand_dims(P, axis=2) - tf.expand_dims(apex, 1), axis=3)  # BxNxKx3
    P_minus_apex_normalized_dot_axis = tf.reduce_sum(tf.expand_dims(axis, axis=1) * P_minus_apex_normalized, axis=3)  # BxNxK
    # flip direction of axis if wrong
    sgn_axis = tf.sign(tf.reduce_sum(W * P_minus_apex_normalized_dot_axis, axis=1))  # BxK
    sgn_axis += tf.to_float(tf.equal(sgn_axis, 0.0))  # prevent sgn == 0
    axis *= tf.expand_dims(sgn_axis, axis=2)  # BxKx3
    tmp = W * acos_safe_tensorflow(tf.abs(P_minus_apex_normalized_dot_axis))  # BxNxK
    W_sum = tf.reduce_sum(W, axis=1)  # BxK
    half_angle = tf.reduce_sum(tmp, axis=1) / W_sum  # BxK
    tf.clip_by_value(half_angle, 1e-3, np.pi / 2 - 1e-3)  # angle cannot be too weird
    return apex, axis, half_angle

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
    apex_torch, axis_torch, half_angle_torch = compute_parameters(P_torch, W_torch, X_torch)
    apex_torch = apex_torch.detach().cpu().numpy()
    axis_torch = axis_torch.detach().cpu().numpy()
    half_angle_torch = half_angle_torch.detach().cpu().numpy()
    print('apex_torch', apex_torch)
    print('axis_torch', axis_torch)
    print('half_angle_torch', half_angle_torch)
    # Debugging with Tensorflow
    P_tensorflow = tf.constant(P, dtype=tf.float32)
    W_tensorflow = tf.constant(W, dtype=tf.float32)
    X_tensorflow = tf.constant(X, dtype=tf.float32)
    apex_tensorflow, axis_tensorflow, half_angle_tensorflow = compute_parameters_tensorflow(P_tensorflow, W_tensorflow, X_tensorflow)
    sess = tf.Session()
    apex_tensorflow, axis_tensorflow, half_angle_tensorflow = sess.run([apex_tensorflow, axis_tensorflow, half_angle_tensorflow])
    print(np.abs(apex_tensorflow - apex_torch).max())
    print(np.minimum(np.abs(axis_tensorflow - axis_torch), np.abs(axis_tensorflow + axis_torch)).max())
    print(np.abs(half_angle_tensorflow - half_angle_torch).max())

def compute_residue_single(apex, axis, half_angle, p):
    # apex: ...x3, axis: ...x3, half_angle: ..., p: ...x30
    v = p - apex
    v_normalized = torch.nn.functional.normalize(v, p=2, dim=-1, eps=1e-12)
    alpha = acos_safe(torch.sum(v_normalized * axis, dim=-1))
    return (torch.sin(torch.clamp(torch.abs(alpha - half_angle), min=None, max=np.pi / 2)))**2 * torch.sum(v * v, dim=-1)

def compute_residue_single_tensorflow(apex, axis, half_angle, p):
    # apex: ...x3, axis: ...x3, half_angle: ..., p: ...x3
    v = p - apex
    v_normalized = tf.nn.l2_normalize(v, axis=-1)
    alpha = acos_safe_tensorflow(tf.reduce_sum(v_normalized * axis, axis=-1))
    return tf.square(tf.sin(tf.minimum(tf.abs(alpha - half_angle), np.pi / 2))) * tf.reduce_sum(v * v, axis=-1)

if __name__ == '__main__':
    batch_size = 100
    num_points = 1024
    device = torch.device('cuda:0')
    np.random.seed(0)
    apex = np.random.randn(batch_size, num_points, 3)
    axis = np.random.randn(batch_size, num_points, 3)
    half_angle = np.random.randn(batch_size, num_points)
    p = np.random.randn(batch_size, num_points, 3)
    apex_torch = torch.from_numpy(apex).float().to(device)
    axis_torch = torch.from_numpy(axis).float().to(device)
    half_angle_torch = torch.from_numpy(half_angle).float().to(device)
    p_torch = torch.from_numpy(p).float().to(device)
    loss_torch = compute_residue_single(apex_torch, axis_torch, half_angle_torch, p_torch)
    loss_torch = loss_torch.detach().cpu().numpy()
    print('loss_torch', loss_torch)
    # Debugging with Tensorflow
    apex_tensorflow = tf.constant(apex, dtype=tf.float32)
    axis_tensorflow = tf.constant(axis, dtype=tf.float32)
    half_angle_tensorflow = tf.constant(half_angle, dtype=tf.float32)
    p_tensorflow = tf.constant(p, dtype=tf.float32)
    loss_tensorflow = compute_residue_single_tensorflow(apex_tensorflow, axis_tensorflow, half_angle_tensorflow, p_tensorflow)
    sess = tf.Session()
    loss_tensorflow = sess.run(loss_tensorflow)
    print(np.abs(loss_torch - loss_tensorflow).max())

def compute_parameter_loss(predicted_axis, gt_axis, matching_indices, angle_diff):
    # predicted_axis: BxK1x3
    # gt_axis: BXK2x3
    # matching indices: BxK2
    batch_size, nb_primitives, _ = gt_axis.size()
    predicted_axis = torch.gather(predicted_axis, 1, matching_indices.unsqueeze(2).expand(batch_size, nb_primitives, 3))
    dot_abs = torch.abs(torch.sum(predicted_axis * gt_axis, axis=2))
    if angle_diff:
        return acos_safe(dot_abs)  # BxK
    else:
        return 1.0 - dot_abs  # BxK

def batched_gather(data, indices, axis):
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

def compute_parameter_loss_tensorflow(predicted_axis, gt_axis, matching_indices, angle_diff):
    axis = batched_gather(predicted_axis, matching_indices, axis=1)
    dot_abs = tf.abs(tf.reduce_sum(axis * gt_axis, axis=2))
    if angle_diff:
        return acos_safe_tensorflow(dot_abs)  # BxK
    else:
        return 1.0 - dot_abs  # BxK

if __name__ == '__main__':
    batch_size = 100
    num_primitives1 = 15
    num_primitives2 = 5
    device = torch.device('cuda:0')
    np.random.seed(0)
    predicted_axis = np.random.randn(batch_size, num_primitives1, 3)
    gt_axis = np.random.randn(batch_size, num_primitives2, 3)
    matching_indices = np.random.randint(0, 15, (batch_size, num_primitives2))
    angle_diff = True
    predicted_axis_torch = torch.from_numpy(predicted_axis).float().to(device)
    gt_axis_torch = torch.from_numpy(gt_axis).float().to(device)
    matching_indices_torch = torch.from_numpy(matching_indices).long().to(device)
    loss_torch = compute_parameter_loss(predicted_axis_torch, gt_axis_torch, matching_indices_torch, angle_diff)
    loss_torch = loss_torch.detach().cpu().numpy()
    print('loss_torch', loss_torch)
    # Debugging with Tensorflow
    predicted_axis_tensorflow = tf.constant(predicted_axis, dtype=tf.float32)
    gt_axis_tensorflow = tf.constant(gt_axis, dtype=tf.float32)
    matching_indices_tensorflow = tf.constant(matching_indices, dtype=tf.int32)
    loss_tensorflow = compute_parameter_loss_tensorflow(predicted_axis_tensorflow, gt_axis_tensorflow, matching_indices_tensorflow, angle_diff)
    sess = tf.Session()
    loss_tensorflow = sess.run(loss_tensorflow)
    print(np.abs(loss_torch - loss_tensorflow).max())

def create_primitive_from_dict(d):
    assert d['type'] == 'cone'
    apex = np.array([d['apex_x'], d['apex_y'], d['apex_z']], dtype=float)
    axis = np.array([d['axis_x'], d['axis_y'], d['axis_z']], dtype=float)
    half_angle = float(d['semi_angle'])
    return Cone(apex=apex, axis=axis, half_angle=half_angle)

def extract_parameter_data_as_dict(primitives, n_max_instances):
    axis_gt = np.zeros(dtype=float, shape=[n_max_instances, 3])
    apex_gt = np.zeros(dtype=float, shape=[n_max_instances, 3])
    half_angle_gt = np.zeros(dtype=float, shape=[n_max_instances])
    for i, primitive in enumerate(primitives):
        if isinstance(primitive, Cone):
            axis_gt[i] = primitive.axis
            apex_gt[i] = primitive.apex
            half_angle_gt[i] = primitive.half_angle
    return {
        'cone_axis_gt': axis_gt,
    }

def extract_predicted_parameters_as_json(cone_apex, cone_axis, cone_half_angle, k):
    cone = Cone(cone_apex, cone_axis, cone_half_angle, z_min=0.0, z_max=5.0)
    return {
        'type': 'cone',
        'apex_x': float(cone.apex[0]),
        'apex_y': float(cone.apex[1]),
        'apex_z': float(cone.apex[2]),
        'axis_x': float(cone.axis[0]),
        'axis_y': float(cone.axis[1]),
        'axis_z': float(cone.axis[2]),
        'angle': float(cone.half_angle * 2),
        'z_min': float(cone.z_min),
        'z_max': float(cone.z_max),
        'label': k,
    }