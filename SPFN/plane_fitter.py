# Importatiomn of packages
import torch
import numpy as np
if __name__ == '__main__':
    import tensorflow as tf
from SPFN.primitives import Plane
from SPFN.geometry_utils import weighted_plane_fitting, weighted_plane_fitting_tensorflow

def compute_parameters(P, W):
    batch_size, n_points, _ = P.size()
    _, _, n_max_instances = W.size()
    W_reshaped = W.transpose(1, 2).contiguous().view(batch_size * n_max_instances, n_points)
    P_tiled = P.unsqueeze(1).expand(batch_size, n_max_instances, n_points, 3).contiguous().view(batch_size * n_max_instances, n_points, 3)
    n, c = weighted_plane_fitting(P_tiled, W_reshaped)  # BKx3
    n = n.view(batch_size, n_max_instances, 3)
    c = c.view(batch_size, n_max_instances)
    return n, c

def compute_parameters_tensorflow(P, W):
    batch_size = tf.shape(P)[0]
    n_points = tf.shape(P)[1]
    n_max_instances = tf.shape(W)[2]
    W_reshaped = tf.reshape(tf.transpose(W, [0, 2, 1]), [batch_size * n_max_instances, n_points])  # BKxN
    P_tiled = tf.reshape(tf.tile(tf.expand_dims(P, axis=1), [1, n_max_instances, 1, 1]), [batch_size * n_max_instances, n_points, 3])  # BKxNx3, important there to match indices in W_reshaped!!!
    n, c = weighted_plane_fitting_tensorflow(P_tiled, W_reshaped)  # BKx3
    n = tf.reshape(n, [batch_size, n_max_instances, 3])  # BxKx3
    c = tf.reshape(c, [batch_size, n_max_instances])  # BxK
    return n, c

if __name__ == '__main__':
    batch_size = 100
    num_points = 1024
    n_max_instances = 12
    device = torch.device('cuda:0')
    np.random.seed(0)
    P = np.random.randn(batch_size, num_points, 3)
    W = np.random.rand(batch_size, num_points, n_max_instances)
    P_torch = torch.from_numpy(P).float().to(device)
    W_torch = torch.from_numpy(W).float().to(device)
    n_torch, c_torch = compute_parameters(P_torch, W_torch)
    n_torch = n_torch.detach().cpu().numpy()
    c_torch = c_torch.detach().cpu().numpy()
    print('n_torch', n_torch)
    print('c_torch', c_torch)
    # Debugging with Tensorflow
    P_tensorflow = tf.constant(P, dtype=tf.float32)
    W_tensorflow = tf.constant(W, dtype=tf.float32)
    n_tensorflow, c_tensorflow = compute_parameters_tensorflow(P_tensorflow, W_tensorflow)
    sess = tf.Session()
    n_tensorflow, c_tensorflow = sess.run([n_tensorflow, c_tensorflow])
    print(np.minimum(np.abs(n_tensorflow - n_torch), np.abs(n_tensorflow + n_torch)).max())
    print(np.minimum(np.abs(c_tensorflow - c_torch), np.abs(c_tensorflow + c_torch)).max())

def compute_residue_single(n, c, p):
    return (torch.sum(p * n, dim=-1) - c)**2

def compute_residue_single_tensorflow(n, c, p):
    # n: ...x3, c: ..., p: ...x3
    return tf.square(tf.reduce_sum(p * n, axis=-1) - c)

if __name__ == '__main__':
    batch_size = 100
    num_points = 1024
    device = torch.device('cuda:0')
    np.random.seed(0)
    n = np.random.randn(batch_size, num_points, 3)
    c = np.random.rand(batch_size, num_points)
    p = np.random.rand(batch_size, num_points, 3)
    n_torch = torch.from_numpy(n).float().to(device)
    c_torch = torch.from_numpy(c).float().to(device)
    p_torch = torch.from_numpy(p).float().to(device)
    residue_loss_torch = compute_residue_single(n_torch, c_torch, p_torch)
    residue_loss_torch = residue_loss_torch.detach().cpu().numpy()
    print('residue_loss_torch', residue_loss_torch)
    # Debugging with Tensorflow
    n_tensorflow = tf.constant(n, dtype=tf.float32)
    c_tensorflow = tf.constant(c, dtype=tf.float32)
    p_tensorflow = tf.constant(p, dtype=tf.float32)
    residue_loss_torch_tensorflow = compute_residue_single_tensorflow(n_tensorflow, c_tensorflow, p_tensorflow)
    sess = tf.Session()
    residue_loss_torch_tensorflow = sess.run(residue_loss_torch_tensorflow)
    print(np.abs(residue_loss_torch_tensorflow-residue_loss_torch).max())

def acos_safe(x):
    return torch.acos(torch.clamp(x, min=-1.0+1e-6, max=1.0-1e-6))

def compute_parameter_loss(predicted_n, gt_n, matching_indices, angle_diff):
    # predicted_axis: BxK1x3
    # gt_axis: BXK2x3
    # matching indices: BxK2
    batch_size, nb_primitives, _ = gt_n.size()
    predicted_n = torch.gather(predicted_n, 1, matching_indices.unsqueeze(2).expand(batch_size, nb_primitives, 3))
    dot_abs = torch.abs(torch.sum(predicted_n * gt_n, axis=2))
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

def acos_safe_tensorflow(x):
    return tf.math.acos(tf.clip_by_value(x, -1.0+1e-6, 1.0-1e-6))

def compute_parameter_loss_tensorflow(predicted_n, gt_n, matching_indices, angle_diff):
    n = batched_gather(predicted_n, matching_indices, axis=1)
    dot_abs = tf.abs(tf.reduce_sum(n * gt_n, axis=2))
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
    loss_tensorflow = compute_parameter_loss_tensorflow(predicted_axis_tensorflow, gt_axis_tensorflow,
                                                        matching_indices_tensorflow, angle_diff)
    sess = tf.Session()
    loss_tensorflow = sess.run(loss_tensorflow)
    print(np.abs(loss_torch - loss_tensorflow).max())

def create_primitive_from_dict(d):
    assert d['type'] == 'plane'
    location = np.array([d['location_x'], d['location_y'], d['location_z']], dtype=float)
    axis = np.array([d['axis_x'], d['axis_y'], d['axis_z']], dtype=float)
    return Plane(n=axis, c=np.dot(location, axis))

def extract_parameter_data_as_dict(primitives, n_max_instances):
    n = np.zeros(dtype=float, shape=[n_max_instances, 3])
    for i, primitive in enumerate(primitives):
        if isinstance(primitive, Plane):
            n[i] = primitive.n
    return {
        'plane_n_gt': n
    }

def extract_predicted_parameters_as_json(plane_normal, plane_center, k):
    # This is only for a single plane
    plane = Plane(plane_normal, plane_center)
    json_info = {
        'type': 'plane',
        'center_x': float(plane.center[0]),
        'center_y': float(plane.center[1]),
        'center_z': float(plane.center[2]),
        'normal_x': float(plane.n[0]),
        'normal_y': float(plane.n[1]),
        'normal_z': float(plane.n[2]),
        'x_size': float(plane.x_range[1] - plane.x_range[0]),
        'y_size': float(plane.y_range[1] - plane.y_range[0]),
        'x_axis_x': float(plane.x_axis[0]),
        'x_axis_y': float(plane.x_axis[1]),
        'x_axis_z': float(plane.x_axis[2]),
        'y_axis_x': float(plane.y_axis[0]),
        'y_axis_y': float(plane.y_axis[1]),
        'y_axis_z': float(plane.y_axis[2]),
        'label': k,
    }
    return json_info