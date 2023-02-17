# Importation of pqckqges
import torch
import numpy as np
if __name__ == '__main__':
    import tensorflow as tf
from SPFN.primitives import Cylinder
from SPFN.differentiable_tls import solve_weighted_tls, solve_weighted_tls_tensorflow
from SPFN.geometry_utils import compute_consistent_plane_frame, compute_consistent_plane_frame_tensorflow, weighted_sphere_fitting, weighted_sphere_fitting_tensorflow

def compute_parameters(P, W, X):
    # First determine n as the solution to \min \sum W_i (X_i \cdot n)^2
    batch_size, n_points, _ = P.size()
    _, _, n_max_primitives = W.size()
    W_reshaped = W.transpose(1,2).contiguous().view(batch_size * n_max_primitives, n_points)  # BKxN
    X_reshaped = X.unsqueeze(1).expand(batch_size, n_max_primitives, n_points, 3).contiguous().view(batch_size * n_max_primitives, n_points, 3)
    n = solve_weighted_tls(X_reshaped, W_reshaped).view(batch_size, n_max_primitives, 3)  # BxKx3
    x_axes, y_axes = compute_consistent_plane_frame(n.view(batch_size * n_max_primitives, 3))
    x_axes = x_axes.view(batch_size, n_max_primitives, 3)  # BxKx3
    y_axes = y_axes.view(batch_size, n_max_primitives, 3)  # BxKx3
    x_coord = torch.sum(P.unsqueeze(1) * x_axes.unsqueeze(2), dim=3)  # BxKxN
    y_coord = torch.sum(P.unsqueeze(1) * y_axes.unsqueeze(2), dim=3)  # BxKxN
    P_proj = torch.stack([x_coord, y_coord], dim=3)  # BxKxNx2, 2D projection point
    P_proj_reshaped = P_proj.view(batch_size * n_max_primitives, n_points, 2)  # BKxNx2
    circle_center, circle_radius_squared = weighted_sphere_fitting(P_proj_reshaped, W_reshaped)
    circle_center = circle_center.view(batch_size, n_max_primitives, 2)  # BxKx2
    center = circle_center[:,:,0].unsqueeze(2) * x_axes + circle_center[:,:,1].unsqueeze(2) * y_axes  # BxKx3
    radius_square = circle_radius_squared.view(batch_size, n_max_primitives)  # BxK
    return n, center, radius_square

def compute_parameters_tensorflow(P, W, X):
    # First determine n as the solution to \min \sum W_i (X_i \cdot n)^2
    batch_size = tf.shape(P)[0]
    n_points = tf.shape(P)[1]
    n_max_primitives = tf.shape(W)[2]
    W_reshaped = tf.reshape(tf.transpose(W, [0, 2, 1]), [batch_size * n_max_primitives, n_points])  # BKxN
    X_reshaped = tf.reshape(tf.tile(tf.expand_dims(X, axis=1), [1, n_max_primitives, 1, 1]), [batch_size * n_max_primitives, n_points, 3])  # BKxNx3
    n = tf.reshape(solve_weighted_tls_tensorflow(X_reshaped, W_reshaped), [batch_size, n_max_primitives, 3])  # BxKx3
    x_axes, y_axes = compute_consistent_plane_frame_tensorflow(tf.reshape(n, [batch_size * n_max_primitives, 3]))
    x_axes = tf.reshape(x_axes, [batch_size, n_max_primitives, 3])  # BxKx3
    y_axes = tf.reshape(y_axes, [batch_size, n_max_primitives, 3])  # BxKx3
    x_coord = tf.reduce_sum(tf.expand_dims(P, axis=1) * tf.expand_dims(x_axes, axis=2), axis=3)  # BxKxN
    y_coord = tf.reduce_sum(tf.expand_dims(P, axis=1) * tf.expand_dims(y_axes, axis=2), axis=3)  # BxKxN
    P_proj = tf.stack([x_coord, y_coord], axis=3)  # BxKxNx2, 2D projection point
    P_proj_reshaped = tf.reshape(P_proj, [batch_size * n_max_primitives, n_points, 2])  # BKxNx2
    circle_center, circle_radius_squared = weighted_sphere_fitting_tensorflow(P_proj_reshaped, W_reshaped)
    circle_center = tf.reshape(circle_center, [batch_size, n_max_primitives, 2])  # BxKx2
    center = tf.expand_dims(circle_center[:, :, 0], axis=2) * x_axes + tf.expand_dims(circle_center[:, :, 1], axis=2) * y_axes  # BxKx3
    radius_square = tf.reshape(circle_radius_squared, [batch_size, n_max_primitives])  # BxK
    return n, center, radius_square

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
    n_torch, center_torch, radius_square_torch = compute_parameters(P_torch, W_torch, X_torch)
    n_torch = n_torch.detach().cpu().numpy()
    center_torch = center_torch.detach().cpu().numpy()
    radius_square_torch = radius_square_torch.detach().cpu().numpy()
    print('n_torch', n_torch)
    print('center_torch', center_torch)
    print('radius_square_torch', radius_square_torch)
    # Debugging with Tensorflow
    P_tensorflow = tf.constant(P, dtype=tf.float32)
    W_tensorflow = tf.constant(W, dtype=tf.float32)
    X_tensorflow = tf.constant(X, dtype=tf.float32)
    n_tensorflow, center_tensorflow, radius_square_tensorflow = compute_parameters_tensorflow(P_tensorflow, W_tensorflow, X_tensorflow)
    sess = tf.Session()
    n_tensorflow, center_tensorflow, radius_square_tensorflow = sess.run([n_tensorflow, center_tensorflow, radius_square_tensorflow])
    print(np.minimum(np.abs(n_tensorflow - n_torch), np.abs(n_tensorflow + n_torch)).max())
    print(np.abs(center_tensorflow - center_torch).max())
    print(np.abs(radius_square_tensorflow - radius_square_torch).max())

def sqrt_safe(x):
    return torch.sqrt(torch.abs(x) + 1e-10)

def compute_residue_single(axis, center, radius_squared, p):
    p_minus_c = p - center
    p_minus_c_sqr = torch.sum(p_minus_c**2, dim=-1)
    p_minus_c_dot_n = torch.sum(p_minus_c * axis, dim=-1)
    return (sqrt_safe(p_minus_c_sqr - p_minus_c_dot_n**2) - sqrt_safe(radius_squared))**2

def sqrt_safe_tensorflow(x):
    return tf.sqrt(tf.abs(x) + 1e-10)

def compute_residue_single_tensorflow(axis, center, radius_squared, p):
    p_minus_c = p - center
    p_minus_c_sqr = tf.reduce_sum(tf.square(p_minus_c), axis=-1)
    p_minus_c_dot_n = tf.reduce_sum(p_minus_c * axis, axis=-1)
    return tf.square(sqrt_safe_tensorflow(p_minus_c_sqr - tf.square(p_minus_c_dot_n)) - sqrt_safe_tensorflow(radius_squared))

if __name__ == '__main__':
    batch_size = 100
    num_points = 1024
    device = torch.device('cuda:0')
    np.random.seed(0)
    axis = np.random.randn(batch_size, num_points, 3)
    center = np.random.randn(batch_size, num_points, 3)
    radius_squared = np.random.randn(batch_size, num_points)
    p = np.random.randn(batch_size, num_points, 3)
    axis_torch = torch.from_numpy(axis).float().to(device)
    center_torch = torch.from_numpy(center).float().to(device)
    radius_squared_torch = torch.from_numpy(radius_squared).float().to(device)
    p_torch = torch.from_numpy(p).float().to(device)
    loss_torch = compute_residue_single(axis_torch, center_torch, radius_squared_torch, p_torch)
    loss_torch = loss_torch.detach().cpu().numpy()
    print('loss_torch', loss_torch)
    # Debugging with Tensorflow
    axis_tensorflow = tf.constant(axis, dtype=tf.float32)
    center_tensorflow = tf.constant(center, dtype=tf.float32)
    radius_squared_tensorflow = tf.constant(radius_squared, dtype=tf.float32)
    p_tensorflow = tf.constant(p, dtype=tf.float32)
    loss_tensorflow = compute_residue_single_tensorflow(axis_tensorflow, center_tensorflow, radius_squared_tensorflow, p_tensorflow)
    sess = tf.Session()
    loss_tensorflow = sess.run(loss_tensorflow)
    print(np.abs(loss_torch - loss_tensorflow).max())

def acos_safe(x):
    return torch.acos(torch.clamp(x, min=-1.0+1e-6, max=1.0-1e-6))

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

def acos_safe_tensorflow(x):
    return tf.math.acos(tf.clip_by_value(x, -1.0+1e-6, 1.0-1e-6))

def compute_parameter_loss_tensorflow(predicted_axis, gt_axis, matching_indices, angle_diff):
    n = batched_gather(predicted_axis, matching_indices, axis=1)
    dot_abs = tf.abs(tf.reduce_sum(n * gt_axis, axis=2))
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
    assert d['type'] == 'cylinder'
    location = np.array([d['location_x'], d['location_y'], d['location_z']], dtype=float)
    axis = np.array([d['axis_x'], d['axis_y'], d['axis_z']], dtype=float)
    radius = float(d['radius'])
    return Cylinder(center=location, radius=radius, axis=axis)

def extract_parameter_data_as_dict(primitives, n_max_primitives):
    n = np.zeros(dtype=float, shape=[n_max_primitives, 3])
    for i, primitive in enumerate(primitives):
        if isinstance(primitive, Cylinder):
            n[i] = primitive.axis
    return {
        'cylinder_axis_gt': n
    }

def extract_predicted_parameters_as_json(cylinder_center, cylinder_radius_squared, cylinder_axis, k):
    cylinder = Cylinder(cylinder_center, np.sqrt(cylinder_radius_squared), cylinder_axis, height=5)
    return {
        'type': 'cylinder',
        'center_x': float(cylinder.center[0]),
        'center_y': float(cylinder.center[1]),
        'center_z': float(cylinder.center[2]),
        'radius': float(cylinder.radius),
        'axis_x': float(cylinder.axis[0]),
        'axis_y': float(cylinder.axis[1]),
        'axis_z': float(cylinder.axis[2]),
        'height': float(cylinder.height),
        'label': k,
    }