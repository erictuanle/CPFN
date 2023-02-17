# Importing packages
import torch
import numpy as np
if __name__ == '__main__':
    import tensorflow as tf
from SPFN.differentiable_tls import solve_weighted_tls, solve_weighted_tls_tensorflow

def compute_consistent_plane_frame(normal):
    # Input:  normal is Bx3
    # Returns: x_axis, y_axis, both of dimension Bx3
    device = normal.get_device()
    batch_size, _ = normal.size()
    candidate_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # Actually, 2 should be enough. This may still cause singularity TODO!!!
    y_axes = []
    for tmp_axis in candidate_axes:
        torch_axis = torch.FloatTensor(tmp_axis).to(device).unsqueeze(0)
        y_axes.append(torch.cross(normal, torch_axis.expand(batch_size, 3)))
    y_axes = torch.stack(y_axes, dim=0) # QxBx3
    y_axes_norm = torch.norm(y_axes, dim=2) # QxB
    # choose the axis with largest norm
    y_axes_chosen_idx = torch.argmax(y_axes_norm, dim=0) # B
    y_axes_chosen_idx = y_axes_chosen_idx.view(1, batch_size, 1).expand(1, batch_size, 3)
    # y_axes_chosen[b, :] = y_axes[y_axes_chosen_idx[b], b, :]
    y_axes = torch.gather(y_axes, 0, y_axes_chosen_idx).squeeze(0)
    y_axes = torch.nn.functional.normalize(y_axes, p=2, dim=1, eps=1e-12)
    x_axes = torch.cross(y_axes, normal) # Bx3
    return x_axes, y_axes

def compute_consistent_plane_frame_tensorflow(normal):
    # Input:  normal is Bx3
    # Returns: x_axis, y_axis, both of dimension Bx3
    batch_size = tf.shape(normal)[0]
    candidate_axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] # Actually, 2 should be enough. This may still cause singularity TODO!!!
    y_axes = []
    for tmp_axis in candidate_axes:
        tf_axis = tf.tile(tf.expand_dims(tf.constant(dtype=tf.float32, value=tmp_axis), axis=0), [batch_size, 1]) # Bx3
        y_axes.append(tf.cross(normal, tf_axis))
    y_axes = tf.stack(y_axes, axis=0) # QxBx3
    y_axes_norm = tf.norm(y_axes, axis=2) # QxB
    # choose the axis with largest norm
    y_axes_chosen_idx = tf.argmax(y_axes_norm, axis=0) # B
    # y_axes_chosen[b, :] = y_axes[y_axes_chosen_idx[b], b, :]
    indices_0 = tf.tile(tf.expand_dims(y_axes_chosen_idx, axis=1), [1, 3]) # Bx3
    indices_1 = tf.tile(tf.expand_dims(tf.range(batch_size), axis=1), [1, 3]) # Bx3
    indices_2 = tf.tile(tf.expand_dims(tf.range(3), axis=0), [batch_size, 1]) # Bx3
    indices = tf.stack([tf.cast(indices_0, tf.int32), indices_1, indices_2], axis=2) # Bx3x3
    y_axes = tf.gather_nd(y_axes, indices=indices) # Bx3
    if tf.VERSION == '1.4.1':
        y_axes = tf.nn.l2_normalize(y_axes, dim=1)
    else:
        y_axes = tf.nn.l2_normalize(y_axes, axis=1)
    x_axes = tf.cross(y_axes, normal) # Bx3
    return x_axes, y_axes

if __name__ == '__main__':
    batch_size = 100
    device = torch.device('cuda:0')
    np.random.seed(0)
    normal = np.random.randn(batch_size, 3)
    normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)
    normal_torch = torch.from_numpy(normal).float().to(device)
    x_axes_torch, y_axes_torch = compute_consistent_plane_frame(normal_torch)
    x_axes_torch = x_axes_torch.detach().cpu().numpy()
    y_axes_torch = y_axes_torch.detach().cpu().numpy()
    print('x_axes_torch', x_axes_torch)
    print('y_axes_torch', y_axes_torch)
    # Debugging with Tensorflow
    normal_tensorflow = tf.constant(normal, dtype=tf.float32)
    x_axes_tensorflow, y_axes_tensorflow = compute_consistent_plane_frame_tensorflow(normal_tensorflow)
    sess = tf.Session()
    x_axes_tensorflow, y_axes_tensorflow = sess.run([x_axes_tensorflow, y_axes_tensorflow])
    print(np.max(np.abs(x_axes_tensorflow-x_axes_torch)))

def weighted_plane_fitting(P, W, division_eps=1e-10):
    # P - BxNx3
    # W - BxN
    # Returns n, c, with n - Bx3, c - B
    WP = P * W.unsqueeze(2) # BxNx3
    W_sum = torch.sum(W, dim=1, keepdim=True) # Bx1
    P_weighted_mean = torch.sum(WP, dim=1) / torch.clamp(W_sum, min=division_eps, max=None) # Bx3
    A = P - P_weighted_mean.unsqueeze(1) # BxNx3
    n = solve_weighted_tls(A, W) # Bx3
    c = torch.sum(n * P_weighted_mean, dim=1)
    return n, c

def weighted_plane_fitting_tensorflow(P, W, division_eps=1e-10):
    # P - BxNx3
    # W - BxN
    # Returns n, c, with n - Bx3, c - B
    WP = P * tf.expand_dims(W, axis=2) # BxNx3
    W_sum = tf.reduce_sum(W, axis=1) # B
    P_weighted_mean = tf.reduce_sum(WP, axis=1) / tf.maximum(tf.expand_dims(W_sum, 1), division_eps) # Bx3
    A = P - tf.expand_dims(P_weighted_mean, axis=1) # BxNx3
    n = solve_weighted_tls_tensorflow(A, W) # Bx3
    c = tf.reduce_sum(n * P_weighted_mean, axis=1)
    return n, c

if __name__ == '__main__':
    batch_size = 100
    num_points = 1024
    device = torch.device('cuda:0')
    np.random.seed(0)
    P = np.random.randn(batch_size, num_points, 3)
    W = np.random.rand(batch_size, num_points)
    P_torch = torch.from_numpy(P).float().to(device)
    W_torch = torch.from_numpy(W).float().to(device)
    n_torch, c_torch = weighted_plane_fitting(P_torch, W_torch)
    n_torch = n_torch.detach().cpu().numpy()
    c_torch = c_torch.detach().cpu().numpy()
    print('n_torch', n_torch)
    print('c_torch', c_torch)
    # Debugging with Tensorflow
    P_tensorflow = tf.constant(P, dtype=tf.float32)
    W_tensorflow = tf.constant(W, dtype=tf.float32)
    n_tensorflow, c_tensorflow = weighted_plane_fitting_tensorflow(P_tensorflow, W_tensorflow)
    sess = tf.Session()
    n_tensorflow, c_tensorflow = sess.run([n_tensorflow, c_tensorflow])
    print(np.minimum(np.abs(n_tensorflow - n_torch), np.abs(n_tensorflow + n_torch)).max())
    print(np.minimum(np.abs(c_tensorflow - c_torch), np.abs(c_tensorflow + c_torch)).max())

def guarded_matrix_solve_ls(A, b, W, condition_number_cap=1e5, sqrt_eps=1e-10, ls_l2_regularizer=1e-8):
    # Solve weighted least square ||\sqrt(W)(Ax-b)||^2
    # A - BxNxD
    # b - BxNx1
    # W - BxN
    batch_size, _, dim = A.size()
    sqrt_W = torch.sqrt(torch.clamp(W, min=sqrt_eps, max=None)).unsqueeze(2) # BxN
    A = A * sqrt_W # BxNxD
    b = b * sqrt_W # BxNx1
    # Compute singular value, trivializing the problem when condition number is too large
    AtA = torch.bmm(A.transpose(1,2), A)
    _, s, _ = torch.svd(AtA, compute_uv=False) # s will be BxD
    s = s.detach()
    mask = s[:,0] / s[:,-1] < condition_number_cap # B
    #import pdb; pdb.set_trace()
    #A = A * mask.float().view(batch_size, 1, 1)
    #x = torch.linalg.lstsq(A, b).solution
    AtA = AtA * mask.float().view(batch_size, 1, 1) + ls_l2_regularizer * torch.eye(dim).unsqueeze(0).to(A.device) # zero out badly conditioned data
    Atb = torch.bmm(A.transpose(1, 2) * mask.float().view(batch_size, 1, 1), b)
    x, _ = torch.solve(Atb, AtA)
    x = x.squeeze(2)
    return x # BxD

if __name__ == '__main__':
    sqrt_eps = 1e-10
    ls_l2_regularizer = 1e-8

    batch_size = 100
    num_points = 1024
    dimension = 3
    device = torch.device('cuda:0')
    np.random.seed(0)
    A = np.random.randn(batch_size, num_points, dimension)
    b = np.random.randn(batch_size, num_points, 1)
    W = np.random.rand(batch_size, num_points)
    A = torch.from_numpy(A).float().to(device)
    b = torch.from_numpy(b).float().to(device)
    W = torch.from_numpy(W).float().to(device)

    sqrt_W = torch.sqrt(torch.clamp(W, sqrt_eps)).unsqueeze(2)  # BxN
    A = A * sqrt_W  # BxNxD
    b = b * sqrt_W  # BxNx1
    AtA = torch.bmm(A.transpose(1, 2), A)
    mask = torch.zeros([batch_size]).float().to(A.device)  # B
    AtA = AtA * mask.view(batch_size, 1, 1) + ls_l2_regularizer * torch.eye(dimension).unsqueeze(0).to(device)  # zero out badly conditioned data
    Atb = torch.bmm(A.transpose(1, 2) * mask.view(batch_size, 1, 1), b)
    x = torch.solve(Atb, AtA)

def guarded_matrix_solve_ls_tensorflow(A, b, W, condition_number_cap=1e5, sqrt_eps=1e-10, ls_l2_regularizer=1e-8):
    # Solve weighted least square ||\sqrt(W)(Ax-b)||^2
    # A - BxNxD
    # b - BxNx1
    # W - BxN
    sqrt_W = tf.sqrt(tf.maximum(W, sqrt_eps)) # BxN
    A *= tf.expand_dims(sqrt_W, axis=2) # BxNxD
    b *= tf.expand_dims(sqrt_W, axis=2) # BxNx1
    # Compute singular value, trivializing the problem when condition number is too large
    AtA = tf.matmul(a=A, b=A, transpose_a=True)
    s, _, _ = [tf.stop_gradient(u) for u in tf.svd(AtA)] # s will be BxD
    mask = tf.less(s[:, 0] / s[:, -1], condition_number_cap) # B
    A *= tf.to_float(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=2)) # zero out badly conditioned data
    x = tf.matrix_solve_ls(A, b, l2_regularizer=ls_l2_regularizer, fast=True) # BxDx1
    return tf.squeeze(x, axis=2) # BxD

if __name__ == '__main__':
    batch_size = 100
    num_points = 1024
    dimension = 3
    device = torch.device('cuda:0')
    np.random.seed(0)
    A = np.random.randn(batch_size, num_points, dimension)
    b = np.random.randn(batch_size, num_points, 1)
    W = np.random.rand(batch_size, num_points)
    A_torch = torch.from_numpy(A).float().to(device)
    b_torch = torch.from_numpy(b).float().to(device)
    W_torch = torch.from_numpy(W).float().to(device)
    x_torch = guarded_matrix_solve_ls(A_torch, b_torch, W_torch)
    x_torch = x_torch.detach().cpu().numpy()
    print('x_torch', x_torch)
    # Debugging with Tensorflow
    A_tensorflow = tf.constant(A, dtype=tf.float32)
    b_tensorflow = tf.constant(b, dtype=tf.float32)
    W_tensorflow = tf.constant(W, dtype=tf.float32)
    x_tensorflow = guarded_matrix_solve_ls_tensorflow(A_tensorflow, b_tensorflow, W_tensorflow)
    sess = tf.Session()
    x_tensorflow = sess.run(x_tensorflow)
    print(np.max(np.abs(x_tensorflow - x_torch)))

def weighted_sphere_fitting(P, W, division_eps=1e-10):
    # P - BxNxD
    # W - BxN
    W_sum = torch.sum(W, axis=1) # B
    WP_sqr_sum = torch.sum(W * torch.sum(P**2, axis=2), axis=1) # B
    P_sqr = torch.sum(P**2, axis=2) # BxN
    b = ((WP_sqr_sum / torch.clamp(W_sum, min=division_eps, max=None)).unsqueeze(1) - P_sqr).unsqueeze(2) # BxNx1
    WP_sum = torch.sum(W.unsqueeze(2) * P, dim=1) # BxD
    A = 2 * ((WP_sum / torch.clamp(W_sum, min=division_eps, max=None).unsqueeze(1)).unsqueeze(1) - P) # BxNxD
    # Seek least norm solution to the least square
    center = guarded_matrix_solve_ls(A, b, W) # BxD
    W_P_minus_C_sqr_sum = P - center.unsqueeze(1) # BxNxD
    W_P_minus_C_sqr_sum = W * torch.sum(W_P_minus_C_sqr_sum**2, dim=2) # BxN
    r_sqr = torch.sum(W_P_minus_C_sqr_sum, dim=1) / torch.clamp(W_sum, min=division_eps, max=None) # B
    return center, r_sqr

def weighted_sphere_fitting_tensorflow(P, W, division_eps=1e-10):
    # P - BxNxD
    # W - BxN
    W_sum = tf.reduce_sum(W, axis=1) # B
    WP_sqr_sum = tf.reduce_sum(W * tf.reduce_sum(tf.square(P), axis=2), axis=1) # B
    P_sqr = tf.reduce_sum(tf.square(P), axis=2) # BxN
    b = tf.expand_dims(tf.expand_dims(WP_sqr_sum / tf.maximum(W_sum, division_eps), axis=1) - P_sqr, axis=2) # BxNx1
    WP_sum = tf.reduce_sum(tf.expand_dims(W, axis=2) * P, axis=1) # BxD
    A = 2 * (tf.expand_dims(WP_sum / tf.expand_dims(tf.maximum(W_sum, division_eps), axis=1), axis=1) - P) # BxNxD
    # Seek least norm solution to the least square
    center = guarded_matrix_solve_ls_tensorflow(A, b, W) # BxD
    W_P_minus_C_sqr_sum = P - tf.expand_dims(center, axis=1) # BxNxD
    W_P_minus_C_sqr_sum = W * tf.reduce_sum(tf.square(W_P_minus_C_sqr_sum), axis=2) # BxN
    r_sqr = tf.reduce_sum(W_P_minus_C_sqr_sum, axis=1) / tf.maximum(W_sum, division_eps) # B
    return center, r_sqr

if __name__ == '__main__':
    batch_size = 100
    num_points = 1024
    dimension = 3
    device = torch.device('cuda:0')
    np.random.seed(0)
    P = np.random.randn(batch_size, num_points, dimension)
    W = np.random.rand(batch_size, num_points)
    P_torch = torch.from_numpy(P).float().to(device)
    W_torch = torch.from_numpy(W).float().to(device)
    center_torch, r_sqr_torch = weighted_sphere_fitting(P_torch, W_torch)
    center_torch = center_torch.detach().cpu().numpy()
    r_sqr_torch = r_sqr_torch.detach().cpu().numpy()
    print('center_torch', center_torch)
    print('r_sqr_torch', r_sqr_torch)
    # Debugging with Tensorflow
    P_tensorflow = tf.constant(P, dtype=tf.float32)
    W_tensorflow = tf.constant(W, dtype=tf.float32)
    center_tensorflow, r_sqr_tensorflow = weighted_sphere_fitting_tensorflow(P_tensorflow, W_tensorflow)
    sess = tf.Session()
    center_tensorflow, r_sqr_tensorflow = sess.run([center_tensorflow, r_sqr_tensorflow])
    print(np.max(np.abs(center_tensorflow - center_torch)))
    print(np.max(np.abs(r_sqr_tensorflow - r_sqr_torch)))