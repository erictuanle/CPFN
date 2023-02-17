# Importatiomn of packages
import torch
import numpy as np
if __name__ == '__main__':
    import tensorflow as tf
from SPFN.primitives import Sphere
from SPFN.geometry_utils import weighted_sphere_fitting, weighted_sphere_fitting_tensorflow

def compute_parameters(P, W):
    batch_size, n_points, _ = P.size()
    _, _, n_max_primitives = W.size()
    P = P.unsqueeze(1).expand(batch_size, n_max_primitives, n_points, 3).contiguous()
    W = W.transpose(1, 2).contiguous()
    P = P.view(batch_size * n_max_primitives, n_points, 3)
    W = W.view(batch_size * n_max_primitives, n_points)
    center, radius_squared = weighted_sphere_fitting(P, W)
    center = center.view(batch_size, n_max_primitives, 3)
    radius_squared = radius_squared.view(batch_size, n_max_primitives)
    return center, radius_squared

def compute_parameters_tensorflow(P, W):
    batch_size = tf.shape(P)[0]
    n_points = tf.shape(P)[1]
    n_max_primitives = tf.shape(W)[2]
    P = tf.tile(tf.expand_dims(P, axis=1), [1, n_max_primitives, 1, 1])  # BxKxNx3
    W = tf.transpose(W, perm=[0, 2, 1])  # BxKxN
    P = tf.reshape(P, [batch_size * n_max_primitives, n_points, 3])  # BKxNx3
    W = tf.reshape(W, [batch_size * n_max_primitives, n_points])  # BKxN
    center, radius_squared = weighted_sphere_fitting_tensorflow(P, W)
    center = tf.reshape(center, [batch_size, n_max_primitives, 3])
    radius_squared = tf.reshape(radius_squared, [batch_size, n_max_primitives])
    return center, radius_squared

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
    center_torch, radius_squared_torch = compute_parameters(P_torch, W_torch)
    center_torch = center_torch.detach().cpu().numpy()
    radius_squared_torch = radius_squared_torch.detach().cpu().numpy()
    print('center_torch', center_torch)
    print('radius_squared_torch', radius_squared_torch)
    # Debugging with Tensorflow
    P_tensorflow = tf.constant(P, dtype=tf.float32)
    W_tensorflow = tf.constant(W, dtype=tf.float32)
    center_tensorflow, radius_squared_tensorflow = compute_parameters_tensorflow(P_tensorflow, W_tensorflow)
    sess = tf.Session()
    center_tensorflow, radius_squared_tensorflow = sess.run([center_tensorflow, radius_squared_tensorflow])
    print(np.abs(center_tensorflow-center_torch).max())
    print(np.abs(radius_squared_tensorflow-radius_squared_torch).max())

def sqrt_safe(x):
    return torch.sqrt(torch.abs(x) + 1e-10)

def compute_residue_single(center, radius_squared, p):
    return (sqrt_safe(torch.sum((p - center)**2, dim=-1)) - sqrt_safe(radius_squared))**2

def sqrt_safe_tensorflow(x):
    return tf.sqrt(tf.abs(x) + 1e-10)

def compute_residue_single_tensorflow(center, radius_squared, p):
    return tf.square(sqrt_safe_tensorflow(tf.reduce_sum(tf.square(p - center), axis=-1)) - sqrt_safe_tensorflow(radius_squared))

if __name__ == '__main__':
    batch_size = 100
    num_points = 1024
    device = torch.device('cuda:0')
    np.random.seed(0)
    center = np.random.randn(batch_size, num_points, 3)
    radius_squared = np.random.rand(batch_size, num_points)
    p = np.random.rand(batch_size, num_points, 3)
    center_torch = torch.from_numpy(center).float().to(device)
    radius_squared_torch = torch.from_numpy(radius_squared).float().to(device)
    p_torch = torch.from_numpy(p).float().to(device)
    residue_loss_torch = compute_residue_single(center_torch, radius_squared_torch, p_torch)
    residue_loss_torch = residue_loss_torch.detach().cpu().numpy()
    print('residue_loss_torch', residue_loss_torch)
    # Debugging with Tensorflow
    center_tensorflow = tf.constant(center, dtype=tf.float32)
    radius_squared_tensorflow = tf.constant(radius_squared, dtype=tf.float32)
    p_tensorflow = tf.constant(p, dtype=tf.float32)
    residue_loss_torch_tensorflow = compute_residue_single_tensorflow(center_tensorflow, radius_squared_tensorflow, p_tensorflow)
    sess = tf.Session()
    residue_loss_torch_tensorflow = sess.run(residue_loss_torch_tensorflow)
    print(np.abs(residue_loss_torch_tensorflow - residue_loss_torch).max())

def create_primitive_from_dict(d):
    assert d['type'] == 'sphere'
    location = np.array([d['location_x'], d['location_y'], d['location_z']], dtype=float)
    radius = float(d['radius'])
    return Sphere(center=location, radius=radius)

def extract_parameter_data_as_dict(primitives, n_max_primitives):
    return {}

def extract_predicted_parameters_as_json(sphere_center, sphere_radius_squared, k):
    sphere = Sphere(sphere_center, np.sqrt(sphere_radius_squared))
    return {
        'type': 'sphere',
        'center_x': float(sphere.center[0]),
        'center_y': float(sphere.center[1]),
        'center_z': float(sphere.center[2]),
        'radius': float(sphere.radius),
        'label': k,
    }