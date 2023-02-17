# Importation of packages
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from SPFN import plane_fitter, sphere_fitter, cylinder_fitter, cone_fitter
from SPFN import losses_implementation

def hungarian_matching(W_pred, I_gt):
    # This non-tf function does not backprob gradient, only output matching indices
    # W_pred - BxNxK
    # I_gt - BxN, may contain -1's
    # Output: matching_indices - BxK, where (b,k)th ground truth primitive is matched with (b, matching_indices[b, k])
    #   where only n_gt_labels entries on each row have meaning. The matching does not include gt background instance
    batch_size, n_points, n_max_labels = W_pred.size()
    matching_indices = torch.zeros([batch_size, n_max_labels], dtype=torch.long).to(W_pred.device)
    mask = torch.zeros([batch_size, n_max_labels], dtype=torch.bool).to(W_pred.device)
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
        mask[b, :n_gt_labels] = True
    return matching_indices, mask

# Converting W to hard encoding
def hard_W_encoding(W):
    # W - BxNxK
    _, _, num_labels = W.size()
    hardW = torch.eye(num_labels).to(W.device)[torch.argmax(W, dim=2)]
    return hardW

if __name__ == '__main__' and 1:
    batch_size = 8
    num_points = 1024
    n_max_instances = 12
    device = torch.device('cuda:0')
    np.random.seed(0)
    W = np.random.rand(batch_size, num_points, n_max_instances)
    W = W / np.linalg.norm(W, axis=2, keepdims=True)
    W_torch = torch.from_numpy(W).float().to(device)
    hardW = hard_W_encoding(W_torch)
    print('hardW', hardW.size())

# Getting the per instance type
def get_instance_type(T, W):
    instance_type = torch.bmm(W.transpose(1,2), T)
    instance_type = torch.argmax(instance_type, dim=2)
    return instance_type
if __name__ == '__main__' and 1:
    batch_size = 8
    num_points = 1024
    n_max_instances = 12
    n_type = 4
    device = torch.device('cuda:0')
    np.random.seed(0)
    W = np.random.rand(batch_size, num_points, n_max_instances)
    W = W / np.linalg.norm(W, axis=2, keepdims=True)
    T = np.random.rand(batch_size, num_points, n_type)
    T = T / np.linalg.norm(T, axis=2, keepdims=True)
    W_torch = torch.from_numpy(W).float().to(device)
    T_torch = torch.from_numpy(T).float().to(device)
    instance_type = get_instance_type(T_torch, W_torch)
    print('instance_type', instance_type.size())

def sqrt_safe(x):
    return torch.sqrt(torch.abs(x) + 1e-10)

# Getting the residual loss
def get_residual_loss(parameters, matching_indices, points_per_instance, T, classes=['plane','sphere','cylinder','cone']):
    batch_size, num_primitives, num_primitive_points, _ = points_per_instance.shape
    _, residue_per_point_array = losses_implementation.compute_residue_loss(parameters, matching_indices, points_per_instance, torch.gather(T, 1, matching_indices), classes=classes)
    residue_per_point_array = torch.gather(residue_per_point_array, 3, T.view(batch_size, num_primitives, 1, 1).expand(batch_size, num_primitives, num_primitive_points, 1)).squeeze(3)
    residual_loss = sqrt_safe(residue_per_point_array)
    return residual_loss
if __name__ == '__main__' and 1:
    batch_size = 8
    num_points = 1024
    num_points_instance = 512
    n_max_instances = 12
    device = torch.device('cuda:0')
    np.random.seed(0)
    P = np.random.randn(batch_size, num_points, 3)
    W = np.random.rand(batch_size, num_points, n_max_instances)
    W = W / np.linalg.norm(W, axis=2, keepdims=True)
    T = np.random.rand(batch_size, num_points, 4)
    T = T / np.linalg.norm(T, axis=2, keepdims=True)
    X = np.random.randn(batch_size, num_points, 3)
    X = X / np.linalg.norm(X, axis=2, keepdims=True)
    points_per_instance = np.random.randn(batch_size, n_max_instances, num_points_instance, 3)
    T_gt = np.random.randint(0, 4, (batch_size, n_max_instances))
    I_gt = np.random.randint(0, n_max_instances, (batch_size, num_points))
    P_torch = torch.from_numpy(P).float().to(device)
    W_torch = torch.from_numpy(W).float().to(device)
    X_torch = torch.from_numpy(X).float().to(device)
    points_per_instance_torch = torch.from_numpy(points_per_instance).float().to(device)
    T_torch = torch.from_numpy(T).float().to(device)
    T_gt_torch = torch.from_numpy(T_gt).long().to(device)
    I_gt_torch = torch.from_numpy(I_gt).long().to(device)

    W_torch = hard_W_encoding(W_torch)
    parameters_torch = losses_implementation.compute_parameters(P_torch, W_torch, X_torch)
    matching_indices_torch, _ = hungarian_matching(W_torch, I_gt_torch)
    T_torch = get_instance_type(T_torch, W_torch)
    residual_loss = get_residual_loss(parameters_torch, matching_indices_torch, points_per_instance_torch, T_torch, classes=['plane','sphere','cylinder','cone'])
    print('residual_loss', residual_loss.size())

# Arccos safe
def acos_safe(x):
    return torch.acos(torch.clamp(x, min=-1.0+1e-6, max=1.0-1e-6))

# Segmentation mIoU
def compute_segmentation_iou(W, I_gt, matching_indices, mask):# W - BxNxK
    mIoU = 1 - losses_implementation.compute_miou_loss(W, I_gt, matching_indices)[0]
    mIoU = torch.sum(mask * mIoU, dim=1) / torch.sum(mask, dim=1)
    return mIoU

if __name__ == '__main__' and 1:
    batch_size = 8
    num_points = 1024
    num_points_instance = 512
    n_max_instances = 12
    device = torch.device('cuda:0')
    np.random.seed(0)
    W = np.random.rand(batch_size, num_points, n_max_instances)
    W = W / np.linalg.norm(W, axis=2, keepdims=True)
    I_gt = np.random.randint(0, n_max_instances, (batch_size, num_points))
    W_torch = torch.from_numpy(W).float().to(device)
    I_gt_torch = torch.from_numpy(I_gt).long().to(device)
    W_torch = hard_W_encoding(W_torch)
    matching_indices_torch, mask_torch = hungarian_matching(W_torch, I_gt_torch)
    mIou = compute_segmentation_iou(W_torch, I_gt_torch, matching_indices_torch, mask_torch)
    print('mIou', mIou.size())

# Mean primitive type accuracy
def compute_type_accuracy(T, T_gt, matching_indices, mask):
    T_reordered = torch.gather(T, 1, matching_indices)  # BxNxK
    type_accuracy = torch.sum(mask*(T_reordered == T_gt), dim=1) / torch.sum(mask, dim=1)
    return type_accuracy

if __name__ == '__main__' and 1:
    batch_size = 8
    num_points = 1024
    n_max_instances = 12
    device = torch.device('cuda:0')
    np.random.seed(0)
    W = np.random.rand(batch_size, num_points, n_max_instances)
    W = W / np.linalg.norm(W, axis=2, keepdims=True)
    T = np.random.rand(batch_size, num_points, 4)
    T = T / np.linalg.norm(T, axis=2, keepdims=True)
    T_gt = np.random.randint(0, 4, (batch_size, n_max_instances))
    mask = np.random.randint(0, 2, (batch_size, n_max_instances))
    W_torch = torch.from_numpy(W).float().to(device)
    T_torch = torch.from_numpy(T).float().to(device)
    T_gt_torch = torch.from_numpy(T_gt).long().to(device)
    mask_torch = torch.from_numpy(mask).float().to(device)

    W_torch = hard_W_encoding(W_torch)
    T_torch = get_instance_type(T_torch, W_torch)
    type_accuracy = compute_type_accuracy(T_torch, T_gt_torch, mask_torch)
    print('type_accuracy', type_accuracy.size())

# Mean point normal difference
def compute_normal_difference(X, X_gt):
    normal_difference = torch.mean(acos_safe(torch.abs(torch.sum(X*X_gt, dim=2))), dim=1)
    return normal_difference

if __name__ == '__main__' and 1:
    batch_size = 8
    num_points = 1024
    device = torch.device('cuda:0')
    np.random.seed(0)
    X = np.random.randn(batch_size, num_points, 3)
    X = X / np.linalg.norm(X, axis=2, keepdims=True)
    X_gt = np.random.randn(batch_size, num_points, 3)
    X_gt = X_gt / np.linalg.norm(X_gt, axis=2, keepdims=True)
    X_torch = torch.from_numpy(X).float().to(device)
    X_gt_torch = torch.from_numpy(X_gt).float().to(device)
    normal_difference = compute_normal_difference(X_gt_torch, X_gt_torch)
    print('normal_difference', normal_difference.size())

# Mean primitive axis difference
def compute_axis_difference(predicted_parameters, gt_parameters, matching_indices, T, T_gt, mask, classes=['plane','sphere','cylinder','cone'], div_eps=1e-10):
    mask = mask * (T == T_gt).float()
    parameter_loss = losses_implementation.compute_parameter_loss(predicted_parameters, gt_parameters, matching_indices, T_gt, is_eval=True, classes=classes)
    axis_difference = torch.sum(mask * parameter_loss, dim=1) / torch.clamp(torch.sum(parameter_loss, dim=1), min=div_eps, max=None)
    return axis_difference

if __name__ == '__main__' and 1:
    batch_size = 8
    num_points = 1024
    n_max_instances = 12
    device = torch.device('cuda:0')
    np.random.seed(0)
    P = np.random.randn(batch_size, num_points, 3)
    W = np.random.rand(batch_size, num_points, n_max_instances)
    W = W / np.linalg.norm(W, axis=2, keepdims=True)
    T = np.random.rand(batch_size, num_points, 4)
    T = T / np.linalg.norm(T, axis=2, keepdims=True)
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
    T_torch = torch.from_numpy(T).float().to(device)
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

    W_torch = hard_W_encoding(W_torch)
    predicted_parameters_torch = losses_implementation.compute_parameters(P_torch, W_torch, X_torch)
    matching_indices_torch, mask_torch = hungarian_matching(W_torch, I_gt_torch)
    T_torch = get_instance_type(T_torch, W_torch)
    axis_difference = compute_axis_difference(predicted_parameters_torch, gt_parameters_torch, matching_indices_torch, T_torch, T_gt_torch, mask_torch, classes=['plane', 'sphere', 'cylinder', 'cone'])
    print('axis_difference', axis_difference.size())

# Mean/Std Sk residual
def compute_meanstd_Sk_residual(residue_loss, mask):
    mean_residual = torch.sum(mask * torch.mean(residue_loss, dim=2), dim=1) / torch.sum(mask, dim=1)
    std_residual = torch.sum(mask * torch.std(residue_loss, dim=2), dim=1) / torch.sum(mask, dim=1)
    return mean_residual, std_residual

if __name__ == '__main__' and 1:
    batch_size = 8
    num_points = 1024
    num_points_instance = 512
    n_max_instances = 12
    device = torch.device('cuda:0')
    np.random.seed(0)
    P = np.random.randn(batch_size, num_points, 3)
    W = np.random.rand(batch_size, num_points, n_max_instances)
    W = W / np.linalg.norm(W, axis=2, keepdims=True)
    T = np.random.rand(batch_size, num_points, 4)
    T = T / np.linalg.norm(T, axis=2, keepdims=True)
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
    points_per_instance = np.random.randn(batch_size, n_max_instances, num_points_instance, 3)
    P_torch = torch.from_numpy(P).float().to(device)
    W_torch = torch.from_numpy(W).float().to(device)
    T_torch = torch.from_numpy(T).float().to(device)
    X_torch = torch.from_numpy(X).float().to(device)
    T_gt_torch = torch.from_numpy(T_gt).long().to(device)
    I_gt_torch = torch.from_numpy(I_gt).long().to(device)
    gt_parameters_torch = {'plane_normal': torch.from_numpy(gt_parameters['plane_normal']).float().to(device),
                           'plane_center': torch.from_numpy(gt_parameters['plane_center']).float().to(device),
                           'sphere_center': torch.from_numpy(gt_parameters['sphere_center']).float().to(device),
                           'sphere_radius_squared': torch.from_numpy(gt_parameters['sphere_radius_squared']).float().to(
                               device),
                           'cylinder_axis': torch.from_numpy(gt_parameters['cylinder_axis']).float().to(device),
                           'cylinder_center': torch.from_numpy(gt_parameters['cylinder_center']).float().to(device),
                           'cylinder_radius_square': torch.from_numpy(
                               gt_parameters['cylinder_radius_square']).float().to(device),
                           'cone_apex': torch.from_numpy(gt_parameters['cone_apex']).float().to(device),
                           'cone_axis': torch.from_numpy(gt_parameters['cone_axis']).float().to(device),
                           'cone_half_angle': torch.from_numpy(gt_parameters['cone_half_angle']).float().to(device)}
    points_per_instance_torch = torch.from_numpy(points_per_instance).float().to(device)

    W_torch = hard_W_encoding(W_torch)
    predicted_parameters_torch = losses_implementation.compute_parameters(P_torch, W_torch, X_torch)
    matching_indices_torch, mask_torch = hungarian_matching(W_torch, I_gt_torch)
    T_torch = get_instance_type(T_torch, W_torch)

    residue_loss_torch = get_residual_loss(predicted_parameters_torch, matching_indices_torch, points_per_instance_torch, T_torch, classes=['plane', 'sphere', 'cylinder', 'cone'])
    mean_residual, std_residual = compute_meanstd_Sk_residual(residue_loss_torch, mask_torch)
    print('Mean Sk Residual Loss: ', mean_residual)
    print('Std Sk Residual Loss: ', std_residual)

# Sk coverage
def compute_Sk_coverage(residue_loss, epsilon, mask):
    residue_loss = torch.mean((residue_loss < epsilon).float(), dim=2)
    Sk_coverage = torch.sum(mask * residue_loss, dim=1) / torch.sum(mask, dim=1)
    return Sk_coverage

if __name__ == '__main__' and 1:
    batch_size = 8
    num_points = 1024
    num_points_instance = 512
    n_max_instances = 12
    device = torch.device('cuda:0')
    np.random.seed(0)
    P = np.random.randn(batch_size, num_points, 3)
    W = np.random.rand(batch_size, num_points, n_max_instances)
    W = W / np.linalg.norm(W, axis=2, keepdims=True)
    T = np.random.rand(batch_size, num_points, 4)
    T = T / np.linalg.norm(T, axis=2, keepdims=True)
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
    points_per_instance = np.random.randn(batch_size, n_max_instances, num_points_instance, 3)
    epsilon = 0.01
    P_torch = torch.from_numpy(P).float().to(device)
    W_torch = torch.from_numpy(W).float().to(device)
    T_torch = torch.from_numpy(T).float().to(device)
    X_torch = torch.from_numpy(X).float().to(device)
    T_gt_torch = torch.from_numpy(T_gt).long().to(device)
    I_gt_torch = torch.from_numpy(I_gt).long().to(device)
    gt_parameters_torch = {'plane_normal': torch.from_numpy(gt_parameters['plane_normal']).float().to(device),
                           'plane_center': torch.from_numpy(gt_parameters['plane_center']).float().to(device),
                           'sphere_center': torch.from_numpy(gt_parameters['sphere_center']).float().to(device),
                           'sphere_radius_squared': torch.from_numpy(gt_parameters['sphere_radius_squared']).float().to(
                               device),
                           'cylinder_axis': torch.from_numpy(gt_parameters['cylinder_axis']).float().to(device),
                           'cylinder_center': torch.from_numpy(gt_parameters['cylinder_center']).float().to(device),
                           'cylinder_radius_square': torch.from_numpy(
                               gt_parameters['cylinder_radius_square']).float().to(device),
                           'cone_apex': torch.from_numpy(gt_parameters['cone_apex']).float().to(device),
                           'cone_axis': torch.from_numpy(gt_parameters['cone_axis']).float().to(device),
                           'cone_half_angle': torch.from_numpy(gt_parameters['cone_half_angle']).float().to(device)}
    points_per_instance_torch = torch.from_numpy(points_per_instance).float().to(device)

    W_torch = hard_W_encoding(W_torch)
    predicted_parameters_torch = losses_implementation.compute_parameters(P_torch, W_torch, X_torch)
    matching_indices_torch, mask_torch = hungarian_matching(W_torch, I_gt_torch)
    T_torch = get_instance_type(T_torch, W_torch)

    residue_loss_torch = get_residual_loss(predicted_parameters_torch, matching_indices_torch,
                                           points_per_instance_torch, T_torch,
                                           classes=['plane', 'sphere', 'cylinder', 'cone'])
    Sk_coverage = compute_Sk_coverage(residue_loss_torch, epsilon, mask_torch)
    print('Sk Coverage : ', Sk_coverage)

# P coverage
def compute_P_coverage(P, T, matching_indices, predicted_parameters, epsilon, classes=['plane', 'sphere', 'cylinder', 'cone']):
    batch_size, num_points, _ = P.size()
    _, num_primitives = T.size()
    residue_loss = get_residual_loss(predicted_parameters, matching_indices, P.unsqueeze(1).expand(batch_size, num_primitives, num_points, 3), torch.gather(T, 1, matching_indices), classes=classes)
    residue_loss, _ = torch.min(residue_loss, dim=1)
    P_coverage = torch.mean((residue_loss < epsilon).float(), dim=1)
    return P_coverage

if __name__ == '__main__' and 1:
    batch_size = 8
    num_points = 1024
    num_points_instance = 512
    n_max_instances = 12
    device = torch.device('cuda:0')
    np.random.seed(0)
    P = np.random.randn(batch_size, num_points, 3)
    W = np.random.rand(batch_size, num_points, n_max_instances)
    W = W / np.linalg.norm(W, axis=2, keepdims=True)
    T = np.random.rand(batch_size, num_points, 4)
    T = T / np.linalg.norm(T, axis=2, keepdims=True)
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
    points_per_instance = np.random.randn(batch_size, n_max_instances, num_points_instance, 3)
    epsilon = 0.01
    P_torch = torch.from_numpy(P).float().to(device)
    W_torch = torch.from_numpy(W).float().to(device)
    T_torch = torch.from_numpy(T).float().to(device)
    X_torch = torch.from_numpy(X).float().to(device)
    T_gt_torch = torch.from_numpy(T_gt).long().to(device)
    I_gt_torch = torch.from_numpy(I_gt).long().to(device)
    gt_parameters_torch = {'plane_normal': torch.from_numpy(gt_parameters['plane_normal']).float().to(device),
                           'plane_center': torch.from_numpy(gt_parameters['plane_center']).float().to(device),
                           'sphere_center': torch.from_numpy(gt_parameters['sphere_center']).float().to(device),
                           'sphere_radius_squared': torch.from_numpy(gt_parameters['sphere_radius_squared']).float().to(
                               device),
                           'cylinder_axis': torch.from_numpy(gt_parameters['cylinder_axis']).float().to(device),
                           'cylinder_center': torch.from_numpy(gt_parameters['cylinder_center']).float().to(device),
                           'cylinder_radius_square': torch.from_numpy(
                               gt_parameters['cylinder_radius_square']).float().to(device),
                           'cone_apex': torch.from_numpy(gt_parameters['cone_apex']).float().to(device),
                           'cone_axis': torch.from_numpy(gt_parameters['cone_axis']).float().to(device),
                           'cone_half_angle': torch.from_numpy(gt_parameters['cone_half_angle']).float().to(device)}
    points_per_instance_torch = torch.from_numpy(points_per_instance).float().to(device)

    W_torch = hard_W_encoding(W_torch)
    predicted_parameters_torch = losses_implementation.compute_parameters(P_torch, W_torch, X_torch)
    matching_indices_torch, mask_torch = hungarian_matching(W_torch, I_gt_torch)
    T_torch = get_instance_type(T_torch, W_torch)

    P_coverage = compute_P_coverage(P_torch, T_torch, matching_indices_torch, predicted_parameters_torch, classes=['plane', 'sphere', 'cylinder', 'cone'])
    print('P Coverage : ', P_coverage)

def compute_all_metrics(P, X, X_gt, W, I_gt, T, T_gt, points_per_instance, gt_parameters, list_epsilon=[0.01, 0.02],  classes=['plane', 'sphere', 'cylinder', 'cone']):
    W = hard_W_encoding(W)
    T = get_instance_type(T, W)
    diff = T.size(1) - T_gt.size(1)
    if diff>0:
        T_gt = torch.cat((T_gt, torch.zeros_like(T_gt[:, 0:1]).expand(-1, diff)), dim=1)
    elif diff < 0:
        W = torch.cat((W, torch.zeros_like(W[:,:,0:1]).expand(-1, -1, -diff)), dim=2)
        T = torch.cat((T, torch.zeros_like(T[:, 0:1]).expand(-1, -diff)), dim=1)
    matching_indices, mask = hungarian_matching(W, I_gt)
    mask = mask.float()
    mIoU = compute_segmentation_iou(W, I_gt, matching_indices, mask)
    type_accuracy = compute_type_accuracy(T, T_gt, matching_indices, mask)
    normal_difference = compute_normal_difference(X, X_gt)
    predicted_parameters = losses_implementation.compute_parameters(P, W, X)
    if diff > 0:
        gt_parameters['plane_normal'] = torch.cat((gt_parameters['plane_normal'], torch.zeros_like(gt_parameters['plane_normal'][:, 0:1]).expand(-1, diff, 3)), dim=1)
        gt_parameters['cylinder_axis'] = torch.cat((gt_parameters['cylinder_axis'], torch.zeros_like(gt_parameters['cylinder_axis'][:, 0:1]).expand(-1, diff, 3)), dim=1)
        gt_parameters['cone_axis'] = torch.cat((gt_parameters['cone_axis'], torch.zeros_like(gt_parameters['cone_axis'][:, 0:1]).expand(-1, diff, 3)), dim=1)
        points_per_instance = torch.cat((points_per_instance, torch.zeros_like(points_per_instance[:,0:1]).expand(-1, diff, 512, 3)), dim=1)
    axis_difference = compute_axis_difference(predicted_parameters, gt_parameters, matching_indices, T, T_gt, mask, classes=classes)
    residue_loss = get_residual_loss(predicted_parameters, matching_indices, points_per_instance, T_gt, classes=classes)
    mean_residual, std_residual = compute_meanstd_Sk_residual(residue_loss, mask)
    Sk_coverage = []
    for epsilon in list_epsilon:
        Sk_coverage.append(compute_Sk_coverage(residue_loss, epsilon, mask))
    P_coverage = []
    for epsilon in list_epsilon:
        P_coverage.append(compute_P_coverage(P, T, matching_indices, predicted_parameters, epsilon, classes=classes))
    return mIoU, type_accuracy, normal_difference, axis_difference, mean_residual, std_residual, Sk_coverage, P_coverage, W, predicted_parameters, T

if __name__ == '__main__' and 1:
    batch_size = 8
    num_points = 1024
    num_points_instance = 512
    n_max_instances = 12
    device = torch.device('cuda:0')
    np.random.seed(0)
    P = np.random.randn(batch_size, num_points, 3)
    W = np.random.rand(batch_size, num_points, n_max_instances)
    W = W / np.linalg.norm(W, axis=2, keepdims=True)
    T = np.random.rand(batch_size, num_points, 4)
    T = T / np.linalg.norm(T, axis=2, keepdims=True)
    X = np.random.randn(batch_size, num_points, 3)
    X = X / np.linalg.norm(X, axis=2, keepdims=True)
    X_gt = np.random.randn(batch_size, num_points, 3)
    X_gt = X_gt / np.linalg.norm(X_gt, axis=2, keepdims=True)
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
    points_per_instance = np.random.randn(batch_size, n_max_instances, num_points_instance, 3)
    P_torch = torch.from_numpy(P).float().to(device)
    W_torch = torch.from_numpy(W).float().to(device)
    T_torch = torch.from_numpy(T).float().to(device)
    X_torch = torch.from_numpy(X).float().to(device)
    X_gt_torch = torch.from_numpy(X_gt).float().to(device)
    T_gt_torch = torch.from_numpy(T_gt).long().to(device)
    I_gt_torch = torch.from_numpy(I_gt).long().to(device)
    gt_parameters_torch = {'plane_normal': torch.from_numpy(gt_parameters['plane_normal']).float().to(device),
                           'plane_center': torch.from_numpy(gt_parameters['plane_center']).float().to(device),
                           'sphere_center': torch.from_numpy(gt_parameters['sphere_center']).float().to(device),
                           'sphere_radius_squared': torch.from_numpy(gt_parameters['sphere_radius_squared']).float().to(
                               device),
                           'cylinder_axis': torch.from_numpy(gt_parameters['cylinder_axis']).float().to(device),
                           'cylinder_center': torch.from_numpy(gt_parameters['cylinder_center']).float().to(device),
                           'cylinder_radius_square': torch.from_numpy(
                               gt_parameters['cylinder_radius_square']).float().to(device),
                           'cone_apex': torch.from_numpy(gt_parameters['cone_apex']).float().to(device),
                           'cone_axis': torch.from_numpy(gt_parameters['cone_axis']).float().to(device),
                           'cone_half_angle': torch.from_numpy(gt_parameters['cone_half_angle']).float().to(device)}
    points_per_instance_torch = torch.from_numpy(points_per_instance).float().to(device)
    mIoU, type_accuracy, normal_difference, axis_difference, mean_residual, std_residual, Sk_coverage, P_coverage = compute_all_metrics(P_torch, X_torch, X_gt_torch, W_torch, I_gt_torch, T_torch, T_gt_torch, points_per_instance_torch, gt_parameters_torch, classes=['plane', 'sphere', 'cylinder', 'cone'])
    print('mIoU', mIoU.size())
    print('type_accuracy', type_accuracy.size())
    print('normal_difference', normal_difference.size())
    print('axis_difference', axis_difference.size())
    print('mean_residual', mean_residual.size())
    print('std_residual', std_residual.size())
    for i in range(len(Sk_coverage)):
        print('Sk_coverage_%d'%i, Sk_coverage[i].size())
    for i in range(len(P_coverage)):
        print('P_coverage_%d'%i, P_coverage[i].size())

def creates_json(T, predicted_parameters):
    list_json = []
    for i, type_id in enumerate(T):
        if type_id == 0:
            json = plane_fitter.extract_predicted_parameters_as_json(predicted_parameters['plane_normal'][0,i].cpu().numpy(), predicted_parameters['plane_center'][0,i].cpu().numpy(), i)
        elif type_id == 1:
            json = sphere_fitter.extract_predicted_parameters_as_json(predicted_parameters['sphere_center'][0,i].cpu().numpy(), predicted_parameters['sphere_radius_squared'][0,i].cpu().numpy(), i)
        elif type_id == 2:
            json = cylinder_fitter.extract_predicted_parameters_as_json(predicted_parameters['cylinder_center'][0,i].cpu().numpy(), predicted_parameters['cylinder_radius_squared'][0,i].cpu().numpy(), predicted_parameters['cylinder_axis'][0,i].cpu().numpy(), i)
        elif type_id == 3:
            json = cone_fitter.extract_predicted_parameters_as_json(predicted_parameters['cone_apex'][0,i].cpu().numpy(), predicted_parameters['cone_axis'][0,i].cpu().numpy(), predicted_parameters['cone_half_angle'][0,i].cpu().numpy(), i)
        list_json.append(json)
    return list_json