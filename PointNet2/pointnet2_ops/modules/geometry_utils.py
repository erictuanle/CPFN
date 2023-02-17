import torch
from .. import cuda_ops

def pairwise_squared_distance(src, dst):
    """
    Calculate squared euclidean distance between each pair of points from src to dst.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Args:
        src: source points, [B, C, N]
        dst: target points, [B, C, M]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, _, N = src.shape
    _, _, M = dst.shape
    dist = -2 * torch.matmul(src.permute(0, 2, 1), dst)
    dist += torch.sum(src ** 2, dim=1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=1).view(B, 1, M)
    return dist


def select_point_subset(points, idx):
    """
    Select a different subset of points in each batch (same number, but different indices in each batch).
    If the indices have more than one dimension per batch, the returned point tensor is shaped like the indices
    (see args/returns for details).
    Args:
        points: input points data, [B, C, N]
        idx: sample index data, [B]+[*] (* may be any number of dimensions)
    Returns:
        new_points:, indexed points data, [B, C]+[*]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=idx.dtype, device=idx.device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, :, idx].permute(0, -1, *range(1, points.dim()+idx.dim()-3))
    return new_points


class _FastFarthestPointSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative farthest point sampling to select a set of npoint features that have the largest
        minimum distance
        Parameters
        ----------
        xyz : torch.Tensor
            [B, N, 3] tensor where N > npoint
        npoint : int32
            number of features in the sampled set
        Returns
        -------
        torch.Tensor
            [B, num_point] tensor containing the set
        Based on: https://github.com/erikwijmans/Pointnet2_PyTorch
        """
        return cuda_ops.farthest_point_sampling(xyz, npoint)

    @staticmethod
    def backward(xyz, a=None):
        return None, None

_fast_farthest_point_sample = _FastFarthestPointSample.apply

def farthest_point_sample(point_pos, num_point, fast=True):
    """
    Args:
        point_pos: pointcloud data, [B, C, N]
        num_point: number of samples
        fast: use faster version with custom CUDA kernel (only works with C==3)
    Returns:
        farthest_indices: sampled pointcloud index, [B, num_point]
    """
    if fast:
        if point_pos.shape[1] != 3:
            raise ValueError('Points must have exactly three position dimensions when using the fast method.')
        return _fast_farthest_point_sample(point_pos.permute(0, 2, 1).contiguous(), num_point).to(dtype=torch.long)
    else:
        device = point_pos.device
        B, C, N = point_pos.shape
        farthest_indices = torch.zeros(B, num_point, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest_index = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(num_point):
            farthest_indices[:, i] = farthest_index
            far_pos = point_pos[batch_indices, :, farthest_index].view(B, C, 1)
            dist = torch.sum((point_pos - far_pos) ** 2, dim=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest_index = torch.max(distance, -1)[1]
        return farthest_indices


class _FastBallQuery(torch.autograd.Function):
    @staticmethod
    def forward(ctx, radius, num_samples, point_pos, query_pos):
        # type: (Any, float, int, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        radius : float
            radius of the balls
        num_samples : int
            maximum number of features in the balls
        point_pos : torch.Tensor
            [B, N, 3] xyz coordinates of the features
        query_pos : torch.Tensor
            [B, S, 3] centers of the ball query
        Returns
        -------
        torch.Tensor
            [B, S, num_samples] tensor with the indicies of the features that form the query balls
        """
        return cuda_ops.ball_query(query_pos, point_pos, radius, num_samples)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


_fast_ball_query = _FastBallQuery.apply

def ball_query(radius, num_samples, point_pos, query_pos, fast=True):
    """
    Return the smaller set of: all points within a fixed radius of the query point, or the num_samples nearest neighbors.
    Args:
        radius: local region radius
        num_samples: max sample number in local region
        point_pos: all points, [B, C, N]
        query_pos: query points, [B, C, S]
        fast: use faster version with custom CUDA kernel (only works with C==3)
    Returns:
        group_indices: grouped point indices, [B, S, num_samples]
    """
    if fast:
        if point_pos.shape[1] != 3:
            raise ValueError('Points must have exactly three position dimensions when using the fast method.')
        return _fast_ball_query(
            radius, num_samples, point_pos.permute(0, 2, 1).contiguous(), query_pos.permute(0, 2, 1).contiguous()).to(dtype=torch.long)
    else:
        device = point_pos.device
        B, _, N = point_pos.shape
        _, _, S = query_pos.shape
        group_indices = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
        sqrdists = pairwise_squared_distance(query_pos, point_pos)
        group_indices[sqrdists > radius ** 2] = N
        group_indices = group_indices.sort(dim=-1)[0][:, :, :num_samples]
        group_first = group_indices[:, :, 0].view(B, S, 1).repeat([1, 1, num_samples])
        mask = group_indices == N
        group_indices[mask] = group_first[mask]
        return group_indices

class _FastThreeNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, unknown, known):
        # type: (Any, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
        r"""
            Find the three nearest neighbors of unknown in known
        Parameters
        ----------
        unknown : torch.Tensor
            [B, S, 3] tensor of known features
        known : torch.Tensor
            [B, N, 3] tensor of unknown features
        Returns
        -------
        dist : torch.Tensor
            [B, S, 3] l2 distance to the three nearest neighbors
        idx : torch.Tensor
            [B, S, 3] index of 3 nearest neighbors
        """
        dist2, idx = cuda_ops.three_nn(unknown, known)

        return torch.sqrt(dist2), idx

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None

_fast_three_nn = _FastThreeNN.apply

def three_nn(point_pos, query_pos, fast=True):
    """
    Return the three nearest neighbors for each of the query points.
    Args:
        point_pos: all points, [B, C, N]
        query_pos: query points, [B, C, S]
        fast: use faster version with custom CUDA kernel (only works with C==3)
    Returns:
        dists: squared euclidean distances, [B, S, 3]
        indices: indices of the nearest neighbors, [B, S, 3]
    """
    if fast:
        if point_pos.shape[1] != 3:
            raise ValueError('Points must have exactly three position dimensions when using the fast method.')
        dists, indices = _fast_three_nn(
            query_pos.permute(0, 2, 1).contiguous(),
            point_pos.permute(0, 2, 1).contiguous())
        indices = indices.to(dtype=torch.long)
        return dists, indices
    else:
        dists = pairwise_squared_distance(query_pos, point_pos)
        dists, indices = dists.sort(dim=-1)
        dists, indices = dists[:, :, :3], indices[:, :, :3]
        return dists, indices

class _FastThreeWeightedSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        # type(Any, torch.Tensor, torch.Tensor, torch.Tensor) -> Torch.Tensor
        r"""
            Performs weight linear interpolation on 3 features
        Parameters
        ----------
        features : torch.Tensor
            [B, C, N] Features descriptors to be interpolated from
        idx : torch.Tensor
            [B, S, 3] three nearest neighbors of the target features in features
        weight : torch.Tensor
            [B, S, 3] weights
        Returns
        -------
        torch.Tensor
            [B, C, S] tensor of the interpolated features
        """
        _, _, N = features.size()
        # S = idx.size(1)

        ctx.three_weighted_sum_for_backward = (idx, weight, N)
        return cuda_ops.three_weighted_sum(features, idx.int(), weight)

    @staticmethod
    def backward(ctx, grad_out):
        # type: (Any, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        r"""
        Parameters
        ----------
        grad_out : torch.Tensor
            (B, C, S) tensor with gradients of ouputs
        Returns
        -------
        grad_features : torch.Tensor
            (B, C, N) tensor with gradients of features
        None
        None
        """
        idx, weight, N = ctx.three_weighted_sum_for_backward

        grad_features = cuda_ops.three_weighted_sum_grad(
            grad_out.contiguous(), idx.int(), weight, N
        )

        return grad_features, None, None

_fast_three_weighted_sum = _FastThreeWeightedSum.apply

def three_weighted_sum(point_feats, indices, weights, fast=True):
    """
    Intrepolate three nearest neighbors for each of the query points.
    Args:
        point_feats: all points, [B, C, N]
        indices: indices of the points to be summed, [B, S, 3]
        weights: weights of the points to be summed, [B, S, 3]
        fast: use faster version with custom CUDA kernel
    Returns:
        weighted sum of each triple [B, C, S]
    """
    if fast:
        return _fast_three_weighted_sum(point_feats, indices, weights)
    else:
        return torch.sum(
            select_point_subset(point_feats, indices) *
            weights.view(indices.shape[0], 1, indices.shape[1], indices.shape[2]), dim=-1)
