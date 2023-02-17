# Importation of packages
import torch
import numpy as np
if __name__ == '__main__':
    import tensorflow as tf
from torch.autograd import gradcheck

def guard_one_over_matrix(M, min_abs_value=1e-10):
    _, row, _ = M.size()
    device = M.get_device()
    up = torch.triu(torch.clamp(M, min=min_abs_value, max=None), diagonal=0)
    low = torch.tril(torch.clamp(M, min=None, max=-min_abs_value), diagonal=0)
    M = up + low
    M = M + torch.eye(row).to(device)
    M = 1 / M
    M = M - torch.eye(row).to(device)
    return M

def guard_one_over_matrix_tensorflow(M, min_abs_value=1e-10):
    up = tf.matrix_band_part(tf.maximum(min_abs_value, M), 0, -1)
    low = tf.matrix_band_part(tf.minimum(-min_abs_value, M), -1, 0)
    M = up + low
    M += tf.eye(tf.shape(M)[1])
    M = 1 / M
    M -= tf.eye(tf.shape(M)[1])
    return M

if __name__ == '__main__':
    batch_size = 100
    P = 5
    device = torch.device('cuda:0')
    np.random.seed(0)
    M = np.random.randn(batch_size, P, P)
    M_torch = torch.from_numpy(M).float().to(device)
    M_torch = guard_one_over_matrix(M_torch)
    M_torch = M_torch.detach().cpu().numpy()
    print('M_torch', M_torch)
    # Debugging with Tensorflow
    M_tensorflow = tf.constant(M, dtype=tf.float32)
    M_tensorflow_ = guard_one_over_matrix_tensorflow(M_tensorflow)
    sess = tf.Session()
    M_tensorflow = sess.run(M_tensorflow_)
    print(np.max(np.abs(M_tensorflow - M_torch)))

def compute_svd_K(s):
    # s should be BxP
    # res[b,i,j] = 1/(s[b,i]^2 - s[b,j]^2) if i != j, 0 otherwise
    # res will be BxPxP
    s = s**2
    res = s.unsqueeze(2) - s.unsqueeze(1)
    # making absolute value in res is at least 1e-10
    res = guard_one_over_matrix(res)
    return res

def compute_svd_K_tensorflow(s):
    # s should be BxP
    # res[b,i,j] = 1/(s[b,i]^2 - s[b,j]^2) if i != j, 0 otherwise
    # res will be BxPxP
    s = tf.square(s)
    res = tf.expand_dims(s, 2) - tf.expand_dims(s, 1)
    # making absolute value in res is at least 1e-10
    res = guard_one_over_matrix_tensorflow(res)
    return res

if __name__ == '__main__':
    batch_size = 100
    P = 5
    device = torch.device('cuda:0')
    np.random.seed(0)
    s = np.random.randn(batch_size, P)
    s_torch = torch.from_numpy(s).float().to(device)
    res_torch = compute_svd_K(s_torch)
    res_torch = res_torch.detach().cpu().numpy()
    print('res_torch', res_torch)
    # Debugging with Tensorflow
    s_tensorflow = tf.constant(s, dtype=tf.float32)
    res_tensorflow = compute_svd_K_tensorflow(s_tensorflow)
    sess = tf.Session()
    res_tensorflow = sess.run(res_tensorflow)
    print(np.max(np.abs(res_tensorflow - res_torch)))

def custom_svd_v_column_tensorflow(M, col_index=-1):
    # Must make sure M is finite. Otherwise cudaSolver might fail.
    assert_op = tf.Assert(tf.logical_not(tf.reduce_any(tf.logical_not(tf.is_finite(M)))), [M], summarize=10)
    with tf.control_dependencies([assert_op]):
        with tf.get_default_graph().gradient_override_map({'Svd': 'CustomSvd'}):
            s, u, v = tf.svd(M, name='Svd')  # M = usv^T
    return v[:, :, col_index]

def register_custom_svd_gradient_tensorflow():
    tf.RegisterGradient('CustomSvd')(custom_gradient_svd_tensorflow)

def custom_gradient_svd_tensorflow(op, grad_s, grad_u, grad_v):
    s, u, v = op.outputs
    # s - BxP
    # u - BxNxP, N >= P
    # v - BxPxP
    v_t = tf.transpose(v, [0, 2, 1])
    K = compute_svd_K_tensorflow(s)
    inner = tf.transpose(K, [0, 2, 1]) * tf.matmul(v_t, grad_v)
    inner = (inner + tf.transpose(inner, [0, 2, 1])) / 2
    # ignoring gradient coming from grad_s and grad_u for our purpose
    res = tf.matmul(u, tf.matmul(2 * tf.matmul(tf.matrix_diag(s), inner), v_t))
    return res

if __name__ == '__main__' and 1:
    register_custom_svd_gradient_tensorflow()

    batch_size = 100
    P = 5
    device = torch.device('cuda:0')
    np.random.seed(0)
    M = np.random.randn(batch_size, P, P)
    M_tensorflow = tf.constant(M, dtype=tf.float32)
    M_input = tf.placeholder(dtype=tf.float32, shape=[None, P, P])

    with tf.get_default_graph().gradient_override_map({'Svd': 'CustomSvd'}):
        s, u, v = tf.svd(M_input, name='Svd')  # M = usv^T
    with tf.Session() as sess:
        error = tf.test.compute_gradient_error(M_input, [batch_size, P, P], v, [batch_size, P, P])
        print('Error: ', error)

class Custom_svd_v_colum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, M, col_index=-1):
        u, s, v = torch.svd(M, some=True)
        out = v[:,:,col_index]
        ctx.save_for_backward(u, s, v)
        ctx.col_index = col_index
        return out
    @staticmethod
    def backward(ctx, grad_out):
        u, s, v = ctx.saved_tensors
        col_index = ctx.col_index
        grad_v = torch.zeros_like(v)
        grad_v[:,:,col_index] = grad_out
        v_t = v.transpose(1, 2)
        K = compute_svd_K(s)
        inner = K.transpose(1,2) * torch.bmm(v_t, grad_v)
        inner = (inner + inner.transpose(1, 2)) / 2
        # ignoring gradient coming from grad_s and grad_u for our purpose
        res = torch.bmm(u, torch.bmm(2 * torch.bmm(torch.diag_embed(s, offset=0, dim1=-2, dim2=-1), inner), v_t))
        return res, None

if __name__ == '__main__':
    batch_size = 100
    P = 5
    device = torch.device('cuda:0')
    np.random.seed(0)
    M = np.random.randn(batch_size, P, P)
    M_torch = torch.from_numpy(M).float().to(device)
    out_torch = Custom_svd_v_colum().apply(M_torch)
    out_torch = out_torch.detach().cpu().numpy()
    print('out_torch', out_torch)
    # Debugging with Tensorflow
    M_tensorflow = tf.constant(M, dtype=tf.float32)
    out_tensorflow = custom_svd_v_column_tensorflow(M_tensorflow)
    sess = tf.Session()
    out_tensorflow = sess.run(out_tensorflow)
    print(np.minimum(np.abs(out_tensorflow-out_torch), np.abs(out_tensorflow+out_torch)).max())

if __name__ == '__main__' and 1:
    batch_size = 4
    P = 5
    device = torch.device('cuda:0')
    np.random.seed(0)
    M = np.random.randn(batch_size, P, P)
    M_torch = torch.from_numpy(M).float().to(device)
    M_torch = torch.nn.Parameter(M_torch, requires_grad=True)
    try:
        custom_svd_v_colum = Custom_svd_v_colum.apply
        torch.autograd.gradcheck(custom_svd_v_colum, (M_torch, -1), raise_exception=True)
        print('Test on Custom_svd_v_colum: Success')
    except:
        print('Test on Custom_svd_v_colum: Failure')
        raise

if __name__ == '__main__' and 1:
    register_custom_svd_gradient_tensorflow()

    batch_size = 100
    P = 5
    device = torch.device('cuda:0')
    np.random.seed(0)
    M = np.random.randn(batch_size, P, P)
    M_torch = torch.from_numpy(M).float().to(device)
    M_torch = torch.nn.Parameter(M_torch, requires_grad=True)
    out = Custom_svd_v_colum().apply(M_torch)
    out.backward(torch.ones_like(out))
    M_grad_torch = M_torch.grad.detach().cpu().numpy()

    M_tensorflow = tf.constant(M, dtype=tf.float32)
    out = custom_svd_v_column_tensorflow(M_tensorflow)
    M_grad_tensorflow = tf.gradients(out, [M_tensorflow])[0]
    sess = tf.Session()
    M_grad_tensorflow = sess.run(M_grad_tensorflow)

    print(np.minimum(np.abs(M_grad_tensorflow - M_grad_torch), np.abs(M_grad_tensorflow + M_grad_torch)).max())

def solve_weighted_tls(A, W):
    # A - BxNx3
    # W - BxN, positive weights
    # Find solution to min x^T A^T diag(W) A x = min ||\sqrt{diag(W)} A x||^2, subject to ||x|| = 1
    batch_size, num_points, _ = A.size()
    A_p = A.unsqueeze(2) * A.unsqueeze(3) # BxNx3x3
    W_p = W.view(batch_size, num_points, 1, 1)
    M = torch.sum(W_p * A_p, dim=1) # Bx3x3
    x = Custom_svd_v_colum().apply(M) # Bx3
    return x

def solve_weighted_tls_tensorflow(A, W):
    # A - BxNx3
    # W - BxN, positive weights
    # Find solution to min x^T A^T diag(W) A x = min ||\sqrt{diag(W)} A x||^2, subject to ||x|| = 1
    A_p = tf.expand_dims(A, axis=2) * tf.expand_dims(A, axis=3) # BxNx3x3
    W_p = tf.expand_dims(tf.expand_dims(W, axis=2), axis=3) # BxNx1x1
    M = tf.reduce_sum(W_p * A_p, axis=1) # Bx3x3
    x = custom_svd_v_column_tensorflow(M) # Bx3
    return x

if __name__ == '__main__':
    batch_size = 100
    num_points = 1024
    device = torch.device('cuda:0')
    np.random.seed(0)
    A = np.random.randn(batch_size, num_points, 3)
    W = np.random.randn(batch_size, num_points)
    A_torch = torch.from_numpy(A).float().to(device)
    W_torch = torch.from_numpy(W).float().to(device)
    x_torch = solve_weighted_tls(A_torch, W_torch)
    x_torch = x_torch.detach().cpu().numpy()
    print('x_torch', x_torch)
    # Debugging with Tensorflow
    A_tensorflow = tf.constant(A, dtype=tf.float32)
    W_tensorflow = tf.constant(W, dtype=tf.float32)
    x_tensorflow = solve_weighted_tls_tensorflow(A_tensorflow, W_tensorflow)
    sess = tf.Session()
    x_tensorflow = sess.run(x_tensorflow)
    print(np.minimum(np.abs(x_tensorflow-x_torch), np.abs(x_tensorflow+x_torch)).max())