import torch
import matplotlib.pyplot as plt
import numpy as np
from pointnet2_ops import pointnet2_utils
# torch.set_printoptions(precision=9)


def normalize_point_cloud(input):
    if len(input.shape) == 2:
        axis = 0
    elif len(input.shape) == 3:
        axis = 1
    else:
        raise "input data format error"
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)), axis=axis, keepdims=True)
    input = input / furthest_distance
    return input, centroid, furthest_distance


def read_xyz_file(file_path):
    points = np.loadtxt(file_path).astype(np.float32)
    return points


def pairwise_distance(point_cloud):
    """

    Args:
        point_cloud: 1 2048 3   type:cuda tensor

    Returns:
        final_result: 1 2048 3   type:cuda tensor
    """

    og_batch_size = point_cloud.shape[0]  # batch size
    point_cloud = torch.squeeze(point_cloud)  # b n 3 -> n 3 :2048 3
    if og_batch_size == 1:
        point_cloud = torch.unsqueeze(point_cloud, 0)  # 1 2048 3

    point_cloud_transpose = torch.transpose(point_cloud, 1, 2)  # transpose matrix b n 3 -> b 3 n : 1  3 2048
    point_cloud_inner = torch.matmul(point_cloud, point_cloud_transpose)   # P*P^T 矩阵与其转置相乘 1  2048 2048
    point_cloud_inner = -2 * point_cloud_inner  # 所有元素乘以  -2
    temp = torch.square(point_cloud)  # 输入张量中的每个元素都平方
    point_cloud_square = torch.sum(temp, dim=-1, keepdim=True)  # 沿着坐标维度求和
    point_cloud_square_tranpose = torch.transpose(point_cloud_square, 1, 2)
    final_result = point_cloud_square + point_cloud_inner + point_cloud_square_tranpose
    return final_result

def High_Pass_Graph_Filter(point_cloud, k=350, dist=0.5, sigma=2.0):
    """
    Args:
        point_cloud: B N C    type：cuda tensor
        k:  number points of result
        dist: 两个点之间的距离小于等于dist，那么它们会被连接起来
        sigma: Gaussian kernel size variation

    Returns:  Edge_points:B N C  type：cuda tensor

    """
    B = point_cloud.shape[0]  # batch size
    N = point_cloud.shape[1]  # points num

    adj = pairwise_distance(point_cloud)  # adjacent matrix
    zero = torch.zeros_like(adj)
    one = torch.ones_like(adj)
    mask = torch.where(adj <= dist, one, zero)

    variation = adj / (sigma * sigma * -1)
    W_fake = torch.exp(variation)
    W = mask * W_fake  # 逐元素相乘
    I = torch.eye(N)
    I = I.unsqueeze(0)
    I = I.expand(B, -1, -1).cuda()
    W = W - I
    sum_W = torch.sum(input=W, dim=-1, keepdim=True)

    normalization = sum_W.repeat(1, 1, N)
    normalization = torch.where(normalization != 0, normalization, one)
    A = W / normalization
    H_A = I - A
    Filtered_signal = torch.matmul(H_A, point_cloud)
    temp = torch.square(Filtered_signal)

    L2_square = torch.sum(temp, dim=-1, keepdim=True)
    temp = torch.squeeze(L2_square, 2)
    Value, index = torch.topk(temp, k=k)

    temp = torch.arange(0, B)
    temp_2 = torch.reshape(temp, [B, 1])
    idx_bais = (temp_2 * N).cuda()
    idx_bais_tile = idx_bais.repeat(1, k)  # 若B=1，则创建了一个空tensor，都是0

    index_new = index + idx_bais_tile  # 都是0加index还是index
    point_cloud_reshape = torch.reshape(point_cloud, [B * N, -1])

    temp = torch.reshape(index_new, [B * k])
    new_point = torch.index_select(point_cloud_reshape, 0, temp)  # dim=0
    Edge_points = torch.reshape(new_point, [B, k, -1])

    return Edge_points


def main():
    # debug的时候需要的步骤
    file_path = 'laptop_2048.xyz'
    point_cloud = read_xyz_file(file_path)  # 2048 3
    point_cloud, centroid, furthest_distance = normalize_point_cloud(point_cloud)
    point_cloud = torch.from_numpy(point_cloud).cuda()  # 放到gpu上  2048 3

    # 这里开始都是cuda tensor
    temp = torch.unsqueeze(point_cloud, dim=0)  # 1 2048 3
    edge_points = High_Pass_Graph_Filter(temp, k=512, dist=0.25, sigma=2)

    point_cloud_np = point_cloud.cpu().numpy()
    point_cloud_np = point_cloud_np * furthest_distance + centroid

    edge_points_np = torch.squeeze(edge_points).cpu().numpy()

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.scatter(point_cloud_np[:, 0], point_cloud_np[:, 1], point_cloud_np[:, 2], c='b', s=1)
    ax1.set_title('Original Point Cloud')

    ax2.scatter(edge_points_np[:, 0], edge_points_np[:, 1], edge_points_np[:, 2], c='r', s=1)
    ax2.set_title('High Frequency Points')

    plt.show()


if __name__ == '__main__':
    main()
