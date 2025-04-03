from torch_cluster import fps
import torch


def farthest_point_sampling(pcls, num_pnts):
    """
    Args:
        pcls:  A batch of point clouds, (B, N, 3) or (B, N, 3+3)(cor + nor).
        num_pnts:  Target number of points.
    Returns:
        sampled:  Target number of points, (B, S, 3).
    """
    ratio = 0.01 + num_pnts / pcls.size(1)
    sampled = []
    for i in range(pcls.size(0)):
        idx = fps(pcls[i], ratio=ratio, random_start=False)[:num_pnts]  # (24,)
        sampled.append(pcls[i:i + 1, idx, :])
    sampled = torch.cat(sampled, dim=0)
    return sampled