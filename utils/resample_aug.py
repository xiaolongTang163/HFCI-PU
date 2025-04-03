import os

import numpy
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from torch_cluster import knn_graph
from kmeans_pytorch import kmeans
import pytorch3d
from utils.data_loss import *
from utils.net_utils import *
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation


def make_patches_for_pcl_pair(pcl_low, pcl_high, patch_size, num_patches, ratio, is_middle=True):
    """
    准备训练对  一般是256 1024 对
    Args:
        pcl_low:  Low-resolution point cloud, (N, 3).
        pcl_high: High-resolution point cloud, (rN, 3).
        patch_size:   Patch size M.
        num_patches, self.args.ratio:  Number of patches P. 1
        ratio:    Ratio r.
    Returns:
        (P, M, 3), (P, rM, 3)
    """
    N = pcl_low.size(0)
    seed_idx = torch.randperm(N)[:num_patches]   # (P, )
    seed_pnts = pcl_low[seed_idx].unsqueeze(0)   # (1, P, 3)  选取195个种子点
    _, _, pat_low = pytorch3d.ops.knn_points(seed_pnts, pcl_low.unsqueeze(0), K=patch_size, return_nn=True)
    pat_low = pat_low[0]    # (P, M, 3)
    _, _, pat_high = pytorch3d.ops.knn_points(seed_pnts, pcl_high.unsqueeze(0), K=patch_size*ratio, return_nn=True)
    pat_high = pat_high[0]
    if is_middle:
        pat_middle_idx = furthest_point_sample(pat_high.cuda(), patch_size * ratio // 2)
        pat_middle = gather_operation(pat_high.cuda().permute(0, 2, 1).contiguous(), pat_middle_idx).permute(0, 2, 1).contiguous().cpu()
        return pat_low, pat_high, pat_middle
    else:
        return pat_low, pat_high


class Make_Patch(Dataset):
    """
    Args:
        pc: (N * 3)
    return:
        pcl_low: (N1 * 3)
    """

    def __init__(self, args, root, subset, cat_low, transform=None):
        super().__init__()
        self.dir = os.path.join(root, subset)
        self.dir_cat_low = os.path.join(self.dir, cat_low)
        self.args = args
        self.transform = transform
        self.patches_low = []
        self.patches_high =[]
        self.patches = []
        for fn in tqdm(os.listdir(self.dir_cat_low), desc='Loading'):
            if fn[-3:] != 'xyz':
                continue
            pc1_path = os.path.join(self.dir_cat_low, fn)
            pc1 = torch.FloatTensor(np.loadtxt(pc1_path, dtype=np.float32))
            N, _ = pc1.shape

           ## fast point sample
            pcl_1 = farthest_point_sampling(pc1.unsqueeze(0), int(N / 4)).squeeze(0)  # 最远点采样  /4
            pcl_1 = pcl_1.squeeze(0)
            self.name = fn[:-4]
            pat1, pat2, pat3 = make_patches_for_pcl_pair(pcl_1, pc1, args.num_point, args.num_patches, args.up_ratio)

            for i in range(pat1.size(0)):
                self.patches.append((pat1[i], pat2[i], pat3[i]))  # + '_ %s' % i))

        #     self.patches_low.append(pat1)
        #     self.patches_high.append(pat2)
        # a = torch.cat(self.patches_low, dim=0).numpy()
        # b = torch.cat(self.patches_high, dim=0).numpy()
        # from dataset.a_Helper.PC_utils import h5_utils
        # h5_utils.save_h5_file([a, b], ["poisson_256", "poisson_1024"], "sketchfab_256_1024_poisson.h5")
        # print("finish")

    def __len__(self):
        # assert len(self.pointclouds_cat_low) == len(self.pointclouds_cat_high)
        return len(self.patches)

    def __getitem__(self, idx):
        patches = {
            'pcl_low': self.patches[idx][0].clone(),
            'pcl_high': self.patches[idx][1].clone(),
            'pcl_middle': self.patches[idx][2].clone(),
        }
        if self.transform is not None:
            patches = self.transform(patches)
        return patches


class Make_Patch_Supervised(Dataset):
    """
    Args:
        pc: (N * 3)
    return:
        pcl_low: (N1 * 3)
    """

    def __init__(self, args, root, subset, cat_low, cat_high, transform=None):
        super().__init__()
        self.dir = os.path.join(root, subset)
        self.dir_cat_low = os.path.join(self.dir, cat_low)
        self.dir_cat_high = os.path.join(self.dir, cat_high)
        self.args = args
        self.transform = transform
        self.patches = []
        self.patches_low = []
        self.patches_middle = []
        self.patches_high = []
        for fn in tqdm(os.listdir(self.dir_cat_low), desc='Loading'):
            if fn[-3:] != 'xyz':
                continue
            pc1_path = os.path.join(self.dir_cat_low, fn)
            pc1 = torch.FloatTensor(np.loadtxt(pc1_path, dtype=np.float32))
            pc2_path = os.path.join(self.dir_cat_high, fn)
            pc2 = torch.FloatTensor(np.loadtxt(pc2_path, dtype=np.float32))
            N, _ = pc1.shape

            ## random select
            self.name = fn[:-4]
            pat1, pat2, pat_middle = make_patches_for_pcl_pair(pc1, pc2, self.args.patch_size, self.args.num_patches, args.upsample_rate)

            for i in range(pat1.size(0)):
                self.patches.append((pat1[i], pat2[i], pat_middle[i]))


    def __len__(self):
        # assert len(self.pointclouds_cat_low) == len(self.pointclouds_cat_high)
        return len(self.patches)

    def __getitem__(self, idx):
        patches = {
            'pcl_low': self.patches[idx][0].clone(),
            'pcl_high': self.patches[idx][1].clone(),
            'pcl_middle': self.patches[idx][2].clone(),
        }
        if self.transform is not None:
            patches = self.transform(patches)  # middle我加的没有数据增强 所以效果差
        return patches