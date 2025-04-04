import sys

import torch
import torch.nn as nn

sys.path.append("../")
import math

import pointnet2_ops.pointnet2_utils as pn2_utils
from pc_eval.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from pc_eval.emd_module.emd_module import emdModule

from knn_cuda import KNN


class Loss(nn.Module):
    def __init__(self, radius=1.0):
        super(Loss, self).__init__()
        self.radius = radius
        self.knn_uniform = KNN(k=2, transpose_mode=True)
        self.knn_repulsion = KNN(k=20, transpose_mode=True)
        self.chamLoss = chamfer_3DDist()
        self.emd = emdModule()

    def get_emd_loss(self, pred, gt, radius=1.0, eps=1.0, iters=512):
        """
        pred and gt is B N 3
        """
        dis, _ = self.emd(pred, gt, eps, iters)
        dis = torch.mean(torch.sqrt(dis), dim=1)
        dis = dis / radius
        return torch.mean(dis)

    def get_cd_loss(self, pred, gt, radius=1.0):
        """
        pred and gt is B N 3
        """
        dist1, dist2, _, _ = self.chamLoss(pred, gt)
        CD_dist = torch.mean(dist1 + dist2, dim=1, keepdims=True)
        CD_dist = CD_dist / radius
        return torch.mean(CD_dist)

    def get_hd_loss(self, pred, gt, radius=1.0):
        dist1, dist2, _, _ = self.chamLoss(pred, gt)
        HD_dist = (
            torch.max(dist1, dim=1, keepdims=True)[0]
            + torch.max(dist2, dim=1, keepdims=True)[0]
        )
        HD_dist = HD_dist / radius
        return torch.mean(HD_dist)

    def get_uniform_loss(
        self, pcd, percentage=[0.004, 0.006, 0.008, 0.010, 0.012], radius=1.0
    ):
        B, N, C = pcd.shape[0], pcd.shape[1], pcd.shape[2]
        npoint = int(N * 0.05)
        loss = 0
        further_point_idx = pn2_utils.furthest_point_sample(
            pcd.permute(0, 2, 1).contiguous(), npoint
        )
        new_xyz = pn2_utils.gather_operation(
            pcd.permute(0, 2, 1).contiguous(), further_point_idx
        )  # B,C,N
        for p in percentage:
            nsample = int(N * p)
            r = math.sqrt(p * radius)
            disk_area = math.pi * (radius ** 2) / N

            idx = pn2_utils.ball_query(
                r, nsample, pcd.contiguous(), new_xyz.permute(0, 2, 1).contiguous()
            )  # b N nsample

            expect_len = math.sqrt(disk_area)

            grouped_pcd = pn2_utils.grouping_operation(
                pcd.permute(0, 2, 1).contiguous(), idx
            )  # B C N nsample
            grouped_pcd = grouped_pcd.permute(0, 2, 3, 1)  # B N nsample C

            grouped_pcd = torch.cat(
                torch.unbind(grouped_pcd, dim=1), dim=0
            )  # B*N nsample C

            dist, _ = self.knn_uniform(grouped_pcd, grouped_pcd)
            uniform_dist = dist[:, :, 1:]  # B*N nsample 1
            uniform_dist = torch.abs(uniform_dist + 1e-8)
            uniform_dist = torch.mean(uniform_dist, dim=1)
            uniform_dist = (uniform_dist - expect_len) ** 2 / (expect_len + 1e-8)
            mean_loss = torch.mean(uniform_dist)
            mean_loss = mean_loss * math.pow(p * 100, 2)
            loss += mean_loss
        return loss / len(percentage)

    def get_repulsion_loss(self, pcd, h=0.0005):
        dist, idx = self.knn_repulsion(pcd, pcd)  # B N k

        dist = dist[:, :, 1:5] ** 2  # top 4 cloest neighbors

        loss = torch.clamp(-dist + h, min=0)
        loss = torch.mean(loss)
        # print(loss)
        return loss

    def get_l2_regular_loss(self, model, alpha):
        l2_loss = torch.tensor(0.0, requires_grad=True)
        for name, param in model.named_parameters():
            if "bias" not in name:
                l2_loss = l2_loss +  torch.norm(param)
        return alpha * l2_loss

    def get_discriminator_loss(self, pred_fake, pred_real):
        real_loss = torch.mean((pred_real - 1) ** 2)
        fake_loss = torch.mean(pred_fake ** 2)
        loss = real_loss + fake_loss
        return loss

    def get_generator_loss(self, pred_fake):
        fake_loss = torch.mean((pred_fake - 1) ** 2)
        return fake_loss

    def get_discriminator_loss_single(self, pred, label=True):
        if label == True:
            loss = torch.mean((pred - 1) ** 2)
            return loss
        else:
            loss = torch.mean((pred) ** 2)
            return loss


if __name__ == "__main__":
    loss = Loss().cuda()
    point_cloud = torch.rand(4, 4096, 3).cuda()
    uniform_loss = loss.get_uniform_loss(point_cloud)
    repulsion_loss = loss.get_repulsion_loss(point_cloud)
    emd_Loss = loss.get_emd_loss(point_cloud, point_cloud)
    cd_loss = loss.get_cd_loss(point_cloud, point_cloud)
    hd_Loss = loss.get_hd_loss(point_cloud, point_cloud)
    print(emd_Loss, cd_loss, emd_Loss.shape, uniform_loss.shape)
    print(hd_Loss)
