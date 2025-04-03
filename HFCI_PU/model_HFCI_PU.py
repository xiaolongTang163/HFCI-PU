import torch
import torch.nn as nn
import numpy as np
from torch import einsum
from pointnet2_ops.pointnet2_utils import grouping_operation
from utils import high_f_torch as helper
from pointnet2_ops import pointnet2_utils
import math
import argparse


def is_square_root_integer(num):
    sqrt_num = math.sqrt(num)
    return sqrt_num.is_integer()


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_knn_point(k, xyz, new_xyz):
    dist = square_distance(new_xyz, xyz)
    _, group_idx = dist.topk(k, largest=False)
    return group_idx.int()


class MLP_CONV(nn.Module):  #
    def __init__(self, in_channel, layer_dims, bn=None):
        super(MLP_CONV, self).__init__()
        layers = []
        last_channel = in_channel
        for out_channel in layer_dims[:-1]:
            layers.append(nn.Conv1d(last_channel, out_channel, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        layers.append(nn.Conv1d(last_channel, layer_dims[-1], 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.mlp(inputs)


class MLP_Res(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=None, out_dim=128):
        super(MLP_Res, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_dim
        self.conv_1 = nn.Conv1d(in_dim, hidden_dim, 1)
        self.conv_2 = nn.Conv1d(hidden_dim, out_dim, 1)
        self.conv_shortcut = nn.Conv1d(in_dim, out_dim, 1)

    def forward(self, x):
        """
        Args:
            x: (B, out_dim, n)
        """
        shortcut = self.conv_shortcut(x)
        out = self.conv_2(torch.relu(self.conv_1(x))) + shortcut
        return out


class SubNetwork(nn.Module):
    def __init__(self, up_ratio=2, stage=1, high_pass=False):
        super(SubNetwork, self).__init__()
        if stage == 1:
            self.feature_extractor = TransformerExtractorDense(dim_feat=128, hidden_dim=64)
        elif stage == 2:
            self.feature_extractor = TransformerExtractor(dim_feat=128, hidden_dim=64, channel=True)
        else:
            self.feature_extractor = TransformerExtractor(dim_feat=128, hidden_dim=64)

        self.up_unit = UpSamplingUnit(up_ratio=up_ratio)
        self.regressor = MLP_CONV(in_channel=128, layer_dims=[64, 3])
        if high_pass:
            self.high_pass = high_pass
        else:
            self.high_pass = False

    def forward(self, points):

        point_feat = self.feature_extractor(points)
        up_feat, duplicated_point = self.up_unit(point_feat, points)
        offest = self.regressor(up_feat)
        up_point = duplicated_point + torch.tanh(offest)
        if self.high_pass:
            """
                Args:
                    cuda tensor b 3 n
                output:
                    cuda tensor b 3 n
            """
            points_high = helper.High_Pass_Graph_Filter(
                up_point.permute(0, 2, 1),
                k=up_point.shape[2] // 2, dist=0.5, sigma=2.0
            ).permute(0, 2, 1)

            added_point = torch.cat([up_point, points_high], dim=-1)  # (2,3,768)

            sampled_index = pointnet2_utils.furthest_point_sample(
                added_point.permute(0, 2, 1).contiguous(), up_point.shape[2])

            sampled_index = sampled_index.unsqueeze(-1).expand(-1, -1,
                                                               added_point.shape[1]).long()  # B npoint -> B npoint 3
            selected_points = torch.gather(added_point.permute(0, 2, 1), 1, sampled_index).permute(0, 2, 1).contiguous()

            return selected_points
        else:
            return up_point


class TransformerExtractor(nn.Module):
    def __init__(self, dim_feat, hidden_dim, channel=False):
        super(TransformerExtractor, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, dim_feat])
        self.mlp_2 = MLP_CONV(in_channel=dim_feat * 2, layer_dims=[dim_feat * 2, dim_feat])

        if channel:
            self.point_transformer = TransformerChannel(dim_feat, dim=hidden_dim)
        else:
            self.point_transformer = Transformer(dim_feat, dim=hidden_dim)

    def forward(self, points):
        feature_1 = self.mlp_1(points)
        global_feature = torch.max(feature_1, 2, keepdim=True)[0]

        feature_2 = torch.cat([feature_1, global_feature.repeat((1, 1, feature_1.size(2)))], 1)
        feature_3 = self.mlp_2(feature_2)
        point_feat = self.point_transformer(feature_3, points)
        return point_feat


class Transformer(nn.Module):
    """
    [Point Transformer](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_Point_Transformer_ICCV_2021_paper.pdf)

    feed forward of transformer
    Args:
        x: Tensor of features, (B, in_channel, n)
        pos: Tensor of positions, (B, 3, n)

    Returns:
        y: Tensor of features with attention, (B, in_channel, n)

    """

    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(Transformer, self).__init__()

        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        identity = x

        x = self.linear_start(x)
        b, dim, n = x.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn_point(self.n_knn, pos_flipped, pos_flipped)

        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn

        qk_rel = query.reshape((b, -1, n, 1)) - key
        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn

        qk_rel_list = [qk_rel]
        pos_rel_list = [pos_rel]
        att_mlp_list = [self.pos_mlp]
        attn_mlp_list = [self.attn_mlp]
        linear_end_list = [self.linear_end]

        y_list = []

        for qk_rel, pos_rel, pos_mlp, attn_mlp, linear_end in zip(
                qk_rel_list, pos_rel_list, att_mlp_list, attn_mlp_list, linear_end_list):
            pos_embedding = pos_mlp(pos_rel)  # b, dim, n, n_knn

            attention = attn_mlp(qk_rel + pos_embedding)
            attention = torch.softmax(attention, -1)

            value_temp = value.reshape((b, -1, n, 1)) + pos_embedding

            agg = einsum('b c i j, b c i j -> b c i', attention, value_temp)  # b, dim, n
            y = linear_end(agg)
            y_list.append(y)

        y = torch.cat(y_list, dim=1)
        return y + identity


class TransformerChannel(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(TransformerChannel, self).__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, 16, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(16, 16 * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(16 * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(16 * attn_hidden_multiplier, 16, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)

        self.psi = 4
        self.w = 64 // self.psi  # 16
        self.d = self.w // 2  # 8
        self.heads = 2 * self.psi - 1  # 7
        self.linear_end = nn.Conv1d(self.heads * self.w, in_channel, 1)

    def forward(self, x, pos):
        identity = x
        x = self.linear_start(x)
        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn_point(self.n_knn, pos_flipped, pos_flipped)

        key = self.conv_key(x)  # 4 64 256
        value = self.conv_value(x)
        query = self.conv_query(x)

        d = self.d  # 16
        w = self.w  # 32
        agg_list = []
        b, _, n = key.shape
        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)
        pos_embedding = self.pos_mlp(pos_rel)  # b 16 n k
        for m in range(self.heads):  # head:7
            q_m = query[:, m * d:m * d + w, :]  # b 16 n
            k_m = key[:, m * self.d:m * self.d + self.w, :]
            v_m = value[:, m * self.d:m * self.d + self.w, :]

            k_m = grouping_operation(k_m.contiguous(), idx_knn)  # b 16 n k
            qk_rel = q_m.reshape((b, -1, n, 1)) - k_m
            attention = self.attn_mlp(qk_rel + pos_embedding)
            attention = torch.softmax(attention, -1)
            v_m = v_m.reshape((b, -1, n, 1)) + pos_embedding
            agg = einsum('b c i j, b c i j -> b c i', attention, v_m)
            agg_list.append(agg)  #
        agg = torch.cat(agg_list, dim=1)

        y = self.linear_end(agg)

        return y + identity


class TransformerExtractorDense(nn.Module):
    def __init__(self, dim_feat, hidden_dim):
        super(TransformerExtractorDense, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, dim_feat])
        self.mlp_2 = MLP_CONV(in_channel=dim_feat * 2, layer_dims=[dim_feat * 2, dim_feat])

        self.point_transformer = Transformer(dim_feat, dim=hidden_dim)

        self.comp_1 = nn.Sequential(nn.Conv1d(128 * 2, 128, 1),
                                    nn.BatchNorm1d(128),
                                    nn.ReLU()
                                    )
        self.comp_2 = nn.Sequential(nn.Conv1d(128 * 3, 128 * 2, 1),
                                    nn.BatchNorm1d(128 * 2),
                                    nn.ReLU(),
                                    nn.Conv1d(128 * 2, 128, 1),
                                    )
        self.comp_3 = nn.Sequential(nn.Conv1d(128 * 4, 128 * 3, 1),
                                    nn.BatchNorm1d(128 * 3),
                                    nn.ReLU(),
                                    nn.Conv1d(128 * 3, 128 * 2, 1),
                                    nn.ReLU(),
                                    nn.Conv1d(128 * 2, 128, 1),
                                    )

    def forward(self, points):
        feature_1 = self.mlp_1(points)
        global_feature = torch.max(feature_1, 2, keepdim=True)[0]
        feature_2 = torch.cat([feature_1, global_feature.repeat((1, 1, feature_1.size(2)))], 1)
        feature_3 = self.mlp_2(feature_2)

        point_feat_1 = self.point_transformer(feature_3, points)
        point_feat_1 = self.comp_1(torch.cat([point_feat_1, feature_3], dim=1))
        point_feat_2 = self.point_transformer(point_feat_1, points)
        point_feat_2 = self.comp_2(torch.cat([point_feat_2, point_feat_1, feature_3], dim=1))
        point_feat_3 = self.point_transformer(point_feat_2, points)
        point_feat_3 = self.comp_3(torch.cat([point_feat_3, point_feat_2, point_feat_1, feature_3], dim=1))

        return point_feat_3


class UpSamplingUnit(nn.Module):
    def __init__(self, up_ratio=2):
        super(UpSamplingUnit, self).__init__()
        self.mlp_1 = MLP_CONV(in_channel=128, layer_dims=[64, 32])
        self.mlp_2 = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)
        self.deconv_branch = nn.ConvTranspose1d(32, 128, up_ratio, up_ratio, bias=False)
        self.duplicated_branch = nn.Upsample(scale_factor=up_ratio)

    def forward(self, point_feat, points):
        deconved_feat = self.deconv_branch(self.mlp_1(point_feat))
        duplicated_feat = self.duplicated_branch(point_feat)
        up_feat = self.mlp_2(torch.cat([deconved_feat, duplicated_feat], 1))
        up_feat = torch.relu(up_feat)
        duplicated_point = self.duplicated_branch(points)
        return up_feat, duplicated_point


class HFCIPU(torch.nn.Module):
    def __init__(self, args):
        super(HFCIPU, self).__init__()
        step_up_rate = int(np.sqrt(args.up_ratio))
        if is_square_root_integer(args.up_ratio):
            self.upsampling_stage_1 = SubNetwork(up_ratio=step_up_rate, stage=1, high_pass=True)
            self.upsampling_stage_2 = SubNetwork(up_ratio=step_up_rate, stage=2, high_pass=True)
            self.refinement_stage = SubNetwork(up_ratio=1, stage=3, high_pass=True)
        else:
            self.upsampling_stage_1 = SubNetwork(up_ratio=args.up_ratio, stage=1, high_pass=True)
            self.upsampling_stage_2 = SubNetwork(up_ratio=1, stage=2, high_pass=True)
            self.refinement_stage = SubNetwork(up_ratio=1, stage=3, high_pass=True)

    def forward(self, point_cloud, gt=None):
        point_cloud = point_cloud.float().contiguous()

        p1_pred = self.upsampling_stage_1(point_cloud)

        p2_pred = self.upsampling_stage_2(p1_pred)
        p3_pred = self.refinement_stage(p2_pred)

        p3_pred = p3_pred.permute(0, 2, 1).contiguous()
        p2_pred = p2_pred.permute(0, 2, 1).contiguous()
        p1_pred = p1_pred.permute(0, 2, 1).contiguous()

        return p1_pred, p2_pred, p3_pred


