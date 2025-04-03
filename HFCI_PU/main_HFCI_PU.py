import os
import time
import torch
import shutil
import random
import pathlib
import argparse
import datetime
import numpy as np
from glob import glob
from tqdm import tqdm
from utils import pc_utils
from utils import operations
from utils.model import Model
from utils.data import H5Dataset
from utils.misc import get_logger
from HFCI_PU.model_HFCI_PU import HFCIPU
from pc_eval.eval import pc2_eval_all, pc2_eval_che, make_folder, delete_folder


parser = argparse.ArgumentParser()
parser.add_argument('--phase', default='train')
parser.add_argument('--dataset', type=str, default='PU-GAN', help="ModelNet40 or Sketchfab or PU-GAN or PU1K")
parser.add_argument('--h5_data', default='data/PU_GAN/train/PUGAN_poisson_256_poisson_1024.h5')
parser.add_argument('--gt_dir', default='data/PU_GAN/test/input_2048/gt')
parser.add_argument('--up_test_xyz', default='data/PU_GAN/test/input_2048/input')
parser.add_argument('--mesh_dir', default='data/PU_GAN/test/original_meshes')
parser.add_argument('--up_ratio', type=int, default=4, help='Upsampling Ratio [default: 4]')
parser.add_argument('--log_dir', default='checkpoint')
parser.add_argument('--resume_ckpt', default="", help='test and restore model to restore from')

parser.add_argument('--num_point', type=int, default='256', help='Input patch point number to network [default: 256]')
parser.add_argument('--up_lr_init', type=float, default=0.001)
parser.add_argument('--up_seed', type=int, default=42)
parser.add_argument('--up_patch_num_ratio', type=float, default=3)
parser.add_argument('--up_batch_size', type=int, default=64, help='Batch Size during training')
parser.add_argument('--up_max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--print_step', type=int, default=100, help='print_step during training')
parser.add_argument('--decay_iter', type=int, default=50000)
parser.add_argument('--jitter', action="store_true", help="jitter augmentation")
parser.add_argument('--jitter_sigma', type=float, default=0.005, help="jitter augmentation")
parser.add_argument('--jitter_max', type=float, default=0.01, help="jitter augmentation")
# *******************************************************************************************
# *******************************************************************************************
args = parser.parse_args()


def train():

    experiment_string = "HFCI_PU_{}_{}".format(args.dataset, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    args.log_dir = os.path.join(args.log_dir, experiment_string)
    args.ckpt_up_save_dir = os.path.join(args.log_dir, "ckpt_up")
    args.prd_dir = os.path.join(args.log_dir, "output_xyz")
    args.code_dir = os.path.join(args.log_dir, "code")
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.ckpt_up_save_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.prd_dir).mkdir(parents=True, exist_ok=True)
    shutil.copytree("HFCI_PU", args.code_dir)

    torch.manual_seed(args.up_seed)
    np.random.seed(args.up_seed)
    random.seed(args.up_seed)

    logging = get_logger('train', args.log_dir)

    net = HFCIPU(args)
    net = net.cuda()
    net.train()
    model = Model(net, "train", args)
    dataset = H5Dataset(h5_path=args.h5_data, num_patch_point=args.num_point, phase="train",
                        batch_size=args.up_batch_size, up_ratio=args.up_ratio, jitter=args.jitter)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.up_batch_size, pin_memory=True, num_workers=4, shuffle=True, drop_last=True)
    for epoch in range(1, args.up_max_epoch + 1):
        net.train()

        for i, examples in enumerate(dataloader):
            total_batch = i + (epoch - 1) * len(dataloader)
            input_pc, label_pc, radius = examples
            input_pc = input_pc.cuda()
            label_pc = label_pc.cuda()
            radius = radius.cuda()
            model.set_input(input_pc, radius, label_pc=label_pc)
            loss, lr = model.optimize(total_batch, epoch)
            if i % args.print_step == 0:
                logging.info("epoch: %d, iteration: %d, Lr: %.6f, Loss_1: %.6f, Loss_2: %.6f, Loss_3: %.6f" % (
                    epoch, i, lr, loss[0], loss[1], loss[2]))
        make_folder(args.prd_dir)
        net.eval()
        target_folder = test(args.up_test_xyz,result_dir=args.prd_dir, net=net)
        cd, hd, emd = pc2_eval_che(target_folder, args.gt_dir, args.log_dir)
        logging.info("[test] Iter {}, CD:{}, HD:{}, EMD:{}".format(epoch, cd, hd, emd))
        delete_folder(args.prd_dir)
        path = os.path.join(args.ckpt_up_save_dir, '{:6f}_model_{:d}.pth'.format(cd, epoch))
        torch.save(
            {'net_state_dict': net.state_dict(),
             'optimizer': model.optimizer.state_dict(),
             "epoch": epoch,
             'scheduler': model.lr_scheduler.state_dict()},
            path
        )
        delete_folder(args.prd_dir)


def pc_prediction(net, input_pc, patch_num_ratio=3):
    """
    upsample patches of a point cloud
    :param
        input_pc        1x3xN
        patch_num_ratio int, impacts number of patches and overlapping
    :return
        input_list      list of [3xM]
        up_point_list   list of [3xMr]
    """
    num_patches = int(input_pc.shape[2] / args.num_point * patch_num_ratio)
    idx, seeds = operations.fps_subsample(input_pc, num_patches, NCHW=True)
    input_list = []
    up_point_list = []
    patches, _, _ = operations.group_knn(args.num_point, seeds, input_pc, NCHW=True)  # 1 3 24 256
    patch_time = 0.
    for k in range(num_patches):
        patch = patches[:, :, k, :]  # 1,3,n
        patch, centroid, furthest_distance = operations.normalize_point_batch(
            patch, NCHW=True)

        start = time.time()
        _, _, up_point = net(patch.detach(), True)
        end = time.time()
        patch_time += end - start

        if (up_point.shape[0] != 1):
            up_point = torch.cat(
                torch.split(up_point, 1, dim=0), dim=2)  # (B, C, N) => (1, C, N*B)
            _, up_point = operations.fps_subsample(
                up_point, args.num_point)

        if up_point.size(1) != 3:
            assert (up_point.size(2) == 3), "ChamferLoss is implemented for 3D points"
            up_point = up_point.transpose(2, 1).contiguous()
        up_point = up_point * furthest_distance + centroid
        input_list.append(patch)
        up_point_list.append(up_point)

    return input_list, up_point_list, patch_time / num_patches


def test(up_test_xyz, result_dir, up_ratio=4, net=None, shape_count=2048):
    target_folder = None
    test_files = glob(up_test_xyz + '/*.xyz', recursive=True)
    total_time = 0.
    for point_path in tqdm(test_files):
        target_folder = result_dir
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        out_path = os.path.join(target_folder, point_path.split('/')[-1][:-4] + '.xyz')
        data = pc_utils.load(point_path, shape_count)
        data = data[np.newaxis, ...]
        data, centroid, furthest_distance = pc_utils.normalize_point_cloud(data)

        data = torch.from_numpy(data).transpose(2, 1).cuda().float()

        with torch.no_grad():
            # 1x3xN
            input_pc_list, pred_pc_list, avg_patch_time = pc_prediction(net, data, patch_num_ratio=args.up_patch_num_ratio)

        pred_pc = torch.cat(pred_pc_list, dim=-1)
        total_time += avg_patch_time

        _, pred_pc = operations.fps_subsample(
            pred_pc, shape_count * up_ratio, NCHW=True)
        pred_pc = pred_pc.transpose(2, 1).cpu().numpy()
        pred_pc = (pred_pc * furthest_distance) + centroid
        pred_pc = pred_pc[0, ...]
        np.savetxt(out_path[:-4] + '.xyz', pred_pc, fmt='%.6f')
    print('Average Inference Time: {} ms'.format(round(total_time / len(test_files) * 1000., 3)))
    return target_folder


def test_hfci_pu():
    args.prd_dir = os.path.join(args.log_dir, "output_xyz")
    if args.up_ratio == 16:
        args.up_ratio = 4
        net = HFCIPU(args)
        net = net.cuda()
        net.load_state_dict(torch.load(args.resume_ckpt, map_location='cuda:0')['net_state_dict'])
        net.eval()
        target_folder = test(up_test_xyz=args.up_test_xyz, result_dir=args.prd_dir, net=net)  # 4
        prd_dir_16 = os.path.join(args.log_dir, "output_xyz_16")
        target_folder = test(up_test_xyz=target_folder, result_dir=prd_dir_16, net=net, up_ratio=16, shape_count=8192)  # 16
    else:
        net = HFCIPU(args)
        net = net.cuda()
        net.load_state_dict(torch.load(args.resume_ckpt, map_location='cuda:0')['net_state_dict'])
        net.eval()
        target_folder = test(up_test_xyz=args.up_test_xyz, result_dir=args.prd_dir, net=net)  # 4
    returns = pc2_eval_all(target_folder, args.gt_dir, args.mesh_dir, args.log_dir)
    print(returns)


if __name__ == "__main__":
    if args.phase == "train":
        train()
    elif args.phase == "test_hfci_pu":
        test_hfci_pu()
    else:
        raise "error"



