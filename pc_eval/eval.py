import argparse
import csv
import os
import subprocess
from collections import OrderedDict
from glob import glob
from time import time
import numpy as np
import torch
from tqdm import tqdm

from pc_eval.common.loss import Loss
from pc_eval.utils.pc_util import load, normalize_point_cloud


def compute_p2f(prd_xyz, mesh_off):
    if __name__ == "__main__":
        commands = ["../pc_eval/evaluate_code/evaluate", mesh_off, prd_xyz]
    else:
        commands = ["pc_eval/evaluate_code/evaluate", mesh_off, prd_xyz]
    subprocess.run(commands, stdout=open(os.devnull, "w"))  # 将子进程的标准输出丢弃


def make_folder(folder_dir):
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)


def delete_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)
    os.rmdir(folder_path)


def pc2_eval_all(prd_dir, gt_dir, mesh_dir, csv_dir):
    returns = []
    loss = Loss()

    gt_paths = glob(os.path.join(gt_dir, "*.xyz"))
    gt_names = [os.path.basename(p) for p in gt_paths]

    # 生成 point2mesh_distance.txt文件
    for name in gt_names:
        prd_path = os.path.join(prd_dir, name)
        mesh_path = os.path.join(mesh_dir, name[:-3] + "off")
        if os.path.isfile(prd_path[:-4] + "_point2mesh_distance.txt"):  # 字符串切块生成新文件
            print("P2F already computed")
            break
        else:
            compute_p2f(prd_path, mesh_path)

    # 计算CD HD 生成csv文件
    make_folder(csv_dir)

    avg_cd_value = 0
    avg_hd_value = 0
    avg_emd_value = 0
    global_p2f = []
    counter = len(gt_paths)
    fieldnames = ["name", "CD", "HD", "p2f avg", "p2f std", "EMD"]

    csv_file = os.path.join(csv_dir, "eval.csv")
    with open(csv_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="-", extrasaction="ignore")
        writer.writeheader()
        # 计算每一个点云对象
        for name in tqdm(gt_names):
            row = OrderedDict()
            prd_xyz = load(os.path.join(prd_dir, name))[:, :3]
            gt_xyz = load(os.path.join(gt_dir, name))[:, :3]

            prd_xyz, _, _ = normalize_point_cloud(prd_xyz)
            gt_xyz, _, _ = normalize_point_cloud(gt_xyz)

            prd_xyz = torch.from_numpy(prd_xyz).unsqueeze(0).contiguous().cuda()
            gt_xyz = torch.from_numpy(gt_xyz).unsqueeze(0).contiguous().cuda()  # 1 8192 3

            cd_loss = loss.get_cd_loss(prd_xyz, gt_xyz).cpu().item()
            hd_loss = loss.get_hd_loss(prd_xyz, gt_xyz).cpu().item()
            emd_loss = loss.get_emd_loss(prd_xyz, gt_xyz).cpu().item()

            avg_cd_value += cd_loss
            avg_hd_value += hd_loss
            avg_emd_value += emd_loss

            row["name"] = name[:-4]
            row["CD"] = "{:.9f}".format(cd_loss)
            row["HD"] = "{:.9f}".format(hd_loss)
            row["EMD"] = "{:.9f}".format(emd_loss)
            if os.path.isfile(os.path.join(prd_dir, name)[:-4] + "_point2mesh_distance.txt"):
                point2mesh_distance = load(os.path.join(prd_dir, name)[:-4] + "_point2mesh_distance.txt")
                if point2mesh_distance.size == 0:
                    continue
                point2mesh_distance = point2mesh_distance[:, 3]
                row["p2f avg"] = "{:.9f}".format(np.nanmean(point2mesh_distance))  # 计算每个对象的P2F均值和方差
                row["p2f std"] = "{:.9f}".format(np.nanstd(point2mesh_distance))
                global_p2f.append(point2mesh_distance)
            writer.writerow(row)

        # 计算数据集上的平均指标
        row = OrderedDict()
        rounded_row = OrderedDict()

        avg_cd_value /= counter
        avg_hd_value /= counter
        avg_emd_value /= counter
        row["name"] = "average"
        row["CD"] = "{:.9f}".format(avg_cd_value)
        row["HD"] = "{:.9f}".format(avg_hd_value)
        row["EMD"] = "{:.9f}".format(avg_emd_value)

        rounded_row["name"] = "rounded_average"
        rounded_row["CD"] = round(avg_cd_value * 1e3, 3)
        rounded_row["HD"] = round(avg_hd_value * 1e3, 3)
        rounded_row["EMD"] = round(avg_emd_value * 1e2, 3)

        returns.append(round(avg_cd_value * 1e3, 3))
        returns.append(round(avg_hd_value * 1e3, 3))
        returns.append(round(avg_emd_value * 1e2, 3))

        if global_p2f:
            global_p2f = np.concatenate(global_p2f, axis=0)  # concat后计算整个数据集所有对象的点的P2F的均值和方差
            mean_p2f = np.nanmean(global_p2f)
            std_p2f = np.nanstd(global_p2f)
            row["p2f avg"] = "{:.9f}".format(mean_p2f)
            row["p2f std"] = "{:.9f}".format(std_p2f)

            rounded_row["p2f avg"] = round(mean_p2f * 1e3, 3)  # *1000
            rounded_row["p2f std"] = round(std_p2f * 1e3, 3)

            returns.append(round(mean_p2f * 1e3, 3))
            returns.append(round(std_p2f * 1e3, 3))

        writer.writerow(row)
        writer.writerow(rounded_row)
        with open(os.path.join(csv_dir, "finalresult.text"), "w") as text:
            text.write(str(row))
            text.write(str("\n"))
            text.write(str(rounded_row))
    return returns


def pc2_eval_che(prd_dir, gt_dir, csv_dir):
    returns = []
    loss = Loss()

    gt_paths = glob(os.path.join(gt_dir, "*.xyz"))
    gt_names = [os.path.basename(p) for p in gt_paths]

    # 计算CD HD 生成csv文件
    make_folder(csv_dir)

    avg_cd_value = 0
    avg_hd_value = 0
    avg_emd_value = 0
    counter = len(gt_paths)
    fieldnames = ["name", "CD", "HD", "EMD"]

    csv_file = os.path.join(csv_dir, "eval.csv")
    with open(csv_file, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="-", extrasaction="ignore")
        writer.writeheader()
        # 计算每一个点云对象
        for name in gt_names:
            row = OrderedDict()
            prd_xyz = load(os.path.join(prd_dir, name))[:, :3]
            gt_xyz = load(os.path.join(gt_dir, name))[:, :3]

            prd_xyz, _, _ = normalize_point_cloud(prd_xyz)
            gt_xyz, _, _ = normalize_point_cloud(gt_xyz)

            prd_xyz = torch.from_numpy(prd_xyz).unsqueeze(0).contiguous().cuda()
            gt_xyz = torch.from_numpy(gt_xyz).unsqueeze(0).contiguous().cuda()  # 1 8192 3

            cd_loss = loss.get_cd_loss(prd_xyz, gt_xyz).cpu().item()
            hd_loss = loss.get_hd_loss(prd_xyz, gt_xyz).cpu().item()
            emd_loss = loss.get_emd_loss(prd_xyz, gt_xyz).cpu().item()

            avg_cd_value += cd_loss
            avg_hd_value += hd_loss
            avg_emd_value += emd_loss

            row["name"] = name[:-4]
            row["CD"] = "{:.9f}".format(cd_loss)
            row["HD"] = "{:.9f}".format(hd_loss)
            row["EMD"] = "{:.9f}".format(emd_loss)
            writer.writerow(row)

        # 计算数据集上的平均指标
        row = OrderedDict()
        rounded_row = OrderedDict()

        avg_cd_value /= counter
        avg_hd_value /= counter
        avg_emd_value /= counter
        row["name"] = "average"
        row["CD"] = "{:.9f}".format(avg_cd_value)
        row["HD"] = "{:.9f}".format(avg_hd_value)
        row["EMD"] = "{:.9f}".format(avg_emd_value)

        rounded_row["name"] = "rounded_average"
        rounded_row["CD"] = round(avg_cd_value * 1e3, 3)
        rounded_row["HD"] = round(avg_hd_value * 1e3, 3)
        rounded_row["EMD"] = round(avg_emd_value * 1e2, 3)

        returns.append(round(avg_cd_value * 1e3, 3))
        returns.append(round(avg_hd_value * 1e3, 3))
        returns.append(round(avg_emd_value * 1e2, 3))

        writer.writerow(row)
        writer.writerow(rounded_row)
        with open(os.path.join(csv_dir, "finalresult.text"), "w") as text:
            text.write(str(row))
            text.write(str("\n"))
            text.write(str(rounded_row))
    return returns
