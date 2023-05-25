import argparse
import os
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
import provider
import numpy as np
import time
from tqdm import tqdm
import laspy
import glob
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import time
import open3d as o3d
import pickle

def PCA(data, correlation=False, sort=True):
    average_data = np.mean(data, axis=0)  # 求 NX3 向量的均值
    decentration_matrix = data - average_data  # 去中心化
    H = np.dot(decentration_matrix.T, decentration_matrix)  # 求解协方差矩阵 H
    eigenvectors, eigenvalues, eigenvectors_T = np.linalg.svd(H)  # SVD求解特征值、特征向量
    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]  # 降序排列
        eigenvalues = eigenvalues[sort]  # 索引
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def collFeatures(pcd, length, size=0.8, num_of_files=1):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)  # set a kd tree for tha point cloud, make searching faster
    normals = []
    llambda = []
    lp = []
    lo = []
    lc = []
    non_idx = []
    # print(point_cloud_o3d)  #geometry::PointCloud with 10000 points.
    print(length)  # 10000
    for i in range(length):
        # search_knn_vector_3d， input[point，x]      returns [int, open3d.utility.IntVector, open3d.utility.DoubleVector]
        [_, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], size)
        # asarray is the same as array  but asarray will save the memeory
        k_nearest_point = np.asarray(pcd.points)[idx,
                          :]  # find the surrounding points for each point, set them as a curve and use PCA to find the normal
        lamb, v = PCA(k_nearest_point)
        if len(k_nearest_point) == 1:
            non_idx.append(i)  # record the index that has no knn point
            p = 0
            o = 0
            c = 0
        else:
            p = (lamb[1] - lamb[2]) / lamb[0]  # calculate features based on eigenvalues
            o = pow(lamb[0] * lamb[1] * lamb[2], 1.0 / 3.0)
            c = lamb[2] / sum(lamb)
        normals.append(v[:, 1])
        llambda.append(lamb)
        lp.append(p)
        lo.append(o)
        lc.append(c)

    normals_array = np.array(normals) 
    llambda_array = np.array(llambda)  # Convert the list of eigenvalues to a NumPy array
    lp_array = np.array(lp).reshape(length, -1)  # Convert the list of lp features to a NumPy array
    lo_array = np.array(lo).reshape(length, -1)  # Convert the list of lo features to a NumPy array
    lc_array = np.array(lc).reshape(length, -1) 

    lp_split = np.split(lp_array, num_of_files, axis=1)
    lo_split = np.split(lo_array, num_of_files, axis=1)
    lc_split = np.split(lc_array, num_of_files, axis=1)
    llambda_split = np.split(llambda_array, num_of_files, axis=1)
    normals_split = np.split(normals_array, num_of_files, axis=1)

    print(lp_split)
    print("shape of lp: ", lp_split.shape)
    for i, column in enumerate(lp_split, start=1):
        print(f"Column {i}:\n", column)
        
    return normals_array.tolist(), llambda_array.tolist(), lp_split.tolist(), lo_array.tolist(), lc_array.tolist(), non_idx


def downsamplingPCD(pcd, dataset):
    #Downsample
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    downsampled_points = np.asarray(downpcd.points)
    downsampled_labels = np.asarray(downpcd.get_point_attr("labels"))

    #Update dataset
    dataset.room_points = [downsampled_points]
    dataset.room_labels = [downsampled_labels]
    dataset.room_idxs = np.array([0])

    #Update pcd after downsampling
    all_points = np.vstack(dataset.room_points)
    all_labels = np.concatenate(dataset.room_labels)
    pcd_update = o3d.geometry.PointCloud(o3ddevice)
    pcd_update.points = o3d.utility.Vector3dVector(all_points)

    return pcd_update, all_points, all_labels, dataset

def createPCD(dataset):
    # Concatenate room_points and room_labels from all rooms
    all_points = np.vstack(dataset.room_points)
    all_labels = np.concatenate(dataset.room_labels)

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    # Set the point positions using all_points
    pcd.points = o3d.utility.Vector3dVector(all_points)

    # Set the point labels using all_labels
    all_labels_np = all_labels.reshape(-1, 1)  # Reshape the labels to have shape (N, 1)

    # Create a tensor for point labels and assign it to the custom attribute 'labels'
    labels_tensor = o3d.core.Tensor(all_labels_np, dtype=o3d.core.Dtype.Int32)
    pcd.set_point_attr("labels", labels_tensor)

    return pcd, all_points, all_labels

def add_geofeature(dataset, dwnsample, visualize):
    # Open3D
    pcd, points, labels = createPCD(dataset)

    # Downsampling
    if dwnsample is True:
        pcd, points, labels, dataset = downsamplingPCD(pcd, dataset)
        print("downsampled room_idx")
        print(len(dataset))

    # Visualization
    if  visualize is True:
        colors = plt.get_cmap("tab20")(np.array(labels).reshape(-1) / 17.0)
        colors = colors[:, 0:3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], window_name='test the color', width=800, height=600)

    # Geometric Feature Addition
    # add features, normals, lambda, p, o, c, radius is 0.8m
    dataset_total_len = len(dataset)
    eigenNorm, llambda, lp, lo, lc, non_index = collFeatures(pcd, dataset_total_len)

    print("eigenvector len = %" % len(eigenNorm))
    print("non-index = %" % len(non_index))

    # Store the additional features in the CustomDataset instance
    dataset.lp_data = lp
    dataset.lo_data = lo
    dataset.lc_data = lc
    dataset.non_index = non_index

    # Filter the points and labels using the non_index variable
    if len(non_index) != 0:
        filtered_indices = dataset.filtered_indices()
        dataset.filtered_update(filtered_indices)


def cal_geofeature(dataset, dwnsample, visualize):
    # Open3D
    pcd, points, labels = createPCD(dataset)

    # Downsampling
    if dwnsample is True:
        pcd, points, labels, dataset = downsamplingPCD(pcd, dataset)
        print("downsampled room_idx")
        print(len(dataset))

    # Visualization
    if  visualize is True:
        colors = plt.get_cmap("tab20")(np.array(labels).reshape(-1) / 17.0)
        colors = colors[:, 0:3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd], window_name='test the color', width=800, height=600)

    # Geometric Feature Addition
    # add features, normals, lambda, p, o, c, radius is 0.8m
    dataset_total_len = len(dataset)
    eigenNorm, llambda, lp, lo, lc, non_index = collFeatures(pcd, dataset_total_len)

    print("eigenvector len = %" % len(eigenNorm))
    print("non-index = %" % len(non_index))

    return lp, lo, lc, non_index
