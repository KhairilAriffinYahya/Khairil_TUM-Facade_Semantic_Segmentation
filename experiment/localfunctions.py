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
from tqdm import tqdm
import laspy
import glob
from collections import Counter



def read_las_file_with_labels(file_path):
    las_data = laspy.read(file_path)
    coords = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()
    labels = np.array(las_data.classification, dtype=np.uint8)
    return coords, labels

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def random_point_cloud_crop(points, num_points):
    assert points.shape[0] >= num_points, "Number of points in the point cloud should be greater than or equal to num_points."

    indices = np.random.choice(points.shape[0], num_points, replace=False)
    cropped_points = points[indices]

    return cropped_points


def compute_class_weights(las_dataset):
    # Count the number of points per class
    class_counts = Counter()
    for _, labels in las_dataset:
        class_counts.update(labels)
    # Compute the number of points in the dataset
    num_points = sum(class_counts.values())
    # Compute class weights
    class_weights = {}
    for class_label, count in class_counts.items():
        class_weights[class_label] = num_points / (len(class_counts) * count)
    # Create a list of weights in the same order as class labels
    weight_list = [class_weights[label] for label in sorted(class_weights.keys())]

    return np.array(weight_list, dtype=np.float32)

