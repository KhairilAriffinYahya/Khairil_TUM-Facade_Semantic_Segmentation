import argparse
import os
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
import numpy as np
import laspy
import glob
import matplotlib.pyplot as plt
import time
import pickle
import pytz
import open3d as o3d
import h5py
import provider
from torch.utils.data import Dataset, DataLoader, random_split
from geofunction import cal_geofeature
from models.localfunctions import timePrint, CurrentTime, inplace_relu, modelTraining
from tqdm import tqdm
from collections import Counter



if __name__ == '__main__':
    tmp = [1,2,3,4,5]
    os.environ["let_see"] = tmp