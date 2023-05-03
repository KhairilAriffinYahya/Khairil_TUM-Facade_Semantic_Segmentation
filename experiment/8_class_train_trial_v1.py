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
import pickle
import open3d as o3d
import h5py
from models.localfunctions import timePrint, CurrentTime, inplace_relu, modelTraining
import pytz

'''Adjust permanent/file/static variables here'''

timezone = pytz.timezone('Asia/Singapore')
print("Check current time")
CurrentTime(timezone)
saveTrain = "8cla_traindata.pkl"
saveEval = "8cla_evaldata.pkl"
saveDir = "/content/Khairil_PN2_experiment/experiment/data/saved_data/"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 0: wall, # 1: window, # 2: door, # 3: molding, # 4: other, # 5: terrain, # 6: column, # 7: arch
classes = ["wall", "window",  "door",  "molding", "other", "terrain", "column", "arch"]
NUM_CLASSES = 8
train_ratio = 0.7

''''''

sys.path.append(os.path.join(BASE_DIR, 'models'))
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

print(seg_label_to_cat)

# Adjust parameters here if there no changes to reduce line
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_trial', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]') #maybe static
    parser.add_argument('--log_dir', type=str, default='pointnet2_sem_seg', help='Log path [default: None]')
    parser.add_argument('--exp_dir', type=str, default='./log/', help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=str, default='DEBY_LOD2_4959323.las', help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--output_model', type=str, default='/best_model.pth', help='model output name')
    parser.add_argument('--rootdir', type=str, default='/content/drive/MyDrive/ data/tum/tum-facade/training/selected/', help='directory to data')
    parser.add_argument('--load', type=bool, default=False, help='load saved data or new')
    parser.add_argument('--save', type=bool, default=False, help='save data')


    return parser.parse_args()


class TrainCustomDataset(Dataset):
    def __init__(self, las_file_list=None, num_classes=8, num_point=4096, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.num_classes = num_classes
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        self.room_idxs = np.array([])

        # Return early if las_file_list is None
        if las_file_list is None:
            return

        adjustedclass = num_classes
        range_class = num_classes+1

        # Use glob to find all .las files in the data_root directory
        las_files = las_file_list
        print(las_file_list)
        rooms = sorted(las_files)
        num_point_all = []
        labelweights = np.zeros(adjustedclass)

        new_class_mapping = {1: 0, 2: 1, 3:2, 6: 3, 13: 4, 11: 5, 7: 6, 8: 7}

        for room_path in rooms:
            # Read LAS file
            print("Reading = " + room_path)
            las_data = laspy.read(room_path)
            coords = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()
            labels = np.array(las_data.classification, dtype=np.uint8)

            # Merge labels as per instructions
            labels[(labels == 5) | (labels == 6)] = 6  # Merge molding and decoration
            labels[(labels == 1) |(labels == 9) | (labels == 15) | (labels == 10)] = 1  # Merge wall, drainpipe, outer ceiling surface, and stairs
            labels[(labels == 12) | (labels == 11)] = 11  # Merge terrain and ground surface
            labels[(labels == 13) | (labels == 16) | (labels == 17)] = 13  # Merge interior, roof, and other
            labels[labels == 14] = 2  # Add blinds to window

            # Map merged labels to new labels (0 to 7)
            labels = np.vectorize(new_class_mapping.get)(labels)

            room_data = np.concatenate((coords, labels[:, np.newaxis]), axis=1)  # xyzl, N*4
            points, labels = room_data[:, 0:3], room_data[:, 3]  # xyz, N*3; l, N
            tmp, _ = np.histogram(labels, range(range_class))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0), np.amax(points, axis=0)
            self.room_points.append(points), self.room_labels.append(labels)
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)

        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []

        for index in range(len(rooms)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)

        print("Totally {} samples in dataset.".format(len(self.room_idxs)))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & 
                              (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 6))  # num_point * 6
        current_points[:, 3] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 4] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 5] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        current_points[:, 0:3] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)

        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)

    def calculate_labelweights(self):
        print("Calculate Weights")
        labelweights = np.zeros(self.num_classes)
        for labels in self.room_labels:
            tmp, _ = np.histogram(labels, range(self.num_classes + 1))
            labelweights += tmp

        print(labelweights)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)  # normalize weights to 1
        labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)  # balance weights

        print(labelweights)

        return labelweights

    def copy(self, indices=None):
        copied_dataset = TrainCustomDataset()
        copied_dataset.num_point = self.num_point
        copied_dataset.block_size = self.block_size
        copied_dataset.transform = self.transform
        copied_dataset.room_points = self.room_points.copy()
        copied_dataset.room_labels = self.room_labels.copy()
        copied_dataset.room_coord_min = self.room_coord_min.copy()
        copied_dataset.room_coord_max = self.room_coord_max.copy()

        if indices is not None:
            copied_dataset.room_idxs = self.room_idxs[indices]
        else:
            copied_dataset.room_idxs = self.room_idxs.copy()

        print("Totally {} samples in dataset.".format(len(copied_dataset.room_idxs)))
        return copied_dataset

    def index_update(self, newIndices):
        self.room_idxs = newIndices

    def save_data(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_data(file_path):
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset


def main(args):
    #Log and print string
    def log_string(str):
        logger.info(str)
        print(str)


    '''Initialize Variables'''
    root = args.rootdir
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size
    savetrain_path = saveDir+saveTrain
    saveeval_path = saveDir+saveEval
    las_file_list = [file for file in glob.glob(root + '/*.las') if not file.endswith(args.test_area )]


    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    if args.exp_dir is None:
        experiment_dir = Path('./log/')
    else:
        experiment_dir = Path(args.exp_dir)
        print(experiment_dir)
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    print("Logging Directory = " +str(experiment_dir))
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    print("Logs Directory = " +str(log_dir))
    
    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''Load Dataset'''
    if args.load is False:
        datasetTime = time.time()
        lidar_dataset = TrainCustomDataset(las_file_list, num_classes=NUM_CLASSES, num_point=NUM_POINT, transform=None)
        print("Dataset taken")

        # Split the dataset into training and evaluation sets
        train_size = int(train_ratio * len(lidar_dataset))
        eval_size = len(lidar_dataset) - train_size
        train_indices, eval_indices = random_split(range(len(lidar_dataset)), [train_size, eval_size])

        print("start loading training data ...")
        TRAIN_DATASET = lidar_dataset.copy(indices=train_indices)

        print("start loading eval data ...")
        EVAL_DATASET = lidar_dataset.copy(indices=eval_indices)

        timePrint(datasetTime)
        CurrentTime(timezone)
    else:
        print("Load previously saved dataset")
        loadtime=time.time()
        TRAIN_DATASET=TrainCustomDataset.load_data(saveDir+saveTrain)
        print("Total {} samples in training dataset.".format(len(TRAIN_DATASET)))
        EVAL_DATASET=TrainCustomDataset.load_data(saveDir+saveEval)
        print("Total {} samples in evaluation dataset.".format(len(EVAL_DATASET)))
        timePrint(loadtime)
        CurrentTime(timezone)

    if args.save is True:
        print("Save Dataset")
        savetime=time.time()
        TRAIN_DATASET.save_data(saveDir+saveTrain)
        EVAL_DATASET.save_data(saveDir + saveEval)
        timePrint(savetime)
        CurrentTime(timezone)

    trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    evalDataLoader = DataLoader(EVAL_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True)

    train_labelweights = TRAIN_DATASET.calculate_labelweights()

    print("wall", "window", "door", "molding", "other", "terrain", "column", "arch")
    train_weights = torch.Tensor(train_labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of eval data is: %d" % len(EVAL_DATASET))

    print("Length of the training dataset:", len(TRAIN_DATASET))
    print("Length of the trainDataLoader:", len(trainDataLoader))

    print("Length of the evaluation dataset:", len(EVAL_DATASET))
    print("Length of the evalDataLoader:", len(evalDataLoader))
    
    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)
    model_name = args.output_model


    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints'+model_name)
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('Linear') != -1:
                torch.nn.init.xavier_normal_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)

        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    '''Training Model'''

    print("Identified Weights")
    print(train_weights)

    print("Data Preparation Complete")
    timePrint(start)
    CurrentTime(timezone)

    accuracyChart = modelTraining(start_epoch, args.epoch, args.learning_rate, args.lr_decay, args.step_size, BATCH_SIZE,
                                  NUM_POINT, NUM_CLASSES,trainDataLoader, evalDataLoader, classifier, optimizer, criterion,
                                  train_weights, checkpoints_dir, model_name, seg_label_to_cat, logger)

    return accuracyChart




if __name__ == '__main__':
    args = parse_args()
    start = time.time()
    accuracyChart = main(args)

    max_value = max(accuracyChart)
    max_index = accuracyChart.index(max_value)

    print(max_index)
    print("Training Complete")
    timePrint(start)
    CurrentTime(timezone)