import argparse
import os
import torch
import logging
import sys
import importlib
import laspy
import glob
import numpy as np
import open3d as o3d
import pickle
import h5py
import pytz
import matplotlib.pyplot as plt
import time
from geofunction import cal_geofeature
from models.localfunctions import timePrint, CurrentTime, modelTesting
from tqdm import tqdm
from pathlib import Path

'''Adjust permanent/file/static variables here'''

timezone = pytz.timezone('Asia/Singapore')
print("Check current time")
CurrentTime(timezone)
saveTest = "geo_testdata.pkl"
saveDir = "/content/Khairil_PN2_experiment/experiment/data/saved_data/"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 0: wall, # 1: window, # 2: door, # 3: molding, # 4: other, # 5: terrain, # 6: column, # 7: arch
classes = ["wall", "window", "door", "molding", "other", "terrain", "column", "arch"]
NUM_CLASSES = 8
train_ratio = 0.7
dataColor = True #if data lack color set this to False

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
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, default='pointnet2_sem_seg', help='log directory')
    parser.add_argument('--exp_dir', type=str, default='log/sem_seg/', help='Log path [default: None]')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=str, default='cc_o_DEBY_LOD2_4959323.las',
                        help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=5,
                        help='aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_extra_feature_trial',
                        help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--output_model', type=str, default='/best_model.pth', help='model output name')
    parser.add_argument('--rootdir', type=str, default='/content/drive/MyDrive/ data/tum/tum-facade/training/cc_selected/',
                        help='directory to data')
    parser.add_argument('--load', type=bool, default=False, help='load saved data or new')
    parser.add_argument('--save', type=bool, default=False, help='save data')
    parser.add_argument('--visualizeModel', type=str, default=False, help='directory to data')
    parser.add_argument('--downsample', type=bool, default=False, help='downsample data')
    parser.add_argument('--calculate_geometry', type=bool, default=False, help='decide where to calculate geometry')
    parser.add_argument('--geometry_features', type=list, default=['p', 'o', 'c'],
                        help='select which geometry_features to add')

    return parser.parse_args()

class TestCustomDataset():
    # prepare to give prediction on each points
    def __init__(self, root, las_file_list='trainval_fullarea', num_classes=8, block_points=4096, stride=0.5,
                 block_size=1.0, padding=0.001):
        self.block_points = block_points
        self.block_size = block_size
        self.padding = padding
        self.file_list = las_file_list
        self.stride = stride
        self.scene_points_num = []
        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []
        self.labelweights = np.zeros(num_classes)
        self.num_classes = num_classes
        self.num_extra_features = 0
        self.extra_features_data = []
        
        # For Geometric Features
        self.lp_data = []
        self.lo_data = []
        self.lc_data = []
        self.non_index = []

        # Return early if las_file_list is None
        if las_file_list is None:
            self.room_idxs = np.array([])
            return

        adjustedclass = num_classes
        range_class = adjustedclass + 1

        new_class_mapping = {1: 0, 2: 1, 3: 2, 6: 3, 13: 4, 11: 5, 7: 6, 8: 7}

        feature_list=[]
        if dataColor is True:
          feature_list=['red','blue','green']
          for feature in feature_list:
              self.num_extra_features += 1

        if 'p' in args.geometry_features:
            self.num_extra_features += 1
        if 'o' in args.geometry_features:
            self.num_extra_features += 1
        if 'c' in args.geometry_features:
            self.num_extra_features += 1

        for files in self.file_list:
            file_path = os.path.join(root, files)
            # Read LAS file
            print("Reading = " + file_path)
            in_file = laspy.read(file_path)
            points = np.vstack((in_file.x, in_file.y, in_file.z)).T
            labels = np.array(in_file.classification, dtype=np.int32)
            if args.calculate_geometry is False:
                if 'p' in args.geometry_features:
                    tmp_p = np.array(in_file.planarity, dtype=np.float64)
                    self.lp_data.append(tmp_p)
                if 'o' in args.geometry_features:
                    tmp_o = np.array(in_file.Omnivariance, dtype=np.float64)
                    self.lo_data.append(tmp_o)
                if 'c' in args.geometry_features:
                    tmp_c = np.array(in_file.surface_variation, dtype=np.float64)
                    self.lc_data.append(tmp_c)
                    
            # Retrieve color points
            tmp_features = []
            for feature in feature_list:
                # Retrieve the variable with the same name as the feature from `las_data`
                feature_value = getattr(las_data, feature)
                tmp_features.append(feature_value)

            if len(feature_list) > 0:
                self.extra_features_data.append(tmp_features)

            # Merge labels as per instructions
            labels[(labels == 5) | (labels == 6)] = 6  # Merge molding and decoration
            labels[(labels == 1) | (labels == 9) | (labels == 15) | (
                    labels == 10)] = 1  # Merge wall, drainpipe, outer ceiling surface, and stairs
            labels[(labels == 12) | (labels == 11)] = 11  # Merge terrain and ground surface
            labels[(labels == 13) | (labels == 16) | (labels == 17)] = 13  # Merge interior, roof, and other
            labels[labels == 14] = 2  # Add blinds to window

            # Map merged labels to new labels (0 to 7)
            labels = np.vectorize(new_class_mapping.get)(labels)

            data = np.hstack((points, labels.reshape((-1, 1))))
            self.scene_points_list.append(data[:, :3])
            self.semantic_labels_list.append(data[:, 3])
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
        assert len(self.scene_points_list) == len(self.semantic_labels_list)

        labelweights = np.zeros(num_classes)
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(range_class))
            self.scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        points = point_set_ini[:, :3]
        labels = self.semantic_labels_list[index]
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]), np.array([])


        extra_num = self.num_extra_features

        for index_y in range(grid_y):
            for index_x in range(grid_x):
                s_x = coord_min[0] + index_x * self.stride
                e_x = min(s_x + self.block_size, coord_max[0])
                s_x = e_x - self.block_size
                s_y = coord_min[1] + index_y * self.stride
                e_y = min(s_y + self.block_size, coord_max[1])
                s_y = e_y - self.block_size
                point_idxs = np.where((points[:, 0] >= s_x - self.padding) & (points[:, 0] <= e_x + self.padding) &
                                      (points[:, 1] >= s_y - self.padding) & (points[:, 1] <= e_y + self.padding))[0]
                if point_idxs.size == 0:
                    continue

                num_batch = int(np.ceil(point_idxs.size / self.block_points))
                point_size = int(num_batch * self.block_points)
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
                np.random.shuffle(point_idxs)
                data_batch = points[point_idxs, :]
                normlized_xyz = np.zeros((point_size, 3))
                normlized_xyz[:, 0] = data_batch[:, 0] / coord_max[0]
                normlized_xyz[:, 1] = data_batch[:, 1] / coord_max[1]
                normlized_xyz[:, 2] = data_batch[:, 2] / coord_max[2]
                data_batch[:, 0] = data_batch[:, 0] - (s_x + self.block_size / 2.0)
                data_batch[:, 1] = data_batch[:, 1] - (s_y + self.block_size / 2.0)
                data_batch = np.concatenate((data_batch, normlized_xyz), axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]
                
                #Extra features to be included
                
                geo_features = []
                if 'p' in args.geometry_features:  # Load the lp features
                    lp_room = self.lp_data[index]
                    selected_lp = lp_room[point_idxs]  # num_point * lp_features
                    geo_features.append(selected_lp)
                    extra_num -=1
                if 'o' in args.geometry_features:  # Load the lo features
                    lo_room = self.lo_data[index]
                    selected_lo = lo_room[point_idxs]  # num_point * lo_features
                    geo_features.append(selected_lo)
                    extra_num -=1
                if 'c' in args.geometry_features:  # Load the lc features
                    lc_room = self.lc_data[index]
                    selected_lc = lc_room[point_idxs]  # num_point * lc_features
                    geo_features.append(selected_lc)
                    extra_num -=1
                tmp_geo_features = np.array(geo_features).reshape(-1, 1)


                tmp_features = []
                for ix in  range(extra_num):
                    features_room = self.extra_features_data[index] # Load the features
                    features_points = features_room[ix]
                    selected_feature = features_points[point_idxs]  # num_point * lp_features
                    tmp_features.append(selected_feature)
                tmp_np_features = np.array(tmp_features).reshape(-1, 1)
                
                data_batch = np.concatenate((data_batch, tmp_np_features,tmp_geo_features), axis=1)

                #Compile the data extracted
                data_room = np.vstack([data_room, data_batch]) if data_room.size else data_batch
                label_room = np.hstack([label_room, label_batch]) if label_room.size else label_batch
                sample_weight = np.hstack([sample_weight, batch_weight]) if label_room.size else batch_weight
                index_room = np.hstack([index_room, point_idxs]) if index_room.size else point_idxs

        data_room = data_room.reshape((-1, self.block_points, data_room.shape[1]))
        label_room = label_room.reshape((-1, self.block_points))
        sample_weight = sample_weight.reshape((-1, self.block_points))
        index_room = index_room.reshape((-1, self.block_points))

        return data_room, label_room, sample_weight, index_room

    def __len__(self):
        return len(self.scene_points_list)

    def filtered_indices(self):
        total_indices = set(range(len(self.scene_points_list)))
        non_index_set = set(self.non_index)
        filtered_indices = list(total_indices - non_index_set)
        return filtered_indices

    def index_update(self, new_indices):
        tmp_scene_points_num = []
        tmp_scene_points_list = [self.scene_points_list[i] for i in new_indices]
        tmp_semantic_labels_list = [self.semantic_labels_list[i] for i in new_indices]

        self.scene_points_list = tmp_scene_points_list
        self.semantic_labels_list = tmp_semantic_labels_list

        # Recompute labelweights
        num_classes = len(self.labelweights)
        assert num_classes == self.num_classes
        labelweights = np.zeros(num_classes)
        for seg in tmp_semantic_labels_list:
            tmp, _ = np.histogram(seg, range(num_classes + 1))
            tmp_scene_points_num.append(seg.shape[0])
            labelweights += tmp
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        self.scene_points_num = tmp_scene_points_num


    def copy(self, new_indices=None):
        new_dataset = TestCustomDataset()
        new_dataset.block_points = self.block_points
        new_dataset.block_size = self.block_size
        new_dataset.padding = self.padding
        new_dataset.file_list = self.file_list
        new_dataset.stride = self.stride
        new_dataset.num_classes = self.num_classes
        new_dataset.room_coord_min = self.room_coord_min
        new_dataset.room_coord_max = self.room_coord_max
        new_dataset.lp_data = self.lp_data
        new_dataset.lo_data = self.lo_data
        new_dataset.lc_data = self.lc_data
        new_dataset.non_index = self.non_index
        new_dataset.num_extra_features = self.num_extra_features
        new_dataset.extra_features_data = self.extra_features_data

        new_dataset.scene_points_list = [self.scene_points_list[i] for i in new_indices]
        new_dataset.semantic_labels_list = [self.semantic_labels_list[i] for i in new_indices]

        labelweights, tmp_scene_points_num = new_dataset.calculate_labelweights()
        new_dataset.labelweights = labelweights
        new_dataset.scene_points_num = tmp_scene_points_num

        assert len(new_dataset.scene_points_list) == len(new_dataset.semantic_labels_list)

        return new_dataset

    def calculate_labelweights(self):
        print("Calculate Weights")
        num_classes = self.num_classes
        labelweights = np.zeros(num_classes)
        tmp_scene_points_num = []
        for seg in self.semantic_labels_list:
            tmp, _ = np.histogram(seg, range(num_classes + 1))
            tmp_scene_points_num.append(seg.shape[0])
            labelweights += tmp

        print(labelweights)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)  # normalize weights to 1
        labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)  # balance weights

        print(labelweights)
        assert len(labelweights) == num_classes

        return labelweights, tmp_scene_points_num

    def save_data(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_data(file_path):
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)

        print("Extra features to be included = %d" % dataset.num_extra_features)
        print("Number of Classes in dataset = %d" %dataset.num_classes)
        print("Totally {} samples in dataset.".format(len(dataset.room_idxs)))
        return dataset


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''Initialize'''
    root = args.rootdir
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point
    savetest_path = saveDir + saveTest
    test_file = glob.glob(root + args.test_area)
    print("Number of Classes = %d" %NUM_CLASSES)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.exp_dir is None:
        tmp_dir = 'log/sem_seg/'
    else:
        tmp_dir = args.exp_dir
        print(tmp_dir)
    experiment_dir = tmp_dir + args.log_dir
    print("Logging Directory = " + str(experiment_dir))
    visual_dir = experiment_dir + '/visual/'
    print("Visual Directory = " + str(visual_dir))
    visual_dir = Path(visual_dir)
    visual_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''Dataset'''
    testdatatime = time.time()

    print("start loading test data ...")
    if args.load is False:
        TEST_DATASET_WHOLE_SCENE = TestCustomDataset(root, test_file, num_classes=NUM_CLASSES, block_points=NUM_POINT)

        if args.calculate_geometry is True:
            print("room_idx test")
            print(len(TEST_DATASET_WHOLE_SCENE))
            add_geofeature(TEST_DATASET_WHOLE_SCENE, args.downsample, args.visualizeModel)
            print("geometric room_idx test")
            print(len(TEST_DATASET_WHOLE_SCENE))
    else:
        TEST_DATASET_WHOLE_SCENE = TestCustomDataset.load_data(saveDir + saveTest)

    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))
    timePrint(testdatatime)
    CurrentTime(timezone)

    if args.save is True:
        print("Save Test dataset")
        savetesttime = time.time()
        TEST_DATASET_WHOLE_SCENE.save_data(saveDir + saveTest)
        timePrint(savetesttime)
        CurrentTime(timezone)

    '''MODEL LOADING'''
    model_name = args.output_model
    tmp_model = args.model
    if tmp_model == None:
        model_dir = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    else:
        model_dir = tmp_model
    print(model_dir)
    MODEL = importlib.import_module(model_dir)
    num_extra_features = TEST_DATASET_WHOLE_SCENE.num_extra_features
    classifier = MODEL.get_model(NUM_CLASSES, num_extra_features).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints' + model_name)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    num_of_features = 6 + num_extra_features

    '''Model testing'''
    with torch.no_grad():
        print("Begin testing")
        modelTesting(TEST_DATASET_WHOLE_SCENE, NUM_CLASSES, NUM_POINT, BATCH_SIZE, args, timezone,
                     num_of_features, log_string, visual_dir, classifier, seg_label_to_cat)
        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    start = time.time()
    main(args)

    timePrint(start)
    CurrentTime(timezone)