"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import logging
from pathlib import Path
import sys
import importlib
from tqdm import tqdm
import laspy
import glob
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ["wall", "window",  "door",  "molding", "other", "terrain", "column", "arch"]
#classes = ["total", "wall", "window",  "door",  "balcony","molding", "deco", "column", "arch","drainpipe","stairs",  "ground surface",
# "terrain",  "roof",  "blinds", "outer ceiling surface", "interior", "other"]
class2label = {cls: i for i, cls in enumerate(classes)}
NUM_CLASSES = 8
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

print(seg_label_to_cat)


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=4096, help='point number [default: 4096]')
    parser.add_argument('--log_dir', type=str, required=True, help='experiment root')
    parser.add_argument('--exp_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--test_area', type=str, default='DEBY_LOD2_4959323.las', help='area for testing, option: 1-6 [default: 5]')
    parser.add_argument('--num_votes', type=int, default=5, help='aggregate segmentation scores with voting [default: 5]')
    parser.add_argument('--model', type=str, help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--output_model', type=str, default='/best_model.pth', help='model output name')
    parser.add_argument('--rootdir', type=str, default='/content/drive/MyDrive/ data/tum/tum-facade/training/selected/', help='directory to data')
    parser.add_argument('--visualizeModel', type=str, default=False, help='directory to data')

    return parser.parse_args()


def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


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


def collFeatures(pcd, length, size=0.8):
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
    return np.array(normals), np.array(llambda), np.array(lp).reshape(length, -1), np.array(lo).reshape(length,
                                                                                                        -1), np.array(
        lc).reshape(length, -1), np.array(non_idx)


def downsamplingPCD(pcd, dataset):
    # Downsample
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    downsampled_points = np.asarray(downpcd.points)
    downsampled_labels = np.asarray(downpcd.get_point_attr("labels"))

    # Update dataset
    dataset.room_points = [downsampled_points]
    dataset.room_labels = [downsampled_labels]
    dataset.room_idxs = np.array([0])

    # Update pcd after downsampling
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

        # For Geometric Features
        self.eigenNorm = None
        self.llambda = None
        self.lp = None
        self.lo = None
        self.lc = None
        self.non_index = None

        adjustedclass = num_classes
        range_class = adjustedclass + 1

        self.scene_points_list = []
        self.semantic_labels_list = []
        self.room_coord_min, self.room_coord_max = [], []

        new_class_mapping = {1: 0, 2: 1, 3: 2, 6: 3, 13: 4, 11: 5, 7: 6, 8: 7}

        for files in self.file_list:
            file_path = os.path.join(root, files)
            in_file = laspy.read(file_path)
            points = np.vstack((in_file.x, in_file.y, in_file.z)).T
            labels = np.array(in_file.classification, dtype=np.int32)

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
        lp = self.lp  # Load the lp features
        lo = self.lo  # Load the lo features
        lc = self.lc  # Load the lc features
        coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
        grid_x = int(np.ceil(float(coord_max[0] - coord_min[0] - self.block_size) / self.stride) + 1)
        grid_y = int(np.ceil(float(coord_max[1] - coord_min[1] - self.block_size) / self.stride) + 1)
        data_room, label_room, sample_weight, index_room = np.array([]), np.array([]), np.array([]), np.array([])

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
                data_batch = np.concatenate((data_batch, lp[point_idxs, :], lo[point_idxs, :], lc[point_idxs, :]),
                                            axis=1)
                label_batch = labels[point_idxs].astype(int)
                batch_weight = self.labelweights[label_batch]

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
        total_indices = set(range(len(self.room_points)))
        non_index_set = set(self.non_index)
        filtered_indices = list(total_indices - non_index_set)
        return filtered_indices

    def filtered_update(self, filtered_indices):
        self.room_points = [self.room_points[i] for i in filtered_indices]
        self.room_labels = [self.room_labels[i] for i in filtered_indices]
        self.room_coord_min = [self.room_coord_min[i] for i in filtered_indices]
        self.room_coord_max = [self.room_coord_max[i] for i in filtered_indices]

        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(filtered_indices)}
        new_room_idxs = [index_mapping[old_idx] for old_idx in self.room_idxs if old_idx in index_mapping]
        self.room_idxs = np.array(new_room_idxs)

    def save_data(self, file_path):
        with h5py.File(file_path, 'w') as f:
            # Save room points, labels, and additional attributes
            for i, (points, labels) in enumerate(zip(self.room_points, self.room_labels)):
                f.create_dataset(f'room_points/{i}', data=points)
                f.create_dataset(f'room_labels/{i}', data=labels)

            # Save filtered room points and labels
            for i, (points, labels) in enumerate(zip(self.filtered_room_points, self.filtered_room_labels)):
                f.create_dataset(f'filtered_room_points/{i}', data=points)
                f.create_dataset(f'filtered_room_labels/{i}', data=labels)

            # Save additional features
            f.create_dataset('eigenNorm', data=self.eigenNorm)
            f.create_dataset('llambda', data=self.llambda)
            f.create_dataset('lp', data=self.lp)
            f.create_dataset('lo', data=self.lo)
            f.create_dataset('lc', data=self.lc)
            f.create_dataset('non_index', data=self.non_index)

    # @staticmethod
    def load_data(file_path):
        dataset = CustomDataset()  # Initialize with default or placeholder parameters
        with h5py.File(file_path, 'r') as f:
            # Load room points, labels, and additional attributes
            dataset.room_points = [f[f'room_points/{i}'][()] for i in range(len(f['room_points']))]
            dataset.room_labels = [f[f'room_labels/{i}'][()] for i in range(len(f['room_labels']))]

            # Load filtered room points and labels
            dataset.filtered_room_points = [f[f'filtered_room_points/{i}'][()] for i in
                                            range(len(f['filtered_room_points']))]
            dataset.filtered_room_labels = [f[f'filtered_room_labels/{i}'][()] for i in
                                            range(len(f['filtered_room_labels']))]

            # Load additional features
            dataset.eigenNorm = f['eigenNorm'][()]
            dataset.llambda = f['llambda'][()]
            dataset.lp = f['lp'][()]
            dataset.lo = f['lo'][()]
            dataset.lc = f['lc'][()]
            dataset.non_index = f['non_index'][()]

        return dataset


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    root = args.rootdir
    
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.exp_dir is None:
        tmp_dir = 'log/sem_seg/'
    else:
        tmp_dir = args.exp_dir
        print(tmp_dir)
    experiment_dir = tmp_dir + args.log_dir
    visual_dir = experiment_dir + '/visual/'
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

    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    test_file = glob.glob(root + args.test_area )

    print("start loading test data ...")
    TEST_DATASET_WHOLE_SCENE = TestCustomDataset(root, test_file, num_classes=NUM_CLASSES, block_points=NUM_POINT)
    log_string("The number of test data is: %d" % len(TEST_DATASET_WHOLE_SCENE))

    print("room_idx evaluation")
    print(TEST_DATASET_WHOLE_SCENE.room_idxs)
    print(len(TEST_DATASET_WHOLE_SCENE))

    #Open3D
    pcd_test, test_points, test_labels = createPCD(TEST_DATASET_WHOLE_SCENE)

    #Downsampling
    #pcd_test, test_points, test_labels, TRAIN_DATASET = downsamplingPCD(pcd_test, TRAIN_DATASET)
    print("downsampled room_idx evaluation")
    print(TEST_DATASET_WHOLE_SCENE.room_idxs)

    # Visualization
    if args.visualizeModel is True:
        colors = plt.get_cmap("tab20")(np.array(test_labels).reshape(-1) / 17.0)
        colors = colors[:, 0:3]
        pcd_test.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd_test], window_name='test the color', width=800, height=600)

    #Geometric Feature Addition
    # add features, normals, lambda, p, o, c, radius is 0.8m
    test_total_len = len(TEST_DATASET_WHOLE_SCENE)
    eigenNorm, llambda, lp, lo, lc, non_index = collFeatures(pcd_test, test_total_len)

    print("eigenvector len = %" %len(eigenNorm))
    print("non-index = %" %len(non_index))

    # Store the additional features in the CustomDataset instance
    TEST_DATASET_WHOLE_SCENE.eigenNorm = eigenNorm
    TEST_DATASET_WHOLE_SCENE.llambda = llambda
    TEST_DATASET_WHOLE_SCENE.lp = lp
    TEST_DATASET_WHOLE_SCENE.lo = lo
    TEST_DATASET_WHOLE_SCENE.lc = lc
    TEST_DATASET_WHOLE_SCENE.non_index = non_index

    # Filter the points and labels using the non_index variable
    if len(non_index) != 0:
        filtered_indices = TEST_DATASET.filtered_indices()
        TEST_DATASET_WHOLE_SCENE.filtered_update(filtered_indices)

    print("geometric room_idx evaluation")
    print(TEST_DATASET_WHOLE_SCENE.room_idxs)
    print(len(TEST_DATASET_WHOLE_SCENE))

    TestTime = time.time()
    timetaken = TestTime-start
    sec = timetaken%60
    t1 = timetaken/60
    mint = t1%60
    hour = t1/60

    print("Time taken to load test = %i:%i:%i" % (hour, mint, sec))


    '''MODEL LOADING'''
    model_name = args.output_model
    tmp_model = args.model
    if tmp_model == None:
        model_dir = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    else:
        model_dir = tmp_model
    print(model_dir)
    MODEL = importlib.import_module(model_dir)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints'+model_name)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        scene_id = TEST_DATASET_WHOLE_SCENE.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(TEST_DATASET_WHOLE_SCENE)

        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]

        log_string('---- EVALUATION WHOLE SCENE----')

        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            total_seen_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_correct_class_tmp = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class_tmp = [0 for _ in range(NUM_CLASSES)]
            if args.visual:
                fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
                fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')
            whole_scene_data = TEST_DATASET_WHOLE_SCENE.scene_points_list[batch_idx]
            whole_scene_label = TEST_DATASET_WHOLE_SCENE.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))
            for _ in tqdm(range(args.num_votes), total=args.num_votes):
                scene_data, scene_label, scene_smpw, scene_point_index = TEST_DATASET_WHOLE_SCENE[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
                batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 6))  # Change to 6 (from 9) as there's no color

                batch_label = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_point_index = np.zeros((BATCH_SIZE, NUM_POINT))
                batch_smpw = np.zeros((BATCH_SIZE, NUM_POINT))

                for sbatch in range(s_batch_num):
                    start_idx = sbatch * BATCH_SIZE
                    end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
                    real_batch_size = end_idx - start_idx
                    batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                    batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                    batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                    batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]

                    torch_data = torch.Tensor(batch_data)
                    torch_data = torch_data.float().cuda()
                    torch_data = torch_data.transpose(2, 1)
                    seg_pred, _ = classifier(torch_data)
                    batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                    vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                               batch_pred_label[0:real_batch_size, ...],
                                               batch_smpw[0:real_batch_size, ...])

            pred_label = np.argmax(vote_label_pool, 1)

            for l in range(NUM_CLASSES):
                total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
                total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=float) + 1e-6)
            print(iou_map)
            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
            print('----------------------------')

            filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
            with open(filename, 'w') as pl_save:
                for i in pred_label:
                    pl_save.write(str(int(i)) + '\n')
                pl_save.close()

            if args.visual:
                for i in range(whole_scene_label.shape[0]):
                    fout.write('v %f %f %f\n' % (
                        whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2]))
                    fout_gt.write('v %f %f %f\n' % (
                        whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2]))

            if args.visual:
                fout.close()
                fout_gt.close()

        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        for l in range(NUM_CLASSES):
            tmp = float(total_iou_deno_class[l])

            if tmp == 0:
                tmp = 0
            else:
                tmp = total_correct_class[l] / float(total_iou_deno_class[l])


            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),tmp )


        log_string(iou_per_class_str)
        log_string('eval point avg class IoU: %f' % np.mean(IoU))
        log_string('eval whole scene point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))
        log_string('eval whole scene point accuracy: %f' % (
                np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

        print("Done!")


if __name__ == '__main__':
    args = parse_args()
    main(args)