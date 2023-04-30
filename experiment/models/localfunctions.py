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
from datetime import datetime
import pytz



def timePrint(start):
    currTime = time.time()
    timetaken = currTime-start
    sec = timetaken%60
    t1 = timetaken/60
    mint = t1%60
    hour = t1/60

    print("Time taken = %i:%i:%i" % (hour, mint, sec))

def CurrentTime(timezone):
    now = datetime.now(timezone)
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)



'''Training'''
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

# Training
def modelTraining(start_epoch, endepoch, alearning_rate, alr_decay, astep_size, BATCH_SIZE, NUM_POINT, NUM_CLASSES,
                  trainDataLoader, testDataLoader, classifier, optimizer, criterion, train_weights, checkpoints_dir,
                  model_name, seg_label_to_cat, logger):

    #Log and print string
    def log_string(str):
        logger.info(str)
        print(str)

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = astep_size
    accuracyChart = []

    global_epoch = 0
    best_iou = 0

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum


    for epoch in range(start_epoch, endepoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, endepoch))
        lr = max(alearning_rate * (alr_decay ** (epoch // astep_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, train_weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
        print("loss value = %f" % loss_sum)
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                # print("Batch shape:", points.shape)  # Debug
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, train_weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                denom = float(total_iou_deno_class[l])

                if denom == 0:
                    tmp = denom
                else:
                    tmp = total_correct_class[l] / float(total_iou_deno_class[l])

                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l]
                    + ' ' * (14 - len(seg_label_to_cat[l])),
                    labelweights[l - 1], tmp)

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

            tmpVal = (total_correct / float(total_seen))
            accuracyChart.append(tmpVal)

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + model_name
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1

    return accuracyChart



'''Testing'''
def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool
