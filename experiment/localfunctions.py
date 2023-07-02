import os
import torch
import datetime
import provider
import numpy as np
from tqdm import tqdm
import laspy
import time
import pickle
import open3d as o3d
import h5py
import matplotlib.pyplot as plt
import pytz
import logging
import sys
import importlib
import shutil
import glob
from collections import Counter
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)


#classes_8 = ["wall", "window", "door", "molding", "other", "terrain", "column", "arch"]
'''              
                 #eggshell	#FCE6C9	RGB(252,230,201) wall
                 #cornflowerblue	#6495ED	RGB(100,149,237) window
                 #cadmiumorange	#FF6103	RGB(255,97,3) door
                 #teal	#008080	RGB(0,128,128) balcony
                 #blueviolet	#8A2BE2	RGB(138,43,226) molding
                 #cyan2	#00EEEE	RGB(0,238,238) deco
                 #red1	#FF0000	RGB(255,0,0) column
                 #cobalt	#3D59AB	RGB(61,89,171) arch
                 #orange1	#FFA500	RGB(255,165,0) drainpipe
                 #rosybrown	#BC8F8F	RGB(188,143,143) stairs
                 #lawngreen	#7CFC00	RGB(124,252,0) ground surface
                 #mint	#BDFCC9	RGB(189,252,201) terrain
                 #firebrick4	#8B1A1A	RGB(139,26,26) roof
                 #alegreen4	#548B54	RGB(84,139,84) blinds
                 #darkgoldenrod	#B8860B	RGB(184,134,11) outer ceiling surface
                 #yellow1	#FFFF00	RGB(255,255,0) interior
                 #dimgray	#696969	RGB(105,105,105) other
                 
g_class2color = {'total':	                [0,0,0],
                 'wall':	                [0,255,0],
                 'window':	              [0,0,255],
                 'door':                  [0,255,255],
                 'molding':               [255,255,0],
                 'other':	                [255,0,255],
                 'terrain':               [100,100,255], 
                 'column':                [200,200,100],
                 'arch':                  [170,120,200],
                 'balcony':               [255,0,0],
                 'deco':                  [0,0,150],
                 'drainpipe':             [20,70,0],
                 'stairs':                [0,50,70],
                 'ground surface':        [50,50,50],
                 'roof':	                [10,20,10],
                 'blinds':	              [100,255,100],
                 'outer ceiling surface':	[200,100,100],
                 'interior':	            [50,50,50]}       
'''


g_classes = ["total", "wall", "window",  "door",  "balcony","molding", "deco", "column", "arch", "drainpipe", "stairs",
           "ground surface", "terrain",  "roof",  "blinds", "outer ceiling surface", "interior", "other"]
g_class2label = {cls: i for i,cls in enumerate(g_classes)}
g_class2color = {'total':	                [255,255,255],    #papayawhip	#FFEFD5	RGB(255,239,213)
                 'wall':	                [255,240,180],    #eggshell	#FCE6C9	RGB(252,230,201) wall
                 'window':	              [100,149,237],    #cornflowerblue	#6495ED	RGB(100,149,237) window
                 'door':                  [255,97,3],       #cadmiumorange	#FF6103	RGB(255,97,3) door
                 'molding':               [138,43,226],     #blueviolet	#8A2BE2	RGB(138,43,226) molding
                 'other':	                [105,105,105],    #dimgray	#696969	RGB(105,105,105) other
                 'terrain':               [189,252,201],    #mint	#BDFCC9	RGB(189,252,201) terrain
                 'column':                [255,0,0],        #red1	#FF0000	RGB(255,0,0) column
                 'arch':                  [61,89,171],      #cobalt	#3D59AB	RGB(61,89,171) arch
                 'balcony':               [0,128,128],      #teal	#008080	RGB(0,128,128) balcony
                 'deco':                  [0,238,238],      #cyan2	#00EEEE	RGB(0,238,238) deco
                 'drainpipe':             [255,165,0],      #orange1	#FFA500	RGB(255,165,0) drainpipe
                 'stairs':                [188,143,143],    #rosybrown	#BC8F8F	RGB(188,143,143) stairs
                 'ground surface':        [124,252,0],      #lawngreen	#7CFC00	RGB(124,252,0) ground surface
                 'roof':	                [139,26,26],      #firebrick4	#8B1A1A	RGB(139,26,26) roof
                 'blinds':	              [84,139,84],      #alegreen4	#548B54	RGB(84,139,84) blinds
                 'outer ceiling surface':	[184,134,11],     #darkgoldenrod	#B8860B	RGB(184,134,11) outer ceiling surface
                 'interior':	            [255,255,0]}      #yellow1	#FFFF00	RGB(255,255,0) interior
g_label2color = {g_classes.index(cls): g_class2color[cls] for cls in g_classes}

g_colorNames  = {'total':	                'papayawhip',         #papayawhip	#FFEFD5	RGB(255,239,213)
                 'wall':	                'eggshell',           #eggshell	#FCE6C9	RGB(252,230,201) wall
                 'window':	              'cornflowerblue',     #cornflowerblue	#6495ED	RGB(100,149,237) window
                 'door':                  'cadmiumorange',      #cadmiumorange	#FF6103	RGB(255,97,3) door
                 'molding':               'blueviolet',         #blueviolet	#8A2BE2	RGB(138,43,226) molding
                 'other':	                'dimgray',            #dimgray	#696969	RGB(105,105,105) other
                 'terrain':               'mint',               #mint	#BDFCC9	RGB(189,252,201) terrain
                 'column':                'red1',               #red1	#FF0000	RGB(255,0,0) column
                 'arch':                  'cobalt',             #cobalt	#3D59AB	RGB(61,89,171) arch
                 'balcony':               'teal',               #teal	#008080	RGB(0,128,128) balcony
                 'deco':                  'cyan2',              #cyan2	#00EEEE	RGB(0,238,238) deco
                 'drainpipe':             'orange1',            #orange1	#FFA500	RGB(255,165,0) drainpipe
                 'stairs':                'rosybrown',          #rosybrown	#BC8F8F	RGB(188,143,143) stairs
                 'ground surface':        'lawngreen',          #lawngreen	#7CFC00	RGB(124,252,0) ground surface
                 'roof':	                'firebrick4',         #firebrick4	#8B1A1A	RGB(139,26,26) roof
                 'blinds':	              'lawngreen',          #lawngreen	#548B54	RGB(84,139,84) blinds
                 'outer ceiling surface':	'darkgoldenrod',      #darkgoldenrod	#B8860B	RGB(184,134,11) outer ceiling surface
                 'interior':	            'yellow1'}            #yellow1	#FFFF00	RGB(255,255,0) interior


tz = pytz.timezone('Asia/Singapore')

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
    MLChart = []
    IoUChart = []

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
            CurrentTime(tz)
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
            
            tmpAcc = (total_correct / float(total_seen))
            tmpML = (loss_sum / num_batches)
            
            accuracyChart.append(tmpAcc)
            MLChart.append(float(tmpML))
            IoUChart.append(best_iou)
            
        global_epoch += 1

    return accuracyChart, MLChart, IoUChart



'''Testing'''
def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool


def modelTesting(dataset, NUM_CLASSES, NUM_POINT, BATCH_SIZE, args, timezone,
                 num_of_features, log_string, visual_dir, classifier, seg_label_to_cat, resultColor):
    scene_id = dataset.file_list
    scene_id = [x[:-4] for x in scene_id]
    num_batches = len(dataset)

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

        whole_scene_data = dataset.scene_points_list[batch_idx]
        whole_scene_label = dataset.semantic_labels_list[batch_idx]
        vote_label_pool = np.zeros((whole_scene_label.shape[0], NUM_CLASSES))

        for _ in tqdm(range(args.num_votes), total=args.num_votes):
            CurrentTime(timezone)
            scene_data, scene_label, scene_smpw, scene_point_index = dataset[batch_idx]
            num_blocks = scene_data.shape[0]
            s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE
            batch_data = np.zeros((BATCH_SIZE, NUM_POINT, num_of_features))  # Change to number of features being used 6+x

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
            if resultColor is True:

                
                for i in range(whole_scene_label.shape[0]):
                    color = g_label2color[pred_label[i]]
                    color_gt = g_label2color[whole_scene_label[i]]

                    
                    
                    fout.write('v %f %f %f %d %d %d\n' % 
                              (whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], 
                               color[0], color[1],color[2]))
                    fout_gt.write('v %f %f %f %d %d %d\n' % 
                              (whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], 
                               color_gt[0],color_gt[1], color_gt[2]))
                

                
                
            else:
                for i in range(whole_scene_label.shape[0]):
                    fout.write('v %f %f %f\n' % (
                        whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2]))
                    fout_gt.write('v %f %f %f\n' % (
                        whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2]))
          


    IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float64) + 1e-6)
    iou_per_class_str = '------- IoU --------\n'
    for l in range(NUM_CLASSES):
        tmp = float(total_iou_deno_class[l])
        if tmp == 0:
            tmp = 0
        else:
            tmp = total_correct_class[l] / float(total_iou_deno_class[l])
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                                  seg_label_to_cat[l] + ' ' * 
                                  (14 - len(seg_label_to_cat[l])), tmp)
                                  
    # Logging results
    log_string(iou_per_class_str)
    log_string('eval point avg class IoU: %f' % np.mean(IoU))
    log_string('eval whole scene point avg class acc: %f' % (np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float64) + 1e-6))))
    log_string('eval whole scene point accuracy: %f' %      (np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))