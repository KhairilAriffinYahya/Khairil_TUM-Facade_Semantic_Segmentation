with torch.no_grad():
    print("Begin testing")
    modelTesting(TEST_DATASET_WHOLE_SCENE, NUM_CLASSES, NUM_POINT, BATCH_SIZE, args, timezone,
                 num_of_features, log_string, visual_dir, classifier, seg_label_to_cat)
    print("Done!")

def modelTesting(dataset, NUM_CLASSES, NUM_POINT, BATCH_SIZE, args, timezone,
                 num_of_features, log_string, visual_dir, classifier, seg_label_to_cat):

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
            for i in range(whole_scene_label.shape[0]):
                fout.write('v %f %f %f\n' % (
                    whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2]))
                fout_gt.write('v %f %f %f\n' % (
                    whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2]))

        if args.visual:
            fout.close()
            fout_gt.close()

        CurrentTime(timezone)

    IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6)
    iou_per_class_str = '------- IoU --------\n'
    for l in range(NUM_CLASSES):
        tmp = float(total_iou_deno_class[l])
        if tmp == 0:
            tmp = 0
        else:
            tmp = total_correct_class[l] / float(total_iou_deno_class[l])
        iou_per_class_str += 'class %s, IoU: %.3f \n' % (
            seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), tmp)

    # Logging results
    log_string(iou_per_class_str)
    log_string('eval point avg class IoU: %f' % np.mean(IoU))
    log_string('eval whole scene point avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))))
    log_string('eval whole scene point accuracy: %f' % (
            np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))

    print("Done!")