class TrainCustomDataset(Dataset):
    def __init__(self, las_file_list=None, feature_list=[], num_classes=8, num_point=4096, block_size=1.0,
                 sample_rate=1.0,
                 transform=None, indices=None):
        super().__init__()
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        self.num_classes = num_classes
        self.num_extra_features = 0
        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []

        # For Extra Features
        self.extra_features_data = []
        self.non_index = []

        # Return early if las_file_list is None
        if las_file_list is None:
            self.room_idxs = np.array([])
            return

        adjustedclass = num_classes
        range_class = num_classes + 1

        # Use glob to find all .las files in the data_root directory
        las_files = las_file_list
        print(las_file_list)
        rooms = sorted(las_files)
        num_point_all = []
        labelweights = np.zeros(adjustedclass)

        new_class_mapping = {1: 0, 2: 1, 3: 2, 6: 3, 13: 4, 11: 5, 7: 6, 8: 7}

        if dataColor is True:
            feature_list.append("red")
            feature_list.append("blue")
            feature_list.append("green")

        for feature in feature_list:
            self.num_extra_features += 1

        for room_path in rooms:
            # Read LAS file
            print("Reading = " + room_path)
            las_data = laspy.read(room_path)
            coords = np.vstack((las_data.x, las_data.y, las_data.z)).transpose()
            labels = np.array(las_data.classification, dtype=np.uint8)

            tmp_features = []
            for feature in feature_list:
                # Retrieve the variable with the same name as the feature from `las_data`
                feature_value = getattr(las_data, feature)
                tmp_features.append(feature_value)

            if self.num_extra_features > 0:
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

            room_data = np.concatenate((coords, labels[:, np.newaxis]), axis=1)  # xyzl, N*4
            points, labels = room_data[:, 0:3], room_data[:, 3]  # xyz, N*3; l, N
            tmp, _ = np.histogram(labels, range(range_class))
            labelweights += tmp
            coord_min, coord_max = np.amin(points, axis=0), np.amax(points, axis=0)
            self.room_points.append(points)
            self.room_labels.append(labels)
            self.room_coord_min.append(coord_min)
            self.room_coord_max.append(coord_max)
            num_point_all.append(labels.size)

        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []

        for index in range(len(rooms)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)

        if indices is not None:
            self.room_idxs = self.room_idxs[indices]

        assert self.num_extra_features == len(self.extra_features_data)

        print("Extra features to be included = %d" % self.num_extra_features)
        print("Totally {} samples in dataset.".format(len(self.room_idxs)))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]  # N * 6
        labels = self.room_labels[room_idx]  # N
        N_points = points.shape[0]
        extra_num = self.num_extra_features

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1]
                                                                                                     >= block_min[
                                                                                                         1]) & (
                                          points[:, 1] <= block_max[1]))[0]
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

        # Extra feature to be added
        num_of_features = current_points.shape[1]
        current_features = current_points

        ex_features = []
        for ix in range(extra_num):
            features_room = self.extra_features_data[room_idx]
            features_points = features_room[ix]
            selected_feature = features_points[selected_point_idxs]  # num_point * lp_features
            ex_features.append(selected_feature)
            num_of_features += 1

        tmp_np_features = np.zeros((self.num_point, num_of_features))
        tmp_np_features[:, 0:current_points.shape[1]] = current_features
        features_loop = num_of_features - current_points.shape[1]
        for i in range(features_loop):
            col_pointer = i + current_points.shape[1]
            tmp_np_features[:, col_pointer] = ex_features[i]
        current_features = tmp_np_features

        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_features, current_labels = self.transform(current_features, current_labels)

        return current_features, current_labels

    def __len__(self):
        return len(self.room_idxs)

    def calculate_labelweights(self):  # calculate weight of each label/class
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

    def filtered_indices(self):  # get new index list
        total_indices = set(range(len(self.room_points)))
        non_index_set = set(self.non_index)
        filtered_indices = list(total_indices - non_index_set)
        return filtered_indices

    def index_update(self, newIndices):  # adjust index
        self.room_idxs = newIndices

    def copy(self, indices=None):
        # COPY EVERYTHING EXCEPT FOR INDEX
        copied_dataset = TrainCustomDataset()
        copied_dataset.num_point = self.num_point
        copied_dataset.block_size = self.block_size
        copied_dataset.transform = self.transform
        copied_dataset.num_classes = self.num_classes
        copied_dataset.room_points = self.room_points.copy()
        copied_dataset.room_labels = self.room_labels.copy()
        copied_dataset.room_coord_min = self.room_coord_min.copy()
        copied_dataset.room_coord_max = self.room_coord_max.copy()
        copied_dataset.num_extra_features = self.num_extra_features
        copied_dataset.extra_features_data = self.extra_features_data

        # Index to be adjusted
        if indices is not None:
            copied_dataset.room_idxs = self.room_idxs[indices]
        else:
            copied_dataset.room_idxs = self.room_idxs.copy()

        print("Totally {} samples in dataset.".format(len(copied_dataset.room_idxs)))
        return copied_dataset

    def save_data(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_data(file_path):
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)

        print("Extra features to be included = %d" % dataset.num_extra_features)
        print("Number of Classes in dataset = %d" % dataset.num_classes)
        print("Totally {} samples in dataset.".format(len(dataset.room_idxs)))
        return dataset