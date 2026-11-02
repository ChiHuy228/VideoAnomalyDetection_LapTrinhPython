import torch
import torch.utils.data as data
import os
import numpy as np
import utils

class UCF_crime(data.Dataset):
    def __init__(self, root_dir, modal, mode, num_segments, len_feature, seed=-1, is_normal=None):
        if seed >= 0:
            utils.set_seed(seed)
        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        split_path = os.path.join('list', 'UCF_{}.list'.format(self.mode))
        with open(split_path, 'r') as split_file:
            self.vid_list = [line.split() for line in split_file if line.strip()]
        if self.mode == "Train":
            if is_normal is True:
                self.vid_list = self.vid_list[8100:]
            elif is_normal is False:
                self.vid_list = self.vid_list[:8100]
            else:
                print("Please sure is_normal=[True/False]")
                self.vid_list = []

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        if self.mode == "Test":
            return self.get_data(index)
        data_, label = self.get_data(index)[:2]
        return data_, label

    def get_data(self, index):
        vid_info = self.vid_list[index][0]
        name = vid_info.split("/")[-1].split("_x264")[0]
        video_feature = np.load(vid_info).astype(np.float32)

        label = 0 if "Normal" in vid_info.split("/")[-1] else 1
        if self.mode == "Train":
            new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(np.float32)
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype=int)
            for i in range(self.num_segments):
                if r[i] != r[i + 1]:
                    new_feat[i, :] = np.mean(video_feature[r[i]:r[i + 1], :], 0)
                else:
                    new_feat[i:i + 1, :] = video_feature[r[i]:r[i] + 1, :]
            video_feature = new_feat
        return video_feature, label, name