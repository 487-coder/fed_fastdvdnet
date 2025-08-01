
import os
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import imageio.v3 as iio
from utils import open_sequence
from torch.utils.data import Dataset

NUMFRXSEQ_VAL = 15
VALSEQPATT = '*'


def partition_data(user_num, dataset_path,alpha=0.5, distribution="iid", noise_level_range=(0.1, 1.0),
                   num_noise_classes=5):
    """
    实现在noise_level维度上的IID和Non-IID数据分区

    Args:
        user_num: 用户数量
        alpha: Dirichlet分布参数（仅在non-iid时使用）
        distribution: "iid" 或 "non_iid"
        dataset_path: 包含mp4文件的数据集文件夹路径
        noise_level_range: noise level的范围 (min, max)
        num_noise_classes: 将noise level离散化为多少个类别

    Returns:
        dict: {user_id: [(video_path, noise_level), ...]}
    """

    # 获取所有mp4文件
    video_files = []
    for file in os.listdir(dataset_path):
        if file.endswith('.mp4'):
            video_files.append(os.path.join(dataset_path, file))

    if not video_files:
        raise ValueError("数据集文件夹中未找到mp4文件")
    print(len(video_files))

    # 为每个视频分配noise level并分类
    video_data = []
    noise_class_to_videos = defaultdict(list)

    for video_path in video_files:
        # 随机分配noise level
        noise_level = random.uniform(noise_level_range[0], noise_level_range[1])
        # 将连续的noise_level离散化为类别
        noise_class = min(int((noise_level - noise_level_range[0]) /
                              (noise_level_range[1] - noise_level_range[0]) * num_noise_classes),
                          num_noise_classes - 1)

        video_item = (video_path, noise_level)
        video_data.append(video_item)
        noise_class_to_videos[noise_class].append(video_item)

    if distribution == "iid":
        return _partition_iid(user_num, video_data)
    elif distribution == "non_iid":
        return _partition_non_iid(user_num, alpha, noise_class_to_videos, num_noise_classes)
    else:
        raise ValueError(f"不支持的分布类型: {distribution}")
    '''
    {
    0: [(video_path_1, noise_level_1), (video_path_2, noise_level_2), ...],
    1: [(video_path_a, noise_level_a), ...],
    ...
    user_num-1: [...]
}

    '''


def _partition_iid(user_num, video_data):
    """
    IID分区：每个用户获得所有noise level的均匀分布
    """
    # 随机打乱所有数据
    random.shuffle(video_data)

    # 均匀分配给各个用户
    videos_per_user = len(video_data) // user_num
    user_data = {}

    for user_id in range(user_num):
        start_idx = user_id * videos_per_user
        if user_id == user_num - 1:  # 最后一个用户获得剩余所有数据
            end_idx = len(video_data)
        else:
            end_idx = start_idx + videos_per_user

        user_data[user_id] = video_data[start_idx:end_idx]

    return user_data


def _partition_non_iid(user_num, alpha, noise_class_to_videos, num_noise_classes):
    """
    Non-IID分区：使用Dirichlet分布控制每个用户的noise level分布
    """
    user_data = {user_id: [] for user_id in range(user_num)}

    # 对每个noise class，使用Dirichlet分布决定分配给各用户的比例
    for noise_class, videos in noise_class_to_videos.items():
        if not videos:
            continue

        # 使用Dirichlet分布生成该noise class在各用户间的分配比例
        proportions = np.random.dirichlet([alpha] * user_num)

        # 随机打乱该类别的视频
        random.shuffle(videos)

        # 根据比例分配给各用户
        start_idx = 0
        for user_id in range(user_num):
            if user_id == user_num - 1:  # 最后一个用户获得剩余数据
                user_videos = videos[start_idx:]
            else:
                num_videos = int(len(videos) * proportions[user_id])
                user_videos = videos[start_idx:start_idx + num_videos]
                start_idx += num_videos

            user_data[user_id].extend(user_videos)

    # 打乱每个用户的数据顺序
    for user_id in user_data:
        random.shuffle(user_data[user_id])

    return user_data





class LocalVideoSequenceDataset(Dataset):
    """
    针对单个客户端的本地 Dataset。
    传入 partition[user_id] 对应的 [(video_path, noise_level), ...]。
    """
    def __init__(self,
                 local_user_data,                 # list[(path, noise_level)]
                 sequence_length,
                 crop_size,
                 epoch_size=-1,
                 random_shuffle=True,
                 temp_stride=-1,
                 preload=True):
        super().__init__()

        assert len(local_user_data) > 0, "This client has no videos."
        self.seq_len = sequence_length
        self.half = (sequence_length - 1) // 2
        self.stride = temp_stride if temp_stride > 0 else sequence_length
        self.crop_size = crop_size
        self.random_shuffle = random_shuffle
        self.epoch_size = epoch_size
        self.preload = preload

        self.to_tensor = T.ToTensor()

        # 1) 读入所有视频（或仅保留路径以便懒加载）
        self.video_paths = []
        self.video_noise_levels = []
        self.videos = []  # 若 preload=True，这里放 numpy array；否则放 None

        for vp, nl in local_user_data:
            self.video_paths.append(Path(vp))
            self.video_noise_levels.append(nl)
            if self.preload:
                frames = iio.imread(str(vp), plugin="pyav")  # [T, H, W, 3], uint8
                self.videos.append(np.asarray(frames))
            else:
                self.videos.append(None)

        # 2) 为该客户端生成 (vid_idx, center_idx, noise_level) 采样对
        self.pairs = []
        for vid_idx, (vp, nl) in enumerate(zip(self.video_paths, self.video_noise_levels)):
            if self.preload:
                T_frames = self.videos[vid_idx].shape[0]
            else:
                # 懒加载模式下，先开一次获取帧数（避免全量保留）
                meta_frames = iio.imread(str(vp), plugin="pyav").shape[0]
                T_frames = meta_frames

            valid_centers = list(range(
                self.half * self.stride,
                T_frames - self.half * self.stride
            ))
            for c in valid_centers:
                self.pairs.append((vid_idx, c, nl))

        # 3) 设定 epoch_size
        if self.epoch_size <= 0:
            self.epoch_size = len(self.pairs)

    def __len__(self):
        return self.epoch_size

    def _load_video(self, vid_idx):
        if self.preload:
            return self.videos[vid_idx]
        # 懒加载读取
        frames = iio.imread(str(self.video_paths[vid_idx]), plugin="pyav")
        return np.asarray(frames)

    def __getitem__(self, idx):
        # 选择一个 pair
        if self.random_shuffle:
            vid_idx, center, nl = random.choice(self.pairs)
        else:
            vid_idx, center, nl = self.pairs[idx % len(self.pairs)]

        video = self._load_video_if_needed(vid_idx)  # [T, H, W, 3]

        # 取连续的 F 帧（带 stride）
        seq_imgs = []
        for i in range(self.seq_len):
            frame_idx = center + (i - self.half) * self.stride
            arr = video[frame_idx]  # H, W, 3
            t = torch.from_numpy(arr.astype(np.float32).transpose(2, 0, 1) / 255.0)  # [C, H, W]
            seq_imgs.append(t)
        seq = torch.stack(seq_imgs, dim=0)  # [F, C, H, W]

        # 随机裁剪（注意：torchvision 的 RandomCrop 是对 [C,H,W] 裁剪）
        i, j, h, w = T.RandomCrop.get_params(seq[0], output_size=(self.crop_size, self.crop_size))
        seq = seq[:, :, i:i + h, j:j + w]  # [F, C, Hc, Wc]

        # 中心帧 GT（修正：应为 seq[self.half]，shape [C, H, W]）
        gt = seq[self.half]

        return seq, gt, float(nl)


class ValDataset(Dataset):

    def __init__(self, data_dir=None, gray_mode=False, num_input_frames=NUMFRXSEQ_VAL):
        self.gray_mode = gray_mode
        self.data_dir = Path(data_dir)
        self.num_input_frames = num_input_frames

        seq_dirs = sorted(self.data_dir.glob(VALSEQPATT))

        self.sequences = []
        for seq_path in seq_dirs:
            seq, _, _ = open_sequence(
                str(seq_path),  # open_sequence likely expects string path
                gray_mode=gray_mode,
                expand_if_needed=False,
                max_num_fr=num_input_frames
            )
            self.sequences.append(seq)

    def __getitem__(self, index):
        return torch.from_numpy(self.sequences[index])

    def __len__(self):
        return len(self.sequences)

