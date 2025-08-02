import torch
import torch.nn as nn
from torchvision.transforms import Compose

import cv2
from collections import defaultdict
from skimage.metrics import peak_signal_noise_ratio
from typing import Optional, Tuple, List

import copy
import numpy as np
from pathlib import Path
import random
image_types = ['*.png', '*.jpg', '*.jpeg', '*.bmp']

class Normalize(nn.Module):
    def forward(self, x):
        print("cnm",x.shape)
        x = x.float() / 255.0
        n, f, c, h, w = x.shape
        return x.view(n, f * c, h, w)


class Augment(nn.Module):
    def __init__(self):
        super().__init__()
        self.op_names = [
            'do_nothing', 'flipud', 'rot90', 'rot90_flipud',
            'rot180', 'rot180_flipud', 'rot270', 'rot270_flipud', 'add_noise'
        ]
        self.weights = [32, 12, 12, 12, 12, 12, 12, 12, 12]

    def augment(self, op_name, img):

        match op_name:
            case 'do_nothing':
                return img
            case 'flipud':
                return torch.flip(img, dims=[1])
            case 'rot90':
                return torch.rot90(img, k=1, dims=[1, 2])
            case 'rot90_flipud':
                return torch.flip(torch.rot90(img, k=1, dims=[1, 2]), dims=[1])
            case 'rot180':
                return torch.rot90(img, k=2, dims=[1, 2])
            case 'rot180_flipud':
                return torch.flip(torch.rot90(img, k=2, dims=[1, 2]), dims=[1])
            case 'rot270':
                return torch.rot90(img, k=3, dims=[1, 2])
            case 'rot270_flipud':
                return torch.flip(torch.rot90(img, k=3, dims=[1, 2]), dims=[1])
            case 'add_noise':
                noise = torch.randn(1, 1, 1, device=img.device) * (5.0 / 255.0)
                return torch.clamp(img + noise.expand_as(img), 0.0, 1.0)
            case _:
                raise ValueError(f"Unsupported op name: {op_name}")

    def forward(self, x):
        N, FC, H, W = x.shape
        F = FC // 3
        out = torch.empty_like(x)
        op_name = random.choices(self.op_names, weights=self.weights, k=1)[0]
        print(op_name, "new")
        for n in range(N):
            for f in range(F):
                img = x[n, f * 3:f * 3 + 3]  # [3, H, W]
                out[n, f * 3:f * 3 + 3] = self.augment(op_name, img)
        return out
def normalize_augment(data_input, ctrl_fr_idx):
    video_transform = Compose([
        Normalize(),
        Augment(),
    ])
    img_train = video_transform(data_input)
    gt_train = img_train[:, 3 * ctrl_fr_idx:3 * ctrl_fr_idx + 3, :, :]
    return img_train, gt_train

def orthogonal_conv_weights(layer):
    if not isinstance(layer, nn.Conv2d):
        return
    weight_tmp = layer.weight.data.clone()
    c_out, c_in, kh, kw = weight_tmp.shape
    dtype = weight_tmp.dtype
    weight_flat = weight_tmp.permute(2, 3, 1, 0).contiguous().view(-1, c_out)
    try:
        U, _, V = torch.linalg.svd(weight_flat, full_matrices=False)
        weight_ortho = torch.matmul(U, V)

        weight_new = weight_ortho.view(kh, kw, c_in, c_out).permute(3, 2, 0, 1).contiguous()
        layer.weight.data.copy_(weight_new.to(dtype))
    except RuntimeError as e:
        print(f"SVD failed for {layer.__class__.__name__}: {e}")
def get_image_names(seq_dir, pattern=None):
    seq_path = Path(seq_dir)
    files = []

    for image_type in image_types:
        files.extend(seq_path.glob(image_type))

    if pattern is not None:
        files = [file for file in files if pattern in file.name]

    files.sort(key=lambda file: int(''.join(filter(str.isdigit, file.name))))
    return [str(file) for file in files]


def open_image(fpath, gray_mode, expand_if_needed=False, expand_axis0=True, normalize_data=True):
    if gray_mode:
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(fpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)

    if expand_axis0:
        img = np.expand_dims(img, axis=0)

    expanded_h, expanded_w = False, False
    h, w = img.shape[-2], img.shape[-1]

    if expand_if_needed:
        if h % 2 == 1:
            expanded_h = True
            last_row = img[..., -1:, :]
            img = np.concatenate((img, last_row), axis=-2)

        if w % 2 == 1:
            expanded_w = True
            last_col = img[..., -1:]  # slice last col
            img = np.concatenate((img, last_col), axis=-1)

    if normalize_data:
        img = img.astype(np.float32) / 255.0

    return img, expanded_h, expanded_w
def open_sequence(seq_dir, gray_mode, expand_if_needed=False, max_num_fr=100):
    file_paths = get_image_names(seq_dir)
    file_paths = file_paths[:max_num_fr]

    print(f"Open sequence in folder: {seq_dir} ({len(file_paths)} frames)")

    seq_list = []
    expanded_h, expanded_w = False, False

    for fpath in file_paths:
        img, h_exp, w_exp = open_image(
            fpath,
            gray_mode=gray_mode,
            expand_if_needed=expand_if_needed,
            expand_axis0=False
        )
        seq_list.append(img)
        expanded_h |= h_exp
        expanded_w |= w_exp

    # Stack to [T, C, H, W]
    seq = np.stack(seq_list, axis=0)
    return seq, expanded_h, expanded_w





class MemReporter():
    """A memory reporter that collects tensors and memory usages

    Parameters:
        - model: an extra nn.Module can be passed to infer the name
        of Tensors

    """
    def __init__(self, model: Optional[torch.nn.Module] = None):
        self.tensor_name = {}
        self.device_mapping = defaultdict(list)
        self.device_tensor_stat = {}
        # to numbering the unknown tensors
        self.name_idx = 0

        tensor_names = defaultdict(list)
        if model is not None:
            assert isinstance(model, torch.nn.Module)
            # for model with tying weight, multiple parameters may share
            # the same underlying tensor
            for name, param in model.named_parameters():
                tensor_names[param].append(name)

        for param, name in tensor_names.items():
            self.tensor_name[id(param)] = '+'.join(name)
def batch_psnr(images, images_clean, data_range):
    images_cpu = images.data.cpu().numpy().astype(np.float32)
    images_clean = images_clean.data.cpu().numpy().astype(np.float32)
    psnr = 0.0
    for index in range(images_cpu.shape[0]):
        psnr += peak_signal_noise_ratio(images_clean[index, :, :, :], images_cpu[index, :, :, :],
                                        data_range=data_range)
    return psnr / images_cpu.shape[0]
def average_weights(w):
    """
    average the weights from all local models
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg
