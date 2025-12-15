"""Core utility functions used across training/testing.

This file consolidates the previously split utils modules:
- io.py
- base_functions.py
- fastdvdnet_utils.py
- warp.py
- raw.py
- ssim.py

The goal is to keep the project structure simple while preserving behavior.
"""

from __future__ import annotations

import glob
import math
import os
import re
from random import choices
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import imageio
import numpy as np
import torch

# -----------------------------
# Logging / IO
# -----------------------------

def list_dir(dir: str, postfix: Optional[str] = None, full_path: bool = False) -> List[str]:
    if full_path:
        if postfix is None:
            names = sorted([name for name in os.listdir(dir) if not name.startswith('.')])
            return sorted([os.path.join(dir, name) for name in names])
        else:
            names = sorted([name for name in os.listdir(dir) if (not name.startswith('.') and name.endswith(postfix))])
            return sorted([os.path.join(dir, name) for name in names])
    else:
        if postfix is None:
            return sorted([name for name in os.listdir(dir) if not name.startswith('.')])
        else:
            return sorted([name for name in os.listdir(dir) if (not name.startswith('.') and name.endswith(postfix))])


def open_images_uint8(image_files: Sequence[str]) -> np.ndarray:
    image_list = []
    for image_file in image_files:
        image = imageio.imread(image_file).astype(np.uint8)
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        image_list.append(image)
    seq = np.stack(image_list, axis=0)
    return seq


def log(log_file: str, s: str, also_print: bool = True) -> None:
    with open(log_file, 'a+', encoding='utf-8') as f:
        f.write(s)
    if also_print:
        print(s, end='')


# return pytorch image in shape 1xCxHxW
def image2tensor(image_file: str) -> torch.Tensor:
    image = imageio.imread(image_file).astype(np.float32) / np.float32(255.0)
    if len(image.shape) == 3:
        image = np.transpose(image, (2, 0, 1))
    elif len(image.shape) == 2:
        image = np.expand_dims(image, 0)
    image = np.asarray(image, dtype=np.float32)
    image = torch.from_numpy(image).unsqueeze(0)
    return image


# save numpy image in shape CxHxW
def np2image(image: np.ndarray, image_file: str) -> None:
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0).astype(np.uint8)
    imageio.imwrite(image_file, image)


def np2image_bgr(image: np.ndarray, image_file: str) -> None:
    image = np.transpose(image, (1, 2, 0))
    image = np.clip(image, 0.0, 1.0)
    image = (image * 255.0).astype(np.uint8)
    cv2.imwrite(image_file, image)


# save tensor image in shape 1xCxHxW
def tensor2image(image: torch.Tensor, image_file: str) -> None:
    image = image.detach().cpu().squeeze(0).numpy()
    np2image(image, image_file)


# -----------------------------
# Training helpers
# -----------------------------

def resume_training(args, model, optimizer, scheduler):
    """Resumes previous training or starts anew."""
    model_files = glob.glob(os.path.join(args['log_dir'], '*.pth'))

    if len(model_files) == 0:
        start_epoch = 0
    else:
        epochs_exist = []
        for model_file in model_files:
            result = re.findall('ckpt_e(.*).pth', model_file)
            epochs_exist.append(int(result[0]))
        max_epoch = max(epochs_exist)
        max_epoch_model_file = os.path.join(args['log_dir'], 'ckpt_e%d.pth' % max_epoch)
        checkpoint = torch.load(max_epoch_model_file, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        start_epoch = max_epoch
        log(args['log_file'], f'> Resuming previous training from epoch:{max_epoch}\n')

    return start_epoch


def save_model(args, model, optimizer, scheduler, epoch: int) -> None:
    save_dict = {
        'args': args,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }

    torch.save(save_dict, os.path.join(args['log_dir'], 'ckpt_e{}.pth'.format(epoch)))


# the same as skimage.metrics.peak_signal_noise_ratio
def batch_psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.clamp(a, 0, 1)
    b = torch.clamp(b, 0, 1)
    x = torch.mean((a - b) ** 2, dim=[-3, -2, -1])
    return 20 * torch.log(1 / torch.sqrt(x)) / math.log(10)


# -----------------------------
# FastDVDNet-style utilities
# -----------------------------

from skimage.metrics import peak_signal_noise_ratio as compare_psnr

IMAGETYPES = ('*.bmp', '*.png', '*.jpg', '*.jpeg', '*.tif')


def fastdvdnet_batch_psnr(img: torch.Tensor, imclean: torch.Tensor, data_range: float = 1.0) -> float:
    """Compute PSNR along batch dimension."""
    img_cpu = img.data.cpu().numpy().astype(np.float32)
    imgclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0.0
    for i in range(img_cpu.shape[0]):
        psnr += compare_psnr(imgclean[i].clip(0, 1), img_cpu[i], data_range=data_range)
    return psnr / img_cpu.shape[0]


def get_imagenames(seq_dir: str, pattern: Optional[str] = None) -> List[str]:
    files: List[str] = []
    for typ in IMAGETYPES:
        files.extend(glob.glob(os.path.join(seq_dir, typ)))

    if pattern is not None:
        files = [f for f in files if pattern in os.path.split(f)[-1]]

    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return files


def open_sequence(seq_dir: str, gray_mode: bool, expand_if_needed: bool = False, max_num_fr: int = 85):
    """Open an image sequence into a numpy array of shape [T, C, H, W]."""
    files = get_imagenames(seq_dir)
    seq_list = []
    for fpath in files[0:max_num_fr]:
        img, expanded_h, expanded_w = open_image(
            fpath,
            gray_mode=gray_mode,
            expand_if_needed=expand_if_needed,
            expand_axis0=False,
        )
        seq_list.append(img)
    seq = np.stack(seq_list, axis=0)
    return seq, expanded_h, expanded_w


def open_image(
    fpath: str,
    gray_mode: bool,
    expand_if_needed: bool = False,
    expand_axis0: bool = True,
    normalize_data: bool = True,
):
    """Open a single image file."""
    if not gray_mode:
        img = cv2.imread(fpath)
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
    else:
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        img = np.expand_dims(img, 0)

    if expand_axis0:
        img = np.expand_dims(img, 0)

    expanded_h = False
    expanded_w = False
    sh_im = img.shape
    if expand_if_needed:
        if sh_im[-2] % 2 == 1:
            expanded_h = True
            if expand_axis0:
                img = np.concatenate((img, img[:, :, -1, :][:, :, np.newaxis, :]), axis=2)
            else:
                img = np.concatenate((img, img[:, -1, :][:, np.newaxis, :]), axis=1)

        if sh_im[-1] % 2 == 1:
            expanded_w = True
            if expand_axis0:
                img = np.concatenate((img, img[:, :, :, -1][:, :, :, np.newaxis]), axis=3)
            else:
                img = np.concatenate((img, img[:, :, -1][:, :, np.newaxis]), axis=2)

    if normalize_data:
        img = normalize(img)
    return img, expanded_h, expanded_w


def normalize(data: np.ndarray) -> np.ndarray:
    """Normalize uint8 image to float32 [0,1]."""
    return np.float32(data / 255.0)


def normalize_augment(img_train: torch.Tensor, aug: bool = False) -> torch.Tensor:
    """Normalize and (optionally) augment a patch of shape [N, T, C, H, W]."""

    def transform(sample: torch.Tensor) -> torch.Tensor:
        do_nothing = lambda x: x
        flipud = lambda x: torch.flip(x, dims=[2])
        rot90 = lambda x: torch.rot90(x, k=1, dims=[2, 3])
        rot90_flipud = lambda x: torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[2])
        rot180 = lambda x: torch.rot90(x, k=2, dims=[2, 3])
        rot180_flipud = lambda x: torch.flip(torch.rot90(x, k=2, dims=[2, 3]), dims=[2])
        rot270 = lambda x: torch.rot90(x, k=3, dims=[2, 3])
        rot270_flipud = lambda x: torch.flip(torch.rot90(x, k=3, dims=[2, 3]), dims=[2])
        add_csnt = lambda x: x + torch.normal(
            mean=torch.zeros(x.size()[0], 1, 1, 1),
            std=(5 / 255.0),
        ).expand_as(x).to(x.device)

        aug_list = [do_nothing, flipud, rot90, rot90_flipud, rot180, rot180_flipud, rot270, rot270_flipud, add_csnt]
        w_aug = [32, 12, 12, 12, 12, 12, 12, 12, 0]
        transf = choices(aug_list, w_aug)
        return transf[0](sample)

    N, T, C, H, W = img_train.shape
    img_train = img_train.type(torch.float32).view(N, -1, H, W) / 255.0

    if aug:
        img_train = transform(img_train)

    img_train = img_train.view(N, T, C, H, W)
    return img_train


def remove_dataparallel_wrapper(state_dict):
    """Remove the 'module.' prefix in DataParallel checkpoints."""
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict


# -----------------------------
# RAW / warping / SSIM
# -----------------------------

def demosaic(raw_seq: torch.Tensor) -> torch.Tensor:
    """Convert raw seq (N,T,4,H,W) to rgb seq (N,T,3,H,W) for flow."""
    N, T, C, H, W = raw_seq.shape
    rgb_seq = torch.empty((N, T, 3, H, W), dtype=raw_seq.dtype, device=raw_seq.device)
    rgb_seq[:, :, 0] = raw_seq[:, :, 0]
    rgb_seq[:, :, 1] = (raw_seq[:, :, 1] + raw_seq[:, :, 2]) / 2
    rgb_seq[:, :, 2] = raw_seq[:, :, 3]
    return rgb_seq


def warp(x: torch.Tensor, flo: torch.Tensor):
    """Warp an image/tensor back according to optical flow."""
    B, C, H, W = x.size()
    xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(x.device)
    vgrid = torch.autograd.Variable(grid) + flo

    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(x, vgrid, mode='nearest', align_corners=True)

    return output, None


from skimage.metrics import structural_similarity as compare_ssim


def batch_ssim(img: torch.Tensor, imclean: torch.Tensor, data_range):
    """Compute SSIM along the temporal dimension (T)."""
    img = img.data.cpu().numpy().astype(np.float32)
    img = np.transpose(img, (0, 2, 3, 1))
    img_clean = imclean.data.cpu().numpy().astype(np.float32)
    img_clean = np.transpose(img_clean, (0, 2, 3, 1))

    ssim = 0.0
    for i in range(img.shape[0]):
        origin_i = img_clean[i, :, :, :]
        denoised_i = img[i, :, :, :]
        ssim += compare_ssim(
            origin_i.astype(float),
            denoised_i.astype(float),
            channel_axis=2,
            win_size=11,
            K1=0.01,
            K2=0.03,
            sigma=1.5,
            gaussian_weights=True,
            data_range=1,
        )
    return ssim / img.shape[0]
