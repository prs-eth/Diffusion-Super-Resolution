# Original Code from: https://github.com/prs-eth/graph-super-resolution
import re
import csv
import random
import warnings

import numpy as np
import torch
from torchvision.transforms import RandomCrop, RandomRotation
import torchvision.transforms.functional as F
from skimage.measure import block_reduce
from scipy import interpolate

ROTATION_EXPAND = False
ROTATION_CENTER = None  # image center
ROTATION_FILL = 0.


def downsample(image, scaling_factor):
    """
    Performs average pooling, ignoring nan values
    :param image: torch tensor or numpy ndarray of shape (B, C, H, W)
    """
    if image.ndim != 4:
        raise ValueError(f'Image should have four dimensions, got {image.ndim}')

    is_tensor = torch.is_tensor(image)
    if is_tensor:
        device = image.device
        image = image.detach().cpu().numpy()

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'Mean of empty slice')
        image = block_reduce(image, (1, 1, scaling_factor, scaling_factor), np.nanmean)

    return torch.from_numpy(image).to(device) if is_tensor else image


def bicubic_with_mask(source, mask, scaling_factor):
    source_size = source.shape[0]
    H, W = source.shape[0] * scaling_factor, source.shape[1] * scaling_factor

    source_r = source.flatten()
    mask_r = mask.flatten()

    x = np.arange((scaling_factor - 1) / 2, H, scaling_factor)  # (0,source.shape[0])
    y = np.arange((scaling_factor - 1) / 2, H, scaling_factor)  # (0,source.shape[1])

    x_g, y_g = np.meshgrid(x, y)  # indexing="ij"
    x_g_r = x_g.flatten()
    y_g_r = y_g.flatten()

    source_r = source_r[mask_r == 1]
    x_g_r = x_g_r[mask_r == 1]
    y_g_r = y_g_r[mask_r == 1]
    xy_g_r = np.concatenate([x_g_r[:, None], y_g_r[:, None]], axis=1)

    x_HR = np.linspace(0, W, endpoint=False, num=W)
    y_HR = np.linspace(0, H, endpoint=False, num=H)

    x_HR_g, y_HR_g = np.meshgrid(x_HR, y_HR)
    x_HR_g, y_HR_g = x_HR_g.flatten(), y_HR_g.flatten()
    xy_HR_g_r = np.concatenate([x_HR_g[:, None], y_HR_g[:, None]], axis=1)

    depth_HR = interpolate.griddata(xy_g_r, source_r, xy_HR_g_r, method="cubic")
    depth_HR_nearest = interpolate.griddata(xy_g_r, source_r, xy_HR_g_r, method="nearest")
    depth_HR[np.isnan(depth_HR)] = depth_HR_nearest[np.isnan(depth_HR)]
    depth_HR = depth_HR.reshape(source_size * scaling_factor, -1)

    return depth_HR


def random_horizontal_flip(images, p=0.5):
    if random.random() < p:
        return [image.flip(-1) for image in images]
    return images


def random_rotate(images, max_rotation_angle, interpolation, crop_valid=False):
    angle = RandomRotation.get_params([-max_rotation_angle, max_rotation_angle])
    if crop_valid:
        rotated = [F.rotate(image, angle, interpolation, True, ROTATION_CENTER, ROTATION_FILL) for image in images]
        crop_params = np.floor(np.asarray(rotated[0].shape[1:3]) - 2. *
                      (np.sin(np.abs(angle * np.pi / 180.)) * np.asarray(images[0].shape[1:3][::-1]))).astype(int)
        return [F.center_crop(image, crop_params) for image in rotated]
    else:
        return [F.rotate(image, angle, interpolation, ROTATION_EXPAND, ROTATION_CENTER, ROTATION_FILL) for image in images]


def random_crop(images, crop_size):
    crop_params = RandomCrop.get_params(images[0], crop_size)
    return [F.crop(image, *crop_params) for image in images]


# Following contents were adapted from https://www.programmersought.com/article/2506939342/.
def _read_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode('utf-8').rstrip()
        channels = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(pfm_file.readline().decode('utf-8').rstrip())
        if scale < 0:
            endian = '<'  # little endian
        else:
            endian = '>'  # big endian

        disparity = np.fromfile(pfm_file, endian + 'f')

    return disparity, (height, width, channels)


def read_calibration(calib_file_path):
    with open(calib_file_path, 'r') as calib_file:
        calib = {}
        csv_reader = csv.reader(calib_file, delimiter='=')
        for attr, value in csv_reader:
            calib.setdefault(attr, value)

    return calib


def create_depth_from_pfm(pfm_file_path, calib=None):
    disparity, shape = _read_pfm(pfm_file_path)

    if calib is None:
        raise Exception('No calibration information available')
    else:
        fx = float(calib['cam0'].split(' ')[0].lstrip('['))
        base_line = float(calib['baseline'])
        doffs = float(calib['doffs'])

        depth_map = fx * base_line / (disparity + doffs)
        depth_map = np.reshape(depth_map, newshape=shape)
        depth_map = np.flipud(depth_map).transpose((2, 0, 1)).copy()

        depth_map[depth_map == 0.] = np.nan

        return depth_map
