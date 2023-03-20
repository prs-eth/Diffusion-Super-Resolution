# Original Code from: https://github.com/prs-eth/graph-super-resolution
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Resize, InterpolationMode
from PIL import Image

from .utils import downsample, bicubic_with_mask, random_crop, random_rotate, random_horizontal_flip, \
    read_calibration, create_depth_from_pfm

VAL_SET_2005_2006 = ['Moebius', 'Lampshade1', 'Lampshade2']
VAL_SET_2014 = ['Shelves-perfect', 'Playtable-perfect']
TEST_SET_2005_2006 = ['Reindeer', 'Bowling1', 'Bowling2']
TEST_SET_2014 = ['Adirondack-perfect', 'Motorcycle-perfect']


class MiddleburyDataset(Dataset):

    def __init__(
            self,
            data_dir: str,
            datasets=('2005', '2006', '2014'),
            split='train',
            crop_size=(128, 128),
            do_horizontal_flip=True,
            max_rotation_angle=15,
            scale_interpolation=InterpolationMode.BILINEAR,
            rotation_interpolation=InterpolationMode.BILINEAR,
            image_transform=None,
            depth_transform=None,
            use_ambient_images=False,
            crop_deterministic=False,
            scaling=8,
            **kwargs
    ):
        if split not in ('train', 'val', 'test'):
            raise ValueError(split)

        if max_rotation_angle > 0 and crop_deterministic:
            max_rotation_angle = 0
            print('Set max_rotation_angle to zero because of deterministic cropping')

        self.split = split
        self.crop_size = crop_size
        self.do_horizontal_flip = do_horizontal_flip
        self.max_rotation_angle = max_rotation_angle
        self.rotation_interpolation = rotation_interpolation
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.data = []
        self.crop_deterministic = crop_deterministic
        self.scaling = scaling

        data_dir = Path(data_dir)

        # read in various Middlebury datasets using the respective global load_{name} function
        for name in ('2005', '2006', '2014'):
            if name in datasets:
                self.data.extend(globals()[f'load_{name}'](data_dir / name, 1.0, scale_interpolation,
                                                           use_ambient_images, split))
        if self.crop_deterministic:
            assert not use_ambient_images
            # construct deterministic mapping
            self.deterministic_map = []
            for i, datum in enumerate(self.data):
                H, W = datum[0][0].shape[1:]
                num_crops_h, num_crops_w = H // self.crop_size[0], W // self.crop_size[1]
                self.deterministic_map.extend(((i, j, k) for j in range(num_crops_h) for k in range(num_crops_w)))

    def __getitem__(self, index, called=0):
        if called >= 32:  # it has been called 32 times, enough of that
            raise ValueError

        if self.crop_deterministic:
            im_index, crop_index_h, crop_index_w = self.deterministic_map[index]
        else:
            im_index = index

        image, depth_map = random.choice(self.data[im_index])
        image, depth_map = image.clone(), depth_map.clone()

        if self.do_horizontal_flip and not self.crop_deterministic:
            image, depth_map = random_horizontal_flip((image, depth_map))

        if self.max_rotation_angle > 0 and not self.crop_deterministic:
            image, depth_map = random_rotate((image, depth_map), self.max_rotation_angle, self.rotation_interpolation)
            # passing fill=np.nan to rotate sets all pixels to nan, so set it here explicitly
            depth_map[depth_map == 0.] = np.nan

        if self.crop_deterministic:
            slice_h = slice(crop_index_h * self.crop_size[0], (crop_index_h + 1) * self.crop_size[0])
            slice_w = slice(crop_index_w * self.crop_size[1], (crop_index_w + 1) * self.crop_size[1])
            image, depth_map = image[:, slice_h, slice_w], depth_map[:, slice_h, slice_w]
        else:
            image, depth_map = random_crop((image, depth_map), self.crop_size)

        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.depth_transform is not None:
            depth_map = self.depth_transform(depth_map)

        source = downsample(depth_map.unsqueeze(0), self.scaling).squeeze().unsqueeze(0)

        mask_hr = (~torch.isnan(depth_map)).float()
        mask_lr = (~torch.isnan(source)).float()

        depth_map[mask_hr == 0.] = 0.
        source[mask_lr == 0.] = 0.

        if self.split == 'train' and (torch.mean(mask_lr) < 0.9 or torch.mean(mask_hr) < 0.8):
            # omit patch due to too many depth holes, try another one
            return self.__getitem__(index, called=called + 1)

        try:
            y_bicubic = torch.from_numpy(bicubic_with_mask(
                source.squeeze().numpy(), mask_lr.squeeze().numpy(), self.scaling)).float()
            y_bicubic = y_bicubic.reshape((1, self.crop_size[0], self.crop_size[1]))
            return {'guide': image, 'y': depth_map, 'source': source, 'mask_hr': mask_hr, 'mask_lr': mask_lr,
                    'im_idx': im_index, 'y_bicubic': y_bicubic}
        except:
            return self.__getitem__(index, called=called + 1)

    def __len__(self):
        return len(self.deterministic_map if self.crop_deterministic else self.data)


def load_2014(data_dir: Path, scale, scale_interpolation, use_ambient_images, split):
    data = []
    for scene in sorted(data_dir.iterdir()):
        # ignore scenes with imperfect rectification, these are only included in the 2014 dataset anyway
        if not scene.is_dir() or scene.name.endswith('-imperfect'):
            continue

        # make train val test split
        last_dir = scene.parts[-1]

        if (split == 'test' and last_dir in TEST_SET_2014) or (split == 'val' and last_dir in VAL_SET_2014) or \
                (split == 'train' and (last_dir not in TEST_SET_2014) and (last_dir not in VAL_SET_2014)):
            calibration = read_calibration(scene / 'calib.txt')

            # add left and right view, as well as corresponding depth maps
            for view in (0, 1):
                resize = Resize((int(int(calibration['height']) * scale), int(int(calibration['width']) * scale)),
                                scale_interpolation)
                depth_map = resize(torch.from_numpy(create_depth_from_pfm(scene / f'disp{view}.pfm', calibration)))
                transform = Compose([ToTensor(), resize])
                if use_ambient_images:
                    data.append(
                        [(transform(Image.open(path)), depth_map) for path in scene.glob(f'ambient/L*/im{view}*.png')])
                else:
                    data.append([(transform(Image.open(scene / f'im{view}.png')), depth_map)])

    return data


def load_2006(data_dir: Path, scale, scale_interpolation, use_ambient_images, split):
    f = 3740  #px
    baseline = 160  #mm

    data = []
    for scene in sorted(data_dir.iterdir()):
        if not scene.is_dir():
            continue

        # make train val test split
        last_dir = scene.parts[-1]
        if (split == 'test' and last_dir in TEST_SET_2005_2006) or (
                split == 'val' and last_dir in VAL_SET_2005_2006) or (
                split == 'train' and (last_dir not in TEST_SET_2005_2006) and (last_dir not in VAL_SET_2005_2006)):

            # add left and right view, as well as corresponding depth maps
            for view in (1, 5):
                disparities = torch.from_numpy(np.array(Image.open(scene / f'disp{view}.png'))).float().unsqueeze(0)
                # zero disparities are to be interpreted as inf, set them to nan so they result in nan depth
                disparities[disparities == 0.] = np.nan
                with open(scene / 'dmin.txt') as fh:
                    dmin = int(fh.read().strip())
                # add dmin to disparities because disparity maps and images have been cropped to the joint field of view
                disparities += dmin

                depth_map = baseline * f / disparities
                resize = Resize((int(depth_map.shape[1] * scale), int(depth_map.shape[2] * scale)), scale_interpolation)
                depth_map = resize(depth_map)
                transform = Compose([ToTensor(), resize])
                if use_ambient_images:
                    data.append(
                        [(transform(Image.open(path)), depth_map) for path in
                         scene.glob(f'Illum*/Exp*/view{view}.png')])
                else:
                    data.append([(transform(Image.open(scene / f'view{view}.png')), depth_map)])

    return data


# 2005 dataset same as 2006
def load_2005(*args, **kwargs):
    return load_2006(*args, **kwargs)
