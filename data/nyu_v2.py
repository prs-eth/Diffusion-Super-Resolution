# Original Code from: https://github.com/prs-eth/graph-super-resolution
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

from .utils import downsample, bicubic_with_mask, random_crop, random_rotate, random_horizontal_flip


class NYUv2Dataset(Dataset):

    def __init__(
            self,
            data_dir: str,
            crop_size=(128, 128),
            do_horizontal_flip=True,
            max_rotation_angle: int = 15,
            rotation_interpolation=InterpolationMode.BILINEAR,
            image_transform=None,
            depth_transform=None,
            in_memory=True,
            split='train',
            crop_valid=False,
            crop_deterministic=False,
            scaling=8,
            **kwargs
    ):
        self.crop_size = crop_size
        self.do_horizontal_flip = do_horizontal_flip
        self.max_rotation_angle = max_rotation_angle
        self.rotation_interpolation = rotation_interpolation
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.crop_valid = crop_valid
        self.crop_deterministic = crop_deterministic
        self.scaling = scaling

        import h5py
        file = h5py.File(Path(data_dir) / 'nyu_depth_v2_labeled.mat')

        with open(Path('data') / 'split_idc_nyuv2.json') as fh: # might want to change the location of that file later
            self.split_idc = np.array(json.load(fh)[split])

        if max_rotation_angle > 0 and crop_deterministic:
            raise ValueError('Max rotation angle has to be zero when cropping deterministically')

        self.images = np.array(file['images']) if in_memory else file['images']
        self.depth_maps = np.array(file['depths']) if in_memory else file['depths']
        self.instances = np.array(file['instances']) if in_memory else file['instances']
        self.labels = np.array(file['labels']) if in_memory else file['labels']

        self.W, self.H = self.images.shape[2:]

        if self.crop_valid:
            if self.max_rotation_angle > 45:
                raise ValueError('When crop_valid=True, only rotation angles up to 45° are supported for now')

            # make sure that max rotation angle is valid, else decrease
            max_angle = np.floor(min(
                2 * np.arctan((np.sqrt(-(crop_size[0] ** 2) + self.H ** 2 + self.W ** 2) - self.W) / (crop_size[0] + self.H)),
                2 * np.arctan((np.sqrt(-(crop_size[1] ** 2) + self.W ** 2 + self.H ** 2) - self.H) / (crop_size[1] + self.W))
            ) * (180. / np.pi))

            if self.max_rotation_angle > max_angle:
                print(f'Max rotation angle too large for given image size and crop size, decreased to {max_angle}')
                self.max_rotation_angle = max_angle

    def __getitem__(self, index):
        if self.crop_deterministic:
            num_crops_h, num_crops_w = self.H // self.crop_size[0], self.W // self.crop_size[1]
            im_index = self.split_idc[index // (num_crops_h * num_crops_w)]
        else:
            im_index = self.split_idc[index]

        image = torch.from_numpy(self.images[im_index].astype('float32')).transpose(1, 2) / 255.
        depth_map = torch.from_numpy(self.depth_maps[im_index].astype('float32')).T.unsqueeze(0) * 1000
        instances = torch.from_numpy(self.instances[im_index].astype('int16')).T.unsqueeze(0)
        labels = torch.from_numpy(self.labels[im_index].astype('int16')).T.unsqueeze(0)
        image, depth_map, instances, labels = image.clone(), depth_map.clone(), instances.clone(), labels.clone()

        outputs = [image, depth_map, instances, labels]

        if self.do_horizontal_flip and not self.crop_deterministic:
            outputs = random_horizontal_flip(outputs)

        if self.max_rotation_angle > 0 and not self.crop_deterministic:
            outputs = random_rotate(outputs, self.max_rotation_angle, self.rotation_interpolation,
                                    crop_valid=self.crop_valid)
            # passing fill=np.nan to rotate sets all pixels to nan, so set it here explicitly
            outputs[1][outputs[1] == 0.] = np.nan

        if self.crop_deterministic:
            crop_index = index % (num_crops_h * num_crops_w)
            crop_index_h, crop_index_w = crop_index // num_crops_w, crop_index % num_crops_w
            slice_h = slice(crop_index_h * self.crop_size[0], (crop_index_h + 1) * self.crop_size[0])
            slice_w = slice(crop_index_w * self.crop_size[1], (crop_index_w + 1) * self.crop_size[1])
            outputs = [o[:, slice_h, slice_w] for o in outputs]
        else:
            outputs = random_crop(outputs, self.crop_size)

        # apply user transforms
        if self.image_transform is not None:
            outputs[0] = self.image_transform(outputs[0])
        if self.depth_transform is not None:
            outputs[1] = self.depth_transform(outputs[1])

        source = downsample(outputs[1].unsqueeze(0), self.scaling).squeeze().unsqueeze(0)

        mask_hr = (~torch.isnan(outputs[1])).float()
        mask_lr = (~torch.isnan(source)).float()

        outputs[1][mask_hr == 0.] = 0.
        source[mask_lr == 0.] = 0.

        y_bicubic = torch.from_numpy(
            bicubic_with_mask(source.squeeze().numpy(), mask_lr.squeeze().numpy(), self.scaling)).float()
        y_bicubic = y_bicubic.reshape((1, self.crop_size[0], self.crop_size[1]))

        return {'guide': outputs[0], 'y': outputs[1], 'source': source, 'mask_hr': mask_hr, 'mask_lr': mask_lr,
                'instances': outputs[2], 'labels': outputs[3], 'y_bicubic': y_bicubic}

    def __len__(self):
        if self.crop_deterministic:
            return len(self.split_idc) * (self.H // self.crop_size[0]) * (self.W // self.crop_size[1])
        return len(self.split_idc)
