# Original Code from: https://github.com/prs-eth/graph-super-resolution
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode, Resize

from .utils import downsample, bicubic_with_mask, random_crop, random_rotate, random_horizontal_flip

DIML_BASE_SIZE = (756, 1344)


class DIMLDataset(Dataset):

    def __init__(
            self,
            data_dir: str,
            resolution='HR',
            scale=1.0,
            crop_size=(128, 128),
            do_horizontal_flip=True,
            max_rotation_angle: int = 15,
            scale_interpolation=InterpolationMode.BILINEAR,
            rotation_interpolation=InterpolationMode.BILINEAR,
            image_transform=None,
            depth_transform=None,
            in_memory=True,
            split='train',
            crop_valid=False,
            crop_deterministic=False,
            scaling=8
    ):
        self.scale = scale
        self.crop_size = crop_size
        self.do_horizontal_flip = do_horizontal_flip
        self.max_rotation_angle = max_rotation_angle
        self.scale_interpolation = scale_interpolation
        self.rotation_interpolation = rotation_interpolation
        self.image_transform = image_transform
        self.depth_transform = depth_transform
        self.crop_valid = crop_valid
        self.crop_deterministic = crop_deterministic
        self.scaling = scaling
        data_dir = Path(data_dir)

        if max_rotation_angle > 0 and crop_deterministic:
            raise ValueError('Max rotation angle has to be zero when cropping deterministically')

        if split not in ('train', 'val', 'test'):
            raise ValueError(split)

        mmap_mode = None if in_memory else 'c'

        self.images = np.load(str(data_dir / f'npy/images_{split}_{resolution}.npy'), mmap_mode)
        self.depth_maps = np.load(str(data_dir / f'npy/depth_{split}_{resolution}.npy'), mmap_mode)
        assert len(self.images) == len(self.depth_maps)

        self.H, self.W = int(DIML_BASE_SIZE[0] * self.scale), int(DIML_BASE_SIZE[1] * self.scale)

        if self.crop_valid:
            if self.max_rotation_angle > 45:
                raise ValueError('When crop_valid=True, only rotation angles up to 45° are supported for now')

            # make sure that max rotation angle is valid, else decrease
            max_angle = np.floor(min(
                2 * np.arctan
                    ((np.sqrt(-(crop_size[0] ** 2) + self.H ** 2 + self.W ** 2) - self.W) / (crop_size[0] + self.H)),
                2 * np.arctan
                    ((np.sqrt(-(crop_size[1] ** 2) + self.W ** 2 + self.H ** 2) - self.H) / (crop_size[1] + self.W))
            ) * (180. / np.pi))

            if self.max_rotation_angle > max_angle:
                print(f'max rotation angle too large for given image size and crop size, decreased to {max_angle}')
                self.max_rotation_angle = max_angle

    def __getitem__(self, index):
        if self.crop_deterministic:
            num_crops_h, num_crops_w = self.H // self.crop_size[0], self.W // self.crop_size[1]
            im_index = index // (num_crops_h * num_crops_w)
        else:
            im_index = index

        image = torch.from_numpy(self.images[im_index].astype('float32')) / 255.
        depth_map = torch.from_numpy(self.depth_maps[im_index].astype('float32')).unsqueeze(0)
        resize = Resize((self.H, self.W), self.scale_interpolation)
        image, depth_map = resize(image), resize(depth_map)

        if self.do_horizontal_flip and not self.crop_deterministic:
            image, depth_map = random_horizontal_flip((image, depth_map))

        if self.max_rotation_angle > 0  and not self.crop_deterministic:
            image, depth_map = random_rotate((image, depth_map), self.max_rotation_angle, self.rotation_interpolation,
                                             crop_valid=self.crop_valid)
            # passing fill=np.nan to rotate sets all pixels to nan, so set it here explicitly
            depth_map[depth_map == 0.] = np.nan

        if self.crop_deterministic:
            crop_index = index % (num_crops_h * num_crops_w)
            crop_index_h, crop_index_w = crop_index // num_crops_w, crop_index % num_crops_w
            slice_h = slice(crop_index_h * self.crop_size[0], (crop_index_h + 1) * self.crop_size[0])
            slice_w = slice(crop_index_w * self.crop_size[1], (crop_index_w + 1) * self.crop_size[1])
            image, depth_map = image[:, slice_h, slice_w], depth_map[:, slice_h, slice_w]
        else:
            image, depth_map = random_crop((image, depth_map), self.crop_size)

        # apply user transforms
        if self.image_transform is not None:
            image = self.image_transform(image)
        if self.depth_transform is not None:
            depth_map = self.depth_transform(depth_map)

        source = downsample(depth_map.unsqueeze(0), self.scaling).squeeze().unsqueeze(0)

        mask_hr = (~torch.isnan(depth_map)).float()
        mask_lr = (~torch.isnan(source)).float()

        depth_map[mask_hr == 0.] = 0.
        source[mask_lr == 0.] = 0.

        y_bicubic = torch.from_numpy(
            bicubic_with_mask(source.squeeze().numpy(), mask_lr.squeeze().numpy(), self.scaling)).float()
        y_bicubic = y_bicubic.reshape((1, self.crop_size[0], self.crop_size[1]))

        return {'guide': image, 'y': depth_map, 'source': source, 'mask_hr': mask_hr, 'mask_lr': mask_lr,
                'y_bicubic': y_bicubic}

    def __len__(self):
        if self.crop_deterministic:
            return len(self.depth_maps) * (self.H // self.crop_size[0]) * (self.W // self.crop_size[1])
        return len(self.depth_maps)
