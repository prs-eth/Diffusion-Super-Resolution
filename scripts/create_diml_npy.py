import sys
from pathlib import Path

from PIL import Image
import numpy as np

RESOLUTIONS = ['HR']
DIML_PATH = Path(sys.argv[1]).resolve()
SAVE_PATH = DIML_PATH / 'npy'
TRAIN_RATIO = 0.9

assert DIML_PATH.exists()
SAVE_PATH.mkdir(exist_ok=True, parents=True)

for split in ('train', 'val', 'test'):
    for resolution in RESOLUTIONS:
        images, depth_maps = [], []

        # use last images of each train scene as val data
        split_file = 'train' if split == 'val' else split
        scenes = sorted([s for s in (DIML_PATH / f'{split_file}/{resolution}').iterdir() if not s.stem.startswith('.')])

        for scene in scenes:
            image_files = sorted([s for s in scene.glob(f'color/*.png') if not s.stem.startswith('.')])
            boundary = int(len(image_files) * TRAIN_RATIO)
            if split == 'train':
                slc = slice(0, boundary)
            elif split == 'val':
                slc = slice(boundary, None)
            else:
                # use all images from test set
                slc = slice(None)

            for image_file in image_files[slc]:
                images.append(np.array(Image.open(image_file)).transpose((2, 0, 1)))

                depth_file = image_file.parent.parent / f'depth_filled/{image_file.stem[:-2]}_depth_filled.png'
                depth_map = np.array(Image.open(depth_file))
                # fit in uint16 to save memory
                assert depth_map.max() <= np.iinfo(np.uint16).max
                depth_maps.append(depth_map.astype(np.uint16))

        print(f'{split}/{resolution}: {len(images)} images')

        np.save(str(SAVE_PATH / f'images_{split}_{resolution}.npy'), images)
        np.save(str(SAVE_PATH / f'depth_{split}_{resolution}.npy'), depth_maps)
