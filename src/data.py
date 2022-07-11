from __future__ import annotations
from pathlib import Path
import torch
import torchvision
import pandas as pd
import monai
from monai import transforms
import numpy as np
import skimage.io as skio
from PIL import Image
from copy import deepcopy

def make_dataloader(
    dataset,
    batch_size=1,
    num_workers=0,
    pin_memory=True,
    shuffle=False,
    drop_last=False,
):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, pin_memory=pin_memory, 
        num_workers=num_workers, shuffle=shuffle, drop_last=drop_last,
    )


class RegistrationDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data: dict[int, str],
        indexer: list[int] | None = None,
        transforms=None,
    ):
        self.data = data
        self.len = len(data)
        self.indexer = indexer if indexer else list(range(self.len))
        self.transforms = transforms
        
    def set_indexer(self, indexer: list[int]): 
        self.indexer = indexer
        self.len = len(indexer)
        
    def __len__(self): return self.len

    def __getitem__(self, k):
        moving_index = self.indexer[k]
        fixed_index = self.indexer[k-1]
        # fixed_index = self.indexer[np.random.randint(0, len(self.indexer))]
        moving_img = np.load(self.data[moving_index][0]).astype('float32')
        fixed_img = np.load(self.data[fixed_index][0]).astype('float32')
        moving_seg = np.where(np.load(self.data[moving_index][1]).sum(-1) > 0, 1, 0).astype('float32')
        fixed_seg = np.where(np.load(self.data[fixed_index][1]).sum(-1) > 0, 1, 0).astype('float32')
        if self.transforms:
            moving_img = self.transforms(moving_img)
            fixed_img = self.transforms(fixed_img)
            moving_seg = self.transforms(moving_seg)
            fixed_seg = self.transforms(fixed_seg)
        return np.stack([moving_img, fixed_img]), np.stack([moving_seg, fixed_seg])
    
class MultiRegistrationDataset(RegistrationDataset):
    def __init__(
        self,
        datasets: list[datasets],
    ):
        self.datasets=[deepcopy(ds) for ds in datasets]
        
    def __len__(self): return sum(map(len, self.datasets))
    def __getitem__(self, k: int):
        selector = np.random.randint(0, len(self.datasets))
        dataset = self.datasets[selector]
        return dataset[k%len(dataset)]
        
    
def make_datasets(root, img, seg):
    root = Path(root)
    f = lambda f: pd.read_csv(root/f, header=None, dtype=str)
    train_df, val_df, test_df = map(f, ('train.txt', 'val.txt', 'test.txt'))
    g = lambda df: [
        (p/img, p/seg) for p in root.iterdir() if p.stem 
        in df.iloc[:, 0].apply(lambda x: Path(x).stem).values
    ]
    sets = map(g, (train_df, val_df, test_df))
    k = lambda s: RegistrationDataset(data=s)
    train_ds, val_ds, test_ds = map(k, sets)
    return train_ds, val_ds, test_ds
    
def make_msd_datasets(
    root='/mnt/qtim/data/registration/v3.2-midslice/'
         'MSD/Heart/Mono',
    use_256=False,
):
    size = str(256 if use_256 else 128)
    img=f'resize_image_{size}_2.npy'
    seg=f'resize_seg_{size}_2.npy'
    return make_datasets(root=root, img=img, seg=seg)

def make_idrid_datasets(
    root='/mnt/qtim/data/registration/v3.2-midslice/'
         'IDRID/retreived_2022_03_04/Retinal',
    use_256=False,
):
    size = str(256 if use_256 else 128)
    img=f'resize_image_{size}_0.npy'
    seg=f'resize_seg_{size}_0.npy'
    return make_datasets(root=root, img=img, seg=seg)
    
def make_t1mix_datasets(
    root='/mnt/qtim/data/registration/v3.2-midslice/'
         'T1mix/retrieved_2021_06_10/T1',
    use_256=False,
    plane=2, # 0, 1 or 2
):
    size = str(256 if use_256 else 128)
    img=f'resize_image_{size}_{plane}.npy'
    seg=f'resize_seg_{size}_{plane}.npy'
    return make_datasets(root=root, img=img, seg=seg)

def make_isbi_datasets(
    root='/mnt/qtim/data/registration/v3.2-midslice/'
         'ISBI/retrieved_2021_10_12/MRI',
    use_256=False,
    plane=2, # 0 or 2
):
    size = str(256 if use_256 else 128)
    img=f'resize_image_{size}_{plane}.npy'
    seg=f'resize_seg_{size}_{plane}.npy'
    return make_datasets(root=root, img=img, seg=seg)

def make_oasis_datasets(
    root='/mnt/qtim/data/registration/v3.2-midslice/'
         'OASIS/retrieved_2021_01_26/T1',
    use_256=False,
    plane=1, # 0, 1 or 2
):
    size = str(256 if use_256 else 128)
    img=f'resize_image_{size}_{plane}.npy'
    seg=f'resize_seg_{size}_{plane}.npy'
    return make_datasets(root=root, img=img, seg=seg)
    
def make_feta_datasets(
    root='/mnt/qtim/data/registration/v3.2-midslice/'
         'FeTA/retrieved_2022_02_16/MRI',
    use_256=False,
    plane=1, # 0, 1 or 2
):
    size = str(256 if use_256 else 128)
    img=f'resize_image_{size}_{plane}.npy'
    seg=f'resize_seg_{size}_{plane}.npy'
    return make_datasets(root=root, img=img, seg=seg)
    
def make_i2cvb_datasets(
    root='/mnt/qtim/data/registration/v3.2-midslice/'
         'I2CVB/retrieved_2022_03_08/MRI',
    use_256=False,
    plane=2, # 0 or 2
):
    size = str(256 if use_256 else 128)
    img=f'resize_image_{size}_{plane}.npy'
    seg=f'resize_seg_{size}_{plane}.npy'
    return make_datasets(root=root, img=img, seg=seg)

def make_spineweb_datasets(
    root='/mnt/qtim/data/registration/v3.2-midslice/'
         'SpineWeb/Dataset7/MR',
    use_256=False,
):
    size = str(256 if use_256 else 128)
    img=f'resize_image_{size}_0.npy'
    seg=f'resize_seg_{size}_0.npy'
    return make_datasets(root=root, img=img, seg=seg)

def make_braindev_datasets(
    root='/mnt/qtim/data/registration/v3.2-midslice/'
         'BrainDevelopment/HammersAtlasDatabase/T1',
    use_256=False,
    plane=2, # 0, 1, or 2
):
    size = str(256 if use_256 else 128)
    img=f'resize_image_{size}_{plane}.npy'
    seg=f'resize_seg_{size}_{plane}.npy'
    return make_datasets(root=root, img=img, seg=seg)

def make_wbc_datasets(
    root='/mnt/qtim/data/registration/v3.2-midslice/'
         'WBC/JTSC/EM',
    use_256=False,
):
    size = str(256 if use_256 else 128)
    img=f'resize_image_{size}_0.npy'
    seg=f'resize_seg_{size}_0.npy'
    return make_datasets(root=root, img=img, seg=seg)