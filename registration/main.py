from __future__ import annotations
import data, model, utils
import click
import pandas as pd
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from importlib import reload
import matplotlib.pyplot as plt; plt.style.use('seaborn-pastel'); plt.set_cmap('bone')
import matplotlib.image as mpimg
import neurite as ne
import numpy as np
import os; os.environ['VXM_BACKEND'] = 'pytorch'
import sys
from time import perf_counter
import torch
import torchvision
from tqdm import tqdm 
import random
import voxelmorph as vm
print(sys.version)
print('numpy version', np.__version__)
print('torch version', torch.__version__)
print(f'voxelmorph using {vm.py.utils.get_backend()} backend')


@click.command()
@click.option(
    '--root', '-r', default='/mnt/qtim/data/registration/v3.2-midslice', help='path to data directory', 
)
@click.option(
    '--output', '-o', default='experiments/debug', help='output to this folder', 
)
@click.option(
    '--use_256', '-256', is_flag=True, default=False, help='use higher resolution images'
)
@click.option(
    '--batch-size', '-bs', default=16, help='batch size for data loader', 
)
@click.option(
    '--learn-rate', '-lr', 'lr', default=1e-3, help='learning rate for optimizer', 
)
@click.option(
    '--epochs', '-e', default=2000, help='number of epochs for single models', 
)
@click.option(
    '--multi-epochs', '-me', default=8000, help='number of epochs for multi models', 
)
@click.option(
    '--debug', '-d', is_flag=True, default=False, help='run in debugging mode', 
)
def main(root, output, use_256, batch_size, lr, epochs, multi_epochs, debug):
    if debug:
        use_256 = False
        epochs=10
        multi_epochs=20
        
    idrid_train_ds, idrid_val_ds, idrid_test_ds = data.make_idrid_datasets(use_256=use_256)
    isbi_train_ds, isbi_val_ds, isbi_test_ds = data.make_isbi_datasets(use_256=use_256)
    feta_train_ds, feta_val_ds, feta_test_ds = data.make_feta_datasets(use_256=use_256)
    wbc_train_ds, wbc_val_ds, wbc_test_ds = data.make_wbc_datasets(use_256=use_256)
    t1mix_train_ds, t1mix_val_ds, t1mix_test_ds = data.make_t1mix_datasets(use_256=use_256)
    oasis_train_ds, oasis_val_ds, oasis_test_ds = data.make_oasis_datasets(use_256=use_256)
    i2cvb_test_ds = data.make_i2cvb_datasets(use_256=use_256, combine=True)
    spineweb_test_ds = data.make_spineweb_datasets(use_256=use_256, combine=True)
    braindev_test_ds = data.make_braindev_datasets(use_256=use_256, combine=True)
    msd_test_ds = data.make_msd_datasets(use_256=use_256, combine=True)

    single_train_1 = deepcopy(wbc_train_ds)
    single_val_1 = deepcopy(wbc_val_ds)
    single_test_1 = deepcopy(wbc_test_ds)

    single_train_2 = deepcopy(t1mix_train_ds)
    single_val_2 = deepcopy(t1mix_val_ds)
    single_test_2 = deepcopy(t1mix_test_ds)

    single_train_3 = deepcopy(oasis_train_ds)
    single_val_3 = deepcopy(oasis_val_ds)
    single_test_3 = deepcopy(oasis_test_ds)

    t1mix_train_ds.set_indexer(list(range(90)))
    wbc_train_ds.set_indexer(list(range(90)))
    multi_train_1 = data.MultiRegistrationDataset([idrid_train_ds, isbi_train_ds, wbc_train_ds, t1mix_train_ds])

    t1mix_train_ds.set_indexer(list(range(50)))
    oasis_train_ds.set_indexer(list(range(50)))
    wbc_train_ds.set_indexer(list(range(50)))
    multi_train_2 = data.MultiRegistrationDataset([idrid_train_ds, isbi_train_ds, wbc_train_ds, t1mix_train_ds, oasis_train_ds, feta_train_ds])

    t1mix_val_ds.set_indexer(list(range(30)))
    wbc_val_ds.set_indexer(list(range(30)))
    multi_val_1 = data.MultiRegistrationDataset([idrid_val_ds, t1mix_val_ds, wbc_val_ds, isbi_val_ds])

    t1mix_val_ds.set_indexer(list(range(15)))
    oasis_val_ds.set_indexer(list(range(15)))
    wbc_val_ds.set_indexer(list(range(15)))
    multi_val_2 = data.MultiRegistrationDataset([idrid_val_ds, t1mix_val_ds, wbc_val_ds, isbi_val_ds, oasis_val_ds, feta_val_ds])

    t1mix_test_ds.set_indexer(list(range(30)))
    wbc_test_ds.set_indexer(list(range(30))) 
    multi_test_1 = data.MultiRegistrationDataset([idrid_test_ds, t1mix_test_ds, wbc_test_ds, isbi_test_ds])

    t1mix_test_ds.set_indexer(list(range(15)))
    oasis_test_ds.set_indexer(list(range(15))) 
    wbc_test_ds.set_indexer(list(range(15))) 
    multi_test_2 = data.MultiRegistrationDataset([idrid_test_ds, t1mix_test_ds, wbc_test_ds, isbi_test_ds, oasis_test_ds, feta_test_ds])
    
    if debug:
        print(
            'training sizes:\t',
            f'{len(single_train_1)=}', 
            f'{len(single_train_2)=}', 
            f'{len(single_train_3)=}', 
            f'{len(multi_train_1)=}', 
            f'{len(multi_train_2)=}', 
        )


    train_dl_1 = data.make_dataloader(single_train_1, batch_size=batch_size, shuffle=True)
    train_dl_2 = data.make_dataloader(single_train_2, batch_size=batch_size, shuffle=True)
    train_dl_3 = data.make_dataloader(single_train_3, batch_size=batch_size, shuffle=True)
    train_dl_4 = data.make_dataloader(multi_train_1, batch_size=batch_size, shuffle=True)
    train_dl_5 = data.make_dataloader(multi_train_2, batch_size=batch_size, shuffle=True)

    val_dl_1 = data.make_dataloader(single_val_1, batch_size=batch_size)
    val_dl_2 = data.make_dataloader(single_val_2, batch_size=batch_size)
    val_dl_3 = data.make_dataloader(single_val_3, batch_size=batch_size)
    val_dl_4 = data.make_dataloader(multi_val_1, batch_size=batch_size)
    val_dl_5 = data.make_dataloader(multi_val_2, batch_size=batch_size)

    test_dl_1 = data.make_dataloader(single_test_1, batch_size=batch_size)
    test_dl_2 = data.make_dataloader(single_test_2, batch_size=batch_size)
    test_dl_3 = data.make_dataloader(single_test_3, batch_size=batch_size)
    test_dl_4 = data.make_dataloader(multi_test_1, batch_size=batch_size)
    test_dl_5 = data.make_dataloader(multi_test_2, batch_size=batch_size)

    ext_dl_1 = data.make_dataloader(i2cvb_test_ds, batch_size=1)
    ext_dl_2 = data.make_dataloader(spineweb_test_ds, batch_size=1)
    ext_dl_3 = data.make_dataloader(braindev_test_ds, batch_size=1)
    ext_dl_4 = data.make_dataloader(msd_test_ds, batch_size=1)
    
    if use_256:
        inshape = (256, 256)
    else:
        inshape = (128, 128)
        
    model1 = model.Model(inshape).to('cuda')
    model2 = model.Model(inshape).to('cuda')
    model3 = model.Model(inshape).to('cuda')
    model4 = model.Model(inshape).to('cuda')
    model5 = model.Model(inshape).to('cuda')
    
    opt1 = torch.optim.Adam(model1.parameters(), lr=lr)
    opt2 = torch.optim.Adam(model2.parameters(), lr=lr)
    opt3 = torch.optim.Adam(model3.parameters(), lr=lr)
    opt4 = torch.optim.Adam(model4.parameters(), lr=lr)
    opt5 = torch.optim.Adam(model5.parameters(), lr=lr)
    
    history_1 = train(model1, opt1, train_dl_1, val_dl_1, epochs=epochs)
    history_2 = train(model2, opt2, train_dl_2, val_dl_2, epochs=epochs)
    history_3 = train(model3, opt3, train_dl_3, val_dl_3, epochs=epochs)
    history_4 = train(model4, opt4, train_dl_4, val_dl_4, epochs=multi_epochs)
    history_5 = train(model5, opt5, train_dl_5, val_dl_5, epochs=multi_epochs)

if __name__ == '__main__':
    main()