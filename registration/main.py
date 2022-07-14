from __future__ import annotations
import data, model, utils
import click
import pandas as pd
from collections import defaultdict
import json
from copy import deepcopy
from pathlib import Path
from importlib import reload
from scipy import signal
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
    '--num-workers', '-nw', default=8, help='number of workers for data loading', 
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
def train(root, output, use_256, batch_size, num_workers, lr, epochs, multi_epochs, debug):
    if debug:
        use_256 = False
        epochs=2
        multi_epochs=2
        # output = None
        
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

    dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers)
    train_dl_1 = data.make_dataloader(single_train_1,  shuffle=True, **dl_kwargs)
    train_dl_2 = data.make_dataloader(single_train_2, shuffle=True, **dl_kwargs)
    train_dl_3 = data.make_dataloader(single_train_3, shuffle=True, **dl_kwargs)
    train_dl_4 = data.make_dataloader(multi_train_1, shuffle=True, **dl_kwargs)
    train_dl_5 = data.make_dataloader(multi_train_2, shuffle=True, **dl_kwargs)

    val_dl_1 = data.make_dataloader(single_val_1, **dl_kwargs)
    val_dl_2 = data.make_dataloader(single_val_2, **dl_kwargs)
    val_dl_3 = data.make_dataloader(single_val_3, **dl_kwargs)
    val_dl_4 = data.make_dataloader(multi_val_1, **dl_kwargs)
    val_dl_5 = data.make_dataloader(multi_val_2, **dl_kwargs)

    test_dl_1 = data.make_dataloader(single_test_1, **dl_kwargs)
    test_dl_2 = data.make_dataloader(single_test_2, **dl_kwargs)
    test_dl_3 = data.make_dataloader(single_test_3, **dl_kwargs)
    test_dl_4 = data.make_dataloader(multi_test_1, **dl_kwargs)
    test_dl_5 = data.make_dataloader(multi_test_2, **dl_kwargs)

    ext_dl_1 = data.make_dataloader(i2cvb_test_ds, batch_size=1, num_workers=num_workers)
    ext_dl_2 = data.make_dataloader(spineweb_test_ds, batch_size=1, num_workers=num_workers)
    ext_dl_3 = data.make_dataloader(braindev_test_ds, batch_size=1, num_workers=num_workers)
    ext_dl_4 = data.make_dataloader(msd_test_ds, batch_size=1, num_workers=num_workers)
    
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
    
    history_1 = model.train(model1, opt1, train_dl_1, val_dl_1, epochs=epochs)
    history_2 = model.train(model2, opt2, train_dl_2, val_dl_2, epochs=epochs)
    history_3 = model.train(model3, opt3, train_dl_3, val_dl_3, epochs=epochs)
    history_4 = model.train(model4, opt4, train_dl_4, val_dl_4, epochs=multi_epochs)
    history_5 = model.train(model5, opt5, train_dl_5, val_dl_5, epochs=multi_epochs)
    
    if output is not None:
        output = Path(output)
        output.mkdir(exist_ok=True)
        
        config = dict(
            root=root, 
            output=output,
            use_256=use_256, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            lr=lr, 
            epochs=epochs, 
            multi_epochs=multi_epochs, 
        )
        with open(output/'config.json', 'w') as f: 
            f.write(json.dumps(config, default=str))
            
        history_dir = output / 'history'
        history_dir.mkdir(exist_ok=True)
        with open(history_dir / 'model_1.json', 'w') as f: 
            f.write(json.dumps(history_1, default=float))
        with open(history_dir / 'model_2.json', 'w') as f: 
            f.write(json.dumps(history_2, default=float))
        with open(history_dir / 'model_3.json', 'w') as f: 
            f.write(json.dumps(history_3, default=float))
        with open(history_dir / 'model_4.json', 'w') as f: 
            f.write(json.dumps(history_4, default=float))
        with open(history_dir / 'model_5.json', 'w') as f: 
            f.write(json.dumps(history_5, default=float))
            
        model_dir = output / 'models'
        model_dir.mkdir(exist_ok=True)
        torch.save(model1.state_dict(), model_dir / 'model.pth')
        torch.save(model2.state_dict(), model_dir / 'model.pth')
        torch.save(model3.state_dict(), model_dir / 'model.pth')
        torch.save(model4.state_dict(), model_dir / 'model.pth')
        torch.save(model5.state_dict(), model_dir / 'model.pth')
                   
        figure_dir = output / 'figures'
        figure_dir.mkdir(exist_ok=True)
        
        k = 32
        window = signal.windows.triang(k)
        avg = lambda sig: signal.convolve(sig, window, mode='valid')
        fs=32
        lw=4
        fig, ax = plt.subplots(ncols=1, nrows=5, figsize=(24, 24))
        ax[0].plot(avg(history_1['train_dice']), label='train_1',  lw=lw, c='C0')
        ax[0].plot(avg(history_1['val_dice']), label='val_1', ls='--', lw=lw, c='C0')
        ax[1].plot(avg(history_2['train_dice']), label='train_2',  lw=lw, c='C1')
        ax[1].plot(avg(history_2['val_dice']), label='val_2', ls='--', lw=lw, c='C1')
        ax[2].plot(avg(history_3['train_dice']), label='train_3',  lw=lw, c='C2')
        ax[2].plot(avg(history_3['val_dice']), label='val_3', ls='--', lw=lw, c='C2')
        ax[3].plot(avg(history_4['train_dice']), label='train_4',  lw=lw, c='C3')
        ax[3].plot(avg(history_4['val_dice']), label='val_4', ls='--', lw=lw, c='C3')
        ax[4].plot(avg(history_5['train_dice']), label='train_5',  lw=lw, c='C3')
        ax[4].plot(avg(history_5['val_dice']), label='val_5', ls='--', lw=lw, c='C3')
        ax[0].tick_params(axis='both', which='major', labelsize=fs, pad=15)
        ax[1].tick_params(axis='both', which='major', labelsize=fs, pad=15)
        ax[2].tick_params(axis='both', which='major', labelsize=fs, pad=15)
        ax[3].tick_params(axis='both', which='major', labelsize=fs, pad=15)
        ax[4].tick_params(axis='both', which='major', labelsize=fs, pad=15)
        ax[0].set_title('single model 1')
        ax[1].set_title('single model 2')
        ax[2].set_title('single model 3')
        ax[3].set_title('multi model 1')
        ax[4].set_title('multi model 2')
        ax[0].set_xlabel('epochs', fontsize=fs)
        ax[1].set_xlabel('epochs', fontsize=fs)
        ax[2].set_xlabel('epochs', fontsize=fs)
        ax[3].set_xlabel('epochs', fontsize=fs)
        ax[4].set_xlabel('epochs', fontsize=fs)
        ax[0].set_ylabel('dice', fontsize=fs)
        ax[1].set_ylabel('dice', fontsize=fs)
        ax[2].set_ylabel('dice', fontsize=fs)
        ax[3].set_ylabel('dice', fontsize=fs)
        ax[4].set_ylabel('dice', fontsize=fs)
        plt.tight_layout(w_pad=4)
        plt.legend(fontsize=fs, ncol=3, bbox_to_anchor=(0.60, -0.30))
        plt.savefig(figure_dir / 'midslice-dice.png')
        plt.show()
        
    m1_ext_1_loss, m1_ext_1_dice = model.evaluate(model1, ext_dl_1)
    m2_ext_1_loss, m2_ext_1_dice = model.evaluate(model2, ext_dl_1)
    m3_ext_1_loss, m3_ext_1_dice = model.evaluate(model3, ext_dl_1)
    m4_ext_1_loss, m4_ext_1_dice = model.evaluate(model4, ext_dl_1)
    m5_ext_1_loss, m5_ext_1_dice = model.evaluate(model5, ext_dl_1)
    m1_ext_2_loss, m1_ext_2_dice = model.evaluate(model1, ext_dl_2)
    m2_ext_2_loss, m2_ext_2_dice = model.evaluate(model2, ext_dl_2)
    m3_ext_2_loss, m3_ext_2_dice = model.evaluate(model3, ext_dl_2)
    m4_ext_2_loss, m4_ext_2_dice = model.evaluate(model4, ext_dl_2)
    m5_ext_2_loss, m5_ext_2_dice = model.evaluate(model5, ext_dl_2)
    m1_ext_3_loss, m1_ext_3_dice = model.evaluate(model1, ext_dl_3)
    m2_ext_3_loss, m2_ext_3_dice = model.evaluate(model2, ext_dl_3)
    m3_ext_3_loss, m3_ext_3_dice = model.evaluate(model3, ext_dl_3)
    m4_ext_3_loss, m4_ext_3_dice = model.evaluate(model4, ext_dl_3)
    m5_ext_3_loss, m5_ext_3_dice = model.evaluate(model5, ext_dl_3)
    m1_ext_4_loss, m1_ext_4_dice = model.evaluate(model1, ext_dl_4)
    m2_ext_4_loss, m2_ext_4_dice = model.evaluate(model2, ext_dl_4)
    m3_ext_4_loss, m3_ext_4_dice = model.evaluate(model3, ext_dl_4)
    m4_ext_4_loss, m4_ext_4_dice = model.evaluate(model4, ext_dl_4)
    m5_ext_4_loss, m5_ext_4_dice = model.evaluate(model5, ext_dl_4)

    print('\t\tsing1   sing2   mult1   mult2')
    print(
        'I2CVB'.ljust(10),
        *map(lambda x: f'{x:.3f}', (
        m1_ext_1_dice, 
        m2_ext_1_dice, 
        m3_ext_1_dice, 
        m4_ext_1_dice, 
        m5_ext_1_dice, 
    )), sep='\t')
    print(
        'SpineWeb'.ljust(10),
        *map(lambda x: f'{x:.3f}', (
        m1_ext_2_dice, 
        m2_ext_2_dice, 
        m3_ext_2_dice, 
        m4_ext_2_dice, 
        m5_ext_2_dice, 
    )), sep='\t')
    print(
        'BrainDev'.ljust(10),
        *map(lambda x: f'{x:.3f}', (
        m1_ext_3_dice, 
        m2_ext_3_dice, 
        m3_ext_3_dice, 
        m4_ext_3_dice, 
        m5_ext_3_dice, 
    )), sep='\t')
    print(
        'WBC\t',
        *map(lambda x: f'{x:.3f}', (
        m1_ext_4_dice, 
        m2_ext_4_dice, 
        m3_ext_4_dice, 
        m4_ext_4_dice, 
        m5_ext_4_dice, 
    )), sep='\t')
    
if __name__ == '__main__':
    train()