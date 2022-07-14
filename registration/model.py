from __future__ import annotations
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import torch
import voxelmorph as vm


def loss_func(moving, fixed, displacement, weight=0.05):
    mse = vm.losses.MSE()
    grad = vm.losses.Grad('l2')
    mse_loss = mse.loss(fixed, moving)
    grad_loss = grad.loss(displacement)
    return mse_loss + weight * grad_loss


def dice_func(a, b, eps=1e-8):
    intersection = 2 * torch.logical_and(a, b).sum()
    union = a.sum() + b.sum() + eps
    return intersection / union

        
class Model(torch.nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int] | tuple[int, int, int],
        input_channels: int = 2,
    ):
        super().__init__()
        self.image_size = image_size
        self.input_channels= input_channels
        self.ndim = len(image_size)
        self.unet = vm.networks.Unet(inshape=image_size, infeats=input_channels)
        self.flow = getattr(torch.nn, f'Conv{self.ndim}d')(16, 2, 3, padding=1)
        self.spatial_transformer = vm.layers.SpatialTransformer(image_size)
        
    def transform(self, x: torch.Tensor, warp: torch.Tensor, mode='bilinear') -> torch.Tensor:
        self.spatial_transformer.mode = mode
        x = self.spatial_transformer(x, warp)
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # img, seg = x[:2], x[
        displacement = self.flow(self.unet(x))
        moving_image = x[:, 0].unsqueeze(1)
        moved_image = self.transform(moving_image, displacement)
        return moved_image, displacement
    
    
def iterate(
    model,
    loader,
    optimizer=None,
    training: bool = True,
):
    average_loss = 0
    average_dice = 0
    if training:
        model.train()
    else:
        model.eval()
    # for i, images in tqdm(enumerate(loader)):
    for i, (images, segmentations) in enumerate(loader):
        images = images.to('cuda')
        segmentations = segmentations.to('cuda')
        moved_images, displacements = model(images)
        moved_segmentations = model.transform(segmentations[:, 0].unsqueeze(1), displacements, mode='nearest')
        fixed_images = images[:, 1].unsqueeze(1)
        fixed_segmentations = segmentations[:, 1].unsqueeze(1)
        loss = loss_func(moved_images, fixed_images, displacements)
        dice = dice_func(moved_segmentations, fixed_segmentations)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        average_loss += loss.detach().item()
        average_dice += dice.detach().item()
    average_loss /= len(loader.dataset)
    average_dice /= len(loader.dataset)
    return average_loss, average_dice
    
def train(
    model,
    optimizer,
    train_loader,
    val_loader,
    epochs: int = 10,
    verbose: bool = True,
):
    history = defaultdict(list)
    for i in tqdm(range(epochs)):
        train_loss, train_dice = iterate(model, train_loader, optimizer)
        history[f'train_loss'].append(train_loss)
        history[f'train_dice'].append(train_dice)
        val_loss, val_dice = iterate(model, val_loader, training=False)
        history[f'val_loss'].append(val_loss)
        history[f'val_dice'].append(val_dice)
        if verbose and (i+1) % 500 == 0: print(f'epoch {i} finished')
    return history


def evaluate(model, dl):
    detach = lambda x: x.detach().cpu().numpy()
    err_loss = 0
    err_dice = 0
    for i, (img, seg) in enumerate(dl):
        moved_img, warp = model(img.cuda())
        moved_seg = model.transform(seg[:, 0].unsqueeze(1).cuda(), warp.cuda(), mode='nearest')
        fixed_img = img[:, 1].unsqueeze(1).cuda()
        fixed_seg = seg[:, 1].unsqueeze(1).cuda()
        loss = loss_func(moved_img, fixed_img, warp)
        dice = dice_func(moved_seg, fixed_seg)
        err_loss += detach(loss)
        err_dice += detach(dice)
        
    return err_loss / len(dl.dataset),  err_dice / len(dl.dataset)
        