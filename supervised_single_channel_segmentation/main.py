import os
import random
import glob
from natsort import natsorted
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import kornia
import cv2
import albumentations as AB

train_image_dir = ['/u01/data/iXrayCT_COVID/data_resized/train/xr2lung2/NLMMC/neg/images/']
train_label_dir = ['/u01/data/iXrayCT_COVID/data_resized/train/xr2lung2/NLMMC/neg/labels/']
val_image_dir = ['/u01/data/iXrayCT_COVID/data_resized/test/xr2lung2/JSRT/neg/images/']
val_label_dir = ['/u01/data/iXrayCT_COVID/data_resized/test/xr2lung2/JSRT/neg/labels/']

class CustomNativeDataset(Dataset):
    def __init__(self, 
        imagedir, 
        labeldir, 
        split='train',
        size=10000, 
        transforms=None
    ):
        self.size = size
        self.is_train = True if split=='train' else False
        self.imagedir = imagedir 
        self.labeldir = labeldir 
        
        self.imagefiles = [glob.glob(os.path.join(folder, '*.*')) for folder in self.imagedir]
        self.labelfiles = [glob.glob(os.path.join(folder, '*.*')) for folder in self.labeldir]
        self.imagefiles = natsorted([item for sublist in self.imagefiles for item in sublist])
        self.labelfiles = natsorted([item for sublist in self.labelfiles for item in sublist])
        self.transforms = transforms
        assert len(self.imagefiles) == len(self.labelfiles)
        # print(self.imagefiles)
    def __len__(self):
        if self.size > len(self.imagefiles) or self.size is None: 
            return len(self.imagefiles)
        else:
            return self.size
 
 
    def __getitem__(self, idx):
        # print(self.imagefiles[idx], self.labelfiles[idx])
        image = cv2.imread(self.imagefiles[idx], cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(self.labelfiles[idx], cv2.IMREAD_GRAYSCALE)
        assert image.shape[:-2] == label.shape[:-2]
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']

        return kornia.image_to_tensor(image).float(), \
               kornia.image_to_tensor(label).float()

#
# Custom Data Module
#
class CustomDataModule(pl.LightningDataModule):
    def __init__(self, arg):
        super(CustomDataModule, self).__init__()
        self.arg = arg
        self.batch_size = self.arg.batch_size
        
    def prepare_data(self):
        pass

    def train_dataloader(self):
        train_tfm = AB.Compose([
            AB.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            AB.Resize(height=self.arg.shape, width=self.arg.shape, p=1.0), 
            AB.CropNonEmptyMaskIfExists(height=480, width=480, p=0.8), 
            AB.RandomScale(scale_limit=(0.8, 1.2), p=0.8),
            AB.Equalize(p=0.8),
            # AB.CLAHE(p=0.8),
            AB.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
            AB.RandomGamma(gamma_limit=(80, 120), p=0.8),
            AB.GaussianBlur(p=0.8),
            AB.GaussNoise(p=0.8),
            AB.Resize(width=self.arg.shape, height=self.arg.shape, p=1.0),
            # AB.ToTensor(),
        ])
        train_ds = CustomNativeDataset(imagedir=train_image_dir, 
                                       labeldir=train_label_dir, 
                                       split='train', 
                                       transforms=train_tfm)
        train_loader = DataLoader(train_ds, 
            num_workers=self.arg.num_workers, 
            batch_size=self.arg.batch_size, 
            pin_memory=True, 
            shuffle=True
        )

        return train_loader


    def val_dataloader(self):
        val_tfm = AB.Compose([
            AB.Equalize(p=1.0),
            AB.Resize(width=self.arg.shape, height=self.arg.shape, p=1.0),
        ])
        val_ds = CustomNativeDataset(imagedir=val_image_dir, 
                                     labeldir=val_label_dir, 
                                     split='val', 
                                     transforms=val_tfm)
        val_loader = DataLoader(val_ds, 
            num_workers=self.arg.num_workers, 
            batch_size=self.arg.batch_size, 
            pin_memory=True, 
            shuffle=False
        )
        return val_loader

    def test_dataloader(self):
        test_tfm = AB.Compose([
            AB.Equalize(p=1.0),
            AB.Resize(width=self.arg.shape, height=self.arg.shape, p=1.0),
        ])
        test_ds = CustomNativeDataset(imagedir=test_image_dir, 
                                      labeldir=test_label_dir, 
                                      split='test', 
                                      transforms=test_tfm)
        test_loader = DataLoader(test_ds, 
            num_workers=self.arg.num_workers, 
            batch_size=self.arg.batch_size, 
            pin_memory=True, 
            shuffle=False
        )
        return test_loader

#
# Custom Lightning Model
#
class UNet(nn.Module):
    """
    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_
    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox
    Implemented by:
        - `Annika Brundyn <https://github.com/annikabrundyn>`_
        - `Akshay Kulkarni <https://github.com/akshaykvnit>`_
    Args:
        num_classes: Number of output classes required
        input_channels: Number of channels in input images (default 3)
        num_layers: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        bilinear: Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
    """

    def __init__(
            self,
            num_classes: int,
            input_channels: int = 3,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False
    ):
        super().__init__()
        self.num_layers = num_layers

        layers = [DoubleConv(input_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)
        self.act = nn.Sigmoid()
    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1:self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        out = self.layers[-1](xi[-1])
        out = self.act(out)
        return out


class DoubleConv(nn.Module):
    """
    [ Conv2d => BatchNorm (optional) => ReLU ] x 2
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Downscale with MaxPool => DoubleConvolution block
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """
    Upsampling (by either bilinear interpolation or transpose convolutions)
    followed by concatenation of feature map from contracting path, followed by DoubleConv.
    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class CustomLightningModel(pl.LightningModule):
    """docstring for ClassName"""

    def __init__(self, arg):
        super(CustomLightningModel, self).__init__()
        self.arg = arg
        self.net = UNet(input_channels=1, num_classes=1)
        self.learning_rate = arg.lr
        self.example_input_array = torch.randn(8, 1, 512, 512)
        

    def forward(self, x):
        return self.net(x)

    def loss(self, estim, label):
        loss = torch.nn.functional.mse_loss(estim, label, reduce=True)
        return loss

    def training_step(self, batch, batch_idx):
        input, label = batch
        input, label = input / 255.0, label / 255.0
        estim = self.net(input)
        loss = self.loss(estim, label)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)

        # Visualization
        if batch_idx==0:
            tensorboard = self.logger.experiment
            viz = torch.cat([input, label, estim], dim=-1)#[:8]
            grd = torchvision.utils.make_grid(viz, nrow=2, padding=0)
            tensorboard.add_image('train_viz', grd, self.current_epoch)
        return loss

    # def training_step_end(self, outputs):
    #     return outputs

    # def training_epoch_end(self, outputs) -> None:
    #     torch.stack([x["train_loss"] for x in outputs]).mean()

    def validation_step(self, batch, batch_idx):
        input, label = batch
        input, label = input / 255.0, label / 255.0
        estim = self.net(input)
        loss = self.loss(estim, label)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)

        # Visualization
        if batch_idx==0:
            tensorboard = self.logger.experiment
            viz = torch.cat([input, label, estim], dim=-1)#[:8]
            grd = torchvision.utils.make_grid(viz, nrow=2, padding=0)
            tensorboard.add_image('val_viz', grd, self.current_epoch)
        return {'val_loss': loss}
    
    # def validation_step_end(self, outputs):
    #     return outputs

    def validation_epoch_end(self, outputs) -> None:
        torch.stack([x['val_loss'] for x in outputs]).mean()

    def test_step(self, batch, batch_idx):
        input, label = batch
        input, label = input / 255.0, label / 255.0
        estim = self.net(input)
        loss = self.loss(estim, label)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        return {'test_loss': loss}
    
    # def test_step_end(self, outputs):
    #     return outputs

    def test_epoch_end(self, outputs) -> None:
        torch.stack([x["test_loss"] for x in outputs]).mean()

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]
       

def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    if args.distributed_backend == 'ddp':
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / max(1, args.gpus))
        args.workers = int(args.workers / max(1, args.gpus))

    model = CustomLightningModel(args) #CustomLightningModel(**vars(args))
    dm = CustomDataModule(args)

    if args.load:
        trainer = pl.Trainer(resume_from_checkpoint=args.load).from_argparse_args(args)
    else:
        trainer = pl.Trainer.from_argparse_args(args)

    if args.test:
        # trainer.test(model)
        result = trainer.test(test_dataloaders=dm.test_dataloader())
        print(result)
    else:
        # trainer.fit(model)
        trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())
        # trainer.fit(model, dm)

def run_cli():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, help="path where dataset is stored")
    parser.add_argument("--gpus", type=str, default='0,1', help="number of available GPUs")
    parser.add_argument("--load", action='store_true')
    parser.add_argument('--distributed-backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'), help='supports three options dp, ddp, ddp2')
    parser.add_argument('--use_amp', action='store_true', help='if true uses 16 bit precision')
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--num_workers", type=int, default=8, help="size of the workers")
    parser.add_argument("--grad_batches", type=int, default=1, help="number of batches to accumulate")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--shape", type=int, default=512, help="shape of the images")
    parser.add_argument("--seed", type=int, default=2020, help="reproducibility")
    parser.add_argument("--test", action='store_true', help="run test")

    parser.set_defaults(
        profiler=True,
        deterministic=True,
        max_epochs=101,
    )
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
