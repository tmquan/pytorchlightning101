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
from pytorch_lightning.callbacks import ModelCheckpoint

import kornia
import cv2
import albumentations as AB

import pandas as pd
import torchxrayvision as xrv
from skimage.io import imread, imsave
import sklearn.metrics

from data import CustomDataset
#
# Custom Data Module
#
thispath = os.path.dirname(os.path.realpath(__file__))


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, arg):
        super(CustomDataModule, self).__init__()
        self.arg = arg
        self.batch_size = self.arg.batch_size
        
    def prepare_data(self):
        pass

    def train_dataloader(self):
        train_tfm = AB.Compose([
            AB.ImageCompression(quality_lower=98, quality_upper=100, p=1.0), 
            # AB.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=1.0),
            # AB.Resize(height=self.arg.shape, width=self.arg.shape, p=1.0), 
            # AB.CropNonEmptyMaskIfExists(height=480, width=480, p=0.8), 
            # AB.RandomScale(scale_limit=(0.8, 1.2), p=0.5),
            AB.Equalize(p=0.5),
            # AB.CLAHE(p=0.5),
            AB.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            AB.RandomGamma(gamma_limit=(80, 120), p=0.5),
            AB.GaussianBlur(p=0.5),
            AB.GaussNoise(p=0.5),
            AB.Resize(width=self.arg.shape, height=self.arg.shape, p=1.0),
            # AB.ToTensor(),
        ])
        train_ds = CustomDataset(imgpath="/u01/data/COVID_Data_Relabel/data/",
                                 csvpath="train_covid_quan.csv",
                                 is_train="train",
                                 transform=train_tfm)
        train_loader = DataLoader(train_ds, 
            num_workers=self.arg.num_workers, 
            batch_size=self.arg.batch_size, 
            pin_memory=True, 
            shuffle=True
        )

        return train_loader


    def val_dataloader(self):
        val_tfm = AB.Compose([
            # AB.Equalize(p=1.0),
            AB.Resize(width=self.arg.shape, height=self.arg.shape, p=1.0),
        ])
        val_ds = CustomDataset(imgpath="/u01/data/COVID_Data_Relabel/data/",
                               csvpath="valid_covid_quan.csv",
                               is_train="valid",
                               transform=val_tfm)
        val_loader = DataLoader(val_ds, 
            num_workers=self.arg.num_workers, 
            batch_size=self.arg.batch_size, 
            pin_memory=True, 
            shuffle=False
        )

        return val_loader

    def test_dataloader(self):
        test_tfm = AB.Compose([
            # AB.Equalize(p=1.0),
            AB.Resize(width=self.arg.shape, height=self.arg.shape, p=1.0),
        ])
        test_ds = CustomDataset(imgpath="/u01/data/COVID_Data_Relabel/data/",
                                csvpath="valid_covid_quan.csv",
                                is_train="test",
                                transform=test_tfm)
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
            num_classes: int = 1,
            input_channels: int = 3,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = True
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
  
        self.learning_rate = arg.lr
        self.example_input_array = torch.randn(8, 1, self.arg.shape, self.arg.shape)
        self.average_type = 'binary'

        self.generatorAB, self.generatorBA = self.init_generator()
        self.discriminator = self.init_discriminator()
        self.classifier = self.init_classifier()
        self.pathologies = ['COVID']

    def forward(self, x):
        out = self.classifier(x)
        return out

    def configure_optimizers(self):
        d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=self.arg.lr)
        g_optim = torch.optim.Adam([model.parameters() for model in list(self.generatorAB, self.generatorBA)],
                                   lr=self.arg.lr)
        c_optim = torch.optim.Adam(self.classifier.parameters(), lr=self.arg.lr)

        d_sched = torch.optim.lr_scheduler.CosineAnnealingLR(d_optim, T_max=10)
        g_sched = torch.optim.lr_scheduler.CosineAnnealingLR(g_optim, T_max=10)
        c_sched = torch.optim.lr_scheduler.CosineAnnealingLR(c_optim, T_max=10)

        return [d_optim, g_optim, c_optim], [d_sched, g_sched, c_sched]

    def init_discriminator(self):
        discriminator = xrv.models.DenseNet(weights='all')
        discriminator.classifier = nn.Sequential(
                nn.Linear(in_features=1024, out_features=1, bias=True),
                # nn.Sigmoid()
        )
        return discriminator

    def init_generator(self):
        generatorAB = UNet(input_channels=1, num_classes=1)
        generatorBA = UNet(input_channels=1, num_classes=1)
        return generatorAB, generatorBA

    def init_classifier(self):
        classifier = xrv.models.DenseNet(weights='all')
        classifier.classifier = nn.Sequential(
                nn.Linear(in_features=1024, out_features=1, bias=True),
                # nn.Sigmoid()
        )
        return classifier

    def generator_loss(self, p_fake):
        b = p_fake.size(0)
        y = torch.ones(b, 1, device=self.device)
        # ground truth result (ie: all real)
        g_loss = F.binary_cross_entropy_with_logits(p_fake, y)
        return g_loss


    def discriminator_loss(self, p_real, p_fake):
        # train discriminator on real
        b = p_real.size(0)
        y_real = torch.ones(b, 1, device=self.device)
        d_real_loss = F.binary_cross_entropy_with_logits(p_real, y_real)

        b = p_fake.size(0)
        y_fake = torch.zeros(b, 1, device=self.device)
        d_fake_loss = F.binary_cross_entropy_with_logits(p_fake, y_fake)

        # gradient backprop & optimize ONLY D's parameters
        d_loss = d_real_loss + d_fake_loss
        return d_loss

    def classifier_loss(self, estim, label):
        # estim = self.classifier(image)
        c_loss = F.binary_cross_entropy_with_logits(estim, label)
        return c_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        prefix = 'train'
        imageA, labelA, imageB, labelB = batch
        imageA, imageB = imageA / 255.0, imageB / 255.0

        # Generator
        imageAB = self.generatorAB(imageA)
        imageBA = self.generatorBA(imageB)

        imageABA = self.generatorBA(imageAB)
        imageBAB = self.generatorAB(imageBA)

        # Discriminator
        probaA = self.discriminator(imageA)
        probaB = self.discriminator(imageB)

        probaAB = self.discriminator(imageAB)
        probaBA = self.discriminator(imageBA)

        # Classifier
        estimA = self.classifier(imageA)
        estimB = self.classifier(imageB)

        estimAB = self.classifier(imageAB)
        estimBA = self.classifier(imageBA)

        # Visualization
        if batch_idx % 20 == 0: # and self.current_epoch==0:
            tensorboard = self.logger.experiment
            vizA = torch.cat([imageA, imageAB, imageABA, 20*torch.abs(imageA-imageAB)], dim=-1)
            vizB = torch.cat([imageB, imageBA, imageBAB, 20*torch.abs(imageB-imageBA)], dim=-1)
            viz = torch.cat([vizA, vizB], dim=-2)
            grd = torchvision.utils.make_grid(viz, nrow=1, padding=0)
            tensorboard.add_image(f'{prefix}_viz', grd, self.current_epoch)

        d_loss = self.discriminator_loss(probaA, probaAB) \
               + self.discriminator_loss(probaB, probaBA)            
        self.log('d_loss', d_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        g_loss = self.generator_loss(probaAB) \
               + self.generator_loss(probaBA) 
        r_loss = nn.L1Loss()(imageA, imageABA) \
               + nn.L1Loss()(imageB, imageBAB)
        self.log('g_loss', g_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('r_loss', r_loss, on_step=True, on_epoch=True, prog_bar=True)

        c_loss = self.classifier_loss(estimA, labelA) \
               + self.classifier_loss(estimB, labelB) \
               + self.classifier_loss(estimAB, labelB) \
               + self.classifier_loss(estimBA, labelA)
        self.log('c_loss', c_loss, on_step=True, on_epoch=True, prog_bar=True)

        if optimizer_idx==0: #d
            return d_loss

        elif optimizer_idx==1: #g
            return g_loss + 1e2*r_loss

        elif optimizer_idx==2: #c
            return c_loss 

    def validation_step(self, batch, batch_idx):
        prefix = 'val'
        imageA, labelA, imageB, labelB = batch
        imageA, imageB = imageA / 255.0, imageB / 255.0
        estimA = self.classifier(imageA)
        # print(estim.shape, label.shape)
        loss = self.classifier_loss(estimA, labelA)
        # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        # Visualization
        imageAB = self.generatorAB(imageA)
        imageBA = self.generatorBA(imageB)

        imageABA = self.generatorBA(imageAB)
        imageBAB = self.generatorAB(imageBA)

        if batch_idx % 20 == 0: # and self.current_epoch==0:
            tensorboard = self.logger.experiment
            vizA = torch.cat([imageA, imageAB, imageABA, 20*torch.abs(imageA-imageAB)], dim=-1)
            vizB = torch.cat([imageB, imageBA, imageBAB, 20*torch.abs(imageB-imageBA)], dim=-1)
            viz = torch.cat([vizA, vizB], dim=-2)
            grd = torchvision.utils.make_grid(viz, nrow=1, padding=0)
            tensorboard.add_image(f'{prefix}_viz', grd, self.current_epoch)

        return {f'{prefix}_loss': loss, 
                f'{prefix}_estim': estimA, 
                f'{prefix}_label': labelA, 
                }
    def validation_epoch_end(self, outputs) -> None:
        prefix = 'val'
        mean_loss = torch.stack([x[f'{prefix}_loss'] for x in outputs]).mean()
        np_estim = torch.cat([x[f'{prefix}_estim'].squeeze_(0) for x in outputs], axis=0).to('cpu').numpy()
        np_label = torch.cat([x[f'{prefix}_label'].squeeze_(0) for x in outputs], axis=0).to('cpu').numpy()

        sigmoid = True
        if sigmoid: 
            np_estim = 1/(1 + np.exp(-np_estim))
        # Casting to binary
        np_estim = 1.0 * (np_estim >= self.arg.threshold).astype(np.float32)
        np_label = 1.0 * (np_label >= self.arg.threshold).astype(np.float32)

        if np_estim.shape[0] > 0 and np_label.shape[0] > 0:
            for p in range(1):
                f1_score = sklearn.metrics.fbeta_score(np_label[:, p], np_estim[:, p], beta=1, average=self.average_type, zero_division=0)
                precision_score = sklearn.metrics.precision_score(np_label[:, p], np_estim[:, p], average=self.average_type, zero_division=0)
                recall_score = sklearn.metrics.recall_score(np_label[:, p], np_estim[:, p], average=self.average_type, zero_division=0)

                self.log(f'{prefix}_f1_score_{self.pathologies[p]}', f1_score, on_epoch=True, logger=True)
                self.log(f'{prefix}_precision_score_{self.pathologies[p]}', precision_score, on_epoch=True, logger=True)
                self.log(f'{prefix}_recall_score_{self.pathologies[p]}', recall_score, on_epoch=True, logger=True)

        self.log(f'{prefix}_loss', mean_loss, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)



    def test_step(self, batch, batch_idx):
        prefix = 'test'
        imageA, labelA, imageB, labelB = batch
        imageA, imageB = imageA / 255.0, imageB / 255.0
        estimA = self.classifier(imageA)
        # print(estim.shape, label.shape)
        loss = self.classifier_loss(estimA, labelA)
        # self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, enable_graph=True)
        # Visualization
        imageAB = self.generatorAB(imageA)
        imageBA = self.generatorBA(imageB)

        imageABA = self.generatorBA(imageAB)
        imageBAB = self.generatorAB(imageBA)

        return {f'{prefix}_loss': loss, 
                f'{prefix}_estim': estimA, 
                f'{prefix}_label': labelA, 
                }
    
    # def test_step_end(self, outputs):
    #     return outputs

    def test_epoch_end(self, outputs) -> None:
        prefix = 'test'
        mean_loss = torch.stack([x[f'{prefix}_loss'] for x in outputs]).mean()
        np_estim = torch.cat([x[f'{prefix}_estim'].squeeze_(0) for x in outputs], axis=0).to('cpu').numpy()
        np_label = torch.cat([x[f'{prefix}_label'].squeeze_(0) for x in outputs], axis=0).to('cpu').numpy()

        sigmoid = True
        if sigmoid: 
            np_estim = 1/(1 + np.exp(-np_estim))
        # Casting to binary
        np_estim = 1.0 * (np_estim >= self.arg.threshold).astype(np.float32)
        np_label = 1.0 * (np_label >= self.arg.threshold).astype(np.float32)

        if np_estim.shape[0] > 0 and np_label.shape[0] > 0:
            for p in range(1):
                f1_score = sklearn.metrics.fbeta_score(np_label[:, p], np_estim[:, p], beta=1, average=self.average_type, zero_division=0)
                precision_score = sklearn.metrics.precision_score(np_label[:, p], np_estim[:, p], average=self.average_type, zero_division=0)
                recall_score = sklearn.metrics.recall_score(np_label[:, p], np_estim[:, p], average=self.average_type, zero_division=0)

                print(f'{prefix}_f1_score_{p}', f1_score)
                print(f'{prefix}_precision_score_{p}', precision_score)
                print(f'{prefix}_recall_score_{p}', recall_score)


    def configure_optimizers(self):
        d_optim = torch.optim.Adam(self.discriminator.parameters(), 
                                   lr=1e-4)
        g_optim = torch.optim.Adam((params for model in [self.generatorAB, self.generatorBA] for params in model.parameters()),
                                    lr=2e-4)
        c_optim = torch.optim.Adam(self.classifier.parameters(), 
                                   lr=1e-4)

        d_sched = torch.optim.lr_scheduler.CosineAnnealingLR(d_optim, T_max=10)
        g_sched = torch.optim.lr_scheduler.CosineAnnealingLR(g_optim, T_max=10)
        c_sched = torch.optim.lr_scheduler.CosineAnnealingLR(c_optim, T_max=10)

        return [d_optim, g_optim, c_optim], [d_sched, g_sched, c_sched]
       

def main(args: Namespace) -> None:
    if args.seed is not None:
        pl.seed_everything(args.seed)

    if args.distributed_backend == 'ddp':
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / max(1, args.gpus))
        args.num_workers = int(args.num_workers / max(1, args.gpus))

    model = CustomLightningModel(args) #CustomLightningModel(**vars(args))
    dm = CustomDataModule(args)
    # checkpoint_callback = ModelCheckpoint(monitor='val_loss',
    #     dirpath='.',
    #     filename='torchxrayvision-{epoch:02d}-{val_loss:.2f}',
    #     save_top_k=30,
    #     mode='min'
    # )
    checkpoint_callback = ModelCheckpoint(
        filepath='torchxrayvision-{epoch:02d}-{val_loss:.2f}',
        save_top_k=-1,
        verbose=True,
        monitor='val_loss',  # TODO
        mode='min'
    )
    if args.load:
        trainer = pl.Trainer(checkpoint_callback=checkpoint_callback,
                             progress_bar_refresh_rate=1,
                             gpus=args.gpus, 
                             distributed_backend=args.distributed_backend,
                             amp_level='O2' if args.use_amp else 0, 
                             precision=16 if args.use_amp else 32, 
                             resume_from_checkpoint=args.load,
                             )
    else:
        trainer = pl.Trainer(checkpoint_callback=checkpoint_callback,
                             progress_bar_refresh_rate=1,
                             gpus=args.gpus, 
                             distributed_backend=args.distributed_backend,
                             amp_level='O2' if args.use_amp else 0, 
                             precision=16 if args.use_amp else 32, 
                             )

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
    parser.add_argument("--gpus", default=-1, help="number of available GPUs")
    parser.add_argument("--load", action='store_true')
    parser.add_argument('--distributed_backend', type=str, default='dp', choices=('dp', 'ddp', 'ddp2'), help='supports three options dp, ddp, ddp2')
    parser.add_argument('--use_amp', action='store_true', help='if true uses 16 bit precision')
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--num_workers", type=int, default=16, help="size of the workers")
    parser.add_argument("--grad_batches", type=int, default=1, help="number of batches to accumulate")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs to train")
    parser.add_argument("--shape", type=int, default=512, help="shape of the images")
    parser.add_argument("--seed", type=int, default=2020, help="reproducibility")
    parser.add_argument("--test", action='store_true', help="run test")
    parser.add_argument("--threshold", type=float, default=0.5, help="")
    parser.add_argument("--num", type=int, default=18, help="")

    parser.set_defaults(
        profiler=True,
        deterministic=True,
        max_epochs=301,
    )
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    run_cli()
