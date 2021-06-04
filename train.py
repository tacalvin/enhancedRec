import argparse

import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from model import DQLR, DQLRnn
from scheduler import CycleScheduler
from dataloader import VideoDataset


def train(epoch, loader, model, optimizer, scheduler, sample_path, alpha, ltriplet=.25):
    loader = tqdm(loader)

    criterion = nn.MSELoss()
    crit_triplet = nn.TripletMarginLoss(margin=alpha)
    latent_loss_weight = 0.25
    triplet_loss_weight = ltriplet
    sample_size = 12

    mse_sum = 0
    mse_n = 0
    model.train()

    for i, (img, img_s) in enumerate(loader):
        # print(i)
        model.zero_grad()

        img = img.cuda()  # .to(device)
        img_s = img_s.cuda()
        # print(img_s.shape)
        # print(img.shape)
        out, latent_loss = model(img_s)
        # recon_loss = 0.0
        # print(len(out))
        # print(out.size())
        # quit()
        # for j in range(len(out)):
        # recon_loss = criterion(out[:,[0,2],:,:,:], img_s[:,[0,2],:,:,:])
        recon_loss = criterion(out, img_s)
        recon_loss /= 3.0

        # print(recon_loss.size(), recon_loss)
        # print(latent_loss.size())
        # quit()

        # apply triplet loss with middle frame as the anchor
        triplet_loss = crit_triplet(
            out[:, 1, :, :, :], out[:, 0, :, :, :], out[:, 2, :, :, :])
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * \
            latent_loss + (triplet_loss_weight * triplet_loss)
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        mse_sum += recon_loss.item() * img.shape[0]
        mse_n += img.shape[0]

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                f'triplet: {triplet_loss.item():.3f}'
                f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                f'lr: {lr:.5f}'
            )
        )

        if i % 180 == 0:
            # print("SAMPLE")
            model.eval()

            sample = img_s[:sample_size]
            # print("SAMPLE_SIZE:{}".format(sample.size()))

            with torch.no_grad():
                out, _ = model(sample)

            # print("SAMPLE_SIZE:{}".format(out.size()))
            sample = img_s[:sample_size, 1, :, :, :]
            # print("NEW SAMPLE IMAGE:{}".format(sample.size()))
            #sample = sample.div(255)
            #out[2] = out[2].div(255)

            utils.save_image(
                torch.cat([sample, out[:sample_size, 1, :, :, :]], 0),
                f'./{sample_path}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )
            model.train()


def evaluate(test_loader, model, alpha, ltriplet=.25):
    model.eval()
    test_loader = tqdm(test_loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 6
    crit_triplet = nn.TripletMarginLoss(margin=alpha)
    triplet_loss_weight = ltriplet
    mse_sum = 0
    mse_n = 0
    with torch.no_grad():

        for i, (img, img_s) in enumerate(test_loader):
            img = img.cuda()  # .to(device)
            img_s = img_s.cuda()

            out, latent_loss = model(img_s)
            # recon_loss = 0.0
            # for j in range(len(out)):
            # recon_loss += criterion(out[j], img_s[:,j,:,:])
            recon_loss = criterion(out, img_s)
            recon_loss /= 3.0
            latent_loss = latent_loss.mean()
            triplet_loss = crit_triplet(
                out[:, 1, :, :, :], out[:, 0, :, :, :], out[:, 2, :, :, :])
            # loss = recon_loss + latent_loss_weight * latent_loss
            loss = recon_loss + latent_loss_weight * \
                latent_loss + (triplet_loss_weight * triplet_loss)

            mse_sum += recon_loss.item() * img.shape[0]
            mse_n += img.shape[0]

            test_loader.set_description(
                (
                    f'Evaluating; mse: {recon_loss.item():.5f}; '
                    f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                )
            )
    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=5600)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--ckp', type=str, default="./checkpoint")
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--sample_path')
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--ltriplet', type=float, default=.05)
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    print(args)

    device = 'cuda'

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    #dataset = datasets.ImageFolder(args.path, transform=transform)
    #loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    train_data = VideoDataset(args.path, split='train', clip_len=3, clip_jump=args.skip,
                              size=args.size, preprocess=False, transform=transform)
    loader = DataLoader(train_data, batch_size=args.bs,
                        shuffle=True, num_workers=4)

    test_data = VideoDataset(args.path, split='val', clip_len=3, 
                             size=args.size, clip_jump=args.skip, preprocess=False, transform=transform)
    test_loader = DataLoader(
        test_data, batch_size=args.bs, shuffle=True, num_workers=4)
    # raw_input('Enter')
    model = DQLRnn()
    if not args.pretrained == None:
        print('Loading pretrained weights...')
        pre_w = torch.load(args.pretrained)
        for key in pre_w.keys():
            model.state_dict()[key] = pre_w[key]
            if 'dec' in key:
                key2 = key.replace('dec', 'dec1')
                model.state_dict()[key2] = pre_w[key]
                key2 = key.replace('dec', 'dec2')
                model.state_dict()[key2] = pre_w[key]
            if 'upsample_t' in key:
                key2 = key.replace('upsample_t', 'upsample_t1')
                model.state_dict()[key2] = pre_w[key]
                key2 = key.replace('upsample_t', 'upsample_t2')
                model.state_dict()[key2] = pre_w[key]
    else:
        print('Training from scratch')

    model = nn.DataParallel(model).cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, args.sample_path, args.alpha, args.ltriplet)
        torch.save(
            model.module.state_dict(
            ), f'{args.ckp}/vqvae_{str(i + 1).zfill(3)}.pt'
        )
        evaluate(test_loader, model, args.alpha, args.ltriplet)
