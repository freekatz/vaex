import os
import random
import sys
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import PIL.Image as PImage

from models import VAR, build_vae_var_eval
from models.vqvae import VQVAE
from utils.data import build_dataset
from utils.dataset import UnlabeledImageItem, UnlabeledImageFolder, FFHQ, CelebAHQ
from utils.dataset.my_transforms import denormalize_pm1_into_01
from utils.dataset.options import DataOptions
from utils.common import seed_everything
from utils import dist_utils


def inference(vqvae: VQVAE, var: VAR, lq, hq) -> [torch.Tensor]:
    hq0_lq = var.autoregressive_infer_cfg(lq)
    lq_res = vqvae.inference(hq0_lq)

    hq0_hq = var.autoregressive_infer_cfg(hq)
    hq_res = vqvae.inference(hq0_hq)
    res = [hq, lq, hq_res, lq_res]
    return res


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=False, default='./local_output/visual_output_tmp')
    parser.add_argument('--image_path', type=str, required=False, default='')
    parser.add_argument('--data_path', type=str, required=False, default='')
    parser.add_argument('--dataset_name', type=str, required=False, default='image_item_blind')
    parser.add_argument('--split', type=str, required=False, default='test', choices=DataOptions.get_splits())
    parser.add_argument('--vae_ckpt_path', type=str, required=False, default='')
    parser.add_argument('--origin_vae_ckpt_path', type=str, required=False, default='')
    parser.add_argument('--var_ckpt_path', type=str, required=False, default='')
    parser.add_argument('--task', type=str, required=False, default='vqvae',
                        choices=['var', 'var_all', 'vqvae', 'vqvae_all'])
    parser.add_argument('--batch_size', type=int, required=False, default=16)
    parser.add_argument('--nums', type=int, required=False, default=1)
    parser.add_argument('--seed', type=int, required=False, default=random.randint(1, 10000))
    parser.add_argument('--nrow', type=int, required=False, default=0)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--cpu', action='store_true')

    args = parser.parse_args()
    pprint(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = dist_utils.get_device() if not args.cpu else 'cpu'

    seed_everything(args.seed, benchmark=True)

    if args.image_path == '' and args.data_path == '':
        sys.exit('You must specify an image or data path.')

    if args.image_path != '':
        data_path = args.image_path
        n = 1
    else:
        data_path = args.data_path
        n = args.nums

    data_params = {'opt': DataOptions.get_options(args.split)}
    dataset = build_dataset(
        dataset_name=args.dataset_name,
        data_path=data_path,
        params=data_params,
        split=args.split,
    )
    task_type = args.task
    vqvae, var = build_vae_var_eval(device=device)
    vqvae.eval()
    var.eval()

    var_ckpt = torch.load(args.var_ckpt_path, map_location='cpu')
    if 'trainer' in var_ckpt.keys():
        var.load_state_dict(var_ckpt['trainer']['var_local'], strict=True, compat=False)
    else:
        missing_keys, unexpected_keys = var.load_state_dict(var_ckpt, strict=False, compat=True)
        print('missing_keys: ', [k for k in missing_keys])
        print('unexpected_keys: ', [k for k in unexpected_keys])
    print(f'Loaded var ckpt from {args.var_ckpt_path}')

    origin_vae_ckpt = torch.load(args.origin_vae_ckpt_path, map_location='cpu')
    var.vae_proxy[0].load_state_dict(origin_vae_ckpt, strict=True)
    print(f'Loaded var.vae ckpt from {args.origin_vae_ckpt_path}')

    vae_ckpt = torch.load(args.vae_ckpt_path, map_location='cpu')
    if 'trainer' in vae_ckpt.keys():
        vqvae.load_state_dict(vae_ckpt['trainer']['vae_ema'], strict=True)
    else:
        vqvae.load_state_dict(vae_ckpt, strict=True)
    print(f'Loaded vae ckpt from {args.vae_ckpt_path}')

    if n > len(dataset):
        n = len(dataset)
    batch_size = args.batch_size
    if batch_size > n:
        batch_size = n
    data_loader = DataLoader(
        dataset, num_workers=0, pin_memory=True,
        batch_size=batch_size,
        shuffle=True, drop_last=False,
    )

    count = 0
    res_list = []
    for i, data in enumerate(data_loader):
        data: dict
        lq, hq = data['lq'], data['gt']
        if args.random:
            lq = torch.randn_like(lq ,device=device)
        else:
            lq = lq.to(device, non_blocking=True)

        hq = hq.to(device, non_blocking=True)
        print(lq.shape, hq.shape)

        if task_type == 'var':
            raise NotImplementedError('VAR is not implemented.')
        elif task_type == 'var_all':
            raise NotImplementedError('VAR_all is not implemented.')
        elif task_type == 'vqvae':
            res = inference(vqvae, var, lq, hq)
        elif task_type == 'vqvae_all':
            raise NotImplementedError('VQVAE_all is not implemented.')
        else:
            raise NotImplementedError(f'Unknown task type {task_type}.')

        res_list.append(res)

        count += lq.size(0)
        if count >= n:
            break

    res = []
    for i, r in enumerate(res_list):
        if len(res) == 0:
            res = r
        else:
            for j in range(len(r)):
                res[j] = torch.cat((res[j], r[j]), dim=0)
    res_img = torch.stack(res, dim=1)
    res_img = torch.reshape(res_img, (-1, 3, 256, 256))
    img = denormalize_pm1_into_01(res_img)
    chw = torchvision.utils.make_grid(img, nrow=args.nrow if args.nrow > 0 else len(res), padding=0, pad_value=1.0)
    chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
    chw = PImage.fromarray(chw.astype(np.uint8))

    timestamp = int(time.time())
    out_path = out_dir / f'{task_type}-{args.dataset_name}_{args.split}-{args.seed}-{timestamp}.png'
    chw.save(out_path)
    print(f'Results ({res_img.shape}) saved to {out_path}')
