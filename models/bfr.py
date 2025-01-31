from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Mapping
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import torch
import torch.nn as nn

from models.vqvae import VQVAE
from models.basic_vae import Decoder, Encoder
from models.quant import VectorQuantizer2

class BFR(nn.Module):
    def __init__(self,
                 vae_lq: VQVAE,
                 vae_hq: VQVAE,
                 ):
        super().__init__()
        self.vae_lq = vae_lq  # freeze encoder, quant_conv
        self.vae_hq = vae_hq  # freeze decoder, post_quant_conv

    def forward(self, lq, ret_usages=False):
        hs_lq = self.vae_lq.encoder(lq)
        h_lq = hs_lq['out']
        f_hat_lq, vq_loss_lq, usage_lq = self.vae_lq.quantize(self.vae_lq.quant_conv(h_lq), ret_usages=ret_usages)
        hq_pred = self.vae_lq.decoder(self.vae_lq.post_quant_conv(f_hat_lq), hs_lq)

        hs_hq = self.vae_hq.encoder(hq_pred)
        h_hq = hs_hq['out']
        f_hat_hq, vq_loss_hq, usage_hq = self.vae_hq.quantize(self.vae_hq.quant_conv(h_hq), ret_usages=ret_usages)
        lq_pred = self.vae_hq.decoder(self.vae_hq.post_quant_conv(f_hat_hq), hs_lq)
        return lq_pred, hq_pred, vq_loss_lq, vq_loss_hq, usage_lq, usage_hq

    def inference(self, lq):
        lq_pred, hq_pred, vq_loss_lq, vq_loss_hq, usage_lq, usage_hq  = self.forward(lq)
        return lq_pred.clamp_(min=-1, max=1), hq_pred.clamp_(min=-1, max=1), vq_loss_lq, vq_loss_hq, usage_lq, usage_hq

    # def load_state_dict(self, state_dict: Tuple[Dict[str, Any], Dict[str, Any]], strict=True, assign=False, compat=False):
    #     state_lq, state_hq = state_dict
    #     self.vae_lq.load_state_dict(state_lq, strict=strict, assign=assign, compat=compat)
    #     self.vae_hq.load_state_dict(state_hq, strict=strict, assign=assign, compat=compat)


if __name__ == '__main__':
    import argparse
    import glob
    import math

    import PIL.Image as PImage
    from torchvision.transforms import InterpolationMode, transforms
    import torch

    from utils import dist_utils
    from utils.dataset.options import DataOptions

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--out', type=str, default='res')
    parser.add_argument('--opts', type=str, default='{}')
    args = parser.parse_args()

    device = 'cpu'
    # dist_utils.init_distributed_mode(local_out_path='../tmp', timeout_minutes=30)

    seed = args.seed
    import random
    def seed_everything(self):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        if self.seed is not None:
            print(f'[in seed_everything] {self.deterministic=}', flush=True)
            if self.deterministic:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True)
                os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
            seed = self.seed + dist_utils.get_rank()*16384
            os.environ['PYTHONHASHSEED'] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.same_seed_for_all_ranks = seed

    def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
        return x.add(x).add_(-1)

    def denormalize_pm1_into_01(x):  # normalize x from [-1, 1] to [0, 1] by (x + 1)/2
        return x.add(1)/2

    import sys
    from pathlib import Path
    import os
    root = Path(os.path.dirname(__file__)).parent
    vae_ckpt = args.ckpt
    B, C, H, W = 4, 3, 256, 256
    vae_lq = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=True,
                share_quant_resi=4, v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
                   head_size=1, fix_modules=['encoder', 'quant_conv']).to(device)
    vae_lq.eval()

    vae_hq = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=True,
                share_quant_resi=4, v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
                   head_size=1, fix_modules=['decoder', 'post_quant_conv']).to(device)
    vae_hq.eval()
    bfr = BFR(vae_lq, vae_hq)
    bfr.eval()

    # trainer
    state_dict = torch.load(vae_ckpt, map_location='cpu')
    if 'trainer' in state_dict.keys():
        state_dict = state_dict['trainer']['vae_wo_ddp']
    bfr.load_state_dict(state_dict, strict=True)


    # vae_ckpt = '/Users/katz/Downloads/vae_ch160v4096z32.pth'
    # vae_lq.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True, compat=True)
    # vae_hq.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True, compat=True)
    # torch.save(vae.state_dict(), '/Users/katz/Downloads/pt_vae_ch160v4096z32_new2.pth')

    from utils.dataset.ffhq_blind import FFHQBlind
    import torchvision
    import numpy as np

    # validate
    from pprint import pprint
    opt = DataOptions.val_options()
    pprint(opt)

    import json
    arg_opts = json.loads(args.opts)
    for k, v in arg_opts.items():
        print(f'Option updates: {k} {opt[k]} -> {k} {v}')
        opt[k] = v

    import torch.utils.data as data


    def inference(i, ds: data.Dataset):
        lq_in_list = []
        hq_in_list = []
        for idx in range(len(ds)):
            if idx > 20:
                break
            res = ds[idx]
            lq, hq = res['lq'], res['gt']
            lq_in_list.append(lq)
            hq_in_list.append(hq)
        lq = torch.stack(lq_in_list, dim=0)
        hq = torch.stack(hq_in_list, dim=0)
        print(i, lq.shape, hq.shape)

        lq_pred, hq_pred, vq_loss_lq, vq_loss_hq, _, _ = bfr.inference(lq)
        print(i, vq_loss_lq, vq_loss_hq)

        lq_pred_from_hq, hq_pred_from_hq, vq_loss_lq, vq_loss_hq, _, _ = bfr.inference(hq)
        print(i, vq_loss_lq, vq_loss_hq)

        res = [hq, lq, lq_pred_from_hq, hq_pred_from_hq, lq_pred, hq_pred]
        res_img = torch.stack(res, dim=1)
        res_img = torch.reshape(res_img, (-1, 3, 256, 256))
        img = denormalize_pm1_into_01(res_img)
        chw = torchvision.utils.make_grid(img, nrow=len(res), padding=0, pad_value=1.0)
        chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
        chw = PImage.fromarray(chw.astype(np.uint8))
        filename = f'dataset{i}-{args.out}.png'
        chw.save(os.path.join(root, filename))
        print(f'Saved {filename}...')

    for i in range(4):
        ds = FFHQBlind(root=f'{args.data}{i+1}', split='train', opt=opt)
        inference(i+1, ds)

