"""
References:
- VectorQuantizer2: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L110
- GumbelQuantize: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
- VQVAE (VQModel): https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/autoencoder.py#L14
"""
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import torch
import torch.nn as nn

from models.basic_vae import Decoder, Encoder
from models.quant import VectorQuantizer2
from models.vqvae import VQVAE


class BFR(VQVAE):
    def __init__(
            self,
            vqvae_local: VQVAE=None,
            vocab_size=4096,  # 词元大小
            z_channels=32,  # latent 维度
            ch=128,  # ResnetBlock 的起始维度，按照 ch_mult=(1, 1, 2, 2, 4) 得到每一层的 in_channels 和 out_channels
            dropout=0.0,
            beta=0.25,              # 表示 self.quant_conv(self.encoder(inp)) 与 f_hat 的 loss 的权重，只在 train vae 时候用到
            using_znorm=False,      # whether to normalize when computing the nearest neighbors
            quant_conv_ks=3,        # quant conv kernel size
            quant_resi=0.5,         # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x，残差块输出特征的权重
            share_quant_resi=4,     # use 4 \phi layers for K scales: partially-shared \phi，共享残差块的数量
            default_qresi_counts=0, # if is 0: automatically set to len(v_patch_nums)，残差块的数量
            v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]，代表每个尺度下的词元图像 h 和 w
            head_size=1,
            test_mode=False,
            fix_modules=[],
            quant_fix_modules=['quant_resi', 'embedding'],
            quat_use_predict=True,
    ):
        super().__init__()
        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        self.head_size = head_size
        # ddconfig is copied from https://github.com/CompVis/latent-diffusion/blob/e66308c7f2e64cb581c6d27ab6fbeb846828253b/models/first_stage_models/vq-f16/config.yaml
        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,  # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True, head_size=head_size  # from vq-f16/config.yaml above
            # resamp_with_conv=True,   # always True, removed.
        )
        ddconfig.pop('double_z', None)  # only KL-VAE should use double_z=True
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)

        if vqvae_local is not None:
            quant: VectorQuantizer2 = vqvae_local.quantize
            self.vae_proxy: Tuple[VQVAE] = (vqvae_local,)
            self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)

        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult']) - 1)
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi,
            share_quant_resi=share_quant_resi,
            fix_modules=quant_fix_modules, use_predict=quat_use_predict
        )
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2)
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1,
                                               padding=quant_conv_ks // 2)

        for module in fix_modules:
            for p in getattr(self, module).parameters():
                p.requires_grad_(False)
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]

        # ===================== `forward` is only used in VAE training =====================

    def forward(self, lq, **kwargs):  # -> rec_B3HW, idx_N, loss
        VectorQuantizer2.forward
        hs = self.encoder(lq)
        h = hs['out']
        f = self.quant_conv(h)
        f_hat_list, logits_list = self.quantize.f_to_logits_and_fhat(f, to_fhat=True, predict=True)
        hq_pred = self.decoder(self.post_quant_conv(f_hat_list[-1]), hs)
        logits = torch.cat(logits_list, dim=1)
        return f, hq_pred, logits
        # ===================== `forward` is only used in VAE training =====================

    def inference(self, lq):
        hs = self.encoder(lq)
        h = hs['out']
        f_hat_list = self.quantize.f_to_idxBl_or_fhat(self.quant_conv(h), to_fhat=True, predict=True)
        img = self.fhat_to_img(f_hat_list[-1], hs)
        return img

    def inference2(self, lq):
        hs = self.encoder(lq)
        h = hs['out']
        f_hat_list = self.quantize.f_to_idxBl_or_fhat(self.quant_conv(h), to_fhat=True, predict=True)
        imgs = [self.fhat_to_img(f_hat_list[i], hs) for i in range(len(f_hat_list))]
        return imgs


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
    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--out', type=str, default='res')
    parser.add_argument('--opts', type=str, default='{}')
    args = parser.parse_args()

    device = 'cpu'
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
    vae = BFR(vocab_size=4096, z_channels=32, ch=160, test_mode=True,
                share_quant_resi=4, v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
                head_size=1,
                ).to(device)
    vae.eval()

    # trainer
    # vae_ckpt = '/Users/katz/Downloads/vae_ch160v4096z32.pth'
    state_dict = torch.load(vae_ckpt, map_location='cpu')
    if 'trainer' in state_dict.keys():
        state_dict = state_dict['trainer']['vae_ema']
    vae.load_state_dict(state_dict, strict=False, compat=False)
    # vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=False, compat=True)
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

        lq_res = vae.inference(lq)
        hq_res = vae.inference(hq)

        res = [hq, lq, hq_res, lq_res]
        res_img = torch.stack(res, dim=1)
        res_img = torch.reshape(res_img, (-1, 3, 256, 256))
        img = denormalize_pm1_into_01(res_img)
        chw = torchvision.utils.make_grid(img, nrow=len(res), padding=0, pad_value=1.0)
        chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
        chw = PImage.fromarray(chw.astype(np.uint8))
        filename = f'dataset{i}-{args.out}.png'
        chw.save(os.path.join(root, filename))
        print(f'Saved {filename}...')

    def inference2(i, ds: data.Dataset):
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

        lq_res2 = vae.inference2(lq)
        res = [hq, lq]
        res.extend(lq_res2)
        res_img = torch.stack(res, dim=1)
        res_img = torch.reshape(res_img, (-1, 3, 256, 256))
        img = denormalize_pm1_into_01(res_img)
        chw = torchvision.utils.make_grid(img, nrow=len(res), padding=0, pad_value=1.0)
        chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
        chw = PImage.fromarray(chw.astype(np.uint8))
        filename = f'dataset{i}-{args.out}-2.png'
        chw.save(os.path.join(root, filename))
        print(f'Saved {filename}...')


    for i in range(4):
        ds = FFHQBlind(root=f'{args.data}{i+1}', split='train', opt=opt)
        if args.mode == 0:
            inference(i + 1, ds)
        elif args.mode == 1:
            if i == 0:
                inference2(i + 1, ds)
