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


class VQVAE(nn.Module):
    def __init__(
            self,
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
            test_mode=True,
            fix_modules=[],
    ):
        super().__init__()
        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        # ddconfig is copied from https://github.com/CompVis/latent-diffusion/blob/e66308c7f2e64cb581c6d27ab6fbeb846828253b/models/first_stage_models/vq-f16/config.yaml
        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2,   # from vq-f16/config.yaml above
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            # resamp_with_conv=True,   # always True, removed.
        )
        ddconfig.pop('double_z', None)  # only KL-VAE should use double_z=True
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult'])-1)
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        )
        self.quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        self.post_quant_conv = torch.nn.Conv2d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)

        for module in fix_modules:
            for p in getattr(self, module).parameters():
                p.requires_grad_(False)
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]

    # ===================== `forward` is only used in VAE training =====================
    def forward(self, lq, w=1, ret_usages=False):   # -> rec_B3HW, idx_N, loss
        VectorQuantizer2.forward
        hs = self.encoder(lq)
        h = hs['out']
        f_hat, vq_loss, usages = self.quantize(self.quant_conv(h), ret_usages=ret_usages)
        return self.decoder(self.post_quant_conv(f_hat), hs, w=w), vq_loss, usages
    # ===================== `forward` is only used in VAE training =====================

    def inference(self, lq, ret_usages=False):
        VectorQuantizer2.forward
        hs = self.encoder(lq)
        h = hs['out']
        f_hat, vq_loss, usages = self.quantize(self.quant_conv(h), ret_usages=ret_usages)
        img = self.fhat_to_img(f_hat, hs)
        return img, vq_loss, usages

    def fhat_to_img(self, f_hat: torch.Tensor, hs, w=1):
        return self.decoder(self.post_quant_conv(f_hat), hs, w=w).clamp_(min=-1, max=1)

    def img_to_idxBl(self, inp_img_no_grad: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]:    # return List[Bl]
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)

    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            ms_h_BChw.append(self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn))
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)

    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False, w=1) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True)), w=w).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat), w=w).clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]

    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False, w=1) -> List[torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1]), w=w).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat), w=w).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]

    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False, compat=False):
        print(f'Loading state dict with strict={strict}, assign={assign}, compat={compat}')
        if 'quantize.ema_vocab_hit_SV' in state_dict and state_dict['quantize.ema_vocab_hit_SV'].shape[0] != self.quantize.ema_vocab_hit_SV.shape[0]:
            state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV

        if compat:
            # compat encoder and decoder
            new_state_dict = OrderedDict()
            for key in list(state_dict.keys()):
                if key.startswith('decoder.'):
                    comped = False
                    if key.find('attn_1.qkv') != -1:
                        param = state_dict.pop(key)
                        dim = param.shape[0] // 3
                        param_list = torch.split(param, dim, dim=0)
                        q_key = key.replace('qkv', 'q')
                        k_key = key.replace('qkv', 'k')
                        v_key = key.replace('qkv', 'v')
                        new_state_dict[q_key] = param_list[0].contiguous()
                        new_state_dict[k_key] = param_list[1].contiguous()
                        new_state_dict[v_key] = param_list[2].contiguous()
                        comped = True
                    for i in range(5):
                        if key.find(f'attn.{i}.qkv') != -1:
                            param = state_dict.pop(key)
                            dim = param.shape[0] // 3
                            param_list = torch.split(param, dim, dim=0)
                            q_key = key.replace('qkv', 'q')
                            k_key = key.replace('qkv', 'k')
                            v_key = key.replace('qkv', 'v')
                            new_state_dict[q_key] = param_list[0].contiguous()
                            new_state_dict[k_key] = param_list[1].contiguous()
                            new_state_dict[v_key] = param_list[2].contiguous()
                            comped = True
                    if key.find('attn_1.norm') != -1:
                        param = state_dict.pop(key)
                        norm1_key = key.replace('norm', 'norm1')
                        norm2_key = key.replace('norm', 'norm2')
                        new_state_dict[norm1_key] = param
                        new_state_dict[norm2_key] = param.clone()
                        comped = True
                    for i in range(5):
                        if key.find(f'attn.{i}.norm') != -1:
                            param = state_dict.pop(key)
                            norm1_key = key.replace('norm', 'norm1')
                            norm2_key = key.replace('norm', 'norm2')
                            new_state_dict[norm1_key] = param
                            new_state_dict[norm2_key] = param.clone()
                            comped = True
                    if not comped:
                        new_state_dict[key] = state_dict.pop(key)
                elif key.startswith('encoder.'):
                    comped = False
                    if key.find('attn_1.qkv') != -1:
                        param = state_dict.pop(key)
                        dim = param.shape[0] // 3
                        param_list = torch.split(param, dim, dim=0)
                        q_key = key.replace('qkv', 'q')
                        k_key = key.replace('qkv', 'k')
                        v_key = key.replace('qkv', 'v')
                        new_state_dict[q_key] = param_list[0].contiguous()
                        new_state_dict[k_key] = param_list[1].contiguous()
                        new_state_dict[v_key] = param_list[2].contiguous()
                        comped = True
                    for i in range(5):
                        if key.find(f'attn.{i}.qkv') != -1:
                            param = state_dict.pop(key)
                            dim = param.shape[0] // 3
                            param_list = torch.split(param, dim, dim=0)
                            q_key = key.replace('qkv', 'q')
                            k_key = key.replace('qkv', 'k')
                            v_key = key.replace('qkv', 'v')
                            new_state_dict[q_key] = param_list[0].contiguous()
                            new_state_dict[k_key] = param_list[1].contiguous()
                            new_state_dict[v_key] = param_list[2].contiguous()
                            comped = True
                    if key.find('attn_1.norm') != -1:
                        param = state_dict.pop(key)
                        norm1_key = key.replace('norm', 'norm1')
                        norm2_key = key.replace('norm', 'norm2')
                        new_state_dict[norm1_key] = param
                        new_state_dict[norm2_key] = param.clone()
                        comped = True
                    for i in range(5):
                        if key.find(f'attn.{i}.norm') != -1:
                            param = state_dict.pop(key)
                            norm1_key = key.replace('norm', 'norm1')
                            norm2_key = key.replace('norm', 'norm2')
                            new_state_dict[norm1_key] = param
                            new_state_dict[norm2_key] = param.clone()
                            comped = True
                    if not comped:
                        new_state_dict[key] = state_dict.pop(key)
                else:
                    new_state_dict[key] = state_dict.pop(key)
        else:
            new_state_dict = state_dict

        # 	Missing key(s) in state_dict: "decoder.mid.attn_1.norm1.weight", "decoder.mid.attn_1.norm1.bias", "decoder.mid.attn_1.norm2.weight", "decoder.mid.attn_1.norm2.bias", "decoder.mid.attn_1.q.weight", "decoder.mid.attn_1.q.bias", "decoder.mid.attn_1.k.weight", "decoder.mid.attn_1.k.bias", "decoder.mid.attn_1.v.weight", "decoder.mid.attn_1.v.bias", "decoder.up.4.attn.0.norm1.weight", "decoder.up.4.attn.0.norm1.bias", "decoder.up.4.attn.0.norm2.weight", "decoder.up.4.attn.0.norm2.bias", "decoder.up.4.attn.0.q.weight", "decoder.up.4.attn.0.q.bias", "decoder.up.4.attn.0.k.weight", "decoder.up.4.attn.0.k.bias", "decoder.up.4.attn.0.v.weight", "decoder.up.4.attn.0.v.bias", "decoder.up.4.attn.1.norm1.weight", "decoder.up.4.attn.1.norm1.bias", "decoder.up.4.attn.1.norm2.weight", "decoder.up.4.attn.1.norm2.bias", "decoder.up.4.attn.1.q.weight", "decoder.up.4.attn.1.q.bias", "decoder.up.4.attn.1.k.weight", "decoder.up.4.attn.1.k.bias", "decoder.up.4.attn.1.v.weight", "decoder.up.4.attn.1.v.bias", "decoder.up.4.attn.2.norm1.weight", "decoder.up.4.attn.2.norm1.bias", "decoder.up.4.attn.2.norm2.weight", "decoder.up.4.attn.2.norm2.bias", "decoder.up.4.attn.2.q.weight", "decoder.up.4.attn.2.q.bias", "decoder.up.4.attn.2.k.weight", "decoder.up.4.attn.2.k.bias", "decoder.up.4.attn.2.v.weight", "decoder.up.4.attn.2.v.bias".
        # 	Unexpected key(s) in state_dict: "decoder.mid.attn_1.norm.weight", "decoder.mid.attn_1.norm.bias", "decoder.mid.attn_1.qkv.weight", "decoder.mid.attn_1.qkv.bias", "decoder.up.4.attn.0.norm.weight", "decoder.up.4.attn.0.norm.bias", "decoder.up.4.attn.0.qkv.weight", "decoder.up.4.attn.0.qkv.bias", "decoder.up.4.attn.1.norm.weight", "decoder.up.4.attn.1.norm.bias", "decoder.up.4.attn.1.qkv.weight", "decoder.up.4.attn.1.qkv.bias", "decoder.up.4.attn.2.norm.weight", "decoder.up.4.attn.2.norm.bias", "decoder.up.4.attn.2.qkv.weight", "decoder.up.4.attn.2.qkv.bias".
        return super().load_state_dict(state_dict=new_state_dict, strict=strict, assign=assign)


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

    def img_folder_to_tensor(img_folder: str, transform: transforms.Compose, img_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ori_aug = transforms.Compose(
            [
                transforms.Resize((img_size, img_size), interpolation=InterpolationMode.LANCZOS),
                transforms.ToTensor()
            ]
        )
        # mid_reso = 1.125
        # final_reso = 256
        # mid_reso = round(min(mid_reso, 2) * final_reso)
        # ori_aug = transforms.Compose(
        #     [
        #         transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
        #         transforms.CenterCrop((final_reso, final_reso)),
        #         # transforms.Resize(final_reso, interpolation=InterpolationMode.LANCZOS),
        #         transforms.ToTensor(), normalize_01_into_pm1
        #     ]
        # )
        img_list = glob.glob(f'{img_folder}/*.png')
        img_all = []
        ori_img_all = []
        for img_path in img_list:
            img_tensor = transform(PImage.open(img_path))
            origin_img_tensor = ori_aug(PImage.open(img_path))
            img_all.append(img_tensor)
            ori_img_all.append(origin_img_tensor)
        img_tensor = torch.stack(img_all, dim=0)
        origin_img_tensor = torch.stack(ori_img_all, dim=0)
        return origin_img_tensor, img_tensor

    def tensor_to_img(img_tensor: torch.Tensor) -> PImage.Image:
        B, C, H, W = img_tensor.shape
        assert int(math.sqrt(B)) * int(math.sqrt(B)) == B
        b = int(math.sqrt(B))
        img_tensor = torch.permute(img_tensor, (1, 0, 2, 3))
        img_tensor = torch.reshape(img_tensor, (C, b, b * H, W))
        img_tensor = torch.permute(img_tensor, (0, 2, 1, 3))
        img_tensor = torch.reshape(img_tensor, (C, b * H, b * W))
        img = transforms.ToPILImage()(img_tensor)
        return img

    import sys
    from pathlib import Path
    import os
    root = Path(os.path.dirname(__file__)).parent
    vae_ckpt = args.ckpt
    B, C, H, W = 4, 3, 256, 256
    vae = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=True,
                share_quant_resi=4, v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)).to(device)
    vae.eval()

    # trainer
    state_dict = torch.load(vae_ckpt, map_location='cpu')
    if 'trainer' in state_dict.keys():
        state_dict = state_dict['trainer']['vae_ema']
    vae.load_state_dict(state_dict, strict=True, compat=False)
    # vae_ckpt = '/Users/katz/Downloads/vae_ch160v4096z32.pth'
    # vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=False, compat=True)
    # torch.save(vae.state_dict(), os.path.join('./', 'pt_vae_ch160v4096z32_new2.pth'))

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

        lq_res, vq_loss, usages = vae.inference(lq, ret_usages=True)
        print(i, vq_loss)
        print(i, usages)

        hq_res, vq_loss, usages = vae.inference(hq, ret_usages=True)
        print(i, vq_loss)
        print(i, usages)

        res_img = torch.stack([hq, lq, hq_res, lq_res], dim=1)
        res_img = torch.reshape(res_img, (-1, 3, 256, 256))
        img = denormalize_pm1_into_01(res_img)
        chw = torchvision.utils.make_grid(img, nrow=4, padding=0, pad_value=1.0)
        chw = chw.permute(1, 2, 0).mul_(255).cpu().numpy()
        chw = PImage.fromarray(chw.astype(np.uint8))
        chw.save(os.path.join(root, f'dataset{i}-{args.out}.png'))

    for i in range(4):
        ds = FFHQBlind(root=f'{args.data}{i+1}', split='train', opt=opt)
        inference(i+1, ds)
