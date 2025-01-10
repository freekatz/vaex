"""
References:
- VectorQuantizer: VectorQuantizer2 from https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L110
- VQVAE: VQModel from https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/autoencoder.py#L14
"""
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basic_vae import CNNDecoder, CNNEncoder
from models.quant import VectorQuantizer2


def identity(x, inplace=False): return x


class VQVAE(nn.Module):
    def __init__(
        self,
        # for all:
        grad_ckpt=False,            # whether to use gradient checkpointing
        
        # vitamin encoder:
        vitamin='',                 # 's', 'b', 'l' for using vitamin; 'cnn' or '' for using CNN
        drop_path_rate=0.1,
        
        # CNN encoder & CNN decoder:
        ch=128,                     # basic width of CNN encoder and CNN decoder
        ch_mult=(1, 1, 2, 2, 4),    # downsample_ratio would be 2 ** (len(ch_mult) - 1)
        dropout=0.0,                # dropout in CNN encoder and CNN decoder
        
        # quantizer:
        vocab_size=4096,
        vocab_width=32,
        vocab_norm=False,           # whether to limit the codebook vectors to have unit norm
        beta=0.25,                  # commitment loss weight
        quant_conv_k=3,             # quant conv kernel size
        share_quant_resi=4,         # use 4 \phi layers for K scales: partially-shared \phi
        default_qresi_counts=0,     # if is 0: automatically set to len(v_patch_nums)
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
        quant_resi=-0.5,            #
    ):
        super().__init__()
        self.downsample_ratio = 2 ** (len(ch_mult) - 1)
        
        # 1. build encoder
        print(f'[VQVAE] create CNN Encoder with {ch=}, {ch_mult=} {dropout=:g} ...', flush=True)
        self.encoder: CNNEncoder = CNNEncoder(
            ch=ch, ch_mult=ch_mult, num_res_blocks=2, dropout=dropout,
            img_channels=3, output_channels=vocab_width, using_sa=True, using_mid_sa=True,
            grad_ckpt=grad_ckpt,
        )
        # 2. build conv before quant
        self.quant_conv = nn.Conv2d(vocab_width, vocab_width, quant_conv_k, stride=1, padding=quant_conv_k // 2)
        
        # 3. build quant
        print(f'[VQVAE] create VectorQuantizer with {vocab_size=}, {vocab_width=} {vocab_norm=}, {beta=:g} ...', flush=True)
        # self.quantize: VectorQuantizer2 = VectorQuantizer2(vocab_size=vocab_size, vocab_width=vocab_width, vocab_norm=vocab_norm, beta=beta, quant_resi=quant_resi)
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=vocab_width, using_znorm=vocab_norm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        )
        # 4. build conv after quant
        self.post_quant_conv = nn.Conv2d(vocab_width, vocab_width, quant_conv_k, stride=1, padding=quant_conv_k // 2)
        print(f'[VQVAE] create CNN Decoder with {ch=}, {ch_mult=} {dropout=:g} ...', flush=True)
        
        # 5. build decoder
        self.decoder = CNNDecoder(
            ch=ch, ch_mult=ch_mult, num_res_blocks=3, dropout=dropout,
            input_channels=vocab_width, using_sa=True, using_mid_sa=True,
            grad_ckpt=grad_ckpt,
        )
        self.maybe_record_function = nullcontext
    
    def forward(self, img_B3HW, ret_usages=False):
        f_BChw = self.encoder(img_B3HW).float()
        with torch.cuda.amp.autocast(enabled=False):
            VectorQuantizer2.forward
            f_BChw, vq_loss, entropy_loss, usages = self.quantize(self.quant_conv(f_BChw), ret_usages=ret_usages)
            f_BChw = self.post_quant_conv(f_BChw)
        return self.decoder(f_BChw).float(), vq_loss, entropy_loss, usages
    
    def img_to_idx(self, img_B3HW: torch.Tensor) -> torch.LongTensor:
        f_BChw = self.encoder(img_B3HW)
        f_BChw = self.quant_conv(f_BChw)
        return self.quantize.f_to_idx(f_BChw)
    
    def idx_to_img(self, idx_Bhw: torch.Tensor) -> torch.Tensor:
        f_hat_BChw = self.quantize.quant_resi(self.quantize.embedding(idx_Bhw).permute(0, 3, 1, 2))
        f_hat_BChw = self.post_quant_conv(f_hat_BChw)
        return self.decoder(f_hat_BChw).clamp_(-1, 1)
    
    def img_to_reconstructed_img(self, img_B3HW) -> torch.Tensor:
        return self.idx_to_img(self.img_to_idx(img_B3HW))
    
    def state_dict(self, *args, **kwargs):
        d = super().state_dict(*args, **kwargs)
        d['vocab_usage_record_times'] = self.quantize.vocab_usage_record_times
        return d
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        if 'quantize.vocab_usage' not in state_dict or state_dict['quantize.vocab_usage'].shape[0] != self.quantize.vocab_usage.shape[0]:
            # state_dict['quantize.vocab_usage'] = self.quantize.vocab_usage
            state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        if 'vocab_usage_record_times' in state_dict:
            # self.quantize.vocab_usage_record_times = state_dict.pop('vocab_usage_record_times')
            self.quantize.record_hit = state_dict.pop('vocab_usage_record_times')
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)


if __name__ == '__main__':
    # TODO 解决
    #   RuntimeError: Error(s) in loading state_dict for VQVAE:
    # 	Missing key(s) in state_dict: "quantize.quant_resi.weight", "quantize.quant_resi.bias".
    # 	Unexpected key(s) in state_dict: "quantize.ema_vocab_hit_SV", "quantize.quant_resi.qresi_ls.0.weight", "quantize.quant_resi.qresi_ls.0.bias", "quantize.quant_resi.qresi_ls.1.weight", "quantize.quant_resi.qresi_ls.1.bias", "quantize.quant_resi.qresi_ls.2.weight", "quantize.quant_resi.qresi_ls.2.bias", "quantize.quant_resi.qresi_ls.3.weight", "quantize.quant_resi.qresi_ls.3.bias".

    import glob
    import math

    import PIL.Image as PImage
    from torchvision.transforms import InterpolationMode, transforms
    import torch

    from utils import dist_utils

    dist_utils.init_distributed_mode(local_out_path='../tmp', timeout_minutes=30)

    def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
        return x.add(x).add_(-1)

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

    vae_ckpt = r'/Users/katz/Downloads/vae_ch160v4096z32.pth'
    B, C, H, W = 4, 3, 256, 256
    vae = VQVAE(
        grad_ckpt=False,
        vitamin='cnn', drop_path_rate=0.0,
        ch=160, ch_mult=(1, 1, 2, 2, 4), dropout=0.0,
        vocab_size=4096, vocab_width=32, vocab_norm=False, beta=0.25,
        quant_conv_k=3, quant_resi=-0.5,
    ).to('cpu')
    vae.eval()
    state_dict = torch.load(vae_ckpt, map_location='cpu')
    vae.load_state_dict(state_dict, strict=True)

    mid_reso = 1.125
    final_reso = 256
    mid_reso = round(min(mid_reso, 2) * final_reso)
    aug = transforms.Compose(
        [
            # transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
            # transforms.CenterCrop((final_reso, final_reso)),
            transforms.Resize((final_reso, final_reso), interpolation=InterpolationMode.LANCZOS),
            transforms.ToTensor()
        ]
    )
    origin_img, img = img_folder_to_tensor('../tmp', aug, img_size=final_reso)
    print(img.shape)

    in_img = tensor_to_img(origin_img)
    in_img.save('../inp.png')

    img = torch.clamp(img, 0., 1.)
    img = img.add(img).add_(-1)
    res, vq_loss, entropy_loss, usages = vae.forward(img, ret_usages=True)
    print(res.shape)
    print(usages)

    res_img = tensor_to_img(res)
    res_img.save('../out.png')
