"""
References:
- VectorQuantizer2: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L110
- GumbelQuantize: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
- VQVAE (VQModel): https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/autoencoder.py#L14
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from models_var.basic_vae import Decoder, Encoder
from models_var.quant import VectorQuantizer2


class VQVAE(nn.Module):
    def __init__(
        self, vocab_size=4096, z_channels=32, ch=128, dropout=0.0,
        beta=0.25,              # commitment loss weight
        using_znorm=False,      # whether to normalize when computing the nearest neighbors
        quant_conv_ks=3,        # quant conv kernel size
        quant_resi=0.5,         # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
        share_quant_resi=4,     # use 4 \phi layers for K scales: partially-shared \phi
        default_qresi_counts=0, # if is 0: automatically set to len(v_patch_nums)
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16), # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
        test_mode=True,
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
        
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, inp, ret_usages=False):   # -> rec_B3HW, idx_N, loss
        VectorQuantizer2.forward
        f_hat, usages, vq_loss = self.quantize(self.quant_conv(self.encoder(inp)), ret_usages=ret_usages)
        return self.decoder(self.post_quant_conv(f_hat)), usages, vq_loss
    # ===================== `forward` is only used in VAE training =====================
    
    def fhat_to_img(self, f_hat: torch.Tensor):
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
    
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
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True))).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        if 'quantize.ema_vocab_hit_SV' in state_dict and state_dict['quantize.ema_vocab_hit_SV'].shape[0] != self.quantize.ema_vocab_hit_SV.shape[0]:
            state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)


if __name__ == '__main__':
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
    vae = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=True,
                share_quant_resi=4, v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)).to('cpu')
    vae.eval()
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

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
    res, vq_loss, usages = vae.forward(img, ret_usages=True)
    print(res.shape)
    print(vq_loss)
    print(usages)

    res_img = tensor_to_img(res)
    res_img.save('../out.png')
