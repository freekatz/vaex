import os
import sys
from typing import Tuple

import torch.nn as nn

from utils.arg_util import Args
from .quant import VectorQuantizer2
from .vqvae import VQVAE
from .dino import DinoDisc
from .basic_vae import Encoder, Decoder


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))


def build_vae_disc(args: Args) -> Tuple[VQVAE, DinoDisc]:
    # disable built-in initialization for speed
    for clz in (
        nn.Linear, nn.Embedding,
        nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
        nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    ):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    # build models
    vae = VQVAE(vocab_size=args.vocab_size, z_channels=args.vocab_width, ch=args.ch,
                using_znorm=args.vocab_norm, beta=args.vq_beta, dropout=args.drop_out,
                share_quant_resi=4, quant_conv_ks=3, quant_resi=0.5,
                v_patch_nums=args.patch_nums, test_mode=False).to(args.device)
    disc = DinoDisc(
        device=args.device, dino_ckpt_path=args.dino_path, depth=args.dino_depth, key_depths=(2, 5, 8, 11),
        ks=args.dino_kernel_size, norm_type=args.disc_norm, using_spec_norm=args.disc_spec_norm, norm_eps=1e-6,
    ).to(args.device)
    
    # init weights
    need_init = [
        vae.quant_conv,
        vae.quantize,
        vae.post_quant_conv,
        vae.decoder,
    ]
    if isinstance(vae.encoder, Encoder):
        need_init.insert(0, vae.encoder)
    for vv in need_init:
        init_weights(vv, args.vae_init)
    init_weights(disc, args.disc_init)
    vae.quantize.eini(args.vocab_init)
    
    return vae, disc


def init_weights(model, conv_std_or_gain):
    print(f'[init_weights] {type(model).__name__} with {"std" if conv_std_or_gain > 0 else "gain"}={abs(conv_std_or_gain):g}')
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, nn.Embedding):
            nn.init.trunc_normal_(m.weight.data, std=0.02)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            if conv_std_or_gain > 0:
                nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
            else:
                nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            if m.bias is not None: nn.init.constant_(m.bias.data, 0.)
            if m.weight is not None: nn.init.constant_(m.weight.data, 1.)
