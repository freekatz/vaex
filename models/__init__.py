import os
import sys
from typing import Tuple

import torch.nn as nn
from torch.cuda import device

from utils import dist_utils
from utils.arg_util import Args
from .vqvae import VectorQuantizer2,VQVAE, DinoDisc, Encoder, Decoder
from .var import VAR


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
                v_patch_nums=args.patch_nums, head_size=1, test_mode=False).to(args.device)
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


def build_var(
        device='cpu',
        # Shared args
         patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
        # VQVAE args
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        # VAR args
        depth=16, shared_aln=False, attn_l2_norm=True,
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,  # init_std < 0: automated
) -> VAR:
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth / 24

    # build models
    vae_local = VQVAE(vocab_size=V, z_channels=Cvae, ch=ch, test_mode=True, share_quant_resi=share_quant_resi,
                      v_patch_nums=patch_nums).to(device)
    var_local = VAR(
        vae_local=vae_local,
        depth=depth, embed_dim=width, num_heads=heads, drop_rate=0., attn_drop_rate=0.,
        drop_path_rate=dpr,
        norm_eps=1e-6, shared_aln=shared_aln, cond_drop_rate=0.1,
        attn_l2_norm=attn_l2_norm,
        patch_nums=patch_nums,
        flash_if_available=flash_if_available,
        fused_if_available=fused_if_available,
    ).to(device)
    return var_local


def build_vae_var_eval(device='cpu') -> Tuple[VQVAE, VAR]:
    # build models
    vae = VQVAE(vocab_size=4096, z_channels=32, ch=160,
                test_mode=True,
                share_quant_resi=4,
                v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),).to(device)

    var = build_var(
        device=device,
        V=4096, Cvae=32, ch=160, share_quant_resi=4,  # hard-coded VQVAE hyperparameters
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        depth=16, shared_aln=False, attn_l2_norm=True,
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
    )
    var.eval()
    return vae, var


def build_vae_disc_var(args: Args) -> Tuple[VQVAE, DinoDisc, VAR]:
    # disable built-in initialization for speed
    for clz in (
            nn.Linear, nn.Embedding,
            nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
            nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm,
            nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
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

    var = build_var(
        device=args.device,
        V=4096, Cvae=32, ch=160, share_quant_resi=4,  # hard-coded VQVAE hyperparameters
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        depth=16, shared_aln=False, attn_l2_norm=True,
        flash_if_available=True, fused_if_available=True,
        init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=-1,
    )
    var.eval()

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

    return vae, disc, var


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
