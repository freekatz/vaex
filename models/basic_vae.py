import torch
import torch.nn as nn
import torch.nn.functional as F

# this file only provides the 2 modules used in VQVAE
__all__ = ['Encoder', 'Decoder', ]

"""
References: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py
"""


# swish
def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='nearest'))


class Downsample2x(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, pad=(0, 1, 0, 1), mode='constant', value=0))


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None,
                 dropout):  # conv_shortcut=False,  # conv_shortcut: always False in VAE
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 1e-6 else nn.Identity()
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x), inplace=True))
        h = self.conv2(self.dropout(F.silu(self.norm2(h), inplace=True)))
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.C = in_channels

        self.norm = Normalize(in_channels)
        self.qkv = torch.nn.Conv2d(in_channels, 3 * in_channels, kernel_size=1, stride=1, padding=0)
        self.w_ratio = int(in_channels) ** (-0.5)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        qkv = self.qkv(self.norm(x))
        B, _, H, W = qkv.shape  # should be B,3C,H,W
        C = self.C
        q, k, v = qkv.reshape(B, 3, C, H, W).unbind(1)

        # compute attention
        q = q.view(B, C, H * W).contiguous()
        q = q.permute(0, 2, 1).contiguous()  # B,HW,C
        k = k.view(B, C, H * W).contiguous()  # B,C,HW
        w = torch.bmm(q, k).mul_(self.w_ratio)  # B,HW,HW    w[B,i,j]=sum_c q[B,i,C]k[B,C,j]
        w = F.softmax(w, dim=2)

        # attend to values
        v = v.view(B, C, H * W).contiguous()
        w = w.permute(0, 2, 1).contiguous()  # B,HW,HW (first HW of k, second of q)
        h = torch.bmm(v, w)  # B, C,HW (HW of q) h[B,C,j] = sum_i v[B,C,i] w[B,i,j]
        h = h.view(B, C, H, W).contiguous()

        return x + self.proj_out(h)


class MultiHeadAttnBlock(nn.Module):
    def __init__(self, in_channels, head_size=1):
        super().__init__()
        self.in_channels = in_channels
        self.head_size = head_size
        self.att_size = in_channels // head_size
        assert(in_channels % head_size == 0), 'The size of head should be divided by the number of channels.'

        self.norm1 = Normalize(in_channels)
        self.norm2 = Normalize(in_channels)

        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.w_ratio = int(in_channels) ** (-0.5)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)
        self.num = 0

    def forward(self, kv, q=None):
        h_ = kv
        h_ = self.norm1(h_)
        if q is None:
            y = h_
        else:
            y = self.norm2(q)

        q = self.q(y)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b, self.head_size, self.att_size ,h*w)
        q = q.permute(0, 3, 1, 2) # b, hw, head, att

        k = k.reshape(b, self.head_size, self.att_size ,h*w)
        k = k.permute(0, 3, 1, 2)

        v = v.reshape(b, self.head_size, self.att_size ,h*w)
        v = v.permute(0, 3, 1, 2)


        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2,3)

        scale = int(self.att_size)**(-0.5)
        q.mul_(scale)
        w_ = torch.matmul(q, k).mul_(self.w_ratio)
        w_ = F.softmax(w_, dim=3)

        w_ = w_.matmul(v)

        w_ = w_.transpose(1, 2).contiguous() # [b, h*w, head, att]
        w_ = w_.view(b, h, w, -1)
        w_ = w_.permute(0, 3, 1, 2)

        w_ = self.proj_out(w_)

        return kv+w_


def make_attn(in_channels, using_sa=True):
    return AttnBlock(in_channels) if using_sa else nn.Identity()


def make_multi_cross_attn(in_channels, head_size=1, using_sa=True):
    return MultiHeadAttnBlock(in_channels, head_size) if using_sa else nn.Identity()


class Encoder(nn.Module):
    def __init__(
            self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
            dropout=0.0, in_channels=3, head_size=0,
            z_channels, double_z=False, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.downsample_ratio = 2 ** (self.num_resolutions - 1)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.attn_fn = make_attn if head_size == 0 else make_multi_cross_attn

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(self.attn_fn(block_in, head_size, using_sa=True))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample2x(block_in)
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        block_in = ch * in_ch_mult[self.num_resolutions]
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = self.attn_fn(block_in, head_size, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, (2 * z_channels if double_z else z_channels), kernel_size=3, stride=1,
                                        padding=1)

    def forward(self, x):
        # downsampling
        h = self.conv_in(x)
        hs = {}
        hs['in'] = h
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                hs['block_' + str(i_level)] = h
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h)
        hs['block_' + str(self.num_resolutions-1) + '_attn'] = h
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        hs['mid_attn'] = h

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        hs['out'] = h
        return hs


class Decoder(nn.Module):
    def __init__(
            self, *, ch=128, ch_mult=(1, 2, 4, 8), num_res_blocks=2,
            dropout=0.0, in_channels=3, head_size=0, # in_channels: raw img channels
            z_channels, using_sa=True, using_mid_sa=True,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.attn_fn = make_attn if head_size == 0 else make_multi_cross_attn

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = self.attn_fn(block_in, head_size, using_sa=using_mid_sa)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        self.up_fuse = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if i_level == self.num_resolutions - 1 and using_sa:
                    attn.append(self.attn_fn(block_in, head_size, using_sa=True))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample2x(block_in)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z, hs):
        # z to block_in
        # middle
        enc_feat = hs['mid_attn']
        dec_feat = self.mid.block_1(self.conv_in(z))
        h = self.mid.block_2(self.mid.attn_1(dec_feat, enc_feat))

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    if 'block_' + str(i_level) + '_attn' in hs:
                        enc_feat = hs['block_' + str(i_level) + '_attn']
                    else:
                        enc_feat = hs['block_' + str(i_level)]
                    h = self.up[i_level].attn[i_block](h, enc_feat)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.conv_out(F.silu(self.norm_out(h), inplace=True))
        return h
