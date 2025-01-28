"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import torch
import torch.nn as nn
from torchvision import models

class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, lpips_path, use_dropout=False):    # do not use dropout by default because we use .eval mode by default
        super().__init__()
        # build models
        pretrained = True
        state_dict = torch.load(lpips_path, map_location='cpu')
        for key in state_dict.keys():
            if key.startswith('net.'):
                pretrained = False
        if pretrained:
            print('[LPIPS] warn: will load net from pretrained weights')
        self.net = Vgg16(requires_grad=False, pretrained=pretrained)

        self.lin0 = NetLinLayer(64, use_dropout=use_dropout)
        self.lin1 = NetLinLayer(128, use_dropout=use_dropout)
        self.lin2 = NetLinLayer(256, use_dropout=use_dropout)
        self.lin3 = NetLinLayer(512, use_dropout=use_dropout)
        self.lin4 = NetLinLayer(512, use_dropout=use_dropout)
        self.lins = nn.ModuleList([self.lin0, self.lin1, self.lin2, self.lin3, self.lin4])

        # detach parameters & set to eval mode
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        if not pretrained:
            self.load_state_dict(state_dict, strict=True)
        else:
            self.load_state_dict(state_dict, strict=False)

        # register helper tensors
        self.register_buffer('shift', torch.tensor([-.030, -.088, -.188], dtype=torch.float32).view(1, 3, 1, 1).contiguous())
        self.register_buffer('scale_inv', 1. / torch.tensor([.458, .448, .450], dtype=torch.float32).view(1, 3, 1, 1).contiguous())

    def forward(self, inp, rec):
        """
        :param inp: image for calculating LPIPS loss, [-1, 1]
        :param rec: image for calculating LPIPS loss, [-1, 1]
        :return: lpips loss (scalar)
        """
        B = inp.shape[0]
        inp_and_recs = torch.cat((inp, rec), dim=0).sub(self.shift).mul_(self.scale_inv)  # first use dataset_mean,std to denormalize to [-1, 1], then use vgg_inp_mean,std to normalize again
        inp_and_recs = self.net(inp_and_recs)   # inp_and_recs: List[Tensor], len(inp_and_recs) == 5
        diff = 0.
        for inp_and_rec, lin in zip(inp_and_recs, self.lins):
            diff += lin.model((normalize_tensor(inp_and_rec[:B]) - normalize_tensor(inp_and_rec[B:])).square_()).mean()
        return diff


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if use_dropout else [nn.Identity()]
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1_2 = self.slice1(x)
        h_relu2_2 = self.slice2(h_relu1_2)
        h_relu3_3 = self.slice3(h_relu2_2)
        h_relu4_3 = self.slice4(h_relu3_3)
        h_relu5_3 = self.slice5(h_relu4_3)
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3
        # vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        # out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        # return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sum(x.square(), dim=1, keepdim=True).add_(1e-9).sqrt_()
    return x / (norm_factor + eps)


if __name__ == '__main__':
    from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    l = LPIPS(r'/Users/katz/Downloads/lpips_with_vgg.pth', use_dropout=False)
    s = l.state_dict()
    print(s.keys())

    # for k in ['shift', 'scale_inv']:
    #     s.pop(k)
    # torch.save(s, r'/Users/katz/Downloads/lpips_with_vgg.pth')

