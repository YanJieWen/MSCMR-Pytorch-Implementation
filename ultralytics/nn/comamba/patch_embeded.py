'''
@File: patch_embeded.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 01, 2024
@HomePage: https://github.com/YanJieWen
'''


import torch
import torch.nn as nn

from .sub_blocks import LayerNorm2d,Permute

__all__ = (
    'Patchembeded'
)

_NORMLAYERS = dict(ln=nn.LayerNorm,
                   ln2d=LayerNorm2d,
                   bn=nn.BatchNorm2d)
class Patchembeded(nn.Module):
    def __init__(self,
                 in_chans:int=3,
                 out_chans:int=128,
                 patch_size:int=4,
                 patch_norm:bool=True,
                 norm_layer:str='ln2d',
                 version:str='v1'):
        super().__init__()
        channel_first = (norm_layer.lower() in ['bn','ln2d'])
        norm_layer = _NORMLAYERS.get(norm_layer.lower(),None)
        _make_patch_embed = dict(
            v1 = self._make_patch_embed,
            v2 = self._make_patch_embed_v2,
        ).get(version,None)
        assert _make_patch_embed is not None, f'{version} is only supported by v1,v2'
        self.patch_embed = _make_patch_embed(in_chans,out_chans,patch_size,patch_norm,
                                             norm_layer,channel_first)

    @staticmethod
    def _make_patch_embed(in_chans=3,embed_dim=96,patch_size=4,patch_norm=True,
                          norm_layer=nn.LayerNorm,channel_first=True):
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_patch_embed_v2(in_chans,embed_dim=96,patch_size=4, patch_norm=True,
                             norm_layer=nn.LayerNorm, channel_first=True):
        stride = patch_size // 2
        kernel_size = stride + 1
        padding = 1
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (nn.Identity() if (channel_first or (not patch_norm)) else Permute(0, 3, 1, 2)),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    def forward(self,x:torch.Tensor):
        '''

        Args:
            x: b,3,h,w

        Returns: b,d,h,w

        '''
        x = self.patch_embed(x)
        return x


