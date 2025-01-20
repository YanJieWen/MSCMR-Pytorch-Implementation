'''
@File: vssdown.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 01, 2024
@HomePage: https://github.com/YanJieWen
'''


from collections import OrderedDict

import torch
import torch.nn as nn
from .ssm_zoo.ss2d import SS2D
from .sub_blocks import LayerNorm2d,Permute,PatchMerging2D
from .mamba_zoo.visual_mamba import VSSBlock

__all__ = ('VmambaDown')

_NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )

_ACTLAYERS=dict(
            silu=nn.SiLU,
            gelu=nn.GELU,
            relu=nn.ReLU,
            sigmoid=nn.Sigmoid,
        )

class VmambaDown(nn.Module):
    def __init__(self,
                 dims: int = 128,
                 layer_idx:int=0,
                 depths:list=[2,2,15,2],
                 in_dims:int=128,
                 drop_path_rate:float=0.1,
                 use_checkpoint:bool=False,
                 norm_layer:str='ln2d',
                 downsample:str='v3',
                 ssm_d_state:int=1,
                 ssm_ratio:float=2.0,
                 ssm_dt_rank:str='auto',
                 ssm_act_layer:str='silu',
                 ssm_conv:int=3,
                 ssm_conv_bias:bool=False,
                 ssm_drop_rate:float=0.0,
                 ssm_init:str='v0',
                 forward_type:str='v05_noz',
                 mlp_ratio:float=4.0,
                 mlp_act_layer:str='gelu',
                 mlp_drop_rate:float=0.,
                 gmlp:bool=False,
                 _SS2D=SS2D):
        super().__init__()
        self.channel_first = (norm_layer.lower() in ['bn','ln2d'])
        norm_layer =_NORMLAYERS.get(norm_layer.lower(),None)
        ssm_act_layer = _ACTLAYERS.get(ssm_act_layer.lower(),None)
        mlp_act_layer = _ACTLAYERS.get(mlp_act_layer.lower(),None)
        self.num_layers = len(depths)
        _make_downsample = dict(
            v1=PatchMerging2D,
            v2=self._make_downsample,
            v3=self._make_downsample_v3
        ).get(downsample,None)
        self.dims = int(dims)
        dpr = [x.item() for x in torch.linspace(0,drop_path_rate,sum(depths))]
        drop_path = dpr[sum(depths[:layer_idx]):sum(depths[:layer_idx+1])]
        downsample = _make_downsample(
                in_dims,
                self.dims,
                norm_layer = norm_layer,
                channel_first=self.channel_first
            ) if (layer_idx<self.num_layers-1) else nn.Identity()
        self.layers = self._make_layer(
            dim=in_dims,
            drop_path=drop_path,
            use_checkpoint=use_checkpoint,
            norm_layer=norm_layer,
            downsample=downsample,
            channel_first=self.channel_first,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_act_layer=ssm_act_layer,
            ssm_conv=ssm_conv,
            ssm_conv_bias=ssm_conv_bias,
            ssm_drop_rate=ssm_drop_rate,
            ssm_init=ssm_init,
            forward_type=forward_type,
            mlp_ratio=mlp_ratio,
            mlp_act_layer=mlp_act_layer,
            mlp_drop_rate=mlp_drop_rate,
            gmlp=gmlp,
            _SS2D=SS2D
        )
        self.apply(self._init_weights)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm, channel_first=False):
        # if channel first, then Norm and Output are both channel_first
        return nn.Sequential(
            (nn.Identity() if channel_first else Permute(0, 3, 1, 2)),
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
            (nn.Identity() if channel_first else Permute(0, 2, 3, 1)),
            norm_layer(out_dim),
        )

    @staticmethod
    def _make_layer(
            dim=96,
            drop_path=[0.1, 0.1],
            use_checkpoint=False,
            norm_layer=nn.LayerNorm,
            downsample=nn.Identity(),
            channel_first=False,
            # ===========================
            ssm_d_state=16,
            ssm_ratio=2.0,
            ssm_dt_rank='auto',
            ssm_act_layer=nn.SiLU,
            ssm_conv=3,
            ssm_conv_bias=True,
            ssm_drop_rate=0.0,
            ssm_init='v0',
            forward_type='v2',
            # ===========================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate=0.0,
            gmlp=False,
            # ==========================
            _SS2D=SS2D
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                channel_first=channel_first,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                gmlp=gmlp,
                use_checkpoint=use_checkpoint,
                _SS2D=_SS2D,
            ))
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks),
            downsample=downsample
        ))

    def forward(self,x):
        '''

        Args:
            x: b,d,h,w

        Returns:b,2d,h/2,w/2

        '''
        x = self.layers(x)
        return x



# if __name__ == '__main__':
#     x = torch.rand(3,128,224,224)
#     model = VmambaDown()
#     print(model(x).shape)
