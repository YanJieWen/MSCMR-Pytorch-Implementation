'''
@File: mambayolo.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 8月 05, 2024
@HomePage: https://github.com/YanJieWen
'''

from typing import Any

import torch
import torch.nn as nn

from .sub_blocks import LayerNorm2d,Mlp,gMlp,DropPath
from .ssm_zoo.ss2d import SS2D

from ultralytics.nn.modules import C2f,Conv


class RIGBlock(nn.Module):
    def __init__(self,in_dims,hidden_dims=None,out_dims=None,act_layer=nn.GELU,drop=0.,channels_first=True):
        super().__init__()
        out_dims = out_dims or in_dims
        hidden_dims = hidden_dims or in_dims
        hidden_dims = int(2*hidden_dims/3)
        self.fc1 = nn.Conv2d(in_dims,hidden_dims*2,kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_dims,hidden_dims,kernel_size=3,stride=1,padding=1,bias=True,groups=hidden_dims)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_dims,out_dims,kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self,x):#残差倒置块
        x,v = self.fc1(x).chunk(2,dim=1)
        x = self.act(self.dwconv(x) + x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSBlock(nn.Module):
    def __init__(self,in_dims,hidden_dims=None,n=1,act_layer=nn.GELU,drop=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(in_dims,hidden_dims,kernel_size=3,padding=3//2,groups=hidden_dims)
        self.norm = nn.BatchNorm2d(hidden_dims)
        self.fc2 = nn.Conv2d(hidden_dims,hidden_dims,kernel_size=1,padding=0)
        self.act = act_layer()
        self.fc3 = nn.Conv2d(hidden_dims,in_dims,kernel_size=1,padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self,x):
        input = x
        x = self.norm(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.fc3(x)
        x = input+self.drop(x)
        return x

class CIB(nn.Module):
    def __init__(self,c1,c2,shortcut=True,e=0.5):
        super().__init__()
        c_ = int(c2*e)
        self.cv1 = nn.Sequential(
            Conv(c1,c1,3,g=c1),
            Conv(c1,2*c_,1),
            Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2*c_,c2,1),
            Conv(c2,c2,3,g=c2)
        )
        self.add = shortcut and c1==c2

    def forward(self,x):
        return x+self.cv1(x) if self.add else self.cv1(x)

class C2fCIB(C2f):
    def __init__(self,c1,c2,n,shortcut=True,g=1,e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c,self.c,shortcut,e=1.0) for _ in range(n))

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)#降低qk的维度，仅进行部分维度的注意力计算-->空间注意力，仅发生在最低分辨率
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        attn = (
            (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

class PSA(nn.Module):

    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert (c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(
            Conv(self.c, self.c * 2, 1),
            Conv(self.c * 2, self.c, 1, act=False)
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


_NORMLAYER = dict(
    ln=nn.LayerNorm,
    ln2d=LayerNorm2d,
    bn=nn.BatchNorm2d,
)

_MLP = dict(
    mlp=Mlp,
    gmlp=gMlp,
    rg=RIGBlock,
    attn=PSA,
)

_SPATIAL = dict(
    ls=LSBlock,
    c2fcib=C2fCIB,
)


class CoMamabaX(nn.Module):
    def __init__(self,
                 in_dims:int=0,
                 hidden_dims:int=0,
                 num_layers:int=1,
                 proj_type: str = 'ls',
                 forward_type: str = 'v2',
                 mlp_type='rg',
                 norm_layer:str='ln2d',
                 drop_path: float = 0.,
                 #ss2d
                 ssm_d_state:int=16,
                 ssm_ratio:float=2.0,
                 ssm_dt_rank:Any='auto',
                 ssm_act_layer:nn.Module=nn.SiLU,
                 ssm_conv:int=3,
                 ssm_conv_bias:bool=True,
                 ssm_drop_rate:float=0.,
                 ssm_init:bool='v0',
                 #mlp
                 mlp_ratio:float=4.0,
                 mlp_act_layer:nn.Module=nn.GELU,
                 mlp_drop_rate:float=0.0,

                 post_norm:bool=False,):
        super().__init__()
        #参数定义
        self.ssm_branch = ssm_ratio>0
        self.mlp_branch = mlp_ratio>0
        channel_first = (norm_layer.lower() in ['ln2d','bn'])
        norm_layer = _NORMLAYER.get(norm_layer.lower(),nn.Identity)
        spatial_layer = _SPATIAL.get(proj_type.lower(),nn.Identity)
        mlp_layer = _MLP.get(mlp_type.lower(),nn.Identity)
        self.post_norm = post_norm

        self.drop_path = DropPath(drop_path)


        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_dims,hidden_dims,kernel_size=1,stride=1,padding=0,bias=True),
        nn.BatchNorm2d(hidden_dims),
        nn.SiLU())

        self.lsblock = spatial_layer(hidden_dims, hidden_dims,num_layers)

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dims)
            self.ss2d = nn.Sequential(*(SS2D(
                d_model=hidden_dims,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            ) for _ in range(num_layers)))
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dims)
            mlp_hidde_dim = int(hidden_dims*mlp_ratio) if (mlp_type in ('mlp','gmlp')) else hidden_dims
            args = [hidden_dims,mlp_hidde_dim]
            if mlp_type in ('mlp','gmlp','rg'):
                args.extend([None,mlp_act_layer,mlp_drop_rate,channel_first])
            else:
                pass
            self.mlp = mlp_layer(*args)
    def forward(self,x):
        input = self.proj_conv(x)
        x1 = self.lsblock(input)
        if self.ssm_branch:
            if self.post_norm:
                input = input+self.drop_path(self.norm(self.ss2d(x1)))
            else:
                input = input+self.drop_path(self.ss2d(self.norm(x1)))
        if self.mlp_branch:
            if self.post_norm:
                input = input+self.drop_path(self.norm2(self.mlp(input)))
            else:
                input = input+self.drop_path(self.mlp(self.norm2(input)))

        return input

class CoMamabaV(nn.Module):
    def __init__(self,
                 in_dims:int=0,
                 hidden_dims:int=0,
                 num_layers:int=1,
                 proj_type: str = 'ls',
                 forward_type: str = 'v2',
                 mlp_type='rg',
                 norm_layer:str='ln2d',
                 drop_path: float = 0.,
                 #ss2d
                 ssm_d_state:int=16,
                 ssm_ratio:float=2.0,
                 ssm_dt_rank:Any='auto',
                 ssm_act_layer:nn.Module=nn.SiLU,
                 ssm_conv:int=3,
                 ssm_conv_bias:bool=True,
                 ssm_drop_rate:float=0.,
                 ssm_init:bool='v0',
                 #mlp
                 mlp_ratio:float=4.0,
                 mlp_act_layer:nn.Module=nn.GELU,
                 mlp_drop_rate:float=0.0,

                 post_norm:bool=False,):
        super().__init__()
        #参数定义
        self.ssm_branch = ssm_ratio>0
        self.mlp_branch = mlp_ratio>0
        channel_first = (norm_layer.lower() in ['ln2d','bn'])
        norm_layer = _NORMLAYER.get(norm_layer.lower(),nn.Identity)
        spatial_layer = _SPATIAL.get(proj_type.lower(),nn.Identity)
        mlp_layer = _MLP.get(mlp_type.lower(),nn.Identity)
        self.post_norm = post_norm

        self.drop_path = DropPath(drop_path)


        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_dims,hidden_dims,kernel_size=1,stride=1,padding=0,bias=True),
        nn.BatchNorm2d(hidden_dims),
        nn.SiLU())

        self.lsblock = spatial_layer(hidden_dims, hidden_dims,num_layers)

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dims)
            self.op = SS2D(
                d_model=hidden_dims,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                channel_first=channel_first,
            )
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dims)
            mlp_hidde_dim = int(hidden_dims*mlp_ratio) if (mlp_type in ('mlp','gmlp')) else hidden_dims
            args = [hidden_dims,mlp_hidde_dim]
            if mlp_type in ('mlp','gmlp','rg'):
                args.extend([None,mlp_act_layer,mlp_drop_rate,channel_first])
            else:
                pass
            self.mlp = mlp_layer(*args)
    def forward(self,x):
        input = self.proj_conv(x)
        x1 = self.lsblock(input)
        if self.ssm_branch:
            if self.post_norm:
                input = input+self.drop_path(self.norm(self.op(x1)))
            else:
                input = input+self.drop_path(self.op(self.norm(x1)))
        if self.mlp_branch:
            if self.post_norm:
                input = input+self.drop_path(self.norm2(self.mlp(input)))
            else:
                input = input+self.drop_path(self.mlp(self.norm2(input)))

        return input


# if __name__ == '__main__':
#     x = torch.rand(1,256,20,20).to('cuda:0')
#     model = CoMamaba(256,512,proj_type='ls',forward_type='v052dc_noz',mlp_type='attn').to('cuda:0')
#     print(model(x).shape)

