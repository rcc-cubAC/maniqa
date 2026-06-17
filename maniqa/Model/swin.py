import torch
import torch.nn as nn

from einops import rearrange

from maniqa.Model.Layer.basic_layer import BasicLayer


class SwinTransformer(nn.Module):
    '''MANIQA 中的 Scale Swin Transformer：多个 BasicLayer 残差堆叠，残差按 scale 缩放。'''

    def __init__(self, patches_resolution, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 embed_dim=256, drop=0.1, drop_rate=0., drop_path_rate=0.1, dropout=0., window_size=7,
                 dim_mlp=1024, qkv_bias=True, qk_scale=None, attn_drop_rate=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, scale=0.8, **kwargs):
        super().__init__()
        self.scale = scale
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.dropout = nn.Dropout(p=drop)
        self.num_features = embed_dim
        self.num_layers = len(depths)
        self.patches_resolution = (patches_resolution[0], patches_resolution[1])
        self.downsample = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1)
        # stochastic depth 衰减率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=self.embed_dim,
                input_resolution=patches_resolution,
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                dim_mlp=dim_mlp,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=dropout,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsample,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

    def forward(self, x):
        x = self.dropout(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for layer in self.layers:
            _x = x
            x = layer(x)
            x = self.scale * x + _x
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.patches_resolution[0], w=self.patches_resolution[1])
        return x
