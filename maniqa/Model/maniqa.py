import timm
import torch
import torch.nn as nn

from einops import rearrange
from timm.models.vision_transformer import Block

from maniqa.Model.swin import SwinTransformer
from maniqa.Model.Layer.ta_block import TABlock
from maniqa.Model.Layer.save_output import SaveOutput


class MANIQA(nn.Module):
    '''Multi-dimension Attention Network for No-Reference IQA。

    ViT 提取多层 patch 特征 -> TAB（通道注意力）+ Scale Swin（空间注意力）交替增强
    -> 双分支按 patch 加权得到一张图的质量分。``forward`` 输出 [B] 的质量分，
    设备跟随输入（不再硬编码 ``.cuda()``）。
    '''

    def __init__(self, embed_dim=72, num_outputs=1, patch_size=8, drop=0.1,
                 depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                 img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)

        self.vit = timm.create_model('vit_base_patch8_224', pretrained=False)
        self.save_output = SaveOutput()
        self._registerHooks()

        self.tablock1 = nn.ModuleList([TABlock(self.input_size ** 2) for _ in range(num_tab)])

        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale,
        )

        self.tablock2 = nn.ModuleList([TABlock(self.input_size ** 2) for _ in range(num_tab)])

        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale,
        )

        self.fc_score = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU(),
        )
        self.fc_weight = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid(),
        )

    def _registerHooks(self):
        '''给 ViT 的每个 Block 挂上 SaveOutput，前向后即可按层取出中间特征。'''
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                layer.register_forward_hook(self.save_output)

    def extract_feature(self, save_output):
        '''拼接 ViT 第 6~9 个 Block 的 patch token 特征（去掉 cls token）。'''
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        return torch.cat((x6, x7, x8, x9), dim=2)

    def forward(self, x):
        self.vit(x)
        x = self.extract_feature(self.save_output)
        self.save_output.clear()

        # stage 1: TAB（通道注意力） -> conv 降维 -> Scale Swin（空间注意力）
        x = rearrange(x, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)
        x = self.swintransformer1(x)

        # stage 2
        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)
        x = self.swintransformer2(x)

        # 双分支按 patch 加权：score = sum(f * w) / sum(w)，对每个样本在 patch 维归约
        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)
        f = self.fc_score(x)                       # [B, N, num_outputs]
        w = self.fc_weight(x)                      # [B, N, num_outputs]
        score = (f * w).sum(dim=1) / w.sum(dim=1)  # [B, num_outputs]
        return score.squeeze(-1)                   # [B]
