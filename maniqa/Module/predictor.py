import os
import torch
import numpy as np
from typing import Union

from maniqa.Model.maniqa import MANIQA
from maniqa.Method.predict import (
    resizeMinSide,
    randomCropPatches,
    preprocessPatches,
    scorePatches,
    predictFile,
)

# MANIQA 三个官方 checkpoint（pipal / kadid10k / koniq10k）共用同一套结构超参，
# 与 dino_detect 的 DINOV3_COMMON_KWARGS 思路一致：一处定义，需要新变体再加键。
MANIQA_DEFAULT_CONFIG = {
    'embed_dim': 768,
    'num_outputs': 1,
    'dim_mlp': 768,
    'patch_size': 8,
    'img_size': 224,
    'window_size': 4,
    'depths': [2, 2],
    'num_heads': [4, 4],
    'num_tab': 2,
    'scale': 0.8,
}


class Predictor(object):
    '''MANIQA 无参考图像质量打分器（对齐 dino_detect.Module.detector.Detector 的接口风格）。

    一张图打分 = 随机裁 ``num_crops`` 个 ``crop_size`` 方块、各自前向、取平均。
    质量分在 fp32 下标定，故 ``dtype`` 默认 float32（``'auto'`` 亦解析为 float32）。
    '''

    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        model_config: Union[dict, None] = None,
        num_crops: int = 20,
        crop_size: int = 224,
        mean: float = 0.5,
        std: float = 0.5,
        dtype='auto',
        device: str = 'cpu',
    ) -> None:
        self.device = device
        # MANIQA 分数在 fp32 标定，'auto' 不走 bf16，保持与原版数值一致。
        self.dtype = torch.float32 if dtype in ('auto', None) else dtype
        self.num_crops = num_crops
        self.crop_size = crop_size
        self.mean = mean
        self.std = std

        config = dict(MANIQA_DEFAULT_CONFIG if model_config is None else model_config)
        self.crop_size = config.get('img_size', self.crop_size)

        self.model = MANIQA(**config)
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()
        self.model.requires_grad_(False)

        self.is_valid = False
        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print('[ERROR][Predictor::loadModel]')
            print('\t model file not exist!')
            print('\t model_file_path:', model_file_path)
            self.is_valid = False
            return False

        state_dict = torch.load(model_file_path, map_location='cpu', weights_only=True)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # checkpoint 含完整 vit.* 权重，故 pretrained=False + 此处整体加载；
        # strict=False 容忍 vit 分类头等无关键，并把 missing/unexpected 数量打印出来便于核对。
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        print('[INFO][Predictor::loadModel]')
        print('\t model loaded from:', model_file_path)
        if len(missing) > 0:
            print('\t missing keys:', len(missing))
        if len(unexpected) > 0:
            print('\t unexpected keys:', len(unexpected))
        self.is_valid = True
        return True

    @torch.no_grad()
    def predict(self, image_chw: np.ndarray) -> float:
        '''对一张 CHW RGB([0,1]) 图打质量分（裁块 -> 前向 -> 平均）。'''
        image_chw = resizeMinSide(image_chw, self.crop_size + 1)
        patches = randomCropPatches(image_chw, self.num_crops, self.crop_size)
        patches_tensor = preprocessPatches(patches, self.mean, self.std, self.device, self.dtype)
        scores = scorePatches(self.model, patches_tensor)  # [N]
        return float(scores.mean().item())

    @torch.no_grad()
    def predictFile(self, image_file_path: str) -> Union[float, None]:
        return predictFile(self, image_file_path)
