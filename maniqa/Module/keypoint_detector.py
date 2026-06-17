import os
import torch
import numpy as np
from typing import Union

from maniqa.Model.superpoint import SuperPoint
from maniqa.Method.detect_keypoints import imageToTensor, countKeypoints, countKeypointsFile

# 与 rpautrat SuperPoint.default_conf 对齐，需要调灵敏度时覆盖（如 detection_threshold / nms_radius）。
SUPERPOINT_DEFAULT_CONF = {
    'nms_radius': 4,
    'detection_threshold': 0.005,
    'remove_borders': 4,
    'max_num_keypoints': None,
}


class KeypointDetector(object):
    '''SuperPoint 关键点检测器（对齐 dino_detect.Module.detector.Detector 的接口风格）。

    主要给筛帧提供"特征点数量"信号：``count(image)`` / ``countFile(path)``；
    也可 ``detect(image)`` 拿完整输出（keypoints / scores / descriptors）。
    '''

    def __init__(
        self,
        model_file_path: Union[str, None] = None,
        model_conf: Union[dict, None] = None,
        dtype='auto',
        device: str = 'cpu',
    ) -> None:
        self.device = device
        self.dtype = torch.float32 if dtype in ('auto', None) else dtype

        conf = dict(SUPERPOINT_DEFAULT_CONF if model_conf is None else model_conf)
        self.model = SuperPoint(**conf)
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()
        self.model.requires_grad_(False)

        self.is_valid = False
        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path: str) -> bool:
        if not os.path.exists(model_file_path):
            print('[ERROR][KeypointDetector::loadModel]')
            print('\t model file not exist!')
            print('\t model_file_path:', model_file_path)
            self.is_valid = False
            return False

        state_dict = torch.load(model_file_path, map_location='cpu', weights_only=True)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        print('[INFO][KeypointDetector::loadModel]')
        print('\t model loaded from:', model_file_path)
        if len(missing) > 0:
            print('\t missing keys:', len(missing))
        if len(unexpected) > 0:
            print('\t unexpected keys:', len(unexpected))
        self.is_valid = True
        return True

    @torch.no_grad()
    def detect(self, image_chw: np.ndarray) -> dict:
        '''对一张 CHW RGB([0,1]) 图跑 SuperPoint -> 完整输出 dict。'''
        image = imageToTensor(image_chw, self.device, self.dtype)
        return self.model({"image": image})

    @torch.no_grad()
    def count(self, image_chw: np.ndarray) -> int:
        '''关键点数量（特征丰富度信号）。'''
        return countKeypoints(self.detect(image_chw))

    @torch.no_grad()
    def countFile(self, image_file_path: str) -> Union[int, None]:
        return countKeypointsFile(self, image_file_path)
