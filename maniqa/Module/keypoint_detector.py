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
        is_offload_cpu: bool = False,
    ) -> None:
        self.device = device
        self.dtype = torch.float32 if dtype in ('auto', None) else dtype
        # offload 模式：权重常驻 CPU，``count()``/``detect()`` 推理窗口内才上 GPU；
        # ``toGPU()`` 可把模型钉在 GPU 上做批量推理。
        self.is_offload_cpu = bool(is_offload_cpu)

        conf = dict(SUPERPOINT_DEFAULT_CONF if model_conf is None else model_conf)
        load_device = 'cpu' if self.is_offload_cpu else self.device
        self.model = SuperPoint(**conf)
        self.model = self.model.to(load_device, dtype=self.dtype)
        self.model.eval()
        self.model.requires_grad_(False)

        self.is_valid = False
        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def _isModelOnDevice(self) -> bool:
        '''模型参数是否已在 self.device。'''
        try:
            param = next(self.model.parameters())
        except StopIteration:
            return True
        return param.device.type == torch.device(self.device).type

    def _ensureModelOnDevice(self) -> bool:
        '''确保模型在 self.device，返回本次是否发生了搬运。'''
        if self._isModelOnDevice():
            return False
        self.model = self.model.to(self.device)
        return True

    def _offloadModelToCPU(self) -> None:
        '''把模型卸回 CPU 并清显存。'''
        self.model = self.model.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def toGPU(self) -> 'KeypointDetector':
        '''把模型常驻到 self.device（批量推理前调用）。'''
        self.model = self.model.to(self.device)
        return self

    def toCPU(self) -> 'KeypointDetector':
        '''把模型卸回 CPU。'''
        self._offloadModelToCPU()
        return self

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
        moved_to_gpu_this_call = self._ensureModelOnDevice()
        try:
            image = imageToTensor(image_chw, self.device, self.dtype)
            return self.model({"image": image})
        finally:
            if moved_to_gpu_this_call and self.is_offload_cpu:
                self._offloadModelToCPU()

    @torch.no_grad()
    def count(self, image_chw: np.ndarray) -> int:
        '''关键点数量（特征丰富度信号）。'''
        return countKeypoints(self.detect(image_chw))

    @torch.no_grad()
    def countFile(self, image_file_path: str) -> Union[int, None]:
        return countKeypointsFile(self, image_file_path)
