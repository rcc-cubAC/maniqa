import os
from typing import List, Tuple, Union

from tqdm import tqdm

from maniqa.Module.predictor import Predictor
from maniqa.Module.keypoint_detector import KeypointDetector
from maniqa.Method.predict import loadImageCHW
from maniqa.Method.video import extractFrames, selectTopRatio, copyFiles
from maniqa.Method.fuse import fuseQualityAndKeypoints


class FrameQualityFilter(object):
    '''视频抽帧 -> 打分 -> 保留最优的前 ``keep_ratio`` 帧。

    打分信号：MANIQA 画质分（``predictor``，必选）；若再传入 SuperPoint
    ``keypoint_detector``，则同时统计每帧关键点数量，并把两路信号各自 min-max
    归一化后按 ``quality_weight`` 融合（``fused = w*质量 + (1-w)*特征点``），
    据此选帧；``min_keypoints`` 可作硬门槛先剔除特征过少的帧。
    只传 ``predictor`` 时退化为纯画质筛选（向后兼容）。
    '''

    def __init__(
        self,
        predictor: Predictor,
        keypoint_detector: Union[KeypointDetector, None] = None,
        quality_weight: float = 0.5,
        min_keypoints: int = 0,
    ) -> None:
        self.predictor = predictor
        self.keypoint_detector = keypoint_detector
        self.quality_weight = quality_weight
        self.min_keypoints = min_keypoints

    def scoreFrames(self, image_file_paths: List[str]) -> List[Tuple[str, float, Union[int, None]]]:
        '''逐帧打分 -> [(path, quality, n_keypoints|None)]；图片只读一次给两个模型复用。'''
        rows: List[Tuple[str, float, Union[int, None]]] = []
        for image_file_path in tqdm(image_file_paths, desc='scoring frames'):
            image_chw = loadImageCHW(image_file_path)
            if image_chw is None:
                continue
            quality = self.predictor.predict(image_chw)
            n_keypoints = self.keypoint_detector.count(image_chw) if self.keypoint_detector is not None else None
            rows.append((image_file_path, quality, n_keypoints))
        return rows

    def filterVideo(
        self,
        video_file_path: str,
        output_dir: str,
        target_fps: float = 10.0,
        keep_ratio: float = 0.7,
        frames_dir: str = None,
    ) -> List[Tuple[str, float]]:
        '''抽帧 + 打分 + 保留 top ``keep_ratio`` 帧到 ``output_dir``。

        返回保留帧的 ``[(frame_path, selection_score), ...]``（时间顺序），并在
        output_dir 写 ``scores.txt`` 记录每帧的质量 / 关键点 / 融合分 / 是否保留。
        '''
        frames_dir = frames_dir or os.path.join(output_dir, '_frames')
        frame_file_paths = extractFrames(video_file_path, frames_dir, target_fps)
        if not frame_file_paths:
            return []

        rows = self.scoreFrames(frame_file_paths)
        scored, fused_by_path = self._selectionScores(rows)

        kept = selectTopRatio(scored, keep_ratio)
        kept_set = set(p for p, _ in kept)
        copyFiles([p for p, _ in kept], output_dir)

        self._writeScores(rows, fused_by_path, kept_set, os.path.join(output_dir, 'scores.txt'))

        mode = 'quality+keypoints' if self.keypoint_detector is not None else 'quality'
        print('[INFO][FrameQualityFilter::filterVideo]')
        print(f'\t mode={mode} frames={len(frame_file_paths)} kept={len(kept)} (top {keep_ratio:.0%}) -> {output_dir}')
        return kept

    def _selectionScores(self, rows):
        '''把每帧 (path, quality, n_keypoints) 折算成用于排序的 (path, selection_score)。

        纯画质：selection_score = quality。
        融合：先按 min_keypoints 硬门槛剔除，再对剩余帧融合质量与关键点数量。
        返回 (scored, fused_by_path)；fused_by_path 仅在融合模式下非空，供清单记录。
        '''
        if self.keypoint_detector is None:
            return [(p, q) for (p, q, _) in rows], {}

        eligible = [(p, q, k) for (p, q, k) in rows if k is not None and k >= self.min_keypoints]
        if not eligible:  # 门槛过严则退回全部，避免一帧都不留
            eligible = [(p, q, k) for (p, q, k) in rows if k is not None]

        fused = fuseQualityAndKeypoints(
            [q for (_, q, _) in eligible],
            [k for (_, _, k) in eligible],
            self.quality_weight,
        )
        scored = [(p, f) for (p, _, _), f in zip(eligible, fused)]
        fused_by_path = {p: f for (p, _), f in zip(scored, fused)}
        return scored, fused_by_path

    def _writeScores(self, rows, fused_by_path, kept_set, txt_path):
        '''按时间顺序写每帧清单：纯画质两列，融合模式四列。'''
        with open(txt_path, 'w') as f:
            if self.keypoint_detector is None:
                f.write('# frame\tquality\tkept\n')
                for path, quality, _ in rows:
                    f.write(f'{os.path.basename(path)}\t{quality:.6f}\t{int(path in kept_set)}\n')
            else:
                f.write('# frame\tquality\tkeypoints\tfused\tkept\n')
                for path, quality, n_keypoints in rows:
                    fused = fused_by_path.get(path, float('nan'))
                    kp = -1 if n_keypoints is None else n_keypoints
                    f.write(f'{os.path.basename(path)}\t{quality:.6f}\t{kp}\t{fused:.6f}\t{int(path in kept_set)}\n')
