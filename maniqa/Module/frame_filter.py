import os
from typing import List, Tuple

from tqdm import tqdm

from maniqa.Module.predictor import Predictor
from maniqa.Method.video import extractFrames, selectTopRatio, copyFiles


class FrameQualityFilter(object):
    '''视频抽帧 -> MANIQA 打分 -> 保留质量最高的前 ``keep_ratio`` 帧。

    组合 ``Predictor``（打分）与 ``Method.video`` 的原子函数（抽帧 / 选 top / 拷贝），
    每一步都可单独复用。
    '''

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def scoreImageFiles(self, image_file_paths: List[str]) -> List[Tuple[str, float]]:
        '''逐张打分 -> [(path, score), ...]（按输入顺序），跳过读失败的图。'''
        scored: List[Tuple[str, float]] = []
        for image_file_path in tqdm(image_file_paths, desc='scoring frames'):
            score = self.predictor.predictFile(image_file_path)
            if score is None:
                continue
            scored.append((image_file_path, score))
        return scored

    def filterVideo(
        self,
        video_file_path: str,
        output_dir: str,
        target_fps: float = 10.0,
        keep_ratio: float = 0.7,
        frames_dir: str = None,
    ) -> List[Tuple[str, float]]:
        '''抽帧 + 打分 + 保留 top ``keep_ratio`` 帧到 ``output_dir``。

        返回保留帧的 ``[(frame_path, score), ...]``（时间顺序），并在 output_dir 写
        ``scores.txt`` 记录所有帧分数与是否保留。
        '''
        frames_dir = frames_dir or os.path.join(output_dir, '_frames')
        frame_file_paths = extractFrames(video_file_path, frames_dir, target_fps)
        if not frame_file_paths:
            return []

        scored = self.scoreImageFiles(frame_file_paths)
        kept = selectTopRatio(scored, keep_ratio)
        kept_set = set(p for p, _ in kept)
        copyFiles([p for p, _ in kept], output_dir)

        self._writeScores(scored, kept_set, os.path.join(output_dir, 'scores.txt'))

        print('[INFO][FrameQualityFilter::filterVideo]')
        print(f'\t frames={len(frame_file_paths)} kept={len(kept)} (top {keep_ratio:.0%}) -> {output_dir}')
        return kept

    @staticmethod
    def _writeScores(scored: List[Tuple[str, float]], kept_set: set, txt_path: str) -> None:
        '''把每帧 (name, score, kept) 写成清单，按时间顺序。'''
        with open(txt_path, 'w') as f:
            f.write('# frame\tscore\tkept\n')
            for path, score in scored:
                f.write(f'{os.path.basename(path)}\t{score:.6f}\t{int(path in kept_set)}\n')
