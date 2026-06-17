import os
import cv2
import shutil
from typing import List, Tuple


def extractFrames(
    video_file_path: str,
    output_dir: str,
    target_fps: float,
    prefix: str = 'frame',
) -> List[str]:
    '''按 target_fps 等间隔从视频抽帧存为 PNG -> 返回帧路径列表（时间顺序）。

    通过 ``step = round(src_fps / target_fps)`` 每隔 step 帧取一帧，无需 ffmpeg
    （OpenCV 自带解码后端）。失败返回空列表。
    '''
    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print('[ERROR][extractFrames]')
        print('\t failed to open video!')
        print('\t video_file_path:', video_file_path)
        return []

    src_fps = cap.get(cv2.CAP_PROP_FPS) or float(target_fps)
    step = max(1, int(round(src_fps / float(target_fps))))

    os.makedirs(output_dir, exist_ok=True)
    frame_file_paths: List[str] = []
    idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            out_path = os.path.join(output_dir, f'{prefix}_{kept:05d}.png')
            cv2.imwrite(out_path, frame)
            frame_file_paths.append(out_path)
            kept += 1
        idx += 1
    cap.release()

    print('[INFO][extractFrames]')
    print(f'\t src_fps={src_fps:.3f} step={step} -> kept {len(frame_file_paths)} frames @ ~{target_fps}fps')
    return frame_file_paths


def selectTopRatio(
    scored: List[Tuple[str, float]],
    ratio: float,
) -> List[Tuple[str, float]]:
    '''按分数降序保留前 ``ratio`` 比例，结果仍按输入（时间）顺序返回。

    scored: ``[(path, score), ...]``。至少保留 1 个。
    '''
    if not scored:
        return []
    n_keep = max(1, int(round(len(scored) * float(ratio))))
    ranked = sorted(scored, key=lambda ps: ps[1], reverse=True)
    keep_paths = set(p for p, _ in ranked[:n_keep])
    return [(p, s) for (p, s) in scored if p in keep_paths]


def copyFiles(file_paths: List[str], output_dir: str) -> List[str]:
    '''把若干文件复制到 output_dir -> 返回新路径列表。'''
    os.makedirs(output_dir, exist_ok=True)
    out_paths: List[str] = []
    for src in file_paths:
        dst = os.path.join(output_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        out_paths.append(dst)
    return out_paths
