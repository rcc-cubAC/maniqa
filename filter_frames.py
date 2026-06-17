import os
import argparse

from maniqa.Module.predictor import Predictor
from maniqa.Module.keypoint_detector import KeypointDetector
from maniqa.Module.frame_filter import FrameQualityFilter
from maniqa.Method.predict import setupSeed

# 与 dino-detect 统一的权重约定位置：~/chLi/Model/<MODEL>/<weight>
DEFAULT_CKPT = os.path.expanduser('~/chLi/Model/MANIQA/ckpt_koniq10k.pt')
DEFAULT_SP_CKPT = os.path.expanduser('~/chLi/Model/SuperPoint/superpoint_v6_from_tf.pth')


def main():
    parser = argparse.ArgumentParser(
        description='抽帧 + 打分（MANIQA 画质 + SuperPoint 特征点数量），保留最优的前若干比例帧。')
    parser.add_argument('video', help='输入视频文件')
    parser.add_argument('output_dir', help='保留帧 + scores.txt 的输出目录')
    parser.add_argument('--ckpt', default=DEFAULT_CKPT, help=f'MANIQA checkpoint，默认 {DEFAULT_CKPT}')
    parser.add_argument('--sp-ckpt', default=DEFAULT_SP_CKPT, help=f'SuperPoint 权重，默认 {DEFAULT_SP_CKPT}')
    parser.add_argument('--fps', type=float, default=10.0, help='抽帧帧率 (default 10)')
    parser.add_argument('--keep', type=float, default=0.7, help='按融合分保留的比例 (default 0.7)')
    parser.add_argument('--quality-weight', type=float, default=0.5,
                        help='融合权重 w：fused=w*画质+(1-w)*特征点 (default 0.5)')
    parser.add_argument('--min-keypoints', type=int, default=0,
                        help='关键点数量硬门槛，低于此值的帧先剔除 (default 0=不启用)')
    parser.add_argument('--no-keypoints', action='store_true', help='只用画质筛选，不考虑特征点')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-crops', type=int, default=20, help='每帧随机裁块数 (default 20)')
    parser.add_argument('--seed', type=int, default=20)
    args = parser.parse_args()

    setupSeed(args.seed)

    predictor = Predictor(
        model_file_path=args.ckpt,
        num_crops=args.num_crops,
        device=args.device,
    )

    keypoint_detector = None
    if not args.no_keypoints:
        if os.path.exists(args.sp_ckpt):
            keypoint_detector = KeypointDetector(model_file_path=args.sp_ckpt, device=args.device)
        else:
            print('[WARN][filter_frames] SuperPoint 权重不存在，退回纯画质筛选：', args.sp_ckpt)

    frame_filter = FrameQualityFilter(
        predictor,
        keypoint_detector=keypoint_detector,
        quality_weight=args.quality_weight,
        min_keypoints=args.min_keypoints,
    )
    frame_filter.filterVideo(
        args.video,
        args.output_dir,
        target_fps=args.fps,
        keep_ratio=args.keep,
    )


if __name__ == '__main__':
    main()
