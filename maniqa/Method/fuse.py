from typing import List


def minMaxNormalize(values: List[float]) -> List[float]:
    '''线性归一化到 [0, 1]；所有值相等时返回全 0.5（不偏向任何一方）。

    把量纲差异很大的信号（如质量分 ~[0.1,0.5] 与关键点数 ~[0,1000]）拉到同一尺度，
    使融合权重可解释。
    '''
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return [0.5 for _ in values]
    span = hi - lo
    return [(v - lo) / span for v in values]


def fuseQualityAndKeypoints(
    qualities: List[float],
    keypoint_counts: List[float],
    quality_weight: float = 0.5,
) -> List[float]:
    '''各自 min-max 归一化后按 quality_weight 加权求和 -> 每帧的融合分。

    fused = w * quality_norm + (1 - w) * keypoints_norm，w = quality_weight。
    w=1 即纯质量，w=0 即纯特征点数量。
    '''
    q_norm = minMaxNormalize([float(q) for q in qualities])
    k_norm = minMaxNormalize([float(c) for c in keypoint_counts])
    w = float(quality_weight)
    return [w * q + (1.0 - w) * k for q, k in zip(q_norm, k_norm)]
