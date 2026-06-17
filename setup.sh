pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

pip install timm einops opencv-python numpy tqdm

# 可选：仅 `maniqa.API.image_iqa`（按 Camera 批量打分）需要 camera-control。
# 它不是 pip 包，是自研库 github.com/565353780/camera-control —— 与本仓库平级 clone
# 后装为可编辑依赖（或加入 PYTHONPATH）。其余功能（demo.py / filter_frames.py）无需它。
# cd .. && git clone https://github.com/565353780/camera-control.git && cd camera-control && ./setup.sh
