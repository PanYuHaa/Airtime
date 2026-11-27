# syntax=docker/dockerfile:1.7
#
# Airtime 运行镜像
#
#   docker build -t airtime:lean .                            # 轻量档（2D 流水线）
#   docker build --build-arg WITH_GEMX=1 -t airtime:full .   # 全量档（+ torch/GEM-X/SAM2）
#
# 基镜像必须用 Ubuntu 24.04：src/airtime/pipeline/m0_prep.py 用 `-fps_mode cfr`，
# 需要 ffmpeg >= 5.1；22.04 自带 4.4.2 会直接失败。这里从静态 FFmpeg 6.1.1
# 镜像取 ffmpeg + ffprobe（只复制两个静态二进制），避免 apt 版 ffmpeg 把
# Mesa/LLVM/SDL 整套图形栈拉进来（实测 apt ffmpeg 层 619 MB，静态版 187 MB）。
# M0 的 `-fps_mode cfr` 和测试里对 ffprobe 的调用都已在本镜像内实测通过。
FROM mwader/static-ffmpeg:6.1.1 AS ffmpeg-static

# 用 CUDA base 档（288 MB，只带 cudart）。原计划的 runtime 档实测 3.25 GB，
# 其中 cusparse/NCCL/cusolver/NPP/NVRTC 共 ~1.8 GB 是 ORT 与 torch wheel 都不
# 需要的；而删这些库只会多出一个 whiteout 层，不会让镜像变小（已实测）。
# 所以改为「按需取库」：cuBLAS/cuFFT/cuRAND/cuDNN 全部走 pip 的 Nvidia wheel，
# 版本与 cudart 12.8 对齐；三套库的 DT_NEEDED 已实测可满足 ORT 的 CUDA EP。
FROM nvidia/cuda:12.8.1-base-ubuntu24.04
COPY --from=ffmpeg-static /ffmpeg /ffprobe /usr/local/bin/

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

# ---------------------------------------------------------------- 系统依赖
# 不装 build-essential：SAM2 的 CUDA 扩展在 torch 环境里用 SAM2_BUILD_CUDA=0 跳过编译。
# 不装 libgl1 / libglib2.0-0t64：主环境统一用 opencv-python-headless，已实测
# cv2 只依赖其 wheel 自带的 .libs；Python 3.12 的 venv 会自带 ensurepip，不需要
# 系统 python3-pip。git 仍保留给全量档拉 GEM-X / SAM2，curl 供运行期下载权重。
RUN sed -i \
        -e 's@http://archive.ubuntu.com/ubuntu@https://mirrors.aliyun.com/ubuntu@g' \
        -e 's@http://security.ubuntu.com/ubuntu@https://mirrors.aliyun.com/ubuntu@g' \
        /etc/apt/sources.list.d/ubuntu.sources \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        ca-certificates \
        git \
        curl \
 && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------- 主环境（2D 流水线）
# 国内镜像（实测 39 MB/s）写进 venv 的 pip 配置；torch 的 cu128 索引单独指定。
# venv 自带 ensurepip，不依赖系统 python3-pip。
# ORT 1.23.2 的 CUDA provider 实测 DT_NEEDED：libcublasLt/libcublas/libcudart/
# libcufft/libcurand/libcudnn，按此只装这几套，版本锚定 CUDA 12.8。
# cuBLAS 用 12.4.5.8（解压 527 MB）而非 12.8.4.1（829 MB）：两者都是 CUDA 12 的
# soname（libcublas.so.12 / libcublasLt.so.12），ORT 与实际推理均已实测可用。
# cuDNN 选 9.2.0.82（解压 838.7 MB）而非 9.28（1.11 GB），ORT 两者都可用。
# libcudnn_adv（RNN/LSTM/attention，230 MB）主 venv 用不到：2D 流水线只有
# 卷积网络（YOLOX / RTMPose），全量档的 torch 用自己 venv 里的 cuDNN。
# 删除与安装放在同一层，才能真的减小镜像（跨层删除只会多一个 whiteout）。
# --no-deps 用于 cuDNN：它的 metadata 只声明 nvidia-cublas-cu12，下面已显式装同版本。
# pip 缓存走 BuildKit cache mount：不进镜像层，但重建时不用重新下载。
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m venv /opt/venvs/main \
 && /opt/venvs/main/bin/pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
 && /opt/venvs/main/bin/pip config set global.trusted-host mirrors.aliyun.com \
 && /opt/venvs/main/bin/pip install --upgrade pip setuptools wheel \
 && /opt/venvs/main/bin/pip install \
        nvidia-cublas-cu12==12.4.5.8 \
        nvidia-cufft-cu12==11.3.3.83 \
        nvidia-curand-cu12==10.3.9.90 \
 && /opt/venvs/main/bin/pip install --no-deps nvidia-cudnn-cu12==9.2.0.82 \
 && rm -f /opt/venvs/main/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn_adv.so.9*

WORKDIR /app

# editable 安装 → cfg.REPO_ROOT 解析到 /app（T4 的 CI 契约依赖这一点）
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# rtmlib 同时声明 opencv-python + opencv-contrib-python（实测会装出两套 cv2），
# 项目只 import cv2。先按上游依赖装，再统一收敛到 headless 版本；CPU 版
# onnxruntime 也必须先卸载，再装回 GPU 版，避免 .so 与 dist-info 残留。
# headless 选 5.1.0.95 与仓库 pyproject 的 numpy>=1.24 实际解析一致（numpy 2.5），
# 4.12 会带去 numpy<2.3 的元数据冲突。
# editable 安装保留，cfg.REPO_ROOT 必须解析到 /app（T4 CI 契约）。
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venvs/main/bin/pip install -e . \
 && /opt/venvs/main/bin/pip install rtmlib \
 && /opt/venvs/main/bin/pip uninstall -y onnxruntime \
 && /opt/venvs/main/bin/pip install --force-reinstall --no-deps \
        onnxruntime-gpu==1.23.2 coloredlogs humanfriendly sympy "mpmath<1.4,>=1.1.0" \
 && /opt/venvs/main/bin/pip uninstall -y opencv-python opencv-contrib-python \
 && /opt/venvs/main/bin/pip install --no-deps opencv-python-headless==5.1.0.95 \
 && /opt/venvs/main/bin/pip install pytest \
 && find /opt/venvs/main -type d -name __pycache__ -prune -exec rm -rf {} + \
 && find /opt/venvs/main/lib -path '*/onnxruntime/capi/libonnxruntime.so.*' ! -name 'libonnxruntime.so.1.23.2' -delete \
 && test ! -e /opt/venvs/main/lib/python3.12/site-packages/onnxruntime-1.32.0.dist-info \
 && /opt/venvs/main/bin/python -c "import cv2, onnxruntime as o; assert o.__version__ == '1.23.2'; print('venv cleaned:', cv2.__version__, o.__version__)"

COPY configs ./configs
COPY tests ./tests
COPY scripts ./scripts

# ---------------------------------------------------------------- 运行时路径
# 注意 AIRTIME_SAM2_CONFIG 不在这里设：cfg.resolve_sam2_config() 是「有环境变量就
# 无条件返回」的语义，全局设上会让 tests/test_eval_and_contracts 里的自动匹配用例
# 永远失败。它由 entrypoint 在检测到 /opt/sam2（全量档）时才导出默认值。
ENV PATH="/opt/venvs/main/bin:${PATH}" \
    AIRTIME_MODELS=/models \
    AIRTIME_DATA=/data \
    AIRTIME_OUTPUT=/runs \
    AIRTIME_GEMX=/opt/GEM-X \
    AIRTIME_SAM2=/opt/sam2 \
    AIRTIME_GEMX_PYTHON=/opt/venvs/torch/bin/python \
    AIRTIME_SAM2_PYTHON=/opt/venvs/torch/bin/python \
    LD_LIBRARY_PATH="/opt/venvs/main/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venvs/main/lib/python3.12/site-packages/nvidia/cublas/lib:/opt/venvs/main/lib/python3.12/site-packages/nvidia/cufft/lib:/opt/venvs/main/lib/python3.12/site-packages/nvidia/curand/lib:/usr/local/cuda/lib64"

# pip 把 CUDA 库放在 site-packages/nvidia/*/lib，动态链接器不会自动搜索；
# 写进 ld.so.conf.d 并 ldconfig，即使 compose 覆盖了 LD_LIBRARY_PATH 也能加载。
# 同时用 ctypes 逐库 dlopen 一次，避免镜像构建成功、运行时却缺 CUDA 库
# （不需要 GPU，provider 的 DT_NEEDED 在构建期就能验证）。
RUN mkdir -p /models /data /runs /etc/ld.so.conf.d \
 && printf '%s\n' \
      '/opt/venvs/main/lib/python3.12/site-packages/nvidia/cudnn/lib' \
      '/opt/venvs/main/lib/python3.12/site-packages/nvidia/cublas/lib' \
      '/opt/venvs/main/lib/python3.12/site-packages/nvidia/cufft/lib' \
      '/opt/venvs/main/lib/python3.12/site-packages/nvidia/curand/lib' \
      > /etc/ld.so.conf.d/zz-airtime-nvidia.conf \
 && ldconfig \
 && test -e /usr/local/bin/ffmpeg \
 && test -e /usr/local/bin/ffprobe \
 && ffmpeg -version \
 && python -c "import ctypes; \
        [ctypes.CDLL(l) for l in ('libcudart.so.12','libcublas.so.12','libcublasLt.so.12','libcufft.so.11','libcurand.so.10','libcudnn.so.9')]; \
        print('CUDA runtime libs OK')"

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# GPU 冒烟脚本：docker run --rm --gpus all airtime:lean python /app/docker/gpu_smoke.py
# 用它验证 ORT 真的能在 CUDA EP 上完成一次推理（只列 provider 不够）。
COPY docker/gpu_smoke.py /app/docker/gpu_smoke.py

# ---------------------------------------------------------------- 全量档（可选）
# 注意：本轮只验证轻量档；全量档（torch 2.10 cu128 + GEM-X/SAM2）未在本轮
# 真实跑完（耗时过长），保留原分档逻辑与 SHA 不变，依赖上面更精简的公共底座。
# 若全量档构建失败，优先检查这里对 torch 索引 / 阿里云镜像的拉取。
# 代码按 SHA 烘入：GEM-X 与 SAM2 都没有可用 tag。
ARG WITH_GEMX=0
ARG GEMX_SHA=4aef1452e5df171deff4f293c08e643eb450981e
ARG SAM2_SHA=2b90b9f5ceec907a1c18123530e92e794ad901a4

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${WITH_GEMX}" = "1" ]; then set -eux; \
      python3 -m venv /opt/venvs/torch; \
      /opt/venvs/torch/bin/pip install --upgrade pip setuptools wheel; \
      \
      # torch cu128：cu124 最高只到 2.6.0，而 GEM-X 要求 torch>=2.10。
      # 加阿里云做 extra-index，让 cuda-bindings 之类的依赖走镜像
      # （PyPI 原站实测只有几 KB/s，会卡到读超时）。torch 本身仍取 cu128 本地版本号，优先于 PyPI 的 2.10.0。
      /opt/venvs/torch/bin/pip install \
          torch==2.10.0 torchvision==0.25.0 \
          --index-url https://download.pytorch.org/whl/cu128 \
          --extra-index-url https://mirrors.aliyun.com/pypi/simple/ \
          --trusted-host mirrors.aliyun.com; \
      \
      # GEM-X 运行时依赖（install_requires 的子集；不用 --no-deps 会重装 PyPI 的 CPU torch）。
      # 这些包不在 torch 索引里，显式指回阿里云镜像（实测比 files.pythonhosted.org 快一个量级）。
      /opt/venvs/torch/bin/pip install \
          --index-url https://mirrors.aliyun.com/pypi/simple/ \
          --trusted-host mirrors.aliyun.com \
          transformers==5.3.0 \
          py-soma-x==0.3.3 \
          "numpy>=1.26,<3" \
          "scipy>=1.11,<2" \
          "einops>=0.8,<1" \
          "pillow>=10,<13" \
          "opencv-python-headless>=4.10,<5" \
          "omegaconf>=2.3,<2.4"; \
      \
      # SAM2 运行时依赖
      /opt/venvs/torch/bin/pip install \
          --index-url https://mirrors.aliyun.com/pypi/simple/ \
          --trusted-host mirrors.aliyun.com \
          "tqdm>=4.66.1" \
          "hydra-core>=1.3.2" \
          "iopath>=0.1.10"; \
      \
      # 按 SHA 烘入代码：两个仓库都没有可用 tag。
      # 用 init + fetch --depth 1 <sha>，比 clone 全史再 checkout 快得多、也小得多。
      git init -q /opt/GEM-X; \
      git -C /opt/GEM-X remote add origin https://github.com/NVlabs/GEM-X.git; \
      git -C /opt/GEM-X fetch -q --depth 1 origin "${GEMX_SHA}"; \
      git -C /opt/GEM-X checkout -q FETCH_HEAD; \
      # 必须 --no-deps，否则会把 torch 重装成 PyPI 的 CPU 版
      /opt/venvs/torch/bin/pip install --no-deps -e "/opt/GEM-X[video]"; \
      \
      git init -q /opt/sam2; \
      git -C /opt/sam2 remote add origin https://github.com/facebookresearch/sam2.git; \
      git -C /opt/sam2 fetch -q --depth 1 origin "${SAM2_SHA}"; \
      git -C /opt/sam2 checkout -q FETCH_HEAD; \
      SAM2_BUILD_CUDA=0 /opt/venvs/torch/bin/pip install --no-deps -e /opt/sam2; \
      \
      # 只清代码仓库的 .git；pip 缓存挂在 cache mount 上，重建时复用
      rm -rf /opt/GEM-X/.git /opt/sam2/.git; \
    fi

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "airtime.cfg"]
