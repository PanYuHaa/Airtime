# Docker 运行指南

用容器把 Airtime 的运行依赖（CUDA 运行库、cuDNN、ffmpeg、onnxruntime-gpu；全量档另含 torch）
一次性装好，宿主机只需要 Docker + NVIDIA 驱动。权重不进镜像，通过卷挂载只读提供。

> 相关文件：`docker-compose.yml`（本目录上一级）、`Dockerfile` / `.dockerignore` / `docker/`（构建）、
> `Makefile` 里的 `docker-*` 目标。

---

## 1. 前置条件

1. **Docker**
   - Linux：安装 Docker Engine（Compose v2 已内置为 `docker compose`）。
   - Windows：安装 Docker Desktop，并在 **Settings → Resources → WSL Integration** 中为所用发行版打开集成。
2. **NVIDIA 驱动 + GPU 直通**
   - Linux：安装 NVIDIA 驱动和 NVIDIA Container Toolkit。
   - WSL2（Docker Desktop）：确认已启用 GPU 支持（较新的 Docker Desktop 自带，无需单独装 toolkit）。
3. **验证 GPU 在容器里可见**（这一步不通过，后面全白搭）：

   ```bash
   docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi
   ```

   能看到 `nvidia-smi` 表格（GPU 型号、驱动版本、CUDA Version）即代表宿主 → 容器的 GPU 链路正常。
   这一步只验证驱动链路，不代表 Airtime 能用到 CUDA；镜像建好后用第 7 节的 GPU 冒烟脚本做端到端确认。

4. **确认 Compose 版本**：

   ```bash
   docker compose version   # 需要 v2.x
   ```

---

## 2. 基础镜像与 ffmpeg 从哪来

**实际基础镜像是 `nvidia/cuda:12.8.1-base-ubuntu24.04`（最小 base 档，只带 cudart）。**

不选 `cudnn-runtime` 档的原因：该档实测 7.33 GB，其中 cusparse / NCCL / cusolver / NPP / NVRTC
约 1.8 GB 是 ORT 和 torch wheel 都不需要的；而**删掉 base 层里的文件只会多出一个 whiteout 层，
镜像并不会变小**。所以改用最小 base 档，再按需用 pip 装 CUDA 库：

- `nvidia-cublas-cu12` / `nvidia-cufft-cu12` / `nvidia-curand-cu12` / `nvidia-cudnn-cu12`
  （cuDNN 用 `pip install --no-deps nvidia-cudnn-cu12==9.2.0.82`，解压后 838 MB）；
- 动态链接靠 `/etc/ld.so.conf.d/zz-airtime-nvidia.conf` + `ldconfig` 和 `LD_LIBRARY_PATH` 兜底，
  即使 compose 覆盖了 `LD_LIBRARY_PATH`，`ldconfig -p` 也能找到这些库。
- 有意删除了 `libcudnn_adv.so.9`（约 230 MB，RNN/LSTM/attention 用不到，2D 流水线只有
  YOLOX / RTMPose 这类卷积网络）；若将来引入这类算子，需要改 Dockerfile 恢复。

**ffmpeg 不走 apt，也不用发行版自带版本。**

- M0 预处理在 `src/airtime/pipeline/m0_prep.py:212` 调用了 `ffmpeg -fps_mode cfr` 做恒定帧率转码，
  `-fps_mode` 是 **ffmpeg 5.1 才引入**的参数。
- apt 版 ffmpeg 会拖进整套 Mesa / LLVM / SDL 图形栈，实测 619 MB；镜像改为从
  `mwader/static-ffmpeg:6.1.1` 多阶段复制 **静态 ffmpeg + ffprobe 两个二进制**（共 195 MB）。
- 版本仍是 **6.1.1**，`-fps_mode cfr` 可用（镜像内已实测）。

所以发行版仓库自带的 ffmpeg 版本与构建无关；但如果你自建镜像并改用 apt 版 ffmpeg，
必须保证 ≥ 5.1，否则 M0 跑不通。

---

## 3. 两档构建

镜像名固定为 `airtime`，tag 由 `IMAGE_TAG` 控制，默认 `dev`。全量档用
`--build-arg WITH_GEMX=1`，tag 约定为 `full`。

| 档位 | tag | 构建参数 | 体积 | 能跑到哪些阶段 | 构建命令 |
|---|---|---|---|---|---|
| 精简 | `dev` | `WITH_GEMX=0`（默认） | 4.83 GB | M0 / M1（检测，无 SAM2 补帧）/ M2（2D 关键点） | `make docker-build` |
| 全量 | `full` | `WITH_GEMX=1` | 未验证 | 全部，含 M1 的 SAM2 补帧与 M3 的 GEM-X 三维重建 | `make docker-build-full` |

> 精简档 4.83 GB 是实测值：`docker image inspect .Size` = 4,828,369,719 B。
> 全量档本轮**未构建、未验证**，体积未知；它只通过了 `docker buildx build --check` 语法检查，
> 首次构建耗时很长，构建失败时优先检查 torch 索引与阿里云镜像的拉取。

各档实际装了什么：

- **精简档**：只有主 venv `/opt/venvs/main`；**没有** `/opt/venvs/torch`、**没有** `/opt/GEM-X`、
  **没有** `/opt/sam2`。
- **全量档**：额外装 torch 2.10.0+cu128 与 GEM-X / SAM2 代码（按 SHA 烘入，无可用 tag）。

对应的 compose 变量：

```bash
# 精简
IMAGE_TAG=dev  WITH_GEMX=0 docker compose build

# 全量
IMAGE_TAG=full WITH_GEMX=1 docker compose build
```

精简档里 SAM2 代码/venv 不存在，M1 会**自动跳过** SAM2 补帧（`m1_track.sam2_fill` 在
权重或代码缺失时返回空）；但 **M3 GEM-X 无法跳过**——它需要 torch 子进程，精简档一跑到
`gemx` 阶段就会失败。所以：

- 只要 2D 链路（M0 / M1 检测 / M2 2D 关键点）→ 精简档够用，`--stage pose2d` 即可。
- **M3F 融合依赖 3D 结果**：`m3f_fuse.py` 只从 `gem/` 目录读 GEM-X 产出的投影块，`gem/` 为空时
  直接打印「x 无 gem 目录」并退出，不会产出骨架；M8 叠加又只画 `skeleton/*.json`。因此精简档
  **跑不出 M3F / M8 的有意义结果**。
- 要最终 3D 骨架（M3/M3F/M8）→ 必须全量档。

---

## 4. 权重放置

权重统一放在宿主机的 `models/`，容器内挂载为 `/models`（**只读**）。
按下面的目录树放好：

```
models/
├── yolox_x_8xb8-300e_humanart-a39d44ed.onnx
├── rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288.onnx
├── sam2_hiera_large.pt
└── GEM-X/
    ├── weights/          # body_s240000.ckpt, vitpose.pth, yolox_humanart.onnx,
    │                     # motion_native77_e774_s292175.ckpt, motion_stats_raw.npz
    └── assets/soma/      # SOMA_neutral.npz, SOMA_template_rig.usda,
                          # SOMA_procedural_transforms.json, correctives_model.pt
```

### 下载指引

**(1) YOLOX 人体检测（ONNX）** —— 来自 OpenMMLab 官方 ONNX 归档（zip 内 `end2end.onnx` 改名）：

```bash
curl --fail --location -o /tmp/yolox.zip \
  https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip
# zip 内路径 .../yolox_x_8xb8-300e_humanart-a39d44ed/end2end.onnx
```

**(2) RTMPose-x Halpe26（ONNX）** —— 同一个 OpenMMLab ONNX 归档：

```bash
curl --fail --location -o /tmp/rtmpose.zip \
  https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip
# zip 内路径 .../rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606/end2end.onnx
```

两个 zip 都把里面的 `end2end.onnx` 取出，分别重命名为目录树里的两个 `.onnx` 文件名。

**(3) SAM2 权重**

```bash
curl --fail --location -o models/sam2_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

> 文件名保持 `sam2_hiera_large.pt` 即可；`cfg.resolve_sam2_config()` 会按文件名/目录结构自动匹配
> Hydra 配置（`configs/sam2.1/sam2.1_hiera_l.yaml`）。如需强制指定，用 `AIRTIME_SAM2_CONFIG`。
> **SAM2 代码只在全量档烘入镜像**（`/opt/sam2`），精简档没有，SAM2 补帧会自动跳过。

**(4) GEM-X 权重**（HuggingFace，`nvidia/GEM-X`，gated=false，无需申请权限）：

```bash
python -m pip install -U huggingface_hub   # 提供 hf 命令
hf download nvidia/GEM-X \
  body_s240000.ckpt body_config.json \
  motion_native77_e774_s292175.ckpt motion_stats_raw.npz \
  motion_native77.yaml body_video_native77.yaml vitpose.pth \
  --local-dir models/GEM-X/weights
```

另外把 YOLOX HumanArt 检测器也放进 `weights/`（GEM-X 的配置按相对路径找 `yolox_humanart.onnx`）：
可直接复用 (1) 里解压出的 `end2end.onnx`，复制为 `models/GEM-X/weights/yolox_humanart.onnx`。

**(5) SOMA v0.3.3 几何资产**（HuggingFace，`nvidia/SOMA-X`，rev `v0.3.3`，gated=false）：

```bash
hf download nvidia/SOMA-X \
  SOMA_neutral.npz SOMA_template_rig.usda SOMA_procedural_transforms.json \
  --revision v0.3.3 --local-dir models/GEM-X/assets/soma

# GEM-X 渲染/校正还需要（同为 v0.3.3）：
hf download nvidia/SOMA-X correctives_model.pt \
  --revision v0.3.3 --local-dir models/GEM-X/assets/soma
```

> **代码与权重的分工（T1 约定）**：GEM-X 代码烘在镜像 `/opt/GEM-X`，SAM2 代码烘在 `/opt/sam2`
> （都只在全量档）；权重和数据都在 `/models`。GEM-X 的视频配置 `docs/configs/body_video_native77.yaml`
> 以「相对配置自身」解析 `../../weights/` 与 `assets/soma/`，因此需要把
> `/opt/GEM-X/weights` → `/models/GEM-X/weights`、`/opt/GEM-X/assets/soma` → `/models/GEM-X/assets/soma`
> 建好链接（由 `docker/entrypoint.sh` 在容器启动时负责，不是构建期）。链接要在容器启动后才有，
> 用下面命令核对（**不要**加 `--entrypoint bash`，那会绕过 entrypoint.sh，链接不会建立）：

```bash
IMAGE_TAG=full docker compose run --rm airtime \
  ls -l /opt/GEM-X/weights /opt/GEM-X/assets/soma
```

---

## 5. 三种用法

### 5.1 Compose（推荐）

```bash
docker compose build                              # 精简档，image: airtime:dev
docker compose run --rm --entrypoint /opt/venvs/main/bin/airtime airtime /data/input.mp4
docker compose run --rm --entrypoint /opt/venvs/main/bin/airtime airtime /data/input.mp4 --out /runs/demo

# 全量档
IMAGE_TAG=full WITH_GEMX=1 docker compose build
IMAGE_TAG=full docker compose run --rm --entrypoint /opt/venvs/main/bin/airtime airtime /data/input.mp4
```

> 用 `--entrypoint` 显式指定入口，不依赖镜像 ENTRYPOINT/CMD 的具体写法。
> 入口用绝对路径 `/opt/venvs/main/bin/airtime`（主 venv，T1 约定），不依赖 PATH。
>
> 注意：`--entrypoint` 会**绕过**镜像的 `/usr/local/bin/entrypoint.sh`，而 GEM-X 需要的
> `/opt/GEM-X/weights` → `/models/GEM-X/weights` 等链接正是由该脚本在启动时建立的。
> 精简档不受影响（没有 GEM-X）；**全量档若走 `--entrypoint`，需要先手动执行一次
> `/usr/local/bin/entrypoint.sh <命令>`，或进入容器后自行建链接**，否则 GEM-X 找不到权重。

### 5.2 原生 docker run（不改 compose 也能跑）

```bash
docker build -t airtime:dev .

docker run --rm --gpus all --ipc=host --shm-size=2g \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/data:/data" \
  -v "$PWD/runs:/runs" \
  -e AIRTIME_MODELS=/models -e AIRTIME_DATA=/data -e AIRTIME_OUTPUT=/runs \
  -w /app --entrypoint /opt/venvs/main/bin/airtime \
  airtime:dev /data/input.mp4
```

交互式排查：

```bash
docker run --rm -it --gpus all --ipc=host --shm-size=2g \
  -v "$PWD/models:/models:ro" -v "$PWD/data:/data" -v "$PWD/runs:/runs" \
  -e AIRTIME_MODELS=/models -e AIRTIME_DATA=/data -e AIRTIME_OUTPUT=/runs \
  -w /app --entrypoint bash airtime:dev
```

### 5.3 Makefile

```bash
make docker-build
make docker-run VIDEO=data/input.mp4          # VIDEO 必须是 data/ 下的相对路径
make docker-check                             # 容器内 airtime-check --strict
make docker-test                              # 容器内 pytest
make docker-shell                             # 进容器 bash
make docker-build-full                        # 全量档
make docker-clean                             # 删除 airtime:dev 和 airtime:full
```

可覆盖的变量：`IMAGE_TAG`（默认 dev）、`WITH_GEMX`（默认 0）、`COMPOSE`（默认 `docker compose`）。

> `make docker-run` 要求 `VIDEO` 位于 `data/` 下，因为宿主 `./data` 挂在容器 `/data`；
> 它会自动把 `data/` 前缀换成 `/data/`（例：`data/input.mp4` → `/data/input.mp4`）。

---

## 6. 环境变量表

| 变量 | 容器内约定值 | 作用 |
|---|---|---|
| `AIRTIME_MODELS` | `/models` | 权重根目录（只读卷） |
| `AIRTIME_DATA` | `/data` | 输入数据目录 |
| `AIRTIME_OUTPUT` | `/runs` | 输出目录（skeleton、overlay 等） |
| `AIRTIME_GEMX` | `/opt/GEM-X` | GEM-X 代码目录（全量档烘入） |
| `AIRTIME_SAM2` | `/opt/sam2` | SAM2 代码目录（全量档烘入） |
| `AIRTIME_GEMX_PYTHON` | `/opt/venvs/torch/bin/python` | 跑 GEM-X 子进程的 torch 解释器（仅全量档存在） |
| `AIRTIME_SAM2_PYTHON` | `/opt/venvs/torch/bin/python` | 跑 SAM2 子进程的 torch 解释器（缺省复用 GEMX_PYTHON，仅全量档存在） |
| `AIRTIME_SAM2_CONFIG` | 留空（自动推断） | 强制指定 SAM2 Hydra 配置路径，如 `configs/sam2.1/sam2.1_hiera_l.yaml` |

`docker-compose.yml` 已显式设置除 `AIRTIME_SAM2_CONFIG` 外的全部变量（指向镜像内的代码与 venv）；
`AIRTIME_SAM2_CONFIG` 默认留空走自动推断。全量档的 `entrypoint.sh` 在检测到 `/opt/sam2` 时会补上
默认值 `configs/sam2/sam2_hiera_l.yaml`；需要覆盖时用 compose 的 `-e` 或在 `environment:` 段追加。

---

## 7. GPU 冒烟

`nvidia-smi` 能跑、`ort.get_available_providers()` 里有 `CUDAExecutionProvider`，都不等于
ONNX 模型真的在 GPU 上算过。镜像里带了 `docker/gpu_smoke.py`：它逐库 `dlopen` 一遍 CUDA/cuDNN
运行时库，用 `CUDAExecutionProvider` 建一个会话，再用随机张量做一次真实前向。

```bash
docker run --rm --gpus all \
  -v "$PWD/models:/models:ro" \
  airtime:dev python /app/docker/gpu_smoke.py
```

- 默认用 `cfg.RTMPOSE_ONNX`，即 `/models/rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288.onnx`；
  也可以把模型路径作为第一个参数传入。
- 找不到模型时以退出码 2 结束并提示，此时可先放好权重再跑，或直接传模型路径。
- 正常输出以 `session: [...]`（其中包含 `CUDAExecutionProvider`）与 `GPU smoke OK` 结尾；
  若会话实际落在 CPU、或前向直接抛库缺失错误，按第 9 节排查。

---

## 8. 已知限制

- **WSL2 内存 ~7 GB，可能 OOM**：全量档加载 GEM-X + torch 时内存吃紧。
  可通过 Docker Desktop 的 `.wslconfig` 调大内存，或先用精简档跑 M0–M2 验证链路。
- **镜像体积**：精简档实测 4.83 GB；全量档未构建、未验证，体积未知。构建和拉取都耗时，全量档首次构建尤其慢。
- **GPU 推理的端到端结果以冒烟脚本为准**：精简档里 ORT 能列出 `CUDAExecutionProvider`，构建期也已
  逐库 `dlopen` 过 CUDA/cuDNN；但真实推理是否落在 GPU 上，仍以第 7 节 `gpu_smoke.py` 的实际输出为准
  （需先挂好权重）。若报 cuDNN 相关错误见第 9 节排查。
- **权重不进镜像**：镜像里没有模型权重，必须挂载 `models/`，否则只有 `airtime-check`（非 strict）能过。
- **`docker-check --strict` 只在全量档能全绿**：`check_setup(strict=True)` 会检查
  `/opt/GEM-X` 与 `/opt/sam2` 是否存在（compose 把 `AIRTIME_GEMX` / `AIRTIME_SAM2` 指到这两处），
  精简档没烘入这两块代码，strict 必然报缺——这是预期行为，不代表权重放错。
  精简档要验证权重，只能逐项确认 `airtime-check --strict` 的输出里**只剩** `/opt/GEM-X`、
  `/opt/sam2` 两项缺失（YOLOX / RTMPose / SOMA 关节名表都应通过）；不带 `--strict` 的
  `airtime-check` 只查 configs，不会检查权重。
- **GPU 只在运行时可用**：用 `docker compose run` 或 `docker run --gpus all`；直接 `docker build` 不需要 GPU。

---

## 9. 故障排查

### GPU 在容器里不可见（`nvidia-smi` 报错 / 找不到 GPU）

1. 先验证宿主侧：`nvidia-smi` 正常。
2. 再验证容器侧：`docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi`（与实际基础镜像一致）。
3. Linux：确认装了 NVIDIA Container Toolkit，且 Docker daemon 能访问 GPU。
4. WSL2：确认 Docker Desktop 的 WSL Integration 已为该发行版打开，并重启 Docker Desktop。
5. 运行时命令是否带了 `--gpus all`（compose 里对应 `deploy.resources.reservations.devices`，本文件已配好）。

### 容器内报找不到 cuDNN / cuDNN 版本不匹配

- 症状：onnxruntime 或 torch 首次推理时报 `libcudnn*.so` 缺失或版本不兼容。
- 原因：cuDNN 是 pip 装进主 venv 的（`nvidia-cudnn-cu12==9.2.0.82`），不是来自基础镜像；
  若该目录不在动态链接器的搜索路径里，运行期就会找不到。
- 处理：按下面三步排查，而不是换基础镜像（基础镜像故意选最小 base 档）：

  ```bash
  # 1) 确认 pip 装的 cuDNN 在位
  docker compose run --rm --entrypoint bash airtime -lc \
    'ls /opt/venvs/main/lib/python3.12/site-packages/nvidia/cudnn/lib/ | head'

  # 2) 确认 ld.so 配置生效（应当列出上述目录）
  docker compose run --rm --entrypoint bash airtime -lc \
    'cat /etc/ld.so.conf.d/zz-airtime-nvidia.conf; ldconfig -p | grep cudnn'

  # 3) 跑一次真实推理（见第 7 节）
  docker compose run --rm --entrypoint /opt/venvs/main/bin/python airtime \
    /app/docker/gpu_smoke.py
  ```

  若 `ldconfig -p` 里查不到 `libcudnn.so.9`，说明 `zz-airtime-nvidia.conf` 没生效，需要重新
  `ldconfig`；若 `site-packages/nvidia/cudnn/lib/` 为空，说明 pip 安装层没有生效。
  仍不行时记录 onnxruntime 版本与 `nvidia-smi` 的 CUDA Version 反馈给维护者。

### 需要 RNN / LSTM / attention 类 ONNX 算子

- 原因：镜像有意删除了 `libcudnn_adv.so.9`（约 230 MB），2D 流水线（YOLOX / RTMPose）用不到它。
- 症状：这类算子报 `libcudnn_adv` 缺失。
- 处理：改 Dockerfile，去掉删除 `libcudnn_adv.so.9` 的那一行（`rm -f .../libcudnn_adv.so.9*`），
  重建镜像。

### SAM2 报配置不匹配（`Could not find config` / Hydra 解析失败）

- 症状：`build_sam2_video_predictor` 找不到 config。
- 原因：权重文件名与自动推断的配置不一致（例如 2.0 vs 2.1）。
- 处理：显式指定配置，例如加 `-e AIRTIME_SAM2_CONFIG=configs/sam2.1/sam2.1_hiera_l.yaml`。
  2.1 权重用 `configs/sam2.1/` 前缀，2.0 用 `configs/sam2/` 前缀。

### 权重路径错误（`airtime-check --strict` 报 XXX 不存在）

- 先确认宿主机目录树与第 4 节完全一致，再跑：

  ```bash
  make docker-check
  # 或
  docker compose run --rm --entrypoint /opt/venvs/main/bin/airtime-check airtime --strict
  ```

- 若报 GEM-X 相关文件缺失，检查 `/opt/GEM-X/weights` 是否已正确链接到 `/models/GEM-X/weights`
  （见第 4 节末尾的核对命令）。注意 strict 模式还要求 `/opt/GEM-X` 与 `/opt/sam2` 存在，
  精简档必然在这两项上报缺，属预期行为（详见第 8 节）。
- 若你用别的宿主机路径，改 `docker-compose.yml` 的 `volumes` 挂载即可，容器内路径无需变。

### DataLoader 报 Bus error / worker 被杀

- 原因：`/dev/shm` 太小。
- 处理：确认 `ipc: host` 与 `shm_size: 2gb` 生效（本文件已配）；内存紧张时可把 `shm_size` 降到 `1gb`，
  但不要去掉 `ipc: host`。

### `make docker-test` 报 pytest 不存在

- 精简档已实测装有 `pytest 9.2.0`（Dockerfile 在主 venv 里显式安装），正常不该报缺。
- 若报了，先确认 `docker compose run --rm --entrypoint /opt/venvs/main/bin/python airtime -m pytest --version`；
  仍缺时在 Dockerfile 中补 `pip install pytest` 后重建。
