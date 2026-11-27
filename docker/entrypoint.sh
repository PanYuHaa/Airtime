#!/usr/bin/env bash
# 容器入口：先把 GEM-X 期望的权重位置接到挂载的模型目录，再执行用户命令。
#
# GEM-X 的 docs/configs/body_video_native77.yaml 里写的是 `../../weights/...`，
# 相对代码目录解析成 /opt/GEM-X/weights/，而权重实际在 ${AIRTIME_MODELS}/GEM-X/ 下。
set -euo pipefail

MODELS_DIR="${AIRTIME_MODELS:-/models}"
GEM_DIR="${AIRTIME_GEMX:-/opt/GEM-X}"

if [ -d "${GEM_DIR}" ]; then
  mkdir -p "${GEM_DIR}/assets"
  # 先删再建，避免 ln 落进已存在的真实目录里
  rm -rf "${GEM_DIR}/weights" "${GEM_DIR}/assets/soma"
  ln -sfn "${MODELS_DIR}/GEM-X/weights" "${GEM_DIR}/weights"
  ln -sfn "${MODELS_DIR}/GEM-X/assets/soma" "${GEM_DIR}/assets/soma"
fi

mkdir -p "${AIRTIME_DATA:-/data}" "${AIRTIME_OUTPUT:-/runs}"

# 上游 sam2 仓库同时带 configs/sam2.1/，不显式覆盖的话
# cfg.resolve_sam2_config() 会误判成 2.1，与默认权重 sam2_hiera_large.pt（2.0）不匹配。
# 只在全量档（/opt/sam2 存在）导出默认值：轻量档没有 SAM2 代码，保留自动匹配，
# 也让镜像内的 pytest 用例可以正常通过。显式传进来的值优先。
if [ -d "${AIRTIME_SAM2:-/opt/sam2}" ]; then
  export AIRTIME_SAM2_CONFIG="${AIRTIME_SAM2_CONFIG:-configs/sam2/sam2_hiera_l.yaml}"
fi

exec "$@"
