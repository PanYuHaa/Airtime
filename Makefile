PYTHON ?= python3

# ---- Docker ---------------------------------------------------------------
# 镜像 tag 由 IMAGE_TAG 控制（默认 dev）；WITH_GEMX=1 构建全量档（含 GEM-X/SAM2）。
COMPOSE ?= docker compose
IMAGE_TAG ?= dev
WITH_GEMX ?= 0
# 容器内主 venv 的解释器（T1 约定：/opt/venvs/main）。用绝对路径调入口，
# 既不依赖 console script 是否在 PATH 上，也不依赖镜像 ENTRYPOINT 写法。
MAIN_PY ?= /opt/venvs/main/bin/python
MAIN_BIN ?= /opt/venvs/main/bin
VIDEO ?=

.PHONY: help install install-dev check lint format test validate-configs env-check clean \
        docker-build docker-build-full docker-run docker-check docker-test docker-shell docker-clean

help:
	@echo "Airtime Engineering Targets:"
	@echo "  make install           Install package in editable mode"
	@echo "  make install-dev       Install package with dev tools (pytest, ruff)"
	@echo "  make validate-configs  Validate configs/*.json schema integrity"
	@echo "  make lint              Run ruff static checks"
	@echo "  make format            Check/format code with ruff"
	@echo "  make test              Run pytest unit test suite"
	@echo "  make env-check         Run airtime-check environment diagnostics"
	@echo "  make check             Run validate-configs + lint + test"
	@echo "  make clean             Remove Python build/cache artifacts"
	@echo ""
	@echo "Docker Targets (see docs/DOCKER.md):"
	@echo "  make docker-build      Build lightweight image airtime:$(IMAGE_TAG) (WITH_GEMX=0)"
	@echo "  make docker-build-full Build full image airtime:full (WITH_GEMX=1, GEM-X/SAM2)"
	@echo "  make docker-run        Run pipeline; VIDEO=<path> e.g. VIDEO=data/input.mp4"
	@echo "  make docker-check      Run airtime-check --strict inside the container"
	@echo "  make docker-test       Run pytest inside the container"
	@echo "  make docker-shell      Open a bash shell in the container"
	@echo "  make docker-clean      Remove airtime:$(IMAGE_TAG) and airtime:full images"

install:
	$(PYTHON) -m pip install -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

validate-configs:
	$(PYTHON) scripts/validate_configs.py

lint:
	$(PYTHON) -m ruff check src tests scripts

format:
	$(PYTHON) -m ruff format src tests scripts

test:
	PYTHONPATH=src $(PYTHON) -m pytest -v

env-check:
	PYTHONPATH=src $(PYTHON) -m airtime.cfg

check: validate-configs lint test

clean:
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info
	find src tests scripts -name "__pycache__" -type d -exec rm -rf {} +

# ---------------------------------------------------------------------------
# Docker
#
# 所有目标都通过 compose 走，复用 docker-compose.yml 的 GPU / 卷 / 环境变量配置；
# 因此 IMAGE_TAG / WITH_GEMX 同时作用于构建与运行（同一组变量）。
# 每个 run 都显式 --entrypoint，不依赖 T2 镜像里 ENTRYPOINT 的具体写法。
# 注意：Makefile 里 shell 的 `$` 必须写成 `$$`。
# VIDEO 是仓库相对路径，必须位于 data/ 下（宿主 ./data 挂到容器 /data），
# 例如 VIDEO=data/input.mp4 → 容器内 /data/input.mp4。
# ---------------------------------------------------------------------------

# 构建轻量档：airtime:$(IMAGE_TAG)（默认 dev）
docker-build:
	WITH_GEMX=0 IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) build

# 构建全量档：airtime:full（含 GEM-X/SAM2 的 torch 环境）
docker-build-full:
	WITH_GEMX=1 IMAGE_TAG=full $(COMPOSE) build

# 跑整条流程：make docker-run VIDEO=data/input.mp4
docker-run:
	@test -n "$(VIDEO)" || { echo "用法: make docker-run VIDEO=data/input.mp4"; exit 1; }
	@case "$(patsubst ./%,%,$(VIDEO))" in \
	  data/*) ;; \
	  *) echo "VIDEO 必须是 data/ 下的相对路径（宿主 ./data 挂到容器 /data），例如 VIDEO=data/input.mp4"; exit 1;; \
	esac
	IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) run --rm --entrypoint $(MAIN_BIN)/airtime airtime \
	  "$(patsubst data/%,/data/%,$(patsubst ./%,%,$(VIDEO)))"

# 容器内环境自检（strict 会检查权重路径）
docker-check:
	IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) run --rm --entrypoint $(MAIN_BIN)/airtime-check airtime --strict

# 容器内跑单元测试（需镜像内装有 pytest，见 docs/DOCKER.md 的构建说明）
docker-test:
	IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) run --rm --entrypoint $(MAIN_PY) airtime -m pytest -v

# 进容器交互式 shell
docker-shell:
	IMAGE_TAG=$(IMAGE_TAG) $(COMPOSE) run --rm --entrypoint bash airtime

# 删除镜像（两档都删；不存在时 docker rmi 报错可忽略）
docker-clean:
	-docker rmi airtime:$(IMAGE_TAG)
	-docker rmi airtime:full
