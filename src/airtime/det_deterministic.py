"""
确定性补丁：让 M1（YOLOX 检测）和 M2（RTMPose 2D）跨进程可复现。

问题实测（2026-11-23）：
  同一 clip、同一模型、同一个进程内连续调用两次 → 完全 bit-identical（10/10）
  但两个独立进程各跑一遍 → boxes 最大差 0.01–0.03 px、kpts 最大差 0.5–13 px、
  scores 最大差 0.003。这就是「同一视频跑两次骨架不一样」的源头。

根因：rtmlib 用 `ort.InferenceSession(model, providers=[...])` 创建会话，
没给 session_options。默认情况下 onnxruntime CUDA EP 把 cuDNN 卷积算法
交给基准搜索（cudnn_conv_algo_search 默认 EXHAUSTIVE），选中的算法依赖
当时 GPU 的显存/时钟状态，浮点归约顺序就变了。

修法：只改这一项 —— `cudnn_conv_algo_search = HEURISTIC`（不跑基准搜索，
同一输入必得同一算法）。

**为什么只改这一项**（最小改动原则，做了消融实验）：
  V0 默认            → 跨进程不一致（boxes 差 0.021、kpts 差 0.549）
  V1 只 HEURISTIC    → **跨进程完全一致** ✓
  V2 只单线程         → 仍不一致（kpts 差 13.59）
  V3 只关 TF32        → 仍不一致（kpts 差 0.516）
  V4 全套            → 一致 ✓

单线程、关 TF32、关 mem_pattern 都不必要，而且会改变数值路径 ——
实测全套改动会让 video5 抖动 p95 从 11.16 涨到 16.15、video7 从 8.93 涨到
10.02、video8 从 6.30 涨到 7.75，属于对已验收结果的回退。
所以这里**只开 cudnn_conv_algo_search=HEURISTIC**，其余保持默认。

用法（在 M1/M2 建模型之前调用）：
    import det_deterministic as DET
    DET.patch_rtmlib()
"""
import functools

# 唯一需要的一项：不跑 cuDNN 基准搜索
CUDA_OPTS = {"cudnn_conv_algo_search": "HEURISTIC"}


def patch_rtmlib(verbose=True):
    """把 onnxruntime.InferenceSession 换成 CUDA EP 确定性版本。

    rtmlib 的 BaseTool 里是 `import onnxruntime as ort` +
    `ort.InferenceSession(...)`，所以替换 sys.modules 里那个模块的属性即可。
    """
    import onnxruntime as ort

    _orig = ort.InferenceSession

    @functools.wraps(_orig)
    def deterministic_session(*args, **kwargs):
        providers = kwargs.get("providers")
        if providers:
            fixed = []
            for p in providers:
                name = p[0] if isinstance(p, tuple) else p
                if name == "CUDAExecutionProvider":
                    opts = dict(CUDA_OPTS)
                    if isinstance(p, tuple) and len(p) > 1 and p[1]:
                        opts.update(p[1])       # 保留调用方显式给的其它选项
                    fixed.append(("CUDAExecutionProvider", opts))
                else:
                    fixed.append(p)
            kwargs["providers"] = fixed
        return _orig(*args, **kwargs)

    ort.InferenceSession = deterministic_session
    if verbose:
        print("  ✓ 已开启 onnxruntime 确定性会话"
              "（cudnn_conv_algo_search=HEURISTIC）")
    return deterministic_session
