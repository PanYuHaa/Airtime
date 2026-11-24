#!/usr/bin/env python3
"""
P1 T1.1 预处理 v2（修正旋转与尺寸）

关键修正：
  1. **不再手动 transpose**：ffmpeg 默认按 rotation 元数据自动转正，显式 transpose 会叠加两次
  2. **短边 = 1080**：竖屏 → 1080x1920，横屏 → 1920x1080（用表达式自适应）
  3. **镜头切分保守化**：主镜头覆盖 < 60% 时保留整段（避免误切掉落地段）
  4. 转码后用 cv2 实测尺寸并校验，与 meta 一致
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ..cfg import CONFIGS_DIR, DATA_DIR, OUTPUT_DIR

PROJ = CONFIGS_DIR.parent
VIDEOS = DATA_DIR / "videos"
OUT = OUTPUT_DIR / "pkg"
CFG = None  # 惰性加载，见 _videos_cfg()

TARGET_FPS = 30
SHORT_SIDE = 1080


def probe(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:stream_tags=rotate:stream_side_data=rotation",
         "-show_entries",
         "stream=color_transfer,color_primaries,pix_fmt",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True)
    d = json.loads(r.stdout)
    s = d["streams"][0]
    w, h = int(s["width"]), int(s["height"])
    rot = 0
    for sd in s.get("side_data_list", []):
        if "rotation" in sd:
            rot = int(float(sd["rotation"])) % 360
    if rot == 0:
        rot = int(s.get("tags", {}).get("rotate", 0)) % 360
    dw, dh = (h, w) if rot in (90, 270) else (w, h)
    num, den = s["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    dur = float(d.get("format", {}).get("duration", 0))

    # HDR 判定：HLG(arib-std-b67) / PQ(smpte2084) 或 10bit
    tr = (s.get("color_transfer") or "").strip()
    pr = (s.get("color_primaries") or "").strip()
    pf = (s.get("pix_fmt") or "").strip()
    hdr = tr in ("arib-std-b67", "smpte2084") or pf.endswith("10le")
    return {"coded_w": w, "coded_h": h, "rotation": rot,
            "disp_w": dw, "disp_h": dh, "fps": fps, "duration": dur,
            "color_transfer": tr, "color_primaries": pr, "pix_fmt": pf,
            "hdr": bool(hdr), "hdr_transfer": tr if hdr else None}


def detect_cuts(video, mult=4.0, abs_min=0.35):
    """保守镜头切分：只认非常强的突变"""
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 20:
        cap.release()
        return [], n

    step = max(1, n // 300)
    hists, idxs = [], []
    for i in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, im = cap.read()
        if not ok:
            break
        small = cv2.resize(im, (160, 90))
        h = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        hists.append(cv2.normalize(h, h).flatten())
        idxs.append(i)
    cap.release()

    if len(hists) < 3:
        return [], n

    diffs = [(idxs[i], float(np.linalg.norm(hists[i] - hists[i-1])))
             for i in range(1, len(hists))]
    ds = [d for _, d in diffs]
    med = float(np.median(ds))
    thr = max(med * mult, abs_min)
    cuts = [i for i, d in diffs if d > thr]
    return cuts, n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(VIDEOS), help="视频文件夹")
    ap.add_argument("--list", default="", help="清单 videos.json（默认用项目自带的）")
    ap.add_argument("--out", default=str(OUT), help="输出文件夹")
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    videos_dir = Path(a.videos)
    out = Path(a.out)
    if a.list:
        list_doc = json.load(open(a.list, encoding="utf-8"))
        cfg = list_doc if isinstance(list_doc, dict) else {"videos": list_doc}
    else:
        cfg = CFG

    only = [x for x in a.only.split(",") if x]
    (out / "clips").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    index = {}

    for v in cfg["videos"]:
        vid = v["id"]
        if only and vid not in only:
            continue

        src = videos_dir / v["file"]
        if not src.exists():
            print(f"  x {vid}: 源不存在 {src}", flush=True)
            continue

        dst = out / "clips" / f"{vid}.mp4"
        meta_p = out / "meta" / f"{vid}.json"
        _t_vid = time.time()

        print(f"\n{'='*74}", flush=True)
        print(f"  {vid}: {v['file']} [{v['camera']}]", flush=True)
        print(f"{'='*74}", flush=True)

        info = probe(src)
        print(f"  编码 {info['coded_w']}x{info['coded_h']}  旋转 {info['rotation']}°  "
              f"显示 {info['disp_w']}x{info['disp_h']}  {info['fps']:.1f}fps", flush=True)

        # 短边 = 1080 的表达（竖屏 1080x1920，横屏 1920x1080）
        #
        # HDR 处理（关键）：iPhone 的 HLG HDR（bt2020 + arib-std-b67 + 10bit）
        # 如果直接 `format=yuv420p` 转 8bit，等于把 HDR 数据当 SDR 用 ——
        # 结果是一大片灰白的雪、人物轮廓糊掉，看不清。
        # 【2026-11-23 定稿】给模型的输入**保持现状，不做色调映射**。
        # Mac 侧实测 3 条 HLG（IMG_2617 / IMG_2606 / IMG_2451，各约 40 帧，
        # 跑 YOLOX + RTMPose-X）：
        #   A 不做映射(现状)  关键点置信度 0.794 / 0.665 / 0.769
        #   C 只换原色+矩阵    0.763 / 0.672 / 0.760
        #   H mobius npl1000  0.761 / 0.661 / 0.771
        #   M mobius npl100   0.730 / 0.645 / 0.745
        #   R reinhard        0.759 / 0.665 / 0.770
        #   检出率五种方案完全相同；置信度差 ≤0.03，不做映射往往最高。
        # 原因：iPhone 的 HLG 设计上「当 SDR 显示也基本正常」，
        #      模型看到的画面并没有坏到影响识别。
        # 已验收的 12 条（其中 5 条也是 HLG）就是按不做映射跑通的，不受影响。
        #
        # 「人看着发灰/过曝」是**播放器显示**的问题（重编码丢了色彩标签），
        # 已在渲染层解决：见 p1_t16_overlay_hdr.py（产出视频跟随原片格式）。
        #
        # 但 PQ(smpte2084) 不同 —— 它不向下兼容，当普通视频看是错的，
        # 必须先色调映射再进模型。现有素材里没有 PQ，规则保留。
        hdr_chain = ""
        if info.get("hdr_transfer") == "smpte2084":
            hdr_chain = ("zscale=tin=smpte2084:t=linear:npl=1000,"
                         "format=gbrpf32le,zscale=p=bt709,"
                         "tonemap=tonemap=mobius:desat=0,"
                         "zscale=t=bt709:m=bt709:r=tv,")
        vf = (hdr_chain +
              f"scale='if(gt(iw,ih),-2,{SHORT_SIDE})':'if(gt(iw,ih),{SHORT_SIDE},-2)'"
              f":flags=lanczos,fps={TARGET_FPS},format=yuv420p")

        if info.get("hdr"):
            branch = ("PQ → 色调映射到 SDR（PQ 不向下兼容，必须映射）"
                      if info.get("hdr_transfer") == "smpte2084"
                      else "HLG → 保持现状（实测不映射识别最好）")
            print(f"  HDR: {info['color_transfer']} / "
                  f"{info['color_primaries']} / {info['pix_fmt']}"
                  f" → {branch}", flush=True)

        if not dst.exists():
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                   "-vf", vf, "-fps_mode", "cfr",
                   "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                   # 显式写 bt709 标签，避免播放器各自猜色彩
                   "-color_primaries", "bt709", "-color_trc", "bt709",
                   "-colorspace", "bt709", "-color_range", "tv",
                   "-an", str(dst)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  x 转码失败: {r.stderr[-300:]}", flush=True)
                continue
            print(f"  ✓ 转码", flush=True)
        else:
            print(f"  -- 已存在", flush=True)

        # 实测输出尺寸
        cap = cv2.VideoCapture(str(dst))
        ow = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        oh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        on = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 校验方向
        expect_portrait = info["disp_h"] > info["disp_w"]
        got_portrait = oh > ow
        orient_ok = expect_portrait == got_portrait
        print(f"  输出 {ow}x{oh}, {on} 帧  "
              f"{'竖屏' if got_portrait else '横屏'} {'✓' if orient_ok else '✗ 方向错误'}",
              flush=True)

        # 镜头切分（保守）
        cuts, _ = detect_cuts(dst)
        if cuts:
            bounds = [0] + cuts + [on]
            segs = [(bounds[i], bounds[i+1]-1) for i in range(len(bounds)-1)]
            segs = [(s, e) for s, e in segs if e - s + 1 >= 15]
            if segs:
                main = max(segs, key=lambda x: x[1]-x[0])
                cover = (main[1]-main[0]+1) / on
                if cover < 0.6:
                    print(f"  切分: {len(segs)} 段, 主镜头仅 {cover*100:.0f}% → 保留整段", flush=True)
                    main = (0, on-1)
                else:
                    print(f"  切分: {len(segs)} 段, 主镜头 {main[0]}-{main[1]} "
                          f"({cover*100:.0f}%)", flush=True)
            else:
                main = (0, on-1)
        else:
            main = (0, on-1)
            print(f"  切分: 单镜头", flush=True)

        meta = {
            "video": vid, "file": v["file"],
            "camera": v["camera"], "static_cam": v["camera"] == "tripod",
            "sport": v.get("sport"),        # 产品 ① 不需要，缺省 None
            "rotation_deg": info["rotation"],
            "coded_size": [info["coded_w"], info["coded_h"]],
            "display_size": [info["disp_w"], info["disp_h"]],
            "output_size": [ow, oh],
            "orientation": "portrait" if got_portrait else "landscape",
            "fps": TARGET_FPS, "frames": on,
            "duration": round(on / TARGET_FPS, 3),
            "main_shot": [main[0], main[1]],
            "main_shot_frames": main[1] - main[0] + 1,
            "shot_cuts": cuts,
            "source_fps": round(info["fps"], 2),
            "source_hdr": bool(info.get("hdr")),
            "source_color": {
                "transfer": info.get("color_transfer"),
                "primaries": info.get("color_primaries"),
                "pix_fmt": info.get("pix_fmt"),
            },
            # 这个字段记的是「进模型时做了什么」，不是渲染输出。
            # HLG 进模型是不做映射（实测识别最好），所以记 passthrough；
            # 以前一律写 mobius 是错的 —— 与 configs 里 model_input.hlg=passthrough
            # 相互矛盾，会让看包的人以为模型输入被映射过。
            "hdr_tonemap": (
                "passthrough (HLG 原样进模型；实测不映射识别最好)"
                if info.get("hdr_transfer") != "smpte2084"
                else "mobius (PQ→linear npl=1000→BT.709；PQ 不向下兼容，必须映射)"
            ) if info.get("hdr") else None,
            "render_sdr_share_tonemap": (
                "hable npl=203 (HLG→linear→BT.709；苹果系统转换 ΔE 5.7)"
                if info.get("hdr_transfer") != "smpte2084"
                else "hable npl=203 (PQ→linear→BT.709)"
            ) if info.get("hdr") else None,
            "sec": round(time.time() - _t_vid, 1),
        }
        meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        index[vid] = meta

    (out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*74}")
    print(f"  完成 {len(index)} 条")
    print(f"{'='*74}")
    print(f"\n{'视频':<9} {'旋转':>4} {'编码':<11} {'输出':<11} {'方向':<6} {'帧':>4} {'主镜头':<12}", flush=True)
    print("-" * 70, flush=True)
    bad = []
    for vid, m in index.items():
        o = m["output_size"]
        ms = m["main_shot"]
        print(f"{vid:<9} {m['rotation_deg']:>3}° {m['coded_size'][0]}x{m['coded_size'][1]:<6} "
              f"{o[0]}x{o[1]:<6} {m['orientation'][:5]:<6} {m['frames']:>4} "
              f"{ms[0]}-{ms[1]}", flush=True)
        if o[0] != SHORT_SIDE and o[1] != SHORT_SIDE:
            bad.append(f"{vid}: 短边非 1080")
    if bad:
        print(f"\n⚠ {bad}")
    else:
        print(f"\n✓ 全部短边 = {SHORT_SIDE}")


if __name__ == "__main__":
    main()
