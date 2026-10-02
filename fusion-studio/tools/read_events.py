#!/usr/bin/env python3

import argparse
import os
import struct
import sys
import warnings

import numpy as np

MAGIC = b"FSEV"
VERSION = 1
_HEADER = struct.Struct("<4sIIIq")
_T_MASK = (1 << 39) - 1


def read_events(path):
    if os.path.isdir(path):
        path = os.path.join(path, "events.bin")
    size = os.path.getsize(path)
    if size < _HEADER.size:
        raise ValueError(f"{path}: 文件比 24 字节的头部还短({size} 字节),不是 events.bin")
    with open(path, "rb") as f:
        magic, version, width, height, t0_us = _HEADER.unpack(f.read(_HEADER.size))
    if magic != MAGIC:
        raise ValueError(f"{path}: magic 是 {magic!r},不是 {MAGIC!r} —— 不是 events.bin")
    if version != VERSION:
        raise ValueError(f"{path}: version={version},本工具只认 v{VERSION}")

    payload = size - _HEADER.size
    torn = payload % 8
    if torn:
        warnings.warn(f"{path}: 尾部有 {torn} 字节撕裂记录(写入端未正常收尾),已截断")
    words = np.fromfile(path, dtype="<u8", offset=_HEADER.size, count=payload // 8)

    return {
        "width": width,
        "height": height,
        "t0_us": t0_us,
        "n": words.size,
        "t": ((words & _T_MASK) + np.int64(t0_us)).astype(np.int64),
        "x": ((words >> 39) & 0xFFF).astype(np.uint16),
        "y": ((words >> 51) & 0xFFF).astype(np.uint16),
        "p": (words >> 63).astype(np.uint8),
    }


def _main():
    ap = argparse.ArgumentParser(description="解码 fusion-studio 的 events.bin")
    ap.add_argument("path", help="take 目录或 events.bin 路径")
    ap.add_argument("--npz", help="另存为 .npz(数组 t/x/y/p + 标量 width/height/t0_us)")
    args = ap.parse_args()

    ev = read_events(args.path)
    print(f"geometry : {ev['width']}x{ev['height']}")
    print(f"events   : {ev['n']:,}")
    if ev["n"]:
        dur_s = (ev["t"][-1] - ev["t"][0]) / 1e6
        mono = bool(np.all(np.diff(ev["t"]) >= 0))
        print(f"t0_us    : {ev['t0_us']}")
        print(f"span     : {dur_s:.3f} s  ({ev['n'] / dur_s / 1e6:.2f} Mev/s)" if dur_s > 0 else "span     : 0 s")
        print(f"monotonic: {mono}")
        print(f"x range  : [{ev['x'].min()}, {ev['x'].max()}]   y range: [{ev['y'].min()}, {ev['y'].max()}]")
        print(f"polarity : ON {int((ev['p'] == 1).sum()):,} / OFF {int((ev['p'] == 0).sum()):,}")
    else:
        print("(零事件 take:纯头部文件)")
    if args.npz:
        np.savez_compressed(args.npz, **ev)
        print(f"saved -> {args.npz}")


if __name__ == "__main__":
    sys.exit(_main())
