#!/usr/bin/env python3

import argparse
import json
import os
import sys

import numpy as np


def least_squares_ray_intersection(rays):
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c, d in rays:
        d = np.asarray(d, dtype=np.float64)
        d = d / max(np.linalg.norm(d), 1e-12)
        c = np.asarray(c, dtype=np.float64)
        proj = np.eye(3) - np.outer(d, d)
        A += proj
        b += proj @ c
    return np.linalg.solve(A, b)


def _w2c_centre_and_view_dir(w2c):
    R = w2c[:3, :3].astype(np.float64)
    T = w2c[:3, 3].astype(np.float64)
    return -R.T @ T, R[2, :].astype(np.float64)


def normalize_to_unit_sphere(w2c0, w2c1):
    c0, d0 = _w2c_centre_and_view_dir(w2c0)
    c1, d1 = _w2c_centre_and_view_dir(w2c1)
    P = least_squares_ray_intersection([(c0, d0), (c1, d1)])

    out0 = w2c0.copy().astype(np.float64)
    out1 = w2c1.copy().astype(np.float64)
    out0[:3, 3] = w2c0[:3, :3] @ P + w2c0[:3, 3]
    out1[:3, 3] = w2c1[:3, :3] @ P + w2c1[:3, 3]

    max_dist = max(np.linalg.norm(c0 - P), np.linalg.norm(c1 - P))
    scale = 1.0 / max(max_dist, 1e-12)
    out0[:3, 3] *= scale
    out1[:3, 3] *= scale
    return out0, out1, P, float(scale)


def normalize_calibration(path):
    if os.path.isdir(path):
        path = os.path.join(path, "calibration.json")
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("schema") != "stereo_calibration.v1":
        raise ValueError(f"{path}: schema 是 {doc.get('schema')!r},需要 stereo_calibration.v1")

    ex = doc["extrinsics"]
    w2c1 = np.eye(4)
    w2c1[:3, :3] = np.asarray(ex["R_cam0_to_cam1"], dtype=np.float64)
    w2c1[:3, 3] = np.asarray(ex["T_cam0_to_cam1"], dtype=np.float64)

    out0, out1, P, scale = normalize_to_unit_sphere(np.eye(4), w2c1)
    return {
        "scene_normalization": {
            "applied": True,
            "origin_in_extrinsics_frame": [float(v) for v in P],
            "scale": scale,
        },
        "cameras": {
            "cam0": {"w2c": out0.tolist()},
            "cam1": {"w2c": out1.tolist()},
        },
    }


def _main():
    ap = argparse.ArgumentParser(description="双目外参场景尺度归一化(单位球)")
    ap.add_argument("path", help="stereo_calibration.v1.json 或 take 目录")
    ap.add_argument("--out", help="结果另存为 JSON")
    args = ap.parse_args()

    r = normalize_calibration(args.path)
    sn = r["scene_normalization"]
    print(f"origin (extrinsics frame): {np.round(sn['origin_in_extrinsics_frame'], 4).tolist()}")
    print(f"scale                    : {sn['scale']:.6g}")
    for cam in ("cam0", "cam1"):
        w2c = np.asarray(r["cameras"][cam]["w2c"])
        centre = -w2c[:3, :3].T @ w2c[:3, 3]
        print(f"{cam} centre (normalized) : {np.round(centre, 4).tolist()}  |c|={np.linalg.norm(centre):.4f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=2)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    sys.exit(_main())
