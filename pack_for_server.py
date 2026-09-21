#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打包本地 Python 项目依赖为 Linux wheel，供离线部署到服务器。

流程：
  1. 读取 requirements.txt
  2. 用 pip download 下载 manylinux2014_x86_64 / cp311 的 wheel
  3. 可选：把项目目录 + wheel 目录 + requirements.txt 打包成 tar.gz

用法：
  # 只下载 wheel
  python pack_for_server.py --project D:\\pythonPorjects\\airllmProjects\\your_project

  # 下载并打包
  python pack_for_server.py --project D:\\pythonPorjects\\airllmProjects\\your_project --pack

  # 自定义 wheel 目录和输出包名
  python pack_for_server.py --project ./your_project --wheels ./myapp_wheels --pack --out myapp.tar.gz
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def run(cmd):
    print("\n>>> " + " ".join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        print(f"[ERROR] 命令失败，返回码 {ret}")
        sys.exit(ret)


def load_requirements(req_path):
    if not os.path.isfile(req_path):
        print(f"[ERROR] 找不到 {req_path}")
        sys.exit(1)
    with open(req_path, encoding="utf-8") as f:
        lines = [l.strip() for l in f
                 if l.strip() and not l.strip().startswith("#")]
    return lines


def download_wheels(req_path, dest, python_version, abi, platform, mirror,
                    trusted, retries, timeout):
    os.makedirs(dest, exist_ok=True)
    cmd = [
        sys.executable, "-m", "pip", "download",
        "-r", req_path,
        "--dest", dest,
        "--platform", platform,
        "--python-version", python_version,
        "--implementation", "cp",
        "--abi", abi,
        "--only-binary=:all:",
        "-i", mirror,
        "--retries", str(retries),
        "--timeout", str(timeout),
    ]
    if trusted:
        cmd += ["--trusted-host", trusted]
    run(cmd)


def pack_tarball(items, out_path):
    """把 items 列表（文件/目录）打包成 .tar.gz"""
    out_path = os.path.abspath(out_path)
    print(f"\n[INFO] 打包到: {out_path}")
    with tarfile.open(out_path, "w:gz") as tar:
        for item in items:
            if not os.path.exists(item):
                print(f"[WARN] 跳过不存在的: {item}")
                continue
            arcname = os.path.basename(os.path.normpath(item))
            print(f"  添加: {item}  ->  {arcname}")
            tar.add(item, arcname=arcname)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"[DONE] {out_path}  ({size_mb:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True,
                    help="本地项目根目录（含 requirements.txt）")
    ap.add_argument("--req", default=None,
                    help="requirements.txt 路径，默认 <project>/requirements.txt")
    ap.add_argument("--wheels", default="./myapp_wheels",
                    help="wheel 下载目录，默认 ./myapp_wheels")
    ap.add_argument("--python-version", default="3.11",
                    help="目标 Python 版本，默认 3.11")
    ap.add_argument("--abi", default="cp311",
                    help="目标 ABI，默认 cp311")
    ap.add_argument("--platform", default="manylinux2014_x86_64",
                    help="目标平台，默认 manylinux2014_x86_64")
    ap.add_argument("--mirror",
                    default="https://pypi.tuna.tsinghua.edu.cn/simple",
                    help="PyPI 镜像")
    ap.add_argument("--trusted", default="pypi.tuna.tsinghua.edu.cn",
                    help="trusted-host，留空则不加")
    ap.add_argument("--retries", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--pack", action="store_true",
                    help="下完后打包成 tar.gz")
    ap.add_argument("--out", default="myapp.tar.gz",
                    help="打包输出文件名，默认 myapp.tar.gz")
    ap.add_argument("--clean-wheels", action="store_true",
                    help="下载前清空 wheel 目录")
    args = ap.parse_args()

    project = os.path.abspath(args.project)
    if not os.path.isdir(project):
        print(f"[ERROR] 项目目录不存在: {project}")
        sys.exit(1)

    req_path = args.req or os.path.join(project, "requirements.txt")
    wheels_dir = os.path.abspath(args.wheels)

    print("=" * 70)
    print(f"[INFO] 项目目录     : {project}")
    print(f"[INFO] requirements : {req_path}")
    print(f"[INFO] wheel 目录   : {wheels_dir}")
    print(f"[INFO] 目标平台     : {args.platform}")
    print(f"[INFO] 目标 Python  : {args.python_version} ({args.abi})")
    print(f"[INFO] 镜像         : {args.mirror}")
    print("=" * 70)

    reqs = load_requirements(req_path)
    print(f"[INFO] requirements.txt 里 {len(reqs)} 个包")
    for r in reqs:
        print("  -", r)

    if args.clean_wheels and os.path.isdir(wheels_dir):
        print(f"[INFO] 清空 {wheels_dir}")
        for f in os.listdir(wheels_dir):
            p = os.path.join(wheels_dir, f)
            if os.path.isfile(p):
                os.remove(p)

    if os.path.isdir(wheels_dir):
        existing = [f for f in os.listdir(wheels_dir) if f.endswith(".whl")]
        if existing:
            print(f"[INFO] wheel 目录已有 {len(existing)} 个，pip 会复用")

    download_wheels(req_path, wheels_dir, args.python_version, args.abi,
                    args.platform, args.mirror, args.trusted or None,
                    args.retries, args.timeout)

    whls = sorted(f for f in os.listdir(wheels_dir) if f.endswith(".whl"))
    total = sum(os.path.getsize(os.path.join(wheels_dir, f)) for f in whls)

    print("\n" + "=" * 70)
    print(f"[RESULT] 共 {len(whls)} 个 wheel，{total/1024/1024:.1f} MB")
    print(f"[RESULT] 目录: {wheels_dir}")
    print("=" * 70)
    for f in whls:
        print("  +", f)

    if args.pack:
        items = [project, wheels_dir, req_path]
        pack_tarball(items, args.out)

        print("\n下一步（本地执行）：")
        print(f"  scp {args.out} user@服务器IP:/opt/qwen/")
    else:
        print("\n下一步（本地执行）：")
        print(f"  tar -czf myapp.tar.gz {os.path.basename(project)} "
              f"{os.path.basename(wheels_dir)} {os.path.basename(req_path)}")
        print(f"  scp myapp.tar.gz user@服务器IP:/opt/qwen/")

    print("\n服务器上执行：")
    print("  cd /opt/qwen")
    print(f"  tar -xzf {os.path.basename(args.out) if args.pack else 'myapp.tar.gz'}")
    print("  python3.11 -m venv /opt/qwen/myapp_venv")
    print("  source /opt/qwen/myapp_venv/bin/activate")
    print(f"  pip install --no-index --find-links ./{os.path.basename(wheels_dir)} "
          "-r requirements.txt")
    print("  cd " + os.path.basename(project))
    print("  export VLLM_BASE_URL=http://127.0.0.1:9080/v1")
    print("  nohup uvicorn main:app --host 0.0.0.0 --port 8000 "
          "> /opt/qwen/myapp.log 2>&1 &")


if __name__ == "__main__":
    main()