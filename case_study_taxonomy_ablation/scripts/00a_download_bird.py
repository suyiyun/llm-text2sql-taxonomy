"""
下载并解压 BIRD dev 集（见 docs/exp_design_l1l2l3_controlled_backbone.md 第 5.2 节）。

数据源: https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip (官方 dev 集，约 330MB，
2024-06-27 版本，1,534 条样本，11 个数据库)。URL 已于 2026-08 通过
bird-bench.github.io 官网核实为当前有效链接。

用法:
  python 00a_download_bird.py
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

from common import ABLATION_DIR

BIRD_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"
BIRD_DATA_DIR = ABLATION_DIR / "bird_data"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest_dir", type=str, default=str(BIRD_DATA_DIR))
    parser.add_argument("--force", action="store_true", help="Re-download/re-extract even if already present")
    return parser.parse_args()


def run(cmd, **kwargs):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def main():
    args = parse_args()
    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dest_dir / "dev.zip"
    extract_marker = dest_dir / "dev_20240627" / "dev.json"

    if extract_marker.exists() and not args.force:
        print(f"Already extracted: {extract_marker} exists. Use --force to re-download.")
        return

    if not zip_path.exists() or args.force:
        print(f"Downloading BIRD dev set ({BIRD_DEV_URL}) -- ~330MB, this can take a few minutes...")
        run(["curl", "-#", "-o", str(zip_path), BIRD_DEV_URL])
    else:
        print(f"{zip_path} already downloaded, skipping fetch.")

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)

    nested_zip = dest_dir / "dev_20240627" / "dev_databases.zip"
    if nested_zip.exists():
        print(f"Extracting nested {nested_zip}...")
        with zipfile.ZipFile(nested_zip) as zf:
            zf.extractall(dest_dir / "dev_20240627")

    if not extract_marker.exists():
        print(f"ERROR: expected {extract_marker} after extraction but it's missing.", file=sys.stderr)
        sys.exit(1)

    db_dir = dest_dir / "dev_20240627" / "dev_databases"
    n_dbs = len([p for p in db_dir.iterdir() if p.is_dir()]) if db_dir.exists() else 0
    print(f"Done. dev.json present, {n_dbs} database directories found under {db_dir}")


if __name__ == "__main__":
    main()
