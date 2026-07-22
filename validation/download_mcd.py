"""Download a subset of the MCD-rPPG dataset for development and validation.

MCD-rPPG (Multi-Camera rPPG; Yegorov et al., 2025, arXiv:2508.17924) provides facial
videos recorded simultaneously by several cameras, together with synchronized PPG and
ECG signals and per-recording clinical measurements (including reference pulse rate and
respiratory rate). It is hosted on the Hugging Face Hub, which serves large files
reliably without the download-quota limits that affect the original UBFC-rPPG Google
Drive mirror.

This script downloads a small, fixed subset -- the front-facing ``FullHDwebcam``
resting ("before") recording for a handful of subjects -- which mirrors the webcam
modality this project targets. Only a few hundred MB is fetched. Files land in
``data/mcd/`` (git-ignored, never committed).

Layout after download::

    data/mcd/
      db.csv                                  # per-recording metadata + reference HR/RR
      video/<id>_FullHDwebcam_before.avi      # ~20-85 MB compressed AVI, 29.9 fps
      ppg_sync/<id>_FullHDwebcam_before.txt   # PPG waveform + per-sample dt (seconds)

Usage::

    pip install -r requirements-dev.txt
    python validation/download_mcd.py                     # all default subjects
    python validation/download_mcd.py --subjects 1020 1035

Citation: K. Yegorov et al., "Gaze into the Heart: A Multi-View Video Dataset for rPPG
and Health Biomarkers Estimation", 2025 (arXiv:2508.17924). Dataset mirror:
https://huggingface.co/datasets/luoyongkai/mcd_rppg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HF_BASE = "https://huggingface.co/datasets/luoyongkai/mcd_rppg/resolve/main/"

# Default subset: 5 resting front-facing webcam recordings with varied reference HR/RR.
DEFAULT_SUBJECTS = ["1020", "1024", "1035", "1091", "1097"]
CAMERA = "FullHDwebcam"
STATE = "before"

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "mcd"


def _download(session, rel_path: str, dest: Path) -> bool:
    """Stream one Hugging Face file to ``dest``. Returns success; skips if present."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {rel_path} already present ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = HF_BASE + rel_path
    with session.get(url, stream=True, timeout=120) as r:
        if r.status_code != 200:
            print(f"  [FAIL] {rel_path} -> HTTP {r.status_code}")
            return False
        tmp = dest.with_suffix(dest.suffix + ".part")
        n = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                if chunk:
                    f.write(chunk)
                    n += len(chunk)
        tmp.replace(dest)
    print(f"  [ ok ] {rel_path} ({n/1e6:.1f} MB)")
    return True


def main(argv: list[str] | None = None) -> int:
    import requests  # lazy import so --help works without the dev dependency

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS,
                        help=f"Subject IDs to download (default: {DEFAULT_SUBJECTS}).")
    args = parser.parse_args(argv)

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    print(f"Downloading db.csv into {DATA_ROOT}")
    ok = _download(session, "db.csv", DATA_ROOT / "db.csv")

    print(f"Downloading {len(args.subjects)} subject(s): {args.subjects}")
    for pid in args.subjects:
        stem = f"{pid}_{CAMERA}_{STATE}"
        ok &= _download(session, f"video/{stem}.avi", DATA_ROOT / "video" / f"{stem}.avi")
        ok &= _download(session, f"ppg_sync/{stem}.txt", DATA_ROOT / "ppg_sync" / f"{stem}.txt")

    if not ok:
        print("\nOne or more files failed to download. Re-run to resume (present files are skipped).")
        return 1
    print("\nDone. All requested recordings downloaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
