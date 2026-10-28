#!/usr/bin/env python3
"""
Build the local footwear gallery from Kaggle "Fashion Product Images (Small)".

Two modes
---------
1. reproduce (default)  -- dataset/manifest.csv already exists in the repo:
   download the raw dataset, then copy *exactly* the ids listed in the
   manifest into data/gallery/ and rewrite data/metadata.csv.
   Every attribute is re-checked against styles.csv, so if the upstream
   dataset ever changes you get an error instead of a silently different set.

2. resample (--resample) -- draw a new balanced sample with a fixed seed and
   overwrite dataset/manifest.csv. Only the person maintaining the split runs
   this; everyone else uses mode 1.

Balancing policy: gender Men/Women ~50/50, and colour round-robin inside each
gender so no colour is reduced to a handful of images.

Cross-platform: macOS / Windows (pathlib, sys.executable, utf-8 I/O).

Kaggle credentials: KAGGLE_API_TOKEN env, ~/.kaggle/access_token, or kaggle.json.

Usage:
  python dataset/buildGallery.py              # reproduce the committed split
  python dataset/buildGallery.py --resample   # make a new split (maintainer)
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
GALLERY_DIR = ROOT / "data" / "gallery"
META_PATH = ROOT / "data" / "metadata.csv"
MANIFEST_PATH = ROOT / "dataset" / "manifest.csv"
DATASET_SLUG = "paramaggarwal/fashion-product-images-small"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

ATTRS = [
    "id", "category", "articleType", "color",
    "gender", "usage", "season", "year", "display_name",
]
# manifest.csv (committed) holds attributes only; metadata.csv (generated)
# adds the on-disk path column.
META_FIELDS = [ATTRS[0], "path"] + ATTRS[1:]
# styles.csv column backing each attribute
SRC = {
    "id": "id",
    "category": "subCategory",
    "articleType": "articleType",
    "color": "baseColour",
    "gender": "gender",
    "usage": "usage",
    "season": "season",
    "year": "year",
    "display_name": "productDisplayName",
}


def check_token() -> None:
    """Accept KAGGLE_API_TOKEN env, ~/.kaggle/access_token, or legacy kaggle.json."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    kdir = Path.home() / ".kaggle"
    if (kdir / "access_token").is_file() or (kdir / "kaggle.json").is_file():
        return
    raise SystemExit(
        "Missing Kaggle credentials. Provide ONE of:\n"
        "  - env :  export KAGGLE_API_TOKEN=KGAT_...   (new API token)\n"
        "  - file:  ~/.kaggle/access_token  (chmod 600)\n"
        "  - file:  ~/.kaggle/kaggle.json   (legacy username/key)\n"
        "Create at https://www.kaggle.com/settings -> API."
    )


def download(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not force and (
        list(RAW_DIR.rglob("styles.csv")) or list(RAW_DIR.glob("*.zip"))
    ):
        print("Dataset already present; skipping download.")
        return
    print(f"Downloading {DATASET_SLUG} ... (~565MB)")
    subprocess.run(
        [
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", DATASET_SLUG, "-p", str(RAW_DIR), "--force",
        ],
        check=True,
    )


def extract() -> None:
    if list(RAW_DIR.rglob("styles.csv")):
        print("Already extracted; skipping unzip.")
        return
    zips = sorted(RAW_DIR.glob("*.zip"))
    if not zips:
        raise SystemExit(f"No zip found in {RAW_DIR}")
    for z in zips:
        print(f"Extracting {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW_DIR)
    # tidy up: drop the zip now that it is extracted
    for z in zips:
        try:
            z.unlink()
            print(f"Removed {z.name}")
        except OSError:
            pass


def find_styles() -> Path:
    cands = sorted(RAW_DIR.rglob("styles.csv"), key=lambda p: len(p.parts))
    if cands:
        return cands[0]
    raise SystemExit(f"styles.csv not found under {RAW_DIR}")


def find_images_dir() -> Path:
    dirs = [
        d for d in RAW_DIR.rglob("images")
        if d.is_dir() and any(
            f.is_file() and f.suffix.lower() in IMAGE_EXTS for f in d.iterdir()
        )
    ]
    if not dirs:
        raise SystemExit(f"images/ directory not found under {RAW_DIR}")
    return sorted(dirs, key=lambda p: len(p.parts))[0]


def _s(raw: dict, key: str) -> str:
    """Return a stripped string field; tolerate list/None from malformed rows."""
    v = raw.get(key, "")
    return v.strip() if isinstance(v, str) else ""


def load_footwear(styles: Path) -> list[dict]:
    """All footwear rows of styles.csv, normalised to the ATTRS schema."""
    rows: list[dict] = []
    with styles.open(encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            if _s(raw, "masterCategory").lower() != "footwear":
                continue
            r = {a: _s(raw, SRC[a]) for a in ATTRS}
            if not r["id"]:
                continue
            rows.append(r)
    if not rows:
        raise SystemExit("No Footwear rows found in styles.csv")
    return rows


def balanced_sample(rows: list[dict], target: int, seed: int,
                    genders: tuple[str, ...] = ("Men", "Women")) -> list[dict]:
    """Even gender split; round-robin over colour within each gender."""
    rng = random.Random(seed)
    by_gender: dict[str, list[dict]] = {}
    for g in genders:
        pool = [r for r in rows if r["gender"] == g]
        if not pool:
            raise SystemExit(f"No rows for gender {g!r}")
        by_gender[g] = pool

    base = target // len(genders)
    quotas = {g: base for g in genders}
    quotas[genders[0]] += target - base * len(genders)  # remainder -> first

    picked: list[dict] = []
    for g in genders:
        by_color: dict[str, list[dict]] = defaultdict(list)
        for r in by_gender[g]:
            by_color[r["color"] or "unknown"].append(r)
        for c in by_color:
            rng.shuffle(by_color[c])

        # big colours first so the round-robin anchors on common ones, then
        # scarce colours still get their (limited) fair share
        order = sorted(by_color, key=lambda c: len(by_color[c]), reverse=True)
        idx = {c: 0 for c in order}
        active = list(order)
        need = quotas[g]
        got = 0
        while got < need and active:
            progressed = False
            for c in list(active):
                if got >= need:
                    break
                if idx[c] < len(by_color[c]):
                    picked.append(by_color[c][idx[c]])
                    idx[c] += 1
                    got += 1
                    progressed = True
                if idx[c] >= len(by_color[c]):
                    active.remove(c)
            if not progressed:
                break
        print(f"  {g}: picked {got} (quota {need})")
    return picked


def write_manifest(picked: list[dict]) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ATTRS)
        w.writeheader()
        for r in picked:
            w.writerow({a: r[a] for a in ATTRS})
    print(f"Manifest ({len(picked)} ids) -> {MANIFEST_PATH}")


def read_manifest() -> list[dict]:
    if not MANIFEST_PATH.is_file():
        raise SystemExit(
            f"{MANIFEST_PATH} not found. It is committed with the repo and "
            "pins the 1000 images. If you really want a new split, run:\n"
            "  python scripts/download_footwear.py --resample"
        )
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("id")]


def materialise(manifest: list[dict], styles_rows: list[dict],
                images_dir: Path) -> None:
    """Copy exactly the manifest ids; verify attributes against styles.csv."""
    by_id = {r["id"]: r for r in styles_rows}
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    for f in GALLERY_DIR.iterdir():      # clear previous gallery
        if f.is_file():
            f.unlink()

    missing: list[str] = []
    changed: list[str] = []
    written = 0
    with META_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        w.writeheader()
        for m in manifest:
            img_id = m["id"]
            cur = by_id.get(img_id)
            if cur is None:
                missing.append(img_id)
                continue
            diff = [a for a in ATTRS if a != "id" and (m.get(a) or "") != cur[a]]
            if diff:
                changed.append(f"{img_id}: {','.join(diff)}")

            src = None
            for ext in (".jpg", ".jpeg", ".png"):
                cand = images_dir / f"{img_id}{ext}"
                if cand.is_file():
                    src = cand
                    break
            if src is None:
                missing.append(img_id)
                continue
            dst = GALLERY_DIR / src.name
            if not dst.exists():
                dst.write_bytes(src.read_bytes())
            w.writerow(
                {
                    **{a: cur[a] for a in ATTRS},
                    "path": dst.relative_to(ROOT).as_posix(),
                }
            )
            written += 1

    print(f"Wrote {written}/{len(manifest)} images -> {GALLERY_DIR}")
    print(f"Metadata -> {META_PATH}")
    if changed:
        print(f"WARNING: {len(changed)} rows differ from styles.csv, first 5:")
        for line in changed[:5]:
            print("  ", line)
    if missing:
        print(f"ERROR: {len(missing)} manifest ids unusable, first 5: {missing[:5]}")
        print("The Kaggle dataset may have been re-uploaded. Re-check styles.csv /"
              " images, or regenerate the split with --resample.")
        raise SystemExit(1)


def report(rows: list[dict]) -> None:
    print("\n== sampled distribution ==")
    print("gender     :", dict(Counter(r["gender"] for r in rows)))
    print("color      :", dict(Counter(r["color"] for r in rows).most_common()))
    print("articleType:", dict(Counter(r["articleType"] for r in rows).most_common()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resample", action="store_true",
                    help="draw a new balanced split and overwrite data/manifest.csv")
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    check_token()
    download(force=args.force)
    extract()
    styles_rows = load_footwear(find_styles())
    images_dir = find_images_dir()
    print(f"Footwear rows in styles.csv: {len(styles_rows)}")

    if args.resample:
        picked = balanced_sample(styles_rows, args.target, args.seed)
        write_manifest(picked)
        manifest = picked
    else:
        manifest = read_manifest()
        print(f"Reproducing committed split: {len(manifest)} ids")

    materialise(manifest, styles_rows, images_dir)
    report(manifest)


if __name__ == "__main__":
    main()
