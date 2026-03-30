from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


REPO_ID = "bitmind/Kinetics-700"
REPO_REVISION = "b35b15a4e3e0940b9495dd27de325c1ed1b3246a"
BASE_URL = f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REPO_REVISION}"


def download_part(part_name: str, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    target = download_dir / part_name
    if target.exists():
        return target
    url = f"{BASE_URL}/{part_name}?download=true"
    print(f"downloading {url}")
    urlretrieve(url, target)
    return target


def extract_subset(zip_path: Path, output_dir: Path, remaining: int) -> int:
    extracted = 0
    with zipfile.ZipFile(zip_path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".mp4")]
        for name in names:
            if extracted >= remaining:
                break
            destination = output_dir / Path(name).name
            if destination.exists():
                extracted += 1
                continue
            with archive.open(name) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
    return extracted


def write_manifest(video_dir: Path, manifest_path: Path) -> None:
    videos = sorted(path.resolve() for path in video_dir.glob("*.mp4"))
    manifest_path.write_text("\n".join(str(path) for path in videos) + "\n", encoding="utf-8")


def run(output_dir: str, download_dir: str, parts: list[str], max_videos: int) -> None:
    output_path = Path(output_dir).expanduser().resolve()
    video_dir = output_path / "videos"
    manifest_path = output_path / "manifest.txt"
    download_path = Path(download_dir).expanduser().resolve()

    video_dir.mkdir(parents=True, exist_ok=True)
    extracted_total = len(list(video_dir.glob("*.mp4")))

    for part_name in parts:
        if extracted_total >= max_videos:
            break
        zip_path = download_part(part_name, download_path)
        remaining = max_videos - extracted_total
        extracted = extract_subset(zip_path, video_dir, remaining)
        extracted_total += extracted
        print(f"extracted {extracted} videos from {part_name}; total={extracted_total}")

    write_manifest(video_dir, manifest_path)
    print(f"manifest written to {manifest_path}")


def main(cfg: dict | None = None) -> None:
    if cfg is not None:
        data_cfg = cfg.get("data", {})
        run(
            output_dir=data_cfg.get("output_dir", "data/kinetics700_subset"),
            download_dir=data_cfg.get("download_dir", "downloads/kinetics700"),
            parts=data_cfg.get("parts", ["Kinetics700_part_001.zip"]),
            max_videos=int(data_cfg.get("max_videos", 1000)),
        )
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/kinetics700_subset")
    parser.add_argument("--download-dir", default="downloads/kinetics700")
    parser.add_argument("--parts", nargs="+", default=["Kinetics700_part_001.zip"])
    parser.add_argument("--max-videos", type=int, default=1000)
    args = parser.parse_args()
    run(args.output_dir, args.download_dir, args.parts, args.max_videos)


if __name__ == "__main__":
    main()
