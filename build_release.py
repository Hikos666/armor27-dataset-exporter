#!/usr/bin/env python3
"""Build the Armor27 Blender add-on and full resource packages."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

VERSION = "1.0.0"
PACKAGE_NAME = "armor27-dataset-exporter"
ADDON_DIR = "armor27_dataset_exporter"
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "src" / ADDON_DIR
MANIFEST_SOURCE = HERE / "manifest.json"
DOCS = HERE / "docs"
BUILD = HERE / "build"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_tree(destination: Path, include_assets: bool) -> None:
    shutil.copytree(SOURCE, destination / ADDON_DIR)
    package_root = destination / ADDON_DIR
    if not include_assets:
        for name in ("armor_newnewnew.blend", "IDs"):
            target = package_root / name
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    shutil.copy2(MANIFEST_SOURCE, package_root / "manifest.json")
    shutil.copytree(DOCS, package_root / "docs")


def remove_development_files(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_dir() and path.name in {"__pycache__", "runs"}:
            shutil.rmtree(path)
    for path in root.rglob("*"):
        if path.is_file() and (
                path.name.startswith("test_") or path.suffix in {".pyc", ".log", ".blend1"}):
            path.unlink()


def package_manifest(root: Path, kind: str) -> dict:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(root))] = sha256(path)
    return {
        "package_name": PACKAGE_NAME,
        "version": VERSION,
        "package_type": kind,
        "blender_min_version": "5.2.0",
        "files_sha256": files,
    }


def write_zip(source_root: Path, output: Path) -> None:
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(source_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_root))


def build(kind: str, include_assets: bool, temp_root: Path) -> Path:
    staging = temp_root / kind
    staging.mkdir()
    copy_tree(staging, include_assets)
    package_root = staging / ADDON_DIR
    remove_development_files(package_root)
    manifest = package_manifest(package_root, kind)
    (package_root / "manifest.generated.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output = BUILD / f"{PACKAGE_NAME}-v{VERSION}-{kind}.zip"
    write_zip(staging, output)
    return output


def main() -> int:
    if not SOURCE.is_dir():
        print(f"missing source directory: {SOURCE}", file=sys.stderr)
        return 2
    BUILD.mkdir(parents=True, exist_ok=True)
    for old in BUILD.iterdir():
        if old.is_file() or old.is_symlink():
            old.unlink()
        elif old.is_dir():
            shutil.rmtree(old)
    with tempfile.TemporaryDirectory(prefix="armor27-release-") as temporary:
        temp_root = Path(temporary)
        outputs = [
            build("addon", include_assets=False, temp_root=temp_root),
            build("full", include_assets=True, temp_root=temp_root),
        ]
    checksums = []
    for output in outputs:
        checksums.append(f"{sha256(output)}  {output.name}")
    (BUILD / f"{PACKAGE_NAME}-v{VERSION}-SHA256SUMS.txt").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8")
    for output in outputs:
        print(f"{output} ({output.stat().st_size} bytes)")
    print(f"{BUILD / f'{PACKAGE_NAME}-v{VERSION}-SHA256SUMS.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
