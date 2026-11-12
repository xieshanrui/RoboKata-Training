"""Losslessly archive completed native frame caches, verify, then remove exact originals."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def contained(path, root):
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError(f"Path escapes the explicit episode root: {path}")
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        raise ValueError(f"Linked frame paths are not supported: {path}")
    return resolved


def stream_hash(stream):
    digest, size = hashlib.sha256(), 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


def archive_episode(episode, result=None):
    """The collector may supply its completed result before its final JSON write."""
    root = episode.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Expected an episode directory: {root}")
    if result is None:
        result = read(contained(root / "result.json", root))
    if result.get("complete") is not True:
        raise ValueError(f"Refusing incomplete episode: {root}")
    archive = root / "frames.zip"
    receipt_path = root / "frames_archive.json"
    if archive.exists() or archive.is_symlink() or receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError(f"Refusing an existing frame archive or receipt: {root}")
    cache = root / ".cache" / "episode0"
    contained(root / ".cache", root)
    contained(cache, root)
    frames = sorted(cache.glob("*.pkl"))
    if not frames:
        raise ValueError(f"No native frame files to archive: {cache}")
    for path in frames:
        if not contained(path, root).is_file():
            raise ValueError(f"Expected a regular frame file: {path}")
    manifest = dict(version=1, compression="ZIP_DEFLATED", compression_level=1, files=[])
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as output:
        for path in frames:
            name = path.relative_to(root).as_posix()
            with contained(path, root).open("rb") as source, output.open(name, "w", force_zip64=True) as target:
                digest, size = hashlib.sha256(), 0
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(block)
                    digest.update(block)
                    size += len(block)
            manifest["files"].append(dict(path=name, sha256=digest.hexdigest(), size_bytes=size))
        output.writestr("manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
    # Flush archive bytes to storage before the readback and any source removal.
    with archive.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    with zipfile.ZipFile(archive, "r") as source:
        names = [entry["path"] for entry in manifest["files"]] + ["manifest.json"]
        if source.namelist() != names or json.loads(source.read("manifest.json").decode("utf-8")) != manifest:
            raise ValueError("Archive member list or manifest readback mismatch; originals retained")
        for entry in manifest["files"]:
            with source.open(entry["path"], "r") as stream:
                actual = stream_hash(stream)
            if actual != (entry["sha256"], entry["size_bytes"]):
                raise ValueError(f"Archive frame hash mismatch; originals retained: {entry['path']}")
    # Verify the entire original set is unchanged before removing even one file.
    if sorted(cache.glob("*.pkl")) != frames:
        raise ValueError("Frame set changed during archiving; originals retained")
    for path, entry in zip(frames, manifest["files"], strict=True):
        with contained(path, root).open("rb") as stream:
            if stream_hash(stream) != (entry["sha256"], entry["size_bytes"]):
                raise ValueError(f"Original frame changed; originals retained: {path}")
    with archive.open("rb") as stream:
        archive_sha256, archive_bytes = stream_hash(stream)
    receipt = dict(version=1, archive=archive.name, archive_sha256=archive_sha256,
        compression="ZIP_DEFLATED", compression_level=1, verified=True,
        manifest="frames.zip/manifest.json", frame_count=len(frames),
        original_bytes=sum(entry["size_bytes"] for entry in manifest["files"]),
        archive_bytes=archive_bytes, removed_originals=[])
    # No directory removal, wildcard deletion, or shell commands: only verified paths.
    try:
        for path, entry in zip(frames, manifest["files"], strict=True):
            resolved = contained(path, root)
            with resolved.open("rb") as stream:
                if stream_hash(stream) != (entry["sha256"], entry["size_bytes"]):
                    raise ValueError(f"Frame changed before removal; verified archive retained: {path}")
            resolved.unlink()
            receipt["removed_originals"].append(entry["path"])
    finally:
        receipt["removal_complete"] = len(receipt["removed_originals"]) == len(frames)
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode", type=Path, help="One completed native attempt directory")
    source.add_argument("--collection", type=Path, help="Archive completed attempts with raw frame caches below this directory")
    args = parser.parse_args()
    if args.episode:
        episodes = [args.episode]
    else:
        root = args.collection.resolve(strict=True)
        episodes = []
        for path in sorted(root.rglob("result.json")):
            contained(path, root)
            if read(path).get("complete") is True and (path.parent / ".cache" / "episode0").is_dir():
                episodes.append(path.parent)
        if not episodes:
            parser.error("No completed attempts with native frame caches found")
    for episode in episodes:
        receipt = archive_episode(episode)
        print(json.dumps(dict(episode=str(episode.resolve()), **receipt)), flush=True)


if __name__ == "__main__":
    main()
