from __future__ import annotations

import os
import sys
import hashlib
import gzip
import shutil
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from huggingface_hub import HfApi
from tqdm import tqdm

SOURCE_REPO = "ontocord/MixtureVitae-211BT"
TARGET_REPO = "ontocord/MixtureVitae-211BT-decontaminated"
LOCAL_ROOT  = Path("/leonardo_work/AIFAC_L01_028/hraj0000/mixturevitae/cleaned_dataset/mixtuevitae211bt-clean")
UPLOAD_LOG = Path(os.environ.get("UPLOAD_LOG", str(LOCAL_ROOT / ".uploaded_paths.log")))
MAX_WORKERS = 6
DRY_RUN = False
WRITE_README = True
README_TEXT = f"""# MixtureVitae-211BT (Decontaminated)

This repository mirrors the file structure of `ontocord/MixtureVitae-211BT` under the `data/` tree,
but each file has been **decontaminated** offline.

- Source repo: `{SOURCE_REPO}`
- Method: offline decontamination pipeline
"""

def sha256sum(p: Path, block_size: int = 2**20) -> str:
    import hashlib
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()

def _load_resume_set(api: HfApi) -> set[str]:
    """Collect paths to skip: (a) previously logged, (b) already present remotely."""
    done: set[str] = set()
    if UPLOAD_LOG.exists():
        done.update(line.strip() for line in UPLOAD_LOG.read_text().splitlines() if line.strip())
    try:
        # Also skip anything that already exists in the target repo
        remote = api.list_repo_files(repo_id=TARGET_REPO, repo_type="dataset")
        for p in remote:
            low = p.lower()
            if p.startswith("data/") and (low.endswith(".jsonl") or low.endswith(".jsonl.gz")):
                done.add(p)
    except Exception as e:
        print(f"WARN: could not fetch remote file list for resume: {e}")
    return done


def _append_to_log(path_in_repo: str) -> None:
    UPLOAD_LOG.parent.mkdir(parents=True, exist_ok=True)
    with UPLOAD_LOG.open("a", encoding="utf-8") as fh:
        fh.write(path_in_repo + "\n")
        fh.flush()
        os.fsync(fh.fileno())

def _normalize_key(name: str) -> str:
    """
    Normalize a filename key so that:
    - 'foo.jsonl'   -> 'foo.jsonl'
    - 'foo.jsonl.gz'-> 'foo.jsonl'
    We do NOT remove '.jsonl' itself.
    """
    if name.endswith(".jsonl.gz"):
        return name[:-3]  # strip only '.gz'
    return name

def build_source_mapping(api: HfApi, repo_id: str) -> Dict[str, str]:
    """
    Map normalized basename ('foo.jsonl') -> 'data/.../foo.jsonl' OR 'data/.../foo.jsonl.gz'
    Prefers exact paths from source (so we can mirror compression exactly).
    """
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    mapping: Dict[str, str] = {}
    collisions: Dict[str, List[str]] = {}

    for fp in files:
        if not fp.startswith("data/"):
            continue
        lower = fp.lower()
        if not (lower.endswith(".jsonl") or lower.endswith(".jsonl.gz")):
            continue

        base = Path(fp).name                 # e.g. foo.jsonl or foo.jsonl.gz
        key = _normalize_key(base)           # e.g. foo.jsonl

        if key in mapping and mapping[key] != fp:
            collisions.setdefault(key, []).extend([mapping[key], fp])
        mapping[key] = fp

    if collisions:
        msg = ["Duplicate normalized keys detected in source repo:"]
        for key, paths in collisions.items():
            msg.append(f"- {key}: {sorted(set(paths))}")
        raise RuntimeError(
            "\n".join(msg)
            + "\nAmbiguous (.jsonl vs .jsonl.gz or multi-path). Resolve in the source repo."
        )

    if not mapping:
        raise RuntimeError("No JSONL/JSONL.GZ files found under data/ in the source repo.")

    return mapping

def enumerate_local_files(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Local root not found: {root}")
    # Capture both .jsonl and .jsonl.gz
    # return sorted(list(root.rglob("*.jsonl")) + list(root.rglob("*.jsonl.gz")))
    return sorted(list(root.rglob("*.jsonl.gz")))

def plan_transfers(
    src_map: Dict[str, str], local_files: List[Path], local_root: Path
) -> Tuple[List[Tuple[Path, str]], List[Path], List[str]]:
    """
    Returns:
      - uploads: list of (local_path, dest_path_in_repo)
      - extras_local: local files with no destination in source mapping
      - missing_remote: normalized basenames present remotely but not found locally
    """
    local_by_key: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}

    for lp in local_files:
        base = lp.name                       # 'foo.jsonl' or 'foo.jsonl.gz'
        key = _normalize_key(base)           # normalized as 'foo.jsonl'
        if key in local_by_key and local_by_key[key] != lp:
            duplicates.setdefault(key, []).extend([local_by_key[key], lp])
        local_by_key[key] = lp

    if duplicates:
        msg = ["Duplicate normalized basenames detected in local tree:"]
        for key, paths in duplicates.items():
            msg.append(f"- {key}: {[str(p.relative_to(local_root)) for p in paths]}")
        raise RuntimeError(
            "\n".join(msg)
            + "\nEnsure only one of foo.jsonl or foo.jsonl.gz per logical file."
        )

    uploads: List[Tuple[Path, str]] = []
    extras_local: List[Path] = []
    missing_remote: List[str] = []

    # Map each local normalized key to canonical path_in_repo from source mapping
    for key, lpath in local_by_key.items():
        if key in src_map:
            uploads.append((lpath, src_map[key]))  # use exact compression/path from source
        else:
            extras_local.append(lpath)

    # What exists in remote mapping that we didn't find locally?
    for key in src_map.keys():
        if key not in local_by_key:
            missing_remote.append(key)

    return uploads, extras_local, missing_remote

def ensure_target_repo(api: HfApi, repo_id: str):
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)


def _compress_to_temp(src: Path) -> Path:
    """
    Stream-compress a .jsonl file to a temp .gz and return its path.
    Caller is responsible for deleting the temp file after upload.
    """
    # NamedTemporaryFile(delete=False) to avoid race with parallel uploads
    tmp = tempfile.NamedTemporaryFile(prefix=src.stem + "_", suffix=".jsonl.gz", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()  # we'll write to it normally
    with src.open("rb") as fin, gzip.open(tmp_path, "wb") as fout:
        shutil.copyfileobj(fin, fout, length=2**20)
    return tmp_path

def upload_one(api: HfApi, local_path: Path, dest_path_in_repo: str, repo_id: str):
    """
    If dest ends with .gz and local is plain .jsonl, compress on-the-fly.
    If dest is plain .jsonl but local is .jsonl.gz (unlikely here), we can optionally
    refuse or decompress; for safety we refuse with a clear error to avoid silent mismatches.
    """
    dest_is_gz = dest_path_in_repo.lower().endswith(".jsonl.gz")
    local_is_gz = local_path.name.lower().endswith(".jsonl.gz")

    tmp_to_cleanup: Path | None = None
    path_to_upload: Path

    if dest_is_gz:
        if local_is_gz:
            path_to_upload = local_path
        else:
            # local is plain .jsonl, compress first
            tmp_to_cleanup = _compress_to_temp(local_path)
            path_to_upload = tmp_to_cleanup
    else:
        # destination expects plain .jsonl
        if local_is_gz:
            raise RuntimeError(
                f"Destination expects plain JSONL but local is gzipped: {local_path}\n"
                f"Please provide the uncompressed .jsonl for this file or adapt the script to decompress."
            )
        path_to_upload = local_path

    print("str(path_to_upload):", str(path_to_upload))
    print("dest_path_in_repo:", dest_path_in_repo)
    try:
        api.upload_file(
            path_or_fileobj=str(path_to_upload),
            path_in_repo=dest_path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
        )
    finally:
        if tmp_to_cleanup and tmp_to_cleanup.exists():
            tmp_to_cleanup.unlink(missing_ok=True)

    return dest_path_in_repo

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set. Export your token or run `huggingface-cli login`.", file=sys.stderr)
        sys.exit(1)

    api = HfApi()

    print(f"[1/6] Scanning source repo: {SOURCE_REPO}")
    src_map = build_source_mapping(api, SOURCE_REPO)
    print(f"      Found {len(src_map)} JSONL/JSONL.GZ files under data/ (normalized by key)")

    print(f"[2/6] Scanning local cleaned root: {LOCAL_ROOT}")
    local_files = enumerate_local_files(LOCAL_ROOT)
    print(f"      Found {len(local_files)} local JSONL/JSONL.GZ files")

    print("[3/6] Planning transfers (normalize by '.jsonl' key)…")
    uploads, extras_local, missing_remote = plan_transfers(src_map, local_files, LOCAL_ROOT)
    print(f"      Will upload {len(uploads)} files to canonical paths.")
    if extras_local:
        print(f"      WARNING: {len(extras_local)} local files have no match in source mapping.")
        for p in extras_local[:10]:
            print("        extra local:", p.relative_to(LOCAL_ROOT))
        if len(extras_local) > 10:
            print("        … (truncated)")

    if missing_remote:
        print(f"      WARNING: {len(missing_remote)} source files not found locally (normalized keys).")
        for b in missing_remote[:10]:
            print("        missing local for key:", b)
        if len(missing_remote) > 10:
            print("        … (truncated)")

    # Resume support: skip already done (logged or already on Hub)
    already_done = _load_resume_set(api)
    before = len(uploads)
    uploads = [(lp, dest) for (lp, dest) in uploads if dest not in already_done]
    skipped = before - len(uploads)
    if skipped:
        print(f"      Resume: skipping {skipped} files already uploaded.")

    if DRY_RUN:
        print("\n[DRY RUN] No uploads performed.")
        sys.exit(0)

    print(f"[4/6] Ensuring target repo exists: {TARGET_REPO}")
    ensure_target_repo(api, TARGET_REPO)

    if WRITE_README:
        print("[5/6] Writing README.md")
        api.upload_file(
            path_or_fileobj=README_TEXT.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=TARGET_REPO,
            repo_type="dataset",
        )

    print("[6/6] Uploading files (sequential)…")
    for lp, dest in uploads:
        try:
            upload_one(api, lp, dest, TARGET_REPO)
            _append_to_log(dest)  # mark as done only if upload succeeded
        except Exception as e:
            print(f"ERROR uploading {lp} -> {dest}: {e}", file=sys.stderr)
            # keep going; failed ones will be retried on the next run

    print("Done ✅")
    print(f"New dataset: https://huggingface.co/datasets/{TARGET_REPO}")

if __name__ == "__main__":
    main()
