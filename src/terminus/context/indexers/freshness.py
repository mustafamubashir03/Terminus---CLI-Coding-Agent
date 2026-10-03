"""Manifest-based file freshness tracking for incremental re-indexing.

Tracks per-file state (content hash, mtime, size) in a JSON manifest at
``.terminus/index/manifest.json``.  Three detection layers:

1. **mtime + size match**  → file unchanged (fastest, no disk read)
2. **mtime/size differs**  → compute SHA-256, compare hash
3. **hash differs**        → file needs re-embedding
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from terminus.config import CONFIG
from terminus.context.indexers.code_parser import get_source_files
from terminus.observability.logging import get_logger

logger = get_logger(__name__)

_HASH_CHUNK = 65536


@dataclass
class FileSnapshot:
    """Fingerprint of a single source file at a point in time."""
    path: str
    content_hash: str
    mtime: float
    size: int
    indexed_at: str


@dataclass
class DiffResult:
    """Outcome of comparing two manifest snapshots."""
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.deleted)


def compute_file_hash(filepath: str, max_size: int | None = None) -> str:
    """Return ``sha256:<hex>`` for *filepath*, reading in streaming chunks.

    If *max_size* is provided and the file exceeds it, only the first
    *max_size* bytes are hashed (avoids reading huge generated files).
    """
    h = hashlib.sha256()
    read = 0
    limit = max_size or CONFIG.get("index", {}).get("max_file_size", 1_048_576)
    with open(filepath, "rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
            if limit and read >= limit:
                break
    return f"sha256:{h.hexdigest()}"


def _fast_stat(filepath: str) -> tuple[float, int] | None:
    """Return ``(mtime, size)`` or ``None`` if stat fails."""
    try:
        st = os.stat(filepath)
        return (st.st_mtime, st.st_size)
    except OSError:
        return None


_DEFAULT_MANIFEST = ".terminus/index/manifest.json"


def _manifest_path(repo_path: str) -> Path:
    base = Path(repo_path).expanduser().resolve()
    custom = CONFIG.get("index", {}).get("manifest_path")
    if custom:
        configured = Path(custom).expanduser()
        return configured if configured.is_absolute() else base / configured
    return base / _DEFAULT_MANIFEST


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Manifest:
    """Persistent record of the last-indexed state of every source file."""

    def __init__(
        self,
        repo_path: str,
        files: dict[str, FileSnapshot] | None = None,
        last_full_index: str = "",
    ) -> None:
        self.repo_path = repo_path
        self.files: dict[str, FileSnapshot] = files or {}
        self.last_full_index = last_full_index

    @staticmethod
    def load(repo_path: str) -> Manifest:
        """Load manifest from disk.  Returns an empty manifest on any error."""
        path = _manifest_path(repo_path)
        if not path.exists():
            logger.debug(f"No manifest at {path}; starting fresh")
            return Manifest(repo_path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            files = {
                k: FileSnapshot(**v) for k, v in raw.get("files", {}).items()
            }
            return Manifest(
                repo_path=raw.get("repo_path", repo_path),
                files=files,
                last_full_index=raw.get("last_full_index", ""),
            )
        except Exception as exc:
            logger.warning(
                f"Corrupt manifest at {path}: {exc}; starting fresh"
            )
            return Manifest(repo_path)

    def save(self) -> None:
        """Persist current manifest to disk (atomic write via tmp + rename)."""
        path = _manifest_path(self.repo_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "repo_path": self.repo_path,
            "last_full_index": self.last_full_index,
            "files": {k: asdict(v) for k, v in self.files.items()},
        }
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            tmp.replace(path)
            logger.debug(f"Manifest saved ({len(self.files)} files) → {path}")
        except Exception as exc:
            logger.error(f"Failed to save manifest: {exc}")
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise

    @staticmethod
    def snapshot_directory(
        repo_path: str, max_size: int | None = None
    ) -> dict[str, FileSnapshot]:
        """Scan *repo_path* and build a snapshot for every source file.

        Uses ``os.stat()`` for mtime/size and SHA-256 for content.
        """
        files = get_source_files(repo_path)
        snapshots: dict[str, FileSnapshot] = {}
        now = _now_iso()

        for filepath in files:
            stat = _fast_stat(filepath)
            if stat is None:
                continue
            mtime, size = stat
            try:
                content_hash = compute_file_hash(filepath, max_size=max_size)
            except Exception as exc:
                logger.warning(f"Hash failed for {filepath}: {exc}")
                content_hash = "hash_error"
            snapshots[filepath] = FileSnapshot(
                path=filepath,
                content_hash=content_hash,
                mtime=mtime,
                size=size,
                indexed_at=now,
            )
        logger.info(
            f"Snapshot complete: {len(snapshots)} source files in {repo_path}"
        )
        return snapshots

    @staticmethod
    def snapshot_directory_fast(
        repo_path: str,
    ) -> dict[str, FileSnapshot]:
        """Scan files using only mtime + size (no SHA-256).

        This is used for the *initial fast pass*.  If mtime+size match,
        we skip the file entirely.  Only files whose stat differs will
        need a hash check (done in ``compute_diff``).
        """
        files = get_source_files(repo_path)
        snapshots: dict[str, FileSnapshot] = {}
        now = _now_iso()
        for filepath in files:
            stat = _fast_stat(filepath)
            if stat is None:
                continue
            mtime, size = stat
            snapshots[filepath] = FileSnapshot(
                path=filepath,
                content_hash="",
                mtime=mtime,
                size=size,
                indexed_at=now,
            )
        return snapshots

    def compute_diff(
        self, current: dict[str, FileSnapshot], repo_path: str | None = None
    ) -> DiffResult:
        """Compare *current* filesystem state against this manifest.

        Two-pass strategy:
        1.  mtime + size comparison (instant, no disk read)
        2.  Only for files where stat changed → compute SHA-256, compare hash

        Files whose mtime+size changed but content hash matches are
        marked *unchanged* (no re-embedding needed).
        """
        max_size = CONFIG.get("index", {}).get("max_file_size", 1_048_576)
        result = DiffResult()
        prev_paths = set(self.files.keys())
        curr_paths = set(current.keys())

        # They will be indexed, so record a real content hash too.
        for path in sorted(curr_paths - prev_paths):
            snap = current[path]
            try:
                snap.content_hash = compute_file_hash(path, max_size=max_size)
            except Exception as exc:
                logger.warning(f"Hash failed for new file {path}: {exc}")
                snap.content_hash = "hash_error"
            result.added.append(path)

        for path in sorted(prev_paths - curr_paths):
            result.deleted.append(path)

        for path in sorted(curr_paths & prev_paths):
            old = self.files[path]
            new = current[path]

            if old.mtime == new.mtime and old.size == new.size:
                if old.content_hash:
                    result.unchanged.append(path)
                else:
                    # Manifest didn't store hash (old format) → treat as
                    # unchanged if stat matches (conservative)
                    result.unchanged.append(path)
                continue

            try:
                new.content_hash = compute_file_hash(path, max_size=max_size)
            except Exception as exc:
                logger.warning(f"Hash check failed for {path}: {exc}")
                new.content_hash = "hash_error"

            if old.content_hash and old.content_hash == new.content_hash:
                result.unchanged.append(path)
            else:
                result.modified.append(path)

        logger.info(
            f"Diff: +{len(result.added)} added, ~{len(result.modified)} modified, "
            f"-{len(result.deleted)} deleted, ={len(result.unchanged)} unchanged"
        )
        return result
