"""Bounded retries for Windows atomic-replace contention, without changing fits."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time


POLICY = {"schema": "windows-atomic-replace-retry-v1", "timeout_seconds": 30.0,
          "initial_delay_seconds": 0.1, "max_delay_seconds": 1.0,
          "retry_winerrors": [5, 32, 33], "retry_operation": "replace same-directory .tmp only",
          "on_exhaustion": "raise original error; preserve destination and temporary file",
          "fitting_or_prediction_changes": False}


def replace_with_retry(source, destination, *, replace, emit=print, timeout=30.0,
                       sleep=time.sleep, monotonic=time.monotonic):
    if timeout <= 0:
        raise ValueError("Retry timeout must be positive.")
    started = monotonic()
    failures = 0
    next_notice = 5.0
    delay = POLICY["initial_delay_seconds"]
    while True:
        try:
            result = replace(source, destination)
        except OSError as error:
            if getattr(error, "winerror", None) not in POLICY["retry_winerrors"]:
                raise
            elapsed = monotonic() - started
            remaining = timeout - elapsed
            if remaining <= 0:
                error.add_note("Atomic save remained blocked for the retry timeout. Existing files were not deleted.")
                raise
            failures += 1
            if failures == 1 or elapsed >= next_notice:
                emit(f"[save retry] WinError {error.winerror}: {Path(destination).name}; waiting, no model refit")
                next_notice = elapsed + 5.0
            sleep(min(delay, remaining))
            delay = min(delay * 2, POLICY["max_delay_seconds"])
        else:
            if failures:
                emit(f"[save recovered] {Path(destination).name}; {failures} retries, {monotonic()-started:.1f}s")
            return result


class RetryOS:
    """Proxy only the frozen persistence module's os reference, not global os."""
    def __init__(self, original, roots, *, emit=print, timeout=30.0, sleep=time.sleep, monotonic=time.monotonic):
        self.original = original
        self.roots = tuple(Path(root).resolve() for root in roots)
        self.options = dict(emit=emit, timeout=timeout, sleep=sleep, monotonic=monotonic)

    def __getattr__(self, name):
        return getattr(self.original, name)

    def replace(self, source, destination, *args, **kwargs):
        if args or kwargs:
            return self.original.replace(source, destination, *args, **kwargs)
        src, dst = Path(source).resolve(), Path(destination).resolve()
        in_scope = any(dst.is_relative_to(root) for root in self.roots)
        if not in_scope or src != dst.with_name(dst.name + ".tmp"):
            return self.original.replace(source, destination)
        return replace_with_retry(source, destination, replace=self.original.replace, **self.options)


@contextmanager
def installed(persistence_module, roots, **options):
    original = persistence_module.os
    if isinstance(original, RetryOS):
        raise RuntimeError("The save retry adapter is already active.")
    persistence_module.os = RetryOS(original, roots, **options)
    try:
        yield
    finally:
        persistence_module.os = original


def record_compatibility(p, batch_root, audit_root, expected_context, source_hashes):
    """Disclose the I/O-only extension separately; never rewrite frozen identities."""
    for name, digest in source_hashes.items():
        if p.file_digest(Path(name)) != digest:
            raise p.CheckpointError("The I/O retry source changed after it was loaded.")
    saved = json.loads((batch_root / "00_execution_identity.json").read_text(encoding="utf-8"))
    if (saved["identity_sha256"] != expected_context["batch_identity_sha256"]
            or hashlib.sha256(p.json_bytes(saved["identity"])).hexdigest() != saved["identity_sha256"]):
        raise p.CheckpointError("The original batch identity is not the verified execution identity.")
    record = {"policy": POLICY, "batch_identity_sha256": saved["identity_sha256"],
              "batch_identity_file_sha256": p.file_digest(batch_root / "00_execution_identity.json"),
              "retry_source_sha256": source_hashes, "output_root": str(batch_root),
              "compatibility": "Only transient atomic-replace failures are retried. Training, seeds, membership, arrays, scoring and checkpoint checks remain unchanged.",
              "preserved": "174 protocol, 171/176 evidence, 178 and all frozen sources, 179 identities and committed checkpoints",
              "activation": "Applies only during the 181 entrypoint process, including subsequent resumes; completed pairs remain read-only."}
    digest = hashlib.sha256(p.json_bytes(record)).hexdigest()
    path = audit_root / ("io_retry_" + digest + ".json")
    if path.exists():
        if path.read_bytes() != p.json_bytes(record):
            raise p.CheckpointError("The saved I/O compatibility record changed.")
    else:
        p.atomic_json(path, record)
    return path
