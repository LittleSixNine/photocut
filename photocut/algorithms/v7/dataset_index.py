"""Metadata-only development index and permission-gated frozen access."""
from __future__ import annotations
import hashlib, json, secrets, threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from .evaluation_manifest import validate_public_manifest, FrozenManifestError


def resolve_object_path_under_root(dataset_root, object_path):
    """Resolve an explicit object path without following it outside a trusted root."""
    root = Path(dataset_root).resolve(strict=True)
    raw = Path(object_path)
    candidate = raw if raw.is_absolute() else root / raw
    if candidate.is_symlink():
        raise PermissionError("symlink object paths are not allowed")
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise PermissionError("object path must be a regular file") from exc
    if not resolved.is_relative_to(root):
        raise PermissionError("object path escapes dataset_root")
    if resolved.is_symlink() or not resolved.is_file():
        raise PermissionError("object path must be a regular file")
    return resolved

@dataclass(frozen=True)
class IndexedSample:
    image_id: str
    origin_group_id: str
    split: str
    metadata: Mapping[str,Any]
    object_path: str | None = None
    materialized: bool = field(default=False, compare=False)

class FinalizationToken:
    __slots__=("_value","_vault_root","_auth_id")
    _registry = {}
    _registry_lock = threading.RLock()

    def __init__(self, *args, **kwargs):
        raise PermissionError("token must be issued by factory")

    def __setattr__(self, name, value):
        raise AttributeError("finalization token is immutable")

    def __delattr__(self, name):
        raise AttributeError("finalization token is immutable")

    @property
    def value(self):
        return self._value

    @property
    def vault_root(self):
        return self._vault_root

    @classmethod
    def issue(cls, vault_root):
        secret=secrets.token_urlsafe(32)
        auth_id=secrets.token_urlsafe(32)
        token=object.__new__(cls)
        object.__setattr__(token, "_value", secret)
        object.__setattr__(token, "_vault_root", Path(vault_root).resolve())
        object.__setattr__(token, "_auth_id", auth_id)
        with cls._registry_lock:
            cls._registry[auth_id] = {"token": token, "value": secret, "vault_root": token._vault_root, "consumed": False}
        return token

    @classmethod
    def consume(cls, token, expected_token, vault_root):
        with cls._registry_lock:
            auth_id = getattr(token, "_auth_id", None)
            record = cls._registry.get(auth_id)
            if record is None or record["token"] is not token or record["consumed"]:
                raise PermissionError("invalid or consumed finalization authorization")
            if not isinstance(expected_token, str) or not secrets.compare_digest(record["value"], expected_token):
                raise PermissionError("opaque token mismatch")
            if Path(vault_root).resolve() != record["vault_root"]:
                raise PermissionError("vault binding mismatch")
            record["consumed"] = True

class FrozenVaultAccess:
    def __init__(self, vault_path, finalization_token: FinalizationToken | None = None, expected_token: str | None = None):
        if finalization_token is None or not isinstance(finalization_token,FinalizationToken): raise PermissionError("finalization token required")
        self.vault_path=Path(vault_path)
        FinalizationToken.consume(finalization_token, expected_token, self.vault_path)
    def read(self): return json.loads(self.vault_path.read_text())
    def materialize(self, object_path, expected_object_hash=None, decoder=None):
        """Permission-gated frozen object access; no caller can bypass the token."""
        if not expected_object_hash: raise ValueError("expected object hash required")
        op=Path(object_path).resolve()
        if not op.is_relative_to(self.vault_path.resolve().parent): raise PermissionError("object outside vault root")
        data=op.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_object_hash.removeprefix("sha256:"): raise ValueError("object hash mismatch")
        if decoder is None:
            try:
                from .input import decode_bytes
            except ImportError: decode_bytes=None
            decoder=decode_bytes
        if decoder is None: raise RuntimeError("algorithms.v7.input.decode_bytes unavailable")
        return decoder(data)

def load_sample_metadata(manifest_path, image_id=None, origin_group_id=None, split=None, allowed_splits=("train","validation")):
    samples=validate_public_manifest(manifest_path)
    allowed=set(allowed_splits)
    result=[]
    for s in samples:
        if s["split"] not in allowed: continue
        if image_id is not None and s["image_id"] != image_id: continue
        if origin_group_id is not None and s["origin_group_id"] != origin_group_id: continue
        if split is not None and s["split"] != split: continue
        result.append(IndexedSample(s["image_id"],s["origin_group_id"],s["split"],dict(s),s.get("object_path")))
    if image_id is not None and not result: raise KeyError(image_id)
    return result[0] if image_id is not None else result

def materialize_sample(sample: IndexedSample, decoder=None):
    if sample.split not in {"train", "validation"}: raise PermissionError("frozen samples require FrozenVaultAccess")
    if not sample.object_path: raise ValueError("sample has no object path")
    expected=sample.metadata.get("expected_object_hash")
    if not expected: raise ValueError("expected object hash required")
    data=Path(sample.object_path).read_bytes()
    digest=hashlib.sha256(data).hexdigest()
    if digest != str(expected).removeprefix("sha256:"): raise ValueError("object hash mismatch")
    if decoder is None:
        try:
            from .input import decode_bytes
        except ImportError: decode_bytes=None
        decoder=decode_bytes
    if decoder is None: raise RuntimeError("algorithms.v7.input.decode_bytes unavailable")
    return decoder(data)

def assign_splits(origin_group_ids, existing=None, validation_fraction=0.2, conflicts=None):
    if not isinstance(validation_fraction,(int,float)) or isinstance(validation_fraction,bool) or not 0 <= validation_fraction <= 1: raise ValueError("validation_fraction must be in [0,1]")
    existing=dict(existing or {}); result={};
    for g in origin_group_ids:
        if g in existing:
            if existing[g] not in {"train", "validation"} and conflicts is not None:
                conflicts.append((g, existing[g]))
            elif existing[g] not in {"train", "validation"}:
                raise ValueError(f"invalid existing split for {g}: {existing[g]}")
            result[g]=existing[g]; continue
        n=int(hashlib.sha256(str(g).encode()).hexdigest()[:8],16)/0xffffffff
        result[g]="validation" if n < validation_fraction else "train"
    return result
