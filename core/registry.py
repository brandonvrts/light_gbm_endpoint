"""Bundle registry: filesystem + default-version pointer + drop-in for real bundles.

Layout: bundles/{contract}/{version}/bundle.joblib  (+ default.txt -> version).
Drop a production bundle into a new version dir and update default.txt to promote it;
rollback = point default.txt back. Loaded bundles are cached by path+mtime.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent.parent
# vendored platt_rescaler.py lives at ROOT so the real recal bundle unpickles
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BUNDLES = ROOT / "bundles"
_CACHE: dict[str, tuple[float, dict]] = {}


def list_contracts() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not BUNDLES.exists():
        return out
    for c in sorted(p.name for p in BUNDLES.iterdir() if p.is_dir()):
        out[c] = sorted(p.name for p in (BUNDLES / c).iterdir()
                        if p.is_dir() and (p / "bundle.joblib").exists())
    return out


def default_version(contract: str) -> str | None:
    ptr = BUNDLES / contract / "default.txt"
    if ptr.exists():
        v = ptr.read_text().strip()
        if (BUNDLES / contract / v / "bundle.joblib").exists():
            return v
    versions = list_contracts().get(contract, [])
    return versions[-1] if versions else None


def bundle_path(contract: str, version: str | None = None) -> Path | None:
    version = version or default_version(contract)
    if not version:
        return None
    p = BUNDLES / contract / version / "bundle.joblib"
    return p if p.exists() else None


def load(contract: str, version: str | None = None) -> dict:
    p = bundle_path(contract, version)
    if p is None:
        raise FileNotFoundError(f"no bundle for contract '{contract}' (version={version})")
    key = str(p)
    mtime = p.stat().st_mtime
    cached = _CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    bundle = joblib.load(p)
    bundle.setdefault("_resolved_version", p.parent.name)
    _CACHE[key] = (mtime, bundle)
    return bundle


def manifest(contract: str, version: str | None = None) -> dict:
    p = bundle_path(contract, version)
    if p is None:
        return {}
    mp = p.parent / "manifest.json"
    return json.loads(mp.read_text()) if mp.exists() else {}
