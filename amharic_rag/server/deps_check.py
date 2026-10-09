#!/usr/bin/env python3
"""
Dependency pin verification for the Amharic S2S stack.

OmniVoice 0.2.1 requires ``transformers>=5.3.0`` (HiggsAudioV2TokenizerModel).
Sentence-Transformers 5.7.x supports Transformers v5 (``>=4.41,<6``).
Do NOT re-pin Transformers back to 4.x — that breaks OmniVoice imports.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Exact pins for packages we control tightly.
REQUIRED_EXACT: Dict[str, str] = {
    "omnivoice": "0.2.1",
    "sentence-transformers": "5.7.0",
}

# Transformers must stay on the OmniVoice-compatible v5 band.
TRANSFORMERS_MIN = (5, 3, 0)
TRANSFORMERS_MAX_EXCLUSIVE = (6, 0, 0)

# Back-compat alias used by older tests / docs.
REQUIRED_VERSIONS: Dict[str, str] = {
    **REQUIRED_EXACT,
    "transformers": ">=5.3.0,<6.0.0",
}


def installed_version(package: str) -> Optional[str]:
    try:
        import importlib.metadata as md
    except ImportError:  # pragma: no cover
        import importlib_metadata as md  # type: ignore
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return None


def _parse_version(ver: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for chunk in ver.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def transformers_ok(version: Optional[str]) -> bool:
    if not version:
        return False
    parsed = _parse_version(version)
    return parsed >= TRANSFORMERS_MIN and parsed < TRANSFORMERS_MAX_EXCLUSIVE


def check_dependency_versions(
    required: Optional[Dict[str, str]] = None,
    *,
    allow_missing: bool = False,
) -> Tuple[bool, List[Tuple[str, str, str]]]:
    """
    Returns ``(ok, mismatches)`` where each mismatch is
    ``(package, actual_or_NOT_INSTALLED, expected)``.
    """
    mismatches: List[Tuple[str, str, str]] = []

    # If a legacy exact map is passed (tests), honor exact equality.
    if required is not None:
        for pkg, expected in required.items():
            actual = installed_version(pkg)
            if actual is None:
                if allow_missing:
                    continue
                mismatches.append((pkg, "NOT INSTALLED", expected))
            elif actual != expected and not (
                pkg == "transformers" and expected.startswith(">=") and transformers_ok(actual)
            ):
                # Allow range-style expected for transformers.
                if pkg == "transformers" and "5.3" in expected and transformers_ok(actual):
                    continue
                if expected.startswith(">=") and pkg == "transformers" and transformers_ok(actual):
                    continue
                mismatches.append((pkg, actual, expected))
        return (len(mismatches) == 0), mismatches

    for pkg, expected in REQUIRED_EXACT.items():
        actual = installed_version(pkg)
        if actual is None:
            if allow_missing:
                continue
            mismatches.append((pkg, "NOT INSTALLED", expected))
        elif actual != expected:
            mismatches.append((pkg, actual, expected))

    tf_ver = installed_version("transformers")
    if tf_ver is None:
        if not allow_missing:
            mismatches.append(
                ("transformers", "NOT INSTALLED", ">=5.3.0,<6.0.0")
            )
    elif not transformers_ok(tf_ver):
        mismatches.append(
            ("transformers", tf_ver, ">=5.3.0,<6.0.0")
        )

    return (len(mismatches) == 0), mismatches


def assert_dependency_versions(
    required: Optional[Dict[str, str]] = None,
    *,
    allow_missing: bool = False,
) -> None:
    ok, mismatches = check_dependency_versions(
        required, allow_missing=allow_missing
    )
    if not ok:
        raise RuntimeError(
            "Incompatible dependency versions detected. "
            "Install with: pip install omnivoice==0.2.1 "
            "sentence-transformers==5.7.0 'transformers>=5.3.0,<6.0.0'. "
            f"Mismatches: {mismatches}"
        )


def assert_omnivoice_transformers_symbols() -> None:
    """Fail if Transformers is too old for OmniVoice's Higgs audio tokenizer."""
    try:
        from transformers import HiggsAudioV2TokenizerModel  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "transformers is installed but missing HiggsAudioV2TokenizerModel. "
            "OmniVoice 0.2.1 needs transformers>=5.3.0. "
            "Do not re-pin transformers to 4.x."
        ) from exc

    try:
        import omnivoice  # noqa: F401
        from omnivoice import OmniVoice  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "omnivoice could not be imported. Install omnivoice==0.2.1 with "
            "transformers>=5.3.0 (do not force transformers 4.x)."
        ) from exc
