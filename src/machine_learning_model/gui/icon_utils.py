"""Icon / glyph utilities with graceful fallback.

If emoji rendering is not supported, maps statuses to simple ASCII tokens.
Set DISABLE_EMOJI=1 to force ASCII fallback.
"""
from __future__ import annotations

import os
from pathlib import Path

# Determine fallback: explicit opt-out OR missing common emoji fonts
_emoji_forced_off = os.getenv("DISABLE_EMOJI", "0") == "1" or os.getenv("NO_EMOJI", "0") == "1"
_known_emoji_fonts = [
    Path("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"),
    Path("/usr/share/fonts/truetype/seguiemj.ttf"),  # Windows container scenarios
]
_have_emoji_font = any(p.exists() for p in _known_emoji_fonts)
FALLBACK = _emoji_forced_off or not _have_emoji_font

STATUS_ICONS: dict[str, str] = {
    "COMPLETED": "✅" if not FALLBACK else "[OK]",
    "IN_PROGRESS": "🔄" if not FALLBACK else "[...]",
    "FAILED": "❌" if not FALLBACK else "[X]",
    "SKIPPED": "⏭️" if not FALLBACK else "[>>]",
    "NOT_STARTED": "⏳" if not FALLBACK else "[...]",
}


def icon_for_status(status: str) -> str:
    return STATUS_ICONS.get(status.upper(), STATUS_ICONS["NOT_STARTED"])


__all__ = ["icon_for_status"]
