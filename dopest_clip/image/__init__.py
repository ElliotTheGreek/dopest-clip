"""Image subsystem: provider-routed generation/edit/compose/analyze (gen.py) and
local PIL-based image operations (ops.py).

The provider-routed half goes through dopest_clip.providers.registry ("image" capability,
default gemini, BYOK). The local half uses Pillow (a base dependency) for the common
ops and lazily imports rembg (matting extra) for remove_background and resvg-py (graphics
extra) for svg<->png — each raises a clear, actionable error if its optional dep is absent.
No silent fallbacks.
"""

from . import gen, ops  # noqa: F401
