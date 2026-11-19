"""Optional native scene-sync helpers, with a reference path for comparisons."""

import os

from .runner import find_native_module


def accelerator():
    """Disable with VIBRT_PYTHON_EXPORT=1 for reproducible A/B tests."""
    if os.environ.get("VIBRT_PYTHON_EXPORT") == "1":
        return None
    native = find_native_module()
    # An older staged extension can still use the public Blender API path.
    return native if native is not None and hasattr(native, "read_mesh") else None
