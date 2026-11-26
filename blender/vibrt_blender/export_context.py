"""Owned state for one scene export; no Blender imports or global caches."""

from dataclasses import dataclass, field


def _stats():
    return {
        "material_self_s": 0.0,
        "texture_self_s": 0.0,
        "pixel_read_s": 0.0,
        "bake_chain_s": 0.0,
        "texture_count": 0,
        "pixel_bytes": 0,
        "slow_textures": [],  # list of (name, w, h, seconds), only > 0.25s
        "slow_materials": [],  # list of (name, self_seconds), only > 0.10s
        "premix_cache_hits": 0,
        "premix_cache_misses": 0,
        "linear_rgb_cache_hits": 0,
        "linear_rgb_cache_misses": 0,
    }


@dataclass
class ExportContext:
    texture_pct: int | None = None
    material: str = ""
    warnings: set = field(default_factory=set)
    stats: dict = field(default_factory=_stats)
    premix_cache: dict = field(default_factory=dict)
    linear_rgb_cache: dict = field(default_factory=dict)
    mean_rgba_cache: dict = field(default_factory=dict)
    group_stack: list = field(default_factory=list)
    sky_maps: dict = field(default_factory=dict)

    def warn(self, key, message):
        key = (self.material, key)
        if key in self.warnings:
            return
        self.warnings.add(key)
        from ._log import log
        prefix = f"material {self.material!r}: " if self.material else ""
        log(f"[vibrt] warn: {prefix}{message}")
