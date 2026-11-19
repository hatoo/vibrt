"""Array/Blender adapter for vibrt-nodes; color equations live in C++/CUDA."""

import numpy as np

from .runner import find_native_module


def native():
    module = find_native_module()
    return module if module is not None and hasattr(module, "color_pixels") else None


def apply(operation, rgb, other=None, factors=None):
    module = native()
    if module is None:
        raise RuntimeError("Native node evaluator is missing; rebuild with make native-build")
    a = np.asarray(rgb, dtype=np.float32)
    b = None if other is None else np.asarray(other, dtype=np.float32)
    f = None if factors is None else np.asarray(factors, dtype=np.float32)
    if a.ndim == 0 or a.shape[-1] != 3 or (b is not None and (b.ndim == 0 or b.shape[-1] != 3)):
        raise ValueError("Node colors must have three channels")
    shape = np.broadcast_shapes(a.shape, (3,) if b is None else b.shape)
    if f is not None:
        if f.ndim == 0:
            f = f.reshape(1)
        elif f.shape == shape[:-1]:
            f = f[..., None]
        if f.shape[-1] != 1:
            raise ValueError("Mix factors must have one channel")
        shape = np.broadcast_shapes(shape, f.shape)

    def color(value):
        if value is None:
            return None
        if value.size == 3:
            return np.ascontiguousarray(value.reshape(3))
        return np.ascontiguousarray(np.broadcast_to(value, shape))

    if f is not None:
        f = np.ascontiguousarray(f.reshape(1) if f.size == 1 else np.broadcast_to(f, shape[:-1] + (1,)))
    result = module.color_pixels(operation, color(a), color(b), f)
    return result.reshape(shape)
