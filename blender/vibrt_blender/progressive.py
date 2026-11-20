"""Lifetime of one Blender RenderResult across progressive image updates.

No bpy dependency, so cancellation and pass-writing can be tested in CPython.
"""


def _set_pass(rpass, flat):
    if hasattr(rpass.rect, "foreach_set"):
        rpass.rect.foreach_set(flat)
    else:
        channels = rpass.channels
        rpass.rect = [tuple(flat[i:i + channels]) for i in range(0, len(flat), channels)]


class RenderResultDisplay:
    def __init__(self, engine, width, height):
        self.engine = engine
        self.width = width
        self.height = height
        self.result = None
        self.passes = {}

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        if self.result is not None:
            # Merge even on Esc/errors so the last complete sample remains
            # visible. Always release the result, including callback failures.
            result, self.result = self.result, None
            self.passes = {}
            self.engine.end_result(result, cancel=error_type is not None, do_merge_results=True)

    def _write_color(self, pixels):
        expected = (self.height, self.width, 4)
        if pixels.shape != expected:
            raise ValueError(f"native pixels shape {pixels.shape}, expected {expected}")
        if self.result is None:
            self.result = self.engine.begin_result(0, 0, self.width, self.height)
            self.passes = {p.name: p for p in self.result.layers[0].passes}
        combined = self.passes.get("Combined")
        if combined is None:
            raise RuntimeError("Combined pass not found in render result")
        flat = pixels.ravel()
        _set_pass(combined, flat)
        noisy = self.passes.get("Noisy Image")
        if noisy is not None:
            _set_pass(noisy, flat)

    def update(self, pixels):
        self._write_color(pixels)
        self.engine.update_result(self.result)

    def finish(self, pixels, depth, mist, aovs=None):
        expected = (self.height, self.width)
        if depth.shape != expected:
            raise ValueError(f"native depth shape {depth.shape}, expected {expected}")
        self._write_color(pixels)
        z_pass = self.passes.get("Z")
        if z_pass is not None:
            _set_pass(z_pass, depth.ravel())
        mist_pass = self.passes.get("Mist")
        if mist_pass is not None:
            _set_pass(mist_pass, mist(depth).ravel())
        for name, values in (aovs or {}).items():
            rpass = self.passes.get(name)
            if rpass is not None:
                if values.shape != (self.height, self.width, 4):
                    raise ValueError(f"Invalid AOV shape for {name}: {values.shape}")
                _set_pass(
                    rpass, (values if rpass.channels == 4 else values[:, :, :1]).ravel()
                )
        self.engine.update_result(self.result)
