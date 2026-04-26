"""Cornell box with a volume-only smoke cube — synthetic test for the
homogeneous volume implementation.

The smoke material has no Surface and a `volume` block with σ_s = white * 3,
σ_a = 0, isotropic phase. The renderer should treat the cube as a pure
scatterer: light from the area emitter passes through, scatters inside, and
the cube appears as a soft-edged glow with the area light visible behind it
attenuated.

Run: python make_scene.py
"""

import json
import struct
from pathlib import Path

HERE = Path(__file__).parent


def write_blob(buf: bytearray, data: bytes) -> dict:
    off = len(buf)
    buf.extend(data)
    pad = (-len(buf)) & 15
    buf.extend(b"\x00" * pad)
    return {"offset": off, "len": len(data)}


def quad(p0, p1, p2, p3, n):
    verts = [*p0, *p1, *p2, *p3]
    normals = [*n, *n, *n, *n]
    indices = [0, 1, 2, 0, 2, 3]
    return verts, normals, indices


def cube(center, size):
    cx, cy, cz = center
    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
    V, N, I = [], [], []

    def add_face(p0, p1, p2, p3, n):
        base = len(V) // 3
        V.extend([*p0, *p1, *p2, *p3])
        N.extend([*n, *n, *n, *n])
        I.extend([base, base + 1, base + 2, base, base + 2, base + 3])

    add_face([cx + sx, cy - sy, cz - sz], [cx + sx, cy + sy, cz - sz],
             [cx + sx, cy + sy, cz + sz], [cx + sx, cy - sy, cz + sz], [1, 0, 0])
    add_face([cx - sx, cy + sy, cz - sz], [cx - sx, cy - sy, cz - sz],
             [cx - sx, cy - sy, cz + sz], [cx - sx, cy + sy, cz + sz], [-1, 0, 0])
    add_face([cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
             [cx - sx, cy + sy, cz + sz], [cx + sx, cy + sy, cz + sz], [0, 1, 0])
    add_face([cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
             [cx + sx, cy - sy, cz + sz], [cx - sx, cy - sy, cz + sz], [0, -1, 0])
    add_face([cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
             [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz], [0, 0, 1])
    add_face([cx - sx, cy + sy, cz - sz], [cx + sx, cy + sy, cz - sz],
             [cx + sx, cy - sy, cz - sz], [cx - sx, cy - sy, cz - sz], [0, 0, -1])
    return V, N, I


def pack_f32(xs):
    return struct.pack(f"<{len(xs)}f", *xs)


def pack_u32(xs):
    return struct.pack(f"<{len(xs)}I", *xs)


def identity_mat4_rowmajor():
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def main():
    buf = bytearray()
    meshes = []
    objects = []
    materials = []

    M_WHITE = 0
    M_RED = 1
    M_GREEN = 2
    M_SMOKE = 3

    materials.append({"base_color": [0.73, 0.73, 0.73], "metallic": 0.0, "roughness": 0.9})
    materials.append({"base_color": [0.65, 0.05, 0.05], "metallic": 0.0, "roughness": 0.9})
    materials.append({"base_color": [0.12, 0.45, 0.15], "metallic": 0.0, "roughness": 0.9})
    # Volume-only smoke. σ_s = color × density = [3,3,3], σ_a = 0, isotropic.
    materials.append({
        "base_color": [0.8, 0.8, 0.8],
        "metallic": 0.0,
        "roughness": 0.5,
        "volume_only": True,
        "volume": {
            "color": [3.0, 3.0, 3.0],
            "density": 1.0,
            "anisotropy": 0.0,
            "absorption_color": [0.0, 0.0, 0.0],
            "emission_color": [0.0, 0.0, 0.0],
            "emission_strength": 0.0,
        },
    })

    def add_obj(v, n, i, mat):
        mesh = {
            "vertices": write_blob(buf, pack_f32(v)),
            "normals": write_blob(buf, pack_f32(n)),
            "indices": write_blob(buf, pack_u32(i)),
        }
        mesh_id = len(meshes)
        meshes.append(mesh)
        objects.append({"mesh": mesh_id, "material": mat,
                        "transform": identity_mat4_rowmajor()})

    # Cornell box, same as cornell.
    add_obj(*quad([-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 1]), M_WHITE)
    add_obj(*quad([-1, -1, 2], [-1, 1, 2], [1, 1, 2], [1, -1, 2], [0, 0, -1]), M_WHITE)
    add_obj(*quad([-1, 1, 0], [1, 1, 0], [1, 1, 2], [-1, 1, 2], [0, -1, 0]), M_WHITE)
    add_obj(*quad([-1, -1, 0], [-1, 1, 0], [-1, 1, 2], [-1, -1, 2], [1, 0, 0]), M_RED)
    add_obj(*quad([1, 1, 0], [1, -1, 0], [1, -1, 2], [1, 1, 2], [-1, 0, 0]), M_GREEN)

    # Smoke cube right of centre, mid-air.
    v, n, i = cube([0.0, 0.0, 1.0], [0.8, 0.8, 0.8])
    add_obj(v, n, i, M_SMOKE)

    area_transform = [
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, -1, 1.99,
        0, 0, 0, 1,
    ]
    lights = [
        {"type": "area_rect", "transform": area_transform, "size": [0.6, 0.6],
         "color": [1.0, 0.96, 0.85], "power": 45.0},
    ]

    camera_mat = [
        1, 0, 0, 0.0,
        0, 0, -1, -3.0,
        0, 1, 0, 1.0,
        0, 0, 0, 1,
    ]

    scene = {
        "version": 1,
        "binary": "scene.bin",
        "render": {"width": 600, "height": 600, "spp": 256, "max_depth": 12},
        "camera": {"transform": camera_mat, "fov_y_rad": 0.6911112},
        "meshes": meshes,
        "materials": materials,
        "textures": [],
        "objects": objects,
        "lights": lights,
        "world": {"type": "constant", "color": [0.0, 0.0, 0.0], "strength": 0.0},
    }

    (HERE / "scene.bin").write_bytes(bytes(buf))
    (HERE / "scene.json").write_text(json.dumps(scene, indent=2))
    print(f"Wrote scene.json ({(HERE/'scene.json').stat().st_size} bytes)")
    print(f"Wrote scene.bin ({len(buf)} bytes)")


if __name__ == "__main__":
    main()
