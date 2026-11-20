"""Tessellate Blender HAIR particle systems into triangle ribbons.

Cycles renders hair as curve primitives. vibrt currently has only triangle
support, so each strand is converted to a ribbon: a chain of quads, two
triangles each, with width tapering from `root_radius` to `tip_radius`. We
emit strand tangents and Hair Info attributes for the legacy and Chiang hair
closures. Strands with matching key counts are tessellated in bounded NumPy
batches; only the final geometry arrays survive export.

Limitations vs Cycles:
- Children: the exporter reads parent strands via
  `ParticleSystem.particles[i].hair_keys` and approximates "Simple" children.
  It does not reproduce Cycles' evaluated child paths: each parent is
  duplicated `rendered_child_count` times with a random in-plane offset
  within `child_radius` at the root. No clumping or curl.
- Ribbon plane: a per-strand random axis is parallel-transported from the
  root tangent. The ribbon does not rotate toward the camera at render time.
- Modern Curves objects are not exported by this particle-system path.
"""

from __future__ import annotations

import math
import random

import numpy as np

from ._log import log as _emit


def _initial_perp(axis: np.ndarray, seed: int) -> np.ndarray:
    """Random unit vector perpendicular to `axis` (per-strand reproducible)."""
    rng = random.Random(seed)
    for _ in range(8):
        v = np.array(
            (rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)),
            dtype=np.float32,
        )
        cross = np.cross(axis, v)
        n2 = float(np.dot(cross, cross))
        if n2 > 1e-6:
            return (cross / math.sqrt(n2)).astype(np.float32)
    fallback = (
        np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(axis[2]) < 0.99 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    cross = np.cross(axis, fallback)
    return (cross / max(np.linalg.norm(cross), 1e-12)).astype(np.float32)


def _strands_to_ribbons(points, root_w, tip_w, seeds):
    """Tessellate (B, N, 3) points with six corners per segment, strand first.

    Transport frames along each strand; batch independent strands and segments
    to avoid thousands of small NumPy calls. Keep the scalar reduction order,
    random seeds, winding and UVs (U across width, V from root to tip).
    """
    count, n, _ = points.shape
    axes = np.empty_like(points)
    axes[:, 0] = points[:, 1] - points[:, 0]
    axes[:, -1] = points[:, -1] - points[:, -2]
    axes[:, 1:-1] = (points[:, 2:] - points[:, :-2]) * 0.5
    norms = np.linalg.norm(axes, axis=2, keepdims=True)
    axes /= np.where(norms < 1e-12, 1.0, norms)
    perps = np.empty_like(axes)
    perps[:, 0] = [_initial_perp(a, int(seed)) for a, seed in zip(axes[:, 0], seeds)]
    up = perps[:, 0]
    for i in range(1, n):
        a = axes[:, i]
        # Batched dot preserves the scalar np.dot reduction order.
        projection = (up[:, None, :] @ a[:, :, None]).reshape(-1, 1)
        candidate = up - a * projection
        l2 = (candidate[:, None, :] @ candidate[:, :, None]).reshape(-1)
        fallback = l2 < 1e-6
        length = np.sqrt(np.where(fallback, 1.0, l2).astype(np.float64))
        candidate /= length.astype(np.float32)[:, None]
        for j in np.flatnonzero(fallback):
            candidate[j] = _initial_perp(a[j], int(seeds[j]) + i * 1009)
        perps[:, i] = candidate
        up = candidate
    t_param = np.linspace(0.0, 1.0, n, dtype=np.float32)
    half = (root_w[:, None] * (1.0 - t_param) + tip_w[:, None] * t_param) * 0.5
    half_v = perps * half[:, :, None]
    left = points + half_v
    right = points - half_v
    segments = np.arange(n - 1)
    keys = np.stack(
        (segments, segments + 1, segments, segments, segments + 1, segments + 1),
        axis=1,
    ).ravel()
    sides = np.tile([False, False, True, True, False, True], n - 1)
    positions = np.where(
        sides[None, :, None], right[:, keys], left[:, keys]
    ).reshape(-1, 3)
    tangents = axes[:, keys].reshape(-1, 3)
    seg_norm = np.cross(axes[:, :-1] + axes[:, 1:], perps[:, :-1] + perps[:, 1:])
    length = np.sqrt((seg_norm[..., None, :] @ seg_norm[..., :, None])[..., 0, 0])
    regular = length > 1e-12
    seg_norm /= np.where(regular, length, 1.0)[..., None]
    seg_norm[~regular] = [0.0, 0.0, 1.0]
    normals = np.repeat(seg_norm, 6, axis=1).reshape(-1, 3)
    uv = np.stack((sides.astype(np.float32), t_param[keys]), axis=1)
    uvs = np.tile(uv, (count, 1))
    return positions, normals, tangents, uvs


def _strand_to_ribbon(points, root_w, tip_w, seed):
    """Single-strand entry point, with the same layout as batched export."""
    if len(points) < 2:
        return None
    return _strands_to_ribbons(
        points[None, ...], np.array([root_w], np.float32),
        np.array([tip_w], np.float32), [seed],
    )


def _strand_batches(strands, seed_base):
    """Bound temporary geometry and keep consecutive, equal-length strands together."""
    batch = []
    key_count = 0
    for strand_i, (points, root_w, tip_w) in enumerate(strands):
        if len(points) < 2:
            continue
        if batch and (
            len(points) != key_count or len(batch) >= 256
            or (len(batch) + 1) * key_count > 65536
        ):
            yield batch
            batch = []
        key_count = len(points)
        seed = (seed_base ^ (strand_i * 49157)) & 0xFFFFFFFF
        batch.append((points, root_w, tip_w, seed))
    if batch:
        yield batch


def _hair_info(points, roots, tips, seeds, uv):
    """Arc length, width and deterministic random value in ribbon corner order."""
    lengths = np.linalg.norm(np.diff(points, axis=1), axis=2)
    total = lengths.sum(axis=1)
    intercepts = np.concatenate(
        (np.zeros((len(points), 1)), np.cumsum(lengths, axis=1)), axis=1
    ) / np.maximum(total.astype(np.float64), 1e-20)[:, None]
    segments = np.arange(points.shape[1] - 1)
    keys = np.stack(
        (segments, segments + 1, segments, segments, segments + 1, segments + 1),
        axis=1,
    ).ravel()
    data = np.empty((len(points), len(keys), 4), np.float32)
    data[:, :, 0] = intercepts[:, keys]
    data[:, :, 1] = total[:, None]
    v = uv.reshape(len(points), -1, 2)[:, :, 1]
    data[:, :, 2] = roots[:, None] * (1 - v) + tips[:, None] * v
    data[:, :, 3] = np.array(
        [(seed & 0xFFFFFF) / 16777216.0 for seed in seeds], np.float32
    )[:, None]
    return data.reshape(-1, 4)


def _strand_widths(settings) -> tuple[float, float]:
    """Resolve `(root_w, tip_w)` in metres from particle settings.

    Cycles renders curve primitives so a 1-pixel-wide strand is fine; we
    rasterise to triangle ribbons, and a sub-pixel-wide ribbon mostly
    disappears under sub-pixel jitter. Inflate by a fixed factor so the
    visual density at typical preview resolutions matches what an artist
    sees in Cycles. Without this, junk_shop's `Churros` (root_radius=1,
    radius_scale=0.006 → 6mm) end up close to the per-pixel size of a
    1000-px-wide preview and the tip-tapered end is invisible.
    """
    rs = float(getattr(settings, "radius_scale", 0.01) or 0.01)
    root = float(getattr(settings, "root_radius", 1.0) or 0.0) * rs
    tip = float(getattr(settings, "tip_radius", 0.0) or 0.0) * rs
    if root <= 0.0 and tip <= 0.0:
        # Both unset: fall back to 1mm so the ribbon is at least visible. Not
        # a silent default — the caller logs the warning so the artist can
        # spot a particle system that was authored without a real width.
        root = 0.001
        tip = 0.001
    # Floor the tip to a fraction of the root so a fully-tapered ribbon
    # doesn't shrink below the rasteriser's resolution at the very tip,
    # which produces a comb-tooth alpha pattern after denoising.
    tip = max(tip, root * 0.25)
    return root, tip


def _iter_strands(psys, scene_eval, obj_eval, log_tag: str):
    """Yield (object-space points (N×3), root_w, tip_w) per strand.

    Includes parents and an approximation of children. See module docstring
    for why children are synthesised manually.
    """
    parents: list[np.ndarray] = []
    for p in psys.particles:
        keys = p.hair_keys
        if len(keys) < 2:
            continue
        # `ParticleHairKey.co` is object-local (despite the docstring's
        # "world location" claim — see test scene's Face: its co range
        # matches the Face mesh's object-local bbox, *not* world). `co_local`
        # is a different per-particle frame that bakes in subsurf offsets.
        # The caller pairs this with the host's matrix_world.
        co = np.empty(len(keys) * 3, dtype=np.float32)
        keys.foreach_get("co", co)
        parents.append(co.reshape(-1, 3))
    if not parents:
        _emit(
            f"[vibrt] warn: hair {log_tag}: 0 parent strands "
            f"(particle_systems may need 'Hair Dynamics' baked) — skipped"
        )
        return

    settings = psys.settings
    root_w, tip_w = _strand_widths(settings)

    # Both `rendered_child_count` and `child_nbr` are "children **per parent**"
    # (Blender Python doc: ParticleSettings.rendered_child_count) — *not* a
    # global total. The previous code divided by len(parents), which produced
    # 8× too few children on junk_shop's Churros (8 parents × 200/parent →
    # was emitting 200 total instead of 1600).
    n_per_parent = int(getattr(settings, "rendered_child_count", 0) or 0)
    if n_per_parent == 0:
        n_per_parent = int(getattr(settings, "child_nbr", 0) or 0)

    child_radius = float(getattr(settings, "child_radius", 0.0) or 0.0)
    if child_radius <= 0.0:
        # Match Cycles' "Simple" children: scatter within a disk a few times
        # the strand width if the artist didn't pick a value.
        child_radius = max(root_w * 4.0, 0.005)

    for parent in parents:
        yield parent, root_w, tip_w

    if n_per_parent <= 0:
        _emit(
            f"[vibrt] hair {log_tag}: no rendered children "
            f"({len(parents)} parent strand(s) only)"
        )
        return

    rng = random.Random(hash((log_tag, n_per_parent, child_radius)) & 0xFFFFFFFF)
    total_children = n_per_parent * len(parents)
    # `child_length` is a multiplier on each child strand's length
    # relative to its parent (Blender Python doc: ParticleSettings.
    # child_length). Lone_monk's grass strands are 2.55 m long with
    # child_length = 0.79; without this scaling our children stay
    # 2.5 m tall and the courtyard fills with vertical green stalks
    # the size of a person. `child_length_threshold` randomises the
    # range below `child_length` (0 = uniform), which we use as
    # a per-child low-end lerp seed.
    child_length = float(getattr(settings, "child_length", 1.0) or 0.0)
    child_length = max(0.0, min(1.0, child_length))
    child_length_thresh = float(
        getattr(settings, "child_length_threshold", 0.0) or 0.0
    )
    child_length_thresh = max(0.0, min(1.0, child_length_thresh))
    _emit(
        f"[vibrt] hair {log_tag}: {len(parents)} parent + {total_children} "
        f"children ({n_per_parent}/parent) @ {root_w*1000:.1f}-{tip_w*1000:.1f}mm "
        f"child_length={child_length:.2f}"
    )
    for parent in parents:
        # Root tangent → in-plane basis (u, v). Children are "Simple" Cycles
        # children: the entire strand is translated by a random offset
        # sampled uniformly inside the disk. No clumping / curl.
        seg = parent[1] - parent[0]
        seg_n = float(np.linalg.norm(seg))
        a = (seg / seg_n).astype(np.float32) if seg_n > 1e-12 else np.array(
            [0.0, 0.0, 1.0], dtype=np.float32
        )
        ref = (
            np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if abs(a[2]) < 0.99 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        u = np.cross(a, ref)
        u = u / max(np.linalg.norm(u), 1e-12)
        v = np.cross(a, u)
        v = v / max(np.linalg.norm(v), 1e-12)
        u = u.astype(np.float32)
        v = v.astype(np.float32)
        root = parent[0]
        for _ in range(n_per_parent):
            theta = rng.uniform(0.0, 2.0 * math.pi)
            r = child_radius * math.sqrt(rng.random())
            offset = u * (r * math.cos(theta)) + v * (r * math.sin(theta))
            # Per-child length lerp: jitter inside [child_length *
            # (1 - threshold), child_length], tip-anchored on the root.
            jitter = (
                rng.uniform(1.0 - child_length_thresh, 1.0)
                if child_length_thresh > 0.0 else 1.0
            )
            scale = child_length * jitter
            if scale < 1.0:
                # Linearly interpolate each key towards the root so the
                # strand keeps its curve shape but gets shorter.
                shrunk = root + (parent - root) * scale
            else:
                shrunk = parent
            yield shrunk + offset[None, :], root_w, tip_w


def _pick_hair_material(obj_eval) -> tuple[int, object | None]:
    """Return (slot_index, material) for the host mesh's hair material.

    Priority: slot whose material has a ShaderNodeBsdfHair* node →
    slot whose material name contains "hair" → first non-empty slot.
    Falls back to (-1, None) when nothing matches.
    """
    slots = list(obj_eval.material_slots)
    hair_kinds = ("ShaderNodeBsdfHair", "ShaderNodeBsdfHairPrincipled")
    for i, slot in enumerate(slots):
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue
        for node in mat.node_tree.nodes:
            if node.bl_idname in hair_kinds:
                return i, mat
    for i, slot in enumerate(slots):
        mat = slot.material
        if mat is not None and "hair" in mat.name.lower():
            return i, mat
    for i, slot in enumerate(slots):
        if slot.material is not None:
            return i, slot.material
    return -1, None


def export_hair(
    obj,
    obj_eval,
    scene_eval,
    writer,
    resolve_material,
    log_tag: str,
    attribute_requests=None,
):
    """Build a hair mesh (positions/normals/tangents/indices) for one object.

    Parameters
    ----------
    obj, obj_eval
        Source and evaluated Blender objects. Hair particle systems live on
        `obj_eval.particle_systems`.
    scene_eval
        Evaluated scene — needed for `psys.set_resolution` if available.
    writer
        BinWriter used to stream geometry blobs.
    resolve_material
        The exporter's `resolve_material` closure — assigns / dedupes
        a material id for the hair material picked from the host object.
    log_tag
        Short label (object name) used in warnings.

    Returns ``(mesh_desc, obj_desc)`` ready to append to the scene's mesh /
    object lists, or ``None`` when the object has no hair to export.
    """
    hair_psys = [
        ps for ps in obj_eval.particle_systems if ps.settings.type == "HAIR"
    ]
    if not hair_psys:
        return None

    # Only emit ribbons for particle systems whose render_type is PATH —
    # that's the "render strands as connected hair curves" mode our
    # tessellator targets. OBJECT and COLLECTION render-types replace
    # each particle with a separate mesh instance (lone_monk's grass /
    # bushes use COLLECTION → `grass blades` / `leaves bush` source
    # meshes), and turning those into 4 m vertical ribbons fills the
    # courtyard with stretched leaf cards. Skip ribbon tessellation for
    # these systems; mesh instances are visited by the main depsgraph exporter.
    instancing_psys = [
        ps for ps in hair_psys
        if ps.settings.render_type in ("OBJECT", "COLLECTION")
    ]
    if instancing_psys:
        names = ", ".join(p.name for p in instancing_psys)
        _emit(
            f"[vibrt] hair {log_tag}: skipping {len(instancing_psys)} "
            f"object/collection-instancing system(s) [{names}] — only "
            f"PATH-rendered hair is supported (Cycles instances "
            f"`render_type=OBJECT/COLLECTION` particles into the source "
            f"meshes; we'd need a separate instancing path)"
        )
    hair_psys = [
        ps for ps in hair_psys if ps.settings.render_type == "PATH"
    ]
    if not hair_psys:
        return None

    slot_idx, mat = _pick_hair_material(obj_eval)
    mat_id = resolve_material(mat)
    requested = (attribute_requests or {}).get(mat.name if mat else "", set())
    need_info = any(key.startswith("hair:") for key in requested)

    # Some psys evaluators (`set_resolution`) re-run modifier evaluation; we
    # try render-mode in case it widens children for `psys.particles` — but
    # `psys.particles` itself only exposes parents either way.
    for psys in hair_psys:
        try:
            psys.set_resolution(scene_eval, obj_eval, "RENDER")
        except Exception:
            pass

    pos_chunks: list[np.ndarray] = []
    nrm_chunks: list[np.ndarray] = []
    tan_chunks: list[np.ndarray] = []
    uv_chunks: list[np.ndarray] = []
    info_chunks: list[np.ndarray] = []

    psys_seed_base = abs(hash(log_tag)) & 0xFFFFFFFF
    for psys_i, psys in enumerate(hair_psys):
        sub_tag = f"{log_tag}.{psys.name}"
        strands = _iter_strands(psys, scene_eval, obj_eval, sub_tag)
        seed_base = psys_seed_base ^ (psys_i * 999983)
        for batch in _strand_batches(strands, seed_base):
            points = np.stack([s[0] for s in batch])
            roots = np.array([s[1] for s in batch], np.float32)
            tips = np.array([s[2] for s in batch], np.float32)
            seeds = [s[3] for s in batch]
            p, n, t, uv = _strands_to_ribbons(points, roots, tips, seeds)
            pos_chunks.append(p)
            nrm_chunks.append(n)
            tan_chunks.append(t)
            uv_chunks.append(uv)
            if need_info:
                info_chunks.append(_hair_info(points, roots, tips, seeds, uv))

    for psys in hair_psys:
        try:
            psys.set_resolution(scene_eval, obj_eval, "PREVIEW")
        except Exception:
            pass

    if not pos_chunks:
        return None

    positions = np.concatenate(pos_chunks).astype(np.float32, copy=False)
    normals = np.concatenate(nrm_chunks).astype(np.float32, copy=False)
    tangents = np.concatenate(tan_chunks).astype(np.float32, copy=False)
    uvs = np.concatenate(uv_chunks).astype(np.float32, copy=False)
    n_corners = positions.shape[0]
    if n_corners % 3 != 0:
        # Should be impossible because every strand contributes 6*(N-1)
        # corners — emit a warning and bail rather than producing a malformed
        # mesh.
        _emit(
            f"[vibrt] warn: hair {log_tag}: corner count {n_corners} "
            f"not divisible by 3 — skipped"
        )
        return None
    indices = np.arange(n_corners, dtype=np.uint32)

    mesh_desc = {
        "vertices": writer.write_f32(positions.reshape(-1)),
        "normals": writer.write_f32(normals.reshape(-1)),
        "uvs": writer.write_f32(uvs.reshape(-1)),
        "indices": writer.write_u32(indices),
        "tangents": writer.write_f32(tangents.reshape(-1)),
    }
    attributes = {}
    if need_info:
        info = np.concatenate(info_chunks)
        for ch, name in enumerate(("Intercept", "Length", "Thickness", "Random")):
            if "hair:" + name not in requested:
                continue
            values = np.ones((len(info), 4), np.float32)
            values[:, :3] = info[:, ch, None]
            attributes["hair:" + name] = writer.write_f32(values.reshape(-1))
        if "hair:Is Strand" in requested:
            attributes["hair:Is Strand"] = writer.write_f32(np.ones(4, np.float32))
    mesh_desc["attributes"] = attributes

    # Hair vertices live in the host mesh's object-local frame (see
    # `_iter_strands`). Mirror the host's matrix_world on the instance so
    # the ribbon ends up co-located with the mesh's scalp triangles.
    mw = obj_eval.matrix_world
    obj_desc = {
        "material": mat_id,
        "transform": [mw[i][j] for i in range(4) for j in range(4)],
        "cast_shadow": getattr(obj, "visible_shadow", True),
    }
    _emit(
        f"[vibrt] hair {log_tag}: emitted {n_corners // 3} triangles "
        f"({n_corners // 6} segments) using material slot #{slot_idx}"
    )
    return mesh_desc, obj_desc
