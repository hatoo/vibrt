"""Export evaluated Blender hair as OptiX round linear curve segments.

Particle paths come from Blender's evaluated cache, including children, clump
and kink. Radius is half the particle diameter; arc-length taper and closed
tips follow cycles/blender/curves.cpp. Curves data uses evaluated attributes.
"""

from __future__ import annotations
import zlib
import numpy as np
from ._log import log as _emit


def _pack_curves(points, radii, offsets, seed=0, normals=None):
    """Pack segments without joining strands; retain key attributes."""
    points = np.asarray(points, np.float32).reshape(-1, 3)
    radii = np.asarray(radii, np.float32).reshape(-1)
    offsets = np.asarray(offsets, np.int64)
    if (len(radii) != len(points) or len(offsets) < 2 or offsets[0] != 0
            or offsets[-1] != len(points) or np.any(np.diff(offsets) < 0)
            or not np.isfinite(points).all() or not np.isfinite(radii).all()
            or np.any(radii < 0)):
        raise ValueError('Invalid curve keys, radii or offsets')
    starts = np.arange(max(len(points) - 1, 0), dtype=np.uint32)
    valid = np.ones(len(starts), dtype=bool)
    ends = offsets[1:-1] - 1
    valid[ends[(ends >= 0) & (ends < len(valid))]] = False
    delta = np.diff(points, axis=0)
    valid &= np.einsum('ij,ij->i', delta, delta) > 1e-20
    valid &= np.maximum(radii[:-1], radii[1:]) > 0
    starts = starts[valid]
    indices = np.column_stack((starts, starts + 1, starts))
    info = np.zeros((len(points), 4), np.float32)
    tangents = np.zeros_like(points)
    make_normals = normals is None
    normals = np.zeros_like(points) if make_normals else np.asarray(normals, np.float32).reshape(-1, 3)
    for curve, (first, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        if end - first < 2:
            continue
        differences = np.diff(points[first:end], axis=0)
        lengths = np.linalg.norm(differences, axis=1)
        total = float(lengths.sum())
        info[first:end, 0] = np.r_[0, np.cumsum(lengths)] / max(total, 1e-20)
        info[first:end, 1] = total
        info[first:end, 2] = 2 * radii[first:end]
        info[first:end, 3] = zlib.crc32(int(curve).to_bytes(8, 'little'), seed) / 4294967296.0
        axes = np.concatenate((differences[:1], differences[:-1] + differences[1:], differences[-1:]))
        axes /= np.maximum(np.linalg.norm(axes, axis=1, keepdims=True), 1e-20)
        tangents[first:end] = axes
    if make_normals:
        # Independent strands share the same recurrence step, allowing NumPy
        # to process them together without reordering keys within a strand.
        sizes = np.diff(offsets)
        for size in np.unique(sizes[sizes >= 2]):
            firsts = offsets[:-1][sizes == size]
            up = np.zeros((len(firsts), 3), np.float32)
            use_x = np.abs(tangents[firsts, 0]) < 0.9
            up[:, 0] = use_x
            up[:, 1] = ~use_x
            fallback = np.zeros(len(firsts), dtype=bool)
            for i in range(size):
                axis = tangents[firsts + i]
                dot = np.einsum('ij,ij->i', up, axis)
                up = up - axis * dot[:, None]
                squared = np.einsum('ij,ij->i', up, up)
                fallback |= squared < 1e-12
                up = up / np.maximum(np.sqrt(squared), 1e-20)[:, None]
                normals[firsts + i] = up
            # The scalar cross-product fallback promotes to float64. Replay
            # those strands to retain that dtype for all subsequent keys.
            for first in firsts[fallback]:
                axes = tangents[first:first + size]
                up = np.array([1, 0, 0] if abs(axes[0, 0]) < 0.9 else [0, 1, 0], np.float32)
                for i, axis in enumerate(axes):
                    up = up - axis * np.dot(up, axis)
                    squared = np.dot(up, up)
                    if squared < 1e-12:
                        up = np.cross(axis, [0, 0, 1] if abs(axis[2]) < 0.9 else [0, 1, 0])
                        squared = np.dot(up, up)
                    up = up / max(float(np.sqrt(squared)), 1e-20)
                    normals[first + i] = up
    uv = np.column_stack((np.full(len(points), 0.5, np.float32), info[:, 0]))
    return dict(vertices=points, indices=indices, curve_radii=radii,
                tangents=tangents, normals=normals, uvs=uv), info


def _write_curves(writer, arrays, info, requested):
    if not len(arrays['indices']):
        return None
    mesh = {name: (writer.write_u32(values.ravel()) if name == 'indices'
                   else writer.write_f32(values.ravel())) for name, values in arrays.items()}
    attributes = {}
    for ch, name in enumerate(('Intercept', 'Length', 'Thickness', 'Random')):
        key = 'hair:' + name
        if key in requested:
            values = np.ones((len(info), 4), np.float32)
            values[:, :3] = info[:, ch, None]
            attributes[key] = writer.write_f32(values.ravel())
    if 'hair:Is Strand' in requested:
        attributes['hair:Is Strand'] = writer.write_f32(np.ones(4, np.float32))
    mesh['attributes'] = attributes
    return mesh


def _object(obj, matrix, materials):
    return {'material': materials[0], 'materials': materials,
            'transform': [matrix[i][j] for i in range(4) for j in range(4)],
            'cast_shadow': getattr(obj, 'visible_shadow', True),
            'camera_visible': getattr(obj, 'visible_camera', True)}


def _particle_radii(settings, points):
    """Cycles' shaperadius profile, parameterized by object-space arc length."""
    lengths = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    time = lengths / max(float(lengths[-1]), 1e-20)
    shape = float(settings.shape)
    profile = 1 - time
    if shape != 0:
        profile = profile ** (1 + shape if shape < 0 else 1 / max(1 - shape, 1e-8))
    scale = 0.5 * float(settings.radius_scale)
    radius = scale * (profile * (settings.root_radius - settings.tip_radius) + settings.tip_radius)
    if settings.use_close_tip:
        radius[-1] = 0
    return np.asarray(radius, np.float32)


def export_hair(obj, obj_eval, scene_eval, writer, resolve_material, log_tag,
                attribute_requests=None, depsgraph=None):
    """Return one mesh/object pair per PATH particle system and material."""
    pairs = []
    for psys in obj_eval.particle_systems:
        settings = psys.settings
        if settings.type != 'HAIR' or settings.render_type != 'PATH':
            continue
        slot = max(0, min(settings.material - 1, len(obj_eval.material_slots) - 1))
        mat = obj_eval.material_slots[slot].material if obj_eval.material_slots else None
        material = resolve_material(mat)
        requested = (attribute_requests or {}).get(mat.name if mat else '', set())
        # co_hair reads the active dependency graph's cache. Blender 5 no longer
        # exposes ParticleSystem.set_resolution; never invent child paths.
        if depsgraph is None:
            import bpy
            depsgraph = bpy.context.evaluated_depsgraph_get()
        parent_count, child_count = len(psys.particles), len(psys.child_particles)
        first = parent_count if settings.child_type != 'NONE' and child_count else 0
        steps = (1 << (settings.render_step if depsgraph.mode == 'RENDER' else settings.display_step)) + 1
        if settings.kink == 'SPIRAL':
            steps += settings.kink_extra_steps
        inv = np.asarray(obj_eval.matrix_world.inverted(), np.float32)
        points, radii, offsets = [], [], [0]
        for index in range(first, parent_count + child_count):
            world = np.array([psys.co_hair(obj_eval, particle_no=index, step=step)
                              for step in range(steps)], np.float32)
            local = world @ inv[:3, :3].T + inv[:3, 3]
            points.append(local)
            radii.append(_particle_radii(settings, local))
            offsets.append(offsets[-1] + len(local))
        if not points:
            continue
        arrays, info = _pack_curves(np.concatenate(points), np.concatenate(radii), offsets,
                                   zlib.crc32((log_tag + psys.name).encode()))
        mesh = _write_curves(writer, arrays, info, requested)
        if mesh is not None:
            pairs.append((mesh, _object(obj, obj_eval.matrix_world, [material])))
            _emit(f'[vibrt] hair {log_tag}.{psys.name}: {len(offsets)-1} evaluated curves, '
                  f'{len(arrays["indices"])} native segments')
    return pairs


def export_curves(obj, obj_eval, matrix, writer, resolve_material, attribute_requests=None):
    """Export Curves data and requested point/curve domain attributes."""
    curves = obj_eval.data
    materials = [resolve_material(s.material) for s in obj_eval.material_slots] or [resolve_material(None)]
    requested = set().union(*[(attribute_requests or {}).get(s.material.name, set())
                             for s in obj_eval.material_slots if s.material])
    if not len(curves.points):
        return None
    positions = np.empty(len(curves.points) * 3, np.float32)
    curves.position_data.foreach_get('vector', positions)
    offsets = np.empty(len(curves.curves) + 1, np.int32)
    curves.curve_offset_data.foreach_get('value', offsets)
    radii = np.empty(len(curves.points), np.float32)
    curves.points.foreach_get('radius', radii)
    normals = np.empty(len(curves.points) * 3, np.float32)
    curves.normals.foreach_get('vector', normals)
    arrays, info = _pack_curves(positions, radii, offsets, zlib.crc32(obj.name.encode()), normals)
    mesh = _write_curves(writer, arrays, info, requested)
    if mesh is None:
        return None
    curve_ids = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))
    material_attr = curves.attributes.get('material_index')
    if material_attr is not None:
        ids = np.empty(len(curves.curves), np.int32)
        material_attr.data.foreach_get('value', ids)
        mesh['material_indices'] = writer.write_u32(np.clip(ids[curve_ids[arrays['indices'][:, 0]]], 0, len(materials)-1))
    for key in requested:
        prefix, name = key.split(':', 1)
        if prefix not in {'attr', 'color', 'uv'}:
            continue
        attr = curves.attributes.get(name)
        if attr is None or attr.domain not in {'POINT', 'CURVE'}:
            continue
        kind = {'FLOAT': ('value', 1), 'INT': ('value', 1), 'BOOLEAN': ('value', 1),
                'FLOAT_VECTOR': ('vector', 3), 'FLOAT_COLOR': ('color', 4),
                'BYTE_COLOR': ('color', 4), 'FLOAT2': ('vector', 2)}.get(attr.data_type)
        if kind is None:
            continue
        field, channels = kind
        dtype = np.int32 if attr.data_type == 'INT' else np.bool_ if attr.data_type == 'BOOLEAN' else np.float32
        values = np.empty((len(attr.data), channels), dtype)
        attr.data.foreach_get(field, values.ravel())
        if attr.domain == 'CURVE':
            values = values[curve_ids]
        rgba = np.ones((len(curves.points), 4), np.float32)
        rgba[:, :channels if channels > 1 else 3] = values
        mesh['attributes'][key] = writer.write_f32(rgba.ravel())
    return mesh, _object(obj, matrix, materials)
