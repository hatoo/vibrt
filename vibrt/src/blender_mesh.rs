//! Read Blender's evaluated mesh arrays in place, as Cycles' mesh sync does.
//!
//! Blender 5.1's RNA mesh collection elements point into the same float3/int3
//! spans consumed by Cycles. Resolve those spans through RNA rather than
//! duplicating version-dependent Mesh/CustomData C++ struct layouts.
//! References: Blender blender-v5.1-release, intern/cycles/blender/mesh.cpp and
//! source/blender/makesrna/intern/rna_mesh.cc (collection begin/lookup functions).
//! No Blender pointer escapes this call, and the GIL stays held throughout.

use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::marker::PhantomData;
use std::mem::{align_of, size_of};

/// An RNA-owned primitive span, deliberately not Send/Sync and never exposed
/// as an arbitrary integer-address API. Constructors only receive collections
/// obtained below from a real bpy.types.Mesh on a supported Blender version.
struct Span<'py, T> {
    owner: Bound<'py, PyAny>,
    ptr: *const T,
    len: usize,
    marker: PhantomData<&'py T>,
}

impl<'py, T: Copy> Span<'py, T> {
    fn new(collection: Bound<'py, PyAny>) -> PyResult<Self> {
        let len = collection.len()?;
        let pointer = |i| -> PyResult<usize> {
            collection
                .get_item(i)?
                .call_method0("as_pointer")?
                .extract()
        };
        let address = if len == 0 {
            align_of::<T>()
        } else {
            pointer(0)?
        };
        if address == 0
            || address % align_of::<T>() != 0
            || len
                .checked_mul(size_of::<T>())
                .is_none_or(|n| n > isize::MAX as usize)
        {
            return Err(PyValueError::new_err("invalid Blender mesh span"));
        }
        // Check stride at both ends before dereferencing. These collections
        // have constant-sized primitive storage in the supported RNA version.
        for i in [1, len.saturating_sub(1)] {
            if i < len && pointer(i)? != address.checked_add(i * size_of::<T>()).unwrap_or(0) {
                return Err(PyValueError::new_err(
                    "unsupported Blender mesh array layout",
                ));
            }
        }
        Ok(Self {
            owner: collection,
            ptr: address as *const T,
            len,
            marker: PhantomData,
        })
    }

    fn get(&self, index: i32) -> PyResult<T> {
        let index =
            usize::try_from(index).map_err(|_| PyValueError::new_err("negative mesh index"))?;
        if index >= self.len {
            return Err(PyValueError::new_err("mesh index out of bounds"));
        }
        // SAFETY: validated Blender 5.1 primitive collection layout; owner is
        // retained, evaluated mesh is alive, GIL held. Packing below calls no
        // Blender methods and never changes geometry or stores these pointers.
        Ok(unsafe { self.ptr.add(index).read() })
    }
}

#[pyfunction]
#[pyo3(signature = (mesh, material_slots, color_attribute = None, uv_attribute = None))]
pub fn read_mesh<'py>(
    py: Python<'py>,
    mesh: &Bound<'py, PyAny>,
    material_slots: usize,
    color_attribute: Option<&str>,
    uv_attribute: Option<&str>,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let bpy = py.import_bound("bpy")?;
    let version: (u32, u32, u32) = bpy.getattr("app")?.getattr("version")?.extract()?;
    // Fail closed on new Blender versions until their RNA storage is tested.
    // Returning None selects the public foreach_get compatibility path.
    if (version.0, version.1) != (5, 1) {
        return Ok(None);
    }
    if !mesh.is_instance(&bpy.getattr("types")?.getattr("Mesh")?)? {
        return Err(PyValueError::new_err("read_mesh requires a Blender Mesh"));
    }
    if mesh.getattr("is_editmode")?.extract::<bool>()? {
        return Ok(None);
    }

    let attributes = mesh.getattr("attributes")?;
    let colors = if let Some(name) = color_attribute {
        let attributes = mesh.getattr("color_attributes")?;
        let attr = if name.is_empty() {
            attributes.getattr("active_color")?
        } else {
            attributes.call_method1("get", (name,))?
        };
        if attr.is_none() {
            None
        } else {
            let domain: String = attr.getattr("domain")?.extract()?;
            if domain != "POINT" && domain != "CORNER" {
                return Ok(None);
            }
            let data = attr.getattr("data")?;
            let colors = match attr.getattr("data_type")?.extract::<String>()?.as_str() {
                "FLOAT_COLOR" => Span::<[f32; 4]>::new(data)?,
                "BYTE_COLOR" => {
                    // Blender's getter applies the configured scene color
                    // management as well as sRGB decoding. Only this color
                    // attribute needs a temporary; geometry remains direct.
                    let len = data.len()?;
                    let count = len
                        .checked_mul(4)
                        .ok_or_else(|| PyValueError::new_err("color attribute too large"))?;
                    let rgba = PyArray1::<f32>::zeros_bound(py, count, false);
                    data.call_method1("foreach_get", ("color", &rgba))?;
                    Span {
                        ptr: rgba.data().cast(),
                        len,
                        owner: rgba.into_any(),
                        marker: PhantomData,
                    }
                }
                _ => return Ok(None),
            };
            Some((colors, domain == "CORNER"))
        }
    } else {
        None
    };
    let uv_layers = mesh.getattr("uv_layers")?;
    let uv_layer = match uv_attribute {
        Some(name) if !name.is_empty() => uv_layers.call_method1("get", (name,))?,
        _ => uv_layers.getattr("active")?,
    };
    let uvs = if uv_layer.is_none() {
        None
    } else {
        Some(Span::<[f32; 2]>::new(uv_layer.getattr("data")?)?)
    };
    let material_indices = if material_slots > 1 {
        let attr = attributes.call_method1("get", ("material_index",))?;
        if attr.is_none() {
            None
        } else {
            if attr.getattr("data_type")?.extract::<String>()? != "INT"
                || attr.getattr("domain")?.extract::<String>()? != "FACE"
            {
                return Ok(None);
            }
            Some(Span::<i32>::new(attr.getattr("data")?)?)
        }
    } else {
        None
    };
    let vertices = Span::<[f32; 3]>::new(mesh.getattr("vertices")?)?;
    let corners = Span::<i32>::new(mesh.getattr("loops")?)?;
    let normals = Span::<[f32; 3]>::new(mesh.getattr("corner_normals")?)?;
    let triangles = Span::<[i32; 3]>::new(mesh.getattr("loop_triangles")?)?;
    let faces = Span::<i32>::new(mesh.getattr("loop_triangle_polygons")?)?;
    if triangles.len != faces.len
        || corners.len != normals.len
        || uvs.as_ref().is_some_and(|uv| uv.len != corners.len)
        || colors
            .as_ref()
            .is_some_and(|(c, corner)| c.len != if *corner { corners.len } else { vertices.len })
    {
        return Ok(None);
    }
    let count = triangles
        .len
        .checked_mul(3)
        .filter(|&n| n <= u32::MAX as usize)
        .ok_or_else(|| PyValueError::new_err("mesh has too many triangle corners"))?;
    let mut positions = Vec::with_capacity(count * 3);
    let mut split_normals = Vec::with_capacity(count * 3);
    let mut texcoords = Vec::with_capacity(if uvs.is_some() { count * 2 } else { 0 });
    let mut vertex_colors = Vec::with_capacity(if colors.is_some() { count * 3 } else { 0 });
    let mut vertex_alphas = Vec::with_capacity(if colors.is_some() { count } else { 0 });
    let mut materials = Vec::with_capacity(if material_slots > 1 { triangles.len } else { 0 });
    // All RNA lookups have finished. Read Blender memory and write each final
    // renderer array once, without foreach_get temporaries or NumPy gathers.
    for tri in 0..triangles.len {
        for corner in triangles.get(tri as i32)? {
            let vertex = corners.get(corner)?;
            positions.extend_from_slice(&vertices.get(vertex)?);
            split_normals.extend_from_slice(&normals.get(corner)?);
            if let Some(uv) = &uvs {
                texcoords.extend_from_slice(&uv.get(corner)?);
            }
            if let Some((color, per_corner)) = &colors {
                let rgba = color.get(if *per_corner { corner } else { vertex })?;
                vertex_colors.extend_from_slice(&rgba[..3]);
                vertex_alphas.push(rgba[3]);
            }
        }
        if material_slots > 1 {
            let index = match &material_indices {
                Some(m) => m.get(faces.get(tri as i32)?)?,
                None => 0,
            };
            materials.push(index as u32);
        }
    }
    // NumPy takes ownership of these Vecs; this does not copy their payloads.
    let result = PyDict::new_bound(py);
    result.set_item("vertices", positions.into_pyarray_bound(py))?;
    result.set_item("normals", split_normals.into_pyarray_bound(py))?;
    result.set_item(
        "indices",
        (0..count as u32).collect::<Vec<_>>().into_pyarray_bound(py),
    )?;
    if uvs.is_some() {
        result.set_item("uvs", texcoords.into_pyarray_bound(py))?;
    }
    if colors.is_some() {
        result.set_item("vertex_colors", vertex_colors.into_pyarray_bound(py))?;
        result.set_item("vertex_alphas", vertex_alphas.into_pyarray_bound(py))?;
    }
    if material_slots > 1 {
        result.set_item("material_indices", materials.into_pyarray_bound(py))?;
    }
    // Keep owners live until after the final read (also documents the lifetime
    // dependency for future refactors of the packing loop).
    let _ = (&vertices.owner, &triangles.owner);
    Ok(Some(result))
}
