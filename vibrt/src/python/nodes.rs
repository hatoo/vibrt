//! Python bindings contain only shape/lifetime handling; nodes live in vibrt-nodes.
#![allow(clippy::useless_conversion)] // PyO3 0.22's generated wrappers.

use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyo3::pyfunction]
pub fn sky_pixels<'py>(
    py: pyo3::Python<'py>,
    settings: &pyo3::Bound<'py, pyo3::PyAny>,
    width: usize,
    height: usize,
) -> pyo3::PyResult<pyo3::Bound<'py, numpy::PyArray3<f32>>> {
    use numpy::IntoPyArray;
    let settings: vibrt_nodes::sky::SkySettings = super::scene::from_python(settings)?;
    // This is a short scene-sync helper, like read_mesh / pixel conversion.
    // Retain the GIL while preparing the sky so Blender's UI callbacks cannot
    // re-enter its shared Python context in the middle of graph export. The
    // internal C++/Rayon workers operate only on owned numeric data.
    let pixels = vibrt_nodes::sky::pixels(&settings, width, height)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(
        numpy::ndarray::Array::from_shape_vec((height, width, 3), pixels)
            .expect("sky dimensions validated")
            .into_pyarray_bound(py),
    )
}

#[pyfunction]
#[pyo3(signature = (operation, a, b=None, factors=None))]
pub fn color_pixels<'py>(
    py: Python<'py>,
    operation: &Bound<'py, PyAny>,
    a: PyReadonlyArrayDyn<'py, f32>,
    b: Option<PyReadonlyArrayDyn<'py, f32>>,
    factors: Option<PyReadonlyArrayDyn<'py, f32>>,
) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
    let op: vibrt_nodes::color::ColorOp = super::scene::from_python(operation)?;
    for input in std::iter::once(&a).chain(b.as_ref()) {
        if input.as_array().shape().last() != Some(&3) {
            return Err(PyValueError::new_err(
                "Color input must end in three RGB channels",
            ));
        }
    }
    let b = b.as_ref().map(|a| a.as_slice()).transpose()?;
    let factors = factors.as_ref().map(|a| a.as_slice()).transpose()?;
    // Keep the GIL: all buffers remain borrowed, and Blender's render-job
    // Python context must not be re-entered during scene synchronization.
    let pixels = vibrt_nodes::color::apply(&op, a.as_slice()?, b, factors)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(
        numpy::ndarray::Array::from_shape_vec((pixels.len() / 3, 3), pixels)
            .expect("validated RGB batch")
            .into_pyarray_bound(py),
    )
}

#[pyfunction]
pub fn fold_color_constants(graph: &Bound<'_, PyAny>) -> PyResult<Vec<Option<[f32; 3]>>> {
    let graph: vibrt_nodes::graph::ColorGraph = super::scene::from_python(graph)?;
    Ok(vibrt_nodes::fold::constants(&graph))
}

#[pyfunction]
#[pyo3(signature = (graph, contexts, textures=None))]
pub fn shader_pixels<'py>(
    py: Python<'py>,
    graph: &Bound<'py, PyAny>,
    contexts: PyReadonlyArrayDyn<'py, f32>,
    textures: Option<Vec<(PyReadonlyArrayDyn<'py, f32>, usize, usize)>>,
) -> PyResult<Bound<'py, numpy::PyArray3<f32>>> {
    let graph: vibrt_nodes::shader::ShaderGraph = super::scene::from_python(graph)?;
    if contexts.as_array().ndim() != 2 || contexts.as_array().shape()[1] != 128 {
        return Err(PyValueError::new_err(
            "Shader contexts must have shape (N, 128)",
        ));
    }
    let data = contexts.as_slice()?;
    let (contexts, tail) = data.as_chunks::<128>();
    debug_assert!(tail.is_empty());
    let textures = textures.unwrap_or_default();
    let textures = textures
        .iter()
        .map(|(a, w, h)| Ok((a.as_slice()?, *w, *h)))
        .collect::<PyResult<Vec<_>>>()?;
    let output = vibrt_nodes::shader::evaluate_with_textures(&graph, contexts, &textures)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let flat: Vec<f32> = output.into_iter().flatten().flatten().collect();
    Ok(numpy::ndarray::Array::from_shape_vec(
        (contexts.len(), vibrt_nodes::shader::OUTPUTS.len(), 4),
        flat,
    )
    .unwrap()
    .into_pyarray_bound(py))
}
