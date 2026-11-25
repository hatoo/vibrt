//! Fused pixel operations on borrowed NumPy memory. Hold the GIL while Rayon
//! reads the input, so Python cannot mutate it during a native operation.

use numpy::ndarray::{Array, IxDyn};
use numpy::{IntoPyArray, PyArray3, PyArrayDyn, PyReadonlyArray3, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn srgb_to_linear<'py>(
    py: Python<'py>,
    pixels: PyReadonlyArrayDyn<'py, f32>,
) -> PyResult<Bound<'py, PyArrayDyn<f32>>> {
    let view = pixels.as_array();
    let values = pixels.as_slice()?;
    let convert = |x: f32| {
        if x <= 0.04045 {
            x * (1.0f32 / 12.92)
        } else {
            ((x + 0.055) * (1.0f32 / 1.055)).powf(2.4)
        }
    };
    let result: Vec<f32> = if values.len() < 65536 {
        values.iter().map(|&x| convert(x)).collect()
    } else {
        values.par_iter().map(|&x| convert(x)).collect()
    };
    Ok(Array::from_shape_vec(IxDyn(view.shape()), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray_bound(py))
}

#[pyfunction]
pub fn resample_bilinear<'py>(
    py: Python<'py>,
    pixels: PyReadonlyArray3<'py, f32>,
    height: usize,
    width: usize,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let view = pixels.as_array();
    let (src_h, src_w, channels) = view.dim();
    let count = height
        .checked_mul(width)
        .and_then(|n| n.checked_mul(channels))
        .filter(|&n| n <= isize::MAX as usize / size_of::<f32>())
        .ok_or_else(|| PyValueError::new_err("resampled image is too large"))?;
    if height == 0 || width == 0 || src_h == 0 || src_w == 0 || channels == 0 {
        return Err(PyValueError::new_err("image dimensions must be positive"));
    }
    let src = pixels.as_slice()?;
    let axis = |size: usize, source: usize| -> Vec<(usize, usize, f64)> {
        let step = if size > 1 {
            (source - 1) as f64 / (size - 1) as f64
        } else {
            0.0
        };
        (0..size)
            .map(|i| {
                // Match NumPy linspace(dtype=float32), including its endpoint.
                let x = if i + 1 == size && size > 1 {
                    (source - 1) as f32
                } else {
                    (i as f64 * step) as f32
                };
                let lo = x.floor() as usize;
                (lo, (lo + 1).min(source - 1), x as f64 - lo as f64)
            })
            .collect()
    };
    let xs = axis(width, src_w);
    let ys = axis(height, src_h);
    let mut out = vec![0.0f32; count];
    out.par_chunks_mut(width * channels)
        .enumerate()
        .for_each(|(y, row)| {
            let (y0, y1, fy) = ys[y];
            for (x, &(x0, x1, fx)) in xs.iter().enumerate() {
                for channel in 0..channels {
                    let a = src[(y0 * src_w + x0) * channels + channel];
                    let b = src[(y0 * src_w + x1) * channels + channel];
                    let c = src[(y1 * src_w + x0) * channels + channel];
                    let d = src[(y1 * src_w + x1) * channels + channel];
                    // The reference first subtracts in f32, then promotes for
                    // interpolation; preserve that order instead of fusing FMAs.
                    let top = a as f64 + (b - a) as f64 * fx;
                    let bottom = c as f64 + (d - c) as f64 * fx;
                    row[x * channels + channel] = (top + (bottom - top) * fy) as f32;
                }
            }
        });
    Ok(Array::from_shape_vec((height, width, channels), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .into_pyarray_bound(py))
}
