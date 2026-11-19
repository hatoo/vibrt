//! Scene-independent CPU sky evaluation using Blender 5.1.1's atmosphere models.
//! Only owned settings cross the FFI boundary; no Blender pointers or GPU state.

use anyhow::{bail, ensure, Result};
use rayon::prelude::*;
use serde::Deserialize;
use std::ffi::c_void;
use std::ptr::NonNull;

#[derive(Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SkySettings {
    pub model: String,
    pub sun_direction: [f32; 3],
    pub turbidity: f32,
    pub ground_albedo: f32,
    pub sun_elevation: f32,
    pub sun_rotation: f32,
    pub sun_size: f32,
    pub sun_intensity: f32,
    pub altitude: f32,
    pub air_density: f32,
    pub aerosol_density: f32,
    pub ozone_density: f32,
    pub sun_disc: bool,
    pub invert_direction: bool,
    pub xyz_to_rgb: [f32; 9],
}

impl Default for SkySettings {
    fn default() -> Self {
        Self {
            model: "HOSEK_WILKIE".into(),
            sun_direction: [0.0, 0.0, 1.0],
            turbidity: 2.2,
            ground_albedo: 0.3,
            sun_elevation: std::f32::consts::FRAC_PI_4,
            sun_rotation: 0.0,
            sun_size: 0.009512044,
            sun_intensity: 1.0,
            altitude: 0.0,
            air_density: 1.0,
            aerosol_density: 1.0,
            ozone_density: 1.0,
            sun_disc: true,
            invert_direction: false,
            // CIE XYZ D65 to scene-linear Rec.709 (Blender's default OCIO space).
            xyz_to_rgb: [
                3.24097,
                -1.5373832,
                -0.49861076,
                -0.96924365,
                1.8759675,
                0.04155506,
                0.05563008,
                -0.20397696,
                1.0569715,
            ],
        }
    }
}

#[repr(C)]
struct NativeSky {
    model: i32,
    sun_direction: [f32; 3],
    turbidity: f32,
    ground_albedo: f32,
    sun_elevation: f32,
    sun_rotation: f32,
    sun_size: f32,
    sun_intensity: f32,
    altitude: f32,
    air_density: f32,
    aerosol_density: f32,
    ozone_density: f32,
    sun_disc: i32,
    invert_direction: i32,
    xyz_to_rgb: [f32; 9],
}
const _: () = assert!(std::mem::size_of::<NativeSky>() == 100);

extern "C" {
    fn vibrt_sky_create(settings: *const NativeSky) -> *mut c_void;
    fn vibrt_sky_free(sky: *mut c_void);
    fn vibrt_sky_row(sky: *const c_void, width: i32, height: i32, y: i32, rgb: *mut f32);
}

struct Sky(NonNull<c_void>);
// The C++ object owns its coefficients/LUT; evaluate only reads them. Rows have
// disjoint output slices, and Drop runs after all Rayon work has joined.
unsafe impl Send for Sky {}
unsafe impl Sync for Sky {}
impl Sky {
    fn row(&self, width: usize, height: usize, y: usize, row: &mut [f32]) {
        // The caller bounds the dimensions and provides exactly width*3 floats.
        unsafe {
            vibrt_sky_row(
                self.0.as_ptr(),
                width as i32,
                height as i32,
                y as i32,
                row.as_mut_ptr(),
            )
        }
    }
}
impl Drop for Sky {
    fn drop(&mut self) {
        unsafe { vibrt_sky_free(self.0.as_ptr()) }
    }
}

impl SkySettings {
    fn native(&self) -> Result<NativeSky> {
        let model = match self.model.as_str() {
            "PREETHAM" => 0,
            "HOSEK_WILKIE" => 1,
            "SINGLE_SCATTERING" | "NISHITA" => 2,
            "MULTIPLE_SCATTERING" => 3,
            other => bail!("unsupported sky model: {other}"),
        };
        let finite = [
            self.turbidity,
            self.ground_albedo,
            self.sun_elevation,
            self.sun_rotation,
            self.sun_size,
            self.sun_intensity,
            self.altitude,
            self.air_density,
            self.aerosol_density,
            self.ozone_density,
        ];
        ensure!(
            finite
                .iter()
                .chain(self.sun_direction.iter())
                .chain(self.xyz_to_rgb.iter())
                .all(|v| v.is_finite()),
            "sky settings must be finite"
        );
        ensure!(
            self.sun_direction
                .iter()
                .map(|v| (*v as f64).powi(2))
                .sum::<f64>()
                > 1e-12,
            "sun direction must be nonzero"
        );
        ensure!(
            (1.0..=10.0).contains(&self.turbidity),
            "turbidity must be between 1 and 10"
        );
        ensure!(
            (0.0..=1.0).contains(&self.ground_albedo),
            "ground albedo must be between 0 and 1"
        );
        ensure!(
            (0.0..=100_000.0).contains(&self.altitude),
            "altitude must be between 0 and 100000 metres"
        );
        ensure!(
            (0.0..=std::f32::consts::PI).contains(&self.sun_size),
            "sun size must be between 0 and pi"
        );
        ensure!(
            [
                self.sun_intensity,
                self.air_density,
                self.aerosol_density,
                self.ozone_density
            ]
            .iter()
            .all(|v| (0.0..=1000.0).contains(v)),
            "sky densities/intensity must be between 0 and 1000"
        );
        let length = self
            .sun_direction
            .iter()
            .map(|v| (*v as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        // Match Cycles SkyTextureNode::simplify_settings: animated full turns
        // are allowed, and the UI azimuth is reversed before shader evaluation.
        let pi = std::f32::consts::PI;
        let tau = std::f32::consts::TAU;
        let mut elevation = self.sun_elevation % tau;
        let mut rotation = self.sun_rotation;
        if elevation.abs() >= pi {
            elevation -= tau.copysign(elevation);
        }
        if elevation.abs() >= pi / 2.0 {
            elevation = pi.copysign(elevation) - elevation;
            rotation += pi;
        }
        rotation %= tau;
        if rotation < 0.0 {
            rotation += tau;
        }
        rotation = tau - rotation;
        Ok(NativeSky {
            model,
            sun_direction: self.sun_direction.map(|v| (v as f64 / length) as f32),
            turbidity: self.turbidity,
            ground_albedo: self.ground_albedo,
            sun_elevation: elevation,
            sun_rotation: rotation,
            sun_size: self.sun_size,
            sun_intensity: self.sun_intensity,
            altitude: self.altitude,
            air_density: self.air_density,
            aerosol_density: self.aerosol_density,
            ozone_density: self.ozone_density,
            sun_disc: i32::from(self.sun_disc),
            invert_direction: i32::from(self.invert_direction),
            xyz_to_rgb: self.xyz_to_rgb,
        })
    }
}

pub fn pixels(settings: &SkySettings, width: usize, height: usize) -> Result<Vec<f32>> {
    ensure!(
        width > 0 && height > 0 && width <= 8192 && height <= 4096,
        "sky image dimensions must be within 1..8192 by 1..4096"
    );
    ensure!(
        width * height <= 16_777_216,
        "sky image exceeds 16 megapixels"
    );
    let settings = settings.native()?;
    let sky = Sky(NonNull::new(unsafe { vibrt_sky_create(&settings) })
        .ok_or_else(|| anyhow::anyhow!("could not allocate sky model"))?);
    let mut pixels = vec![0.0; width * height * 3];
    pixels
        .par_chunks_mut(width * 3)
        .enumerate()
        .for_each(|(y, row)| {
            sky.row(width, height, y, row);
        });
    ensure!(
        pixels.iter().all(|v| v.is_finite()),
        "sky model produced non-finite radiance"
    );
    Ok(pixels)
}
