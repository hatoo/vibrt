use std::env;
use std::path::PathBuf;

fn find_optix_root() -> PathBuf {
    if let Ok(root) = env::var("OPTIX_ROOT") {
        return PathBuf::from(root);
    }

    // Default install locations
    #[cfg(target_os = "windows")]
    {
        let default = PathBuf::from(r"C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0");
        if default.exists() {
            return default;
        }
    }

    #[cfg(target_os = "linux")]
    {
        let default = PathBuf::from("/usr/local/NVIDIA-OptiX-SDK-9.0.0");
        if default.exists() {
            return default;
        }
    }

    panic!(
        "Could not find OptiX SDK. Set the OPTIX_ROOT environment variable \
         to the OptiX SDK install directory."
    );
}

fn find_libclang() {
    if env::var("LIBCLANG_PATH").is_ok() {
        return;
    }

    let candidates = [
        r"C:\Program Files\LLVM\bin",
        r"C:\Program Files (x86)\LLVM\bin",
    ];
    for path in &candidates {
        if PathBuf::from(path).join("libclang.dll").exists() {
            unsafe { env::set_var("LIBCLANG_PATH", path) };
            return;
        }
    }
}

fn main() {
    find_libclang();

    let optix_root = find_optix_root();
    let include_path = optix_root.join("include");

    println!("cargo:rerun-if-env-changed=OPTIX_ROOT");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:include={}", include_path.display());

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include_path.display()))
        // Generate bindings for OptiX types
        .allowlist_type("Optix.*")
        .allowlist_type("CUdeviceptr")
        .allowlist_type("CUcontext")
        .allowlist_type("CUstream")
        .allowlist_var("OPTIX_.*")
        // Derive useful traits
        .derive_debug(true)
        .derive_copy(true)
        .derive_default(true)
        .derive_eq(false)
        .derive_hash(false)
        // Use newtype enums for type safety
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        })
        .prepend_enum_name(false)
        // Layout tests
        .layout_tests(true)
        .generate()
        .expect("Failed to generate OptiX bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Failed to write bindings");
}
