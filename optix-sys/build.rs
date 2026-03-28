use std::env;
use std::path::PathBuf;
use std::process::Command;

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

    // Compile example CUDA device code to PTX if nvcc is available
    compile_ptx(&include_path, &out_path);
}

fn find_cl_exe() -> Option<PathBuf> {
    // Search for cl.exe in Visual Studio installations
    let vs_root = PathBuf::from(r"C:\Program Files\Microsoft Visual Studio");
    if !vs_root.exists() {
        return None;
    }
    for year_entry in std::fs::read_dir(&vs_root).ok()? {
        let year = year_entry.ok()?.path();
        // Look through editions: Community, Professional, Enterprise, BuildTools
        for edition_entry in std::fs::read_dir(&year).into_iter().flatten().flatten() {
            let msvc_dir = edition_entry.path().join(r"VC\Tools\MSVC");
            if !msvc_dir.exists() {
                continue;
            }
            for version_entry in std::fs::read_dir(&msvc_dir).into_iter().flatten().flatten() {
                let cl = version_entry
                    .path()
                    .join(r"bin\Hostx64\x64\cl.exe");
                if cl.exists() {
                    return cl.parent().map(|p| p.to_path_buf());
                }
            }
        }
    }
    None
}

fn compile_ptx(optix_include: &PathBuf, out_dir: &PathBuf) {
    let cu_file = PathBuf::from("examples/devicecode.cu");
    if !cu_file.exists() {
        return;
    }
    println!("cargo:rerun-if-changed=examples/devicecode.cu");
    println!("cargo:rerun-if-changed=examples/devicecode.h");

    let ptx_path = out_dir.join("devicecode.ptx");

    // Find nvcc
    let nvcc = which_nvcc().unwrap_or_else(|| {
        eprintln!("warning: nvcc not found, skipping PTX compilation");
        eprintln!("         Install the CUDA Toolkit to build the example");
        return PathBuf::from("nvcc");
    });

    let mut cmd = Command::new(&nvcc);
    cmd.arg("-ptx")
        .arg(&cu_file)
        .arg("-o")
        .arg(&ptx_path)
        .arg(format!("-I{}", optix_include.display()))
        .arg("-Iexamples")
        .arg("--use_fast_math")
        .arg("-arch=compute_75")
        .arg("-Wno-deprecated-gpu-targets");

    // On Windows, nvcc needs cl.exe in PATH
    #[cfg(target_os = "windows")]
    if let Some(cl_dir) = find_cl_exe() {
        let path = env::var("PATH").unwrap_or_default();
        cmd.env("PATH", format!("{};{}", cl_dir.display(), path));
    }

    let output = cmd.output();
    match output {
        Ok(result) if result.status.success() => {
            eprintln!("Compiled PTX: {}", ptx_path.display());
        }
        Ok(result) => {
            eprintln!(
                "warning: nvcc failed:\n{}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
        Err(e) => {
            eprintln!("warning: failed to run nvcc: {}", e);
        }
    }
}

fn which_nvcc() -> Option<PathBuf> {
    // Check PATH first
    if Command::new("nvcc").arg("--version").output().is_ok() {
        return Some(PathBuf::from("nvcc"));
    }

    // Check default CUDA toolkit locations
    #[cfg(target_os = "windows")]
    {
        let cuda_path = env::var("CUDA_PATH").ok().map(PathBuf::from);
        if let Some(ref p) = cuda_path {
            let nvcc = p.join("bin").join("nvcc.exe");
            if nvcc.exists() {
                return Some(nvcc);
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        let nvcc = PathBuf::from("/usr/local/cuda/bin/nvcc");
        if nvcc.exists() {
            return Some(nvcc);
        }
    }

    None
}
