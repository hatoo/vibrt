"""Compile the full device kernel with NVRTC on the CPU, without loading CUDA/OptiX.

Run cargo check -p vibrt-nodes first to generate cycles_nodes.h. Outputs live
under this checkout's target/device-check, so no render/optimizer cache is used.
"""
from pathlib import Path
import ctypes as ct
import os
import argparse

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimal", action="store_true", help="Compile a scene without special closures")
    args = parser.parse_args()
    native = ROOT / "vibrt-nodes/native"
    generated = sorted((ROOT / "target").glob("*/build/vibrt-nodes-*/out/cycles_nodes.h"),
                       key=lambda p: p.stat().st_mtime)[-1]
    replacements = {
        "devicecode.h": ROOT / "vibrt/src/devicecode.h",
        "vibrt_nodes_types.h": native / "types.h",
        "vibrt_nodes_core.h": native / "core.h",
        "vibrt_shader.h": native / "shader.h",
        "vibrt_bsdf.h": native / "bsdf.h",
        "cycles_nodes.h": generated,
        "cycles_compat.h": native / "cycles_compat.h",
        "rect_light_sampling.h": ROOT / "vibrt/src/rect_light_sampling.h",
    }
    replacements.update({f"device/{p.name}": p for p in (ROOT / "vibrt/src/device").glob("*.h")})
    source = (ROOT / "vibrt/src/devicecode.cu").read_text(encoding="utf-8")
    for _ in range(5):
        for name, path in replacements.items():
            token = f'#include "{name}"'
            if token in source:
                source = source.replace(token, path.read_text(encoding="utf-8"))
    source = """#define VIBRT_SHADER_OPS 0xffffffffffffffffull
#define VIBRT_MAX_CLOSURES 8
#define VIBRT_SHADER_STACK_SIZE 256
#define VIBRT_SHADER_CONTEXTS 0xffffffffu
#define VIBRT_SHADER_CONTEXT_SIZE 32
#define VIBRT_CLOSURE_TYPES 0xffffffffu
#define VIBRT_SHADER_OUTPUT_COUNT 64
#define VIBRT_AOVS 1
#define VIBRT_EXTENDED_PHASES 1
""" + source
    if args.minimal:
        source = source.replace("#define VIBRT_CLOSURE_TYPES 0xffffffffu", "#define VIBRT_CLOSURE_TYPES 0u")
    directory = ROOT / ("target/device-check-minimal" if args.minimal else "target/device-check")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "kernel.cu").write_text(source, encoding="utf-8")
    cuda = Path(os.environ.get("CUDA_PATH", "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0"))
    candidates = list(cuda.glob("bin/nvrtc64_*.dll")) or list(Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA").glob("v*/bin/nvrtc64_*.dll"))
    library = sorted(candidates)[-1]
    dll_directory = os.add_dll_directory(str(library.parent))
    nvrtc = ct.CDLL(str(library))
    program = ct.c_void_p()
    assert nvrtc.nvrtcCreateProgram(ct.byref(program), source.encode(), b"vibrt.cu", 0, None, None) == 0
    optix = Path(os.environ.get("OPTIX_ROOT", "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0"))
    flags = [b"--std=c++17", b"--use_fast_math", f"--include-path={optix / 'include'}".encode()]
    options = (ct.c_char_p * len(flags))(*flags)
    result = nvrtc.nvrtcCompileProgram(program, len(flags), options)
    size = ct.c_size_t()
    nvrtc.nvrtcGetProgramLogSize(program, ct.byref(size))
    log = ct.create_string_buffer(size.value)
    nvrtc.nvrtcGetProgramLog(program, log)
    (directory / "compile.log").write_bytes(log.value)
    if result:
        print(log.value.decode(errors="replace"))
    else:
        nvrtc.nvrtcGetPTXSize(program, ct.byref(size))
        ptx = ct.create_string_buffer(size.value)
        nvrtc.nvrtcGetPTX(program, ptx)
        (directory / "kernel.ptx").write_bytes(ptx.value)
        variant = "minimal closures" if args.minimal else "all closure/node branches"
        print(f"NVRTC compiled {variant} ({size.value} PTX bytes); no GPU context created.")
    nvrtc.nvrtcDestroyProgram(ct.byref(program))
    dll_directory.close()
    raise SystemExit(result)


if __name__ == "__main__":
    main()
