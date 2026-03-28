/*
 * Wrapper header for bindgen.
 * Prevents OptiX from including cuda.h — we define the required CUDA types ourselves.
 */

#define OPTIX_DONT_INCLUDE_CUDA

/* Minimal CUDA driver API types needed by OptiX headers. */
/* These are ABI-compatible with the real CUDA definitions. */
typedef struct CUctx_st* CUcontext;
typedef struct CUstream_st* CUstream;
/* CUdeviceptr is already defined in optix_types.h */

#include "optix_function_table.h"
