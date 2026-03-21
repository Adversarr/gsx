# GSX

GSX is an experimental cross-platform C library for high-performance 3D Gaussian Splatting. Built for efficiency and flexibility, GSX provides a unified interface for training and inference across multiple hardware backends.

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Cross--platform-green)](#supported-backends)
[![Backend](https://img.shields.io/badge/backend-CPU%20%7C%20CUDA%20%7C%20Metal-blue)](#supported-backends)

<table>
  <tr>
    <th style="width: 45%;">Image Fitting Target</th>
    <th style="width: 45%;">3DGS-FIT (1M, PSNR=26.6)</th>
  </tr>
  <tr>
    <td><img src="demo/vg-starry-night.jpg" alt="Target Image" style="width: 100%;"></td>
    <td><img src="demo/vg-starry-night-1m.jpg" alt="Rendered Output" style="width: 100%;"></td>
  </tr>
</table>


## Features

### Cross-Platform Backend Support
- **CPU**: Universal fallback with SIMD optimizations
- **CUDA**: NVIDIA GPU acceleration (Tensor Core support)
- **Metal**: Apple Silicon (macOS/iOS) optimization

### Comprehensive Gaussian Splatting Pipeline
- **Core**: Arena-based memory, tensor operations, Gaussian state management
- **Rendering**: Forward/inference and backward/training modes
- **Loss Functions**: Differentiable L1, MSE, SSIM losses
- **Metrics**: PSNR, SSIM quality evaluation
- **Optimizer**: Adam with per-parameter-role learning rates
- **Adaptive Density Control**: ABSGS, MCMC, FastGS policies for pruning/duplication/growth
- **Data Loading**: Callback-based datasets with shuffle, resize, prefetch
- **Session Management**: Checkpoint I/O, scheduler, training orchestration

### Developer-Friendly Design
- Clean C API with opaque handles for stable ABI
- Backend-neutral core implementation
- Transactional mutations for structural changes
- Comprehensive error handling with machine-readable codes
- Single-threaded execution model for simplicity

## Quick Start

### Prerequisites

- CMake 3.20+
- C11 / C++17 compiler
- Optional: CUDA Toolkit (for CUDA backend), GTest, Google Benchmark

### Building

```bash
# Basic build (CPU backend only)
cmake -S . -B build
cmake --build build -j8

# CUDA backend (NVIDIA GPU)
cmake -S . -B build-cuda -DGSX_USE_CUDA=ON
cmake --build build-cuda -j8

# Metal backend (Apple Silicon)
cmake -S . -B build-metal -DGSX_USE_METAL=ON
cmake --build build-metal -j8

# With tests
cmake -S . -B build -DGSX_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure

# With benchmarks
cmake -S . -B build-bench -DGSX_BUILD_BENCHMARKS=ON
cmake --build build-bench
```

### CMake Options

| Option | Description | Default |
|--------|-------------|---------|
| `GSX_USE_CUDA` | Enable CUDA backend | OFF |
| `GSX_USE_METAL` | Enable Metal backend | OFF |
| `GSX_BUILD_TESTS` | Build test suite | OFF |
| `GSX_BUILD_BENCHMARKS` | Build benchmarks | OFF |
| `GSX_BUILD_APP_EXTRAS` | Build example applications | OFF |

## Usage Examples

### Backend Discovery

```bash
# List all available backends and devices
./build/apps/gsx-info
```

### Training from Image

```bash
# Fit Gaussians to a target image (this will reproduce the demo results)
build/apps/image-fit-demo \
  --input demo/vg-starry-night.jpg  \
  --steps 1000 \
  --gaussians 500000 \
  --backend metal \
  --lr-mean3d 0.001 \
  --init-opacity 0.1
```

### Render Point Cloud

```bash
# Render PLY file as Gaussian splats
./build/apps/render_pcd \
  --input data/garden/point_cloud.ply \
  --output renders/ \
  --backend metal
```

### Train on MipNerf360 (Experimental)

```bash
# Step 1: Convert MipNerf360/COLMAP dataset to GSX format
python scripts/convert_mipnerf360_to_data_storage.py \
  --source_path data/garden \
  --output_path data/garden_processed \
  --eval

# Step 2: Train on the preprocessed dataset
./build/apps/train-preprocessed \
  --dataset-root data/garden_processed \
  --output-dir data/out_garden \
  --backend metal \
  --enable-adc true
```

### Basic API Usage

```c
#include <gsx/gsx.h>

int main(void) {
    gsx_backend_t backend;
    gsx_arena_t arena;
    gsx_renderer_t renderer;
    gsx_render_context_t render_context;
    gsx_backend_device_t device;
    gsx_backend_buffer_type_t buffer_type;
    gsx_tensor_t out_rgb;
    gsx_backend_desc backend_desc = {0};
    gsx_arena_desc arena_desc = {0};
    gsx_renderer_desc renderer_desc = {0};
    gsx_tensor_desc tensor_desc = {0};
    gsx_camera_intrinsics intrinsics = {0};
    gsx_camera_pose pose = {0};
    gsx_render_forward_request forward_request = {0};
    gsx_gs_t gs;
    gsx_size_t gaussian_count = 100;
    gsx_index_t visible_device_count = 0;

    // ---- Backend Initialization ----
    gsx_backend_registry_init();
    gsx_count_backend_devices_by_type(GSX_BACKEND_TYPE_CPU, &visible_device_count);
    if (visible_device_count == 0) { return 1; }
    gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_CPU, 0, &device);
    backend_desc.device = device;
    gsx_backend_init(&backend, &backend_desc);
    gsx_backend_find_buffer_type(backend, GSX_BACKEND_BUFFER_TYPE_HOST, &buffer_type);

    // ---- Arena Creation ----
    arena_desc.initial_capacity_bytes = 256 * 1024 * 1024;
    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_FIXED;
    gsx_arena_init(&arena, buffer_type, &arena_desc);

    // ---- Renderer Setup ----
    renderer_desc.width = 640;
    renderer_desc.height = 480;
    renderer_desc.output_data_type = GSX_DATA_TYPE_F32;
    gsx_renderer_init(&renderer, backend, &renderer_desc);
    gsx_render_context_init(&render_context, renderer);

    // ---- Gaussian Splatting (GS) Setup ----
    // GS owns all Gaussian parameter tensors (mean3d, rotation, logscale, opacity, sh0).
    gsx_gs_desc gs_desc = {
        .buffer_type = buffer_type,
        .arena_desc = arena_desc,
        .count = gaussian_count,
        .aux_flags = GSX_GS_AUX_DEFAULT
    };
    gsx_gs_init(&gs, &gs_desc);

    // Output tensor for rendered image
    tensor_desc.rank = 3;
    tensor_desc.shape[0] = 3;
    tensor_desc.shape[1] = 480;
    tensor_desc.shape[2] = 640;
    tensor_desc.data_type = GSX_DATA_TYPE_F32;
    tensor_desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    tensor_desc.arena = arena;
    gsx_tensor_init(&out_rgb, &tensor_desc);

    // ---- Forward Render Request ----
    // Camera: pinhole model with focal lengths
    intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
    intrinsics.fx = 500.0f;
    intrinsics.fy = 500.0f;
    intrinsics.width = 640;
    intrinsics.height = 480;
    pose.rot = (gsx_quat){{0, 0, 0, 1}};  // identity rotation

    // Bind camera, Gaussian fields (borrowed from GS), and output
    forward_request.intrinsics = &intrinsics;
    forward_request.pose = &pose;
    forward_request.near_plane = 0.1f;
    forward_request.far_plane = 100.0f;
    forward_request.forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
    gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &forward_request.gs_mean3d);
    gsx_gs_get_field(gs, GSX_GS_FIELD_ROTATION, &forward_request.gs_rotation);
    gsx_gs_get_field(gs, GSX_GS_FIELD_LOGSCALE, &forward_request.gs_logscale);
    gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &forward_request.gs_opacity);
    gsx_gs_get_field(gs, GSX_GS_FIELD_SH0, &forward_request.gs_sh0);
    forward_request.out_rgb = out_rgb;

    // ---- Render ----
    gsx_renderer_render(renderer, render_context, &forward_request);

    // ---- Cleanup ----
    gsx_render_context_free(render_context);
    gsx_renderer_free(renderer);
    gsx_gs_free(gs);
    gsx_arena_free(arena);
    gsx_backend_free(backend);
    return 0;
}
```

## API Overview

GSX provides a modular C API organized into logical layers:

| Header | Purpose |
|--------|---------|
| `gsx-base.h` | Versioning, error types, math primitives, opaque handle declarations |
| `gsx-core.h` | Arenas, tensors, Gaussian state, tensor operations |
| `gsx-backend.h` | Backend registry, device enumeration, buffer management |
| `gsx-render.h` | Renderer, forward/backward passes, camera semantics |
| `gsx-data.h` | Dataset callbacks, dataloader, image resize |
| `gsx-loss.h` | Differentiable loss functions (L1, MSE, SSIM) |
| `gsx-optim.h` | Adam optimizer, parameter groups, learning rate control |
| `gsx-adc.h` | Adaptive Density Control policies |
| `gsx-runtime.h` | Scheduler, session, checkpoint I/O |
| `gsx.h` | Umbrella header (includes all public APIs) |

## Execution Model

GSX is designed for simplicity and predictability:

- **Single-threaded**: All public API calls must be made from one thread, or externally serialized
- **Ordered execution**: Backend-bound operations execute in-order on one backend-owned stream
- **No concurrent work**: Render, optimizer, ADC, and tensor transfers cannot overlap on the same backend
- **Backend-neutral**: Core implementation is backend-agnostic; backend-specific logic stays in `gsx/src/gsx-<backend>/`

## Supported Backends

| Backend | Platform | Status | Notes |
|---------|----------|--------|-------|
| CPU | Cross-platform | Stable | SIMD-optimized (ARM NEON, x86 AVX2/AVX512) |
| CUDA | NVIDIA GPU | Opt-in | Requires CUDA Toolkit, Tensor Core support |
| Metal | Apple Silicon | Opt-in | macOS/iOS, optimized via Metal framework |

## Project Structure

```
gsx/
├── gsx/
│   ├── include/gsx/      # Public C API headers
│   └── src/              # Core and backend implementations
│       ├── gsx-*.c       # Backend-agnostic implementations
│       ├── gsx-cpu/      # CPU backend
│       ├── gsx-cuda/     # CUDA backend
│       └── gsx-metal/    # Metal backend
├── apps/                 # Example applications
│   ├── gsx-info.c        # Backend discovery tool
│   ├── image_fit_demo.c  # Full training pipeline
│   ├── banana-optim.c    # Minimal optimizer example
│   └── render_pcd.c      # Point cloud rendering
├── tests/                # Test suite (CTest, GoogleTest)
├── benchmarks/           # Performance benchmarks
├── data/                 # Sample datasets
├── demo/                 # Demo assets
└── dev-docs/             # Developer documentation
```

## Testing

```bash
# Run all tests
ctest --test-dir build --output-on-failure

# Run specific test category
ctest --test-dir build -R backend_runtime --output-on-failure
```

## License

[MIT](LICENSE)

## Acknowledgments

GSX builds upon foundational work in 3D Gaussian Splatting and draws inspiration from:
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [Khronos Gaussian Splatting Specification](https://www.khronos.org/)
- [LichtField-Studio](https://github.com/MrNeRF/LichtFeld-Studio)