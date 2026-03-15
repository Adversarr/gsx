#pragma once

#include "../../../helper_math.h"

#include <cuda_runtime.h>

namespace tinygs {

static constexpr int kImageTile = 8;
static constexpr int kImageTileLog2 = 3;
static constexpr int kImageTileMask = kImageTile - 1;

struct DensificationInfo {
    float accum_counter;
    float accum_grad_mean2d;
    float accum_absgrad_mean2d;
    float max_radii_screen;
};

static inline __host__ __device__ unsigned int get_linear_index_tiled(unsigned int row, unsigned int col, unsigned int width_in_tile)
{
    const unsigned int tile_row = row >> kImageTileLog2;
    const unsigned int tile_col = col >> kImageTileLog2;
    const unsigned int in_tile_row = row & kImageTileMask;
    const unsigned int in_tile_col = col & kImageTileMask;

    return (((tile_row * width_in_tile) + tile_col) << (2 * kImageTileLog2)) + (in_tile_row << kImageTileLog2) + in_tile_col;
}

static inline __host__ __device__ float activate_scale(float value)
{
    return expf(value);
}

static inline __host__ __device__ float activate_scale_deriv(float value)
{
    return expf(value);
}

static inline __host__ __device__ float activate_opacity(float value)
{
    if(value >= 0.0f) {
        const float z = expf(-value);
        return 1.0f / (1.0f + z);
    }
    {
        const float z = expf(value);
        return z / (1.0f + z);
    }
}

} /* namespace tinygs */
