#pragma once

#include "helper_math.h"

#include "rasterization_config.h"

#include <cub/cub.cuh>
#include <cstdint>

namespace fast_gs::rasterization {

struct mat3x3 {
    float m11;
    float m12;
    float m13;
    float m21;
    float m22;
    float m23;
    float m31;
    float m32;
    float m33;
};

struct __align__(8) mat3x3_triu {
    float m11;
    float m12;
    float m13;
    float m22;
    float m23;
    float m33;
};

template <typename T>
static void obtain(char *&blob, T *&ptr, std::size_t count, std::size_t alignment)
{
    std::size_t offset = (reinterpret_cast<std::uintptr_t>(blob) + alignment - 1) & ~(alignment - 1);

    ptr = reinterpret_cast<T *>(offset);
    blob = reinterpret_cast<char *>(ptr + count);
}

template <typename T, typename... Args>
static std::size_t required(std::size_t count, Args... args)
{
    char *size = nullptr;

    T::from_blob(size, count, args...);
    return (std::size_t)size + 128;
}

struct PerPrimitiveBuffers {
    std::size_t cub_workspace_size;
    char *cub_workspace;
    cub::DoubleBuffer<uint> depth_keys;
    cub::DoubleBuffer<uint> primitive_indices;
    uint *n_touched_tiles;
    uint *offset;
    ushort4 *screen_bounds;
    float2 *mean2d;
    float4 *conic_opacity;
    float3 *color;
    uint *n_visible_primitives;
    uint *n_instances;

    static PerPrimitiveBuffers from_blob(char *&blob, std::size_t n_primitives)
    {
        PerPrimitiveBuffers buffers;
        uint *depth_keys_current = nullptr;
        uint *depth_keys_alternate = nullptr;
        uint *primitive_indices_current = nullptr;
        uint *primitive_indices_alternate = nullptr;
        std::size_t sorting_workspace_size = 0;

        obtain(blob, depth_keys_current, n_primitives, 128);
        obtain(blob, depth_keys_alternate, n_primitives, 128);
        buffers.depth_keys = cub::DoubleBuffer<uint>(depth_keys_current, depth_keys_alternate);
        obtain(blob, primitive_indices_current, n_primitives, 128);
        obtain(blob, primitive_indices_alternate, n_primitives, 128);
        buffers.primitive_indices = cub::DoubleBuffer<uint>(primitive_indices_current, primitive_indices_alternate);
        obtain(blob, buffers.n_touched_tiles, n_primitives, 128);
        obtain(blob, buffers.offset, n_primitives, 128);
        obtain(blob, buffers.screen_bounds, n_primitives, 128);
        obtain(blob, buffers.mean2d, n_primitives, 128);
        obtain(blob, buffers.conic_opacity, n_primitives, 128);
        obtain(blob, buffers.color, n_primitives, 128);
        cub::DeviceScan::ExclusiveSum(nullptr, buffers.cub_workspace_size, buffers.offset, buffers.offset, n_primitives);
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            sorting_workspace_size,
            buffers.depth_keys,
            buffers.primitive_indices,
            n_primitives
        );
        buffers.cub_workspace_size = buffers.cub_workspace_size > sorting_workspace_size ? buffers.cub_workspace_size : sorting_workspace_size;
        obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
        obtain(blob, buffers.n_visible_primitives, 1, 128);
        obtain(blob, buffers.n_instances, 1, 128);
        return buffers;
    }
};

struct PerInstanceBuffers {
    std::size_t cub_workspace_size;
    char *cub_workspace;
    cub::DoubleBuffer<ushort> keys;
    cub::DoubleBuffer<uint> primitive_indices;

    static PerInstanceBuffers from_blob(char *&blob, std::size_t n_instances)
    {
        PerInstanceBuffers buffers;
        ushort *keys_current = nullptr;
        ushort *keys_alternate = nullptr;
        uint *primitive_indices_current = nullptr;
        uint *primitive_indices_alternate = nullptr;

        obtain(blob, keys_current, n_instances, 128);
        obtain(blob, keys_alternate, n_instances, 128);
        buffers.keys = cub::DoubleBuffer<ushort>(keys_current, keys_alternate);
        obtain(blob, primitive_indices_current, n_instances, 128);
        obtain(blob, primitive_indices_alternate, n_instances, 128);
        buffers.primitive_indices = cub::DoubleBuffer<uint>(primitive_indices_current, primitive_indices_alternate);
        cub::DeviceRadixSort::SortPairs(
            nullptr,
            buffers.cub_workspace_size,
            buffers.keys,
            buffers.primitive_indices,
            n_instances
        );
        obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
        return buffers;
    }
};

struct PerTileBuffers {
    std::size_t cub_workspace_size;
    char *cub_workspace;
    uint2 *instance_ranges;
    uint *n_buckets;
    uint *bucket_offsets;
    uint *max_n_contributions;
    uint *n_contributions;

    static PerTileBuffers from_blob(char *&blob, std::size_t n_tiles)
    {
        PerTileBuffers buffers;

        obtain(blob, buffers.instance_ranges, n_tiles, 128);
        obtain(blob, buffers.n_buckets, n_tiles, 128);
        obtain(blob, buffers.bucket_offsets, n_tiles, 128);
        obtain(blob, buffers.max_n_contributions, n_tiles, 128);
        obtain(blob, buffers.n_contributions, n_tiles * config::block_size_blend, 128);
        cub::DeviceScan::InclusiveSum(
            nullptr,
            buffers.cub_workspace_size,
            buffers.n_buckets,
            buffers.bucket_offsets,
            n_tiles
        );
        obtain(blob, buffers.cub_workspace, buffers.cub_workspace_size, 128);
        return buffers;
    }
};

struct PerBucketBuffers {
    uint *tile_index;
    float4 *color_transmittance;

    static PerBucketBuffers from_blob(char *&blob, std::size_t n_buckets)
    {
        PerBucketBuffers buffers;

        obtain(blob, buffers.tile_index, n_buckets * config::block_size_blend, config::block_size_blend);
        obtain(blob, buffers.color_transmittance, n_buckets * config::block_size_blend, config::block_size_blend);
        return buffers;
    }
};

} /* namespace fast_gs::rasterization */
