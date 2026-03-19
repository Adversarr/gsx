
#include <metal_stdlib>
using namespace metal;

#include "simd_utils.metal"

// Radix sort configuration
// Process 8 bits per pass (256 buckets) - standard for GPU radix sort
#define RADIX_BITS 8
#define RADIX_SIZE (1 << RADIX_BITS)  // 256 buckets
#define RADIX_MASK (RADIX_SIZE - 1)   // 0xFF

// Threadgroup configuration
#define THREADGROUP_SIZE 256
#define KEYS_PER_THREAD 4
#define KEYS_PER_THREADGROUP (THREADGROUP_SIZE * KEYS_PER_THREAD)  // 1024
#define SIMD_WIDTH 32
#define SIMD_GROUP_COUNT (THREADGROUP_SIZE / SIMD_WIDTH)

static inline uint gsx_metal_sort_first_set_lane_u64(ulong mask)
{
    uint low = (uint)(mask & 0xFFFFFFFFul);

    if(low != 0u) {
        return (uint)ctz(low);
    }
    return 32u + (uint)ctz((uint)(mask >> 32));
}

static inline uint radix_scan_exclusive_u32(
    uint value,
    uint tid,
    uint simd_lane,
    uint simd_group_id,
    threadgroup uint *simd_totals,
    threadgroup uint *simd_offsets)
{
    if(tid < SIMD_GROUP_COUNT) {
        simd_totals[tid] = 0u;
        simd_offsets[tid] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        uint simd_exclusive = simd_prefix_exclusive_sum(value);

        if(simd_lane == (SIMD_WIDTH - 1u)) {
            simd_totals[simd_group_id] = simd_exclusive + value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if(tid == 0u) {
            uint running_sum = 0u;

            for(uint simd_idx = 0u; simd_idx < SIMD_GROUP_COUNT; ++simd_idx) {
                simd_offsets[simd_idx] = running_sum;
                running_sum += simd_totals[simd_idx];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return simd_offsets[simd_group_id] + simd_exclusive;
    }
}

// ===========================================================================
// Pass 1: Histogram Kernel
// ===========================================================================
kernel void radix_histogram(
    device const uint *keys [[buffer(0)]],
    device uint *histogram [[buffer(1)]],
    constant uint &array_size [[buffer(2)]],
    constant uint &shift [[buffer(3)]],
    threadgroup uint *simd_histograms [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_size [[threads_per_simdgroup]])
{
    uint block_start = tgid * KEYS_PER_THREADGROUP;

    for(uint i = tid; i < SIMD_GROUP_COUNT * RADIX_SIZE; i += tg_size) {
        simd_histograms[i] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for(uint batch = 0u; batch < KEYS_PER_THREAD; ++batch) {
        uint idx = block_start + batch * tg_size + tid;
        bool valid = idx < array_size;
        uint digit = RADIX_SIZE;

        if(valid) {
            digit = (keys[idx] >> shift) & RADIX_MASK;
        }

        if(valid) {
            ulong match_mask = 0ul;
            uint match_count = 0u;
            uint leader_lane = 0u;

            for(uint lane = 0u; lane < simd_size; ++lane) {
                uint other_digit = gsx_metal_simd_shuffle(digit, (ushort)lane);

                if(other_digit == digit) {
                    match_mask |= (1ul << lane);
                }
            }

            match_count = (uint)popcount(match_mask);
            leader_lane = gsx_metal_sort_first_set_lane_u64(match_mask);
            if(simd_lane == leader_lane) {
                simd_histograms[simd_group_id * RADIX_SIZE + digit] += match_count;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if(tid < RADIX_SIZE) {
        uint digit_total = 0u;

        for(uint simd_idx = 0u; simd_idx < SIMD_GROUP_COUNT; ++simd_idx) {
            digit_total += simd_histograms[simd_idx * RADIX_SIZE + tid];
        }
        histogram[tgid * RADIX_SIZE + tid] = digit_total;
    }
}

// ===========================================================================
// Pass 2: Build Global Prefix and Per-Block Scatter Offsets
// ===========================================================================
kernel void radix_prefix_offsets(
    device const uint *histogram [[buffer(0)]],
    device uint *global_histogram [[buffer(1)]],
    device uint *scatter_offsets [[buffer(2)]],
    constant uint &num_threadgroups [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint simd_totals[SIMD_GROUP_COUNT];
    threadgroup uint simd_offsets[SIMD_GROUP_COUNT];
    uint sum = 0u;
    uint exclusive = 0u;
    uint running_sum = 0u;

    if(tid >= RADIX_SIZE) {
        return;
    }

    for(uint tg = 0u; tg < num_threadgroups; ++tg) {
        sum += histogram[tg * RADIX_SIZE + tid];
    }
    exclusive = radix_scan_exclusive_u32(
        sum,
        tid,
        simd_lane,
        simd_group_id,
        simd_totals,
        simd_offsets);
    global_histogram[tid] = exclusive;

    running_sum = exclusive;
    for(uint tg = 0u; tg < num_threadgroups; ++tg) {
        uint histogram_idx = tg * RADIX_SIZE + tid;

        scatter_offsets[histogram_idx] = running_sum;
        running_sum += histogram[histogram_idx];
    }
}

// ===========================================================================
// Pass 3 (SIMD-Optimized): Scatter Kernel using SIMD group operations
// ===========================================================================
// This kernel uses Metal's SIMD group functions for faster rank computation.
//
// KEY OPTIMIZATION: Use a two-phase approach with per-SIMD-group histograms:
// Phase 1: Each SIMD group computes per-digit counts using SIMD ballot
// Phase 2: Compute prefix sum across SIMD groups, then scatter
//
// This reduces per-thread work from O(tid) to O(num_simd_groups * 8) = O(1)
// since num_simd_groups is typically 8 (256 threads / 32 per SIMD).
//
// The algorithm stores per-SIMD-group histograms in shared memory:
// simd_histograms[simd_group_id * RADIX_SIZE + digit] = count in that SIMD group
//
// NOTE: This version requires additional threadgroup memory for simd_histograms.
// The Rust side allocates: RADIX_SIZE (256) * num_simd_groups (8) * 4 bytes = 8KB
kernel void radix_scatter_simd(
    device const uint *keys_in [[buffer(0)]],
    device const uint *values_in [[buffer(1)]],
    device uint *keys_out [[buffer(2)]],
    device uint *values_out [[buffer(3)]],
    device const uint *scatter_offsets [[buffer(4)]],
    constant uint &array_size [[buffer(5)]],
    constant uint &shift [[buffer(6)]],
    threadgroup uint *local_offsets [[threadgroup(0)]],
    threadgroup uint *batch_counts [[threadgroup(1)]],
    threadgroup uint *simd_digit_prefixes [[threadgroup(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_size [[threads_per_simdgroup]])
{
    uint block_start = tgid * KEYS_PER_THREADGROUP;
    uint block_end = min(block_start + KEYS_PER_THREADGROUP, array_size);
    uint block_size = block_end - block_start;

    for(uint i = tid; i < RADIX_SIZE; i += tg_size) {
        local_offsets[i] = scatter_offsets[tgid * RADIX_SIZE + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for(uint batch = 0u; batch < KEYS_PER_THREAD; ++batch) {
        uint local_idx = batch * tg_size + tid;
        bool valid = (local_idx < block_size);
        uint key = 0u;
        uint value = 0u;
        uint digit = RADIX_SIZE;
        ulong match_mask = 0ul;
        uint rank_in_simd = 0u;

        for(uint i = tid; i < SIMD_GROUP_COUNT * RADIX_SIZE; i += tg_size) {
            simd_digit_prefixes[i] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if(valid) {
            uint global_idx = block_start + local_idx;

            key = keys_in[global_idx];
            value = values_in[global_idx];
            digit = (key >> shift) & RADIX_MASK;
        }

        if(valid) {
            for(uint lane = 0u; lane < simd_size; ++lane) {
                uint other_digit = gsx_metal_simd_shuffle(digit, (ushort)lane);

                if(other_digit == digit) {
                    match_mask |= (1ul << lane);
                }
            }

            if(simd_lane > 0u) {
                rank_in_simd = (uint)popcount(match_mask & ((1ul << simd_lane) - 1ul));
            }
            if(simd_lane == gsx_metal_sort_first_set_lane_u64(match_mask)) {
                simd_digit_prefixes[simd_group_id * RADIX_SIZE + digit] = (uint)popcount(match_mask);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if(tid < RADIX_SIZE) {
            uint running_sum = 0u;

            for(uint simd_idx = 0u; simd_idx < SIMD_GROUP_COUNT; ++simd_idx) {
                uint prefix_idx = simd_idx * RADIX_SIZE + tid;
                uint digit_count = simd_digit_prefixes[prefix_idx];

                simd_digit_prefixes[prefix_idx] = running_sum;
                running_sum += digit_count;
            }
            batch_counts[tid] = running_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if(valid) {
            uint out_idx = local_offsets[digit] + simd_digit_prefixes[simd_group_id * RADIX_SIZE + digit] + rank_in_simd;

            keys_out[out_idx] = key;
            values_out[out_idx] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for(uint d = tid; d < RADIX_SIZE; d += tg_size) {
            local_offsets[d] += batch_counts[d];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
