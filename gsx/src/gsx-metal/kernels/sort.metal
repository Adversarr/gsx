#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

#include "simd_utils.metal"

#define RADIX_BITS 6
#define RADIX_SIZE (1u << RADIX_BITS)
#define RADIX_MASK (RADIX_SIZE - 1u)

#define THREADGROUP_SIZE 256u
#define KEYS_PER_THREAD 4u
#define KEYS_PER_THREADGROUP (THREADGROUP_SIZE * KEYS_PER_THREAD)
#define SIMD_WIDTH 32u
#define SIMD_GROUP_COUNT (THREADGROUP_SIZE / SIMD_WIDTH)

template<typename T>
struct gsx_metal_sort_sum_op {
    inline T operator()(thread const T &a, thread const T &b) const { return a + b; }
    inline T operator()(threadgroup const T &a, thread const T &b) const { return a + b; }
    inline T operator()(threadgroup const T &a, threadgroup const T &b) const { return a + b; }
    inline T operator()(volatile threadgroup const T &a, volatile threadgroup const T &b) const { return a + b; }
};

static inline uint gsx_metal_sort_scan_exclusive_u32(
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
        uint simd_exclusive = gsx_metal_simd_prefix_exclusive_sum(value);

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

static inline uint gsx_metal_sort_block_digit_base(
    device const uint *scatter_offsets,
    device const uint *global_histogram,
    device const uint *block_sums,
    uint num_scan_blocks,
    uint block_tgid,
    uint tg_count,
    uint digit)
{
    uint base_offset = global_histogram[digit] + scatter_offsets[digit * tg_count + block_tgid];

    if(num_scan_blocks > 0u) {
        uint block_idx = block_tgid / THREADGROUP_SIZE;

        base_offset += block_sums[digit * num_scan_blocks + block_idx];
    }
    return base_offset;
}

static inline ulong gsx_metal_sort_same_digit_mask_full(uint digit)
{
    ulong same_digit_mask = gsx_metal_simd_ballot(true);

    for(uint bit = 0u; bit < RADIX_BITS; ++bit) {
        ulong bit_mask = gsx_metal_simd_ballot(((digit >> bit) & 1u) != 0u);

        same_digit_mask &= (((digit >> bit) & 1u) != 0u) ? bit_mask : (~bit_mask);
    }
    return same_digit_mask;
}

static inline ulong gsx_metal_sort_same_digit_mask_tail(uint digit, bool valid, ulong active_mask)
{
    ulong same_digit_mask = active_mask;

    for(uint bit = 0u; bit < RADIX_BITS; ++bit) {
        ulong bit_mask = gsx_metal_simd_ballot(valid && (((digit >> bit) & 1u) != 0u));

        same_digit_mask &= (((digit >> bit) & 1u) != 0u) ? bit_mask : (~bit_mask);
    }
    return same_digit_mask;
}

kernel void radix_histogram(
    device const uint * __restrict__ keys [[buffer(0)]],
    device uint *       __restrict__ histogram [[buffer(1)]],
    constant uint &     __restrict__ array_size [[buffer(2)]],
    constant uint &     __restrict__ shift [[buffer(3)]],
    threadgroup uint *  __restrict__ local_histogram [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_count [[threadgroups_per_grid]])
{
    uint block_start = tgid * KEYS_PER_THREADGROUP;
    volatile threadgroup atomic_uint *atomic_histogram = reinterpret_cast<volatile threadgroup atomic_uint *>(local_histogram);

    if(tid < RADIX_SIZE) {
        local_histogram[tid] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for(uint batch = 0u; batch < KEYS_PER_THREAD; ++batch) {
        uint idx = block_start + batch * THREADGROUP_SIZE + tid;

        if(idx < array_size) {
            uint digit = (keys[idx] >> shift) & RADIX_MASK;
            atomic_fetch_add_explicit(&atomic_histogram[digit], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if(tid < RADIX_SIZE) {
        histogram[tid * tg_count + tgid] = local_histogram[tid];
    }
}

kernel void radix_scan_scatter_offsets_blocks(
    device uint *       __restrict__ scatter_offsets [[buffer(0)]],
    device uint *       __restrict__ block_sums [[buffer(1)]],
    device uint *       __restrict__ global_histogram [[buffer(2)]],
    constant uint &num_threadgroups [[buffer(3)]],
    constant uint &num_scan_blocks [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint simd_totals[SIMD_GROUP_COUNT];
    threadgroup uint simd_offsets[SIMD_GROUP_COUNT];
    uint digit = tgid.y;
    uint block_idx = tgid.x;
    uint tg_idx = block_idx * THREADGROUP_SIZE + tid;
    uint value = 0u;
    uint segment_idx = 0u;

    if(digit >= RADIX_SIZE || block_idx >= num_scan_blocks) {
        return;
    }

    if(tg_idx < num_threadgroups) {
        segment_idx = digit * num_threadgroups + tg_idx;
        value = scatter_offsets[segment_idx];
    }
    value = gsx_metal_sort_scan_exclusive_u32(value, tid, simd_lane, simd_group_id, simd_totals, simd_offsets);

    if(tg_idx < num_threadgroups) {
        scatter_offsets[segment_idx] = value;
    }
    if(tid == 0u) {
        uint block_total = 0u;

        for(uint simd_idx = 0u; simd_idx < SIMD_GROUP_COUNT; ++simd_idx) {
            block_total += simd_totals[simd_idx];
        }
        block_sums[digit * num_scan_blocks + block_idx] = block_total;
        if((block_idx + 1u) == num_scan_blocks) {
            global_histogram[digit] = block_total;
        }
    }
}

kernel void radix_scan_block_sums(
    device uint *       __restrict__ block_sums [[buffer(0)]],
    constant uint &num_scan_blocks [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint simd_totals[SIMD_GROUP_COUNT];
    threadgroup uint simd_offsets[SIMD_GROUP_COUNT];
    uint digit = tgid;
    uint value = 0u;

    if(digit >= RADIX_SIZE) {
        return;
    }

    if(tid < num_scan_blocks) {
        value = block_sums[digit * num_scan_blocks + tid];
    }
    value = gsx_metal_sort_scan_exclusive_u32(value, tid, simd_lane, simd_group_id, simd_totals, simd_offsets);
    if(tid < num_scan_blocks) {
        block_sums[digit * num_scan_blocks + tid] = value;
    }
}

kernel void radix_prefix_offsets(
    device const uint * __restrict__ block_sums [[buffer(0)]],
    device uint *       __restrict__ global_histogram [[buffer(1)]],
    constant uint &num_scan_blocks [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint simd_totals[SIMD_GROUP_COUNT];
    threadgroup uint simd_offsets[SIMD_GROUP_COUNT];
    uint digit_total = 0u;

    if(tid >= RADIX_SIZE) {
        return;
    }

    digit_total = global_histogram[tid];
    if(num_scan_blocks > 0u) {
        digit_total += block_sums[tid * num_scan_blocks + (num_scan_blocks - 1u)];
    }
    global_histogram[tid] = gsx_metal_sort_scan_exclusive_u32(
        digit_total,
        tid,
        simd_lane,
        simd_group_id,
        simd_totals,
        simd_offsets);
}

kernel void radix_scatter_simd_full(
    device const uint * __restrict__ keys_in [[buffer(0)]],
    device const uint * __restrict__ values_in [[buffer(1)]],
    device uint *       __restrict__ keys_out [[buffer(2)]],
    device uint *       __restrict__ values_out [[buffer(3)]],
    device const uint * __restrict__ scatter_offsets [[buffer(4)]],
    device const uint * __restrict__ global_histogram [[buffer(5)]],
    device const uint * __restrict__ block_sums [[buffer(6)]],
    constant uint &                  num_scan_blocks [[buffer(7)]],
    constant uint &                  array_size [[buffer(8)]],
    constant uint &                  shift [[buffer(9)]],
    constant uint &                  threadgroup_base [[buffer(10)]],
    constant uint &                  total_threadgroups [[buffer(11)]],
    threadgroup uint *  __restrict__ local_offsets [[threadgroup(0)]],
    threadgroup ushort * __restrict__ subgroup_digit_counts [[threadgroup(1)]],
    threadgroup ushort * __restrict__ subgroup_digit_bases [[threadgroup(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint block_tgid = threadgroup_base + tgid;
    uint block_start = block_tgid * KEYS_PER_THREADGROUP;

    (void)array_size;

    if(tid < RADIX_SIZE) {
        local_offsets[tid] = gsx_metal_sort_block_digit_base(
            scatter_offsets,
            global_histogram,
            block_sums,
            num_scan_blocks,
            block_tgid,
            total_threadgroups,
            tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma unroll
    for(uint batch = 0u; batch < KEYS_PER_THREAD; ++batch) {
        uint local_idx = batch * THREADGROUP_SIZE + tid;
        uint key = keys_in[block_start + local_idx];
        uint value = values_in[block_start + local_idx];
        uint digit = (key >> shift) & RADIX_MASK;
        uint subgroup_digit_idx = simd_group_id * RADIX_SIZE;
        ulong same_digit_mask = gsx_metal_sort_same_digit_mask_full(digit);
        ulong lower_lane_mask = (1ul << simd_lane) - 1ul;
        uint rank_in_simd = popcount(same_digit_mask & lower_lane_mask);
        uint digit_count = popcount(same_digit_mask);
        uint out_idx = 0u;

        for(uint digit_idx = tid; digit_idx < (SIMD_GROUP_COUNT * RADIX_SIZE); digit_idx += THREADGROUP_SIZE) {
            subgroup_digit_counts[digit_idx] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if(rank_in_simd == 0u) {
            subgroup_digit_counts[subgroup_digit_idx + digit] = (ushort)digit_count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if(tid < RADIX_SIZE) {
            uint running = 0u;
            uint batch_base = local_offsets[tid];

            for(uint simd_idx = 0u; simd_idx < SIMD_GROUP_COUNT; ++simd_idx) {
                uint idx = simd_idx * RADIX_SIZE + tid;
                uint count = subgroup_digit_counts[idx];

                subgroup_digit_bases[idx] = (ushort)running;
                running += count;
            }
            local_offsets[RADIX_SIZE + tid] = batch_base;
            local_offsets[tid] = batch_base + running;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        out_idx = local_offsets[RADIX_SIZE + digit] + uint(subgroup_digit_bases[subgroup_digit_idx + digit]) + rank_in_simd;
        keys_out[out_idx] = key;
        values_out[out_idx] = value;
    }
}

kernel void radix_scatter_simd_tail(
    device const uint * __restrict__ keys_in [[buffer(0)]],
    device const uint * __restrict__ values_in [[buffer(1)]],
    device uint *       __restrict__ keys_out [[buffer(2)]],
    device uint *       __restrict__ values_out [[buffer(3)]],
    device const uint * __restrict__ scatter_offsets [[buffer(4)]],
    device const uint * __restrict__ global_histogram [[buffer(5)]],
    device const uint * __restrict__ block_sums [[buffer(6)]],
    constant uint &                  num_scan_blocks [[buffer(7)]],
    constant uint &                  array_size [[buffer(8)]],
    constant uint &                  shift [[buffer(9)]],
    constant uint &                  threadgroup_base [[buffer(10)]],
    constant uint &                  total_threadgroups [[buffer(11)]],
    threadgroup uint *  __restrict__ local_offsets [[threadgroup(0)]],
    threadgroup ushort * __restrict__ subgroup_digit_counts [[threadgroup(1)]],
    threadgroup ushort * __restrict__ subgroup_digit_bases [[threadgroup(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint block_tgid = threadgroup_base + tgid;
    uint block_start = block_tgid * KEYS_PER_THREADGROUP;
    uint block_end = min(block_start + KEYS_PER_THREADGROUP, array_size);
    uint block_size = block_end - block_start;

    if(tid < RADIX_SIZE) {
        local_offsets[tid] = gsx_metal_sort_block_digit_base(
            scatter_offsets,
            global_histogram,
            block_sums,
            num_scan_blocks,
            block_tgid,
            total_threadgroups,
            tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for(uint batch = 0u; batch < KEYS_PER_THREAD; ++batch) {
        uint local_idx = batch * THREADGROUP_SIZE + tid;
        bool valid = local_idx < block_size;
        uint key = 0u;
        uint value = 0u;
        uint digit = 0u;
        uint subgroup_digit_idx = simd_group_id * RADIX_SIZE;
        ulong active_mask = gsx_metal_simd_ballot(valid);
        uint rank_in_simd = 0u;
        uint digit_count = 0u;
        uint out_idx = 0u;

        if(valid) {
            ulong same_digit_mask = 0ul;
            ulong lower_lane_mask = simd_lane == 0u ? 0ul : ((1ul << simd_lane) - 1ul);

            key = keys_in[block_start + local_idx];
            value = values_in[block_start + local_idx];
            digit = (key >> shift) & RADIX_MASK;
            same_digit_mask = gsx_metal_sort_same_digit_mask_tail(digit, true, active_mask);
            rank_in_simd = popcount(same_digit_mask & lower_lane_mask);
            digit_count = popcount(same_digit_mask);
        }

        for(uint digit_idx = tid; digit_idx < (SIMD_GROUP_COUNT * RADIX_SIZE); digit_idx += THREADGROUP_SIZE) {
            subgroup_digit_counts[digit_idx] = 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if(valid && rank_in_simd == 0u) {
            subgroup_digit_counts[subgroup_digit_idx + digit] = (ushort)digit_count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if(tid < RADIX_SIZE) {
            uint running = 0u;
            uint batch_base = local_offsets[tid];

            for(uint simd_idx = 0u; simd_idx < SIMD_GROUP_COUNT; ++simd_idx) {
                uint idx = simd_idx * RADIX_SIZE + tid;
                uint count = subgroup_digit_counts[idx];

                subgroup_digit_bases[idx] = (ushort)running;
                running += count;
            }
            local_offsets[RADIX_SIZE + tid] = batch_base;
            local_offsets[tid] = batch_base + running;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if(valid) {
            out_idx = local_offsets[RADIX_SIZE + digit] + uint(subgroup_digit_bases[subgroup_digit_idx + digit]) + rank_in_simd;
            keys_out[out_idx] = key;
            values_out[out_idx] = value;
        }
    }
}
