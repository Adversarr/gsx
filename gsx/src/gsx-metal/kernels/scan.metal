#include <metal_stdlib>
using namespace metal;

// This implementation assumes dispatch with 256 threads per threadgroup.
// On Apple Silicon this corresponds to 8 SIMD groups of width 32.
constant uint kThreadsPerGroup = 256u;
constant uint kSimdWidth = 32u;
constant uint kSimdGroupsPerThreadgroup = kThreadsPerGroup / kSimdWidth;

static inline uint gsx_metal_scan_exclusive_u32(
    uint value,
    uint ltid,
    uint simd_lane,
    uint simd_group,
    threadgroup uint *simd_totals,
    threadgroup uint *simd_offsets,
    threadgroup uint *block_total)
{
    if(ltid < kSimdGroupsPerThreadgroup) {
        simd_totals[ltid] = 0u;
        simd_offsets[ltid] = 0u;
    }
    if(ltid == 0u) {
        *block_total = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        uint simd_exclusive = simd_prefix_exclusive_sum(value);

        if(simd_lane == (kSimdWidth - 1u)) {
            simd_totals[simd_group] = simd_exclusive + value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if(ltid == 0u) {
            uint running_sum = 0u;

            for(uint simd_idx = 0u; simd_idx < kSimdGroupsPerThreadgroup; ++simd_idx) {
                simd_offsets[simd_idx] = running_sum;
                running_sum += simd_totals[simd_idx];
            }
            *block_total = running_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return simd_offsets[simd_group] + simd_exclusive;
    }
}

kernel void prefix_scan_small_exclusive_u32(
    device uint *data [[buffer(0)]],
    constant uint &count [[buffer(1)]],
    uint ltid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint simd_totals[kSimdGroupsPerThreadgroup];
    threadgroup uint simd_offsets[kSimdGroupsPerThreadgroup];
    threadgroup uint block_total;
    uint value = ltid < count ? data[ltid] : 0u;
    uint scanned_value = gsx_metal_scan_exclusive_u32(
        value,
        ltid,
        simd_lane,
        simd_group,
        simd_totals,
        simd_offsets,
        &block_total);

    if(ltid < count) {
        data[ltid] = scanned_value;
    }
}

// Phase 1:
// - Exclusive scan each 256-element block in-place.
// - Emit per-block totals into block_sums[tgid].
kernel void prefix_scan_blocks(
    device uint* data [[buffer(0)]],
    device uint* block_sums [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint ltid [[thread_position_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup uint simd_totals[kSimdGroupsPerThreadgroup];
    threadgroup uint simd_offsets[kSimdGroupsPerThreadgroup];
    threadgroup uint block_total;
    const uint num_blocks = (count + (kThreadsPerGroup - 1u)) / kThreadsPerGroup;
    if(tgid >= num_blocks) {
        return;
    }

    const uint global_index = tgid * kThreadsPerGroup + ltid;
    const uint value = global_index < count ? data[global_index] : 0u;
    const uint scanned_value = gsx_metal_scan_exclusive_u32(
        value,
        ltid,
        simd_lane,
        simd_group,
        simd_totals,
        simd_offsets,
        &block_total);

    if(global_index < count) {
        data[global_index] = scanned_value;
    }

    if(ltid == 0u) {
        block_sums[tgid] = block_total;
    }
}

kernel void prefix_scan_block_sums(
    device uint *block_sums [[buffer(0)]],
    constant uint &block_count [[buffer(1)]],
    uint ltid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    threadgroup uint simd_totals[kSimdGroupsPerThreadgroup];
    threadgroup uint simd_offsets[kSimdGroupsPerThreadgroup];
    threadgroup uint block_total;
    uint value = ltid < block_count ? block_sums[ltid] : 0u;
    uint scanned_value = gsx_metal_scan_exclusive_u32(
        value,
        ltid,
        simd_lane,
        simd_group,
        simd_totals,
        simd_offsets,
        &block_total);

    if(ltid < block_count) {
        block_sums[ltid] = scanned_value;
    }
}

kernel void prefix_scan_add_block_offsets(
    device uint* data [[buffer(0)]],
    device const uint* scanned_block_sums [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) {
        return;
    }
    uint block_index = gid / kThreadsPerGroup;
    data[gid] += scanned_block_sums[block_index];
}
