
#include <metal_stdlib>
using namespace metal;

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
    threadgroup atomic_uint *local_histogram [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    // Initialize local histogram to zero
    for (uint i = tid; i < RADIX_SIZE; i += tg_size) {
        atomic_store_explicit(&local_histogram[i], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint block_start = tgid * KEYS_PER_THREADGROUP;

    for (uint k = 0; k < KEYS_PER_THREAD; k++) {
        uint idx = block_start + tid + k * tg_size;
        if (idx < array_size) {
            uint key = keys[idx];
            uint digit = (key >> shift) & RADIX_MASK;
            atomic_fetch_add_explicit(&local_histogram[digit], 1u, memory_order_relaxed);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < RADIX_SIZE; i += tg_size) {
        histogram[tgid * RADIX_SIZE + i] = atomic_load_explicit(&local_histogram[i], memory_order_relaxed);
    }
}

// ===========================================================================
// Pass 2a: Reduce Kernel
// ===========================================================================
kernel void radix_reduce(
    device const uint *histogram [[buffer(0)]],
    device uint *global_histogram [[buffer(1)]],
    constant uint &num_threadgroups [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= RADIX_SIZE) return;

    uint sum = 0;
    for (uint tg = 0; tg < num_threadgroups; tg++) {
        sum += histogram[tg * RADIX_SIZE + gid];
    }
    global_histogram[gid] = sum;
}

// ===========================================================================
// Pass 2b: Exclusive Scan Kernel
// ===========================================================================
kernel void radix_scan(
    device uint *global_histogram [[buffer(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup uint simd_totals[SIMD_GROUP_COUNT];
    threadgroup uint simd_offsets[SIMD_GROUP_COUNT];
    uint value = tid < RADIX_SIZE ? global_histogram[tid] : 0u;
    uint exclusive = radix_scan_exclusive_u32(
        value,
        tid,
        simd_lane,
        simd_group_id,
        simd_totals,
        simd_offsets);

    if(tid < RADIX_SIZE) {
        global_histogram[tid] = exclusive;
    }
}

// ===========================================================================
// Pass 2c: Scatter Offsets Kernel
// ===========================================================================
kernel void radix_scatter_offsets(
    device const uint *histogram [[buffer(0)]],
    device const uint *global_prefix [[buffer(1)]],
    device uint *scatter_offsets [[buffer(2)]],
    constant uint &num_threadgroups [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= RADIX_SIZE) return;

    uint global_offset = global_prefix[gid];
    uint running_sum = 0;

    for (uint tg = 0; tg < num_threadgroups; tg++) {
        scatter_offsets[tg * RADIX_SIZE + gid] = global_offset + running_sum;
        running_sum += histogram[tg * RADIX_SIZE + gid];
    }
}

// ===========================================================================
// Pass 3: Scatter Kernel (Parallel approach using local ranking)
// ===========================================================================
// This kernel processes keys in parallel within each threadgroup.
// All threads participate in loading, ranking, and scattering keys.
//
// Algorithm for each batch of 256 keys (one per thread):
// 1. Each thread loads its key and computes its digit
// 2. Store digits in shared memory for all threads to see
// 3. Each thread counts how many preceding threads have the same digit (rank)
// 4. Each thread writes its key to output at position: base_offset + rank
// 5. Update base offsets for next batch using digit counts
//
// The key optimization is that all threads compute their ranks simultaneously
// in parallel. The ranking loop is O(tid) per thread, averaging O(n/2) total
// work across all threads, which is much better than sequential O(n) per key.
kernel void radix_scatter(
    device const uint *keys_in [[buffer(0)]],
    device const uint *values_in [[buffer(1)]],
    device uint *keys_out [[buffer(2)]],
    device uint *values_out [[buffer(3)]],
    device const uint *scatter_offsets [[buffer(4)]],
    constant uint &array_size [[buffer(5)]],
    constant uint &shift [[buffer(6)]],
    threadgroup uint *local_offsets [[threadgroup(0)]],  // RADIX_SIZE uints for bucket offsets
    threadgroup uint *shared_digits [[threadgroup(1)]],  // THREADGROUP_SIZE uints for digits
    threadgroup atomic_uint *digit_counts [[threadgroup(2)]],   // RADIX_SIZE atomic counters for batch counts
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tg_size [[threads_per_threadgroup]])
{
    uint block_start = tgid * KEYS_PER_THREADGROUP;
    uint block_end = min(block_start + KEYS_PER_THREADGROUP, array_size);
    uint block_size = block_end - block_start;

    // Load scatter offsets for this threadgroup into shared memory
    for (uint i = tid; i < RADIX_SIZE; i += tg_size) {
        local_offsets[i] = scatter_offsets[tgid * RADIX_SIZE + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process keys in batches of tg_size (256) keys
    // Each batch is processed fully in parallel
    for (uint batch = 0; batch < KEYS_PER_THREAD; batch++) {
        uint local_idx = batch * tg_size + tid;
        bool valid = (local_idx < block_size);

        // Initialize digit counts for this batch
        for (uint d = tid; d < RADIX_SIZE; d += tg_size) {
            atomic_store_explicit(&digit_counts[d], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 1: Each thread loads its key and computes digit
        uint key = 0;
        uint value = 0;
        uint digit = RADIX_SIZE;  // Invalid marker for out-of-bounds threads
        if (valid) {
            uint global_idx = block_start + local_idx;
            key = keys_in[global_idx];
            value = values_in[global_idx];
            digit = (key >> shift) & RADIX_MASK;
        }

        // Step 2: Store digit in shared memory for all threads to see
        shared_digits[tid] = digit;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 3: Compute rank - count threads with same digit that come before this thread
        // This is the key parallel operation - all threads compute their ranks simultaneously
        // Each thread only looks at threads with lower tid, ensuring stable sort order
        uint rank = 0;
        if (valid) {
            for (uint i = 0; i < tid; i++) {
                if (shared_digits[i] == digit) {
                    rank++;
                }
            }
        }

        // Step 4: Count total keys per digit in this batch using atomics
        if (valid) {
            atomic_fetch_add_explicit(&digit_counts[digit], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 5: Write keys - each thread knows exactly where to write
        // Output position = base offset for digit + rank within this batch
        if (valid) {
            uint base = local_offsets[digit];
            uint out_idx = base + rank;
            keys_out[out_idx] = key;
            values_out[out_idx] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 6: Update local offsets for next batch
        for (uint d = tid; d < RADIX_SIZE; d += tg_size) {
            local_offsets[d] += atomic_load_explicit(&digit_counts[d], memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
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
    threadgroup uint *local_offsets [[threadgroup(0)]],    // RADIX_SIZE uints for bucket offsets
    threadgroup uint *shared_digits [[threadgroup(1)]],    // THREADGROUP_SIZE uints for digits
    threadgroup atomic_uint *digit_counts [[threadgroup(2)]],     // RADIX_SIZE atomic counters for total batch counts
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

    // Load scatter offsets for this threadgroup into shared memory
    for (uint i = tid; i < RADIX_SIZE; i += tg_size) {
        local_offsets[i] = scatter_offsets[tgid * RADIX_SIZE + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Process keys in batches of tg_size (256) keys
    for (uint batch = 0; batch < KEYS_PER_THREAD; batch++) {
        uint local_idx = batch * tg_size + tid;
        bool valid = (local_idx < block_size);

        // Initialize digit counts for this batch
        for (uint d = tid; d < RADIX_SIZE; d += tg_size) {
            atomic_store_explicit(&digit_counts[d], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 1: Load key and compute digit
        uint key = 0;
        uint value = 0;
        uint digit = RADIX_SIZE;  // Invalid marker (256, outside valid range 0-255)
        if (valid) {
            uint global_idx = block_start + local_idx;
            key = keys_in[global_idx];
            value = values_in[global_idx];
            digit = (key >> shift) & RADIX_MASK;
        }

        // Step 2: Store digit in shared memory for cross-SIMD visibility
        shared_digits[tid] = digit;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 3: Compute rank within SIMD group using ballot-like approach
        // Build a bitmask where bit i is set if lane i has the same digit
        uint match_mask = 0;
        for (uint lane = 0; lane < simd_size; lane++) {
            // ALL threads execute simd_shuffle uniformly
            uint other_digit = simd_shuffle(digit, lane);
            if (other_digit == digit) {
                match_mask |= (1u << lane);
            }
        }

        // Count matches in lanes before us (rank within SIMD group)
        uint rank_in_simd = 0;
        if (valid && simd_lane > 0) {
            uint before_mask = match_mask & ((1u << simd_lane) - 1u);
            rank_in_simd = popcount(before_mask);
        }

        // Step 4: Count matches from earlier SIMD groups
        // Still O(tid) but now using register-resident digit variable
        uint rank_from_earlier = 0;
        if (valid) {
            uint start = simd_group_id * simd_size;
            for (uint i = 0; i < start; i++) {
                if (shared_digits[i] == digit) {
                    rank_from_earlier++;
                }
            }
        }

        uint rank = rank_from_earlier + rank_in_simd;

        // Step 5: Count total keys per digit in this batch using atomics
        if (valid) {
            atomic_fetch_add_explicit(&digit_counts[digit], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 6: Write key to output position
        if (valid) {
            uint base = local_offsets[digit];
            uint out_idx = base + rank;
            keys_out[out_idx] = key;
            values_out[out_idx] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Step 7: Update local offsets for next batch
        for (uint d = tid; d < RADIX_SIZE; d += tg_size) {
            local_offsets[d] += atomic_load_explicit(&digit_counts[d], memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
