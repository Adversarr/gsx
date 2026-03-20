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
#define LOCAL_SORT_SENTINEL RADIX_SIZE
#define LOCAL_SORT_BITS_FULL RADIX_BITS
#define LOCAL_SORT_BITS_TAIL (RADIX_BITS + 1u)

#define GSX_METAL_SORT_TOKEN_LANE_SHIFT 16u
#define GSX_METAL_SORT_TOKEN_KEY_MASK ((1u << LOCAL_SORT_BITS_TAIL) - 1u)

template<typename T>
struct gsx_metal_sort_sum_op {
    inline T operator()(thread const T &a, thread const T &b) const { return a + b; }
    inline T operator()(threadgroup const T &a, thread const T &b) const { return a + b; }
    inline T operator()(threadgroup const T &a, threadgroup const T &b) const { return a + b; }
    inline T operator()(volatile threadgroup const T &a, volatile threadgroup const T &b) const { return a + b; }
};

static inline uint gsx_metal_sort_simd_prefix_inclusive_max_u32(uint value, uint simd_lane)
{
    uint temp = 0u;

    temp = gsx_metal_simd_shuffle_up(value, 1u);
    if(simd_lane >= 1u) {
        value = max(value, temp);
    }
    temp = gsx_metal_simd_shuffle_up(value, 2u);
    if(simd_lane >= 2u) {
        value = max(value, temp);
    }
    temp = gsx_metal_simd_shuffle_up(value, 4u);
    if(simd_lane >= 4u) {
        value = max(value, temp);
    }
    temp = gsx_metal_simd_shuffle_up(value, 8u);
    if(simd_lane >= 8u) {
        value = max(value, temp);
    }
    temp = gsx_metal_simd_shuffle_up(value, 16u);
    if(simd_lane >= 16u) {
        value = max(value, temp);
    }
    return value;
}

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

static inline uint gsx_metal_sort_prefix_inclusive_max_u32(
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
        uint simd_inclusive = gsx_metal_sort_simd_prefix_inclusive_max_u32(value, simd_lane);

        if(simd_lane == (SIMD_WIDTH - 1u)) {
            simd_totals[simd_group_id] = simd_inclusive;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if(tid == 0u) {
            uint running_max = 0u;

            for(uint simd_idx = 0u; simd_idx < SIMD_GROUP_COUNT; ++simd_idx) {
                simd_offsets[simd_idx] = running_max;
                running_max = max(running_max, simd_totals[simd_idx]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        return max(simd_offsets[simd_group_id], simd_inclusive);
    }
}

template <ushort LENGTH, int SCAN_TYPE, typename BinaryOp, typename T>
static inline T gsx_metal_sort_thread_scan(threadgroup T *values, BinaryOp op)
{
    for(ushort i = 1u; i < LENGTH; ++i) {
        values[i] = op(values[i], values[i - 1u]);
    }
    T result = values[LENGTH - 1u];

    if(SCAN_TYPE != 0) {
        for(ushort i = LENGTH - 1u; i > 0u; --i) {
            values[i] = values[i - 1u];
        }
        values[0] = T(0);
    }
    return result;
}

template <ushort LENGTH, typename BinaryOp, typename T>
static inline void gsx_metal_sort_thread_uniform_apply(threadgroup T *values, T uniform_value, BinaryOp op)
{
    for(ushort i = 0u; i < LENGTH; ++i) {
        values[i] = op(values[i], uniform_value);
    }
}

template <int SCAN_TYPE, typename BinaryOp, typename T>
static inline T gsx_metal_sort_simdgroup_scan(T value, ushort local_id, BinaryOp op)
{
    ushort lane_id = local_id % (ushort)SIMD_WIDTH;
    T temp = gsx_metal_simd_shuffle_up(value, 1u);

    if(lane_id >= 1u) {
        value = op(value, temp);
    }
    temp = gsx_metal_simd_shuffle_up(value, 2u);
    if(lane_id >= 2u) {
        value = op(value, temp);
    }
    temp = gsx_metal_simd_shuffle_up(value, 4u);
    if(lane_id >= 4u) {
        value = op(value, temp);
    }
    temp = gsx_metal_simd_shuffle_up(value, 8u);
    if(lane_id >= 8u) {
        value = op(value, temp);
    }
    temp = gsx_metal_simd_shuffle_up(value, 16u);
    if(lane_id >= 16u) {
        value = op(value, temp);
    }
    if(SCAN_TYPE != 0) {
        temp = gsx_metal_simd_shuffle_up(value, 1u);
        value = (lane_id == 0u) ? T(0) : temp;
    }
    return value;
}

template <ushort BLOCK_SIZE, int SCAN_TYPE, typename BinaryOp, typename T>
static inline T gsx_metal_sort_threadgroup_prefix_scan_store_sum(
    T value,
    thread T &inclusive_sum,
    threadgroup T *shared,
    ushort local_id,
    BinaryOp op)
{
    shared[local_id] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if(local_id < SIMD_WIDTH) {
        T partial_sum = gsx_metal_sort_thread_scan<BLOCK_SIZE / SIMD_WIDTH, SCAN_TYPE>(&shared[local_id * (BLOCK_SIZE / SIMD_WIDTH)], op);
        T prefix = gsx_metal_sort_simdgroup_scan<1>(partial_sum, local_id, op);

        gsx_metal_sort_thread_uniform_apply<BLOCK_SIZE / SIMD_WIDTH>(&shared[local_id * (BLOCK_SIZE / SIMD_WIDTH)], prefix, op);
        if(local_id == (SIMD_WIDTH - 1u)) {
            shared[0] = prefix + partial_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if(SCAN_TYPE == 0) {
        value = (local_id == 0u) ? value : shared[local_id];
    } else {
        value = (local_id == 0u) ? T(0) : shared[local_id];
    }
    inclusive_sum = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return value;
}

template <ushort BLOCK_SIZE>
static inline uchar gsx_metal_sort_flag_head_discontinuity(uint value, threadgroup uint *shared, ushort local_id)
{
    shared[local_id] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uchar result = (local_id == 0u) ? 1u : (shared[local_id] != shared[local_id - 1u]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

template <ushort BLOCK_SIZE>
static inline uchar gsx_metal_sort_flag_tail_discontinuity(uint value, threadgroup uint *shared, ushort local_id)
{
    shared[local_id] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uchar result = (local_id == (BLOCK_SIZE - 1u)) ? 1u : (shared[local_id] != shared[local_id + 1u]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return result;
}

template <ushort BLOCK_SIZE>
static inline uint gsx_metal_sort_by_bit(
    uint value,
    threadgroup uint * scan_shared,
    threadgroup uint * sort_shared,
    ushort local_id,
    uchar current_bit)
{
    uchar mask = (uchar)((value >> current_bit) & 1u);
    uchar2 partial_sum = uchar2(0u);
    uchar2 scan = uchar2(0u);
    ushort2 offset = ushort2(0u);

    scan[mask] = 1u;
    scan = gsx_metal_sort_threadgroup_prefix_scan_store_sum<BLOCK_SIZE, 1>(
        scan,
        partial_sum,
        reinterpret_cast<threadgroup uchar2 *>(scan_shared),
        local_id,
        gsx_metal_sort_sum_op<uchar2>());

    offset[1] = partial_sum[0];
    sort_shared[scan[mask] + offset[mask]] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = sort_shared[local_id];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return value;
}

template <ushort BLOCK_SIZE>
static inline uint gsx_metal_sort_by_two_bits(
    uint value,
    threadgroup uint * scan_shared,
    threadgroup uint * sort_shared,
    ushort local_id,
    uchar current_bit)
{
    uchar mask = (uchar)((value >> current_bit) & 3u);
    uchar4 partial_sum = uchar4(0u);
    uchar4 scan = uchar4(0u);
    ushort4 offset = ushort4(0u);

    scan[mask] = 1u;
    scan = gsx_metal_sort_threadgroup_prefix_scan_store_sum<BLOCK_SIZE, 1>(
        scan,
        partial_sum,
        reinterpret_cast<threadgroup uchar4 *>(scan_shared),
        local_id,
        gsx_metal_sort_sum_op<uchar4>());

    offset[1] = partial_sum[0];
    offset[2] = offset[1] + partial_sum[1];
    offset[3] = offset[2] + partial_sum[2];
    sort_shared[scan[mask] + offset[mask]] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    value = sort_shared[local_id];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return value;
}

template <ushort BLOCK_SIZE, ushort LOCAL_BITS>
static inline uint gsx_metal_sort_local_batch(
    uint value,
    threadgroup uint * scan_shared,
    threadgroup uint * sort_shared,
    ushort local_id)
{
    uchar current_bit = 0u;

    while(current_bit < LOCAL_BITS) {
        if((current_bit + 1u) < LOCAL_BITS) {
            value = gsx_metal_sort_by_two_bits<BLOCK_SIZE>(value, scan_shared, sort_shared, local_id, current_bit);
            current_bit += 2u;
        } else {
            value = gsx_metal_sort_by_bit<BLOCK_SIZE>(value, scan_shared, sort_shared, local_id, current_bit);
            current_bit += 1u;
        }
    }
    return value;
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

kernel void radix_prefix_offsets(
    device const uint * __restrict__ histogram [[buffer(0)]],
    device uint *       __restrict__ global_histogram [[buffer(1)]],
    device uint *       __restrict__ scatter_offsets [[buffer(2)]],
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
        sum += histogram[tid * num_threadgroups + tg];
    }
    exclusive = gsx_metal_sort_scan_exclusive_u32(sum, tid, simd_lane, simd_group_id, simd_totals, simd_offsets);
    global_histogram[tid] = exclusive;

    running_sum = exclusive;
    for(uint tg = 0u; tg < num_threadgroups; ++tg) {
        uint histogram_idx = tid * num_threadgroups + tg;

        scatter_offsets[tg * RADIX_SIZE + tid] = running_sum;
        running_sum += histogram[histogram_idx];
    }
}

kernel void radix_scatter_simd(
    device const uint * __restrict__ keys_in [[buffer(0)]],
    device const uint * __restrict__ values_in [[buffer(1)]],
    device uint *       __restrict__ keys_out [[buffer(2)]],
    device uint *       __restrict__ values_out [[buffer(3)]],
    device const uint * __restrict__ scatter_offsets [[buffer(4)]],
    constant uint &                  array_size [[buffer(5)]],
    constant uint &                  shift [[buffer(6)]],
    threadgroup uint *  __restrict__ local_offsets [[threadgroup(0)]],
    threadgroup uint *  __restrict__ scan_shared [[threadgroup(1)]],
    threadgroup uint *  __restrict__ sort_shared [[threadgroup(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    uint block_start = tgid * KEYS_PER_THREADGROUP;
    uint block_end = min(block_start + KEYS_PER_THREADGROUP, array_size);
    uint block_size = block_end - block_start;
    threadgroup uint *tag_shared = scan_shared;

    if(tid < RADIX_SIZE) {
        local_offsets[tid] = scatter_offsets[tgid * RADIX_SIZE + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for(uint batch = 0u; batch < KEYS_PER_THREAD; ++batch) {
        uint local_idx = batch * THREADGROUP_SIZE + tid;
        uint batch_start = batch * THREADGROUP_SIZE;
        bool full_batch = (batch_start + THREADGROUP_SIZE) <= block_size;
        bool valid = local_idx < block_size;
        uint token_key = LOCAL_SORT_SENTINEL;
        uint token = 0u;
        uint prefix_value = 0u;
        uint prefix_head = 0u;
        uint out_idx = 0u;
        uint src_lane = 0u;
        uint key = 0u;
        uint value = 0u;
        uchar head_flag = 0u;
        uchar tail_flag = 0u;

        if(valid) {
            uint global_idx = block_start + local_idx;
            uint input_key = keys_in[global_idx];

            token_key = (input_key >> shift) & RADIX_MASK;
        }
        token = token_key | (tid << GSX_METAL_SORT_TOKEN_LANE_SHIFT);

        if(full_batch) {
            token = gsx_metal_sort_local_batch<THREADGROUP_SIZE, LOCAL_SORT_BITS_FULL>(token, scan_shared, sort_shared, (ushort)tid);
        } else {
            token = gsx_metal_sort_local_batch<THREADGROUP_SIZE, LOCAL_SORT_BITS_TAIL>(token, scan_shared, sort_shared, (ushort)tid);
        }
        token_key = token & GSX_METAL_SORT_TOKEN_KEY_MASK;
        head_flag = gsx_metal_sort_flag_head_discontinuity<THREADGROUP_SIZE>(token_key, tag_shared, (ushort)tid);

        prefix_value = (head_flag != 0u) ? tid : 0u;
        prefix_head = gsx_metal_sort_prefix_inclusive_max_u32(
            prefix_value,
            tid,
            simd_lane,
            simd_group_id,
            scan_shared,
            scan_shared + SIMD_GROUP_COUNT);
        if(token_key < RADIX_SIZE) {
            out_idx = local_offsets[token_key] + (tid - prefix_head);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        tail_flag = gsx_metal_sort_flag_tail_discontinuity<THREADGROUP_SIZE>(token_key, tag_shared, (ushort)tid);
        if(tail_flag != 0u && token_key < RADIX_SIZE) {
            local_offsets[token_key] = out_idx + 1u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if(token_key < RADIX_SIZE) {
            src_lane = token >> GSX_METAL_SORT_TOKEN_LANE_SHIFT;
            key = keys_in[block_start + batch_start + src_lane];
            value = values_in[block_start + batch_start + src_lane];
            keys_out[out_idx] = key;
            values_out[out_idx] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
