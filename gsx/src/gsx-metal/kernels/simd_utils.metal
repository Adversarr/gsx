#ifndef GSX_METAL_SIMD_UTILS_METAL
#define GSX_METAL_SIMD_UTILS_METAL

#include <metal_stdlib>
#include <metal_atomic>

template<typename T>
inline T gsx_metal_simd_shuffle(T value, ushort lane)
{
    return metal::simd_shuffle(value, lane);
}

template<typename T>
inline T gsx_metal_simd_shuffle_up(T value, ushort delta)
{
    return metal::simd_shuffle_up(value, delta);
}

template<typename T>
inline T gsx_metal_simd_broadcast_first(T value)
{
    return metal::simd_broadcast_first(value);
}

template<typename T>
inline T gsx_metal_simd_sum(T value)
{
    return metal::simd_sum(value);
}

template<typename T>
inline T gsx_metal_simd_max(T value)
{
    return metal::simd_max(value);
}

inline ulong gsx_metal_simd_active_threads_mask()
{
    return (ulong)metal::simd_active_threads_mask();
}

inline ulong gsx_metal_simd_ballot(bool predicate)
{
    return (ulong)metal::simd_ballot(predicate);
}

inline bool gsx_metal_simd_any(bool predicate)
{
    return metal::simd_any(predicate);
}

inline bool gsx_metal_simd_all(bool predicate)
{
    return metal::simd_all(predicate);
}

template<typename T>
inline T gsx_metal_simd_prefix_exclusive_sum(T value)
{
    return metal::simd_prefix_exclusive_sum(value);
}

inline void gsx_metal_atomic_add_f32(device float *values, uint index, float delta)
{
    device metal::atomic_float *atomic_values = reinterpret_cast<device metal::atomic_float *>(values);
    metal::atomic_fetch_add_explicit(&atomic_values[index], delta, metal::memory_order_relaxed);
}

inline void gsx_metal_atomic_add_f32x2(device float *values, uint index, float2 delta)
{
    gsx_metal_atomic_add_f32(values, index, delta.x);
    gsx_metal_atomic_add_f32(values, index + 1u, delta.y);
}

inline void gsx_metal_atomic_add_f32x3(device float *values, uint index, float3 delta)
{
    gsx_metal_atomic_add_f32(values, index, delta.x);
    gsx_metal_atomic_add_f32(values, index + 1u, delta.y);
    gsx_metal_atomic_add_f32(values, index + 2u, delta.z);
}

inline void gsx_metal_atomic_max_f32_nonnegative(device float *values, uint index, float value)
{
    device metal::atomic_uint *atomic_values = reinterpret_cast<device metal::atomic_uint *>(values);

    if(value <= 0.0f) {
        return;
    }
    metal::atomic_fetch_max_explicit(&atomic_values[index], as_type<uint>(value), metal::memory_order_relaxed);
}

#endif
