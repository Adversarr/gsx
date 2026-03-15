#ifndef GSX_CUDA_COMMON_UTILS_H
#define GSX_CUDA_COMMON_UTILS_H

template <typename T>
static inline __host__ __device__ constexpr T div_round_up(T value, T divisor)
{
    return (value + divisor - 1) / divisor;
}

#endif /* GSX_CUDA_COMMON_UTILS_H */
