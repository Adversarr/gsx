#pragma once

#include "../helper_math.h"

namespace tinygs {

static inline __device__ float3 read_sh0_soa(const float *sh0, int n_primitives, int primitive_idx)
{
    return make_float3(
        sh0[primitive_idx],
        sh0[n_primitives + primitive_idx],
        sh0[2 * n_primitives + primitive_idx]
    );
}

static inline __device__ float3 read_sh_soa(const float *sh, int coeff_idx, int n_primitives, int primitive_idx)
{
    const int coeff_stride = 3 * n_primitives;
    const int base = coeff_idx * coeff_stride + primitive_idx;

    return make_float3(
        sh[base],
        sh[base + n_primitives],
        sh[base + 2 * n_primitives]
    );
}

static inline __device__ void accum_sh0_soa(float *sh0, int n_primitives, int primitive_idx, float3 value)
{
    atomicAdd(&sh0[primitive_idx], value.x);
    atomicAdd(&sh0[n_primitives + primitive_idx], value.y);
    atomicAdd(&sh0[2 * n_primitives + primitive_idx], value.z);
}

static inline __device__ void accum_sh_soa(float *sh, int coeff_idx, int n_primitives, int primitive_idx, float3 value)
{
    const int coeff_stride = 3 * n_primitives;
    const int base = coeff_idx * coeff_stride + primitive_idx;

    atomicAdd(&sh[base], value.x);
    atomicAdd(&sh[base + n_primitives], value.y);
    atomicAdd(&sh[base + 2 * n_primitives], value.z);
}

} /* namespace tinygs */
