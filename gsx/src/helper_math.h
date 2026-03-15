#ifndef GSX_HELPER_MATH_H
#define GSX_HELPER_MATH_H

#include <cuda_runtime.h>
#include <math.h>

static inline __host__ __device__ float2 make_float2(float value)
{
    return make_float2(value, value);
}

static inline __host__ __device__ float3 make_float3(float value)
{
    return make_float3(value, value, value);
}

static inline __host__ __device__ float3 make_float3(float4 value)
{
    return make_float3(value.x, value.y, value.z);
}

static inline __host__ __device__ float4 make_float4(float3 xyz, float w)
{
    return make_float4(xyz.x, xyz.y, xyz.z, w);
}

static inline __host__ __device__ float2 operator+(float2 lhs, float2 rhs)
{
    return make_float2(lhs.x + rhs.x, lhs.y + rhs.y);
}

static inline __host__ __device__ float2 operator+(float2 lhs, float rhs)
{
    return make_float2(lhs.x + rhs, lhs.y + rhs);
}

static inline __host__ __device__ float2 operator+(float lhs, float2 rhs)
{
    return rhs + lhs;
}

static inline __host__ __device__ float2 operator-(float2 lhs, float2 rhs)
{
    return make_float2(lhs.x - rhs.x, lhs.y - rhs.y);
}

static inline __host__ __device__ float2 operator-(float2 lhs, float rhs)
{
    return make_float2(lhs.x - rhs, lhs.y - rhs);
}

static inline __host__ __device__ float2 operator*(float2 lhs, float rhs)
{
    return make_float2(lhs.x * rhs, lhs.y * rhs);
}

static inline __host__ __device__ float2 operator*(float lhs, float2 rhs)
{
    return rhs * lhs;
}

static inline __host__ __device__ float2 operator/(float2 lhs, float rhs)
{
    return make_float2(lhs.x / rhs, lhs.y / rhs);
}

static inline __host__ __device__ float2 operator*(float2 lhs, float2 rhs)
{
    return make_float2(lhs.x * rhs.x, lhs.y * rhs.y);
}

static inline __host__ __device__ float2 &operator+=(float2 &lhs, float2 rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    return lhs;
}

static inline __host__ __device__ float2 &operator-=(float2 &lhs, float2 rhs)
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    return lhs;
}

static inline __host__ __device__ float3 operator+(float3 lhs, float3 rhs)
{
    return make_float3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

static inline __host__ __device__ float3 operator+(float3 lhs, float rhs)
{
    return make_float3(lhs.x + rhs, lhs.y + rhs, lhs.z + rhs);
}

static inline __host__ __device__ float3 operator+(float lhs, float3 rhs)
{
    return rhs + lhs;
}

static inline __host__ __device__ float3 operator-(float3 lhs, float3 rhs)
{
    return make_float3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

static inline __host__ __device__ float3 operator-(float3 lhs, float rhs)
{
    return make_float3(lhs.x - rhs, lhs.y - rhs, lhs.z - rhs);
}

static inline __host__ __device__ float3 operator-(float3 value)
{
    return make_float3(-value.x, -value.y, -value.z);
}

static inline __host__ __device__ float3 operator*(float3 lhs, float rhs)
{
    return make_float3(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}

static inline __host__ __device__ float3 operator*(float lhs, float3 rhs)
{
    return rhs * lhs;
}

static inline __host__ __device__ float3 operator*(float3 lhs, float3 rhs)
{
    return make_float3(lhs.x * rhs.x, lhs.y * rhs.y, lhs.z * rhs.z);
}

static inline __host__ __device__ float3 operator/(float3 lhs, float rhs)
{
    return make_float3(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
}

static inline __host__ __device__ float3 &operator+=(float3 &lhs, float3 rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    return lhs;
}

static inline __host__ __device__ float3 &operator-=(float3 &lhs, float3 rhs)
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    return lhs;
}

static inline __host__ __device__ float4 operator+(float4 lhs, float4 rhs)
{
    return make_float4(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}

static inline __host__ __device__ float4 &operator+=(float4 &lhs, float4 rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    lhs.w += rhs.w;
    return lhs;
}

static inline __host__ __device__ float dot(float2 lhs, float2 rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y;
}

static inline __host__ __device__ float dot(float3 lhs, float3 rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

static inline __host__ __device__ float length(float2 value)
{
    return sqrtf(dot(value, value));
}

static inline __host__ __device__ float length(float3 value)
{
    return sqrtf(dot(value, value));
}

static inline __host__ __device__ float3 normalize(float3 value)
{
    float norm = length(value);

    if(norm <= 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    return value / norm;
}

static inline __host__ __device__ float clamp(float value, float lower, float upper)
{
    return fminf(fmaxf(value, lower), upper);
}

static inline __host__ __device__ float fast_lerp(float a, float b, float t)
{
    return a + t * (b - a);
}

static inline __host__ __device__ float3 fmaxf(float3 value, float lower)
{
    return make_float3(fmaxf(value.x, lower), fmaxf(value.y, lower), fmaxf(value.z, lower));
}

#endif /* GSX_HELPER_MATH_H */
