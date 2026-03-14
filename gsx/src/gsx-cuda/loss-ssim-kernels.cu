#include <cuda_runtime.h>

#include <cstddef>

namespace {

constexpr int GSX_CUDA_SSIM_WINDOW_SIZE = 11;
constexpr float GSX_CUDA_SSIM_C1 = 0.0001f;
constexpr float GSX_CUDA_SSIM_C2 = 0.0009f;

constexpr int GSX_CUDA_SSIM_FUSED_BLOCK_X = 16;
constexpr int GSX_CUDA_SSIM_FUSED_BLOCK_Y = 16;
constexpr int GSX_CUDA_SSIM_FUSED_HALO = 5;
constexpr int GSX_CUDA_SSIM_FUSED_SHARED_X = GSX_CUDA_SSIM_FUSED_BLOCK_X + 2 * GSX_CUDA_SSIM_FUSED_HALO;
constexpr int GSX_CUDA_SSIM_FUSED_SHARED_Y = GSX_CUDA_SSIM_FUSED_BLOCK_Y + 2 * GSX_CUDA_SSIM_FUSED_HALO;
constexpr int GSX_CUDA_SSIM_FUSED_CONV_X = GSX_CUDA_SSIM_FUSED_BLOCK_X;
constexpr int GSX_CUDA_SSIM_FUSED_CONV_Y = GSX_CUDA_SSIM_FUSED_SHARED_Y;

__constant__ float gsx_cuda_ssim_gauss_1d[GSX_CUDA_SSIM_WINDOW_SIZE] = {
    0.001028380123898387f, 0.0075987582094967365f, 0.036000773310661316f, 0.10936068743467331f, 0.21300552785396576f,
    0.26601171493530273f, 0.21300552785396576f, 0.10936068743467331f, 0.036000773310661316f, 0.0075987582094967365f,
    0.001028380123898387f
};

template <bool IsHWC>
__device__ __forceinline__ size_t gsx_cuda_ssim_fused_offset(
    size_t outer, int c, int y, int x, int channels, int height, int width)
{
    if(IsHWC) {
        return ((((outer * (size_t)height + (size_t)y) * (size_t)width + (size_t)x) * (size_t)channels) + (size_t)c);
    }
    return (((outer * (size_t)channels + (size_t)c) * (size_t)height + (size_t)y) * (size_t)width + (size_t)x);
}

template <bool IsHWC>
__device__ __forceinline__ float gsx_cuda_ssim_fused_sample_or_zero(
    const float *values, size_t outer, int c, int y, int x, int channels, int height, int width)
{
    if(y < 0 || x < 0 || y >= height || x >= width) {
        return 0.0f;
    }
    return values[gsx_cuda_ssim_fused_offset<IsHWC>(outer, c, y, x, channels, height, width)];
}

template <bool IsHWC>
__global__ void gsx_cuda_loss_ssim_forward_fused_f32_kernel(
    float *__restrict__ loss_map,
    const float *__restrict__ prediction,
    const float *__restrict__ target,
    float *__restrict__ dm_buffer_a,
    float *__restrict__ dm_buffer_b,
    size_t outer_count,
    int channels,
    int height,
    int width,
    float scale)
{
    const int pix_y = (int)blockIdx.y * GSX_CUDA_SSIM_FUSED_BLOCK_Y + threadIdx.y;
    const int pix_x = (int)blockIdx.x * GSX_CUDA_SSIM_FUSED_BLOCK_X + threadIdx.x;
    const size_t total_elements = outer_count * (size_t)channels * (size_t)height * (size_t)width;
    __shared__ float tile[GSX_CUDA_SSIM_FUSED_SHARED_Y][GSX_CUDA_SSIM_FUSED_SHARED_X][2];
    __shared__ float xconv[GSX_CUDA_SSIM_FUSED_CONV_Y][GSX_CUDA_SSIM_FUSED_CONV_X][5];

    for(size_t outer = (size_t)blockIdx.z; outer < outer_count; outer += (size_t)gridDim.z) {
        for(int c = 0; c < channels; ++c) {
            const int tile_size = GSX_CUDA_SSIM_FUSED_SHARED_Y * GSX_CUDA_SSIM_FUSED_SHARED_X;
            const int threads = GSX_CUDA_SSIM_FUSED_BLOCK_X * GSX_CUDA_SSIM_FUSED_BLOCK_Y;
            const int steps = (tile_size + threads - 1) / threads;
            const int tile_start_y = (int)blockIdx.y * GSX_CUDA_SSIM_FUSED_BLOCK_Y;
            const int tile_start_x = (int)blockIdx.x * GSX_CUDA_SSIM_FUSED_BLOCK_X;

            for(int s = 0; s < steps; ++s) {
                int tid = s * threads + threadIdx.y * GSX_CUDA_SSIM_FUSED_BLOCK_X + threadIdx.x;
                if(tid < tile_size) {
                    int local_y = tid / GSX_CUDA_SSIM_FUSED_SHARED_X;
                    int local_x = tid % GSX_CUDA_SSIM_FUSED_SHARED_X;
                    int gy = tile_start_y + local_y - GSX_CUDA_SSIM_FUSED_HALO;
                    int gx = tile_start_x + local_x - GSX_CUDA_SSIM_FUSED_HALO;

                    tile[local_y][local_x][0] =
                        gsx_cuda_ssim_fused_sample_or_zero<IsHWC>(prediction, outer, c, gy, gx, channels, height, width);
                    tile[local_y][local_x][1] =
                        gsx_cuda_ssim_fused_sample_or_zero<IsHWC>(target, outer, c, gy, gx, channels, height, width);
                }
            }
            __syncthreads();

            int ly = threadIdx.y;
            int lx = threadIdx.x + GSX_CUDA_SSIM_FUSED_HALO;
            float sum_x = 0.0f;
            float sum_x2 = 0.0f;
            float sum_y = 0.0f;
            float sum_y2 = 0.0f;
            float sum_xy = 0.0f;

#pragma unroll
            for(int d = 1; d <= GSX_CUDA_SSIM_FUSED_HALO; ++d) {
                float w = gsx_cuda_ssim_gauss_1d[GSX_CUDA_SSIM_FUSED_HALO - d];
                float x_left = tile[ly][lx - d][0];
                float y_left = tile[ly][lx - d][1];
                float x_right = tile[ly][lx + d][0];
                float y_right = tile[ly][lx + d][1];

                sum_x += (x_left + x_right) * w;
                sum_x2 += (x_left * x_left + x_right * x_right) * w;
                sum_y += (y_left + y_right) * w;
                sum_y2 += (y_left * y_left + y_right * y_right) * w;
                sum_xy += (x_left * y_left + x_right * y_right) * w;
            }
            {
                float center_x = tile[ly][lx][0];
                float center_y = tile[ly][lx][1];
                float w = gsx_cuda_ssim_gauss_1d[GSX_CUDA_SSIM_FUSED_HALO];
                sum_x += center_x * w;
                sum_x2 += center_x * center_x * w;
                sum_y += center_y * w;
                sum_y2 += center_y * center_y * w;
                sum_xy += center_x * center_y * w;
            }

            xconv[ly][threadIdx.x][0] = sum_x;
            xconv[ly][threadIdx.x][1] = sum_x2;
            xconv[ly][threadIdx.x][2] = sum_y;
            xconv[ly][threadIdx.x][3] = sum_y2;
            xconv[ly][threadIdx.x][4] = sum_xy;

            int ly2 = ly + GSX_CUDA_SSIM_FUSED_BLOCK_Y;
            if(ly2 < GSX_CUDA_SSIM_FUSED_CONV_Y) {
                sum_x = 0.0f;
                sum_x2 = 0.0f;
                sum_y = 0.0f;
                sum_y2 = 0.0f;
                sum_xy = 0.0f;

#pragma unroll
                for(int d = 1; d <= GSX_CUDA_SSIM_FUSED_HALO; ++d) {
                    float w = gsx_cuda_ssim_gauss_1d[GSX_CUDA_SSIM_FUSED_HALO - d];
                    float x_left = tile[ly2][lx - d][0];
                    float y_left = tile[ly2][lx - d][1];
                    float x_right = tile[ly2][lx + d][0];
                    float y_right = tile[ly2][lx + d][1];

                    sum_x += (x_left + x_right) * w;
                    sum_x2 += (x_left * x_left + x_right * x_right) * w;
                    sum_y += (y_left + y_right) * w;
                    sum_y2 += (y_left * y_left + y_right * y_right) * w;
                    sum_xy += (x_left * y_left + x_right * y_right) * w;
                }
                {
                    float center_x = tile[ly2][lx][0];
                    float center_y = tile[ly2][lx][1];
                    float w = gsx_cuda_ssim_gauss_1d[GSX_CUDA_SSIM_FUSED_HALO];
                    sum_x += center_x * w;
                    sum_x2 += center_x * center_x * w;
                    sum_y += center_y * w;
                    sum_y2 += center_y * center_y * w;
                    sum_xy += center_x * center_y * w;
                }

                xconv[ly2][threadIdx.x][0] = sum_x;
                xconv[ly2][threadIdx.x][1] = sum_x2;
                xconv[ly2][threadIdx.x][2] = sum_y;
                xconv[ly2][threadIdx.x][3] = sum_y2;
                xconv[ly2][threadIdx.x][4] = sum_xy;
            }
            __syncthreads();

            if(pix_x < width && pix_y < height) {
                ly = threadIdx.y + GSX_CUDA_SSIM_FUSED_HALO;
                lx = threadIdx.x;
                float out0 = 0.0f;
                float out1 = 0.0f;
                float out2 = 0.0f;
                float out3 = 0.0f;
                float out4 = 0.0f;

#pragma unroll
                for(int d = 1; d <= GSX_CUDA_SSIM_FUSED_HALO; ++d) {
                    float w = gsx_cuda_ssim_gauss_1d[GSX_CUDA_SSIM_FUSED_HALO - d];
                    out0 += (xconv[ly - d][lx][0] + xconv[ly + d][lx][0]) * w;
                    out1 += (xconv[ly - d][lx][1] + xconv[ly + d][lx][1]) * w;
                    out2 += (xconv[ly - d][lx][2] + xconv[ly + d][lx][2]) * w;
                    out3 += (xconv[ly - d][lx][3] + xconv[ly + d][lx][3]) * w;
                    out4 += (xconv[ly - d][lx][4] + xconv[ly + d][lx][4]) * w;
                }
                {
                    float w = gsx_cuda_ssim_gauss_1d[GSX_CUDA_SSIM_FUSED_HALO];
                    out0 += xconv[ly][lx][0] * w;
                    out1 += xconv[ly][lx][1] * w;
                    out2 += xconv[ly][lx][2] * w;
                    out3 += xconv[ly][lx][3] * w;
                    out4 += xconv[ly][lx][4] * w;
                }

                float mu1 = out0;
                float mu2 = out2;
                float mu1_sq = mu1 * mu1;
                float mu2_sq = mu2 * mu2;
                float sigma1_sq = out1 - mu1_sq;
                float sigma2_sq = out3 - mu2_sq;
                float sigma12 = out4 - mu1 * mu2;
                float a = mu1_sq + mu2_sq + GSX_CUDA_SSIM_C1;
                float b = sigma1_sq + sigma2_sq + GSX_CUDA_SSIM_C2;
                float c_val = 2.0f * mu1 * mu2 + GSX_CUDA_SSIM_C1;
                float d_val = 2.0f * sigma12 + GSX_CUDA_SSIM_C2;
                float denominator = a * b;
                float ssim = denominator == 0.0f ? 1.0f : (c_val * d_val) / denominator;
                size_t idx = gsx_cuda_ssim_fused_offset<IsHWC>(outer, c, pix_y, pix_x, channels, height, width);

                loss_map[idx] += scale * (1.0f - ssim);

                if(dm_buffer_a != nullptr && dm_buffer_b != nullptr) {
                    float dm_dmu1 = 0.0f;
                    float dm_dsigma1_sq = 0.0f;
                    float dm_dsigma12 = 0.0f;

                    if(denominator != 0.0f) {
                        dm_dmu1 = ((mu2 * 2.0f * d_val) / denominator - (mu2 * 2.0f * c_val) / denominator
                                   - (mu1 * 2.0f * c_val * d_val) / (a * denominator)
                                   + (mu1 * 2.0f * c_val * d_val) / (denominator * b));
                        dm_dsigma1_sq = -(c_val * d_val) / (denominator * b);
                        dm_dsigma12 = (2.0f * c_val) / denominator;
                    }

                    dm_buffer_a[idx] = dm_dmu1;
                    dm_buffer_a[idx + total_elements] = dm_dsigma1_sq;
                    dm_buffer_b[idx] = dm_dsigma12;
                }
            }
            __syncthreads();
        }
    }
}

template <bool IsHWC>
__global__ void gsx_cuda_loss_ssim_backward_fused_f32_kernel(
    float *__restrict__ grad_prediction,
    const float *__restrict__ prediction,
    const float *__restrict__ target,
    const float *__restrict__ dm_buffer_a,
    const float *__restrict__ dm_buffer_b,
    size_t outer_count,
    int channels,
    int height,
    int width,
    float grad_scale)
{
    const int pix_y = (int)blockIdx.y * GSX_CUDA_SSIM_FUSED_BLOCK_Y + threadIdx.y;
    const int pix_x = (int)blockIdx.x * GSX_CUDA_SSIM_FUSED_BLOCK_X + threadIdx.x;
    const size_t total_elements = outer_count * (size_t)channels * (size_t)height * (size_t)width;
    const float chain = -grad_scale;
    __shared__ float sdata[3][GSX_CUDA_SSIM_FUSED_SHARED_Y][GSX_CUDA_SSIM_FUSED_SHARED_X];
    __shared__ float scratch[GSX_CUDA_SSIM_FUSED_CONV_Y][GSX_CUDA_SSIM_FUSED_CONV_X][3];

    for(size_t outer = (size_t)blockIdx.z; outer < outer_count; outer += (size_t)gridDim.z) {
        for(int c = 0; c < channels; ++c) {
            const int tile_size = GSX_CUDA_SSIM_FUSED_SHARED_Y * GSX_CUDA_SSIM_FUSED_SHARED_X;
            const int threads = GSX_CUDA_SSIM_FUSED_BLOCK_X * GSX_CUDA_SSIM_FUSED_BLOCK_Y;
            const int steps = (tile_size + threads - 1) / threads;
            const int tile_start_y = (int)blockIdx.y * GSX_CUDA_SSIM_FUSED_BLOCK_Y;
            const int tile_start_x = (int)blockIdx.x * GSX_CUDA_SSIM_FUSED_BLOCK_X;
            float p1 = 0.0f;
            float p2 = 0.0f;

            if(pix_x < width && pix_y < height) {
                p1 = gsx_cuda_ssim_fused_sample_or_zero<IsHWC>(prediction, outer, c, pix_y, pix_x, channels, height, width);
                p2 = gsx_cuda_ssim_fused_sample_or_zero<IsHWC>(target, outer, c, pix_y, pix_x, channels, height, width);
            }

            for(int s = 0; s < steps; ++s) {
                int tid = s * threads + threadIdx.y * GSX_CUDA_SSIM_FUSED_BLOCK_X + threadIdx.x;
                if(tid < tile_size) {
                    int local_y = tid / GSX_CUDA_SSIM_FUSED_SHARED_X;
                    int local_x = tid % GSX_CUDA_SSIM_FUSED_SHARED_X;
                    int gy = tile_start_y + local_y - GSX_CUDA_SSIM_FUSED_HALO;
                    int gx = tile_start_x + local_x - GSX_CUDA_SSIM_FUSED_HALO;

                    if(gy < 0 || gx < 0 || gy >= height || gx >= width) {
                        sdata[0][local_y][local_x] = 0.0f;
                        sdata[1][local_y][local_x] = 0.0f;
                        sdata[2][local_y][local_x] = 0.0f;
                    } else {
                        size_t idx = gsx_cuda_ssim_fused_offset<IsHWC>(outer, c, gy, gx, channels, height, width);
                        sdata[0][local_y][local_x] = dm_buffer_a[idx] * chain;
                        sdata[1][local_y][local_x] = dm_buffer_a[idx + total_elements] * chain;
                        sdata[2][local_y][local_x] = dm_buffer_b[idx] * chain;
                    }
                }
            }
            __syncthreads();

            int ly = threadIdx.y;
            int lx = threadIdx.x + GSX_CUDA_SSIM_FUSED_HALO;
            for(int pass = 0; pass < 2; ++pass) {
                int yy = ly + pass * GSX_CUDA_SSIM_FUSED_BLOCK_Y;
                if(yy < GSX_CUDA_SSIM_FUSED_CONV_Y) {
                    float sum0 = 0.0f;
                    float sum1 = 0.0f;
                    float sum2 = 0.0f;

#pragma unroll
                    for(int d = 1; d <= GSX_CUDA_SSIM_FUSED_HALO; ++d) {
                        float w = gsx_cuda_ssim_gauss_1d[GSX_CUDA_SSIM_FUSED_HALO - d];
                        sum0 += (sdata[0][yy][lx - d] + sdata[0][yy][lx + d]) * w;
                        sum1 += (sdata[1][yy][lx - d] + sdata[1][yy][lx + d]) * w;
                        sum2 += (sdata[2][yy][lx - d] + sdata[2][yy][lx + d]) * w;
                    }
                    {
                        float w = gsx_cuda_ssim_gauss_1d[GSX_CUDA_SSIM_FUSED_HALO];
                        sum0 += sdata[0][yy][lx] * w;
                        sum1 += sdata[1][yy][lx] * w;
                        sum2 += sdata[2][yy][lx] * w;
                    }
                    scratch[yy][threadIdx.x][0] = sum0;
                    scratch[yy][threadIdx.x][1] = sum1;
                    scratch[yy][threadIdx.x][2] = sum2;
                }
            }
            __syncthreads();

            if(pix_x < width && pix_y < height) {
                ly = threadIdx.y + GSX_CUDA_SSIM_FUSED_HALO;
                lx = threadIdx.x;
                float sum0 = 0.0f;
                float sum1 = 0.0f;
                float sum2 = 0.0f;

#pragma unroll
                for(int d = 1; d <= GSX_CUDA_SSIM_FUSED_HALO; ++d) {
                    float w = gsx_cuda_ssim_gauss_1d[GSX_CUDA_SSIM_FUSED_HALO - d];
                    sum0 += (scratch[ly - d][lx][0] + scratch[ly + d][lx][0]) * w;
                    sum1 += (scratch[ly - d][lx][1] + scratch[ly + d][lx][1]) * w;
                    sum2 += (scratch[ly - d][lx][2] + scratch[ly + d][lx][2]) * w;
                }
                {
                    float w = gsx_cuda_ssim_gauss_1d[GSX_CUDA_SSIM_FUSED_HALO];
                    sum0 += scratch[ly][lx][0] * w;
                    sum1 += scratch[ly][lx][1] * w;
                    sum2 += scratch[ly][lx][2] * w;
                }

                size_t idx = gsx_cuda_ssim_fused_offset<IsHWC>(outer, c, pix_y, pix_x, channels, height, width);
                grad_prediction[idx] += sum0 + 2.0f * p1 * sum1 + p2 * sum2;
            }
            __syncthreads();
        }
    }
}

}  // namespace

extern "C" cudaError_t gsx_cuda_loss_ssim_chw_f32_kernel_launch(
    float *loss_map,
    float *grad_prediction,
    const float *prediction,
    const float *target,
    size_t outer_count,
    int channels,
    int height,
    int width,
    float scale,
    float grad_scale,
    float *ssim_buffer_a,
    float *ssim_buffer_b,
    cudaStream_t stream
)
{
    dim3 block((unsigned int)GSX_CUDA_SSIM_FUSED_BLOCK_X, (unsigned int)GSX_CUDA_SSIM_FUSED_BLOCK_Y, 1u);
    dim3 grid(
        (unsigned int)((width + GSX_CUDA_SSIM_FUSED_BLOCK_X - 1) / GSX_CUDA_SSIM_FUSED_BLOCK_X),
        (unsigned int)((height + GSX_CUDA_SSIM_FUSED_BLOCK_Y - 1) / GSX_CUDA_SSIM_FUSED_BLOCK_Y),
        (unsigned int)((outer_count < (size_t)65535) ? outer_count : (size_t)65535));
    cudaError_t error = cudaSuccess;

    if(outer_count == 0 || channels <= 0 || height <= 0 || width <= 0) {
        return cudaSuccess;
    }
    if(grad_prediction != nullptr && (ssim_buffer_a == nullptr || ssim_buffer_b == nullptr)) {
        return cudaErrorInvalidValue;
    }

    gsx_cuda_loss_ssim_forward_fused_f32_kernel<false><<<grid, block, 0, stream>>>(
        loss_map, prediction, target, ssim_buffer_a, ssim_buffer_b, outer_count, channels, height, width, scale);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        return error;
    }
    if(grad_prediction != nullptr) {
        gsx_cuda_loss_ssim_backward_fused_f32_kernel<false><<<grid, block, 0, stream>>>(
            grad_prediction, prediction, target, ssim_buffer_a, ssim_buffer_b, outer_count, channels, height, width, grad_scale);
        return cudaGetLastError();
    }

    return cudaSuccess;
}

extern "C" cudaError_t gsx_cuda_loss_ssim_hwc_f32_kernel_launch(
    float *loss_map,
    float *grad_prediction,
    const float *prediction,
    const float *target,
    size_t outer_count,
    int channels,
    int height,
    int width,
    float scale,
    float grad_scale,
    float *ssim_buffer_a,
    float *ssim_buffer_b,
    cudaStream_t stream
)
{
    dim3 block((unsigned int)GSX_CUDA_SSIM_FUSED_BLOCK_X, (unsigned int)GSX_CUDA_SSIM_FUSED_BLOCK_Y, 1u);
    dim3 grid(
        (unsigned int)((width + GSX_CUDA_SSIM_FUSED_BLOCK_X - 1) / GSX_CUDA_SSIM_FUSED_BLOCK_X),
        (unsigned int)((height + GSX_CUDA_SSIM_FUSED_BLOCK_Y - 1) / GSX_CUDA_SSIM_FUSED_BLOCK_Y),
        (unsigned int)((outer_count < (size_t)65535) ? outer_count : (size_t)65535));
    cudaError_t error = cudaSuccess;

    if(outer_count == 0 || channels <= 0 || height <= 0 || width <= 0) {
        return cudaSuccess;
    }
    if(grad_prediction != nullptr && (ssim_buffer_a == nullptr || ssim_buffer_b == nullptr)) {
        return cudaErrorInvalidValue;
    }

    gsx_cuda_loss_ssim_forward_fused_f32_kernel<true><<<grid, block, 0, stream>>>(
        loss_map, prediction, target, ssim_buffer_a, ssim_buffer_b, outer_count, channels, height, width, scale);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        return error;
    }
    if(grad_prediction != nullptr) {
        gsx_cuda_loss_ssim_backward_fused_f32_kernel<true><<<grid, block, 0, stream>>>(
            grad_prediction, prediction, target, ssim_buffer_a, ssim_buffer_b, outer_count, channels, height, width, grad_scale);
        return cudaGetLastError();
    }

    return cudaSuccess;
}
