#pragma once

#include <cuda_runtime.h>

#include <string>
#include <stdexcept>

#define CUDA_CHECK_THROW(expr)                                                                                       \
    do {                                                                                                             \
        cudaError_t gsx_cuda_check_throw_err__ = (expr);                                                            \
        if(gsx_cuda_check_throw_err__ != cudaSuccess) {                                                             \
            throw std::runtime_error(cudaGetErrorString(gsx_cuda_check_throw_err__));                               \
        }                                                                                                            \
    } while(false)

#define CHECK_CUDA(debug_enabled, message)                                                                          \
    do {                                                                                                             \
        if(debug_enabled) {                                                                                          \
            cudaError_t gsx_cuda_check_err__ = cudaGetLastError();                                                   \
            if(gsx_cuda_check_err__ != cudaSuccess) {                                                                \
                throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(gsx_cuda_check_err__));   \
            }                                                                                                        \
        }                                                                                                            \
    } while(false)

namespace tinygs {

static inline void maybe_sync(cudaStream_t stream)
{
    (void)stream;
}

} /* namespace tinygs */
