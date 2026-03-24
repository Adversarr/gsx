#include "internal.h"
#include "../extra/gsx-image-impl.h"
#include "../pcg32.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

gsx_error gsx_cuda_backend_buffer_type_get_info(gsx_backend_buffer_type_t buffer_type, gsx_backend_buffer_type_info *out_info)
{
    gsx_cuda_backend_buffer_type *cuda_buffer_type = gsx_cuda_backend_buffer_type_from_base(buffer_type);

    if(out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_info must be non-null");
    }

    *out_info = cuda_buffer_type->info;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_buffer_type_get_alloc_size(gsx_backend_buffer_type_t buffer_type, gsx_size_t requested_size_bytes, gsx_size_t *out_alloc_size_bytes)
{
    gsx_cuda_backend_buffer_type *cuda_buffer_type = gsx_cuda_backend_buffer_type_from_base(buffer_type);

    if(out_alloc_size_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_alloc_size_bytes must be non-null");
    }

    if(gsx_round_up_overflows(requested_size_bytes, cuda_buffer_type->info.alignment_bytes, out_alloc_size_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "allocation size overflow");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_buffer_type_init_buffer(gsx_backend_buffer_type_t buffer_type, const gsx_backend_buffer_desc *desc, gsx_backend_buffer_t *out_buffer)
{
    gsx_cuda_backend_buffer_type *cuda_buffer_type = gsx_cuda_backend_buffer_type_from_base(buffer_type);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer_type->backend);
    gsx_cuda_backend_buffer *cuda_buffer = NULL;
    gsx_size_t alloc_size_bytes = 0;
    gsx_size_t effective_alignment = 0;
    cudaError_t cuda_err = cudaSuccess;
    void *ptr = NULL;

    if(out_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_buffer must be non-null");
    }
    *out_buffer = NULL;

    if(desc == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc must be non-null");
    }
    if(desc->size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc->size_bytes must be non-zero");
    }

    effective_alignment = cuda_buffer_type->info.alignment_bytes;
    if(desc->alignment_bytes > effective_alignment) {
        effective_alignment = desc->alignment_bytes;
    }

    if(gsx_round_up_overflows(desc->size_bytes, effective_alignment, &alloc_size_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "allocation size overflow");
    }

    cuda_buffer = (gsx_cuda_backend_buffer *)calloc(1, sizeof(*cuda_buffer));
    if(cuda_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate buffer struct");
    }

    if(cuda_buffer_type->info.type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        cuda_err = cudaMalloc(&ptr, alloc_size_bytes);
        if(cuda_err != cudaSuccess) {
            free(cuda_buffer);
            return gsx_cuda_make_error(cuda_err, "cudaMalloc failed");
        }
    } else if(cuda_buffer_type->info.type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        cuda_err = cudaMallocHost(&ptr, alloc_size_bytes);
        if(cuda_err != cudaSuccess) {
            free(cuda_buffer);
            return gsx_cuda_make_error(cuda_err, "cudaMallocHost failed");
        }
    } else {
        free(cuda_buffer);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "unsupported buffer type");
    }

    cuda_buffer->base.iface = &gsx_cuda_backend_buffer_iface;
    cuda_buffer->base.buffer_type = buffer_type;
    cuda_buffer->base.size_bytes = desc->size_bytes;
    cuda_buffer->base.alignment_bytes = effective_alignment;
    cuda_buffer->ptr = ptr;
    cuda_buffer->alloc_size_bytes = alloc_size_bytes;

    cuda_backend->base.live_buffer_count += 1;
    *out_buffer = &cuda_buffer->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_buffer_free(gsx_backend_buffer_t buffer)
{
    gsx_cuda_backend_buffer *cuda_buffer = NULL;
    gsx_cuda_backend *cuda_backend = NULL;
    gsx_backend_buffer_type_class type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    cudaError_t cuda_err = cudaSuccess;

    if(buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer must be non-null");
    }
    cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    type = gsx_cuda_backend_buffer_get_type_class(buffer);
    if(cuda_backend->base.live_buffer_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "backend live_buffer_count underflow in buffer free");
    }

    if(cuda_buffer->ptr != NULL) {
        if(type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
            cuda_err = cudaFree(cuda_buffer->ptr);
        } else if(type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
            cuda_err = cudaFreeHost(cuda_buffer->ptr);
        }
        cuda_buffer->ptr = NULL;
    }

    cuda_backend->base.live_buffer_count -= 1;
    free(cuda_buffer);

    if(cuda_err != cudaSuccess) {
        return gsx_cuda_make_error(cuda_err, "CUDA free failed");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_buffer_get_info(gsx_backend_buffer_t buffer, gsx_backend_buffer_info *out_info)
{
    if(out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_info must be non-null");
    }

    out_info->backend = buffer->buffer_type->backend;
    out_info->buffer_type = buffer->buffer_type;
    out_info->size_bytes = buffer->size_bytes;
    out_info->alignment_bytes = buffer->alignment_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_buffer_get_native_handle(gsx_backend_buffer_t buffer, void **out_handle)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);

    if(out_handle == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_handle must be non-null");
    }

    *out_handle = cuda_buffer->ptr;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_buffer_upload(gsx_backend_buffer_t buffer, gsx_size_t dst_offset_bytes, const void *src_bytes, gsx_size_t byte_count)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    gsx_cuda_backend_buffer_type *cuda_buffer_type = gsx_cuda_backend_buffer_type_from_base(buffer->buffer_type);
    cudaError_t cuda_err = cudaSuccess;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_cuda_backend_buffer_check_range(buffer, dst_offset_bytes, byte_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(byte_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(src_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src_bytes must be non-null for non-zero byte_count");
    }

    if(cuda_buffer_type->info.type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        cuda_err = cudaMemcpyAsync(
            (char*)cuda_buffer->ptr + dst_offset_bytes,
            src_bytes,
            byte_count,
            cudaMemcpyHostToDevice,
            cuda_backend->major_stream
        );
    } else {
        cuda_err = cudaMemcpyAsync(
            (char*)cuda_buffer->ptr + dst_offset_bytes,
            src_bytes,
            byte_count,
            cudaMemcpyHostToHost,
            cuda_backend->major_stream
        );
    }

    return gsx_cuda_make_error(cuda_err, "cudaMemcpyAsync upload failed");
}

gsx_error gsx_cuda_backend_buffer_download(gsx_backend_buffer_t buffer, gsx_size_t src_offset_bytes, void *dst_bytes, gsx_size_t byte_count)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    gsx_cuda_backend_buffer_type *cuda_buffer_type = gsx_cuda_backend_buffer_type_from_base(buffer->buffer_type);
    cudaError_t cuda_err = cudaSuccess;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_cuda_backend_buffer_check_range(buffer, src_offset_bytes, byte_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(byte_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(dst_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_bytes must be non-null for non-zero byte_count");
    }

    if(cuda_buffer_type->info.type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        cuda_err = cudaMemcpyAsync(
            dst_bytes,
            (const char*)cuda_buffer->ptr + src_offset_bytes,
            byte_count,
            cudaMemcpyDeviceToHost,
            cuda_backend->major_stream
        );
    } else {
        cuda_err = cudaMemcpyAsync(
            dst_bytes,
            (const char*)cuda_buffer->ptr + src_offset_bytes,
            byte_count,
            cudaMemcpyHostToHost,
            cuda_backend->major_stream
        );
    }

    return gsx_cuda_make_error(cuda_err, "cudaMemcpyAsync download failed");
}

gsx_error gsx_cuda_backend_buffer_set_zero(gsx_backend_buffer_t buffer)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    cudaError_t cuda_err = cudaSuccess;

    if(gsx_cuda_backend_buffer_get_type_class(buffer) == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        memset(cuda_buffer->ptr, 0, (size_t)cuda_buffer->alloc_size_bytes);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    cuda_err = cudaMemsetAsync(cuda_buffer->ptr, 0, cuda_buffer->alloc_size_bytes, cuda_backend->major_stream);
    return gsx_cuda_make_error(cuda_err, "cudaMemsetAsync failed");
}

gsx_error gsx_cuda_backend_buffer_memset_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    uint8_t value,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    cudaError_t cuda_err = cudaSuccess;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t tensor_offset = 0;
    gsx_size_t total_offset = 0;

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view->buffer must match buffer");
    }

    tensor_offset = tensor_view->offset_bytes;
    if(gsx_size_add_overflows(tensor_offset, offset_bytes, &total_offset)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor offset overflow");
    }

    error = gsx_cuda_backend_buffer_check_range(buffer, total_offset, size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(gsx_cuda_backend_buffer_get_type_class(buffer) == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        memset((char*)cuda_buffer->ptr + total_offset, value, (size_t)size_bytes);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    cuda_err = cudaMemsetAsync(
        (char*)cuda_buffer->ptr + total_offset,
        value,
        size_bytes,
        cuda_backend->major_stream
    );
    return gsx_cuda_make_error(cuda_err, "cudaMemsetAsync tensor failed");
}

gsx_error gsx_cuda_backend_buffer_set_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *src_bytes,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    gsx_cuda_backend_buffer_type *cuda_buffer_type = gsx_cuda_backend_buffer_type_from_base(buffer->buffer_type);
    cudaError_t cuda_err = cudaSuccess;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t tensor_offset = 0;
    gsx_size_t total_offset = 0;

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view->buffer must match buffer");
    }

    tensor_offset = tensor_view->offset_bytes;
    if(gsx_size_add_overflows(tensor_offset, offset_bytes, &total_offset)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor offset overflow");
    }

    error = gsx_cuda_backend_buffer_check_range(buffer, total_offset, size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(src_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src_bytes must be non-null for non-zero size_bytes");
    }

    if(cuda_buffer_type->info.type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        cuda_err = cudaMemcpyAsync(
            (char*)cuda_buffer->ptr + total_offset,
            src_bytes,
            size_bytes,
            cudaMemcpyHostToDevice,
            cuda_backend->major_stream
        );
    } else {
        cuda_err = cudaMemcpyAsync(
            (char*)cuda_buffer->ptr + total_offset,
            src_bytes,
            size_bytes,
            cudaMemcpyHostToHost,
            cuda_backend->major_stream
        );
    }
    return gsx_cuda_make_error(cuda_err, "cudaMemcpyAsync set_tensor failed");
}

gsx_error gsx_cuda_backend_buffer_get_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    void *dst_bytes,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    gsx_cuda_backend_buffer_type *cuda_buffer_type = gsx_cuda_backend_buffer_type_from_base(buffer->buffer_type);
    cudaError_t cuda_err = cudaSuccess;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t tensor_offset = 0;
    gsx_size_t total_offset = 0;

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view->buffer must match buffer");
    }

    tensor_offset = tensor_view->offset_bytes;
    if(gsx_size_add_overflows(tensor_offset, offset_bytes, &total_offset)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor offset overflow");
    }

    error = gsx_cuda_backend_buffer_check_range(buffer, total_offset, size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(dst_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_bytes must be non-null for non-zero size_bytes");
    }

    if(cuda_buffer_type->info.type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        cuda_err = cudaMemcpyAsync(
            dst_bytes,
            (const char*)cuda_buffer->ptr + total_offset,
            size_bytes,
            cudaMemcpyDeviceToHost,
            cuda_backend->major_stream
        );
    } else {
        cuda_err = cudaMemcpyAsync(
            dst_bytes,
            (const char*)cuda_buffer->ptr + total_offset,
            size_bytes,
            cudaMemcpyHostToHost,
            cuda_backend->major_stream
        );
    }
    return gsx_cuda_make_error(cuda_err, "cudaMemcpyAsync get_tensor failed");
}

gsx_error gsx_cuda_backend_buffer_copy_tensor(gsx_backend_buffer_t dst_buffer, const gsx_backend_tensor_view *src_view, const gsx_backend_tensor_view *dst_view)
{
    gsx_cuda_backend_buffer *cuda_dst_buffer = NULL;
    gsx_cuda_backend_buffer *cuda_src_buffer = NULL;
    gsx_cuda_backend *cuda_backend = NULL;
    gsx_cuda_backend_buffer_type *cuda_dst_type = NULL;
    gsx_cuda_backend_buffer_type *cuda_src_type = NULL;
    cudaError_t cuda_err = cudaSuccess;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t copy_size = 0;

    if(src_view == NULL || dst_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src_view and dst_view must be non-null");
    }
    if(src_view->buffer == NULL || dst_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor views must have valid buffers");
    }
    if(dst_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_view->buffer must match dst_buffer");
    }
    if(src_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "copy_tensor requires source and destination to belong to the same backend");
    }

    copy_size = src_view->size_bytes;
    if(dst_view->size_bytes != copy_size) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "copy_tensor requires equal source and destination sizes");
    }
    if(copy_size == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    error = gsx_cuda_backend_buffer_check_range(src_view->buffer, src_view->offset_bytes, copy_size);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_backend_buffer_check_range(dst_view->buffer, dst_view->offset_bytes, copy_size);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    {
        gsx_size_t src_end = 0, dst_end = 0;
        gsx_size_add_overflows(src_view->offset_bytes, copy_size, &src_end);
        gsx_size_add_overflows(dst_view->offset_bytes, copy_size, &dst_end);

        if(src_view->buffer == dst_view->buffer) {
            bool overlaps = !(src_end <= dst_view->offset_bytes || dst_end <= src_view->offset_bytes);
            if(overlaps) {
                return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "copy_tensor source and destination regions overlap");
            }
        }
    }

    cuda_dst_buffer = gsx_cuda_backend_buffer_from_base(dst_buffer);
    cuda_src_buffer = gsx_cuda_backend_buffer_from_base(src_view->buffer);
    cuda_backend = gsx_cuda_backend_from_base(dst_buffer->buffer_type->backend);
    cuda_dst_type = gsx_cuda_backend_buffer_type_from_base(dst_buffer->buffer_type);
    cuda_src_type = gsx_cuda_backend_buffer_type_from_base(src_view->buffer->buffer_type);

    {
        enum cudaMemcpyKind kind = cudaMemcpyDefault;
        if(cuda_src_type->info.type == GSX_BACKEND_BUFFER_TYPE_DEVICE && cuda_dst_type->info.type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
            kind = cudaMemcpyDeviceToDevice;
        } else if(cuda_src_type->info.type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED && cuda_dst_type->info.type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
            kind = cudaMemcpyHostToDevice;
        } else if(cuda_src_type->info.type == GSX_BACKEND_BUFFER_TYPE_DEVICE && cuda_dst_type->info.type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
            kind = cudaMemcpyDeviceToHost;
        } else {
            kind = cudaMemcpyHostToHost;
        }

        cuda_err = cudaMemcpyAsync(
            (char*)cuda_dst_buffer->ptr + dst_view->offset_bytes,
            (const char*)cuda_src_buffer->ptr + src_view->offset_bytes,
            copy_size,
            kind,
            cuda_backend->major_stream
        );
    }

    return gsx_cuda_make_error(cuda_err, "cudaMemcpyAsync copy_tensor failed");
}

gsx_error gsx_cuda_backend_buffer_fill_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *value_bytes,
    gsx_size_t value_size_bytes
)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    gsx_backend_buffer_type_class buffer_type_class = gsx_cuda_backend_buffer_get_type_class(buffer);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    cudaError_t cuda_err = cudaSuccess;
    void *value_device_bytes = NULL;

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view->buffer must match buffer");
    }
    if(value_size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "value_size_bytes must be non-zero");
    }
    if(tensor_view->size_bytes != 0 && value_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "value_bytes must be non-null when the tensor is non-empty");
    }
    if(tensor_view->size_bytes % value_size_bytes != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor byte size must be a multiple of value_size_bytes");
    }

    error = gsx_cuda_backend_buffer_check_range(buffer, tensor_view->offset_bytes, tensor_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor_view->size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(buffer_type_class == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        gsx_cuda_backend_fill_host_bytes(
            (char*)cuda_buffer->ptr + tensor_view->offset_bytes,
            tensor_view->size_bytes,
            value_bytes,
            value_size_bytes
        );
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    cuda_err = cudaMalloc(&value_device_bytes, value_size_bytes);
    if(cuda_err != cudaSuccess) {
        return gsx_cuda_make_error(cuda_err, "cudaMalloc for fill_tensor value staging failed");
    }

    cuda_err = cudaMemcpyAsync(
        value_device_bytes,
        value_bytes,
        value_size_bytes,
        cudaMemcpyHostToDevice,
        cuda_backend->major_stream
    );
    if(cuda_err != cudaSuccess) {
        cudaFree(value_device_bytes);
        return gsx_cuda_make_error(cuda_err, "cudaMemcpyAsync for fill_tensor value staging failed");
    }

    gsx_cuda_fill_tensor_kernel_launch(
        (char*)cuda_buffer->ptr + tensor_view->offset_bytes,
        value_device_bytes,
        value_size_bytes,
        tensor_view->size_bytes,
        tensor_view->effective_alignment_bytes,
        cuda_backend->major_stream
    );
    cuda_err = cudaGetLastError();
    if(cuda_err != cudaSuccess) {
        cudaFree(value_device_bytes);
        return gsx_cuda_make_error(cuda_err, "fill_tensor kernel launch failed");
    }

    cuda_err = cudaFree(value_device_bytes);
    if(cuda_err != cudaSuccess) {
        return gsx_cuda_make_error(cuda_err, "cudaFree for fill_tensor value staging failed");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_buffer_fill_rand_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    uint64_t rng_state,
    uint64_t rng_inc
)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    gsx_backend_buffer_type_class buffer_type_class = gsx_cuda_backend_buffer_get_type_class(buffer);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t element_count = 0;

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view->buffer must match buffer");
    }

    error = gsx_cuda_backend_buffer_check_range(buffer, tensor_view->offset_bytes, tensor_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor_view->size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(tensor_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "rand fill only supports float32 tensors on cuda backend");
    }
    if(tensor_view->size_bytes % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "float32 tensor byte size must be divisible by sizeof(float)");
    }

    element_count = tensor_view->size_bytes / sizeof(float);

    if(buffer_type_class == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        float *values = (float *)((char *)cuda_buffer->ptr + tensor_view->offset_bytes);
        gsx_pcg32 rng = { 0 };
        rng.state = rng_state;
        rng.inc = rng_inc;
        for(gsx_size_t i = 0; i < element_count; ++i) {
            values[i] = pcg32_next_float(&rng);
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    gsx_cuda_fill_rand_tensor_f32_kernel_launch(
        (float *)((char *)cuda_buffer->ptr + tensor_view->offset_bytes),
        rng_state,
        rng_inc,
        element_count,
        cuda_backend->major_stream
    );
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_buffer_fill_randn_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    uint64_t rng_state,
    uint64_t rng_inc,
    gsx_float_t sigma
)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    gsx_backend_buffer_type_class buffer_type_class = gsx_cuda_backend_buffer_get_type_class(buffer);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t element_count = 0;

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view->buffer must match buffer");
    }

    error = gsx_cuda_backend_buffer_check_range(buffer, tensor_view->offset_bytes, tensor_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor_view->size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(tensor_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "randn fill only supports float32 tensors on cuda backend");
    }
    if(tensor_view->size_bytes % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "float32 tensor byte size must be divisible by sizeof(float)");
    }

    element_count = tensor_view->size_bytes / sizeof(float);

    if(buffer_type_class == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        float *values = (float *)((char *)cuda_buffer->ptr + tensor_view->offset_bytes);
        gsx_pcg32 rng = { 0 };
        rng.state = rng_state;
        rng.inc = rng_inc;
        for(gsx_size_t i = 0; i < element_count; i += 2) {
            float u1 = pcg32_next_float(&rng);
            float u2 = pcg32_next_float(&rng);
            float radius = 0.0f;
            float theta = 0.0f;
            float z0 = 0.0f;
            float z1 = 0.0f;

            if(u1 < 1e-7f) {
                u1 = 1e-7f;
            }
            radius = sqrtf(-2.0f * logf(u1));
            theta = 6.2831853071795864769f * u2;
            z0 = radius * cosf(theta);
            z1 = radius * sinf(theta);
            values[i] = z0 * sigma;
            if(i + 1 < element_count) {
                values[i + 1] = z1 * sigma;
            }
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    gsx_cuda_fill_randn_tensor_f32_kernel_launch(
        (float *)((char *)cuda_buffer->ptr + tensor_view->offset_bytes),
        rng_state,
        rng_inc,
        element_count,
        (float)sigma,
        cuda_backend->major_stream
    );
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_buffer_fill_randint_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    uint64_t rng_state,
    uint64_t rng_inc,
    uint32_t bound
)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    gsx_backend_buffer_type_class buffer_type_class = gsx_cuda_backend_buffer_get_type_class(buffer);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t element_count = 0;

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view->buffer must match buffer");
    }

    error = gsx_cuda_backend_buffer_check_range(buffer, tensor_view->offset_bytes, tensor_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor_view->size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(buffer_type_class == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        gsx_pcg32 rng = { 0 };
        rng.state = rng_state;
        rng.inc = rng_inc;
        switch(tensor_view->data_type) {
        case GSX_DATA_TYPE_U8: {
            uint8_t *values = (uint8_t *)((char *)cuda_buffer->ptr + tensor_view->offset_bytes);
            element_count = tensor_view->size_bytes / sizeof(uint8_t);
            for(gsx_size_t i = 0; i < element_count; ++i) {
                values[i] = (uint8_t)pcg32_next_uint_bound(&rng, bound);
            }
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        case GSX_DATA_TYPE_I32: {
            if(tensor_view->size_bytes % sizeof(int32_t) != 0) {
                return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "int32 tensor byte size must be divisible by sizeof(int32_t)");
            }
            int32_t *values = (int32_t *)((char *)cuda_buffer->ptr + tensor_view->offset_bytes);
            element_count = tensor_view->size_bytes / sizeof(int32_t);
            for(gsx_size_t i = 0; i < element_count; ++i) {
                values[i] = (int32_t)pcg32_next_uint_bound(&rng, bound);
            }
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        default:
            return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "randint fill only supports uint8 and int32 tensors on cuda backend");
        }
    }

    switch(tensor_view->data_type) {
    case GSX_DATA_TYPE_U8: {
        if(tensor_view->size_bytes % sizeof(uint8_t) != 0) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "uint8 tensor byte size must be divisible by sizeof(uint8_t)");
        }
        element_count = tensor_view->size_bytes / sizeof(uint8_t);
        gsx_cuda_fill_randint_tensor_u8_kernel_launch(
            (uint8_t *)((char *)cuda_buffer->ptr + tensor_view->offset_bytes),
            rng_state,
            rng_inc,
            element_count,
            bound,
            cuda_backend->major_stream
        );
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    case GSX_DATA_TYPE_I32: {
        if(tensor_view->size_bytes % sizeof(int32_t) != 0) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "int32 tensor byte size must be divisible by sizeof(int32_t)");
        }
        element_count = tensor_view->size_bytes / sizeof(int32_t);
        gsx_cuda_fill_randint_tensor_i32_kernel_launch(
            (int32_t *)((char *)cuda_buffer->ptr + tensor_view->offset_bytes),
            rng_state,
            rng_inc,
            element_count,
            bound,
            cuda_backend->major_stream
        );
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    default:
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "randint fill only supports uint8 and int32 tensors on cuda backend");
    }
}

static gsx_size_t gsx_cuda_backend_multinomial_upper_bound(const float *cdf_values, gsx_size_t category_count, float draw)
{
    gsx_size_t low = 0;
    gsx_size_t high = category_count;

    while(low < high) {
        gsx_size_t mid = low + (high - low) / 2u;

        if(draw < cdf_values[mid]) {
            high = mid;
        } else {
            low = mid + 1u;
        }
    }
    if(low >= category_count) {
        return category_count - 1u;
    }
    return low;
}

gsx_error gsx_cuda_backend_buffer_multinomial_tensor(
    gsx_backend_buffer_t out_buffer,
    const gsx_backend_tensor_view *out_view,
    const gsx_backend_tensor_view *cdf_view,
    uint64_t rng_state,
    uint64_t rng_inc
)
{
    gsx_cuda_backend_buffer *cuda_out_buffer = gsx_cuda_backend_buffer_from_base(out_buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(out_buffer->buffer_type->backend);
    gsx_backend_buffer_type_class out_type_class = gsx_cuda_backend_buffer_get_type_class(out_buffer);
    gsx_backend_buffer_type_class cdf_type_class = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_size_t sample_count = 0;
    gsx_size_t category_count = 0;

    if(out_view == NULL || cdf_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "multinomial tensor views must be non-null");
    }
    if(out_view->buffer != out_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_view->buffer must match out_buffer");
    }

    error = gsx_cuda_backend_buffer_check_range(out_buffer, out_view->offset_bytes, out_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_backend_buffer_check_range(cdf_view->buffer, cdf_view->offset_bytes, cdf_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(out_view->data_type != GSX_DATA_TYPE_I32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "multinomial output only supports int32 tensors on cuda backend");
    }
    if(cdf_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "multinomial cdf only supports float32 tensors on cuda backend");
    }
    if(out_view->size_bytes % sizeof(int32_t) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "int32 output tensor byte size must be divisible by sizeof(int32_t)");
    }
    if(cdf_view->size_bytes % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "float32 cdf tensor byte size must be divisible by sizeof(float)");
    }

    sample_count = out_view->size_bytes / sizeof(int32_t);
    category_count = cdf_view->size_bytes / sizeof(float);
    if(sample_count == 0 || category_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(category_count > (gsx_size_t)INT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "multinomial category count exceeds int32 output range");
    }

    cdf_type_class = gsx_cuda_backend_buffer_get_type_class(cdf_view->buffer);
    if(out_type_class == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED || cdf_type_class == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        const float *cdf_values = NULL;
        int32_t *out_values = NULL;
        gsx_pcg32 rng = { 0 };
        gsx_size_t index = 0;
        float total_mass = 0.0f;

        if(out_type_class != GSX_BACKEND_BUFFER_TYPE_HOST_PINNED || cdf_type_class != GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
            return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda multinomial does not support mixed host-pinned and device tensor placement");
        }

        cdf_values = (const float *)((const char *)gsx_cuda_backend_buffer_from_base(cdf_view->buffer)->ptr + cdf_view->offset_bytes);
        out_values = (int32_t *)((char *)cuda_out_buffer->ptr + out_view->offset_bytes);
        total_mass = cdf_values[category_count - 1u];
        rng.state = rng_state;
        rng.inc = rng_inc;
        for(index = 0; index < sample_count; ++index) {
            float draw = pcg32_next_float(&rng) * total_mass;
            gsx_size_t picked = gsx_cuda_backend_multinomial_upper_bound(cdf_values, category_count, draw);

            out_values[index] = (int32_t)picked;
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    gsx_cuda_multinomial_tensor_i32_kernel_launch(
        (int32_t *)((char *)cuda_out_buffer->ptr + out_view->offset_bytes),
        (const float *)((const char *)gsx_cuda_backend_buffer_from_base(cdf_view->buffer)->ptr + cdf_view->offset_bytes),
        rng_state,
        rng_inc,
        sample_count,
        category_count,
        cuda_backend->major_stream
    );
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_buffer_check_finite_tensor(gsx_backend_buffer_t buffer, const gsx_backend_tensor_view *tensor_view, bool *out_is_finite)
{
    gsx_cuda_backend_buffer *cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    gsx_cuda_backend *cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
    gsx_backend_buffer_type_class buffer_type_class = gsx_cuda_backend_buffer_get_type_class(buffer);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    void *has_non_finite_dev_ptr = NULL;
    int has_non_finite_host = 0;
    cudaError_t cuda_err = cudaSuccess;
    gsx_size_t element_count = 0;
    gsx_size_t element_size = 0;

    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view->buffer must match buffer");
    }
    if(out_is_finite == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_is_finite must be non-null");
    }
    *out_is_finite = true;

    error = gsx_cuda_backend_buffer_check_range(buffer, tensor_view->offset_bytes, tensor_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor_view->size_bytes == 0) {
        *out_is_finite = true;
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    switch(tensor_view->data_type) {
    case GSX_DATA_TYPE_F32:
        element_size = 4;
        break;
    case GSX_DATA_TYPE_F16:
        element_size = 2;
        break;
    default:
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "check_finite only supports floating point types");
    }
    if(tensor_view->size_bytes % element_size != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor byte size must be a multiple of the checked element size");
    }

    if(buffer_type_class == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        gsx_size_t element_index = 0;

        if(tensor_view->data_type == GSX_DATA_TYPE_F32) {
            const float *values = (const float *)((const char *)cuda_buffer->ptr + tensor_view->offset_bytes);

            element_count = tensor_view->size_bytes / sizeof(float);
            for(element_index = 0; element_index < element_count; ++element_index) {
                if(!isfinite((double)values[element_index])) {
                    *out_is_finite = false;
                    break;
                }
            }
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }

        {
            const uint16_t *values = (const uint16_t *)((const char *)cuda_buffer->ptr + tensor_view->offset_bytes);

            element_count = tensor_view->size_bytes / element_size;
            for(element_index = 0; element_index < element_count; ++element_index) {
                bool is_value_finite = gsx_cuda_backend_f16_is_finite(values[element_index]);

                if(!is_value_finite) {
                    *out_is_finite = false;
                    break;
                }
            }
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
    }

    element_count = tensor_view->size_bytes / element_size;

    cuda_err = cudaMalloc(&has_non_finite_dev_ptr, sizeof(int));
    if(cuda_err != cudaSuccess) {
        return gsx_cuda_make_error(cuda_err, "cudaMalloc for check_finite flag failed");
    }

    cuda_err = cudaMemsetAsync(has_non_finite_dev_ptr, 0, sizeof(int), cuda_backend->major_stream);
    if(cuda_err != cudaSuccess) {
        cudaFree(has_non_finite_dev_ptr);
        return gsx_cuda_make_error(cuda_err, "cudaMemsetAsync for check_finite flag failed");
    }

    switch(tensor_view->data_type) {
    case GSX_DATA_TYPE_F32:
        gsx_cuda_check_finite_tensor_f32_kernel_launch(
            (const char*)cuda_buffer->ptr + tensor_view->offset_bytes,
            element_count,
            tensor_view->effective_alignment_bytes,
            (int*)has_non_finite_dev_ptr,
            cuda_backend->major_stream
        );
        break;
    case GSX_DATA_TYPE_F16:
        gsx_cuda_check_finite_tensor_f16_kernel_launch(
            (const char*)cuda_buffer->ptr + tensor_view->offset_bytes,
            element_count,
            tensor_view->effective_alignment_bytes,
            (int*)has_non_finite_dev_ptr,
            cuda_backend->major_stream
        );
        break;
    default:
        cudaFree(has_non_finite_dev_ptr);
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "check_finite only supports floating point types");
    }
    cuda_err = cudaGetLastError();
    if(cuda_err != cudaSuccess) {
        cudaFree(has_non_finite_dev_ptr);
        return gsx_cuda_make_error(cuda_err, "check_finite kernel launch failed");
    }

    cuda_err = cudaMemcpyAsync(&has_non_finite_host, has_non_finite_dev_ptr, sizeof(int), cudaMemcpyDeviceToHost, cuda_backend->major_stream);
    if(cuda_err != cudaSuccess) {
        cudaFree(has_non_finite_dev_ptr);
        return gsx_cuda_make_error(cuda_err, "cudaMemcpyAsync for check_finite result failed");
    }

    /* Synchronization is intentional so the finite-check result is deterministic and host-visible at return. */
    cuda_err = cudaStreamSynchronize(cuda_backend->major_stream);
    cudaFree(has_non_finite_dev_ptr);
    if(cuda_err != cudaSuccess) {
        return gsx_cuda_make_error(cuda_err, "cudaStreamSynchronize for check_finite failed");
    }

    *out_is_finite = (has_non_finite_host == 0);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_buffer_gather_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *index_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape
)
{
    gsx_cuda_backend_buffer *x_buffer = NULL;
    gsx_cuda_backend_buffer *index_buffer = NULL;
    gsx_cuda_backend_buffer *out_buffer = NULL;
    gsx_cuda_backend *cuda_backend = NULL;
    gsx_backend_buffer_type_class x_buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_backend_buffer_type_class index_buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_backend_buffer_type_class out_buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_size_t expected_x_bytes = 0;
    gsx_size_t expected_index_bytes = 0;
    gsx_size_t x_row_bytes = 0;
    gsx_size_t row_count = 0;
    gsx_size_t row_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    cudaError_t cuda_err = cudaSuccess;
    int out_of_range_host = 0;
    int *out_of_range_dev = NULL;
    const void *x_data = NULL;
    const void *index_data = NULL;
    void *out_data = NULL;

    if(dst_buffer == NULL || x_view == NULL || index_view == NULL || out_view == NULL || x_shape == NULL || out_shape == NULL
        || x_view->buffer == NULL || index_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer, tensor views, and shapes must be non-null");
    }
    if(out_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_view must reference dst_buffer");
    }
    if(x_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend
        || index_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "all gather tensors must belong to the same backend");
    }

    error = gsx_cuda_backend_buffer_check_range(x_view->buffer, x_view->offset_bytes, x_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_backend_buffer_check_range(index_view->buffer, index_view->offset_bytes, index_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_backend_buffer_check_range(dst_buffer, out_view->offset_bytes, out_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    row_count = (gsx_size_t)out_shape[0];
    if(row_count == 0 || x_rank < 1 || out_rank < 1) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "gather shape metadata is invalid");
    }
    if(out_view->size_bytes % row_count != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "gather output view byte size must be row-aligned");
    }
    x_row_bytes = out_view->size_bytes / row_count;
    if(gsx_size_mul_overflows((gsx_size_t)x_shape[0], x_row_bytes, &expected_x_bytes) || expected_x_bytes != x_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "gather x view byte size is inconsistent with row metadata");
    }
    if(gsx_size_mul_overflows(row_count, sizeof(int32_t), &expected_index_bytes) || expected_index_bytes != index_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "index view byte size must match out leading dimension");
    }

    if(row_count == 0 || x_row_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    x_buffer = gsx_cuda_backend_buffer_from_base(x_view->buffer);
    index_buffer = gsx_cuda_backend_buffer_from_base(index_view->buffer);
    out_buffer = gsx_cuda_backend_buffer_from_base(dst_buffer);
    cuda_backend = gsx_cuda_backend_from_base(dst_buffer->buffer_type->backend);
    x_buffer_type = gsx_cuda_backend_buffer_get_type_class(x_view->buffer);
    index_buffer_type = gsx_cuda_backend_buffer_get_type_class(index_view->buffer);
    out_buffer_type = gsx_cuda_backend_buffer_get_type_class(dst_buffer);
    x_data = (const char*)x_buffer->ptr + x_view->offset_bytes;
    index_data = (const char*)index_buffer->ptr + index_view->offset_bytes;
    out_data = (char*)out_buffer->ptr + out_view->offset_bytes;

    if(x_buffer_type == GSX_BACKEND_BUFFER_TYPE_DEVICE
        && index_buffer_type == GSX_BACKEND_BUFFER_TYPE_DEVICE
        && out_buffer_type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        cuda_err = cudaMalloc((void**) (&out_of_range_dev), sizeof(int));   // TODO: preallocate it at backend initialization time.
        if(cuda_err != cudaSuccess) {
            return gsx_cuda_make_error(cuda_err, "cudaMalloc for gather out-of-range flag failed");
        }
        cuda_err = cudaMemsetAsync(out_of_range_dev, 0, sizeof(int), cuda_backend->major_stream);
        if(cuda_err != cudaSuccess) {
            cudaFree(out_of_range_dev);
            return gsx_cuda_make_error(cuda_err, "cudaMemsetAsync for gather out-of-range flag failed");
        }
        cuda_err = gsx_cuda_gather_rows_kernel_launch(
            x_data,
            out_data,
            x_row_bytes,
            row_count,
            (const int32_t *)index_data,
            (gsx_size_t)x_shape[0],
            out_of_range_dev,
            cuda_backend->major_stream
        );
        if(cuda_err != cudaSuccess) {
            cudaFree(out_of_range_dev);
            return gsx_cuda_make_error(cuda_err, "gather_tensor kernel launch failed");
        }
        cuda_err = cudaMemcpyAsync(&out_of_range_host, out_of_range_dev, sizeof(int), cudaMemcpyDeviceToHost, cuda_backend->major_stream);
        if(cuda_err != cudaSuccess) {
            cudaFree(out_of_range_dev);
            return gsx_cuda_make_error(cuda_err, "cudaMemcpyAsync for gather out-of-range flag failed");
        }
        cuda_err = cudaStreamSynchronize(cuda_backend->major_stream); // must synchronize to check out_of_range_host.
        if(cuda_err != cudaSuccess) {
            cudaFree(out_of_range_dev);
            return gsx_cuda_make_error(cuda_err, "cudaStreamSynchronize for gather failed");
        }
        cuda_err = cudaFree(out_of_range_dev);
        if(cuda_err != cudaSuccess) {
            return gsx_cuda_make_error(cuda_err, "cudaFree for gather out-of-range flag failed");
        }
        if(out_of_range_host != 0) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "gather index is out of range");
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(x_buffer_type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED
        && index_buffer_type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED
        && out_buffer_type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        const unsigned char *x_bytes = (const unsigned char *)x_data;
        const int32_t *indices_host = (const int32_t *)index_data;
        unsigned char *out_bytes = (unsigned char *)out_data;

        for(row_index = 0; row_index < row_count; ++row_index) {
            int32_t src_row = indices_host[row_index];
            if(src_row < 0 || src_row >= x_shape[0]) {
                return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "gather index is out of range");
            }
            memcpy(
                out_bytes + row_index * x_row_bytes,
                x_bytes + (gsx_size_t)src_row * x_row_bytes,
                x_row_bytes
            );
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "gather_tensor requires all tensors to use the same CUDA buffer type");
}

static gsx_error gsx_cuda_backend_buffer_apply_unary_tensor_f32(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_impl_unary_op op
)
{
    gsx_cuda_backend_buffer *x_buffer = NULL;
    gsx_cuda_backend_buffer *out_buffer = NULL;
    gsx_cuda_backend *cuda_backend = NULL;
    gsx_backend_buffer_type_class x_buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_backend_buffer_type_class out_buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_size_t element_count = 0;
    gsx_size_t element_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    cudaError_t cuda_err = cudaSuccess;
    const float *x_values = NULL;
    float *out_values = NULL;

    (void)rank;
    (void)shape;

    if(dst_buffer == NULL || x_view == NULL || out_view == NULL || shape == NULL || x_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer, tensor views, and shape must be non-null");
    }
    if(out_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_view must reference dst_buffer");
    }
    if(x_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out must belong to the same backend");
    }
    error = gsx_cuda_backend_buffer_check_range(x_view->buffer, x_view->offset_bytes, x_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_backend_buffer_check_range(dst_buffer, out_view->offset_bytes, out_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(x_view->size_bytes != out_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out view byte sizes must match");
    }
    if(x_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "unary tensor op only supports float32 tensors on cuda backend");
    }
    if(x_view->size_bytes % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "float32 tensor byte size must be divisible by sizeof(float)");
    }

    x_buffer = gsx_cuda_backend_buffer_from_base(x_view->buffer);
    out_buffer = gsx_cuda_backend_buffer_from_base(dst_buffer);
    x_buffer_type = gsx_cuda_backend_buffer_get_type_class(x_view->buffer);
    out_buffer_type = gsx_cuda_backend_buffer_get_type_class(dst_buffer);
    x_values = (const float *)((const char*)x_buffer->ptr + x_view->offset_bytes);
    out_values = (float *)((char*)out_buffer->ptr + out_view->offset_bytes);
    element_count = x_view->size_bytes / sizeof(float);

    if(x_buffer_type == GSX_BACKEND_BUFFER_TYPE_DEVICE && out_buffer_type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        cuda_backend = gsx_cuda_backend_from_base(dst_buffer->buffer_type->backend);
        switch(op) {
        case GSX_IMPL_UNARY_OP_EXP:
            cuda_err = gsx_cuda_exp_tensor_f32_kernel_launch(x_values, out_values, element_count, cuda_backend->major_stream);
            break;
        case GSX_IMPL_UNARY_OP_SIGMOID:
            cuda_err = gsx_cuda_sigmoid_tensor_f32_kernel_launch(x_values, out_values, element_count, cuda_backend->major_stream);
            break;
        case GSX_IMPL_UNARY_OP_SIGMOID_DERIVATIVE:
            cuda_err = gsx_cuda_sigmoid_derivative_tensor_f32_kernel_launch(x_values, out_values, element_count, cuda_backend->major_stream);
            break;
        case GSX_IMPL_UNARY_OP_ABS:
            cuda_err = gsx_cuda_abs_tensor_f32_kernel_launch(x_values, out_values, element_count, cuda_backend->major_stream);
            break;
        default:
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown unary tensor op");
        }
        if(cuda_err != cudaSuccess) {
            return gsx_cuda_make_error(cuda_err, "unary tensor kernel launch failed");
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(x_buffer_type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED && out_buffer_type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        for(element_index = 0; element_index < element_count; ++element_index) {
            float x_value = x_values[element_index];
            switch(op) {
            case GSX_IMPL_UNARY_OP_EXP:
                out_values[element_index] = expf(x_value);
                break;
            case GSX_IMPL_UNARY_OP_SIGMOID:
                out_values[element_index] = 1.0f / (1.0f + expf(-x_value));
                break;
            case GSX_IMPL_UNARY_OP_SIGMOID_DERIVATIVE: {
                float sigmoid_value = 1.0f / (1.0f + expf(-x_value));
                out_values[element_index] = sigmoid_value * (1.0f - sigmoid_value);
                break;
            }
            case GSX_IMPL_UNARY_OP_ABS:
                out_values[element_index] = fabsf(x_value);
                break;
            default:
                return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown unary tensor op");
            }
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "unary tensor op requires x and out to use the same CUDA buffer type");
}

static gsx_error gsx_cuda_backend_buffer_apply_unary_inplace_tensor_f32(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    gsx_impl_unary_op op
)
{
    gsx_cuda_backend_buffer *cuda_buffer = NULL;
    gsx_cuda_backend *cuda_backend = NULL;
    gsx_backend_buffer_type_class buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_size_t element_count = 0;
    gsx_size_t element_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    cudaError_t cuda_err = cudaSuccess;
    float *values = NULL;

    if(buffer == NULL || tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer and tensor_view must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must reference buffer");
    }
    if(tensor_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "unary tensor op only supports float32 tensors on cuda backend");
    }

    error = gsx_cuda_backend_buffer_check_range(buffer, tensor_view->offset_bytes, tensor_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    buffer_type = gsx_cuda_backend_buffer_get_type_class(buffer);
    values = (float *)((char*)cuda_buffer->ptr + tensor_view->offset_bytes);
    element_count = tensor_view->size_bytes / sizeof(float);

    if(buffer_type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
        switch(op) {
        case GSX_IMPL_UNARY_OP_EXP:
            cuda_err = gsx_cuda_exp_inplace_tensor_f32_kernel_launch(values, element_count, cuda_backend->major_stream);
            break;
        case GSX_IMPL_UNARY_OP_SIGMOID:
            cuda_err = gsx_cuda_sigmoid_inplace_tensor_f32_kernel_launch(values, element_count, cuda_backend->major_stream);
            break;
        case GSX_IMPL_UNARY_OP_SIGMOID_DERIVATIVE:
            cuda_err = gsx_cuda_sigmoid_derivative_inplace_tensor_f32_kernel_launch(values, element_count, cuda_backend->major_stream);
            break;
        case GSX_IMPL_UNARY_OP_ABS:
            cuda_err = gsx_cuda_abs_inplace_tensor_f32_kernel_launch(values, element_count, cuda_backend->major_stream);
            break;
        default:
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown unary tensor op");
        }
        if(cuda_err != cudaSuccess) {
            return gsx_cuda_make_error(cuda_err, "unary inplace tensor kernel launch failed");
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(buffer_type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        for(element_index = 0; element_index < element_count; ++element_index) {
            float x_value = values[element_index];
            switch(op) {
            case GSX_IMPL_UNARY_OP_EXP:
                values[element_index] = expf(x_value);
                break;
            case GSX_IMPL_UNARY_OP_SIGMOID:
                values[element_index] = 1.0f / (1.0f + expf(-x_value));
                break;
            case GSX_IMPL_UNARY_OP_SIGMOID_DERIVATIVE: {
                float sigmoid_value = 1.0f / (1.0f + expf(-x_value));
                values[element_index] = sigmoid_value * (1.0f - sigmoid_value);
                break;
            }
            case GSX_IMPL_UNARY_OP_ABS:
                values[element_index] = fabsf(x_value);
                break;
            default:
                return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown unary tensor op");
            }
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "unary inplace tensor op does not support this CUDA buffer type");
}

gsx_error gsx_cuda_backend_buffer_unary_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_impl_unary_op op
)
{
    return gsx_cuda_backend_buffer_apply_unary_tensor_f32(dst_buffer, x_view, out_view, rank, shape, op);
}

gsx_error gsx_cuda_backend_buffer_unary_tensor_inplace(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    gsx_impl_unary_op op
)
{
    return gsx_cuda_backend_buffer_apply_unary_inplace_tensor_f32(buffer, tensor_view, op);
}

static gsx_error gsx_cuda_backend_reduce_validate_shape_contract(
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_size_t *out_outer_count,
    gsx_size_t *out_reduce_count
)
{
    gsx_index_t dim = 0;
    gsx_size_t outer_count = 1;
    gsx_size_t reduce_count = 1;
    gsx_size_t x_element_count = 1;
    gsx_size_t out_element_count = 1;
    gsx_size_t expected_x_bytes = 0;
    gsx_size_t expected_out_bytes = 0;

    if(x_view == NULL || out_view == NULL || x_shape == NULL || out_shape == NULL || out_outer_count == NULL
        || out_reduce_count == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "reduce shape inputs must be non-null");
    }
    if(start_axis < 0 || start_axis >= x_rank) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "start_axis must be in range [0, x_rank)");
    }
    if(out_rank != start_axis + 1) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_rank must equal start_axis + 1");
    }
    if(out_shape[start_axis] != 1) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out reduced axis extent must be 1");
    }
    for(dim = 0; dim < start_axis; ++dim) {
        if(x_shape[dim] != out_shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out prefix shape must match x prefix shape");
        }
    }
    for(dim = 0; dim < x_rank; ++dim) {
        if(gsx_size_mul_overflows(x_element_count, (gsx_size_t)x_shape[dim], &x_element_count)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "x element count overflows");
        }
    }
    for(dim = 0; dim < out_rank; ++dim) {
        if(gsx_size_mul_overflows(out_element_count, (gsx_size_t)out_shape[dim], &out_element_count)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "out element count overflows");
        }
    }
    for(dim = 0; dim < start_axis; ++dim) {
        if(gsx_size_mul_overflows(outer_count, (gsx_size_t)x_shape[dim], &outer_count)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "outer_count overflows");
        }
    }
    for(dim = start_axis; dim < x_rank; ++dim) {
        if(gsx_size_mul_overflows(reduce_count, (gsx_size_t)x_shape[dim], &reduce_count)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "reduce_count overflows");
        }
    }
    if(gsx_size_mul_overflows(x_element_count, sizeof(float), &expected_x_bytes)
        || gsx_size_mul_overflows(out_element_count, sizeof(float), &expected_out_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "reduce tensor byte size overflows");
    }
    if((x_view->size_bytes != 0 || out_view->size_bytes != 0)
        && (expected_x_bytes != x_view->size_bytes || expected_out_bytes != out_view->size_bytes)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor views do not match provided reduce shape metadata");
    }
    *out_outer_count = outer_count;
    *out_reduce_count = reduce_count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_query_unary_reduce_workspace_size(
    gsx_backend_t backend,
    gsx_backend_buffer_type_class workspace_buffer_type,
    gsx_data_type data_type,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_impl_unary_reduce_op op,
    gsx_size_t *out_workspace_size_bytes,
    gsx_size_t *out_workspace_alignment_bytes
)
{
    gsx_backend_tensor_view x_view = { 0 };
    gsx_backend_tensor_view out_view = { 0 };
    gsx_size_t outer_count = 1;
    gsx_size_t reduce_count = 1;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    cudaError_t cuda_err = cudaSuccess;

    if(backend == NULL || x_shape == NULL || out_shape == NULL || out_workspace_size_bytes == NULL
        || out_workspace_alignment_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "reduce workspace query inputs must be non-null");
    }
    if(workspace_buffer_type != GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda reduce requires device workspace buffer type");
    }
    if(data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda unary_reduce only supports float32 tensors");
    }
    switch(op) {
    case GSX_IMPL_UNARY_REDUCE_OP_SUM:
    case GSX_IMPL_UNARY_REDUCE_OP_MEAN:
    case GSX_IMPL_UNARY_REDUCE_OP_MAX:
        break;
    default:
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown unary_reduce op");
    }
    x_view.data_type = data_type;
    out_view.data_type = data_type;
    x_view.size_bytes = 0;
    out_view.size_bytes = 0;
    error = gsx_cuda_backend_reduce_validate_shape_contract(
        &x_view,
        &out_view,
        x_rank,
        x_shape,
        out_rank,
        out_shape,
        start_axis,
        &outer_count,
        &reduce_count
    );
    if(!gsx_error_is_success(error)) {
        return error;
    }
    cuda_err = gsx_cuda_unary_reduce_workspace_size_query(reduce_count, op, out_workspace_size_bytes);
    if(cuda_err != cudaSuccess) {
        switch(op) {
        case GSX_IMPL_UNARY_REDUCE_OP_SUM:
            return gsx_cuda_make_error(cuda_err, "cuda unary_reduce sum workspace query failed");
        case GSX_IMPL_UNARY_REDUCE_OP_MEAN:
            return gsx_cuda_make_error(cuda_err, "cuda unary_reduce mean workspace query failed");
        case GSX_IMPL_UNARY_REDUCE_OP_MAX:
            return gsx_cuda_make_error(cuda_err, "cuda unary_reduce max workspace query failed");
        default:
            return gsx_cuda_make_error(cuda_err, "cuda unary_reduce workspace query failed");
        }
    }
    *out_workspace_alignment_bytes = 256;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_query_binary_reduce_workspace_size(
    gsx_backend_t backend,
    gsx_backend_buffer_type_class workspace_buffer_type,
    gsx_data_type data_type,
    gsx_index_t lhs_rank,
    const gsx_index_t *lhs_shape,
    gsx_index_t rhs_rank,
    const gsx_index_t *rhs_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_impl_binary_reduce_op op,
    gsx_size_t *out_workspace_size_bytes,
    gsx_size_t *out_workspace_alignment_bytes
)
{
    gsx_backend_tensor_view lhs_view = { 0 };
    gsx_backend_tensor_view rhs_view = { 0 };
    gsx_backend_tensor_view out_view = { 0 };
    gsx_size_t outer_count_lhs = 1;
    gsx_size_t reduce_count_lhs = 1;
    gsx_size_t outer_count_rhs = 0;
    gsx_size_t reduce_count_rhs = 0;
    gsx_size_t lhs_elements = 1;
    gsx_size_t rhs_elements = 1;
    gsx_size_t out_elements = 1;
    gsx_index_t dim = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    cudaError_t cuda_err = cudaSuccess;

    if(backend == NULL || lhs_shape == NULL || rhs_shape == NULL || out_shape == NULL || out_workspace_size_bytes == NULL
        || out_workspace_alignment_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "binary reduce workspace query inputs must be non-null");
    }
    if(workspace_buffer_type != GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda reduce requires device workspace buffer type");
    }
    if(data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda binary_reduce only supports float32 tensors");
    }
    switch(op) {
    case GSX_IMPL_BINARY_REDUCE_OP_MSE:
    case GSX_IMPL_BINARY_REDUCE_OP_MAE:
        break;
    default:
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "unknown binary_reduce op");
    }
    if(rhs_rank != lhs_rank) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "lhs_rank and rhs_rank must match");
    }
    for(dim = 0; dim < lhs_rank; ++dim) {
        if(lhs_shape[dim] != rhs_shape[dim]) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "lhs and rhs shape must match");
        }
        if(gsx_size_mul_overflows(lhs_elements, (gsx_size_t)lhs_shape[dim], &lhs_elements)
            || gsx_size_mul_overflows(rhs_elements, (gsx_size_t)rhs_shape[dim], &rhs_elements)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "binary reduce element count overflows");
        }
    }
    for(dim = 0; dim < out_rank; ++dim) {
        if(gsx_size_mul_overflows(out_elements, (gsx_size_t)out_shape[dim], &out_elements)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "out element count overflows");
        }
    }
    lhs_view.data_type = data_type;
    rhs_view.data_type = data_type;
    out_view.data_type = data_type;
    if(gsx_size_mul_overflows(lhs_elements, sizeof(float), &lhs_view.size_bytes)
        || gsx_size_mul_overflows(rhs_elements, sizeof(float), &rhs_view.size_bytes)
        || gsx_size_mul_overflows(out_elements, sizeof(float), &out_view.size_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "reduce tensor byte size overflows");
    }
    error = gsx_cuda_backend_reduce_validate_shape_contract(
        &lhs_view,
        &out_view,
        lhs_rank,
        lhs_shape,
        out_rank,
        out_shape,
        start_axis,
        &outer_count_lhs,
        &reduce_count_lhs
    );
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_backend_reduce_validate_shape_contract(
        &rhs_view,
        &out_view,
        rhs_rank,
        rhs_shape,
        out_rank,
        out_shape,
        start_axis,
        &outer_count_rhs,
        &reduce_count_rhs
    );
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(outer_count_lhs != outer_count_rhs || reduce_count_lhs != reduce_count_rhs) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "lhs and rhs reduce metadata must match");
    }
    cuda_err = gsx_cuda_binary_reduce_workspace_size_query(reduce_count_lhs, op, out_workspace_size_bytes);
    if(cuda_err != cudaSuccess) {
        switch(op) {
        case GSX_IMPL_BINARY_REDUCE_OP_MSE:
            return gsx_cuda_make_error(cuda_err, "cuda binary_reduce mse workspace query failed");
        case GSX_IMPL_BINARY_REDUCE_OP_MAE:
            return gsx_cuda_make_error(cuda_err, "cuda binary_reduce mae workspace query failed");
        default:
            return gsx_cuda_make_error(cuda_err, "cuda binary_reduce workspace query failed");
        }
    }
    *out_workspace_alignment_bytes = 256;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_cuda_backend_buffer_unary_reduce_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_backend_tensor_view *workspace_view,
    gsx_index_t x_rank,
    const gsx_index_t *x_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_impl_unary_reduce_op op
)
{
    gsx_cuda_backend_buffer *x_buffer = NULL;
    gsx_cuda_backend_buffer *out_buffer = NULL;
    gsx_cuda_backend_buffer *workspace_buffer = NULL;
    gsx_cuda_backend *cuda_backend = NULL;
    gsx_backend_buffer_type_class x_buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_backend_buffer_type_class out_buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_backend_buffer_type_class workspace_buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    const float *x_values = NULL;
    float *out_values = NULL;
    void *workspace_ptr = NULL;
    gsx_size_t outer_count = 1;
    gsx_size_t reduce_count = 1;
    gsx_size_t expected_out_bytes = 0;
    gsx_size_t expected_input_elements = 0;
    gsx_size_t expected_input_bytes = 0;
    gsx_index_t dim = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    cudaError_t cuda_err = cudaSuccess;

    if(dst_buffer == NULL || x_view == NULL || out_view == NULL || workspace_view == NULL || x_shape == NULL || out_shape == NULL
        || x_view->buffer == NULL || out_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "reduce buffers, views, and shapes must be non-null");
    }
    if(out_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_view must reference dst_buffer");
    }
    if(x_view->data_type != GSX_DATA_TYPE_F32 || out_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda unary_reduce only supports float32 tensors");
    }
    if(x_view->data_type != out_view->data_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x_view and out_view data_type must match");
    }
    error = gsx_cuda_backend_buffer_check_range(x_view->buffer, x_view->offset_bytes, x_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_backend_buffer_check_range(dst_buffer, out_view->offset_bytes, out_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(workspace_view->buffer != NULL) {
        error = gsx_cuda_backend_buffer_check_range(
            workspace_view->buffer, workspace_view->offset_bytes, workspace_view->size_bytes);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    } else if(workspace_view->offset_bytes != 0 || workspace_view->size_bytes != 0) {
        return gsx_make_error(
            GSX_ERROR_INVALID_ARGUMENT, "workspace view must have zero offset/size when workspace buffer is null");
    }
    if(x_rank < 1 || out_rank < 1 || start_axis < 0 || start_axis >= x_rank) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "reduce shape metadata is invalid");
    }
    for(dim = 0; dim < start_axis; ++dim) {
        if(gsx_size_mul_overflows(outer_count, (gsx_size_t)x_shape[dim], &outer_count)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "outer_count overflows");
        }
    }
    for(dim = start_axis; dim < x_rank; ++dim) {
        if(gsx_size_mul_overflows(reduce_count, (gsx_size_t)x_shape[dim], &reduce_count)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "reduce_count overflows");
        }
    }
    if(gsx_size_mul_overflows(outer_count, sizeof(float), &expected_out_bytes)
        || gsx_size_mul_overflows(outer_count, reduce_count, &expected_input_elements)
        || gsx_size_mul_overflows(expected_input_elements, sizeof(float), &expected_input_bytes)
        || out_view->size_bytes != expected_out_bytes
        || x_view->size_bytes != expected_input_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "reduce tensor views are inconsistent with dispatch metadata");
    }

    x_buffer_type = gsx_cuda_backend_buffer_get_type_class(x_view->buffer);
    out_buffer_type = gsx_cuda_backend_buffer_get_type_class(dst_buffer);
    if(workspace_view->buffer != NULL) {
        workspace_buffer_type = gsx_cuda_backend_buffer_get_type_class(workspace_view->buffer);
    }
    if(x_buffer_type != GSX_BACKEND_BUFFER_TYPE_DEVICE || out_buffer_type != GSX_BACKEND_BUFFER_TYPE_DEVICE
        || workspace_buffer_type != GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda unary_reduce only supports device buffers");
    }

    x_buffer = gsx_cuda_backend_buffer_from_base(x_view->buffer);
    out_buffer = gsx_cuda_backend_buffer_from_base(dst_buffer);
    if(workspace_view->buffer != NULL) {
        workspace_buffer = gsx_cuda_backend_buffer_from_base(workspace_view->buffer);
    }
    cuda_backend = gsx_cuda_backend_from_base(dst_buffer->buffer_type->backend);
    x_values = (const float *)((const char*)x_buffer->ptr + x_view->offset_bytes);
    out_values = (float *)((char*)out_buffer->ptr + out_view->offset_bytes);
    if(workspace_buffer != NULL) {
        workspace_ptr = (void *)((char*)workspace_buffer->ptr + workspace_view->offset_bytes);
    }

    cuda_err = gsx_cuda_unary_reduce_f32_launch(
        x_values,
        out_values,
        workspace_ptr,
        workspace_view->size_bytes,
        outer_count,
        reduce_count,
        op,
        cuda_backend->major_stream
    );
    return gsx_cuda_make_error(cuda_err, "cuda unary_reduce launch failed");
}

gsx_error gsx_cuda_backend_buffer_binary_reduce_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *lhs_view,
    const gsx_backend_tensor_view *rhs_view,
    const gsx_backend_tensor_view *out_view,
    const gsx_backend_tensor_view *workspace_view,
    gsx_index_t lhs_rank,
    const gsx_index_t *lhs_shape,
    gsx_index_t rhs_rank,
    const gsx_index_t *rhs_shape,
    gsx_index_t out_rank,
    const gsx_index_t *out_shape,
    gsx_index_t start_axis,
    gsx_impl_binary_reduce_op op
)
{
    gsx_cuda_backend_buffer *lhs_buffer = NULL;
    gsx_cuda_backend_buffer *rhs_buffer = NULL;
    gsx_cuda_backend_buffer *out_buffer = NULL;
    gsx_cuda_backend_buffer *workspace_buffer = NULL;
    gsx_cuda_backend *cuda_backend = NULL;
    gsx_backend_buffer_type_class lhs_buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_backend_buffer_type_class rhs_buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_backend_buffer_type_class out_buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_backend_buffer_type_class workspace_buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    const float *lhs_values = NULL;
    const float *rhs_values = NULL;
    float *out_values = NULL;
    void *workspace_ptr = NULL;
    gsx_size_t outer_count_lhs = 1;
    gsx_size_t reduce_count_lhs = 1;
    gsx_size_t expected_out_bytes = 0;
    gsx_size_t expected_input_elements = 0;
    gsx_size_t expected_input_bytes = 0;
    gsx_index_t dim = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    cudaError_t cuda_err = cudaSuccess;

    if(dst_buffer == NULL || lhs_view == NULL || rhs_view == NULL || out_view == NULL || workspace_view == NULL || lhs_shape == NULL
        || rhs_shape == NULL || out_shape == NULL || lhs_view->buffer == NULL || rhs_view->buffer == NULL
        || out_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "binary_reduce buffers, views, and shapes must be non-null");
    }
    if(rhs_rank != lhs_rank) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "lhs_rank and rhs_rank must match");
    }
    if(out_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_view must reference dst_buffer");
    }
    if(lhs_view->data_type != GSX_DATA_TYPE_F32 || rhs_view->data_type != GSX_DATA_TYPE_F32
        || out_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda binary_reduce only supports float32 tensors");
    }
    if(lhs_view->data_type != rhs_view->data_type || lhs_view->data_type != out_view->data_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "binary_reduce tensor data_type must match");
    }
    error = gsx_cuda_backend_buffer_check_range(lhs_view->buffer, lhs_view->offset_bytes, lhs_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_backend_buffer_check_range(rhs_view->buffer, rhs_view->offset_bytes, rhs_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_backend_buffer_check_range(dst_buffer, out_view->offset_bytes, out_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(workspace_view->buffer != NULL) {
        error = gsx_cuda_backend_buffer_check_range(
            workspace_view->buffer, workspace_view->offset_bytes, workspace_view->size_bytes);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    } else if(workspace_view->offset_bytes != 0 || workspace_view->size_bytes != 0) {
        return gsx_make_error(
            GSX_ERROR_INVALID_ARGUMENT, "workspace view must have zero offset/size when workspace buffer is null");
    }
    if(lhs_rank < 1 || rhs_rank < 1 || out_rank < 1 || rhs_rank != lhs_rank || start_axis < 0 || start_axis >= lhs_rank) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "binary_reduce shape metadata is invalid");
    }
    for(dim = 0; dim < start_axis; ++dim) {
        if(gsx_size_mul_overflows(outer_count_lhs, (gsx_size_t)lhs_shape[dim], &outer_count_lhs)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "outer_count overflows");
        }
    }
    for(dim = start_axis; dim < lhs_rank; ++dim) {
        if(gsx_size_mul_overflows(reduce_count_lhs, (gsx_size_t)lhs_shape[dim], &reduce_count_lhs)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "reduce_count overflows");
        }
    }
    if(gsx_size_mul_overflows(outer_count_lhs, sizeof(float), &expected_out_bytes)
        || gsx_size_mul_overflows(outer_count_lhs, reduce_count_lhs, &expected_input_elements)
        || gsx_size_mul_overflows(expected_input_elements, sizeof(float), &expected_input_bytes)
        || out_view->size_bytes != expected_out_bytes
        || lhs_view->size_bytes != expected_input_bytes
        || rhs_view->size_bytes != expected_input_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "binary_reduce tensor views are inconsistent with dispatch metadata");
    }

    lhs_buffer_type = gsx_cuda_backend_buffer_get_type_class(lhs_view->buffer);
    rhs_buffer_type = gsx_cuda_backend_buffer_get_type_class(rhs_view->buffer);
    out_buffer_type = gsx_cuda_backend_buffer_get_type_class(dst_buffer);
    if(workspace_view->buffer != NULL) {
        workspace_buffer_type = gsx_cuda_backend_buffer_get_type_class(workspace_view->buffer);
    }
    if(lhs_buffer_type != GSX_BACKEND_BUFFER_TYPE_DEVICE || rhs_buffer_type != GSX_BACKEND_BUFFER_TYPE_DEVICE
        || out_buffer_type != GSX_BACKEND_BUFFER_TYPE_DEVICE || workspace_buffer_type != GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda binary_reduce only supports device buffers");
    }

    lhs_buffer = gsx_cuda_backend_buffer_from_base(lhs_view->buffer);
    rhs_buffer = gsx_cuda_backend_buffer_from_base(rhs_view->buffer);
    out_buffer = gsx_cuda_backend_buffer_from_base(dst_buffer);
    if(workspace_view->buffer != NULL) {
        workspace_buffer = gsx_cuda_backend_buffer_from_base(workspace_view->buffer);
    }
    cuda_backend = gsx_cuda_backend_from_base(dst_buffer->buffer_type->backend);
    lhs_values = (const float *)((const char*)lhs_buffer->ptr + lhs_view->offset_bytes);
    rhs_values = (const float *)((const char*)rhs_buffer->ptr + rhs_view->offset_bytes);
    out_values = (float *)((char*)out_buffer->ptr + out_view->offset_bytes);
    if(workspace_buffer != NULL) {
        workspace_ptr = (void *)((char*)workspace_buffer->ptr + workspace_view->offset_bytes);
    }

    cuda_err = gsx_cuda_binary_reduce_f32_launch(
        lhs_values,
        rhs_values,
        out_values,
        workspace_ptr,
        workspace_view->size_bytes,
        outer_count_lhs,
        reduce_count_lhs,
        op,
        cuda_backend->major_stream
    );
    return gsx_cuda_make_error(cuda_err, "cuda binary_reduce launch failed");
}

gsx_error gsx_cuda_backend_buffer_clamp_inplace_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *min_value,
    const void *max_value
)
{
    gsx_cuda_backend_buffer *cuda_buffer = NULL;
    gsx_cuda_backend *cuda_backend = NULL;
    gsx_backend_buffer_type_class buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_size_t element_size_bytes = 0;
    gsx_size_t element_count = 0;
    gsx_size_t element_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    cudaError_t cuda_err = cudaSuccess;
    void *tensor_data = NULL;

    if(buffer == NULL || tensor_view == NULL || min_value == NULL || max_value == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer, tensor_view, min_value, and max_value must be non-null");
    }
    if(tensor_view->buffer != buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must reference buffer");
    }

    error = gsx_cuda_backend_buffer_check_range(buffer, tensor_view->offset_bytes, tensor_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_data_type_get_size_bytes(tensor_view->data_type, &element_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(tensor_view->size_bytes % element_size_bytes != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor view byte size is not aligned to element size");
    }

    cuda_buffer = gsx_cuda_backend_buffer_from_base(buffer);
    buffer_type = gsx_cuda_backend_buffer_get_type_class(buffer);
    tensor_data = (void *)((char*)cuda_buffer->ptr + tensor_view->offset_bytes);
    element_count = tensor_view->size_bytes / element_size_bytes;

    if(buffer_type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        cuda_backend = gsx_cuda_backend_from_base(buffer->buffer_type->backend);
        switch(tensor_view->data_type) {
        case GSX_DATA_TYPE_F32:
            cuda_err = gsx_cuda_clamp_inplace_tensor_f32_kernel_launch(
                (float *)tensor_data,
                element_count,
                *(const float *)min_value,
                *(const float *)max_value,
                cuda_backend->major_stream
            );
            break;
        case GSX_DATA_TYPE_I32:
            cuda_err = gsx_cuda_clamp_inplace_tensor_i32_kernel_launch(
                (int32_t *)tensor_data,
                element_count,
                *(const int32_t *)min_value,
                *(const int32_t *)max_value,
                cuda_backend->major_stream
            );
            break;
        case GSX_DATA_TYPE_U8:
            cuda_err = gsx_cuda_clamp_inplace_tensor_u8_kernel_launch(
                (uint8_t *)tensor_data,
                element_count,
                *(const uint8_t *)min_value,
                *(const uint8_t *)max_value,
                cuda_backend->major_stream
            );
            break;
        default:
            return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "clamp_inplace only supports f32, u8, and i32 tensors on cuda backend");
        }
        if(cuda_err != cudaSuccess) {
            return gsx_cuda_make_error(cuda_err, "clamp_inplace_tensor kernel launch failed");
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(buffer_type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        switch(tensor_view->data_type) {
        case GSX_DATA_TYPE_F32: {
            float *values = (float *)tensor_data;
            const float min_bound = *(const float *)min_value;
            const float max_bound = *(const float *)max_value;

            for(element_index = 0; element_index < element_count; ++element_index) {
                if(values[element_index] < min_bound) {
                    values[element_index] = min_bound;
                } else if(values[element_index] > max_bound) {
                    values[element_index] = max_bound;
                }
            }
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        case GSX_DATA_TYPE_U8: {
            uint8_t *values = (uint8_t *)tensor_data;
            const uint8_t min_bound = *(const uint8_t *)min_value;
            const uint8_t max_bound = *(const uint8_t *)max_value;

            for(element_index = 0; element_index < element_count; ++element_index) {
                if(values[element_index] < min_bound) {
                    values[element_index] = min_bound;
                } else if(values[element_index] > max_bound) {
                    values[element_index] = max_bound;
                }
            }
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        case GSX_DATA_TYPE_I32: {
            int32_t *values = (int32_t *)tensor_data;
            const int32_t min_bound = *(const int32_t *)min_value;
            const int32_t max_bound = *(const int32_t *)max_value;

            for(element_index = 0; element_index < element_count; ++element_index) {
                if(values[element_index] < min_bound) {
                    values[element_index] = min_bound;
                } else if(values[element_index] > max_bound) {
                    values[element_index] = max_bound;
                }
            }
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        default:
            return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "clamp_inplace only supports f32, u8, and i32 tensors on cuda backend");
        }
    }

    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "clamp_inplace_tensor does not support this CUDA buffer type");
}

gsx_error gsx_cuda_backend_buffer_image_convert_colorspace(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *src_view,
    gsx_storage_format storage_format,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_image_colorspace src_colorspace,
    const gsx_backend_tensor_view *dst_view,
    gsx_image_colorspace dst_colorspace
)
{
    gsx_cuda_backend_buffer *src_buffer = NULL;
    gsx_cuda_backend_buffer *dst_cuda_buffer = NULL;
    gsx_cuda_backend *cuda_backend = NULL;
    gsx_backend_buffer_type_class src_buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_backend_buffer_type_class dst_buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_index_t channels = 0;
    gsx_index_t height = 0;
    gsx_index_t width = 0;
    gsx_size_t element_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    cudaError_t cuda_err = cudaSuccess;
    const float *src_values = NULL;
    float *dst_values = NULL;

    if(dst_buffer == NULL || src_view == NULL || dst_view == NULL || shape == NULL || src_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "image colorspace conversion inputs must be non-null");
    }
    if(dst_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_view must reference dst_buffer");
    }
    if(src_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "image colorspace tensors must belong to the same backend");
    }
    if(src_view->data_type != GSX_DATA_TYPE_F32 || dst_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda image colorspace conversion only supports float32 tensors");
    }

    error = gsx_cuda_backend_buffer_check_range(src_view->buffer, src_view->offset_bytes, src_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_backend_buffer_check_range(dst_buffer, dst_view->offset_bytes, dst_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_image_get_chw_hwc_dims(rank, shape, storage_format, &channels, &height, &width);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(channels != 3) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "image colorspace conversion requires a 3-channel RGB image");
    }

    src_buffer = gsx_cuda_backend_buffer_from_base(src_view->buffer);
    dst_cuda_buffer = gsx_cuda_backend_buffer_from_base(dst_buffer);
    src_buffer_type = gsx_cuda_backend_buffer_get_type_class(src_view->buffer);
    dst_buffer_type = gsx_cuda_backend_buffer_get_type_class(dst_buffer);
    src_values = (const float *)((const char *)src_buffer->ptr + src_view->offset_bytes);
    dst_values = (float *)((char *)dst_cuda_buffer->ptr + dst_view->offset_bytes);
    element_count = src_view->size_bytes / sizeof(float);

    if(src_buffer_type == GSX_BACKEND_BUFFER_TYPE_DEVICE && dst_buffer_type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        cuda_backend = gsx_cuda_backend_from_base(dst_buffer->buffer_type->backend);
        if(src_colorspace == GSX_IMAGE_COLOR_SPACE_LINEAR && dst_colorspace == GSX_IMAGE_COLOR_SPACE_SRGB) {
            cuda_err = gsx_cuda_image_linear_to_srgb_f32_kernel_launch(src_values, dst_values, element_count, cuda_backend->major_stream);
        } else if(src_colorspace == GSX_IMAGE_COLOR_SPACE_SRGB && dst_colorspace == GSX_IMAGE_COLOR_SPACE_LINEAR) {
            cuda_err = gsx_cuda_image_srgb_to_linear_f32_kernel_launch(src_values, dst_values, element_count, cuda_backend->major_stream);
        } else {
            cuda_err = cudaMemcpyAsync(dst_values, src_values, src_view->size_bytes, cudaMemcpyDeviceToDevice, cuda_backend->major_stream);
        }
        if(cuda_err != cudaSuccess) {
            return gsx_cuda_make_error(cuda_err, "image colorspace conversion on cuda failed");
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(src_buffer_type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED && dst_buffer_type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        gsx_size_t i = 0;
        for(i = 0; i < element_count; ++i) {
            if(src_colorspace == GSX_IMAGE_COLOR_SPACE_LINEAR && dst_colorspace == GSX_IMAGE_COLOR_SPACE_SRGB) {
                dst_values[i] = gsx_image_linear_to_srgb(src_values[i]);
            } else if(src_colorspace == GSX_IMAGE_COLOR_SPACE_SRGB && dst_colorspace == GSX_IMAGE_COLOR_SPACE_LINEAR) {
                dst_values[i] = gsx_image_srgb_to_linear(src_values[i]);
            } else {
                dst_values[i] = src_values[i];
            }
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "image colorspace conversion requires matching CUDA buffer types");
}

gsx_error gsx_cuda_backend_buffer_image_convert_storage_format(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *src_view,
    gsx_index_t src_rank,
    const gsx_index_t *src_shape,
    gsx_storage_format src_storage_format,
    const gsx_backend_tensor_view *dst_view,
    gsx_index_t dst_rank,
    const gsx_index_t *dst_shape,
    gsx_storage_format dst_storage_format
)
{
    gsx_cuda_backend_buffer *src_buffer = NULL;
    gsx_cuda_backend_buffer *dst_cuda_buffer = NULL;
    gsx_cuda_backend *cuda_backend = NULL;
    gsx_backend_buffer_type_class src_buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_backend_buffer_type_class dst_buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_index_t channels = 0;
    gsx_index_t height = 0;
    gsx_index_t width = 0;
    gsx_size_t element_size_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    cudaError_t cuda_err = cudaSuccess;
    const void *src_bytes = NULL;
    void *dst_bytes = NULL;

    if(dst_buffer == NULL || src_view == NULL || dst_view == NULL || src_shape == NULL || dst_shape == NULL || src_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "image storage conversion inputs must be non-null");
    }
    if(dst_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_view must reference dst_buffer");
    }
    if(src_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "image storage tensors must belong to the same backend");
    }
    if(src_view->data_type != dst_view->data_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "image storage conversion requires matching data types");
    }

    error = gsx_cuda_backend_buffer_check_range(src_view->buffer, src_view->offset_bytes, src_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_backend_buffer_check_range(dst_buffer, dst_view->offset_bytes, dst_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_data_type_get_size_bytes(src_view->data_type, &element_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(!gsx_image_same_extents(
            src_rank,
            src_shape,
            src_storage_format,
            dst_rank,
            dst_shape,
            dst_storage_format,
            &channels,
            &height,
            &width)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "image storage conversion requires matching logical extents");
    }

    src_buffer = gsx_cuda_backend_buffer_from_base(src_view->buffer);
    dst_cuda_buffer = gsx_cuda_backend_buffer_from_base(dst_buffer);
    src_buffer_type = gsx_cuda_backend_buffer_get_type_class(src_view->buffer);
    dst_buffer_type = gsx_cuda_backend_buffer_get_type_class(dst_buffer);
    src_bytes = (const char *)src_buffer->ptr + src_view->offset_bytes;
    dst_bytes = (char *)dst_cuda_buffer->ptr + dst_view->offset_bytes;

    if(src_buffer_type == GSX_BACKEND_BUFFER_TYPE_DEVICE && dst_buffer_type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        cuda_backend = gsx_cuda_backend_from_base(dst_buffer->buffer_type->backend);
        if(src_storage_format == GSX_STORAGE_FORMAT_CHW && dst_storage_format == GSX_STORAGE_FORMAT_HWC) {
            cuda_err = gsx_cuda_image_chw_to_hwc_kernel_launch(src_bytes, dst_bytes, channels, height, width, element_size_bytes, cuda_backend->major_stream);
        } else if(src_storage_format == GSX_STORAGE_FORMAT_HWC && dst_storage_format == GSX_STORAGE_FORMAT_CHW) {
            cuda_err = gsx_cuda_image_hwc_to_chw_kernel_launch(src_bytes, dst_bytes, channels, height, width, element_size_bytes, cuda_backend->major_stream);
        } else {
            return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "image storage conversion only supports CHW and HWC");
        }
        if(cuda_err != cudaSuccess) {
            return gsx_cuda_make_error(cuda_err, "image storage conversion kernel launch failed");
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(src_buffer_type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED && dst_buffer_type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        return gsx_image_copy_storage_convert_bytes(dst_bytes, dst_storage_format, src_bytes, src_storage_format, channels, height, width, element_size_bytes);
    }

    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "image storage conversion requires matching CUDA buffer types");
}

gsx_error gsx_cuda_backend_buffer_image_convert_data_type(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *src_view,
    gsx_storage_format storage_format,
    gsx_index_t rank,
    const gsx_index_t *shape,
    const gsx_backend_tensor_view *dst_view
)
{
    gsx_cuda_backend_buffer *src_buffer = NULL;
    gsx_cuda_backend_buffer *dst_cuda_buffer = NULL;
    gsx_cuda_backend *cuda_backend = NULL;
    gsx_backend_buffer_type_class src_buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_backend_buffer_type_class dst_buffer_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
    gsx_index_t channels = 0;
    gsx_index_t height = 0;
    gsx_index_t width = 0;
    gsx_size_t element_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    cudaError_t cuda_err = cudaSuccess;

    if(dst_buffer == NULL || src_view == NULL || dst_view == NULL || shape == NULL || src_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "image data type conversion inputs must be non-null");
    }
    if(dst_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_view must reference dst_buffer");
    }
    if(src_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "image data type tensors must belong to the same backend");
    }

    error = gsx_cuda_backend_buffer_check_range(src_view->buffer, src_view->offset_bytes, src_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_cuda_backend_buffer_check_range(dst_buffer, dst_view->offset_bytes, dst_view->size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_image_get_chw_hwc_dims(rank, shape, storage_format, &channels, &height, &width);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    src_buffer = gsx_cuda_backend_buffer_from_base(src_view->buffer);
    dst_cuda_buffer = gsx_cuda_backend_buffer_from_base(dst_buffer);
    src_buffer_type = gsx_cuda_backend_buffer_get_type_class(src_view->buffer);
    dst_buffer_type = gsx_cuda_backend_buffer_get_type_class(dst_buffer);
    element_count = (gsx_size_t)channels * (gsx_size_t)height * (gsx_size_t)width;

    if(src_buffer_type == GSX_BACKEND_BUFFER_TYPE_DEVICE && dst_buffer_type == GSX_BACKEND_BUFFER_TYPE_DEVICE) {
        cuda_backend = gsx_cuda_backend_from_base(dst_buffer->buffer_type->backend);
        if(src_view->data_type == GSX_DATA_TYPE_F32 && dst_view->data_type == GSX_DATA_TYPE_U8) {
            cuda_err = gsx_cuda_image_f32_to_u8_kernel_launch(
                (const float *)((const char *)src_buffer->ptr + src_view->offset_bytes),
                (uint8_t *)((char *)dst_cuda_buffer->ptr + dst_view->offset_bytes),
                element_count,
                cuda_backend->major_stream);
        } else if(src_view->data_type == GSX_DATA_TYPE_U8 && dst_view->data_type == GSX_DATA_TYPE_F32) {
            cuda_err = gsx_cuda_image_u8_to_f32_kernel_launch(
                (const uint8_t *)((const char *)src_buffer->ptr + src_view->offset_bytes),
                (float *)((char *)dst_cuda_buffer->ptr + dst_view->offset_bytes),
                element_count,
                cuda_backend->major_stream);
        } else {
            return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "cuda image data type conversion only supports float32 and uint8");
        }
        if(cuda_err != cudaSuccess) {
            return gsx_cuda_make_error(cuda_err, "image data type conversion kernel launch failed");
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(src_buffer_type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED && dst_buffer_type == GSX_BACKEND_BUFFER_TYPE_HOST_PINNED) {
        gsx_size_t i = 0;
        if(src_view->data_type == GSX_DATA_TYPE_F32 && dst_view->data_type == GSX_DATA_TYPE_U8) {
            const float *src_values = (const float *)((const char *)src_buffer->ptr + src_view->offset_bytes);
            uint8_t *dst_values = (uint8_t *)((char *)dst_cuda_buffer->ptr + dst_view->offset_bytes);
            for(i = 0; i < element_count; ++i) {
                dst_values[i] = gsx_image_quantize_u8(src_values[i]);
            }
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        if(src_view->data_type == GSX_DATA_TYPE_U8 && dst_view->data_type == GSX_DATA_TYPE_F32) {
            const uint8_t *src_values = (const uint8_t *)((const char *)src_buffer->ptr + src_view->offset_bytes);
            float *dst_values = (float *)((char *)dst_cuda_buffer->ptr + dst_view->offset_bytes);
            for(i = 0; i < element_count; ++i) {
                dst_values[i] = gsx_image_dequantize_u8(src_values[i]);
            }
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
    }

    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "image data type conversion requires matching CUDA buffer types");
}
