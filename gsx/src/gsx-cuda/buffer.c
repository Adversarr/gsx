#include "internal.h"

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
    if(desc->alignment_bytes != 0 && !gsx_is_power_of_two(desc->alignment_bytes)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "desc->alignment_bytes must be a power of two");
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
    case GSX_DATA_TYPE_BF16:
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
                bool is_value_finite = tensor_view->data_type == GSX_DATA_TYPE_F16
                    ? gsx_cuda_backend_f16_is_finite(values[element_index])
                    : gsx_cuda_backend_bf16_is_finite(values[element_index]);

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
    case GSX_DATA_TYPE_BF16:
        gsx_cuda_check_finite_tensor_bf16_kernel_launch(
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
