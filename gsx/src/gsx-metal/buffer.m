#include "internal.h"

#import <Metal/Metal.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *gsx_metal_backend_buffer_bytes(gsx_metal_backend_buffer *buffer)
{
    return (unsigned char *)[(id<MTLBuffer>)buffer->mtl_buffer contents];
}

static unsigned char *gsx_metal_backend_tensor_data(gsx_metal_backend_buffer *metal_buffer, const gsx_backend_tensor_view *tensor_view, gsx_size_t offset_bytes)
{
    return gsx_metal_backend_buffer_bytes(metal_buffer) + (size_t)(tensor_view->offset_bytes + offset_bytes);
}

static bool gsx_metal_backend_buffer_is_cpu_visible(gsx_metal_backend_buffer *buffer)
{
    return buffer->type_class != GSX_BACKEND_BUFFER_TYPE_DEVICE;
}

static MTLResourceOptions gsx_metal_backend_buffer_type_resource_options(gsx_backend_buffer_type_class type_class)
{
    switch(type_class) {
    case GSX_BACKEND_BUFFER_TYPE_DEVICE:
        return MTLResourceStorageModePrivate | MTLResourceCPUCacheModeDefaultCache;
    case GSX_BACKEND_BUFFER_TYPE_HOST_PINNED:
        return MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined;
    case GSX_BACKEND_BUFFER_TYPE_UNIFIED:
        return MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
    case GSX_BACKEND_BUFFER_TYPE_HOST:
        break;
    }
    return MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
}

static gsx_error gsx_metal_backend_commit_copy(
    gsx_metal_backend *metal_backend,
    id<MTLBuffer> src_buffer,
    gsx_size_t src_offset_bytes,
    id<MTLBuffer> dst_buffer,
    gsx_size_t dst_offset_bytes,
    gsx_size_t byte_count
)
{
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLBlitCommandEncoder> blit_encoder = nil;

    if(byte_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(src_buffer == nil || dst_buffer == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "source and destination Metal buffers must be non-null");
    }

    command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
    if(command_buffer == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal command buffer");
    }

    blit_encoder = [command_buffer blitCommandEncoder];
    if(blit_encoder == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal blit encoder");
    }

    [blit_encoder
        copyFromBuffer:src_buffer
        sourceOffset:(NSUInteger)src_offset_bytes
        toBuffer:dst_buffer
        destinationOffset:(NSUInteger)dst_offset_bytes
        size:(NSUInteger)byte_count];
    [blit_encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_backend_commit_fill(
    gsx_metal_backend *metal_backend,
    id<MTLBuffer> buffer,
    gsx_size_t offset_bytes,
    gsx_size_t byte_count,
    uint8_t value
)
{
    id<MTLCommandBuffer> command_buffer = nil;
    id<MTLBlitCommandEncoder> blit_encoder = nil;

    if(byte_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(buffer == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "Metal buffer must be non-null");
    }

    command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
    if(command_buffer == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal command buffer");
    }

    blit_encoder = [command_buffer blitCommandEncoder];
    if(blit_encoder == nil) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal blit encoder");
    }

    [blit_encoder fillBuffer:buffer range:NSMakeRange((NSUInteger)offset_bytes, (NSUInteger)byte_count) value:value];
    [blit_encoder endEncoding];
    [command_buffer commit];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_backend_tensor_compute_total_bytes(
    gsx_data_type data_type,
    gsx_index_t rank,
    const gsx_index_t *shape,
    gsx_size_t *out_total_bytes,
    gsx_size_t *out_row_bytes
)
{
    gsx_size_t element_size_bytes = 0;
    gsx_size_t element_count = 1;
    gsx_size_t row_elements = 1;
    gsx_index_t dim = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(shape == NULL || out_total_bytes == NULL || out_row_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "shape and output pointers must be non-null");
    }
    if(rank < 1) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "rank must be at least 1");
    }

    error = gsx_data_type_get_size_bytes(data_type, &element_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    for(dim = 0; dim < rank; ++dim) {
        if(shape[dim] <= 0) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "shape entries must be positive");
        }
        if(gsx_size_mul_overflows(element_count, (gsx_size_t)shape[dim], &element_count)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor element count overflows");
        }
        if(dim >= 1 && gsx_size_mul_overflows(row_elements, (gsx_size_t)shape[dim], &row_elements)) {
            return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor row element count overflows");
        }
    }

    if(gsx_size_mul_overflows(element_count, element_size_bytes, out_total_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor byte size overflows");
    }
    if(gsx_size_mul_overflows(row_elements, element_size_bytes, out_row_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "tensor row byte size overflows");
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_type_get_info(gsx_backend_buffer_type_t buffer_type, gsx_backend_buffer_type_info *out_info)
{
    gsx_metal_backend_buffer_type *metal_buffer_type = gsx_metal_backend_buffer_type_from_base(buffer_type);

    if(out_info == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_info must be non-null");
    }

    *out_info = metal_buffer_type->info;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_type_get_alloc_size(gsx_backend_buffer_type_t buffer_type, gsx_size_t requested_size_bytes, gsx_size_t *out_alloc_size_bytes)
{
    gsx_metal_backend_buffer_type *metal_buffer_type = gsx_metal_backend_buffer_type_from_base(buffer_type);

    if(out_alloc_size_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_alloc_size_bytes must be non-null");
    }

    if(gsx_round_up_overflows(requested_size_bytes, metal_buffer_type->info.alignment_bytes, out_alloc_size_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "allocation size overflow");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_type_init_buffer(gsx_backend_buffer_type_t buffer_type, const gsx_backend_buffer_desc *desc, gsx_backend_buffer_t *out_buffer)
{
    gsx_metal_backend_buffer_type *metal_buffer_type = gsx_metal_backend_buffer_type_from_base(buffer_type);
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(buffer_type->backend);
    gsx_metal_backend_buffer *metal_buffer = NULL;
    gsx_size_t alloc_size_bytes = 0;
    gsx_size_t effective_alignment = 0;
    MTLResourceOptions resource_options = MTLResourceStorageModeShared;
    id<MTLBuffer> mtl_buffer = nil;

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

    effective_alignment = metal_buffer_type->info.alignment_bytes;
    if(desc->alignment_bytes > effective_alignment) {
        effective_alignment = desc->alignment_bytes;
    }

    if(gsx_round_up_overflows(desc->size_bytes, effective_alignment, &alloc_size_bytes)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "allocation size overflow");
    }

    resource_options = gsx_metal_backend_buffer_type_resource_options(metal_buffer_type->info.type);

    metal_buffer = (gsx_metal_backend_buffer *)calloc(1, sizeof(*metal_buffer));
    if(metal_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate buffer struct");
    }

    mtl_buffer = [(id<MTLDevice>)metal_backend->mtl_device
        newBufferWithLength:(NSUInteger)alloc_size_bytes
        options:resource_options];
    if(mtl_buffer == nil) {
        free(metal_buffer);
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal buffer");
    }

    metal_buffer->base.iface = &gsx_metal_backend_buffer_iface;
    metal_buffer->base.buffer_type = buffer_type;
    metal_buffer->base.size_bytes = desc->size_bytes;
    metal_buffer->base.alignment_bytes = effective_alignment;
    metal_buffer->mtl_buffer = mtl_buffer;
    metal_buffer->alloc_size_bytes = alloc_size_bytes;
    metal_buffer->type_class = metal_buffer_type->info.type;
    metal_buffer->resource_options = (uint32_t)resource_options;

    metal_backend->base.live_buffer_count += 1;
    *out_buffer = &metal_buffer->base;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_free(gsx_backend_buffer_t buffer)
{
    gsx_metal_backend_buffer *metal_buffer = NULL;
    gsx_metal_backend *metal_backend = NULL;

    if(buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer must be non-null");
    }
    metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    metal_backend = gsx_metal_backend_from_base(buffer->buffer_type->backend);

    if(metal_backend->base.live_buffer_count == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "backend live_buffer_count underflow in buffer free");
    }

    if(metal_buffer->mtl_buffer != NULL) {
        [(id<MTLBuffer>)metal_buffer->mtl_buffer release];
        metal_buffer->mtl_buffer = NULL;
    }

    metal_backend->base.live_buffer_count -= 1;
    free(metal_buffer);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_get_info(gsx_backend_buffer_t buffer, gsx_backend_buffer_info *out_info)
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

gsx_error gsx_metal_backend_buffer_upload(gsx_backend_buffer_t buffer, gsx_size_t dst_offset_bytes, const void *src_bytes, gsx_size_t byte_count)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(buffer->buffer_type->backend);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    unsigned char *dst_ptr = NULL;

    error = gsx_metal_backend_buffer_check_range(buffer, dst_offset_bytes, byte_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(byte_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(src_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src_bytes must be non-null for non-zero byte_count");
    }

    if(gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        dst_ptr = gsx_metal_backend_buffer_bytes(metal_buffer);
        if(dst_ptr == NULL) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "metal buffer contents are unavailable");
        }

        memcpy(dst_ptr + (size_t)dst_offset_bytes, src_bytes, (size_t)byte_count);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    {
        id<MTLBuffer> staging_buffer = [(id<MTLDevice>)metal_backend->mtl_device
            newBufferWithBytes:src_bytes
            length:(NSUInteger)byte_count
            options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLBlitCommandEncoder> blit_encoder = nil;

        if(staging_buffer == nil) {
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal upload staging buffer");
        }

        command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
        if(command_buffer == nil) {
            [staging_buffer release];
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal command buffer");
        }

        blit_encoder = [command_buffer blitCommandEncoder];
        if(blit_encoder == nil) {
            [staging_buffer release];
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal blit encoder");
        }

        [blit_encoder
            copyFromBuffer:staging_buffer
            sourceOffset:0
            toBuffer:(id<MTLBuffer>)metal_buffer->mtl_buffer
            destinationOffset:(NSUInteger)dst_offset_bytes
            size:(NSUInteger)byte_count];
        [blit_encoder endEncoding];
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed_buffer) {
            (void)completed_buffer;
            [staging_buffer release];
        }];
        [command_buffer commit];
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_download(gsx_backend_buffer_t buffer, gsx_size_t src_offset_bytes, void *dst_bytes, gsx_size_t byte_count)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(buffer->buffer_type->backend);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    const unsigned char *src_ptr = NULL;

    error = gsx_metal_backend_buffer_check_range(buffer, src_offset_bytes, byte_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(byte_count == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(dst_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_bytes must be non-null for non-zero byte_count");
    }

    if(gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        src_ptr = gsx_metal_backend_buffer_bytes(metal_buffer);
        if(src_ptr == NULL) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "metal buffer contents are unavailable");
        }

        memcpy(dst_bytes, src_ptr + (size_t)src_offset_bytes, (size_t)byte_count);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    {
        id<MTLBuffer> staging_buffer = [(id<MTLDevice>)metal_backend->mtl_device
            newBufferWithLength:(NSUInteger)byte_count
            options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache];
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLBlitCommandEncoder> blit_encoder = nil;

        if(staging_buffer == nil) {
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal download staging buffer");
        }

        command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
        if(command_buffer == nil) {
            [staging_buffer release];
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal command buffer");
        }

        blit_encoder = [command_buffer blitCommandEncoder];
        if(blit_encoder == nil) {
            [staging_buffer release];
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal blit encoder");
        }

        [blit_encoder
            copyFromBuffer:(id<MTLBuffer>)metal_buffer->mtl_buffer
            sourceOffset:(NSUInteger)src_offset_bytes
            toBuffer:staging_buffer
            destinationOffset:0
            size:(NSUInteger)byte_count];
        [blit_encoder endEncoding];
        /* The destination pointer must remain valid until the submitted command buffer completes. */
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed_buffer) {
            const void *staging_ptr = NULL;

            (void)completed_buffer;
            staging_ptr = [staging_buffer contents];
            if(staging_ptr != NULL) {
                memcpy(dst_bytes, staging_ptr, (size_t)byte_count);
            }
            [staging_buffer release];
        }];
        [command_buffer commit];
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_set_zero(gsx_backend_buffer_t buffer)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(buffer->buffer_type->backend);

    if(metal_buffer->alloc_size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        unsigned char *bytes = gsx_metal_backend_buffer_bytes(metal_buffer);

        if(bytes == NULL) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "metal buffer contents are unavailable");
        }

        memset(bytes, 0, (size_t)metal_buffer->alloc_size_bytes);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    return gsx_metal_backend_commit_fill(
        metal_backend,
        (id<MTLBuffer>)metal_buffer->mtl_buffer,
        0,
        metal_buffer->alloc_size_bytes,
        0
    );
}

gsx_error gsx_metal_backend_buffer_memset_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    uint8_t value,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(buffer->buffer_type->backend);
    gsx_error error = gsx_metal_backend_tensor_view_check_range(buffer, tensor_view, offset_bytes, size_bytes);

    if(!gsx_error_is_success(error) || size_bytes == 0) {
        return error;
    }

    if(gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        memset(gsx_metal_backend_tensor_data(metal_buffer, tensor_view, offset_bytes), value, (size_t)size_bytes);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    return gsx_metal_backend_commit_fill(
        metal_backend,
        (id<MTLBuffer>)metal_buffer->mtl_buffer,
        tensor_view->offset_bytes + offset_bytes,
        size_bytes,
        value
    );
}

gsx_error gsx_metal_backend_buffer_set_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *src_bytes,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(buffer->buffer_type->backend);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(size_bytes != 0 && src_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "src_bytes must be non-null when size_bytes is non-zero");
    }

    error = gsx_metal_backend_tensor_view_check_range(buffer, tensor_view, offset_bytes, size_bytes);
    if(!gsx_error_is_success(error) || size_bytes == 0) {
        return error;
    }

    if(gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        memcpy(gsx_metal_backend_tensor_data(metal_buffer, tensor_view, offset_bytes), src_bytes, (size_t)size_bytes);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    {
        id<MTLBuffer> staging_buffer = [(id<MTLDevice>)metal_backend->mtl_device
            newBufferWithBytes:src_bytes
            length:(NSUInteger)size_bytes
            options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLBlitCommandEncoder> blit_encoder = nil;

        if(staging_buffer == nil) {
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal tensor upload staging buffer");
        }

        command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
        if(command_buffer == nil) {
            [staging_buffer release];
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal command buffer");
        }

        blit_encoder = [command_buffer blitCommandEncoder];
        if(blit_encoder == nil) {
            [staging_buffer release];
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal blit encoder");
        }

        [blit_encoder
            copyFromBuffer:staging_buffer
            sourceOffset:0
            toBuffer:(id<MTLBuffer>)metal_buffer->mtl_buffer
            destinationOffset:(NSUInteger)(tensor_view->offset_bytes + offset_bytes)
            size:(NSUInteger)size_bytes];
        [blit_encoder endEncoding];
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed_buffer) {
            (void)completed_buffer;
            [staging_buffer release];
        }];
        [command_buffer commit];
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_get_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    void *dst_bytes,
    gsx_size_t offset_bytes,
    gsx_size_t size_bytes
)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(buffer->buffer_type->backend);
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(size_bytes != 0 && dst_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dst_bytes must be non-null when size_bytes is non-zero");
    }

    error = gsx_metal_backend_tensor_view_check_range(buffer, tensor_view, offset_bytes, size_bytes);
    if(!gsx_error_is_success(error) || size_bytes == 0) {
        return error;
    }

    if(gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        memcpy(dst_bytes, gsx_metal_backend_tensor_data(metal_buffer, tensor_view, offset_bytes), (size_t)size_bytes);
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    {
        id<MTLBuffer> staging_buffer = [(id<MTLDevice>)metal_backend->mtl_device
            newBufferWithLength:(NSUInteger)size_bytes
            options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache];
        id<MTLCommandBuffer> command_buffer = nil;
        id<MTLBlitCommandEncoder> blit_encoder = nil;

        if(staging_buffer == nil) {
            return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate Metal tensor download staging buffer");
        }

        command_buffer = [(id<MTLCommandQueue>)metal_backend->major_command_queue commandBuffer];
        if(command_buffer == nil) {
            [staging_buffer release];
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal command buffer");
        }

        blit_encoder = [command_buffer blitCommandEncoder];
        if(blit_encoder == nil) {
            [staging_buffer release];
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to allocate Metal blit encoder");
        }

        [blit_encoder
            copyFromBuffer:(id<MTLBuffer>)metal_buffer->mtl_buffer
            sourceOffset:(NSUInteger)(tensor_view->offset_bytes + offset_bytes)
            toBuffer:staging_buffer
            destinationOffset:0
            size:(NSUInteger)size_bytes];
        [blit_encoder endEncoding];
        /* The destination pointer must remain valid until the submitted command buffer completes. */
        [command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed_buffer) {
            const void *staging_ptr = NULL;

            (void)completed_buffer;
            staging_ptr = [staging_buffer contents];
            if(staging_ptr != NULL) {
                memcpy(dst_bytes, staging_ptr, (size_t)size_bytes);
            }
            [staging_buffer release];
        }];
        [command_buffer commit];
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_metal_backend_buffer_copy_tensor(gsx_backend_buffer_t dst_buffer, const gsx_backend_tensor_view *src_view, const gsx_backend_tensor_view *dst_view)
{
    gsx_metal_backend_buffer *src_metal_buffer = NULL;
    gsx_metal_backend_buffer *dst_metal_buffer = NULL;
    gsx_size_t src_begin_bytes = 0;
    gsx_size_t src_end_bytes = 0;
    gsx_size_t dst_begin_bytes = 0;
    gsx_size_t dst_end_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dst_buffer == NULL || src_view == NULL || dst_view == NULL || src_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer and tensor views must be non-null");
    }
    if(src_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend || dst_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor copy requires source and destination to belong to the same backend");
    }

    error = gsx_metal_backend_tensor_view_validate(src_view->buffer, src_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_tensor_view_validate(dst_buffer, dst_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(src_view->size_bytes != dst_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor copy requires equal source and destination sizes");
    }
    if(src_view->size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    src_begin_bytes = src_view->offset_bytes;
    src_end_bytes = src_view->offset_bytes + src_view->size_bytes;
    dst_begin_bytes = dst_view->offset_bytes;
    dst_end_bytes = dst_view->offset_bytes + dst_view->size_bytes;
    if(src_view->buffer == dst_buffer) {
        if(src_begin_bytes == dst_begin_bytes && src_end_bytes == dst_end_bytes) {
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
        if(!(dst_end_bytes <= src_begin_bytes || src_end_bytes <= dst_begin_bytes)) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor copy rejects overlapping source and destination ranges");
        }
    }

    src_metal_buffer = gsx_metal_backend_buffer_from_base(src_view->buffer);
    dst_metal_buffer = gsx_metal_backend_buffer_from_base(dst_buffer);
    if(gsx_metal_backend_buffer_is_cpu_visible(src_metal_buffer) && gsx_metal_backend_buffer_is_cpu_visible(dst_metal_buffer)) {
        memcpy(
            gsx_metal_backend_buffer_bytes(dst_metal_buffer) + (size_t)dst_begin_bytes,
            gsx_metal_backend_buffer_bytes(src_metal_buffer) + (size_t)src_begin_bytes,
            (size_t)src_view->size_bytes
        );
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    return gsx_metal_backend_commit_copy(
        gsx_metal_backend_from_base(dst_buffer->buffer_type->backend),
        (id<MTLBuffer>)src_metal_buffer->mtl_buffer,
        src_begin_bytes,
        (id<MTLBuffer>)dst_metal_buffer->mtl_buffer,
        dst_begin_bytes,
        src_view->size_bytes
    );
}

gsx_error gsx_metal_backend_buffer_fill_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *value_bytes,
    gsx_size_t value_size_bytes
)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    gsx_metal_backend *metal_backend = gsx_metal_backend_from_base(buffer->buffer_type->backend);
    unsigned char *dst_bytes = NULL;
    gsx_size_t offset_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(value_size_bytes == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "value_size_bytes must be non-zero");
    }
    if(tensor_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor_view must be non-null");
    }
    if(tensor_view->size_bytes != 0 && value_bytes == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "value_bytes must be non-null when tensor is non-empty");
    }
    if(tensor_view->size_bytes % value_size_bytes != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor byte size must be a multiple of value_size_bytes");
    }

    error = gsx_metal_backend_tensor_view_validate(buffer, tensor_view);
    if(!gsx_error_is_success(error) || tensor_view->size_bytes == 0) {
        return error;
    }

    if(gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        dst_bytes = gsx_metal_backend_tensor_data(metal_buffer, tensor_view, 0);
        for(offset_bytes = 0; offset_bytes < tensor_view->size_bytes; offset_bytes += value_size_bytes) {
            memcpy(dst_bytes + (size_t)offset_bytes, value_bytes, (size_t)value_size_bytes);
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    if(value_size_bytes == 1) {
        return gsx_metal_backend_commit_fill(
            metal_backend,
            (id<MTLBuffer>)metal_buffer->mtl_buffer,
            tensor_view->offset_bytes,
            tensor_view->size_bytes,
            *(const uint8_t *)value_bytes
        );
    }

    return gsx_make_error(
        GSX_ERROR_NOT_SUPPORTED,
        "fill_tensor on Metal device buffers currently supports value_size_bytes == 1 only without explicit synchronization APIs"
    );
}

gsx_error gsx_metal_backend_buffer_check_finite_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    bool *out_is_finite
)
{
    gsx_metal_backend_buffer *metal_buffer = gsx_metal_backend_buffer_from_base(buffer);
    const unsigned char *bytes = NULL;
    gsx_size_t element_count = 0;
    gsx_size_t element_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_is_finite == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_is_finite must be non-null");
    }
    *out_is_finite = true;

    error = gsx_metal_backend_tensor_view_validate(buffer, tensor_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(!gsx_metal_backend_buffer_is_cpu_visible(metal_buffer)) {
        return gsx_make_error(
            GSX_ERROR_NOT_SUPPORTED,
            "finite check on Metal device buffers is not supported yet without explicit synchronization APIs"
        );
    }

    bytes = gsx_metal_backend_tensor_data(metal_buffer, tensor_view, 0);
    switch(tensor_view->data_type) {
    case GSX_DATA_TYPE_F32: {
        const float *values = (const float *)bytes;

        if(tensor_view->size_bytes % sizeof(float) != 0) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "float32 tensors must have a byte size divisible by sizeof(float)");
        }
        element_count = tensor_view->size_bytes / sizeof(float);
        for(element_index = 0; element_index < element_count; ++element_index) {
            if(!isfinite((double)values[element_index])) {
                *out_is_finite = false;
                return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
            }
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    case GSX_DATA_TYPE_F16: {
        const uint16_t *values = (const uint16_t *)bytes;

        if(tensor_view->size_bytes % sizeof(uint16_t) != 0) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "float16 tensors must have a byte size divisible by sizeof(uint16_t)");
        }
        element_count = tensor_view->size_bytes / sizeof(uint16_t);
        for(element_index = 0; element_index < element_count; ++element_index) {
            if(!gsx_metal_backend_f16_is_finite(values[element_index])) {
                *out_is_finite = false;
                return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
            }
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    case GSX_DATA_TYPE_BF16: {
        const uint16_t *values = (const uint16_t *)bytes;

        if(tensor_view->size_bytes % sizeof(uint16_t) != 0) {
            return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "bfloat16 tensors must have a byte size divisible by sizeof(uint16_t)");
        }
        element_count = tensor_view->size_bytes / sizeof(uint16_t);
        for(element_index = 0; element_index < element_count; ++element_index) {
            if(!gsx_metal_backend_bf16_is_finite(values[element_index])) {
                *out_is_finite = false;
                return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
            }
        }
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    default:
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "finite check supports f32/f16/bf16 only");
    }
}

gsx_error gsx_metal_backend_buffer_gather_tensor(
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
    gsx_size_t expected_x_bytes = 0;
    gsx_size_t expected_out_bytes = 0;
    gsx_size_t expected_index_bytes = 0;
    gsx_size_t x_row_bytes = 0;
    gsx_size_t out_row_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_metal_tensor_gather_params params = { 0 };

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
    if(index_view->data_type != GSX_DATA_TYPE_I32) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "index tensor must use int32");
    }
    if(x_rank != out_rank || x_rank < 1) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out ranks must match and be at least 1");
    }
    if(x_view->data_type != out_view->data_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out data types must match");
    }

    error = gsx_metal_backend_tensor_view_validate(x_view->buffer, x_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_tensor_view_validate(index_view->buffer, index_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_tensor_view_validate(dst_buffer, out_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_metal_backend_tensor_compute_total_bytes(x_view->data_type, x_rank, x_shape, &expected_x_bytes, &x_row_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_tensor_compute_total_bytes(out_view->data_type, out_rank, out_shape, &expected_out_bytes, &out_row_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(expected_x_bytes != x_view->size_bytes || expected_out_bytes != out_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor views do not match the provided shape metadata");
    }
    if(x_row_bytes != out_row_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out trailing dimensions must match");
    }
    if(gsx_size_mul_overflows((gsx_size_t)out_shape[0], sizeof(int32_t), &expected_index_bytes) || expected_index_bytes != index_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "index view byte size must match out leading dimension");
    }

    if((gsx_size_t)x_shape[0] > UINT32_MAX || (gsx_size_t)out_shape[0] > UINT32_MAX || out_row_bytes > UINT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "gather launch parameters exceed Metal kernel limits");
    }

    params.x_row_count = (uint32_t)x_shape[0];
    params.out_row_count = (uint32_t)out_shape[0];
    params.row_bytes = (uint32_t)out_row_bytes;
    return gsx_metal_backend_dispatch_tensor_gather(
        dst_buffer->buffer_type->backend,
        x_view,
        index_view,
        out_view,
        &params
    );
}

gsx_error gsx_metal_backend_buffer_exp_tensor(
    gsx_backend_buffer_t dst_buffer,
    const gsx_backend_tensor_view *x_view,
    const gsx_backend_tensor_view *out_view,
    gsx_index_t rank,
    const gsx_index_t *shape
)
{
    gsx_size_t expected_bytes = 0;
    gsx_size_t row_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };
    gsx_metal_tensor_exp_params params = { 0 };

    if(dst_buffer == NULL || x_view == NULL || out_view == NULL || shape == NULL || x_view->buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "buffer, tensor views, and shape must be non-null");
    }
    if(out_view->buffer != dst_buffer) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_view must reference dst_buffer");
    }
    if(x_view->buffer->buffer_type->backend != dst_buffer->buffer_type->backend) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out must belong to the same backend");
    }
    if(x_view->data_type != out_view->data_type) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "x and out data types must match");
    }
    if(x_view->data_type != GSX_DATA_TYPE_F32) {
        return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "exp only supports float32 tensors on metal backend");
    }

    error = gsx_metal_backend_tensor_view_validate(x_view->buffer, x_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_tensor_view_validate(dst_buffer, out_view);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_metal_backend_tensor_compute_total_bytes(x_view->data_type, rank, shape, &expected_bytes, &row_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(row_bytes == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "row byte size must be non-zero");
    }
    if(expected_bytes != x_view->size_bytes || expected_bytes != out_view->size_bytes) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor views do not match the provided shape metadata");
    }
    if(expected_bytes % sizeof(float) != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "float32 tensor byte size must be divisible by sizeof(float)");
    }
    if((expected_bytes / sizeof(float)) > UINT32_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "exp element count exceeds Metal kernel limits");
    }

    params.element_count = (uint32_t)(expected_bytes / sizeof(float));
    return gsx_metal_backend_dispatch_tensor_exp(dst_buffer->buffer_type->backend, x_view, out_view, &params);
}

gsx_error gsx_metal_backend_buffer_clamp_inplace_tensor(
    gsx_backend_buffer_t buffer,
    const gsx_backend_tensor_view *tensor_view,
    const void *min_value,
    const void *max_value
)
{
    (void)buffer;
    (void)tensor_view;
    (void)min_value;
    (void)max_value;
    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal clamp_inplace_tensor is not implemented");
}
