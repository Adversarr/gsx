#include "gsx/gsx-random.h"
#include "gsx-impl.h"

#include "pcg32.h"

#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

GSX_API gsx_error gsx_pcg32_init(gsx_pcg32_t* out_pcg, gsx_pcg32_state_t init_seed) {
    if(out_pcg == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_pcg must be non-null");
    }
    *out_pcg = NULL;

    gsx_pcg32* pcg = (gsx_pcg32*)malloc(sizeof(*pcg));
    if(pcg == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate gsx_pcg32 state");
    }

    pcg32_init(pcg, init_seed, init_seed);

    *out_pcg = pcg;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_pcg32_free(gsx_pcg32_t pcg) {
    free(pcg);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_pcg32_next_uint(gsx_pcg32_t pcg, uint32_t* out_value) {
    if(pcg == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "pcg must be non-null");
    }
    if(out_value == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_value must be non-null");
    }
    *out_value = pcg32_next_uint((gsx_pcg32*)pcg);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_pcg32_next_uint_bound(gsx_pcg32_t pcg, uint32_t* out_value, uint32_t bound) {
    if(pcg == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "pcg must be non-null");
    }
    if(out_value == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_value must be non-null");
    }
    if(bound == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "bound must be non-zero");
    }

    *out_value = pcg32_next_uint_bound((gsx_pcg32*)pcg, bound);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_pcg32_next_float(gsx_pcg32_t pcg, float* out_value) {
    if(pcg == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "pcg must be non-null");
    }
    if(out_value == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_value must be non-null");
    }

    *out_value = pcg32_next_float((gsx_pcg32*)pcg);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_pcg32_next_double(gsx_pcg32_t pcg, double* out_value) {
    if(pcg == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "pcg must be non-null");
    }
    if(out_value == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_value must be non-null");
    }

    *out_value = pcg32_next_double((gsx_pcg32*)pcg);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_pcg32_next_normal(gsx_pcg32_t pcg, float* out_value) {
    float u1 = 0.0f;
    float u2 = 0.0f;
    float mag = 0.0f;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(pcg == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "pcg must be non-null");
    }
    if(out_value == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_value must be non-null");
    }

    error = gsx_pcg32_next_float(pcg, &u1);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_pcg32_next_float(pcg, &u2);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    u1 = fmaxf(1e-7f, u1);
    mag = sqrtf(-2.0f * logf(u1));
    *out_value = mag * cosf(2.0f * (float)M_PI * u2);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_pcg32_advance(gsx_pcg32_t pcg, gsx_pcg32_statediff_t delta) {
    if(pcg == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "pcg must be non-null");
    }

    pcg32_advance((gsx_pcg32*)pcg, delta);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_pcg32_distance(const gsx_pcg32_t a, const gsx_pcg32_t b, gsx_pcg32_statediff_t* out_distance) {
    if(a == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "a must be non-null");
    }
    if(b == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "b must be non-null");
    }
    if(out_distance == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_distance must be non-null");
    }

    *out_distance = (gsx_pcg32_statediff_t)pcg32_distance((const gsx_pcg32*)a, (const gsx_pcg32*)b);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_pcg32_equal(const gsx_pcg32_t a, const gsx_pcg32_t b, bool* out_equal) {
    if(a == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "a must be non-null");
    }
    if(b == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "b must be non-null");
    }
    if(out_equal == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_equal must be non-null");
    }
    *out_equal = pcg32_equal((const gsx_pcg32*)a, (const gsx_pcg32*)b);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_random_tensor_require_accessible_storage(gsx_tensor_t tensor)
{
    if(tensor == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor must be non-null");
    }
    if(tensor->arena == NULL || tensor->arena->dry_run) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "tensor storage is unavailable in dry-run mode");
    }
    if(tensor->backing_buffer == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "tensor backing buffer is unavailable");
    }
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_backend_tensor_view gsx_random_tensor_make_backend_view(gsx_tensor_t tensor)
{
    gsx_backend_tensor_view tensor_view = { 0 };

    tensor_view.buffer = tensor->backing_buffer;
    tensor_view.offset_bytes = tensor->offset_bytes;
    tensor_view.size_bytes = tensor->size_bytes;
    tensor_view.effective_alignment_bytes = tensor->effective_alignment_bytes;
    tensor_view.data_type = tensor->data_type;
    return tensor_view;
}

static bool gsx_random_data_type_is_floating(gsx_data_type data_type)
{
    switch(data_type) {
    case GSX_DATA_TYPE_F32:
    case GSX_DATA_TYPE_F16:
    case GSX_DATA_TYPE_BF16:
        return true;
    default:
        return false;
    }
}

static bool gsx_random_data_type_is_integer(gsx_data_type data_type)
{
    switch(data_type) {
    case GSX_DATA_TYPE_U8:
    case GSX_DATA_TYPE_I16:
    case GSX_DATA_TYPE_U32:
    case GSX_DATA_TYPE_I32:
        return true;
    default:
        return false;
    }
}

static gsx_error gsx_random_tensor_get_element_count(gsx_tensor_t tensor, gsx_size_t *out_element_count)
{
    gsx_size_t element_size_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(out_element_count == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_element_count must be non-null");
    }

    error = gsx_data_type_get_size_bytes(tensor->data_type, &element_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(element_size_bytes == 0 || tensor->size_bytes % element_size_bytes != 0) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "tensor byte size is inconsistent with its data type");
    }

    *out_element_count = tensor->size_bytes / element_size_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_random_advance_after_fill(gsx_pcg32_t pcg, gsx_size_t advance_count)
{
    if(advance_count > (gsx_size_t)INT64_MAX) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "random fill would advance the PCG state beyond supported range");
    }
    return gsx_pcg32_advance(pcg, (gsx_pcg32_statediff_t)advance_count);
}

GSX_API gsx_error gsx_pcg32_fill_rand(gsx_pcg32_t pcg, gsx_tensor_t tensor)
{
    gsx_backend_tensor_view tensor_view = { 0 };
    gsx_size_t element_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(pcg == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "pcg must be non-null");
    }
    error = gsx_random_tensor_require_accessible_storage(tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(!gsx_random_data_type_is_floating(tensor->data_type)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "rand fill requires a floating-point tensor");
    }

    error = gsx_random_tensor_get_element_count(tensor, &element_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    tensor_view = gsx_random_tensor_make_backend_view(tensor);
    error = tensor->backing_buffer->iface->fill_rand_tensor(tensor->backing_buffer, &tensor_view, pcg->state, pcg->inc);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_random_advance_after_fill(pcg, element_count);
}

GSX_API gsx_error gsx_pcg32_fill_randn(gsx_pcg32_t pcg, gsx_tensor_t tensor, gsx_float_t sigma)
{
    gsx_backend_tensor_view tensor_view = { 0 };
    gsx_size_t element_count = 0;
    gsx_size_t advance_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(pcg == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "pcg must be non-null");
    }
    if(!isfinite((double)sigma) || sigma < 0.0f) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "sigma must be finite and non-negative");
    }
    error = gsx_random_tensor_require_accessible_storage(tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(!gsx_random_data_type_is_floating(tensor->data_type)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "randn fill requires a floating-point tensor");
    }

    error = gsx_random_tensor_get_element_count(tensor, &element_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(element_count == (gsx_size_t)INT64_MAX && (element_count % 2u) != 0u) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "random normal fill would advance the PCG state beyond supported range");
    }
    advance_count = element_count + (element_count % 2u);

    tensor_view = gsx_random_tensor_make_backend_view(tensor);
    error = tensor->backing_buffer->iface->fill_randn_tensor(tensor->backing_buffer, &tensor_view, pcg->state, pcg->inc, sigma);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_random_advance_after_fill(pcg, advance_count);
}

GSX_API gsx_error gsx_pcg32_fill_randint(gsx_pcg32_t pcg, gsx_tensor_t tensor, uint32_t bound)
{
    gsx_backend_tensor_view tensor_view = { 0 };
    gsx_size_t element_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(pcg == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "pcg must be non-null");
    }
    if(bound == 0) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "bound must be non-zero");
    }
    error = gsx_random_tensor_require_accessible_storage(tensor);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(!gsx_random_data_type_is_integer(tensor->data_type)) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "randint fill requires an integer tensor");
    }

    error = gsx_random_tensor_get_element_count(tensor, &element_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    tensor_view = gsx_random_tensor_make_backend_view(tensor);
    error = tensor->backing_buffer->iface->fill_randint_tensor(tensor->backing_buffer, &tensor_view, pcg->state, pcg->inc, bound);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_random_advance_after_fill(pcg, element_count);
}
