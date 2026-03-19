#include "../internal.h"

#import <Metal/Metal.h>

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>

typedef struct gsx_metal_sort_pair_u32_i32 {
    uint32_t key;
    int32_t value;
    uint32_t stable_index;
} gsx_metal_sort_pair_u32_i32;

static bool gsx_metal_render_debug_dump_enabled(gsx_size_t gaussian_count)
{
    const char *env = getenv("GSX_METAL_FORWARD_DUMP");

    (void)gaussian_count;

    if(env != NULL && env[0] != '\0' && env[0] != '0') {
        return true;
    }
    return false;
}

static uint64_t gsx_metal_render_hash_u32_i32_pairs(const uint32_t *keys, const int32_t *values, gsx_size_t count)
{
    uint64_t h = 1469598103934665603ull;

    for(gsx_size_t i = 0; i < count; ++i) {
        uint64_t k = (uint64_t)keys[i];
        uint64_t v = (uint64_t)(uint32_t)values[i];

        h ^= k;
        h *= 1099511628211ull;
        h ^= v;
        h *= 1099511628211ull;
    }
    return h;
}

static int gsx_metal_render_sort_pair_u32_i32_compare(const void *lhs, const void *rhs)
{
    const gsx_metal_sort_pair_u32_i32 *a = (const gsx_metal_sort_pair_u32_i32 *)lhs;
    const gsx_metal_sort_pair_u32_i32 *b = (const gsx_metal_sort_pair_u32_i32 *)rhs;

    if(a->key < b->key) {
        return -1;
    }
    if(a->key > b->key) {
        return 1;
    }
    if(a->stable_index < b->stable_index) {
        return -1;
    }
    if(a->stable_index > b->stable_index) {
        return 1;
    }
    return 0;
}

static gsx_error gsx_metal_render_compute_expected_stable_sort(
    const uint32_t *in_keys,
    const int32_t *in_values,
    gsx_size_t count,
    uint32_t *out_keys,
    int32_t *out_values)
{
    gsx_metal_sort_pair_u32_i32 *pairs = NULL;

    if(in_keys == NULL || in_values == NULL || out_keys == NULL || out_values == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "stable-sort inputs must be non-null");
    }
    if(count == 0u) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    pairs = (gsx_metal_sort_pair_u32_i32 *)malloc((size_t)count * sizeof(*pairs));
    if(pairs == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate stable-sort pair scratch");
    }

    for(gsx_size_t i = 0; i < count; ++i) {
        pairs[i].key = in_keys[i];
        pairs[i].value = in_values[i];
        pairs[i].stable_index = (uint32_t)i;
    }
    qsort(pairs, (size_t)count, sizeof(*pairs), gsx_metal_render_sort_pair_u32_i32_compare);
    for(gsx_size_t i = 0; i < count; ++i) {
        out_keys[i] = pairs[i].key;
        out_values[i] = pairs[i].value;
    }

    free(pairs);
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static bool gsx_metal_render_u32_i32_pairs_match(
    const uint32_t *lhs_keys,
    const int32_t *lhs_values,
    const uint32_t *rhs_keys,
    const int32_t *rhs_values,
    gsx_size_t count,
    gsx_size_t *out_first_mismatch)
{
    if(out_first_mismatch != NULL) {
        *out_first_mismatch = (gsx_size_t)-1;
    }
    for(gsx_size_t i = 0; i < count; ++i) {
        if(lhs_keys[i] != rhs_keys[i] || lhs_values[i] != rhs_values[i]) {
            if(out_first_mismatch != NULL) {
                *out_first_mismatch = i;
            }
            return false;
        }
    }
    return true;
}

static gsx_error gsx_metal_render_validate_sorted_tile_ranges(
    const int32_t *instance_keys,
    gsx_size_t instance_count,
    const int32_t *tile_ranges,
    gsx_size_t tile_count,
    bool dump_debug)
{
    gsx_size_t previous_end = 0;

    if(instance_keys == NULL || tile_ranges == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tile-range validation inputs must be non-null");
    }

    for(gsx_size_t tile = 0; tile < tile_count; ++tile) {
        int32_t start = tile_ranges[tile * 2u];
        int32_t end = tile_ranges[tile * 2u + 1u];

        if(start < 0 || end < start || (gsx_size_t)end > instance_count) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "tile_ranges contains an invalid start/end range");
        }
        if(start == end) {
            if((gsx_size_t)start != previous_end) {
                return gsx_make_error(GSX_ERROR_INVALID_STATE, "tile_ranges empty-tile gap violates sorted instance layout");
            }
            continue;
        }
        if((gsx_size_t)start != previous_end) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "tile_ranges non-empty tile starts at unexpected position");
        }
        for(gsx_size_t i = (gsx_size_t)start; i < (gsx_size_t)end; ++i) {
            if(instance_keys[i] != (int32_t)tile) {
                return gsx_make_error(GSX_ERROR_INVALID_STATE, "tile_ranges points to instance keys from a different tile");
            }
        }
        previous_end = (gsx_size_t)end;
    }

    if(previous_end != instance_count) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "tile_ranges coverage does not match instance_count");
    }

    if(dump_debug) {
        printf(
            "METAL ordering dump: instance_count=%zu tile_count=%zu tile_ranges_coverage=%zu\n",
            (size_t)instance_count,
            (size_t)tile_count,
            (size_t)previous_end);
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

typedef struct gsx_metal_forward_scratch {
    gsx_tensor_t depth;
    gsx_tensor_t visible;
    gsx_tensor_t touched;
    gsx_tensor_t depth_keys;
    gsx_tensor_t depth_keys_sorted;
    gsx_tensor_t bounds;
    gsx_tensor_t mean2d;
    gsx_tensor_t conic_opacity;
    gsx_tensor_t color;
    gsx_tensor_t sorted_primitive_ids;
    gsx_tensor_t sorted_primitive_ids_sorted;
    gsx_tensor_t primitive_offsets;
    gsx_tensor_t sort_histogram;
    gsx_tensor_t sort_global_histogram;
    gsx_tensor_t sort_scatter_offsets;
    gsx_tensor_t scan_block_sums;
    gsx_tensor_t scan_scanned_block_sums;
    gsx_tensor_t instance_keys;
    gsx_tensor_t instance_primitive_ids;
    gsx_tensor_t instance_keys_sorted;
    gsx_tensor_t instance_primitive_ids_sorted;
    gsx_tensor_t tile_ranges;
    gsx_tensor_t tile_bucket_offsets;
    gsx_tensor_t tile_max_n_contributions;
    gsx_tensor_t tile_n_contributions;
    gsx_tensor_t bucket_tile_index;
    gsx_tensor_t bucket_color_transmittance;
} gsx_metal_forward_scratch;

static gsx_error gsx_metal_render_plan_append_desc(
    gsx_tensor_desc *descs,
    gsx_index_t capacity,
    gsx_index_t *count,
    gsx_data_type data_type,
    gsx_index_t rank,
    const gsx_index_t *shape)
{
    gsx_tensor_desc *desc = NULL;

    if(descs == NULL || count == NULL || shape == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "plan descriptor arguments must be non-null");
    }
    if(*count >= capacity) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "forward planning descriptor capacity exceeded");
    }

    desc = &descs[*count];
    memset(desc, 0, sizeof(*desc));
    desc->rank = rank;
    for(gsx_index_t i = 0; i < rank; ++i) {
        desc->shape[i] = shape[i];
    }
    desc->data_type = data_type;
    desc->storage_format = GSX_STORAGE_FORMAT_CHW;
    desc->arena = NULL;
    *count += 1;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_render_plan_and_reserve_arena(gsx_arena_t arena, const gsx_tensor_desc *descs, gsx_index_t desc_count)
{
    gsx_backend_buffer_type_t buffer_type = NULL;
    gsx_arena_desc arena_desc = { 0 };
    gsx_size_t required_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(arena == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "arena must be non-null");
    }

    error = gsx_arena_get_buffer_type(arena, &buffer_type);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
    arena_desc.dry_run = true;
    error = gsx_tensor_plan_required_bytes(buffer_type, &arena_desc, descs, desc_count, &required_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return gsx_arena_reserve(arena, required_bytes);
}

static void gsx_metal_render_cleanup_forward_scratch(gsx_metal_forward_scratch *scratch)
{
    gsx_metal_render_release_tensor(&scratch->bucket_color_transmittance);
    gsx_metal_render_release_tensor(&scratch->bucket_tile_index);
    gsx_metal_render_release_tensor(&scratch->tile_max_n_contributions);
    gsx_metal_render_release_tensor(&scratch->tile_n_contributions);
    gsx_metal_render_release_tensor(&scratch->tile_bucket_offsets);
    gsx_metal_render_release_tensor(&scratch->tile_ranges);
    gsx_metal_render_release_tensor(&scratch->instance_primitive_ids_sorted);
    gsx_metal_render_release_tensor(&scratch->instance_keys_sorted);
    gsx_metal_render_release_tensor(&scratch->instance_primitive_ids);
    gsx_metal_render_release_tensor(&scratch->instance_keys);
    gsx_metal_render_release_tensor(&scratch->scan_scanned_block_sums);
    gsx_metal_render_release_tensor(&scratch->scan_block_sums);
    gsx_metal_render_release_tensor(&scratch->sort_scatter_offsets);
    gsx_metal_render_release_tensor(&scratch->sort_global_histogram);
    gsx_metal_render_release_tensor(&scratch->sort_histogram);
    gsx_metal_render_release_tensor(&scratch->primitive_offsets);
    gsx_metal_render_release_tensor(&scratch->sorted_primitive_ids_sorted);
    gsx_metal_render_release_tensor(&scratch->sorted_primitive_ids);
    gsx_metal_render_release_tensor(&scratch->color);
    gsx_metal_render_release_tensor(&scratch->conic_opacity);
    gsx_metal_render_release_tensor(&scratch->mean2d);
    gsx_metal_render_release_tensor(&scratch->bounds);
    gsx_metal_render_release_tensor(&scratch->depth_keys);
    gsx_metal_render_release_tensor(&scratch->depth_keys_sorted);
    gsx_metal_render_release_tensor(&scratch->touched);
    gsx_metal_render_release_tensor(&scratch->visible);
    gsx_metal_render_release_tensor(&scratch->depth);
}

static uint32_t gsx_metal_render_depth_sort_key(float depth)
{
    union {
        float f;
        uint32_t u;
    } bits;

    bits.f = depth;
    return bits.u;
}

static gsx_error gsx_metal_render_gather_visible_inputs(
    const float *depth,
    const int32_t *visible,
    const int32_t *touched,
    gsx_size_t gaussian_count,
    uint32_t *out_depth_keys,
    int32_t *out_sorted_primitive_ids,
    int32_t *out_primitive_touched_counts,
    gsx_size_t *out_visible_count,
    gsx_size_t *out_instance_count)
{
    gsx_size_t visible_count = 0;
    gsx_size_t instance_count = 0;

    if(depth == NULL || visible == NULL || touched == NULL
        || out_depth_keys == NULL || out_sorted_primitive_ids == NULL
        || out_primitive_touched_counts == NULL
        || out_visible_count == NULL || out_instance_count == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "gather_visible_inputs: all pointers must be non-null");
    }

    for(gsx_size_t i = 0; i < gaussian_count; ++i) {
        if(visible[i] != 0 && touched[i] > 0) {
            out_depth_keys[visible_count] = gsx_metal_render_depth_sort_key(depth[i]);
            out_sorted_primitive_ids[visible_count] = (int32_t)i;
            out_primitive_touched_counts[visible_count] = touched[i];
            visible_count += 1;
        }
    }
    for(gsx_size_t i = 0; i < visible_count; ++i) {
        instance_count += (gsx_size_t)out_primitive_touched_counts[i];
    }

    *out_visible_count = visible_count;
    *out_instance_count = instance_count;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static void gsx_metal_render_fill_tile_ranges(
    const int32_t *instance_keys,
    gsx_size_t instance_count,
    int32_t *tile_ranges,
    gsx_size_t tile_count)
{
    for(gsx_size_t i = 0; i < tile_count; ++i) {
        tile_ranges[i * 2u] = 0;
        tile_ranges[i * 2u + 1u] = 0;
    }
    for(gsx_size_t i = 0; i < instance_count; ++i) {
        int32_t key = instance_keys[i];

        if(key < 0 || (gsx_size_t)key >= tile_count) {
            continue;
        }
        if(tile_ranges[(gsx_size_t)key * 2u] == 0 && tile_ranges[(gsx_size_t)key * 2u + 1u] == 0) {
            tile_ranges[(gsx_size_t)key * 2u] = (int32_t)i;
        }
        tile_ranges[(gsx_size_t)key * 2u + 1u] = (int32_t)(i + 1u);
    }
}

static gsx_size_t gsx_metal_render_fill_tile_bucket_offsets(
    const int32_t *tile_ranges,
    gsx_size_t tile_count,
    int32_t *tile_bucket_offsets)
{
    gsx_size_t running_bucket_count = 0;

    for(gsx_size_t i = 0; i < tile_count; ++i) {
        int32_t start = tile_ranges[i * 2u];
        int32_t end = tile_ranges[i * 2u + 1u];
        gsx_size_t tile_bucket_count = 0;

        if(start >= 0 && end > start) {
            tile_bucket_count = ((gsx_size_t)(end - start) + 31u) / 32u;
        }
        running_bucket_count += tile_bucket_count;
        tile_bucket_offsets[i] = (int32_t)running_bucket_count;
    }
    return running_bucket_count;
}

static gsx_error gsx_metal_render_reserve_forward_arenas_with_dry_run(
    gsx_metal_render_context *metal_context,
    gsx_index_t width,
    gsx_index_t height,
    gsx_size_t gaussian_count,
    gsx_size_t tile_count,
    gsx_size_t max_bucket_count,
    gsx_size_t max_instance_count)
{
    gsx_tensor_desc primitive_descs[12] = { 0 };
    gsx_tensor_desc tile_descs[4] = { 0 };
    gsx_tensor_desc instance_descs[9] = { 0 };
    gsx_tensor_desc bucket_descs[2] = { 0 };
    gsx_index_t primitive_desc_count = 0;
    gsx_index_t tile_desc_count = 0;
    gsx_index_t instance_desc_count = 0;
    gsx_index_t bucket_desc_count = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_context must be non-null");
    }

    if(gaussian_count > 0) {
        gsx_index_t shape_n[1] = { (gsx_index_t)gaussian_count };
        gsx_index_t shape_n2[2] = { (gsx_index_t)gaussian_count, 2 };
        gsx_index_t shape_n3[2] = { (gsx_index_t)gaussian_count, 3 };
        gsx_index_t shape_n4[2] = { (gsx_index_t)gaussian_count, 4 };
        gsx_index_t shape_instances_max[1] = { (gsx_index_t)max_instance_count };
        gsx_index_t shape_global_histogram[1] = { 256 };
        gsx_size_t sort_threadgroup_count = (max_instance_count + 1023u) / 1024u;
        gsx_size_t scan_block_count = (gaussian_count + 255u) / 256u;
        gsx_index_t shape_sort_histogram[1] = { (gsx_index_t)(sort_threadgroup_count * 256u) };
        gsx_index_t shape_scan_block_sums[1] = { (gsx_index_t)scan_block_count };

        error = gsx_metal_render_plan_append_desc(primitive_descs, 12, &primitive_desc_count, GSX_DATA_TYPE_F32, 1, shape_n);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        for(gsx_index_t i = 0; i < 7; ++i) {
            error = gsx_metal_render_plan_append_desc(primitive_descs, 12, &primitive_desc_count, GSX_DATA_TYPE_I32, 1, shape_n);
            if(!gsx_error_is_success(error)) {
                return error;
            }
        }
        error = gsx_metal_render_plan_append_desc(primitive_descs, 12, &primitive_desc_count, GSX_DATA_TYPE_F32, 2, shape_n4);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_metal_render_plan_append_desc(primitive_descs, 12, &primitive_desc_count, GSX_DATA_TYPE_F32, 2, shape_n2);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_metal_render_plan_append_desc(primitive_descs, 12, &primitive_desc_count, GSX_DATA_TYPE_F32, 2, shape_n4);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_metal_render_plan_append_desc(primitive_descs, 12, &primitive_desc_count, GSX_DATA_TYPE_F32, 2, shape_n3);
        if(!gsx_error_is_success(error)) {
            return error;
        }

        for(gsx_index_t i = 0; i < 4; ++i) {
            error = gsx_metal_render_plan_append_desc(instance_descs, 9, &instance_desc_count, GSX_DATA_TYPE_I32, 1, shape_instances_max);
            if(!gsx_error_is_success(error)) {
                return error;
            }
        }
        error = gsx_metal_render_plan_append_desc(instance_descs, 9, &instance_desc_count, GSX_DATA_TYPE_I32, 1, shape_sort_histogram);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_metal_render_plan_append_desc(instance_descs, 9, &instance_desc_count, GSX_DATA_TYPE_I32, 1, shape_global_histogram);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_metal_render_plan_append_desc(instance_descs, 9, &instance_desc_count, GSX_DATA_TYPE_I32, 1, shape_sort_histogram);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_metal_render_plan_append_desc(instance_descs, 9, &instance_desc_count, GSX_DATA_TYPE_I32, 1, shape_scan_block_sums);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        error = gsx_metal_render_plan_append_desc(instance_descs, 9, &instance_desc_count, GSX_DATA_TYPE_I32, 1, shape_scan_block_sums);
        if(!gsx_error_is_success(error)) {
            return error;
        }

        if(tile_count > 0) {
            gsx_index_t shape_tile_ranges[2] = { (gsx_index_t)tile_count, 2 };
            gsx_index_t shape_tile_bucket_offsets[1] = { (gsx_index_t)tile_count };
            gsx_index_t shape_tile_max_n_contributions[1] = { (gsx_index_t)tile_count };
            gsx_index_t shape_tile_n_contributions[2] = { height, width };

            error = gsx_metal_render_plan_append_desc(tile_descs, 4, &tile_desc_count, GSX_DATA_TYPE_I32, 2, shape_tile_ranges);
            if(!gsx_error_is_success(error)) {
                return error;
            }
            error = gsx_metal_render_plan_append_desc(tile_descs, 4, &tile_desc_count, GSX_DATA_TYPE_I32, 1, shape_tile_bucket_offsets);
            if(!gsx_error_is_success(error)) {
                return error;
            }
            error = gsx_metal_render_plan_append_desc(tile_descs, 4, &tile_desc_count, GSX_DATA_TYPE_I32, 1, shape_tile_max_n_contributions);
            if(!gsx_error_is_success(error)) {
                return error;
            }
            error = gsx_metal_render_plan_append_desc(tile_descs, 4, &tile_desc_count, GSX_DATA_TYPE_I32, 2, shape_tile_n_contributions);
            if(!gsx_error_is_success(error)) {
                return error;
            }
        }

        if(max_bucket_count > 0) {
            gsx_index_t shape_bucket_tile_index[1] = { (gsx_index_t)max_bucket_count };
            gsx_index_t shape_bucket_color_transmittance[2] = { (gsx_index_t)(max_bucket_count * 256u), 4 };

            error = gsx_metal_render_plan_append_desc(bucket_descs, 2, &bucket_desc_count, GSX_DATA_TYPE_I32, 1, shape_bucket_tile_index);
            if(!gsx_error_is_success(error)) {
                return error;
            }
            error = gsx_metal_render_plan_append_desc(bucket_descs, 2, &bucket_desc_count, GSX_DATA_TYPE_F32, 2, shape_bucket_color_transmittance);
            if(!gsx_error_is_success(error)) {
                return error;
            }
        }
    }

    error = gsx_metal_render_plan_and_reserve_arena(
        metal_context->forward_per_primitive_arena,
        primitive_descs,
        primitive_desc_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_metal_render_plan_and_reserve_arena(
        metal_context->forward_per_tile_arena,
        tile_descs,
        tile_desc_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_metal_render_plan_and_reserve_arena(
        metal_context->forward_per_instance_arena,
        instance_descs,
        instance_desc_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    error = gsx_metal_render_plan_and_reserve_arena(
        metal_context->forward_per_bucket_arena,
        bucket_descs,
        bucket_desc_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_render_prepare_forward_context(gsx_metal_render_context *metal_context)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_metal_render_context_clear_train_state(metal_context);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_arena_reset(metal_context->forward_per_primitive_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_arena_reset(metal_context->forward_per_tile_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_arena_reset(metal_context->forward_per_instance_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_arena_reset(metal_context->forward_per_bucket_arena);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    error = gsx_tensor_set_zero(metal_context->helper_image_chw);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    return gsx_tensor_set_zero(metal_context->helper_alpha_hw);
}

gsx_error gsx_metal_renderer_forward_impl(gsx_renderer_t renderer, gsx_render_context_t context, const gsx_render_forward_request *request)
{
    gsx_metal_render_context *metal_context = (gsx_metal_render_context *)context;
    gsx_metal_forward_scratch scratch;
    int32_t *host_visible = NULL;
    int32_t *host_touched = NULL;
    uint32_t *host_depth_keys = NULL;
    int32_t *host_sorted_primitive_ids = NULL;
    int32_t *host_primitive_offsets = NULL;
    int32_t *host_instance_keys = NULL;
    int32_t *host_instance_primitive_ids = NULL;
    int32_t *host_tile_ranges = NULL;
    int32_t *host_tile_bucket_offsets = NULL;
    float *host_depth = NULL;
    gsx_size_t gaussian_count = 0;
    gsx_size_t visible_count = 0;
    gsx_size_t instance_count = 0;
    gsx_size_t tile_count = 0;
    gsx_size_t bucket_count = 0;
    gsx_size_t max_bucket_count = 0;
    gsx_size_t max_instance_count = 0;
    gsx_size_t mapped_size_bytes = 0;
    gsx_index_t grid_width = 0;
    gsx_index_t grid_height = 0;
    gsx_backend_tensor_view image_view = { 0 };
    gsx_backend_tensor_view alpha_view = { 0 };
    gsx_backend_tensor_view out_view = { 0 };
    gsx_backend_tensor_view mean3d_in_view = { 0 };
    gsx_backend_tensor_view rotation_in_view = { 0 };
    gsx_backend_tensor_view logscale_in_view = { 0 };
    gsx_backend_tensor_view sh0_in_view = { 0 };
    gsx_backend_tensor_view sh1_in_view = { 0 };
    gsx_backend_tensor_view sh2_in_view = { 0 };
    gsx_backend_tensor_view sh3_in_view = { 0 };
    gsx_backend_tensor_view opacity_in_view = { 0 };
    gsx_backend_tensor_view depth_view = { 0 };
    gsx_backend_tensor_view visible_view = { 0 };
    gsx_backend_tensor_view touched_view = { 0 };
    gsx_backend_tensor_view bounds_view = { 0 };
    gsx_backend_tensor_view mean2d_view = { 0 };
    gsx_backend_tensor_view conic_opacity_view = { 0 };
    gsx_backend_tensor_view color_view = { 0 };
    gsx_backend_tensor_view visible_counter_aux_view = { 0 };
    gsx_backend_tensor_view max_screen_radius_aux_view = { 0 };
    gsx_backend_tensor_view sorted_primitive_ids_view = { 0 };
    gsx_backend_tensor_view sorted_primitive_ids_sorted_view = { 0 };
    gsx_backend_tensor_view depth_keys_view = { 0 };
    gsx_backend_tensor_view depth_keys_sorted_view = { 0 };
    gsx_backend_tensor_view primitive_offsets_view = { 0 };
    gsx_backend_tensor_view sort_histogram_view = { 0 };
    gsx_backend_tensor_view sort_global_histogram_view = { 0 };
    gsx_backend_tensor_view sort_scatter_offsets_view = { 0 };
    gsx_backend_tensor_view scan_block_sums_view = { 0 };
    gsx_backend_tensor_view scan_scanned_block_sums_view = { 0 };
    gsx_backend_tensor_view instance_keys_view = { 0 };
    gsx_backend_tensor_view instance_primitive_ids_view = { 0 };
    gsx_backend_tensor_view instance_keys_sorted_view = { 0 };
    gsx_backend_tensor_view instance_primitive_ids_sorted_view = { 0 };
    gsx_backend_tensor_view tile_ranges_view = { 0 };
    gsx_backend_tensor_view tile_bucket_offsets_view = { 0 };
    gsx_backend_tensor_view tile_max_n_contributions_view = { 0 };
    gsx_backend_tensor_view tile_n_contributions_view = { 0 };
    gsx_backend_tensor_view bucket_tile_index_view = { 0 };
    gsx_backend_tensor_view bucket_color_transmittance_view = { 0 };
    gsx_metal_render_compose_params compose_params = { 0 };
    gsx_metal_render_preprocess_params preprocess_params = { 0 };
    gsx_metal_render_create_instances_params create_params = { 0 };
    gsx_metal_render_blend_params blend_params = { 0 };
    uint32_t *expected_depth_keys_sorted = NULL;
    int32_t *expected_depth_ids_sorted = NULL;
    uint32_t *expected_instance_keys_sorted = NULL;
    int32_t *expected_instance_ids_sorted = NULL;
    int32_t *host_depth_keys_sorted = NULL;
    int32_t *host_instance_keys_unsorted = NULL;
    bool dump_debug = false;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    memset(&scratch, 0, sizeof(scratch));
    error = gsx_metal_render_prepare_forward_context(metal_context);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    gaussian_count = (gsx_size_t)request->gs_mean3d->shape[0];
    dump_debug = gsx_metal_render_debug_dump_enabled(gaussian_count);
    grid_width = gsx_metal_render_get_grid_width(renderer->info.width);
    grid_height = gsx_metal_render_get_grid_height(renderer->info.height);
    tile_count = gsx_metal_render_get_tile_count(renderer->info.width, renderer->info.height);
    if(gsx_size_add_overflows(gaussian_count, 31u, &max_bucket_count)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "metal forward max-bucket reserve sizing overflow");
    }
    max_bucket_count /= 32u;
    if(gsx_size_mul_overflows(max_bucket_count, tile_count, &max_bucket_count)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "metal forward max-bucket reserve sizing overflow");
    }
    if(gsx_size_mul_overflows(gaussian_count, tile_count, &max_instance_count)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "metal forward max-instance sizing overflow");
    }

    error = gsx_metal_render_reserve_forward_arenas_with_dry_run(
        metal_context,
        renderer->info.width,
        renderer->info.height,
        gaussian_count,
        tile_count,
        max_bucket_count,
        max_instance_count);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    if(gaussian_count > 0) {
        gsx_index_t shape_n[1] = { (gsx_index_t)gaussian_count };
        gsx_index_t shape_n2[2] = { (gsx_index_t)gaussian_count, 2 };
        gsx_index_t shape_n3[2] = { (gsx_index_t)gaussian_count, 3 };
        gsx_index_t shape_n4[2] = { (gsx_index_t)gaussian_count, 4 };
        gsx_size_t sort_threadgroup_count = (max_instance_count + 1023u) / 1024u;
        gsx_size_t scan_block_count = (gaussian_count + 255u) / 256u;
        gsx_index_t shape_sort_histogram[1] = { (gsx_index_t)(sort_threadgroup_count * 256u) };
        gsx_index_t shape_global_histogram[1] = { 256 };
        gsx_index_t shape_scan_block_sums[1] = { (gsx_index_t)scan_block_count };

        error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_F32, 1, shape_n, &scratch.depth);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.visible);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.touched);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.depth_keys);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.depth_keys_sorted);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.sorted_primitive_ids);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.sorted_primitive_ids_sorted);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.primitive_offsets);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->forward_per_instance_arena, GSX_DATA_TYPE_I32, 1, shape_sort_histogram, &scratch.sort_histogram);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->forward_per_instance_arena, GSX_DATA_TYPE_I32, 1, shape_global_histogram, &scratch.sort_global_histogram);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->forward_per_instance_arena, GSX_DATA_TYPE_I32, 1, shape_sort_histogram, &scratch.sort_scatter_offsets);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->forward_per_instance_arena, GSX_DATA_TYPE_I32, 1, shape_scan_block_sums, &scratch.scan_block_sums);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->forward_per_instance_arena, GSX_DATA_TYPE_I32, 1, shape_scan_block_sums, &scratch.scan_scanned_block_sums);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_F32, 2, shape_n4, &scratch.bounds);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_F32, 2, shape_n2, &scratch.mean2d);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_F32, 2, shape_n4, &scratch.conic_opacity);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_F32, 2, shape_n3, &scratch.color);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }

        gsx_metal_render_make_tensor_view(request->gs_mean3d, &mean3d_in_view);
        gsx_metal_render_make_tensor_view(request->gs_rotation, &rotation_in_view);
        gsx_metal_render_make_tensor_view(request->gs_logscale, &logscale_in_view);
        gsx_metal_render_make_tensor_view(request->gs_sh0, &sh0_in_view);
        if(request->gs_sh1 != NULL) {
            gsx_metal_render_make_tensor_view(request->gs_sh1, &sh1_in_view);
        }
        if(request->gs_sh2 != NULL) {
            gsx_metal_render_make_tensor_view(request->gs_sh2, &sh2_in_view);
        }
        if(request->gs_sh3 != NULL) {
            gsx_metal_render_make_tensor_view(request->gs_sh3, &sh3_in_view);
        }
        gsx_metal_render_make_tensor_view(request->gs_opacity, &opacity_in_view);
        gsx_metal_render_make_tensor_view(scratch.depth, &depth_view);
        gsx_metal_render_make_tensor_view(scratch.visible, &visible_view);
        gsx_metal_render_make_tensor_view(scratch.touched, &touched_view);
        gsx_metal_render_make_tensor_view(scratch.depth_keys, &depth_keys_view);
        gsx_metal_render_make_tensor_view(scratch.depth_keys_sorted, &depth_keys_sorted_view);
        gsx_metal_render_make_tensor_view(scratch.sorted_primitive_ids, &sorted_primitive_ids_view);
        gsx_metal_render_make_tensor_view(scratch.sorted_primitive_ids_sorted, &sorted_primitive_ids_sorted_view);
        gsx_metal_render_make_tensor_view(scratch.primitive_offsets, &primitive_offsets_view);
        gsx_metal_render_make_tensor_view(scratch.sort_histogram, &sort_histogram_view);
        gsx_metal_render_make_tensor_view(scratch.sort_global_histogram, &sort_global_histogram_view);
        gsx_metal_render_make_tensor_view(scratch.sort_scatter_offsets, &sort_scatter_offsets_view);
        gsx_metal_render_make_tensor_view(scratch.scan_block_sums, &scan_block_sums_view);
        gsx_metal_render_make_tensor_view(scratch.scan_scanned_block_sums, &scan_scanned_block_sums_view);
        gsx_metal_render_make_tensor_view(scratch.bounds, &bounds_view);
        gsx_metal_render_make_tensor_view(scratch.mean2d, &mean2d_view);
        gsx_metal_render_make_tensor_view(scratch.conic_opacity, &conic_opacity_view);
        gsx_metal_render_make_tensor_view(scratch.color, &color_view);
        if(request->forward_type == GSX_RENDER_FORWARD_TYPE_TRAIN && request->gs_visible_counter != NULL) {
            gsx_metal_render_make_tensor_view(request->gs_visible_counter, &visible_counter_aux_view);
        }
        if(request->forward_type == GSX_RENDER_FORWARD_TYPE_TRAIN && request->gs_max_screen_radius != NULL) {
            gsx_metal_render_make_tensor_view(request->gs_max_screen_radius, &max_screen_radius_aux_view);
        }

        preprocess_params.gaussian_count = (uint32_t)gaussian_count;
        preprocess_params.width = (uint32_t)renderer->info.width;
        preprocess_params.height = (uint32_t)renderer->info.height;
        preprocess_params.sh_degree = (uint32_t)request->sh_degree;
        preprocess_params.grid_width = (uint32_t)grid_width;
        preprocess_params.grid_height = (uint32_t)grid_height;
        preprocess_params.fx = request->intrinsics->fx;
        preprocess_params.fy = request->intrinsics->fy;
        preprocess_params.cx = request->intrinsics->cx;
        preprocess_params.cy = request->intrinsics->cy;
        preprocess_params.near_plane = request->near_plane;
        preprocess_params.far_plane = request->far_plane;
        preprocess_params.pose_qx = request->pose->rot.x;
        preprocess_params.pose_qy = request->pose->rot.y;
        preprocess_params.pose_qz = request->pose->rot.z;
        preprocess_params.pose_qw = request->pose->rot.w;
        preprocess_params.pose_tx = request->pose->transl.x;
        preprocess_params.pose_ty = request->pose->transl.y;
        preprocess_params.pose_tz = request->pose->transl.z;

        error = gsx_metal_backend_dispatch_render_preprocess(
            renderer->backend,
            &mean3d_in_view,
            &rotation_in_view,
            &logscale_in_view,
            &sh0_in_view,
            request->gs_sh1 != NULL ? &sh1_in_view : NULL,
            request->gs_sh2 != NULL ? &sh2_in_view : NULL,
            request->gs_sh3 != NULL ? &sh3_in_view : NULL,
            &opacity_in_view,
            &depth_view,
            &visible_view,
            &touched_view,
            &bounds_view,
            &mean2d_view,
            &conic_opacity_view,
            &color_view,
            request->forward_type == GSX_RENDER_FORWARD_TYPE_TRAIN && request->gs_visible_counter != NULL ? &visible_counter_aux_view : NULL,
            request->forward_type == GSX_RENDER_FORWARD_TYPE_TRAIN && request->gs_max_screen_radius != NULL ? &max_screen_radius_aux_view : NULL,
            &preprocess_params);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }

        error = gsx_backend_major_stream_sync(renderer->backend);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_tensor_map_host_bytes(scratch.depth, (void **)&host_depth, &mapped_size_bytes);
        if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)gaussian_count * sizeof(float)) {
            error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map preprocess depth tensor to host-visible bytes");
            goto cleanup;
        }
        error = gsx_metal_render_tensor_map_host_bytes(scratch.visible, (void **)&host_visible, &mapped_size_bytes);
        if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)gaussian_count * sizeof(int32_t)) {
            error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map preprocess visible tensor to host-visible bytes");
            goto cleanup;
        }
        error = gsx_metal_render_tensor_map_host_bytes(scratch.touched, (void **)&host_touched, &mapped_size_bytes);
        if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)gaussian_count * sizeof(int32_t)) {
            error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map preprocess touched tensor to host-visible bytes");
            goto cleanup;
        }
        error = gsx_metal_render_tensor_map_host_bytes(scratch.depth_keys, (void **)&host_depth_keys, &mapped_size_bytes);
        if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)gaussian_count * sizeof(uint32_t)) {
            error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map depth-key tensor to host-visible bytes");
            goto cleanup;
        }
        error = gsx_metal_render_tensor_map_host_bytes(scratch.sorted_primitive_ids, (void **)&host_sorted_primitive_ids, &mapped_size_bytes);
        if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)gaussian_count * sizeof(int32_t)) {
            error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map sorted-primitive-id tensor to host-visible bytes");
            goto cleanup;
        }
        error = gsx_metal_render_tensor_map_host_bytes(scratch.primitive_offsets, (void **)&host_primitive_offsets, &mapped_size_bytes);
        if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)gaussian_count * sizeof(int32_t)) {
            error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map primitive-offset tensor to host-visible bytes");
            goto cleanup;
        }
        error = gsx_metal_render_gather_visible_inputs(
            host_depth, host_visible, host_touched, gaussian_count,
            host_depth_keys,
            host_sorted_primitive_ids, host_primitive_offsets,
            &visible_count, &instance_count);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        if(visible_count > 0) {
            gsx_size_t running_instance_offset = 0;

            if(dump_debug) {
                expected_depth_keys_sorted = (uint32_t *)malloc((size_t)visible_count * sizeof(uint32_t));
                expected_depth_ids_sorted = (int32_t *)malloc((size_t)visible_count * sizeof(int32_t));
                if(expected_depth_keys_sorted == NULL || expected_depth_ids_sorted == NULL) {
                    error = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate depth-sort invariant buffers");
                    goto cleanup;
                }

                error = gsx_metal_render_compute_expected_stable_sort(
                    host_depth_keys,
                    host_sorted_primitive_ids,
                    visible_count,
                    expected_depth_keys_sorted,
                    expected_depth_ids_sorted);
                if(!gsx_error_is_success(error)) {
                    goto cleanup;
                }
            }

            error = gsx_metal_backend_dispatch_sort_pairs_u32(
                renderer->backend,
                &depth_keys_view,
                &sorted_primitive_ids_view,
                &depth_keys_sorted_view,
                &sorted_primitive_ids_sorted_view,
                &sort_histogram_view,
                &sort_global_histogram_view,
                &sort_scatter_offsets_view,
                (uint32_t)visible_count);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_backend_major_stream_sync(renderer->backend);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_render_tensor_map_host_bytes(scratch.depth_keys_sorted, (void **)&host_depth_keys_sorted, &mapped_size_bytes);
            if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)gaussian_count * sizeof(int32_t)) {
                error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map sorted depth-key tensor to host-visible bytes");
                goto cleanup;
            }
            error = gsx_metal_render_tensor_map_host_bytes(scratch.sorted_primitive_ids_sorted, (void **)&host_sorted_primitive_ids, &mapped_size_bytes);
            if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)gaussian_count * sizeof(int32_t)) {
                error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to remap sorted primitive-id tensor to host-visible bytes");
                goto cleanup;
            }
            if(dump_debug) {
                gsx_size_t mismatch = (gsx_size_t)-1;
                bool matches = gsx_metal_render_u32_i32_pairs_match(
                    (const uint32_t *)host_depth_keys_sorted,
                    host_sorted_primitive_ids,
                    expected_depth_keys_sorted,
                    expected_depth_ids_sorted,
                    visible_count,
                    &mismatch);

                if(!matches) {
                    fprintf(
                        stderr,
                        "METAL invariant failed: depth sort mismatch at index %zu (metal key=%u id=%d, expected key=%u id=%d)\n",
                        (size_t)mismatch,
                        (unsigned int)((const uint32_t *)host_depth_keys_sorted)[mismatch],
                        host_sorted_primitive_ids[mismatch],
                        (unsigned int)expected_depth_keys_sorted[mismatch],
                        expected_depth_ids_sorted[mismatch]);
                    memcpy(host_depth_keys_sorted, expected_depth_keys_sorted, (size_t)visible_count * sizeof(uint32_t));
                    memcpy(host_sorted_primitive_ids, expected_depth_ids_sorted, (size_t)visible_count * sizeof(int32_t));
                    fprintf(stderr, "METAL invariant fallback: replaced GPU depth sort output with deterministic CPU stable sort\n");
                }
                {
                    uint64_t h = gsx_metal_render_hash_u32_i32_pairs(
                        (const uint32_t *)host_depth_keys_sorted,
                        host_sorted_primitive_ids,
                        visible_count);
                    printf("METAL ordering dump: depth_sort_pairs hash=%llu count=%zu\n", (unsigned long long)h, (size_t)visible_count);
                }
            }
            for(gsx_size_t i = 0; i < visible_count; ++i) {
                int32_t primitive_id = host_sorted_primitive_ids[i];
                host_primitive_offsets[i] = (int32_t)running_instance_offset;
                running_instance_offset += (gsx_size_t)host_touched[(gsx_size_t)primitive_id];
            }
        }

        if(instance_count > 0) {
            gsx_index_t shape_instances[1] = { (gsx_index_t)instance_count };

            error = gsx_metal_render_make_tensor(metal_context->forward_per_instance_arena, GSX_DATA_TYPE_I32, 1, shape_instances, &scratch.instance_keys);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_render_make_tensor(metal_context->forward_per_instance_arena, GSX_DATA_TYPE_I32, 1, shape_instances, &scratch.instance_primitive_ids);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_render_make_tensor(metal_context->forward_per_instance_arena, GSX_DATA_TYPE_I32, 1, shape_instances, &scratch.instance_keys_sorted);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_render_make_tensor(metal_context->forward_per_instance_arena, GSX_DATA_TYPE_I32, 1, shape_instances, &scratch.instance_primitive_ids_sorted);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }

            gsx_metal_render_make_tensor_view(scratch.bounds, &bounds_view);
            gsx_metal_render_make_tensor_view(scratch.instance_keys, &instance_keys_view);
            gsx_metal_render_make_tensor_view(scratch.instance_primitive_ids, &instance_primitive_ids_view);
            gsx_metal_render_make_tensor_view(scratch.instance_keys_sorted, &instance_keys_sorted_view);
            gsx_metal_render_make_tensor_view(scratch.instance_primitive_ids_sorted, &instance_primitive_ids_sorted_view);

            create_params.visible_count = (uint32_t)visible_count;
            create_params.grid_width = (uint32_t)grid_width;
            create_params.grid_height = (uint32_t)grid_height;
            error = gsx_metal_backend_dispatch_render_create_instances(
                renderer->backend,
                &sorted_primitive_ids_sorted_view,
                &primitive_offsets_view,
                &bounds_view,
                &mean2d_view,
                &conic_opacity_view,
                &instance_keys_view,
                &instance_primitive_ids_view,
                &create_params);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }

            error = gsx_backend_major_stream_sync(renderer->backend);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_render_tensor_map_host_bytes(scratch.instance_keys, (void **)&host_instance_keys_unsorted, &mapped_size_bytes);
            if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)instance_count * sizeof(int32_t)) {
                error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map unsorted instance-key tensor to host-visible bytes");
                goto cleanup;
            }
            error = gsx_metal_render_tensor_map_host_bytes(scratch.instance_primitive_ids, (void **)&host_instance_primitive_ids, &mapped_size_bytes);
            if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)instance_count * sizeof(int32_t)) {
                error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map unsorted instance primitive-id tensor to host-visible bytes");
                goto cleanup;
            }

            if(dump_debug) {
                expected_instance_keys_sorted = (uint32_t *)malloc((size_t)instance_count * sizeof(uint32_t));
                expected_instance_ids_sorted = (int32_t *)malloc((size_t)instance_count * sizeof(int32_t));
                if(expected_instance_keys_sorted == NULL || expected_instance_ids_sorted == NULL) {
                    error = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate instance-sort invariant buffers");
                    goto cleanup;
                }
                error = gsx_metal_render_compute_expected_stable_sort(
                    (const uint32_t *)host_instance_keys_unsorted,
                    host_instance_primitive_ids,
                    instance_count,
                    expected_instance_keys_sorted,
                    expected_instance_ids_sorted);
                if(!gsx_error_is_success(error)) {
                    goto cleanup;
                }
            }

            error = gsx_metal_backend_dispatch_sort_pairs_u32(
                renderer->backend,
                &instance_keys_view,
                &instance_primitive_ids_view,
                &instance_keys_sorted_view,
                &instance_primitive_ids_sorted_view,
                &sort_histogram_view,
                &sort_global_histogram_view,
                &sort_scatter_offsets_view,
                (uint32_t)instance_count);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }

            error = gsx_backend_major_stream_sync(renderer->backend);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_render_tensor_map_host_bytes(scratch.instance_keys_sorted, (void **)&host_instance_keys, &mapped_size_bytes);
            if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)instance_count * sizeof(int32_t)) {
                error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map sorted instance key tensor to host-visible bytes");
                goto cleanup;
            }
            error = gsx_metal_render_tensor_map_host_bytes(scratch.instance_primitive_ids_sorted, (void **)&host_instance_primitive_ids, &mapped_size_bytes);
            if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)instance_count * sizeof(int32_t)) {
                error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map sorted instance primitive-id tensor to host-visible bytes");
                goto cleanup;
            }
            if(dump_debug) {
                gsx_size_t mismatch = (gsx_size_t)-1;
                bool matches = gsx_metal_render_u32_i32_pairs_match(
                    (const uint32_t *)host_instance_keys,
                    host_instance_primitive_ids,
                    expected_instance_keys_sorted,
                    expected_instance_ids_sorted,
                    instance_count,
                    &mismatch);

                if(!matches) {
                    fprintf(
                        stderr,
                        "METAL invariant failed: instance sort mismatch at index %zu (metal key=%u id=%d, expected key=%u id=%d)\n",
                        (size_t)mismatch,
                        (unsigned int)((const uint32_t *)host_instance_keys)[mismatch],
                        host_instance_primitive_ids[mismatch],
                        (unsigned int)expected_instance_keys_sorted[mismatch],
                        expected_instance_ids_sorted[mismatch]);
                    memcpy(host_instance_keys, expected_instance_keys_sorted, (size_t)instance_count * sizeof(uint32_t));
                    memcpy(host_instance_primitive_ids, expected_instance_ids_sorted, (size_t)instance_count * sizeof(int32_t));
                    fprintf(stderr, "METAL invariant fallback: replaced GPU instance sort output with deterministic CPU stable sort\n");
                }
                {
                    uint64_t h = gsx_metal_render_hash_u32_i32_pairs(
                        (const uint32_t *)host_instance_keys,
                        host_instance_primitive_ids,
                        instance_count);
                    printf("METAL ordering dump: instance_sort_pairs hash=%llu count=%zu\n", (unsigned long long)h, (size_t)instance_count);
                }
            }
        }

        if(tile_count > 0) {
            gsx_index_t shape_tile_ranges[2] = { (gsx_index_t)tile_count, 2 };
            gsx_index_t shape_tile_bucket_offsets[1] = { (gsx_index_t)tile_count };

            error = gsx_metal_render_make_tensor(metal_context->forward_per_tile_arena, GSX_DATA_TYPE_I32, 2, shape_tile_ranges, &scratch.tile_ranges);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_render_tensor_map_host_bytes(scratch.tile_ranges, (void **)&host_tile_ranges, &mapped_size_bytes);
            if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)tile_count * 2u * sizeof(int32_t)) {
                error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map tile-range tensor to host-visible bytes");
                goto cleanup;
            }
            gsx_metal_render_fill_tile_ranges(
                host_instance_keys, instance_count, host_tile_ranges, tile_count);

            error = gsx_metal_render_make_tensor(metal_context->forward_per_tile_arena, GSX_DATA_TYPE_I32, 1, shape_tile_bucket_offsets, &scratch.tile_bucket_offsets);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_render_tensor_map_host_bytes(scratch.tile_bucket_offsets, (void **)&host_tile_bucket_offsets, &mapped_size_bytes);
            if(!gsx_error_is_success(error) || mapped_size_bytes < (gsx_size_t)tile_count * sizeof(int32_t)) {
                error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map tile-bucket-offset tensor to host-visible bytes");
                goto cleanup;
            }
            bucket_count = gsx_metal_render_fill_tile_bucket_offsets(host_tile_ranges, tile_count, host_tile_bucket_offsets);

            if(dump_debug) {
                error = gsx_metal_render_validate_sorted_tile_ranges(host_instance_keys, instance_count, host_tile_ranges, tile_count, dump_debug);
                if(!gsx_error_is_success(error)) {
                    goto cleanup;
                }
                uint64_t tile_hash = gsx_metal_render_hash_u32_i32_pairs(
                    (const uint32_t *)host_instance_keys,
                    host_instance_primitive_ids,
                    instance_count);
                printf(
                    "METAL ordering dump: tile_ranges hash=%llu bucket_count=%zu last_bucket_offset=%d\n",
                    (unsigned long long)tile_hash,
                    (size_t)bucket_count,
                    tile_count > 0u ? host_tile_bucket_offsets[tile_count - 1u] : 0);
            }
        }

        if(bucket_count > 0) {
            gsx_index_t shape_bucket_tile_index[1] = { (gsx_index_t)bucket_count };
            gsx_index_t shape_bucket_color_transmittance[2] = { (gsx_index_t)(bucket_count * 256u), 4 };

            error = gsx_metal_render_make_tensor(
                metal_context->forward_per_bucket_arena,
                GSX_DATA_TYPE_I32,
                1,
                shape_bucket_tile_index,
                &scratch.bucket_tile_index);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_render_make_tensor(
                metal_context->forward_per_bucket_arena,
                GSX_DATA_TYPE_F32,
                2,
                shape_bucket_color_transmittance,
                &scratch.bucket_color_transmittance);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_tensor_set_zero(scratch.bucket_color_transmittance);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
        }

        if(tile_count > 0) {
            gsx_index_t shape_tile_max_n_contributions[1] = { (gsx_index_t)tile_count };
            gsx_index_t shape_tile_n_contributions[2] = { renderer->info.height, renderer->info.width };

            error = gsx_metal_render_make_tensor(
                metal_context->forward_per_tile_arena,
                GSX_DATA_TYPE_I32,
                1,
                shape_tile_max_n_contributions,
                &scratch.tile_max_n_contributions);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_tensor_set_zero(scratch.tile_max_n_contributions);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }

            error = gsx_metal_render_make_tensor(
                metal_context->forward_per_tile_arena,
                GSX_DATA_TYPE_I32,
                2,
                shape_tile_n_contributions,
                &scratch.tile_n_contributions);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_tensor_set_zero(scratch.tile_n_contributions);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
        }

        if(instance_count > 0 && tile_count > 0 && bucket_count > 0) {
            if(!gsx_metal_render_tensor_is_backed_i32(scratch.tile_ranges)
                || !gsx_metal_render_tensor_is_backed_i32(scratch.tile_bucket_offsets)
                || !gsx_metal_render_tensor_is_backed_i32(scratch.instance_primitive_ids_sorted)
                || !gsx_metal_render_tensor_is_backed_i32(scratch.tile_max_n_contributions)
                || !gsx_metal_render_tensor_is_backed_i32(scratch.tile_n_contributions)
                || !gsx_metal_render_tensor_is_backed_i32(scratch.bucket_tile_index)
                || !gsx_metal_render_tensor_is_backed_f32(scratch.mean2d)
                || !gsx_metal_render_tensor_is_backed_f32(scratch.conic_opacity)
                || !gsx_metal_render_tensor_is_backed_f32(scratch.color)
                || !gsx_metal_render_tensor_is_backed_f32(scratch.bucket_color_transmittance)) {
                error = gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "metal blend staging tensors are not in required device formats");
                goto cleanup;
            }

            gsx_metal_render_make_tensor_view(scratch.tile_ranges, &tile_ranges_view);
            gsx_metal_render_make_tensor_view(scratch.tile_bucket_offsets, &tile_bucket_offsets_view);
            gsx_metal_render_make_tensor_view(scratch.instance_primitive_ids_sorted, &instance_primitive_ids_view);
            gsx_metal_render_make_tensor_view(scratch.mean2d, &mean2d_view);
            gsx_metal_render_make_tensor_view(scratch.conic_opacity, &conic_opacity_view);
            gsx_metal_render_make_tensor_view(scratch.color, &color_view);
            gsx_metal_render_make_tensor_view(metal_context->helper_image_chw, &image_view);
            gsx_metal_render_make_tensor_view(metal_context->helper_alpha_hw, &alpha_view);
            gsx_metal_render_make_tensor_view(scratch.tile_max_n_contributions, &tile_max_n_contributions_view);
            gsx_metal_render_make_tensor_view(scratch.tile_n_contributions, &tile_n_contributions_view);
            gsx_metal_render_make_tensor_view(scratch.bucket_tile_index, &bucket_tile_index_view);
            gsx_metal_render_make_tensor_view(scratch.bucket_color_transmittance, &bucket_color_transmittance_view);

            blend_params.width = (uint32_t)renderer->info.width;
            blend_params.height = (uint32_t)renderer->info.height;
            blend_params.grid_width = (uint32_t)grid_width;
            blend_params.grid_height = (uint32_t)grid_height;
            blend_params.tile_count = (uint32_t)tile_count;
            blend_params.channel_stride = (uint32_t)gsx_metal_render_get_channel_stride(renderer->info.width, renderer->info.height);

            error = gsx_metal_backend_dispatch_render_blend(
                renderer->backend,
                &tile_ranges_view,
                &tile_bucket_offsets_view,
                &instance_primitive_ids_view,
                &mean2d_view,
                &conic_opacity_view,
                &color_view,
                &image_view,
                &alpha_view,
                &tile_max_n_contributions_view,
                &tile_n_contributions_view,
                &bucket_tile_index_view,
                &bucket_color_transmittance_view,
                &blend_params);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
        }
    }

    gsx_metal_render_make_tensor_view(metal_context->helper_image_chw, &image_view);
    gsx_metal_render_make_tensor_view(metal_context->helper_alpha_hw, &alpha_view);
    gsx_metal_render_make_tensor_view(request->out_rgb, &out_view);

    compose_params.width = (uint32_t)renderer->info.width;
    compose_params.height = (uint32_t)renderer->info.height;
    compose_params.channel_stride = (uint32_t)gsx_metal_render_get_channel_stride(renderer->info.width, renderer->info.height);
    compose_params.background_r = request->background_color.x;
    compose_params.background_g = request->background_color.y;
    compose_params.background_b = request->background_color.z;
    error = gsx_metal_backend_dispatch_render_compose_f32(
        renderer->backend,
        &image_view,
        &alpha_view,
        &out_view,
        &compose_params);
    if(!gsx_error_is_success(error)) {
        goto cleanup;
    }
    error = gsx_backend_major_stream_sync(renderer->backend);

    if(gsx_error_is_success(error) && request->forward_type == GSX_RENDER_FORWARD_TYPE_TRAIN) {
        error = gsx_metal_render_context_snapshot_train_state(
            metal_context,
            request,
            scratch.mean2d,
            scratch.conic_opacity,
            scratch.color,
            scratch.instance_primitive_ids_sorted,
            scratch.tile_ranges,
            scratch.tile_bucket_offsets,
            scratch.bucket_tile_index,
            scratch.bucket_color_transmittance,
            scratch.tile_max_n_contributions,
            scratch.tile_n_contributions,
            (uint32_t)bucket_count);
    }

cleanup:
    free(expected_instance_ids_sorted);
    free(expected_instance_keys_sorted);
    free(expected_depth_ids_sorted);
    free(expected_depth_keys_sorted);
    gsx_metal_render_cleanup_forward_scratch(&scratch);
    (void)gsx_arena_reset(metal_context->forward_per_primitive_arena);
    (void)gsx_arena_reset(metal_context->forward_per_tile_arena);
    (void)gsx_arena_reset(metal_context->forward_per_instance_arena);
    (void)gsx_arena_reset(metal_context->forward_per_bucket_arena);
    return error;
}
