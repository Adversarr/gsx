#include "../internal.h"

#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct gsx_metal_sort_pair_u32_i32 {
    uint32_t key;
    int32_t value;
    uint32_t stable_index;
} gsx_metal_sort_pair_u32_i32;

typedef struct gsx_metal_forward_scratch {
    gsx_tensor_t depth_keys;
    gsx_tensor_t depth_keys_sorted;
    gsx_tensor_t visible_primitive_ids;
    gsx_tensor_t visible_primitive_ids_sorted;
    gsx_tensor_t touched_tiles;
    gsx_tensor_t primitive_offsets;
    gsx_tensor_t bounds;
    gsx_tensor_t mean2d;
    gsx_tensor_t conic_opacity;
    gsx_tensor_t color;
    gsx_tensor_t visible_count;
    gsx_tensor_t instance_count;
    gsx_tensor_t visible_sort_histogram;
    gsx_tensor_t visible_sort_global_histogram;
    gsx_tensor_t visible_sort_scatter_offsets;
    gsx_tensor_t primitive_scan_block_sums;
    gsx_tensor_t primitive_scan_scanned_block_sums;
    gsx_tensor_t instance_keys;
    gsx_tensor_t instance_primitive_ids;
    gsx_tensor_t instance_keys_sorted;
    gsx_tensor_t instance_primitive_ids_sorted;
    gsx_tensor_t instance_sort_histogram;
    gsx_tensor_t instance_sort_global_histogram;
    gsx_tensor_t instance_sort_scatter_offsets;
    gsx_tensor_t tile_ranges;
    gsx_tensor_t tile_bucket_counts;
    gsx_tensor_t tile_bucket_offsets;
    gsx_tensor_t tile_max_n_contributions;
    gsx_tensor_t tile_n_contributions;
    gsx_tensor_t tile_scan_block_sums;
    gsx_tensor_t tile_scan_scanned_block_sums;
    gsx_tensor_t bucket_tile_index;
    gsx_tensor_t bucket_color_transmittance;
} gsx_metal_forward_scratch;

typedef struct gsx_metal_forward_primitive_plan {
    gsx_size_t gaussian_count;
} gsx_metal_forward_primitive_plan;

typedef struct gsx_metal_forward_tile_plan {
    gsx_index_t width;
    gsx_index_t height;
    gsx_size_t tile_count;
} gsx_metal_forward_tile_plan;

typedef struct gsx_metal_forward_instance_plan {
    gsx_size_t instance_count;
} gsx_metal_forward_instance_plan;

typedef struct gsx_metal_forward_bucket_plan {
    gsx_size_t bucket_count;
} gsx_metal_forward_bucket_plan;

static bool gsx_metal_render_debug_dump_enabled(gsx_size_t gaussian_count)
{
    const char *env = getenv("GSX_METAL_FORWARD_DUMP");

    (void)gaussian_count;

    return env != NULL && env[0] != '\0' && env[0] != '0';
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
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate stable-sort scratch");
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
    gsx_size_t tile_count)
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

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_render_validate_bucket_counts(
    const int32_t *tile_ranges,
    gsx_size_t tile_count,
    const int32_t *tile_bucket_counts,
    const int32_t *tile_bucket_offsets,
    gsx_size_t *out_expected_bucket_count,
    gsx_size_t *out_mismatch_tile,
    int32_t *out_expected_count,
    int32_t *out_actual_count,
    int32_t *out_expected_offset,
    int32_t *out_actual_offset)
{
    gsx_size_t expected_bucket_count = 0;

    if(tile_ranges == NULL || tile_bucket_counts == NULL || tile_bucket_offsets == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "bucket-count validation inputs must be non-null");
    }
    if(out_mismatch_tile != NULL) {
        *out_mismatch_tile = (gsx_size_t)-1;
    }
    if(out_expected_count != NULL) {
        *out_expected_count = 0;
    }
    if(out_actual_count != NULL) {
        *out_actual_count = 0;
    }
    if(out_expected_offset != NULL) {
        *out_expected_offset = 0;
    }
    if(out_actual_offset != NULL) {
        *out_actual_offset = 0;
    }

    for(gsx_size_t tile = 0; tile < tile_count; ++tile) {
        int32_t start = tile_ranges[tile * 2u];
        int32_t end = tile_ranges[tile * 2u + 1u];
        int32_t expected_count = 0;
        int32_t expected_offset = 0;

        if(start < 0 || end < start) {
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "bucket-count validation saw invalid tile range");
        }

        expected_count = (end - start + 31) / 32;
        expected_bucket_count += (gsx_size_t)expected_count;
        expected_offset = (int32_t)expected_bucket_count;

        if(tile_bucket_counts[tile] != expected_count) {
            if(out_mismatch_tile != NULL) {
                *out_mismatch_tile = tile;
            }
            if(out_expected_count != NULL) {
                *out_expected_count = expected_count;
            }
            if(out_actual_count != NULL) {
                *out_actual_count = tile_bucket_counts[tile];
            }
            if(out_expected_offset != NULL) {
                *out_expected_offset = expected_offset;
            }
            if(out_actual_offset != NULL) {
                *out_actual_offset = tile_bucket_offsets[tile];
            }
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "tile_bucket_counts does not match tile_ranges");
        }
        if(tile_bucket_offsets[tile] != expected_offset) {
            if(out_mismatch_tile != NULL) {
                *out_mismatch_tile = tile;
            }
            if(out_expected_count != NULL) {
                *out_expected_count = expected_count;
            }
            if(out_actual_count != NULL) {
                *out_actual_count = tile_bucket_counts[tile];
            }
            if(out_expected_offset != NULL) {
                *out_expected_offset = expected_offset;
            }
            if(out_actual_offset != NULL) {
                *out_actual_offset = tile_bucket_offsets[tile];
            }
            return gsx_make_error(GSX_ERROR_INVALID_STATE, "tile_bucket_offsets does not match inclusive bucket prefix sum");
        }
    }

    if(out_expected_bucket_count != NULL) {
        *out_expected_bucket_count = expected_bucket_count;
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_render_read_tensor_u32(gsx_tensor_t tensor, uint32_t *out_value)
{
    void *mapped_bytes = NULL;
    gsx_size_t mapped_size_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(tensor == NULL || out_value == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor and out_value must be non-null");
    }

    error = gsx_metal_render_tensor_map_host_bytes(tensor, &mapped_bytes, &mapped_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(mapped_size_bytes < sizeof(uint32_t)) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "mapped scalar tensor is smaller than uint32_t");
    }

    *out_value = *(const uint32_t *)mapped_bytes;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_render_read_tensor_u32_at(gsx_tensor_t tensor, gsx_size_t index, uint32_t *out_value)
{
    void *mapped_bytes = NULL;
    gsx_size_t mapped_size_bytes = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(tensor == NULL || out_value == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tensor and out_value must be non-null");
    }

    error = gsx_metal_render_tensor_map_host_bytes(tensor, &mapped_bytes, &mapped_size_bytes);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    if(mapped_size_bytes < (index + 1u) * sizeof(uint32_t)) {
        return gsx_make_error(GSX_ERROR_INVALID_STATE, "mapped tensor is smaller than requested scalar index");
    }

    *out_value = ((const uint32_t *)mapped_bytes)[index];
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

static gsx_error gsx_metal_render_sort_pairs_u32_or_copy(
    gsx_backend_t backend,
    const gsx_backend_tensor_view *keys_in_view,
    const gsx_backend_tensor_view *values_in_view,
    const gsx_backend_tensor_view *keys_out_view,
    const gsx_backend_tensor_view *values_out_view,
    const gsx_backend_tensor_view *histogram_view,
    const gsx_backend_tensor_view *global_histogram_view,
    const gsx_backend_tensor_view *scatter_offsets_view,
    uint32_t count)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(backend == NULL || keys_in_view == NULL || values_in_view == NULL || keys_out_view == NULL || values_out_view == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "sort-or-copy tensor views must be non-null");
    }

    if(count <= 1u) {
        error = gsx_metal_backend_buffer_copy_tensor(keys_out_view->buffer, keys_in_view, keys_out_view);
        if(!gsx_error_is_success(error)) {
            return error;
        }
        return gsx_metal_backend_buffer_copy_tensor(values_out_view->buffer, values_in_view, values_out_view);
    }

    return gsx_metal_backend_dispatch_sort_pairs_u32(
        backend,
        keys_in_view,
        values_in_view,
        keys_out_view,
        values_out_view,
        histogram_view,
        global_histogram_view,
        scatter_offsets_view,
        count);
}

static void gsx_metal_render_cleanup_forward_scratch(gsx_metal_forward_scratch *scratch)
{
    gsx_metal_render_release_tensor(&scratch->bucket_color_transmittance);
    gsx_metal_render_release_tensor(&scratch->bucket_tile_index);
    gsx_metal_render_release_tensor(&scratch->tile_scan_scanned_block_sums);
    gsx_metal_render_release_tensor(&scratch->tile_scan_block_sums);
    gsx_metal_render_release_tensor(&scratch->tile_n_contributions);
    gsx_metal_render_release_tensor(&scratch->tile_max_n_contributions);
    gsx_metal_render_release_tensor(&scratch->tile_bucket_offsets);
    gsx_metal_render_release_tensor(&scratch->tile_bucket_counts);
    gsx_metal_render_release_tensor(&scratch->tile_ranges);
    gsx_metal_render_release_tensor(&scratch->instance_sort_scatter_offsets);
    gsx_metal_render_release_tensor(&scratch->instance_sort_global_histogram);
    gsx_metal_render_release_tensor(&scratch->instance_sort_histogram);
    gsx_metal_render_release_tensor(&scratch->instance_primitive_ids_sorted);
    gsx_metal_render_release_tensor(&scratch->instance_keys_sorted);
    gsx_metal_render_release_tensor(&scratch->instance_primitive_ids);
    gsx_metal_render_release_tensor(&scratch->instance_keys);
    gsx_metal_render_release_tensor(&scratch->primitive_scan_scanned_block_sums);
    gsx_metal_render_release_tensor(&scratch->primitive_scan_block_sums);
    gsx_metal_render_release_tensor(&scratch->visible_sort_scatter_offsets);
    gsx_metal_render_release_tensor(&scratch->visible_sort_global_histogram);
    gsx_metal_render_release_tensor(&scratch->visible_sort_histogram);
    gsx_metal_render_release_tensor(&scratch->instance_count);
    gsx_metal_render_release_tensor(&scratch->visible_count);
    gsx_metal_render_release_tensor(&scratch->color);
    gsx_metal_render_release_tensor(&scratch->conic_opacity);
    gsx_metal_render_release_tensor(&scratch->mean2d);
    gsx_metal_render_release_tensor(&scratch->bounds);
    gsx_metal_render_release_tensor(&scratch->primitive_offsets);
    gsx_metal_render_release_tensor(&scratch->touched_tiles);
    gsx_metal_render_release_tensor(&scratch->visible_primitive_ids_sorted);
    gsx_metal_render_release_tensor(&scratch->visible_primitive_ids);
    gsx_metal_render_release_tensor(&scratch->depth_keys_sorted);
    gsx_metal_render_release_tensor(&scratch->depth_keys);
}

static gsx_error gsx_metal_render_plan_forward_primitive_scratch(gsx_arena_t dry_run_arena, void *user_data)
{
    gsx_metal_forward_primitive_plan *plan = (gsx_metal_forward_primitive_plan *)user_data;
    gsx_metal_forward_scratch scratch = { 0 };
    gsx_size_t sort_threadgroup_count = 0;
    gsx_size_t scan_block_count = 0;
    gsx_index_t shape_n[1];
    gsx_index_t shape_n2[2];
    gsx_index_t shape_n3[2];
    gsx_index_t shape_n4[2];
    gsx_index_t shape_one[1] = { 1 };
    gsx_index_t shape_sort_histogram[1];
    gsx_index_t shape_global_histogram[1] = { 256 };
    gsx_index_t shape_scan_block_sums[1];
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dry_run_arena == NULL || plan == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dry-run primitive plan arguments must be non-null");
    }
    if(plan->gaussian_count == 0u) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    shape_n[0] = (gsx_index_t)plan->gaussian_count;
    shape_n2[0] = (gsx_index_t)plan->gaussian_count;
    shape_n2[1] = 2;
    shape_n3[0] = (gsx_index_t)plan->gaussian_count;
    shape_n3[1] = 3;
    shape_n4[0] = (gsx_index_t)plan->gaussian_count;
    shape_n4[1] = 4;
    sort_threadgroup_count = (plan->gaussian_count + 1023u) / 1024u;
    scan_block_count = (plan->gaussian_count + 255u) / 256u;
    shape_sort_histogram[0] = (gsx_index_t)(sort_threadgroup_count * 256u);
    shape_scan_block_sums[0] = (gsx_index_t)scan_block_count;

    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.depth_keys);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.depth_keys_sorted);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.visible_primitive_ids);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.visible_primitive_ids_sorted);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.touched_tiles);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.primitive_offsets);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_F32, 2, shape_n4, &scratch.bounds);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_F32, 2, shape_n2, &scratch.mean2d);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_F32, 2, shape_n4, &scratch.conic_opacity);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_F32, 2, shape_n3, &scratch.color);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_one, &scratch.visible_count);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_one, &scratch.instance_count);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_sort_histogram, &scratch.visible_sort_histogram);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_global_histogram, &scratch.visible_sort_global_histogram);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_sort_histogram, &scratch.visible_sort_scatter_offsets);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_scan_block_sums, &scratch.primitive_scan_block_sums);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_scan_block_sums, &scratch.primitive_scan_scanned_block_sums);
    if(!gsx_error_is_success(error)) goto cleanup;

cleanup:
    gsx_metal_render_cleanup_forward_scratch(&scratch);
    return error;
}

static gsx_error gsx_metal_render_plan_forward_tile_scratch(gsx_arena_t dry_run_arena, void *user_data)
{
    gsx_metal_forward_tile_plan *plan = (gsx_metal_forward_tile_plan *)user_data;
    gsx_metal_forward_scratch scratch = { 0 };
    gsx_size_t scan_block_count = 0;
    gsx_index_t shape_tile_ranges[2];
    gsx_index_t shape_tile_counts[1];
    gsx_index_t shape_tile_n_contributions[2];
    gsx_index_t shape_scan_block_sums[1];
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dry_run_arena == NULL || plan == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dry-run tile plan arguments must be non-null");
    }
    if(plan->tile_count == 0u) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    shape_tile_ranges[0] = (gsx_index_t)plan->tile_count;
    shape_tile_ranges[1] = 2;
    shape_tile_counts[0] = (gsx_index_t)plan->tile_count;
    shape_tile_n_contributions[0] = plan->height;
    shape_tile_n_contributions[1] = plan->width;
    scan_block_count = (plan->tile_count + 255u) / 256u;
    shape_scan_block_sums[0] = (gsx_index_t)scan_block_count;

    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 2, shape_tile_ranges, &scratch.tile_ranges);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_tile_counts, &scratch.tile_bucket_counts);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_tile_counts, &scratch.tile_bucket_offsets);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_tile_counts, &scratch.tile_max_n_contributions);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 2, shape_tile_n_contributions, &scratch.tile_n_contributions);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_scan_block_sums, &scratch.tile_scan_block_sums);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_scan_block_sums, &scratch.tile_scan_scanned_block_sums);
    if(!gsx_error_is_success(error)) goto cleanup;

cleanup:
    gsx_metal_render_cleanup_forward_scratch(&scratch);
    return error;
}

static gsx_error gsx_metal_render_plan_forward_instance_scratch(gsx_arena_t dry_run_arena, void *user_data)
{
    gsx_metal_forward_instance_plan *plan = (gsx_metal_forward_instance_plan *)user_data;
    gsx_metal_forward_scratch scratch = { 0 };
    gsx_size_t sort_threadgroup_count = 0;
    gsx_index_t shape_n[1];
    gsx_index_t shape_sort_histogram[1];
    gsx_index_t shape_global_histogram[1] = { 256 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dry_run_arena == NULL || plan == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dry-run instance plan arguments must be non-null");
    }
    if(plan->instance_count == 0u) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    shape_n[0] = (gsx_index_t)plan->instance_count;
    sort_threadgroup_count = (plan->instance_count + 1023u) / 1024u;
    shape_sort_histogram[0] = (gsx_index_t)(sort_threadgroup_count * 256u);

    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.instance_keys);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.instance_primitive_ids);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.instance_keys_sorted);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch.instance_primitive_ids_sorted);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_sort_histogram, &scratch.instance_sort_histogram);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_global_histogram, &scratch.instance_sort_global_histogram);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_sort_histogram, &scratch.instance_sort_scatter_offsets);
    if(!gsx_error_is_success(error)) goto cleanup;

cleanup:
    gsx_metal_render_cleanup_forward_scratch(&scratch);
    return error;
}

static gsx_error gsx_metal_render_plan_forward_bucket_scratch(gsx_arena_t dry_run_arena, void *user_data)
{
    gsx_metal_forward_bucket_plan *plan = (gsx_metal_forward_bucket_plan *)user_data;
    gsx_metal_forward_scratch scratch = { 0 };
    gsx_size_t bucket_entry_count = 0;
    gsx_index_t shape_bucket_tile_index[1];
    gsx_index_t shape_bucket_color_transmittance[2];
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(dry_run_arena == NULL || plan == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "dry-run bucket plan arguments must be non-null");
    }
    if(plan->bucket_count == 0u) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(gsx_size_mul_overflows(plan->bucket_count, 256u, &bucket_entry_count)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "metal forward bucket-entry sizing overflow");
    }

    shape_bucket_tile_index[0] = (gsx_index_t)plan->bucket_count;
    shape_bucket_color_transmittance[0] = (gsx_index_t)bucket_entry_count;
    shape_bucket_color_transmittance[1] = 4;

    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_I32, 1, shape_bucket_tile_index, &scratch.bucket_tile_index);
    if(!gsx_error_is_success(error)) goto cleanup;
    error = gsx_metal_render_make_tensor(dry_run_arena, GSX_DATA_TYPE_F32, 2, shape_bucket_color_transmittance, &scratch.bucket_color_transmittance);
    if(!gsx_error_is_success(error)) goto cleanup;

cleanup:
    gsx_metal_render_cleanup_forward_scratch(&scratch);
    return error;
}

static gsx_error gsx_metal_render_reserve_forward_primitive_scratch(
    gsx_metal_render_context *metal_context,
    gsx_size_t gaussian_count)
{
    gsx_metal_forward_primitive_plan plan = { gaussian_count };

    if(metal_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_context must be non-null");
    }

    return gsx_metal_render_reserve_arena_with_dry_run(
        metal_context->forward_per_primitive_arena,
        gsx_metal_render_plan_forward_primitive_scratch,
        &plan);
}

static gsx_error gsx_metal_render_reserve_forward_tile_scratch(
    gsx_metal_render_context *metal_context,
    gsx_index_t width,
    gsx_index_t height,
    gsx_size_t tile_count)
{
    gsx_metal_forward_tile_plan plan = { width, height, tile_count };

    if(metal_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_context must be non-null");
    }

    return gsx_metal_render_reserve_arena_with_dry_run(
        metal_context->forward_per_tile_arena,
        gsx_metal_render_plan_forward_tile_scratch,
        &plan);
}

static gsx_error gsx_metal_render_reserve_forward_instance_scratch(
    gsx_metal_render_context *metal_context,
    gsx_size_t instance_count)
{
    gsx_metal_forward_instance_plan plan = { instance_count };

    if(metal_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_context must be non-null");
    }

    return gsx_metal_render_reserve_arena_with_dry_run(
        metal_context->forward_per_instance_arena,
        gsx_metal_render_plan_forward_instance_scratch,
        &plan);
}

static gsx_error gsx_metal_render_reserve_forward_bucket_scratch(
    gsx_metal_render_context *metal_context,
    gsx_size_t bucket_count)
{
    gsx_metal_forward_bucket_plan plan = { bucket_count };

    if(metal_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_context must be non-null");
    }

    return gsx_metal_render_reserve_arena_with_dry_run(
        metal_context->forward_per_bucket_arena,
        gsx_metal_render_plan_forward_bucket_scratch,
        &plan);
}

static gsx_error gsx_metal_render_alloc_forward_primitive_scratch(
    gsx_metal_render_context *metal_context,
    gsx_size_t gaussian_count,
    gsx_metal_forward_scratch *scratch)
{
    gsx_size_t sort_threadgroup_count = 0;
    gsx_size_t scan_block_count = 0;
    gsx_index_t shape_n[1];
    gsx_index_t shape_n2[2];
    gsx_index_t shape_n3[2];
    gsx_index_t shape_n4[2];
    gsx_index_t shape_one[1] = { 1 };
    gsx_index_t shape_sort_histogram[1];
    gsx_index_t shape_global_histogram[1] = { 256 };
    gsx_index_t shape_scan_block_sums[1];
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_context == NULL || scratch == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "primitive scratch allocation arguments must be non-null");
    }
    if(gaussian_count == 0u) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    shape_n[0] = (gsx_index_t)gaussian_count;
    shape_n2[0] = (gsx_index_t)gaussian_count;
    shape_n2[1] = 2;
    shape_n3[0] = (gsx_index_t)gaussian_count;
    shape_n3[1] = 3;
    shape_n4[0] = (gsx_index_t)gaussian_count;
    shape_n4[1] = 4;
    sort_threadgroup_count = (gaussian_count + 1023u) / 1024u;
    scan_block_count = (gaussian_count + 255u) / 256u;
    shape_sort_histogram[0] = (gsx_index_t)(sort_threadgroup_count * 256u);
    shape_scan_block_sums[0] = (gsx_index_t)scan_block_count;

    error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch->depth_keys);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch->depth_keys_sorted);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch->visible_primitive_ids);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch->visible_primitive_ids_sorted);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch->touched_tiles);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch->primitive_offsets);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_F32, 2, shape_n4, &scratch->bounds);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_F32, 2, shape_n2, &scratch->mean2d);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_F32, 2, shape_n4, &scratch->conic_opacity);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_F32, 2, shape_n3, &scratch->color);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_one, &scratch->visible_count);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_one, &scratch->instance_count);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_sort_histogram, &scratch->visible_sort_histogram);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_global_histogram, &scratch->visible_sort_global_histogram);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_sort_histogram, &scratch->visible_sort_scatter_offsets);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_scan_block_sums, &scratch->primitive_scan_block_sums);
    if(!gsx_error_is_success(error)) return error;
    return gsx_metal_render_make_tensor(metal_context->forward_per_primitive_arena, GSX_DATA_TYPE_I32, 1, shape_scan_block_sums, &scratch->primitive_scan_scanned_block_sums);
}

static gsx_error gsx_metal_render_alloc_forward_tile_scratch(
    gsx_metal_render_context *metal_context,
    gsx_index_t width,
    gsx_index_t height,
    gsx_size_t tile_count,
    gsx_metal_forward_scratch *scratch)
{
    gsx_size_t scan_block_count = 0;
    gsx_index_t shape_tile_ranges[2];
    gsx_index_t shape_tile_counts[1];
    gsx_index_t shape_tile_n_contributions[2];
    gsx_index_t shape_scan_block_sums[1];
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_context == NULL || scratch == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "tile scratch allocation arguments must be non-null");
    }
    if(tile_count == 0u) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    shape_tile_ranges[0] = (gsx_index_t)tile_count;
    shape_tile_ranges[1] = 2;
    shape_tile_counts[0] = (gsx_index_t)tile_count;
    shape_tile_n_contributions[0] = height;
    shape_tile_n_contributions[1] = width;
    scan_block_count = (tile_count + 255u) / 256u;
    shape_scan_block_sums[0] = (gsx_index_t)scan_block_count;

    error = gsx_metal_render_make_tensor(metal_context->forward_per_tile_arena, GSX_DATA_TYPE_I32, 2, shape_tile_ranges, &scratch->tile_ranges);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_tile_arena, GSX_DATA_TYPE_I32, 1, shape_tile_counts, &scratch->tile_bucket_counts);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_tile_arena, GSX_DATA_TYPE_I32, 1, shape_tile_counts, &scratch->tile_bucket_offsets);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_tile_arena, GSX_DATA_TYPE_I32, 1, shape_tile_counts, &scratch->tile_max_n_contributions);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_tile_arena, GSX_DATA_TYPE_I32, 2, shape_tile_n_contributions, &scratch->tile_n_contributions);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_tile_arena, GSX_DATA_TYPE_I32, 1, shape_scan_block_sums, &scratch->tile_scan_block_sums);
    if(!gsx_error_is_success(error)) return error;
    return gsx_metal_render_make_tensor(metal_context->forward_per_tile_arena, GSX_DATA_TYPE_I32, 1, shape_scan_block_sums, &scratch->tile_scan_scanned_block_sums);
}

static gsx_error gsx_metal_render_alloc_forward_instance_scratch(
    gsx_metal_render_context *metal_context,
    gsx_size_t instance_count,
    gsx_metal_forward_scratch *scratch)
{
    gsx_size_t sort_threadgroup_count = 0;
    gsx_index_t shape_n[1];
    gsx_index_t shape_sort_histogram[1];
    gsx_index_t shape_global_histogram[1] = { 256 };
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_context == NULL || scratch == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "instance scratch allocation arguments must be non-null");
    }
    if(instance_count == 0u) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }

    shape_n[0] = (gsx_index_t)instance_count;
    sort_threadgroup_count = (instance_count + 1023u) / 1024u;
    shape_sort_histogram[0] = (gsx_index_t)(sort_threadgroup_count * 256u);

    error = gsx_metal_render_make_tensor(metal_context->forward_per_instance_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch->instance_keys);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_instance_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch->instance_primitive_ids);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_instance_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch->instance_keys_sorted);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_instance_arena, GSX_DATA_TYPE_I32, 1, shape_n, &scratch->instance_primitive_ids_sorted);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_instance_arena, GSX_DATA_TYPE_I32, 1, shape_sort_histogram, &scratch->instance_sort_histogram);
    if(!gsx_error_is_success(error)) return error;
    error = gsx_metal_render_make_tensor(metal_context->forward_per_instance_arena, GSX_DATA_TYPE_I32, 1, shape_global_histogram, &scratch->instance_sort_global_histogram);
    if(!gsx_error_is_success(error)) return error;
    return gsx_metal_render_make_tensor(metal_context->forward_per_instance_arena, GSX_DATA_TYPE_I32, 1, shape_sort_histogram, &scratch->instance_sort_scatter_offsets);
}

static gsx_error gsx_metal_render_alloc_forward_bucket_scratch(
    gsx_metal_render_context *metal_context,
    gsx_size_t bucket_count,
    gsx_metal_forward_scratch *scratch)
{
    gsx_size_t bucket_entry_count = 0;
    gsx_index_t shape_bucket_tile_index[1];
    gsx_index_t shape_bucket_color_transmittance[2];
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_context == NULL || scratch == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "bucket scratch allocation arguments must be non-null");
    }
    if(bucket_count == 0u) {
        return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
    }
    if(gsx_size_mul_overflows(bucket_count, 256u, &bucket_entry_count)) {
        return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "metal forward bucket-entry sizing overflow");
    }

    shape_bucket_tile_index[0] = (gsx_index_t)bucket_count;
    shape_bucket_color_transmittance[0] = (gsx_index_t)bucket_entry_count;
    shape_bucket_color_transmittance[1] = 4;

    error = gsx_metal_render_make_tensor(metal_context->forward_per_bucket_arena, GSX_DATA_TYPE_I32, 1, shape_bucket_tile_index, &scratch->bucket_tile_index);
    if(!gsx_error_is_success(error)) {
        return error;
    }
    return gsx_metal_render_make_tensor(metal_context->forward_per_bucket_arena, GSX_DATA_TYPE_F32, 2, shape_bucket_color_transmittance, &scratch->bucket_color_transmittance);
}

static gsx_error gsx_metal_render_prepare_forward_context(gsx_metal_render_context *metal_context)
{
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    if(metal_context == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "metal_context must be non-null");
    }

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
    gsx_metal_forward_scratch scratch = { 0 };
    gsx_backend_tensor_view mean3d_in_view = { 0 };
    gsx_backend_tensor_view rotation_in_view = { 0 };
    gsx_backend_tensor_view logscale_in_view = { 0 };
    gsx_backend_tensor_view sh0_in_view = { 0 };
    gsx_backend_tensor_view sh1_in_view = { 0 };
    gsx_backend_tensor_view sh2_in_view = { 0 };
    gsx_backend_tensor_view sh3_in_view = { 0 };
    gsx_backend_tensor_view opacity_in_view = { 0 };
    gsx_backend_tensor_view depth_keys_view = { 0 };
    gsx_backend_tensor_view depth_keys_sorted_view = { 0 };
    gsx_backend_tensor_view visible_primitive_ids_view = { 0 };
    gsx_backend_tensor_view visible_primitive_ids_sorted_view = { 0 };
    gsx_backend_tensor_view touched_tiles_view = { 0 };
    gsx_backend_tensor_view primitive_offsets_view = { 0 };
    gsx_backend_tensor_view bounds_view = { 0 };
    gsx_backend_tensor_view mean2d_view = { 0 };
    gsx_backend_tensor_view conic_opacity_view = { 0 };
    gsx_backend_tensor_view color_view = { 0 };
    gsx_backend_tensor_view visible_count_view = { 0 };
    gsx_backend_tensor_view instance_count_view = { 0 };
    gsx_backend_tensor_view visible_sort_histogram_view = { 0 };
    gsx_backend_tensor_view visible_sort_global_histogram_view = { 0 };
    gsx_backend_tensor_view visible_sort_scatter_offsets_view = { 0 };
    gsx_backend_tensor_view primitive_scan_block_sums_view = { 0 };
    gsx_backend_tensor_view primitive_scan_scanned_block_sums_view = { 0 };
    gsx_backend_tensor_view instance_keys_view = { 0 };
    gsx_backend_tensor_view instance_primitive_ids_view = { 0 };
    gsx_backend_tensor_view instance_keys_sorted_view = { 0 };
    gsx_backend_tensor_view instance_primitive_ids_sorted_view = { 0 };
    gsx_backend_tensor_view instance_sort_histogram_view = { 0 };
    gsx_backend_tensor_view instance_sort_global_histogram_view = { 0 };
    gsx_backend_tensor_view instance_sort_scatter_offsets_view = { 0 };
    gsx_backend_tensor_view tile_ranges_view = { 0 };
    gsx_backend_tensor_view tile_bucket_counts_view = { 0 };
    gsx_backend_tensor_view tile_bucket_offsets_view = { 0 };
    gsx_backend_tensor_view tile_max_n_contributions_view = { 0 };
    gsx_backend_tensor_view tile_n_contributions_view = { 0 };
    gsx_backend_tensor_view tile_scan_block_sums_view = { 0 };
    gsx_backend_tensor_view tile_scan_scanned_block_sums_view = { 0 };
    gsx_backend_tensor_view bucket_tile_index_view = { 0 };
    gsx_backend_tensor_view bucket_color_transmittance_view = { 0 };
    gsx_backend_tensor_view optional_dummy_view = { 0 };
    gsx_backend_tensor_view visible_counter_aux_view = { 0 };
    gsx_backend_tensor_view max_screen_radius_aux_view = { 0 };
    gsx_backend_tensor_view image_view = { 0 };
    gsx_backend_tensor_view alpha_view = { 0 };
    gsx_backend_tensor_view out_view = { 0 };
    gsx_metal_render_preprocess_params preprocess_params = { 0 };
    gsx_metal_render_create_instances_params create_params = { 0 };
    gsx_metal_render_blend_params blend_params = { 0 };
    gsx_metal_render_compose_params compose_params = { 0 };
    uint32_t visible_count_u32 = 0;
    uint32_t instance_count_u32 = 0;
    uint32_t bucket_count_u32 = 0;
    gsx_size_t gaussian_count = 0;
    gsx_size_t visible_count = 0;
    gsx_size_t instance_count = 0;
    gsx_size_t bucket_count = 0;
    gsx_size_t tile_count = 0;
    gsx_index_t grid_width = 0;
    gsx_index_t grid_height = 0;
    uint32_t *expected_depth_keys_sorted = NULL;
    int32_t *expected_depth_ids_sorted = NULL;
    uint32_t *expected_instance_keys_sorted = NULL;
    int32_t *expected_instance_ids_sorted = NULL;
    bool dump_debug = false;
    gsx_error error = { GSX_ERROR_SUCCESS, NULL };

    error = gsx_metal_render_prepare_forward_context(metal_context);
    if(!gsx_error_is_success(error)) {
        return error;
    }

    gaussian_count = (gsx_size_t)request->gs_mean3d->shape[0];
    dump_debug = gsx_metal_render_debug_dump_enabled(gaussian_count);
    grid_width = gsx_metal_render_get_grid_width(renderer->info.width);
    grid_height = gsx_metal_render_get_grid_height(renderer->info.height);
    tile_count = gsx_metal_render_get_tile_count(renderer->info.width, renderer->info.height);

    if(gaussian_count > 0u) {
        error = gsx_metal_render_reserve_forward_primitive_scratch(metal_context, gaussian_count);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_reserve_forward_tile_scratch(metal_context, renderer->info.width, renderer->info.height, tile_count);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_alloc_forward_primitive_scratch(metal_context, gaussian_count, &scratch);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_alloc_forward_tile_scratch(metal_context, renderer->info.width, renderer->info.height, tile_count, &scratch);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }

        error = gsx_tensor_set_zero(scratch.visible_count);
        if(!gsx_error_is_success(error)) goto cleanup;
        error = gsx_tensor_set_zero(scratch.instance_count);
        if(!gsx_error_is_success(error)) goto cleanup;
        error = gsx_tensor_set_zero(scratch.tile_ranges);
        if(!gsx_error_is_success(error)) goto cleanup;
        error = gsx_tensor_set_zero(scratch.tile_bucket_counts);
        if(!gsx_error_is_success(error)) goto cleanup;
        error = gsx_tensor_set_zero(scratch.tile_bucket_offsets);
        if(!gsx_error_is_success(error)) goto cleanup;

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
        gsx_metal_render_make_tensor_view(scratch.depth_keys, &depth_keys_view);
        gsx_metal_render_make_tensor_view(scratch.depth_keys_sorted, &depth_keys_sorted_view);
        gsx_metal_render_make_tensor_view(scratch.visible_primitive_ids, &visible_primitive_ids_view);
        gsx_metal_render_make_tensor_view(scratch.visible_primitive_ids_sorted, &visible_primitive_ids_sorted_view);
        gsx_metal_render_make_tensor_view(scratch.touched_tiles, &touched_tiles_view);
        gsx_metal_render_make_tensor_view(scratch.primitive_offsets, &primitive_offsets_view);
        gsx_metal_render_make_tensor_view(scratch.bounds, &bounds_view);
        gsx_metal_render_make_tensor_view(scratch.mean2d, &mean2d_view);
        gsx_metal_render_make_tensor_view(scratch.conic_opacity, &conic_opacity_view);
        gsx_metal_render_make_tensor_view(scratch.color, &color_view);
        gsx_metal_render_make_tensor_view(scratch.visible_count, &visible_count_view);
        gsx_metal_render_make_tensor_view(scratch.instance_count, &instance_count_view);
        gsx_metal_render_make_tensor_view(scratch.visible_sort_histogram, &visible_sort_histogram_view);
        gsx_metal_render_make_tensor_view(scratch.visible_sort_global_histogram, &visible_sort_global_histogram_view);
        gsx_metal_render_make_tensor_view(scratch.visible_sort_scatter_offsets, &visible_sort_scatter_offsets_view);
        gsx_metal_render_make_tensor_view(scratch.primitive_scan_block_sums, &primitive_scan_block_sums_view);
        gsx_metal_render_make_tensor_view(scratch.primitive_scan_scanned_block_sums, &primitive_scan_scanned_block_sums_view);
        gsx_metal_render_make_tensor_view(scratch.tile_ranges, &tile_ranges_view);
        gsx_metal_render_make_tensor_view(scratch.tile_bucket_counts, &tile_bucket_counts_view);
        gsx_metal_render_make_tensor_view(scratch.tile_bucket_offsets, &tile_bucket_offsets_view);
        gsx_metal_render_make_tensor_view(scratch.tile_max_n_contributions, &tile_max_n_contributions_view);
        gsx_metal_render_make_tensor_view(scratch.tile_n_contributions, &tile_n_contributions_view);
        gsx_metal_render_make_tensor_view(scratch.tile_scan_block_sums, &tile_scan_block_sums_view);
        gsx_metal_render_make_tensor_view(scratch.tile_scan_scanned_block_sums, &tile_scan_scanned_block_sums_view);
        gsx_metal_render_make_tensor_view(metal_context->optional_dummy_f32, &optional_dummy_view);
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
        preprocess_params.has_visible_counter = request->forward_type == GSX_RENDER_FORWARD_TYPE_TRAIN && request->gs_visible_counter != NULL ? 1u : 0u;
        preprocess_params.has_max_screen_radius = request->forward_type == GSX_RENDER_FORWARD_TYPE_TRAIN && request->gs_max_screen_radius != NULL ? 1u : 0u;
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
            request->gs_sh1 != NULL ? &sh1_in_view : &optional_dummy_view,
            request->gs_sh2 != NULL ? &sh2_in_view : &optional_dummy_view,
            request->gs_sh3 != NULL ? &sh3_in_view : &optional_dummy_view,
            &opacity_in_view,
            &depth_keys_view,
            &visible_primitive_ids_view,
            &touched_tiles_view,
            &bounds_view,
            &mean2d_view,
            &conic_opacity_view,
            &color_view,
            &visible_count_view,
            &instance_count_view,
            request->forward_type == GSX_RENDER_FORWARD_TYPE_TRAIN && request->gs_visible_counter != NULL ? &visible_counter_aux_view : &optional_dummy_view,
            request->forward_type == GSX_RENDER_FORWARD_TYPE_TRAIN && request->gs_max_screen_radius != NULL ? &max_screen_radius_aux_view : &optional_dummy_view,
            &preprocess_params);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }

        error = gsx_metal_render_read_tensor_u32(scratch.visible_count, &visible_count_u32);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        error = gsx_metal_render_read_tensor_u32(scratch.instance_count, &instance_count_u32);
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
        visible_count = (gsx_size_t)visible_count_u32;
        instance_count = (gsx_size_t)instance_count_u32;

        if(dump_debug) {
            printf(
                "METAL ordering dump: visible_count=%zu instance_count=%zu tile_count=%zu\n",
                (size_t)visible_count,
                (size_t)instance_count,
                (size_t)tile_count);
        }

        if(visible_count > 0u) {
            if(dump_debug) {
                const uint32_t *host_depth_keys = NULL;
                const int32_t *host_visible_primitive_ids = NULL;
                gsx_size_t mapped_size_bytes = 0;

                error = gsx_metal_render_tensor_map_host_bytes(scratch.depth_keys, (void **)&host_depth_keys, &mapped_size_bytes);
                if(!gsx_error_is_success(error) || mapped_size_bytes < visible_count * sizeof(uint32_t)) {
                    error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map dense depth-key tensor for debug dump");
                    goto cleanup;
                }
                error = gsx_metal_render_tensor_map_host_bytes(scratch.visible_primitive_ids, (void **)&host_visible_primitive_ids, &mapped_size_bytes);
                if(!gsx_error_is_success(error) || mapped_size_bytes < visible_count * sizeof(int32_t)) {
                    error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map dense visible-id tensor for debug dump");
                    goto cleanup;
                }

                expected_depth_keys_sorted = (uint32_t *)malloc((size_t)visible_count * sizeof(uint32_t));
                expected_depth_ids_sorted = (int32_t *)malloc((size_t)visible_count * sizeof(int32_t));
                if(expected_depth_keys_sorted == NULL || expected_depth_ids_sorted == NULL) {
                    error = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate depth-sort debug scratch");
                    goto cleanup;
                }
                error = gsx_metal_render_compute_expected_stable_sort(
                    host_depth_keys,
                    host_visible_primitive_ids,
                    visible_count,
                    expected_depth_keys_sorted,
                    expected_depth_ids_sorted);
                if(!gsx_error_is_success(error)) {
                    goto cleanup;
                }
            }

            error = gsx_metal_render_sort_pairs_u32_or_copy(
                renderer->backend,
                &depth_keys_view,
                &visible_primitive_ids_view,
                &depth_keys_sorted_view,
                &visible_primitive_ids_sorted_view,
                &visible_sort_histogram_view,
                &visible_sort_global_histogram_view,
                &visible_sort_scatter_offsets_view,
                (uint32_t)visible_count);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }

            if(dump_debug) {
                const uint32_t *host_depth_keys_sorted = NULL;
                const int32_t *host_visible_primitive_ids_sorted = NULL;
                gsx_size_t mismatch = (gsx_size_t)-1;
                gsx_size_t mapped_size_bytes = 0;
                bool matches = false;

                error = gsx_backend_major_stream_sync(renderer->backend);
                if(!gsx_error_is_success(error)) {
                    goto cleanup;
                }
                error = gsx_metal_render_tensor_map_host_bytes(scratch.depth_keys_sorted, (void **)&host_depth_keys_sorted, &mapped_size_bytes);
                if(!gsx_error_is_success(error) || mapped_size_bytes < visible_count * sizeof(uint32_t)) {
                    error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map sorted depth-key tensor for debug dump");
                    goto cleanup;
                }
                error = gsx_metal_render_tensor_map_host_bytes(scratch.visible_primitive_ids_sorted, (void **)&host_visible_primitive_ids_sorted, &mapped_size_bytes);
                if(!gsx_error_is_success(error) || mapped_size_bytes < visible_count * sizeof(int32_t)) {
                    error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map sorted visible-id tensor for debug dump");
                    goto cleanup;
                }

                matches = gsx_metal_render_u32_i32_pairs_match(
                    host_depth_keys_sorted,
                    host_visible_primitive_ids_sorted,
                    expected_depth_keys_sorted,
                    expected_depth_ids_sorted,
                    visible_count,
                    &mismatch);
                if(!matches) {
                    fprintf(
                        stderr,
                        "METAL invariant failed: depth sort mismatch at index %zu (metal key=%u id=%d, expected key=%u id=%d)\n",
                        (size_t)mismatch,
                        (unsigned int)host_depth_keys_sorted[mismatch],
                        host_visible_primitive_ids_sorted[mismatch],
                        (unsigned int)expected_depth_keys_sorted[mismatch],
                        expected_depth_ids_sorted[mismatch]);
                    error = gsx_make_error(GSX_ERROR_INVALID_STATE, "metal debug depth sort mismatch");
                    goto cleanup;
                }

                printf(
                    "METAL ordering dump: depth_sort_pairs hash=%llu count=%zu\n",
                    (unsigned long long)gsx_metal_render_hash_u32_i32_pairs(host_depth_keys_sorted, host_visible_primitive_ids_sorted, visible_count),
                    (size_t)visible_count);
            }

            error = gsx_metal_backend_dispatch_render_apply_depth_ordering(
                renderer->backend,
                &visible_primitive_ids_sorted_view,
                &touched_tiles_view,
                &primitive_offsets_view,
                (uint32_t)visible_count);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_backend_dispatch_scan_exclusive_u32(
                renderer->backend,
                &primitive_offsets_view,
                &primitive_scan_block_sums_view,
                &primitive_scan_scanned_block_sums_view,
                (uint32_t)visible_count);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
        }

        if(instance_count > 0u) {
            error = gsx_metal_render_reserve_forward_instance_scratch(metal_context, instance_count);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_render_alloc_forward_instance_scratch(metal_context, instance_count, &scratch);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }

            gsx_metal_render_make_tensor_view(scratch.instance_keys, &instance_keys_view);
            gsx_metal_render_make_tensor_view(scratch.instance_primitive_ids, &instance_primitive_ids_view);
            gsx_metal_render_make_tensor_view(scratch.instance_keys_sorted, &instance_keys_sorted_view);
            gsx_metal_render_make_tensor_view(scratch.instance_primitive_ids_sorted, &instance_primitive_ids_sorted_view);
            gsx_metal_render_make_tensor_view(scratch.instance_sort_histogram, &instance_sort_histogram_view);
            gsx_metal_render_make_tensor_view(scratch.instance_sort_global_histogram, &instance_sort_global_histogram_view);
            gsx_metal_render_make_tensor_view(scratch.instance_sort_scatter_offsets, &instance_sort_scatter_offsets_view);

            create_params.visible_count = (uint32_t)visible_count;
            create_params.grid_width = (uint32_t)grid_width;
            create_params.grid_height = (uint32_t)grid_height;
            error = gsx_metal_backend_dispatch_render_create_instances(
                renderer->backend,
                &visible_primitive_ids_sorted_view,
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

            if(dump_debug) {
                const uint32_t *host_instance_keys = NULL;
                const int32_t *host_instance_primitive_ids = NULL;
                gsx_size_t mapped_size_bytes = 0;

                error = gsx_backend_major_stream_sync(renderer->backend);
                if(!gsx_error_is_success(error)) {
                    goto cleanup;
                }
                error = gsx_metal_render_tensor_map_host_bytes(scratch.instance_keys, (void **)&host_instance_keys, &mapped_size_bytes);
                if(!gsx_error_is_success(error) || mapped_size_bytes < instance_count * sizeof(uint32_t)) {
                    error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map unsorted instance keys for debug dump");
                    goto cleanup;
                }
                error = gsx_metal_render_tensor_map_host_bytes(scratch.instance_primitive_ids, (void **)&host_instance_primitive_ids, &mapped_size_bytes);
                if(!gsx_error_is_success(error) || mapped_size_bytes < instance_count * sizeof(int32_t)) {
                    error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map unsorted instance ids for debug dump");
                    goto cleanup;
                }

                expected_instance_keys_sorted = (uint32_t *)malloc((size_t)instance_count * sizeof(uint32_t));
                expected_instance_ids_sorted = (int32_t *)malloc((size_t)instance_count * sizeof(int32_t));
                if(expected_instance_keys_sorted == NULL || expected_instance_ids_sorted == NULL) {
                    error = gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate instance-sort debug scratch");
                    goto cleanup;
                }
                error = gsx_metal_render_compute_expected_stable_sort(
                    host_instance_keys,
                    host_instance_primitive_ids,
                    instance_count,
                    expected_instance_keys_sorted,
                    expected_instance_ids_sorted);
                if(!gsx_error_is_success(error)) {
                    goto cleanup;
                }
            }

            error = gsx_metal_render_sort_pairs_u32_or_copy(
                renderer->backend,
                &instance_keys_view,
                &instance_primitive_ids_view,
                &instance_keys_sorted_view,
                &instance_primitive_ids_sorted_view,
                &instance_sort_histogram_view,
                &instance_sort_global_histogram_view,
                &instance_sort_scatter_offsets_view,
                (uint32_t)instance_count);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }

            if(dump_debug) {
                const uint32_t *host_instance_keys_sorted = NULL;
                const int32_t *host_instance_ids_sorted = NULL;
                gsx_size_t mismatch = (gsx_size_t)-1;
                gsx_size_t mapped_size_bytes = 0;
                bool matches = false;

                error = gsx_backend_major_stream_sync(renderer->backend);
                if(!gsx_error_is_success(error)) {
                    goto cleanup;
                }
                error = gsx_metal_render_tensor_map_host_bytes(scratch.instance_keys_sorted, (void **)&host_instance_keys_sorted, &mapped_size_bytes);
                if(!gsx_error_is_success(error) || mapped_size_bytes < instance_count * sizeof(uint32_t)) {
                    error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map sorted instance keys for debug dump");
                    goto cleanup;
                }
                error = gsx_metal_render_tensor_map_host_bytes(scratch.instance_primitive_ids_sorted, (void **)&host_instance_ids_sorted, &mapped_size_bytes);
                if(!gsx_error_is_success(error) || mapped_size_bytes < instance_count * sizeof(int32_t)) {
                    error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map sorted instance ids for debug dump");
                    goto cleanup;
                }

                matches = gsx_metal_render_u32_i32_pairs_match(
                    host_instance_keys_sorted,
                    host_instance_ids_sorted,
                    expected_instance_keys_sorted,
                    expected_instance_ids_sorted,
                    instance_count,
                    &mismatch);
                if(!matches) {
                    fprintf(
                        stderr,
                        "METAL invariant failed: instance sort mismatch at index %zu (metal key=%u id=%d, expected key=%u id=%d)\n",
                        (size_t)mismatch,
                        (unsigned int)host_instance_keys_sorted[mismatch],
                        host_instance_ids_sorted[mismatch],
                        (unsigned int)expected_instance_keys_sorted[mismatch],
                        expected_instance_ids_sorted[mismatch]);
                    error = gsx_make_error(GSX_ERROR_INVALID_STATE, "metal debug instance sort mismatch");
                    goto cleanup;
                }

                printf(
                    "METAL ordering dump: instance_sort_pairs hash=%llu count=%zu\n",
                    (unsigned long long)gsx_metal_render_hash_u32_i32_pairs(host_instance_keys_sorted, host_instance_ids_sorted, instance_count),
                    (size_t)instance_count);
            }

            error = gsx_tensor_set_zero(scratch.tile_ranges);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_backend_dispatch_render_extract_instance_ranges(
                renderer->backend,
                &instance_keys_sorted_view,
                &tile_ranges_view,
                (uint32_t)instance_count,
                (uint32_t)tile_count);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
        }

        if(tile_count > 0u) {
            error = gsx_tensor_set_zero(scratch.tile_bucket_counts);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_tensor_set_zero(scratch.tile_bucket_offsets);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }

            error = gsx_metal_backend_dispatch_render_extract_bucket_counts(
                renderer->backend,
                &tile_ranges_view,
                &tile_bucket_counts_view,
                (uint32_t)tile_count);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_backend_buffer_copy_tensor(
                tile_bucket_offsets_view.buffer,
                &tile_bucket_counts_view,
                &tile_bucket_offsets_view);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_backend_dispatch_scan_exclusive_u32(
                renderer->backend,
                &tile_bucket_offsets_view,
                &tile_scan_block_sums_view,
                &tile_scan_scanned_block_sums_view,
                (uint32_t)tile_count);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            if(dump_debug && tile_count > 0u) {
                const int32_t *host_tile_bucket_offsets_pre_finalize = NULL;
                gsx_size_t mapped_size_bytes_pre_finalize = 0;

                error = gsx_metal_render_tensor_map_host_bytes(
                    scratch.tile_bucket_offsets,
                    (void **)&host_tile_bucket_offsets_pre_finalize,
                    &mapped_size_bytes_pre_finalize);
                if(!gsx_error_is_success(error) || mapped_size_bytes_pre_finalize < tile_count * sizeof(int32_t)) {
                    error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map tile bucket offsets after exclusive scan");
                    goto cleanup;
                }
                fprintf(
                    stderr,
                    "METAL pre-finalize scan dump: offsets[0..5]=%d,%d,%d,%d,%d,%d\n",
                    tile_count > 0u ? host_tile_bucket_offsets_pre_finalize[0] : 0,
                    tile_count > 1u ? host_tile_bucket_offsets_pre_finalize[1] : 0,
                    tile_count > 2u ? host_tile_bucket_offsets_pre_finalize[2] : 0,
                    tile_count > 3u ? host_tile_bucket_offsets_pre_finalize[3] : 0,
                    tile_count > 4u ? host_tile_bucket_offsets_pre_finalize[4] : 0,
                    tile_count > 5u ? host_tile_bucket_offsets_pre_finalize[5] : 0);
            }
            error = gsx_metal_backend_dispatch_render_finalize_bucket_offsets(
                renderer->backend,
                &tile_bucket_counts_view,
                &tile_bucket_offsets_view,
                (uint32_t)tile_count);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }

            error = gsx_metal_render_read_tensor_u32_at(scratch.tile_bucket_offsets, tile_count - 1u, &bucket_count_u32);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            bucket_count = (gsx_size_t)bucket_count_u32;

            if(dump_debug && instance_count > 0u) {
                const int32_t *host_instance_keys_sorted = NULL;
                const int32_t *host_instance_ids_sorted = NULL;
                const int32_t *host_tile_ranges = NULL;
                const int32_t *host_tile_bucket_counts = NULL;
                const int32_t *host_tile_bucket_offsets = NULL;
                const int32_t *host_tile_scan_block_sums = NULL;
                const int32_t *host_tile_scan_scanned_block_sums = NULL;
                gsx_size_t mapped_size_bytes = 0;
                gsx_size_t expected_bucket_count = 0;
                gsx_size_t mismatch_tile = (gsx_size_t)-1;
                gsx_size_t tile_scan_block_count = (tile_count + 255u) / 256u;
                int32_t expected_count = 0;
                int32_t actual_count = 0;
                int32_t expected_offset = 0;
                int32_t actual_offset = 0;

                error = gsx_metal_render_tensor_map_host_bytes(scratch.instance_keys_sorted, (void **)&host_instance_keys_sorted, &mapped_size_bytes);
                if(!gsx_error_is_success(error) || mapped_size_bytes < instance_count * sizeof(int32_t)) {
                    error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map sorted instance keys for tile debug dump");
                    goto cleanup;
                }
                error = gsx_metal_render_tensor_map_host_bytes(scratch.instance_primitive_ids_sorted, (void **)&host_instance_ids_sorted, &mapped_size_bytes);
                if(!gsx_error_is_success(error) || mapped_size_bytes < instance_count * sizeof(int32_t)) {
                    error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map sorted instance ids for tile debug dump");
                    goto cleanup;
                }
                error = gsx_metal_render_tensor_map_host_bytes(scratch.tile_ranges, (void **)&host_tile_ranges, &mapped_size_bytes);
                if(!gsx_error_is_success(error) || mapped_size_bytes < tile_count * 2u * sizeof(int32_t)) {
                    error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map tile ranges for debug dump");
                    goto cleanup;
                }
                error = gsx_metal_render_tensor_map_host_bytes(scratch.tile_bucket_counts, (void **)&host_tile_bucket_counts, &mapped_size_bytes);
                if(!gsx_error_is_success(error) || mapped_size_bytes < tile_count * sizeof(int32_t)) {
                    error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map tile bucket counts for debug dump");
                    goto cleanup;
                }
                error = gsx_metal_render_tensor_map_host_bytes(scratch.tile_bucket_offsets, (void **)&host_tile_bucket_offsets, &mapped_size_bytes);
                if(!gsx_error_is_success(error) || mapped_size_bytes < tile_count * sizeof(int32_t)) {
                    error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map tile bucket offsets for debug dump");
                    goto cleanup;
                }
                error = gsx_metal_render_tensor_map_host_bytes(scratch.tile_scan_block_sums, (void **)&host_tile_scan_block_sums, &mapped_size_bytes);
                if(!gsx_error_is_success(error) || mapped_size_bytes < tile_scan_block_count * sizeof(int32_t)) {
                    error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map tile scan block sums for debug dump");
                    goto cleanup;
                }
                error = gsx_metal_render_tensor_map_host_bytes(scratch.tile_scan_scanned_block_sums, (void **)&host_tile_scan_scanned_block_sums, &mapped_size_bytes);
                if(!gsx_error_is_success(error) || mapped_size_bytes < tile_scan_block_count * sizeof(int32_t)) {
                    error = gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to map tile scan scanned block sums for debug dump");
                    goto cleanup;
                }
                error = gsx_metal_render_validate_sorted_tile_ranges(host_instance_keys_sorted, instance_count, host_tile_ranges, tile_count);
                if(!gsx_error_is_success(error)) {
                    goto cleanup;
                }
                error = gsx_metal_render_validate_bucket_counts(
                    host_tile_ranges,
                    tile_count,
                    host_tile_bucket_counts,
                    host_tile_bucket_offsets,
                    &expected_bucket_count,
                    &mismatch_tile,
                    &expected_count,
                    &actual_count,
                    &expected_offset,
                    &actual_offset);
                if(!gsx_error_is_success(error)) {
                    fprintf(
                        stderr,
                        "METAL invariant failed: bucket validation mismatch (tile=%zu expected_count=%d actual_count=%d expected_offset=%d actual_offset=%d expected_bucket_count=%zu actual_last_offset=%d first_counts=%d,%d,%d,%d first_ranges=[%d,%d],[%d,%d] counts_view=(buf=%p off=%zu size=%zu) offsets_view=(buf=%p off=%zu size=%zu)\n",
                        (size_t)mismatch_tile,
                        expected_count,
                        actual_count,
                        expected_offset,
                        actual_offset,
                        (size_t)expected_bucket_count,
                        tile_count > 0u ? host_tile_bucket_offsets[tile_count - 1u] : 0,
                        tile_count > 0u ? host_tile_bucket_counts[0] : 0,
                        tile_count > 1u ? host_tile_bucket_counts[1] : 0,
                        tile_count > 2u ? host_tile_bucket_counts[2] : 0,
                        tile_count > 3u ? host_tile_bucket_counts[3] : 0,
                        tile_count > 0u ? host_tile_ranges[0] : 0,
                        tile_count > 0u ? host_tile_ranges[1] : 0,
                        tile_count > 1u ? host_tile_ranges[2] : 0,
                        tile_count > 1u ? host_tile_ranges[3] : 0,
                        (void *)tile_bucket_counts_view.buffer,
                        (size_t)tile_bucket_counts_view.offset_bytes,
                        (size_t)tile_bucket_counts_view.size_bytes,
                        (void *)tile_bucket_offsets_view.buffer,
                        (size_t)tile_bucket_offsets_view.offset_bytes,
                        (size_t)tile_bucket_offsets_view.size_bytes);
                    fprintf(
                        stderr,
                        "METAL scan dump: offsets[0..5]=%d,%d,%d,%d,%d,%d offsets[254..258]=%d,%d,%d,%d,%d block_sums=%d,%d,%d,%d,%d scanned_block_sums=%d,%d,%d,%d,%d\n",
                        tile_count > 0u ? host_tile_bucket_offsets[0] : 0,
                        tile_count > 1u ? host_tile_bucket_offsets[1] : 0,
                        tile_count > 2u ? host_tile_bucket_offsets[2] : 0,
                        tile_count > 3u ? host_tile_bucket_offsets[3] : 0,
                        tile_count > 4u ? host_tile_bucket_offsets[4] : 0,
                        tile_count > 5u ? host_tile_bucket_offsets[5] : 0,
                        tile_count > 254u ? host_tile_bucket_offsets[254] : 0,
                        tile_count > 255u ? host_tile_bucket_offsets[255] : 0,
                        tile_count > 256u ? host_tile_bucket_offsets[256] : 0,
                        tile_count > 257u ? host_tile_bucket_offsets[257] : 0,
                        tile_count > 258u ? host_tile_bucket_offsets[258] : 0,
                        tile_scan_block_count > 0u ? host_tile_scan_block_sums[0] : 0,
                        tile_scan_block_count > 1u ? host_tile_scan_block_sums[1] : 0,
                        tile_scan_block_count > 2u ? host_tile_scan_block_sums[2] : 0,
                        tile_scan_block_count > 3u ? host_tile_scan_block_sums[3] : 0,
                        tile_scan_block_count > 4u ? host_tile_scan_block_sums[4] : 0,
                        tile_scan_block_count > 0u ? host_tile_scan_scanned_block_sums[0] : 0,
                        tile_scan_block_count > 1u ? host_tile_scan_scanned_block_sums[1] : 0,
                        tile_scan_block_count > 2u ? host_tile_scan_scanned_block_sums[2] : 0,
                        tile_scan_block_count > 3u ? host_tile_scan_scanned_block_sums[3] : 0,
                        tile_scan_block_count > 4u ? host_tile_scan_scanned_block_sums[4] : 0);
                    goto cleanup;
                }

                printf(
                    "METAL ordering dump: tile_ranges hash=%llu bucket_count=%zu expected_bucket_count=%zu last_bucket_offset=%d\n",
                    (unsigned long long)gsx_metal_render_hash_u32_i32_pairs((const uint32_t *)host_instance_keys_sorted, host_instance_ids_sorted, instance_count),
                    (size_t)bucket_count,
                    (size_t)expected_bucket_count,
                    tile_count > 0u ? host_tile_bucket_offsets[tile_count - 1u] : 0);
            }
        }

        if(bucket_count > 0u) {
            error = gsx_metal_render_reserve_forward_bucket_scratch(metal_context, bucket_count);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_metal_render_alloc_forward_bucket_scratch(metal_context, bucket_count, &scratch);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_tensor_set_zero(scratch.bucket_color_transmittance);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }

            gsx_metal_render_make_tensor_view(scratch.bucket_tile_index, &bucket_tile_index_view);
            gsx_metal_render_make_tensor_view(scratch.bucket_color_transmittance, &bucket_color_transmittance_view);
        }

        if(tile_count > 0u) {
            error = gsx_tensor_set_zero(scratch.tile_max_n_contributions);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
            error = gsx_tensor_set_zero(scratch.tile_n_contributions);
            if(!gsx_error_is_success(error)) {
                goto cleanup;
            }
        }

        if(instance_count > 0u && tile_count > 0u && bucket_count > 0u) {
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

            gsx_metal_render_make_tensor_view(metal_context->helper_image_chw, &image_view);
            gsx_metal_render_make_tensor_view(metal_context->helper_alpha_hw, &alpha_view);

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
                &instance_primitive_ids_sorted_view,
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
    if(request->forward_type == GSX_RENDER_FORWARD_TYPE_TRAIN) {
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
        if(!gsx_error_is_success(error)) {
            goto cleanup;
        }
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
