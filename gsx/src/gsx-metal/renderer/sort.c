#include "../internal.h"

#include <stdlib.h>
#include <stdint.h>

static int gsx_metal_sort_pair_u32_cmp(const void *lhs_void, const void *rhs_void)
{
    const gsx_metal_sort_pair_u32 *lhs = (const gsx_metal_sort_pair_u32 *)lhs_void;
    const gsx_metal_sort_pair_u32 *rhs = (const gsx_metal_sort_pair_u32 *)rhs_void;

    if(lhs->key < rhs->key) {
        return -1;
    }
    if(lhs->key > rhs->key) {
        return 1;
    }
    if(lhs->stable_index < rhs->stable_index) {
        return -1;
    }
    if(lhs->stable_index > rhs->stable_index) {
        return 1;
    }
    return 0;
}

/*
 * Sorting is intentionally hosted in renderer/sort.c so we can later swap this
 * implementation for a GPU radix sort without changing forward stage wiring.
 */
void gsx_metal_render_sort_pairs_u32(gsx_metal_sort_pair_u32 *pairs, uint32_t count)
{
    if(pairs == NULL || count <= 1u) {
        return;
    }
    qsort(pairs, (size_t)count, sizeof(*pairs), gsx_metal_sort_pair_u32_cmp);
}
