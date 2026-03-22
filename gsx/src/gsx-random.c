#include "gsx/gsx-random.h"
#include "gsx-impl.h"
#include <stdlib.h>

#define GSX_PCG32_DEFAULT_STATE  0x853c49e6748fea9bULL
#define GSX_PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define GSX_PCG32_MULT           0x5851f42d4c957f2dULL

struct gsx_pcg32 {
    uint64_t state;
    uint64_t inc;
};

static inline uint32_t gsx_pcg32_next_uint_impl(struct gsx_pcg32* pcg) {
    uint64_t oldstate = pcg->state;
    pcg->state = oldstate * GSX_PCG32_MULT + pcg->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
}

GSX_API gsx_error gsx_pcg32_init(gsx_pcg32_t* out_pcg, gsx_pcg32_state_t init_seed) {
    if(out_pcg == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_pcg must be non-null");
    }
    *out_pcg = NULL;

    struct gsx_pcg32* pcg = (struct gsx_pcg32*)malloc(sizeof(struct gsx_pcg32));
    if(pcg == NULL) {
        return gsx_make_error(GSX_ERROR_OUT_OF_MEMORY, "failed to allocate pcg32 state");
    }

    pcg->state = 0u;
    pcg->inc = (init_seed << 1u) | 1u;
    gsx_pcg32_next_uint_impl(pcg);
    pcg->state += init_seed;
    gsx_pcg32_next_uint_impl(pcg);

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
    *out_value = gsx_pcg32_next_uint_impl(pcg);
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

    uint32_t threshold = (~bound + 1u) % bound;
    while(1) {
        uint32_t r = gsx_pcg32_next_uint_impl(pcg);
        if(r >= threshold) {
            *out_value = r % bound;
            return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
        }
    }
}

GSX_API gsx_error gsx_pcg32_next_float(gsx_pcg32_t pcg, float* out_value) {
    if(pcg == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "pcg must be non-null");
    }
    if(out_value == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_value must be non-null");
    }

    union {
        uint32_t u;
        float f;
    } x;
    x.u = (gsx_pcg32_next_uint_impl(pcg) >> 9u) | 0x3f800000u;
    *out_value = x.f - 1.0f;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_pcg32_next_double(gsx_pcg32_t pcg, double* out_value) {
    if(pcg == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "pcg must be non-null");
    }
    if(out_value == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_value must be non-null");
    }

    union {
        uint64_t u;
        double d;
    } x;
    x.u = ((uint64_t)gsx_pcg32_next_uint_impl(pcg) << 20u) | 0x3ff0000000000000ULL;
    *out_value = x.d - 1.0;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

GSX_API gsx_error gsx_pcg32_advance(gsx_pcg32_t pcg, gsx_pcg32_statediff_t delta) {
    if(pcg == NULL) {
        return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "pcg must be non-null");
    }

    uint64_t cur_mult = GSX_PCG32_MULT;
    uint64_t cur_plus = pcg->inc;
    uint64_t acc_mult = 1u;
    uint64_t acc_plus = 0u;
    uint64_t d = (uint64_t)delta;

    while(d > 0u) {
        if((d & 1u) != 0u) {
            acc_mult *= cur_mult;
            acc_plus = acc_plus * cur_mult + cur_plus;
        }
        cur_plus = (cur_mult + 1u) * cur_plus;
        cur_mult *= cur_mult;
        d /= 2u;
    }
    pcg->state = acc_mult * pcg->state + acc_plus;
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

    uint64_t cur_mult = GSX_PCG32_MULT;
    uint64_t cur_plus = a->inc;
    uint64_t cur_state = b->state;
    uint64_t the_bit = 1u;
    uint64_t distance = 0u;

    while(a->state != cur_state) {
        if((a->state & the_bit) != (cur_state & the_bit)) {
            cur_state = cur_state * cur_mult + cur_plus;
            distance |= the_bit;
        }
        the_bit <<= 1u;
        cur_plus = (cur_mult + 1ULL) * cur_plus;
        cur_mult *= cur_mult;
    }
    *out_distance = (gsx_pcg32_statediff_t)distance;
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
    *out_equal = a->state == b->state && a->inc == b->inc;
    return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}
