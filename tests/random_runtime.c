#include "gsx/gsx.h"
#include "gsx/gsx-random.h"

#include <stdalign.h>

#define GSX_STATIC_ASSERT(condition, message) _Static_assert(condition, message)

GSX_STATIC_ASSERT(sizeof(gsx_pcg32_state_t) == sizeof(uint64_t), "gsx_pcg32_state_t must be 64-bit.");
GSX_STATIC_ASSERT(sizeof(gsx_pcg32_statediff_t) == sizeof(int64_t), "gsx_pcg32_statediff_t must be 64-bit signed.");
GSX_STATIC_ASSERT(sizeof(gsx_pcg32_t) == sizeof(void *), "gsx_pcg32_t handle must be pointer-sized.");

static int require_code(gsx_error error, gsx_error_code expected_code)
{
    return error.code == expected_code ? 0 : 1;
}

int main(void)
{
    gsx_pcg32_t pcg = NULL;
    gsx_error err;

    err = gsx_pcg32_init(NULL, 0);
    if(require_code(err, GSX_ERROR_INVALID_ARGUMENT) != 0) {
        return 1;
    }

    err = gsx_pcg32_init(&pcg, 42);
    if(require_code(err, GSX_ERROR_SUCCESS) != 0) {
        return 1;
    }

    err = gsx_pcg32_free(pcg);
    if(require_code(err, GSX_ERROR_SUCCESS) != 0) {
        return 1;
    }

    err = gsx_pcg32_init(&pcg, 12345);
    if(require_code(err, GSX_ERROR_SUCCESS) != 0) {
        return 1;
    }

    uint32_t uval = 0;
    err = gsx_pcg32_next_uint(NULL, &uval);
    if(require_code(err, GSX_ERROR_INVALID_ARGUMENT) != 0) {
        gsx_pcg32_free(pcg);
        return 1;
    }

    err = gsx_pcg32_next_uint(pcg, NULL);
    if(require_code(err, GSX_ERROR_INVALID_ARGUMENT) != 0) {
        gsx_pcg32_free(pcg);
        return 1;
    }

    err = gsx_pcg32_next_uint(pcg, &uval);
    if(require_code(err, GSX_ERROR_SUCCESS) != 0) {
        gsx_pcg32_free(pcg);
        return 1;
    }

    err = gsx_pcg32_next_uint_bound(pcg, NULL, 100);
    if(require_code(err, GSX_ERROR_INVALID_ARGUMENT) != 0) {
        gsx_pcg32_free(pcg);
        return 1;
    }

    err = gsx_pcg32_next_uint_bound(pcg, &uval, 0);
    if(require_code(err, GSX_ERROR_INVALID_ARGUMENT) != 0) {
        gsx_pcg32_free(pcg);
        return 1;
    }

    err = gsx_pcg32_next_uint_bound(pcg, &uval, 100);
    if(require_code(err, GSX_ERROR_SUCCESS) != 0) {
        gsx_pcg32_free(pcg);
        return 1;
    }
    if(uval >= 100) {
        gsx_pcg32_free(pcg);
        return 1;
    }

    float fval = 0.0f;
    err = gsx_pcg32_next_float(NULL, &fval);
    if(require_code(err, GSX_ERROR_INVALID_ARGUMENT) != 0) {
        gsx_pcg32_free(pcg);
        return 1;
    }

    err = gsx_pcg32_next_float(pcg, NULL);
    if(require_code(err, GSX_ERROR_INVALID_ARGUMENT) != 0) {
        gsx_pcg32_free(pcg);
        return 1;
    }

    err = gsx_pcg32_next_float(pcg, &fval);
    if(require_code(err, GSX_ERROR_SUCCESS) != 0) {
        gsx_pcg32_free(pcg);
        return 1;
    }
    if(fval < 0.0f || fval >= 1.0f) {
        gsx_pcg32_free(pcg);
        return 1;
    }

    double dval = 0.0;
    err = gsx_pcg32_next_double(NULL, &dval);
    if(require_code(err, GSX_ERROR_INVALID_ARGUMENT) != 0) {
        gsx_pcg32_free(pcg);
        return 1;
    }

    err = gsx_pcg32_next_double(pcg, NULL);
    if(require_code(err, GSX_ERROR_INVALID_ARGUMENT) != 0) {
        gsx_pcg32_free(pcg);
        return 1;
    }

    err = gsx_pcg32_next_double(pcg, &dval);
    if(require_code(err, GSX_ERROR_SUCCESS) != 0) {
        gsx_pcg32_free(pcg);
        return 1;
    }
    if(dval < 0.0 || dval >= 1.0) {
        gsx_pcg32_free(pcg);
        return 1;
    }

    err = gsx_pcg32_advance(NULL, 100);
    if(require_code(err, GSX_ERROR_INVALID_ARGUMENT) != 0) {
        gsx_pcg32_free(pcg);
        return 1;
    }

    err = gsx_pcg32_advance(pcg, 100);
    if(require_code(err, GSX_ERROR_SUCCESS) != 0) {
        gsx_pcg32_free(pcg);
        return 1;
    }

    gsx_pcg32_t pcg2 = NULL;
    err = gsx_pcg32_init(&pcg2, 42);
    if(require_code(err, GSX_ERROR_SUCCESS) != 0) {
        gsx_pcg32_free(pcg);
        return 1;
    }

    bool equal = false;
    err = gsx_pcg32_equal(NULL, pcg2, &equal);
    if(require_code(err, GSX_ERROR_INVALID_ARGUMENT) != 0) {
        gsx_pcg32_free(pcg);
        gsx_pcg32_free(pcg2);
        return 1;
    }

    err = gsx_pcg32_equal(pcg, NULL, &equal);
    if(require_code(err, GSX_ERROR_INVALID_ARGUMENT) != 0) {
        gsx_pcg32_free(pcg);
        gsx_pcg32_free(pcg2);
        return 1;
    }

    err = gsx_pcg32_equal(pcg, pcg2, NULL);
    if(require_code(err, GSX_ERROR_INVALID_ARGUMENT) != 0) {
        gsx_pcg32_free(pcg);
        gsx_pcg32_free(pcg2);
        return 1;
    }

    err = gsx_pcg32_equal(pcg, pcg2, &equal);
    if(require_code(err, GSX_ERROR_SUCCESS) != 0) {
        gsx_pcg32_free(pcg);
        gsx_pcg32_free(pcg2);
        return 1;
    }
    if(equal) {
        gsx_pcg32_free(pcg);
        gsx_pcg32_free(pcg2);
        return 1;
    }

    gsx_pcg32_statediff_t dist = 0;
    err = gsx_pcg32_distance(NULL, pcg2, &dist);
    if(require_code(err, GSX_ERROR_INVALID_ARGUMENT) != 0) {
        gsx_pcg32_free(pcg);
        gsx_pcg32_free(pcg2);
        return 1;
    }

    err = gsx_pcg32_distance(pcg, NULL, &dist);
    if(require_code(err, GSX_ERROR_INVALID_ARGUMENT) != 0) {
        gsx_pcg32_free(pcg);
        gsx_pcg32_free(pcg2);
        return 1;
    }

    err = gsx_pcg32_distance(pcg, pcg2, NULL);
    if(require_code(err, GSX_ERROR_INVALID_ARGUMENT) != 0) {
        gsx_pcg32_free(pcg);
        gsx_pcg32_free(pcg2);
        return 1;
    }

    err = gsx_pcg32_distance(pcg, pcg2, &dist);
    if(require_code(err, GSX_ERROR_SUCCESS) != 0) {
        gsx_pcg32_free(pcg);
        gsx_pcg32_free(pcg2);
        return 1;
    }

    err = gsx_pcg32_free(pcg);
    if(require_code(err, GSX_ERROR_SUCCESS) != 0) {
        gsx_pcg32_free(pcg2);
        return 1;
    }

    err = gsx_pcg32_free(pcg2);
    if(require_code(err, GSX_ERROR_SUCCESS) != 0) {
        return 1;
    }

    return 0;
}
