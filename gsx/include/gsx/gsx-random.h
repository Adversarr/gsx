#ifndef GSX_RANDOM_H
#define GSX_RANDOM_H

#include "gsx-core.h"

GSX_EXTERN_C_BEGIN

typedef uint64_t          gsx_pcg32_state_t;
typedef int64_t           gsx_pcg32_statediff_t;
typedef struct gsx_pcg32* gsx_pcg32_t;

GSX_API gsx_error gsx_pcg32_init(gsx_pcg32_t* out_pcg, gsx_pcg32_state_t init_seed);
GSX_API gsx_error gsx_pcg32_free(gsx_pcg32_t pcg);

GSX_API gsx_error gsx_pcg32_next_uint(gsx_pcg32_t pcg, uint32_t* out_value);
GSX_API gsx_error gsx_pcg32_next_uint_bound(gsx_pcg32_t pcg, uint32_t* out_value, uint32_t bound);
GSX_API gsx_error gsx_pcg32_next_float(gsx_pcg32_t pcg, float* out_value);
GSX_API gsx_error gsx_pcg32_next_double(gsx_pcg32_t pcg, double* out_value);
GSX_API gsx_error gsx_pcg32_next_normal(gsx_pcg32_t pcg, float* out_value);
GSX_API gsx_error gsx_pcg32_advance(gsx_pcg32_t pcg, gsx_pcg32_statediff_t delta);
GSX_API gsx_error gsx_pcg32_distance(const gsx_pcg32_t a, const gsx_pcg32_t b, gsx_pcg32_statediff_t* out_distance);
GSX_API gsx_error gsx_pcg32_equal(const gsx_pcg32_t a, const gsx_pcg32_t b, bool* out_equal);

/* Fill tensors with random N(0, sigma^2), tensor should be floating point types */
GSX_API gsx_error gsx_pcg32_fill_randn(gsx_pcg32_t pcg, gsx_tensor_t tensor, gsx_float_t sigma);
/* Fill tensors with random U(0, 1), tensor should be floating point types */
GSX_API gsx_error gsx_pcg32_fill_rand(gsx_pcg32_t pcg, gsx_tensor_t tensor);
/* Fill tensors with random integers in the range [0, bound), tensor should be an integer type */
GSX_API gsx_error gsx_pcg32_fill_randint(gsx_pcg32_t pcg, gsx_tensor_t tensor, uint32_t bound);
/* Multinomial sampling with replacement only, used by MCMC; out indices must be int32 and cdf must be float32 */
GSX_API gsx_error gsx_pcg32_multinomial(gsx_pcg32_t pcg, gsx_tensor_t out_indices, gsx_tensor_t cdf);

GSX_EXTERN_C_END

#endif // GSX_RANDOM_H
