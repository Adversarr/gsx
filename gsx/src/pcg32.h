/*
 * Tiny self-contained version of the PCG Random Number Generation for C++
 * put together from pieces of the much larger C/C++ codebase.
 * Wenzel Jakob, February 2015
 *
 * The PCG random number generator was developed by Melissa O'Neill
 * <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     http://www.pcg-random.org
 *
 * Note: This code was modified to work with CUDA by the tiny-cuda-nn authors.
 */

#ifndef GSX_PCG32_H
#define GSX_PCG32_H

#include <stdbool.h>
#include <stdint.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GSX_PCG32_HOST_DEVICE __host__ __device__
#else
#define GSX_PCG32_HOST_DEVICE
#endif

#define PCG32_DEFAULT_STATE  0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT           0x5851f42d4c957f2dULL

typedef struct pcg32 {
	uint64_t state;
	uint64_t inc;
} pcg32;

static GSX_PCG32_HOST_DEVICE inline void pcg32_init_default(pcg32 *rng) {
	rng->state = PCG32_DEFAULT_STATE;
	rng->inc = PCG32_DEFAULT_STREAM;
}

static GSX_PCG32_HOST_DEVICE inline uint32_t pcg32_next_uint(pcg32 *rng) {
	uint64_t oldstate = rng->state;
	rng->state = oldstate * PCG32_MULT + rng->inc;
	uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
	uint32_t rot = (uint32_t)(oldstate >> 59u);
	return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
}

static GSX_PCG32_HOST_DEVICE inline void pcg32_seed(pcg32 *rng, uint64_t initstate, uint64_t initseq) {
	rng->state = 0u;
	rng->inc = (initseq << 1u) | 1u;
	pcg32_next_uint(rng);
	rng->state += initstate;
	pcg32_next_uint(rng);
}

static GSX_PCG32_HOST_DEVICE inline void pcg32_init(pcg32 *rng, uint64_t initstate, uint64_t initseq) {
	pcg32_seed(rng, initstate, initseq);
}

static GSX_PCG32_HOST_DEVICE inline uint32_t pcg32_next_uint_bound(pcg32 *rng, uint32_t bound) {
	uint32_t threshold = (~bound + 1u) % bound;
	for(;;) {
		uint32_t r = pcg32_next_uint(rng);
		if(r >= threshold) {
			return r % bound;
		}
	}
}

static GSX_PCG32_HOST_DEVICE inline float pcg32_next_float(pcg32 *rng) {
	union {
		uint32_t u;
		float f;
	} x;
	x.u = (pcg32_next_uint(rng) >> 9u) | 0x3f800000u;
	return x.f - 1.0f;
}

static GSX_PCG32_HOST_DEVICE inline double pcg32_next_double(pcg32 *rng) {
	union {
		uint64_t u;
		double d;
	} x;
	x.u = ((uint64_t)pcg32_next_uint(rng) << 20u) | 0x3ff0000000000000ULL;
	return x.d - 1.0;
}

static GSX_PCG32_HOST_DEVICE inline void pcg32_advance(pcg32 *rng, int64_t delta_) {
	uint64_t cur_mult = PCG32_MULT;
	uint64_t cur_plus = rng->inc;
	uint64_t acc_mult = 1u;
	uint64_t acc_plus = 0u;
	uint64_t delta = (uint64_t)delta_;
	while(delta > 0u) {
		if((delta & 1u) != 0u) {
			acc_mult *= cur_mult;
			acc_plus = acc_plus * cur_mult + cur_plus;
		}
		cur_plus = (cur_mult + 1u) * cur_plus;
		cur_mult *= cur_mult;
		delta /= 2u;
	}
	rng->state = acc_mult * rng->state + acc_plus;
}

static GSX_PCG32_HOST_DEVICE inline int64_t pcg32_distance(const pcg32 *a, const pcg32 *b) {
	uint64_t cur_mult = PCG32_MULT;
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
	return (int64_t)distance;
}

static GSX_PCG32_HOST_DEVICE inline bool pcg32_equal(const pcg32 *a, const pcg32 *b) {
	return a->state == b->state && a->inc == b->inc;
}

static GSX_PCG32_HOST_DEVICE inline bool pcg32_not_equal(const pcg32 *a, const pcg32 *b) {
	return a->state != b->state || a->inc != b->inc;
}

#undef GSX_PCG32_HOST_DEVICE

#endif
