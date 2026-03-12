#ifndef GSX_IMPL_H
#define GSX_IMPL_H

/*
 * Internal implementation notes for the stable GSX public API surface:
 *
 * - Only headers under gsx/include/gsx are part of the stable ABI.
 * - Public descriptor/state/result structs are plain POD values in v0.
 * - Public backend-bound work is externally observed as one totally ordered
 *   major stream per backend.
 * - Private helper threads or streams are allowed for dataloader prefetch only.
 * - Layering is one-way:
 *   core <- backend/render/data/loss/optim/adc <- runtime
 * - Runtime objects own replay-critical state only. They do not own the caller
 *   training loop.
 * - GS/optimizer structural mutation and ADC steps must remain transactional.
 */

#endif /* GSX_IMPL_H */
