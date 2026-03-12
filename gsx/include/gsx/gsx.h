#ifndef GSX_H
#define GSX_H

/*
 * Umbrella header for the stable GSX public API.
 *
 * Layering:
 * - gsx-base.h: versioning, error handling, handle forward declarations, value types
 * - gsx-core.h: arenas, tensors, gaussian state, core health checks
 * - subsystem headers: backend, render, data, loss, optim, adc
 * - gsx-runtime.h: replay-critical runtime state, scheduling, checkpoint I/O
 */

#include "gsx-base.h"
#include "gsx-core.h"
#include "gsx-backend.h"
#include "gsx-render.h"
#include "gsx-data.h"
#include "gsx-loss.h"
#include "gsx-optim.h"
#include "gsx-adc.h"
#include "gsx-runtime.h"

#endif /* GSX_H */
