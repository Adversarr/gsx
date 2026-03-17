#ifndef GSX_IO_PLY_H
#define GSX_IO_PLY_H

#include "gsx/gsx-core.h"

GSX_EXTERN_C_BEGIN

/**
 * Read a PLY file into a pre-initialized gaussian splat object.
 *
 * The caller must initialize `*out_gs` with `gsx_gs_init` before calling this
 * function. The gs object determines the backend, buffer type, and arena
 * configuration used for storing the loaded data. The gaussian count will be
 * resized to match the number of vertices in the PLY file.
 *
 * @param out_gs    Pointer to an initialized gs handle. Must not be NULL.
 * @param filename  Path to the PLY file. Must not be NULL.
 * @return          GSX_ERROR_SUCCESS on success, or an appropriate error code.
 */
gsx_error gsx_read_ply(gsx_gs_t* out_gs, const char* filename);

gsx_error gsx_write_ply(gsx_gs_t gs, const char* filename);

GSX_EXTERN_C_END

#endif
