#pragma once

namespace gs_nvtx {

static inline int catM(void) { return 0; }
static inline int catK(void) { return 0; }
static inline int catC(void) { return 0; }
static inline int catCp(void) { return 0; }

} /* namespace gs_nvtx */

#define C_GREEN 0
#define C_BLUE 0
#define C_ORANGE 0
#define C_PURPLE 0
#define C_PINK 0
#define C_CYAN 0
#define C_RED 0
#define C_GRAY 0

#define GS_FUNC_RANGE() do { } while(false)
#define GS_RANGE_SCOPE(name, color, category, size) do { (void)(size); } while(false)
