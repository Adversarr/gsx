/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "../../helper_math.h"

#define DEF inline constexpr

namespace fast_gs::rasterization::config {
#ifdef NDEBUG
DEF bool debug = false;
#else
DEF bool debug = true;
#endif
DEF float dilation = 0.3f;
DEF float min_alpha_threshold_rcp = 255.0f;
DEF float min_alpha_threshold = 1.0f / min_alpha_threshold_rcp;
DEF float min_alpha_threshold_deactivated = -5.537334267018537f;
DEF float max_fragment_alpha = 0.999f;
DEF float transmittance_threshold = 1e-4f;
DEF int block_size_preprocess = 128;
DEF int block_size_preprocess_backward = 128;
DEF int block_size_apply_depth_ordering = 256;
DEF int block_size_create_instances = 256;
DEF int block_size_extract_instance_ranges = 256;
DEF int block_size_extract_bucket_counts = 256;
DEF int tile_width = 16;
DEF int tile_width_minus_1 = tile_width - 1;
DEF int tile_width_log2 = 4;
DEF int tile_height = 16;
DEF int block_size_blend = tile_width * tile_height;
DEF int block_size_blend_mask = block_size_blend - 1;
DEF int n_sequential_threshold = 8;
DEF int blend_bwd_n_warps = 8;
DEF int blend_bwd2_n_warps = 4;
DEF float math_pi = 3.14159265358979323846f;
} /* namespace fast_gs::rasterization::config */

namespace config = fast_gs::rasterization::config;

#undef DEF
