#include <metal_stdlib>
using namespace metal;

#include "simd_utils.metal"
#include "render_common.metal"

struct gsx_metal_render_per_pixel_upper {
	float3 grad_color_pixel;
	int last_contributor;
};

struct gsx_metal_render_per_pixel_lower {
	float3 color_pixel_after;
	float transmittance;
};

struct gsx_metal_render_per_pixel {
	float3 grad_color_pixel;
	int last_contributor;
	float3 color_pixel_after;
	float transmittance;
};

kernel void gsx_metal_render_blend_backward_kernel(
	device const int *tile_ranges [[buffer(0)]],
	device const int *tile_bucket_offsets [[buffer(1)]],
	device const int *bucket_tile_index [[buffer(2)]],
	device const int *instance_primitive_ids [[buffer(3)]],
	device const float *mean2d [[buffer(4)]],
	device const float *conic_opacity [[buffer(5)]],
	device const float *color [[buffer(6)]],
	device const float *image_chw [[buffer(7)]],
	device const float *alpha_hw [[buffer(8)]],
	device const int *tile_max_n_contributions [[buffer(9)]],
	device const int *tile_n_contributions [[buffer(10)]],
	device const float *bucket_color_transmittance [[buffer(11)]],
	device const float *grad_rgb [[buffer(12)]],
	device float *grad_mean2d [[buffer(13)]],
	device float *grad_conic [[buffer(14)]],
	device float *grad_raw_opacity_partial [[buffer(15)]],
	device float *grad_color [[buffer(16)]],
	constant gsx_metal_render_blend_backward_params &params [[buffer(17)]],
	uint3 group_id [[threadgroup_position_in_grid]],
	uint thread_rank [[thread_index_in_threadgroup]],
	uint simd_lane_id [[thread_index_in_simdgroup]])
{
	threadgroup gsx_metal_render_per_pixel_upper cached_per_pixel_upper[gsx_metal_render_tile_size / gsx_metal_render_simd_width][gsx_metal_render_simd_width];
	threadgroup gsx_metal_render_per_pixel_lower cached_per_pixel_lower[gsx_metal_render_tile_size / gsx_metal_render_simd_width][gsx_metal_render_simd_width];
	uint simdgroup_id = thread_rank / gsx_metal_render_simd_width;
	uint simdgroup_count = gsx_metal_render_tile_size / gsx_metal_render_simd_width;
	uint bucket_idx = group_id.x * simdgroup_count + simdgroup_id;
	float3 background = float3(params.background_r, params.background_g, params.background_b);
	uint tile_id;
	uint2 tile_coords;
	uint2 start_pixel_coords;
	int start;
	int end;
	int tile_n_primitives;
	int tile_bucket_base;
	int tile_bucket_idx;
	int tile_primitive_idx;
	int instance_idx;
	bool valid_primitive;
	device const float *bucket_ptr;
	float2 mean2d_local = float2(0.0f);
	float3 conic_local = float3(0.0f);
	float opacity_local = 0.0f;
	float3 color_local = float3(0.0f);
	float3 color_grad_factor = float3(0.0f);
	uint primitive_idx = 0u;
	float2 dL_dmean2d_accum = float2(0.0f);
	float3 dL_dconic_accum = float3(0.0f);
	float dL_draw_opacity_partial_accum = 0.0f;
	float3 dL_dcolor_accum = float3(0.0f);
	gsx_metal_render_per_pixel per_pixel;
	thread float3 &grad_pixel = per_pixel.grad_color_pixel;
	thread int &last_contributor = per_pixel.last_contributor;
	thread float3 &color_pixel_after = per_pixel.color_pixel_after;
	thread float &transmittance = per_pixel.transmittance;

	if(bucket_idx >= params.total_bucket_count) {
		return;
	}

	if(bucket_tile_index[bucket_idx] < 0) {
		return;
	}
	tile_id = uint(bucket_tile_index[bucket_idx]);
	if(tile_id >= params.tile_count) {
		return;
	}

	tile_coords = uint2(tile_id % params.grid_width, tile_id / params.grid_width);
	start_pixel_coords = uint2(tile_coords.x * gsx_metal_render_tile_width, tile_coords.y * gsx_metal_render_tile_height);
	start = tile_ranges[tile_id * 2u];
	end = tile_ranges[tile_id * 2u + 1u];
	if(start < 0 || end <= start) {
		return;
	}

	tile_n_primitives = end - start;
	tile_bucket_base = tile_id == 0u ? 0 : tile_bucket_offsets[tile_id - 1u];
	tile_bucket_idx = int(bucket_idx) - tile_bucket_base;
	if(tile_bucket_idx * int(gsx_metal_render_simd_width) >= tile_max_n_contributions[tile_id]) {
		return;
	}
	tile_primitive_idx = tile_bucket_idx * int(gsx_metal_render_simd_width) + int(simd_lane_id);
	instance_idx = start + tile_primitive_idx;
	valid_primitive = instance_idx < end;
	bucket_ptr = bucket_color_transmittance + bucket_idx * gsx_metal_render_tile_size * 4u;

	if(valid_primitive) {
		int primitive_id = instance_primitive_ids[instance_idx];
		uint p2 = uint(primitive_id) * 2u;
		uint p3 = uint(primitive_id) * 3u;
		uint p4 = uint(primitive_id) * 4u;
		float3 color_unclamped = float3(color[p3], color[p3 + 1u], color[p3 + 2u]);

		primitive_idx = uint(primitive_id);
		mean2d_local = float2(mean2d[p2], mean2d[p2 + 1u]);
		conic_local = float3(conic_opacity[p4], conic_opacity[p4 + 1u], conic_opacity[p4 + 2u]);
		opacity_local = conic_opacity[p4 + 3u];
		color_local = max(color_unclamped, float3(0.0f));
		color_grad_factor = float3(
			color_unclamped.x >= 0.0f ? 1.0f : 0.0f,
			color_unclamped.y >= 0.0f ? 1.0f : 0.0f,
			color_unclamped.z >= 0.0f ? 1.0f : 0.0f);
	}

	grad_pixel = float3(0.0f);
	last_contributor = 0;
	color_pixel_after = float3(0.0f);
	transmittance = 0.0f;

	for(uint ii = 0u; ii < gsx_metal_render_tile_size + gsx_metal_render_simd_width - 1u; ii += gsx_metal_render_simd_width) {
		if(ii < gsx_metal_render_tile_size) {
			uint load_idx = ii + simd_lane_id;
			gsx_metal_render_per_pixel_upper local_upper;
			gsx_metal_render_per_pixel_lower local_lower;

			local_upper.grad_color_pixel = float3(0.0f);
			local_upper.last_contributor = 0;
			local_lower.color_pixel_after = float3(0.0f);
			local_lower.transmittance = 0.0f;

			if(load_idx < gsx_metal_render_tile_size) {
				uint load_dx = load_idx % gsx_metal_render_tile_width;
				uint load_dy = load_idx / gsx_metal_render_tile_width;
				uint2 load_pixel_coords = uint2(start_pixel_coords.x + load_dx, start_pixel_coords.y + load_dy);
				bool valid_pixel = load_pixel_coords.x < params.width && load_pixel_coords.y < params.height;

				if(valid_pixel) {
					uint pixel_index = load_pixel_coords.y * params.width + load_pixel_coords.x;
					uint bucket_entry = load_idx * 4u;
					float4 color_transmittance = float4(
						bucket_ptr[bucket_entry],
						bucket_ptr[bucket_entry + 1u],
						bucket_ptr[bucket_entry + 2u],
						bucket_ptr[bucket_entry + 3u]);
					float pixel_transmittance = 1.0f - alpha_hw[pixel_index];

					local_upper.grad_color_pixel = float3(
						grad_rgb[pixel_index],
						grad_rgb[params.channel_stride + pixel_index],
						grad_rgb[2u * params.channel_stride + pixel_index]);
					local_upper.last_contributor = tile_n_contributions[pixel_index];
					local_lower.color_pixel_after = float3(
						image_chw[pixel_index] + pixel_transmittance * background.x,
						image_chw[params.channel_stride + pixel_index] + pixel_transmittance * background.y,
						image_chw[2u * params.channel_stride + pixel_index] + pixel_transmittance * background.z) - color_transmittance.xyz;
					local_lower.transmittance = color_transmittance.w;
				}
			}

			cached_per_pixel_upper[simdgroup_id][simd_lane_id] = local_upper;
			cached_per_pixel_lower[simdgroup_id][simd_lane_id] = local_lower;
			simdgroup_barrier(mem_flags::mem_threadgroup);
		}

		for(uint j = 0u; j < gsx_metal_render_simd_width; ++j) {
			uint i = ii + j;
			uint idx = i - simd_lane_id;
			uint dx = idx % gsx_metal_render_tile_width;
			uint dy = idx / gsx_metal_render_tile_width;
			uint2 pixel_coords = uint2(start_pixel_coords.x + dx, start_pixel_coords.y + dy);
			bool valid_pixel = pixel_coords.x < params.width && pixel_coords.y < params.height;
			bool valid_general = valid_primitive && valid_pixel && idx < gsx_metal_render_tile_size;

			grad_pixel = float3(
				gsx_metal_simd_shuffle_up(grad_pixel.x, 1u),
				gsx_metal_simd_shuffle_up(grad_pixel.y, 1u),
				gsx_metal_simd_shuffle_up(grad_pixel.z, 1u));
			last_contributor = gsx_metal_simd_shuffle_up(last_contributor, 1u);
			color_pixel_after = float3(
				gsx_metal_simd_shuffle_up(color_pixel_after.x, 1u),
				gsx_metal_simd_shuffle_up(color_pixel_after.y, 1u),
				gsx_metal_simd_shuffle_up(color_pixel_after.z, 1u));
			transmittance = gsx_metal_simd_shuffle_up(transmittance, 1u);

			if(simd_lane_id == 0u && valid_general) {
				grad_pixel = cached_per_pixel_upper[simdgroup_id][j].grad_color_pixel;
				last_contributor = cached_per_pixel_upper[simdgroup_id][j].last_contributor;
				color_pixel_after = cached_per_pixel_lower[simdgroup_id][j].color_pixel_after;
				transmittance = cached_per_pixel_lower[simdgroup_id][j].transmittance;
			}

			{
				bool skip = !valid_general || tile_primitive_idx >= last_contributor;
				float2 pixel = float2(float(pixel_coords.x), float(pixel_coords.y));
				float2 delta = (mean2d_local - 0.5f) - pixel;
				float3 delta_coefs = float3(delta.x * delta.x, delta.x * delta.y, delta.y * delta.y);
				float sigma_over_2_gt = 0.5f * (conic_local.x * delta_coefs.x + conic_local.z * delta_coefs.z) + conic_local.y * delta_coefs.y;
				float sigma_over_2 = max(sigma_over_2_gt, 0.0f);
				float gaussian = exp(-sigma_over_2);
				float alpha_prepare = opacity_local * gaussian;
				float color_dot_grad_pixel = dot(color_local, grad_pixel);
				float alpha = 0.0f;
				float blending_weight;
				float one_minus_alpha;
				float one_minus_alpha_safe;
				float one_minus_alpha_rcp;
				float3 dL_dcolor;
				float color_after_dot_grad;
				float2 prepare_dL_dmean2d;
				float dL_dalpha_from_color;
				float dL_draw_opacity_partial;
				float3 dL_dconic;
				float2 dL_dmean2d;

				if(!skip) {
					alpha = min(alpha_prepare, gsx_metal_render_max_alpha);
				}
				blending_weight = transmittance * alpha;
				dL_dcolor = blending_weight * (grad_pixel * color_grad_factor);
				dL_dcolor_accum += dL_dcolor;
				color_pixel_after -= blending_weight * color_local;
				color_after_dot_grad = dot(color_pixel_after, grad_pixel);
				prepare_dL_dmean2d = float2(
					conic_local.x * delta.x + conic_local.y * delta.y,
					conic_local.y * delta.x + conic_local.z * delta.y);
				one_minus_alpha = 1.0f - alpha;
				one_minus_alpha_safe = max(one_minus_alpha, 1.0e-4f);
				one_minus_alpha_rcp = 1.0f / one_minus_alpha_safe;
				dL_dalpha_from_color = transmittance * color_dot_grad_pixel - color_after_dot_grad * one_minus_alpha_rcp;
				dL_draw_opacity_partial = alpha_prepare >= gsx_metal_render_max_alpha ? 0.0f : alpha * dL_dalpha_from_color;
				dL_dconic = float3(
					-0.5f * dL_draw_opacity_partial * delta_coefs.x,
					-dL_draw_opacity_partial * delta_coefs.y,
					-0.5f * dL_draw_opacity_partial * delta_coefs.z);
				dL_dmean2d = dL_draw_opacity_partial * prepare_dL_dmean2d;

				dL_draw_opacity_partial_accum += dL_draw_opacity_partial;
				dL_dconic_accum += dL_dconic;
				dL_dmean2d_accum -= dL_dmean2d;
				transmittance *= one_minus_alpha;
			}
		}
	}

	if(valid_primitive) {
		gsx_metal_atomic_add_f32(grad_mean2d, primitive_idx * 2u, dL_dmean2d_accum.x);
		gsx_metal_atomic_add_f32(grad_mean2d, primitive_idx * 2u + 1u, dL_dmean2d_accum.y);
		gsx_metal_atomic_add_f32(grad_conic, primitive_idx * 3u, dL_dconic_accum.x);
		gsx_metal_atomic_add_f32(grad_conic, primitive_idx * 3u + 1u, dL_dconic_accum.y);
		gsx_metal_atomic_add_f32(grad_conic, primitive_idx * 3u + 2u, dL_dconic_accum.z);
		gsx_metal_atomic_add_f32(grad_raw_opacity_partial, primitive_idx, dL_draw_opacity_partial_accum);
		gsx_metal_atomic_add_f32(grad_color, primitive_idx * 3u, dL_dcolor_accum.x);
		gsx_metal_atomic_add_f32(grad_color, primitive_idx * 3u + 1u, dL_dcolor_accum.y);
		gsx_metal_atomic_add_f32(grad_color, primitive_idx * 3u + 2u, dL_dcolor_accum.z);
	}
}

kernel void gsx_metal_render_preprocess_backward_kernel(
	device const float *mean3d [[buffer(0)]],
	device const float *rotation [[buffer(1)]],
	device const float *logscale [[buffer(2)]],
	device const float *sh0 [[buffer(3)]],
	device const float *opacity_raw [[buffer(4)]],
	device const float *saved_mean2d [[buffer(5)]],
	device const float *saved_conic_opacity [[buffer(6)]],
	device const float *grad_mean2d [[buffer(7)]],
	device const float *grad_conic [[buffer(8)]],
	device const float *grad_raw_opacity_partial [[buffer(9)]],
	device const float *grad_color [[buffer(10)]],
	device float *grad_mean3d [[buffer(11)]],
	device float *grad_rotation [[buffer(12)]],
	device float *grad_logscale [[buffer(13)]],
	device float *grad_sh0 [[buffer(14)]],
	device float *grad_opacity [[buffer(15)]],
	constant gsx_metal_render_preprocess_backward_params &params [[buffer(16)]],
	uint primitive_idx [[thread_position_in_grid]])
{
	if(primitive_idx >= params.gaussian_count) {
		return;
	}

	uint base3 = primitive_idx * 3u;
	uint base4 = primitive_idx * 4u;
	uint base2 = primitive_idx * 2u;
	float3 dL_dcolor = float3(grad_color[base3], grad_color[base3 + 1u], grad_color[base3 + 2u]);
	float2 dL_dmean2d = float2(grad_mean2d[base2], grad_mean2d[base2 + 1u]);
	float3 dL_dconic = float3(grad_conic[base3], grad_conic[base3 + 1u], grad_conic[base3 + 2u]);
	float dL_draw_opacity_partial = grad_raw_opacity_partial[primitive_idx];
	float3 mean = float3(mean3d[base3], mean3d[base3 + 1u], mean3d[base3 + 2u]);
	float raw_opacity = opacity_raw[primitive_idx];
	float opacity = 1.0f / (1.0f + exp(-raw_opacity));
	float sx = exp(logscale[base3]);
	float sy = exp(logscale[base3 + 1u]);
	float sz = exp(logscale[base3 + 2u]);
	float var_x = sx * sx;
	float var_y = sy * sy;
	float var_z = sz * sz;
	float pose_qx = params.pose_qx;
	float pose_qy = params.pose_qy;
	float pose_qz = params.pose_qz;
	float pose_qw = params.pose_qw;
	float pose_qrr_raw = pose_qw * pose_qw;
	float pose_qxx_raw = pose_qx * pose_qx;
	float pose_qyy_raw = pose_qy * pose_qy;
	float pose_qzz_raw = pose_qz * pose_qz;
	float pose_q_norm_sq = pose_qrr_raw + pose_qxx_raw + pose_qyy_raw + pose_qzz_raw;
	float w2c_r11;
	float w2c_r12;
	float w2c_r13;
	float w2c_r21;
	float w2c_r22;
	float w2c_r23;
	float w2c_r31;
	float w2c_r32;
	float w2c_r33;
	float x_cam;
	float y_cam;
	float z;
	float x;
	float y;
	float clip_left;
	float clip_right;
	float clip_top;
	float clip_bottom;
	float tx;
	float ty;
	float j11;
	float j13;
	float j22;
	float j23;
	float raw_qx;
	float raw_qy;
	float raw_qz;
	float raw_qw;
	float qrr_raw;
	float qxx_raw;
	float qyy_raw;
	float qzz_raw;
	float q_norm_sq;
	float qxx;
	float qyy;
	float qzz;
	float qxy;
	float qxz;
	float qyz;
	float qrx;
	float qry;
	float qrz;
	float r11;
	float r12;
	float r13;
	float r21;
	float r22;
	float r23;
	float r31;
	float r32;
	float r33;
	float rs11;
	float rs12;
	float rs13;
	float rs21;
	float rs22;
	float rs23;
	float rs31;
	float rs32;
	float rs33;
	float cov11;
	float cov12;
	float cov13;
	float cov22;
	float cov23;
	float cov33;
	float3 jw1;
	float3 jw2;
	float3 jwc1;
	float3 jwc2;
	float a;
	float b;
	float c;
	float aa;
	float bb;
	float cc;
	float ac;
	float ab;
	float bc;
	float determinant;
	float determinant_rcp;
	float determinant_rcp_sq;
	float3 dL_dcov2d;
	float dtx_dx;
	float dty_dy;
	float dL_dj11;
	float dL_dj22;
	float dL_dj13;
	float dL_dj23;
	float dL_dj13_clamped;
	float dL_dj23_clamped;
	float djwr1_dz_helper;
	float djwr2_dz_helper;
	float3 dL_dmean3d_cam;
	float3 dL_dmean3d_from_splatting;
	float3 dL_dmean3d_total;
	float3 dL_djw_r1;
	float3 dL_djw_r2;
	float dL_dcov3d_11;
	float dL_dcov3d_12;
	float dL_dcov3d_13;
	float dL_dcov3d_22;
	float dL_dcov3d_23;
	float dL_dcov3d_33;
	float rotation_scale11;
	float rotation_scale12;
	float rotation_scale13;
	float rotation_scale21;
	float rotation_scale22;
	float rotation_scale23;
	float rotation_scale31;
	float rotation_scale32;
	float rotation_scale33;
	float dL_drotation_scale11;
	float dL_drotation_scale12;
	float dL_drotation_scale13;
	float dL_drotation_scale21;
	float dL_drotation_scale22;
	float dL_drotation_scale23;
	float dL_drotation_scale31;
	float dL_drotation_scale32;
	float dL_drotation_scale33;
	float3 dL_draw_scale;
	float dL_dR11;
	float dL_dR12;
	float dL_dR13;
	float dL_dR21;
	float dL_dR22;
	float dL_dR23;
	float dL_dR31;
	float dL_dR32;
	float dL_dR33;
	float q_norm;
	float inv_q_norm;
	float qn_w;
	float qn_x;
	float qn_y;
	float qn_z;
	float dL_dqnorm_w;
	float dL_dqnorm_x;
	float dL_dqnorm_y;
	float dL_dqnorm_z;
	float dot_qnorm_dL;
	float4 dL_draw_rotation;

	grad_sh0[base3] = 0.28209479177387814f * dL_dcolor.x;
	grad_sh0[base3 + 1u] = 0.28209479177387814f * dL_dcolor.y;
	grad_sh0[base3 + 2u] = 0.28209479177387814f * dL_dcolor.z;
	grad_opacity[primitive_idx] = dL_draw_opacity_partial * (1.0f - opacity);

	if(pose_q_norm_sq < 1.0e-8f) {
		return;
	}

	{
		float pose_qxx = 2.0f * pose_qxx_raw / pose_q_norm_sq;
		float pose_qyy = 2.0f * pose_qyy_raw / pose_q_norm_sq;
		float pose_qzz = 2.0f * pose_qzz_raw / pose_q_norm_sq;
		float pose_qxy = 2.0f * pose_qx * pose_qy / pose_q_norm_sq;
		float pose_qxz = 2.0f * pose_qx * pose_qz / pose_q_norm_sq;
		float pose_qyz = 2.0f * pose_qy * pose_qz / pose_q_norm_sq;
		float pose_qrx = 2.0f * pose_qw * pose_qx / pose_q_norm_sq;
		float pose_qry = 2.0f * pose_qw * pose_qy / pose_q_norm_sq;
		float pose_qrz = 2.0f * pose_qw * pose_qz / pose_q_norm_sq;

		w2c_r11 = 1.0f - (pose_qyy + pose_qzz);
		w2c_r12 = pose_qxy - pose_qrz;
		w2c_r13 = pose_qry + pose_qxz;
		w2c_r21 = pose_qrz + pose_qxy;
		w2c_r22 = 1.0f - (pose_qxx + pose_qzz);
		w2c_r23 = pose_qyz - pose_qrx;
		w2c_r31 = pose_qxz - pose_qry;
		w2c_r32 = pose_qrx + pose_qyz;
		w2c_r33 = 1.0f - (pose_qxx + pose_qyy);
	}

	x_cam = w2c_r11 * mean.x + w2c_r12 * mean.y + w2c_r13 * mean.z + params.pose_tx;
	y_cam = w2c_r21 * mean.x + w2c_r22 * mean.y + w2c_r23 * mean.z + params.pose_ty;
	z = w2c_r31 * mean.x + w2c_r32 * mean.y + w2c_r33 * mean.z + params.pose_tz;
	if(z <= params.near_plane || z >= params.far_plane || opacity < gsx_metal_render_min_alpha) {
		return;
	}

	raw_qx = rotation[base4];
	raw_qy = rotation[base4 + 1u];
	raw_qz = rotation[base4 + 2u];
	raw_qw = rotation[base4 + 3u];
	qrr_raw = raw_qw * raw_qw;
	qxx_raw = raw_qx * raw_qx;
	qyy_raw = raw_qy * raw_qy;
	qzz_raw = raw_qz * raw_qz;
	q_norm_sq = qrr_raw + qxx_raw + qyy_raw + qzz_raw;
	if(q_norm_sq < 1.0e-8f) {
		return;
	}

	qxx = 2.0f * qxx_raw / q_norm_sq;
	qyy = 2.0f * qyy_raw / q_norm_sq;
	qzz = 2.0f * qzz_raw / q_norm_sq;
	qxy = 2.0f * raw_qx * raw_qy / q_norm_sq;
	qxz = 2.0f * raw_qx * raw_qz / q_norm_sq;
	qyz = 2.0f * raw_qy * raw_qz / q_norm_sq;
	qrx = 2.0f * raw_qw * raw_qx / q_norm_sq;
	qry = 2.0f * raw_qw * raw_qy / q_norm_sq;
	qrz = 2.0f * raw_qw * raw_qz / q_norm_sq;

	r11 = 1.0f - (qyy + qzz);
	r12 = qxy - qrz;
	r13 = qry + qxz;
	r21 = qrz + qxy;
	r22 = 1.0f - (qxx + qzz);
	r23 = qyz - qrx;
	r31 = qxz - qry;
	r32 = qrx + qyz;
	r33 = 1.0f - (qxx + qyy);

	rs11 = r11 * var_x;
	rs12 = r12 * var_y;
	rs13 = r13 * var_z;
	rs21 = r21 * var_x;
	rs22 = r22 * var_y;
	rs23 = r23 * var_z;
	rs31 = r31 * var_x;
	rs32 = r32 * var_y;
	rs33 = r33 * var_z;

	cov11 = rs11 * r11 + rs12 * r12 + rs13 * r13;
	cov12 = rs11 * r21 + rs12 * r22 + rs13 * r23;
	cov13 = rs11 * r31 + rs12 * r32 + rs13 * r33;
	cov22 = rs21 * r21 + rs22 * r22 + rs23 * r23;
	cov23 = rs21 * r31 + rs22 * r32 + rs23 * r33;
	cov33 = rs31 * r31 + rs32 * r32 + rs33 * r33;

	x = x_cam / z;
	y = y_cam / z;
	clip_left = (-0.15f * float(params.width) - params.cx) / params.fx;
	clip_right = (1.15f * float(params.width) - params.cx) / params.fx;
	clip_top = (-0.15f * float(params.height) - params.cy) / params.fy;
	clip_bottom = (1.15f * float(params.height) - params.cy) / params.fy;
	tx = clamp(x, clip_left, clip_right);
	ty = clamp(y, clip_top, clip_bottom);
	j11 = params.fx / z;
	j13 = -j11 * tx;
	j22 = params.fy / z;
	j23 = -j22 * ty;

	jw1 = float3(j11 * w2c_r11 + j13 * w2c_r31, j11 * w2c_r12 + j13 * w2c_r32, j11 * w2c_r13 + j13 * w2c_r33);
	jw2 = float3(j22 * w2c_r21 + j23 * w2c_r31, j22 * w2c_r22 + j23 * w2c_r32, j22 * w2c_r23 + j23 * w2c_r33);
	jwc1 = float3(
		jw1.x * cov11 + jw1.y * cov12 + jw1.z * cov13,
		jw1.x * cov12 + jw1.y * cov22 + jw1.z * cov23,
		jw1.x * cov13 + jw1.y * cov23 + jw1.z * cov33);
	jwc2 = float3(
		jw2.x * cov11 + jw2.y * cov12 + jw2.z * cov13,
		jw2.x * cov12 + jw2.y * cov22 + jw2.z * cov23,
		jw2.x * cov13 + jw2.y * cov23 + jw2.z * cov33);
	a = dot(jwc1, jw1) + 0.3f;
	b = dot(jwc1, jw2);
	c = dot(jwc2, jw2) + 0.3f;
	determinant = a * c - b * b;
	if(determinant <= 1.0e-8f) {
		return;
	}

	aa = a * a;
	bb = b * b;
	cc = c * c;
	ac = a * c;
	ab = a * b;
	bc = b * c;
	determinant_rcp = 1.0f / determinant;
	determinant_rcp_sq = determinant_rcp * determinant_rcp;
	dL_dcov2d = determinant_rcp_sq * float3(
		bc * dL_dconic.y - cc * dL_dconic.x - bb * dL_dconic.z,
		bc * dL_dconic.x - 0.5f * (ac + bb) * dL_dconic.y + ab * dL_dconic.z,
		ab * dL_dconic.y - bb * dL_dconic.x - aa * dL_dconic.z);

	dL_dcov3d_11 = (jw1.x * jw1.x) * dL_dcov2d.x + 2.0f * (jw1.x * jw2.x) * dL_dcov2d.y + (jw2.x * jw2.x) * dL_dcov2d.z;
	dL_dcov3d_12 = (jw1.x * jw1.y) * dL_dcov2d.x + (jw1.x * jw2.y + jw1.y * jw2.x) * dL_dcov2d.y + (jw2.x * jw2.y) * dL_dcov2d.z;
	dL_dcov3d_13 = (jw1.x * jw1.z) * dL_dcov2d.x + (jw1.x * jw2.z + jw1.z * jw2.x) * dL_dcov2d.y + (jw2.x * jw2.z) * dL_dcov2d.z;
	dL_dcov3d_22 = (jw1.y * jw1.y) * dL_dcov2d.x + 2.0f * (jw1.y * jw2.y) * dL_dcov2d.y + (jw2.y * jw2.y) * dL_dcov2d.z;
	dL_dcov3d_23 = (jw1.y * jw1.z) * dL_dcov2d.x + (jw1.y * jw2.z + jw1.z * jw2.y) * dL_dcov2d.y + (jw2.y * jw2.z) * dL_dcov2d.z;
	dL_dcov3d_33 = (jw1.z * jw1.z) * dL_dcov2d.x + 2.0f * (jw1.z * jw2.z) * dL_dcov2d.y + (jw2.z * jw2.z) * dL_dcov2d.z;

	dL_djw_r1 = 2.0f * float3(
		jwc1.x * dL_dcov2d.x + jwc2.x * dL_dcov2d.y,
		jwc1.y * dL_dcov2d.x + jwc2.y * dL_dcov2d.y,
		jwc1.z * dL_dcov2d.x + jwc2.z * dL_dcov2d.y);
	dL_djw_r2 = 2.0f * float3(
		jwc1.x * dL_dcov2d.y + jwc2.x * dL_dcov2d.z,
		jwc1.y * dL_dcov2d.y + jwc2.y * dL_dcov2d.z,
		jwc1.z * dL_dcov2d.y + jwc2.z * dL_dcov2d.z);

	dL_dj11 = w2c_r11 * dL_djw_r1.x + w2c_r12 * dL_djw_r1.y + w2c_r13 * dL_djw_r1.z;
	dL_dj22 = w2c_r21 * dL_djw_r2.x + w2c_r22 * dL_djw_r2.y + w2c_r23 * dL_djw_r2.z;
	dL_dj13 = w2c_r31 * dL_djw_r1.x + w2c_r32 * dL_djw_r1.y + w2c_r33 * dL_djw_r1.z;
	dL_dj23 = w2c_r31 * dL_djw_r2.x + w2c_r32 * dL_djw_r2.y + w2c_r33 * dL_djw_r2.z;

	dtx_dx = (x > clip_left && x < clip_right) ? 1.0f : 0.0f;
	dty_dy = (y > clip_top && y < clip_bottom) ? 1.0f : 0.0f;
	dL_dj13_clamped = dL_dj13 * dtx_dx;
	dL_dj23_clamped = dL_dj23 * dty_dy;
	djwr1_dz_helper = dL_dj11 - 2.0f * tx * dL_dj13_clamped;
	djwr2_dz_helper = dL_dj22 - 2.0f * ty * dL_dj23_clamped;

	dL_dmean3d_cam = float3(
		j11 * (dL_dmean2d.x - dL_dj13_clamped / z),
		j22 * (dL_dmean2d.y - dL_dj23_clamped / z),
		-j11 * (x * dL_dmean2d.x + djwr1_dz_helper / z) - j22 * (y * dL_dmean2d.y + djwr2_dz_helper / z));

	dL_dmean3d_from_splatting = float3(
		w2c_r11 * dL_dmean3d_cam.x + w2c_r21 * dL_dmean3d_cam.y + w2c_r31 * dL_dmean3d_cam.z,
		w2c_r12 * dL_dmean3d_cam.x + w2c_r22 * dL_dmean3d_cam.y + w2c_r32 * dL_dmean3d_cam.z,
		w2c_r13 * dL_dmean3d_cam.x + w2c_r23 * dL_dmean3d_cam.y + w2c_r33 * dL_dmean3d_cam.z);
	dL_dmean3d_total = dL_dmean3d_from_splatting;

	grad_mean3d[base3] = dL_dmean3d_total.x;
	grad_mean3d[base3 + 1u] = dL_dmean3d_total.y;
	grad_mean3d[base3 + 2u] = dL_dmean3d_total.z;

	rotation_scale11 = r11 * sx;
	rotation_scale12 = r12 * sy;
	rotation_scale13 = r13 * sz;
	rotation_scale21 = r21 * sx;
	rotation_scale22 = r22 * sy;
	rotation_scale23 = r23 * sz;
	rotation_scale31 = r31 * sx;
	rotation_scale32 = r32 * sy;
	rotation_scale33 = r33 * sz;

	dL_drotation_scale11 = 2.0f * (dL_dcov3d_11 * rotation_scale11 + dL_dcov3d_12 * rotation_scale21 + dL_dcov3d_13 * rotation_scale31);
	dL_drotation_scale12 = 2.0f * (dL_dcov3d_11 * rotation_scale12 + dL_dcov3d_12 * rotation_scale22 + dL_dcov3d_13 * rotation_scale32);
	dL_drotation_scale13 = 2.0f * (dL_dcov3d_11 * rotation_scale13 + dL_dcov3d_12 * rotation_scale23 + dL_dcov3d_13 * rotation_scale33);
	dL_drotation_scale21 = 2.0f * (dL_dcov3d_12 * rotation_scale11 + dL_dcov3d_22 * rotation_scale21 + dL_dcov3d_23 * rotation_scale31);
	dL_drotation_scale22 = 2.0f * (dL_dcov3d_12 * rotation_scale12 + dL_dcov3d_22 * rotation_scale22 + dL_dcov3d_23 * rotation_scale32);
	dL_drotation_scale23 = 2.0f * (dL_dcov3d_12 * rotation_scale13 + dL_dcov3d_22 * rotation_scale23 + dL_dcov3d_23 * rotation_scale33);
	dL_drotation_scale31 = 2.0f * (dL_dcov3d_13 * rotation_scale11 + dL_dcov3d_23 * rotation_scale21 + dL_dcov3d_33 * rotation_scale31);
	dL_drotation_scale32 = 2.0f * (dL_dcov3d_13 * rotation_scale12 + dL_dcov3d_23 * rotation_scale22 + dL_dcov3d_33 * rotation_scale32);
	dL_drotation_scale33 = 2.0f * (dL_dcov3d_13 * rotation_scale13 + dL_dcov3d_23 * rotation_scale23 + dL_dcov3d_33 * rotation_scale33);

	dL_draw_scale = float3(
		(dL_drotation_scale11 * r11 + dL_drotation_scale21 * r21 + dL_drotation_scale31 * r31) * sx,
		(dL_drotation_scale12 * r12 + dL_drotation_scale22 * r22 + dL_drotation_scale32 * r32) * sy,
		(dL_drotation_scale13 * r13 + dL_drotation_scale23 * r23 + dL_drotation_scale33 * r33) * sz);
	grad_logscale[base3] = dL_draw_scale.x;
	grad_logscale[base3 + 1u] = dL_draw_scale.y;
	grad_logscale[base3 + 2u] = dL_draw_scale.z;

	dL_dR11 = dL_drotation_scale11 * sx;
	dL_dR12 = dL_drotation_scale12 * sy;
	dL_dR13 = dL_drotation_scale13 * sz;
	dL_dR21 = dL_drotation_scale21 * sx;
	dL_dR22 = dL_drotation_scale22 * sy;
	dL_dR23 = dL_drotation_scale23 * sz;
	dL_dR31 = dL_drotation_scale31 * sx;
	dL_dR32 = dL_drotation_scale32 * sy;
	dL_dR33 = dL_drotation_scale33 * sz;

	q_norm = sqrt(q_norm_sq);
	inv_q_norm = 1.0f / q_norm;
	qn_w = raw_qw * inv_q_norm;
	qn_x = raw_qx * inv_q_norm;
	qn_y = raw_qy * inv_q_norm;
	qn_z = raw_qz * inv_q_norm;

	dL_dqnorm_w =
		dL_dR12 * (-2.0f * qn_z) + dL_dR13 * (2.0f * qn_y) +
		dL_dR21 * (2.0f * qn_z) + dL_dR23 * (-2.0f * qn_x) +
		dL_dR31 * (-2.0f * qn_y) + dL_dR32 * (2.0f * qn_x);
	dL_dqnorm_x =
		dL_dR12 * (2.0f * qn_y) + dL_dR13 * (2.0f * qn_z) +
		dL_dR21 * (2.0f * qn_y) + dL_dR22 * (-4.0f * qn_x) + dL_dR23 * (-2.0f * qn_w) +
		dL_dR31 * (2.0f * qn_z) + dL_dR32 * (2.0f * qn_w) + dL_dR33 * (-4.0f * qn_x);
	dL_dqnorm_y =
		dL_dR11 * (-4.0f * qn_y) + dL_dR12 * (2.0f * qn_x) + dL_dR13 * (2.0f * qn_w) +
		dL_dR21 * (2.0f * qn_x) + dL_dR23 * (2.0f * qn_z) +
		dL_dR31 * (-2.0f * qn_w) + dL_dR32 * (2.0f * qn_z) + dL_dR33 * (-4.0f * qn_y);
	dL_dqnorm_z =
		dL_dR11 * (-4.0f * qn_z) + dL_dR12 * (-2.0f * qn_w) + dL_dR13 * (2.0f * qn_x) +
		dL_dR21 * (2.0f * qn_w) + dL_dR22 * (-4.0f * qn_z) + dL_dR23 * (2.0f * qn_y) +
		dL_dR31 * (2.0f * qn_x) + dL_dR32 * (2.0f * qn_y);

	dot_qnorm_dL = qn_w * dL_dqnorm_w + qn_x * dL_dqnorm_x + qn_y * dL_dqnorm_y + qn_z * dL_dqnorm_z;
	dL_draw_rotation = float4(
		(dL_dqnorm_x - qn_x * dot_qnorm_dL) * inv_q_norm,
		(dL_dqnorm_y - qn_y * dot_qnorm_dL) * inv_q_norm,
		(dL_dqnorm_z - qn_z * dot_qnorm_dL) * inv_q_norm,
		(dL_dqnorm_w - qn_w * dot_qnorm_dL) * inv_q_norm);
	grad_rotation[base4] = dL_draw_rotation.x;
	grad_rotation[base4 + 1u] = dL_draw_rotation.y;
	grad_rotation[base4 + 2u] = dL_draw_rotation.z;
	grad_rotation[base4 + 3u] = dL_draw_rotation.w;

	(void)sh0;
	(void)saved_mean2d;
	(void)saved_conic_opacity;
}
