#include <gsx/extra/gsx-io-ply.h>
#include <gsx/extra/gsx-flann.h>
#include <gsx/extra/gsx-stbi.h>
#include <gsx/gsx.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct app_options {
	const char *input_ply_path;
	const char *output_image_path;

	gsx_index_t width;
	gsx_index_t height;
	gsx_float_t fx;
	gsx_float_t fy;
	gsx_float_t cx;
	gsx_float_t cy;

	/* CLI accepts qvec in wxyz order; we convert to xyzw for gsx_camera_pose. */
	gsx_float_t qvec_wxyz[4];
	gsx_float_t tvec_xyz[3];

	gsx_backend_type backend_type;
	gsx_index_t device_index;

	bool flann_enable;
	gsx_index_t flann_num_neighbors;
	gsx_float_t flann_init_scaling;
	gsx_float_t flann_min_distance;
	gsx_float_t flann_max_distance;
	gsx_float_t flann_default_distance;
} app_options;

typedef struct app_state {
	gsx_backend_t backend;
	gsx_renderer_t renderer;
	gsx_render_context_t render_context;
	gsx_arena_t render_arena;
	gsx_tensor_t out_rgb;
	gsx_gs_t gs;

	void *host_rgb;
	gsx_size_t host_rgb_size_bytes;
} app_state;

static bool gsx_check(gsx_error error, const char *context)
{
	if(gsx_error_is_success(error)) {
		return true;
	}

	fprintf(stderr, "error: %s failed (%d)", context, error.code);
	if(error.message != NULL) {
		fprintf(stderr, ": %s", error.message);
	}
	fprintf(stderr, "\n");
	return false;
}

static void set_default_options(app_options *options)
{
	memset(options, 0, sizeof(*options));

	options->input_ply_path = NULL;
	options->output_image_path = "rendered.png";

	options->width = 740;
	options->height = 480;
	options->fx = 740.0f;
	options->fy = 740.0f;
	options->cx = (gsx_float_t)options->width * 0.5f;
	options->cy = (gsx_float_t)options->height * 0.5f;

	options->qvec_wxyz[0] = 0.9848955175050466f;
	options->qvec_wxyz[1] = -0.1423900720001207f;
	options->qvec_wxyz[2] = -0.09478210164018773f;
	options->qvec_wxyz[3] = -0.026874527027207384f;

	options->tvec_xyz[0] = -0.33941549888781136f;
	options->tvec_xyz[1] = -1.9337371871740343f;
	options->tvec_xyz[2] = 3.8356418172368407f;

#if defined(__APPLE__)
	options->backend_type = GSX_BACKEND_TYPE_METAL;
#else
	options->backend_type = GSX_BACKEND_TYPE_CPU;
#endif
	options->device_index = 0;
	options->flann_enable = false;
	options->flann_num_neighbors = 16;
	options->flann_init_scaling = 1.0f;
	options->flann_min_distance = 0.0001f;
	options->flann_max_distance = 10.0f;
	options->flann_default_distance = 0.001f;}

static void print_usage(const char *program_name)
{
	fprintf(stderr, "usage: %s --input <pointcloud.ply> [options]\n", program_name);
	fprintf(stderr, "\nrequired:\n");
	fprintf(stderr, "  --input <path>                 PLY point cloud input path\n");
	fprintf(stderr, "\noutput:\n");
	fprintf(stderr, "  --output <path>                Output image path (default: rendered.png)\n");
	fprintf(stderr, "\ncamera intrinsics:\n");
	fprintf(stderr, "  --width <int>                  Image width (default: 740)\n");
	fprintf(stderr, "  --height <int>                 Image height (default: 480)\n");
	fprintf(stderr, "  --fx <float>                   Focal length x (default: 740)\n");
	fprintf(stderr, "  --fy <float>                   Focal length y (default: 740)\n");
	fprintf(stderr, "  --cx <float>                   Principal point x (default: width / 2)\n");
	fprintf(stderr, "  --cy <float>                   Principal point y (default: height / 2)\n");
	fprintf(stderr, "\ncamera extrinsics:\n");
	fprintf(stderr, "  --qvec <w,x,y,z>               Quaternion in COLMAP wxyz order\n");
	fprintf(stderr, "  --tvec <x,y,z>                 Translation vector\n");
	fprintf(stderr, "\nbackend (optional):\n");
	fprintf(stderr, "  --backend <cpu|cuda|metal>     Backend type (default: metal on macOS, else cpu)\n");
	fprintf(stderr, "  --device <index>               Device index inside selected backend (default: 0)\n");
	fprintf(stderr, "\nflann initialization (optional):\n");
	fprintf(stderr, "  --flann                        Enable FLANN-based scale+rotation initialization (off by default)\n");
	fprintf(stderr, "  --flann-knn <n>                K nearest neighbors (default: 16)\n");
	fprintf(stderr, "  --flann-scale <f>              Scale multiplier (default: 1.0)\n");
	fprintf(stderr, "  --flann-min <f>                Minimum scale distance (default: 0.01)\n");
	fprintf(stderr, "  --flann-max <f>                Maximum scale distance (default: 10.0)\n");
	fprintf(stderr, "  --flann-default <f>            Default distance if no neighbors (default: 1.0)\n");
	fprintf(stderr, "\nnotes:\n");
	fprintf(stderr, "  - qvec is accepted as wxyz and converted to renderer pose xyzw internally.\n");
	fprintf(stderr, "  - renderer uses SH degree 0 for compatibility with plain point-cloud PLY files.\n");
}

static bool parse_i64(const char *value, int64_t *out_value)
{
	char *end_ptr = NULL;
	long long parsed = 0;

	if(value == NULL || out_value == NULL) {
		return false;
	}

	errno = 0;
	parsed = strtoll(value, &end_ptr, 10);
	if(errno != 0 || end_ptr == value || *end_ptr != '\0') {
		return false;
	}

	*out_value = (int64_t)parsed;
	return true;
}

static bool parse_f32(const char *value, gsx_float_t *out_value)
{
	char *end_ptr = NULL;
	float parsed = 0.0f;

	if(value == NULL || out_value == NULL) {
		return false;
	}

	errno = 0;
	parsed = strtof(value, &end_ptr);
	if(errno != 0 || end_ptr == value || *end_ptr != '\0') {
		return false;
	}

	*out_value = parsed;
	return true;
}

static bool parse_f32_csv(const char *value, gsx_float_t *out_values, size_t expected_count)
{
	char *buffer = NULL;
	char *token = NULL;
	size_t count = 0;

	if(value == NULL || out_values == NULL || expected_count == 0u) {
		return false;
	}

	buffer = (char *)malloc(strlen(value) + 1u);
	if(buffer == NULL) {
		return false;
	}
	strcpy(buffer, value);

	token = strtok(buffer, ",");
	while(token != NULL) {
		if(count >= expected_count || !parse_f32(token, &out_values[count])) {
			free(buffer);
			return false;
		}
		++count;
		token = strtok(NULL, ",");
	}

	free(buffer);
	return count == expected_count;
}

static bool parse_backend_type(const char *value, gsx_backend_type *out_backend_type)
{
	if(value == NULL || out_backend_type == NULL) {
		return false;
	}

	if(strcmp(value, "cpu") == 0) {
		*out_backend_type = GSX_BACKEND_TYPE_CPU;
		return true;
	}
	if(strcmp(value, "cuda") == 0) {
		*out_backend_type = GSX_BACKEND_TYPE_CUDA;
		return true;
	}
	if(strcmp(value, "metal") == 0) {
		*out_backend_type = GSX_BACKEND_TYPE_METAL;
		return true;
	}

	return false;
}

static bool parse_args(int argc, char **argv, app_options *options)
{
	int i = 0;

	for(i = 1; i < argc; ++i) {
		const char *arg = argv[i];
		if(strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
			print_usage(argv[0]);
			return false;
		}

		if(strcmp(arg, "--input") == 0) {
			if(i + 1 >= argc) {
				fprintf(stderr, "error: --input requires a value\n");
				return false;
			}
			options->input_ply_path = argv[++i];
			continue;
		}
		if(strcmp(arg, "--output") == 0) {
			if(i + 1 >= argc) {
				fprintf(stderr, "error: --output requires a value\n");
				return false;
			}
			options->output_image_path = argv[++i];
			continue;
		}

		if(strcmp(arg, "--width") == 0) {
			int64_t value = 0;
			if(i + 1 >= argc || !parse_i64(argv[++i], &value) || value <= 0) {
				fprintf(stderr, "error: invalid --width\n");
				return false;
			}
			options->width = (gsx_index_t)value;
			continue;
		}
		if(strcmp(arg, "--height") == 0) {
			int64_t value = 0;
			if(i + 1 >= argc || !parse_i64(argv[++i], &value) || value <= 0) {
				fprintf(stderr, "error: invalid --height\n");
				return false;
			}
			options->height = (gsx_index_t)value;
			continue;
		}
		if(strcmp(arg, "--fx") == 0) {
			if(i + 1 >= argc || !parse_f32(argv[++i], &options->fx)) {
				fprintf(stderr, "error: invalid --fx\n");
				return false;
			}
			continue;
		}
		if(strcmp(arg, "--fy") == 0) {
			if(i + 1 >= argc || !parse_f32(argv[++i], &options->fy)) {
				fprintf(stderr, "error: invalid --fy\n");
				return false;
			}
			continue;
		}
		if(strcmp(arg, "--cx") == 0) {
			if(i + 1 >= argc || !parse_f32(argv[++i], &options->cx)) {
				fprintf(stderr, "error: invalid --cx\n");
				return false;
			}
			continue;
		}
		if(strcmp(arg, "--cy") == 0) {
			if(i + 1 >= argc || !parse_f32(argv[++i], &options->cy)) {
				fprintf(stderr, "error: invalid --cy\n");
				return false;
			}
			continue;
		}

		if(strcmp(arg, "--qvec") == 0) {
			if(i + 1 >= argc || !parse_f32_csv(argv[++i], options->qvec_wxyz, 4u)) {
				fprintf(stderr, "error: invalid --qvec, expected w,x,y,z\n");
				return false;
			}
			continue;
		}
		if(strcmp(arg, "--tvec") == 0) {
			if(i + 1 >= argc || !parse_f32_csv(argv[++i], options->tvec_xyz, 3u)) {
				fprintf(stderr, "error: invalid --tvec, expected x,y,z\n");
				return false;
			}
			continue;
		}

		if(strcmp(arg, "--backend") == 0) {
			if(i + 1 >= argc || !parse_backend_type(argv[++i], &options->backend_type)) {
				fprintf(stderr, "error: invalid --backend, expected cpu|cuda|metal\n");
				return false;
			}
			continue;
		}
		if(strcmp(arg, "--device") == 0) {
			int64_t value = 0;
			if(i + 1 >= argc || !parse_i64(argv[++i], &value) || value < 0) {
				fprintf(stderr, "error: invalid --device\n");
				return false;
			}
			options->device_index = (gsx_index_t)value;
			continue;
		}
		if(strcmp(arg, "--flann") == 0) {
			options->flann_enable = true;
			continue;
		}
		if(strcmp(arg, "--flann-knn") == 0) {
			int64_t value = 0;
			if(i + 1 >= argc || !parse_i64(argv[++i], &value) || value <= 0) {
				fprintf(stderr, "error: invalid --flann-knn\n");
				return false;
			}
			options->flann_num_neighbors = (gsx_index_t)value;
			continue;
		}
		if(strcmp(arg, "--flann-scale") == 0) {
			if(i + 1 >= argc || !parse_f32(argv[++i], &options->flann_init_scaling) || options->flann_init_scaling <= 0.0f) {
				fprintf(stderr, "error: invalid --flann-scale\n");
				return false;
			}
			continue;
		}
		if(strcmp(arg, "--flann-min") == 0) {
			if(i + 1 >= argc || !parse_f32(argv[++i], &options->flann_min_distance) || options->flann_min_distance <= 0.0f) {
				fprintf(stderr, "error: invalid --flann-min\n");
				return false;
			}
			continue;
		}
		if(strcmp(arg, "--flann-max") == 0) {
			if(i + 1 >= argc || !parse_f32(argv[++i], &options->flann_max_distance) || options->flann_max_distance <= 0.0f) {
				fprintf(stderr, "error: invalid --flann-max\n");
				return false;
			}
			continue;
		}
		if(strcmp(arg, "--flann-default") == 0) {
			if(i + 1 >= argc || !parse_f32(argv[++i], &options->flann_default_distance) || options->flann_default_distance <= 0.0f) {
				fprintf(stderr, "error: invalid --flann-default\n");
				return false;
			}
			continue;
		}

		if(arg[0] != '-') {
			if(options->input_ply_path != NULL) {
				fprintf(stderr, "error: multiple input paths provided\n");
				return false;
			}
			options->input_ply_path = arg;
			continue;
		}

		fprintf(stderr, "error: unknown argument: %s\n", arg);
		return false;
	}

	if(options->input_ply_path == NULL) {
		fprintf(stderr, "error: --input <pointcloud.ply> is required\n");
		return false;
	}

	if(options->fx <= 0.0f || options->fy <= 0.0f) {
		fprintf(stderr, "error: fx and fy must be > 0\n");
		return false;
	}

	if(options->output_image_path == NULL || options->output_image_path[0] == '\0') {
		fprintf(stderr, "error: --output path must be non-empty\n");
		return false;
	}

	return true;
}

static void cleanup_state(app_state *state)
{
	if(state == NULL) {
		return;
	}

	free(state->host_rgb);
	state->host_rgb = NULL;
	state->host_rgb_size_bytes = 0;

	if(state->out_rgb != NULL) {
		gsx_check(gsx_tensor_free(state->out_rgb), "gsx_tensor_free(out_rgb)");
		state->out_rgb = NULL;
	}

	if(state->render_arena != NULL) {
		gsx_check(gsx_arena_free(state->render_arena), "gsx_arena_free(render_arena)");
		state->render_arena = NULL;
	}

	if(state->render_context != NULL) {
		gsx_check(gsx_render_context_free(state->render_context), "gsx_render_context_free");
		state->render_context = NULL;
	}

	if(state->renderer != NULL) {
		gsx_check(gsx_renderer_free(state->renderer), "gsx_renderer_free");
		state->renderer = NULL;
	}

	if(state->gs != NULL) {
		gsx_check(gsx_gs_free(state->gs), "gsx_gs_free");
		state->gs = NULL;
	}

	if(state->backend != NULL) {
		gsx_check(gsx_backend_free(state->backend), "gsx_backend_free");
		state->backend = NULL;
	}
}

static bool init_out_rgb_tensor(gsx_arena_t arena, gsx_index_t width, gsx_index_t height, gsx_tensor_t *out_rgb)
{
	gsx_tensor_desc desc = { 0 };

	desc.rank = 3;
	desc.shape[0] = 3;
	desc.shape[1] = height;
	desc.shape[2] = width;
	desc.data_type = GSX_DATA_TYPE_F32;
	desc.storage_format = GSX_STORAGE_FORMAT_CHW;
	desc.arena = arena;

	return gsx_check(gsx_tensor_init(out_rgb, &desc), "gsx_tensor_init(out_rgb)");
}

static bool run_render(const app_options *options, app_state *state)
{
	gsx_index_t visible_device_count = 0;
	gsx_backend_device_t device = NULL;
	gsx_backend_desc backend_desc = { 0 };
	gsx_backend_buffer_type_t device_buffer_type = NULL;
	gsx_gs_desc gs_desc = { 0 };
	gsx_renderer_desc renderer_desc = { 0 };
	gsx_arena_desc render_arena_desc = { 0 };
	gsx_gs_info gs_info = { 0 };
	gsx_tensor_t gs_mean3d = NULL;
	gsx_tensor_t gs_rotation = NULL;
	gsx_tensor_t gs_logscale = NULL;
	gsx_tensor_t gs_sh0 = NULL;
	gsx_tensor_t gs_opacity = NULL;
	gsx_tensor_info out_rgb_info = { 0 };
	gsx_render_forward_request request = { 0 };
	gsx_camera_intrinsics intrinsics = { 0 };
	gsx_camera_pose pose = { 0 };

	memset(state, 0, sizeof(*state));

	if(!gsx_check(gsx_backend_registry_init(), "gsx_backend_registry_init")) {
		return false;
	}

	if(!gsx_check(gsx_count_backend_devices_by_type(options->backend_type, &visible_device_count), "gsx_count_backend_devices_by_type")) {
		return false;
	}
	if(visible_device_count <= 0) {
		fprintf(stderr, "error: no visible devices for backend type %d\n", options->backend_type);
		return false;
	}
	if(options->device_index < 0 || options->device_index >= visible_device_count) {
		fprintf(stderr, "error: device index %d out of range [0, %d)\n", options->device_index, visible_device_count);
		return false;
	}

	if(!gsx_check(gsx_get_backend_device_by_type(options->backend_type, options->device_index, &device), "gsx_get_backend_device_by_type")) {
		return false;
	}

	backend_desc.device = device;
	if(!gsx_check(gsx_backend_init(&state->backend, &backend_desc), "gsx_backend_init")) {
		return false;
	}

	if(!gsx_check(gsx_backend_find_buffer_type(state->backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &device_buffer_type), "gsx_backend_find_buffer_type(device)")) {
		return false;
	}

	gs_desc.buffer_type = device_buffer_type;
	gs_desc.arena_desc.initial_capacity_bytes = (gsx_size_t)(64u << 20);
	gs_desc.arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
	gs_desc.count = 0;
	gs_desc.aux_flags = GSX_GS_AUX_DEFAULT;
	if(!gsx_check(gsx_gs_init(&state->gs, &gs_desc), "gsx_gs_init")) {
		return false;
	}

	if(!gsx_check(gsx_read_ply(&state->gs, options->input_ply_path), "gsx_read_ply")) {
		return false;
	}

	if(options->flann_enable) {
		if(!gsx_check(gsx_gs_recompute_scale_rotation_flann(
			state->gs,
			options->flann_num_neighbors,
			options->flann_init_scaling,
			options->flann_min_distance,
			options->flann_max_distance,
			options->flann_default_distance,
			1.0f,
			false),
			"gsx_gs_recompute_scale_rotation_flann")) {
			return false;
		}
	}

	if(!gsx_check(gsx_gs_get_info(state->gs, &gs_info), "gsx_gs_get_info")) {
		return false;
	}
	if(gs_info.count == 0u) {
		fprintf(stderr, "error: input point cloud produced zero gaussians\n");
		return false;
	}

	renderer_desc.width = options->width;
	renderer_desc.height = options->height;
	renderer_desc.output_data_type = GSX_DATA_TYPE_F32;
	renderer_desc.feature_flags = 0u;
	renderer_desc.enable_alpha_output = false;
	renderer_desc.enable_invdepth_output = false;
	if(!gsx_check(gsx_renderer_init(&state->renderer, state->backend, &renderer_desc), "gsx_renderer_init")) {
		return false;
	}
	if(!gsx_check(gsx_render_context_init(&state->render_context, state->renderer), "gsx_render_context_init")) {
		return false;
	}

	render_arena_desc.initial_capacity_bytes = (gsx_size_t)options->width * (gsx_size_t)options->height * 3u * sizeof(gsx_float_t);
	render_arena_desc.growth_mode = GSX_ARENA_GROWTH_MODE_GROW_ON_DEMAND;
	if(!gsx_check(gsx_arena_init(&state->render_arena, device_buffer_type, &render_arena_desc), "gsx_arena_init(render_arena)")) {
		return false;
	}

	if(!init_out_rgb_tensor(state->render_arena, options->width, options->height, &state->out_rgb)) {
		return false;
	}

	if(!gsx_check(gsx_gs_get_field(state->gs, GSX_GS_FIELD_MEAN3D, &gs_mean3d), "gsx_gs_get_field(mean3d)")) {
		return false;
	}
	if(!gsx_check(gsx_gs_get_field(state->gs, GSX_GS_FIELD_ROTATION, &gs_rotation), "gsx_gs_get_field(rotation)")) {
		return false;
	}
	if(!gsx_check(gsx_gs_get_field(state->gs, GSX_GS_FIELD_LOGSCALE, &gs_logscale), "gsx_gs_get_field(logscale)")) {
		return false;
	}
	if(!gsx_check(gsx_gs_get_field(state->gs, GSX_GS_FIELD_SH0, &gs_sh0), "gsx_gs_get_field(sh0)")) {
		return false;
	}
	if(!gsx_check(gsx_gs_get_field(state->gs, GSX_GS_FIELD_OPACITY, &gs_opacity), "gsx_gs_get_field(opacity)")) {
		return false;
	}

	intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
	intrinsics.width = options->width;
	intrinsics.height = options->height;
	intrinsics.fx = options->fx;
	intrinsics.fy = options->fy;
	intrinsics.cx = options->cx;
	intrinsics.cy = options->cy;
	intrinsics.camera_id = 0;

	pose.rot.x = options->qvec_wxyz[1];
	pose.rot.y = options->qvec_wxyz[2];
	pose.rot.z = options->qvec_wxyz[3];
	pose.rot.w = options->qvec_wxyz[0];
	pose.transl.x = options->tvec_xyz[0];
	pose.transl.y = options->tvec_xyz[1];
	pose.transl.z = options->tvec_xyz[2];
	pose.camera_id = 0;
	pose.frame_id = 0;

	request.intrinsics = &intrinsics;
	request.pose = &pose;
	request.near_plane = 0.01f;
	request.far_plane = 1000.0f;
	request.background_color.x = 0.0f;
	request.background_color.y = 0.0f;
	request.background_color.z = 0.0f;
	request.precision = GSX_RENDER_PRECISION_FLOAT32;
	request.sh_degree = 0;
	request.forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
	request.borrow_train_state = false;
	request.gs_mean3d = gs_mean3d;
	request.gs_rotation = gs_rotation;
	request.gs_logscale = gs_logscale;
	request.gs_sh0 = gs_sh0;
	request.gs_opacity = gs_opacity;
	request.out_rgb = state->out_rgb;

	if(!gsx_check(gsx_renderer_render(state->renderer, state->render_context, &request), "gsx_renderer_render")) {
		return false;
	}

	if(!gsx_check(gsx_tensor_get_info(state->out_rgb, &out_rgb_info), "gsx_tensor_get_info(out_rgb)")) {
		return false;
	}
	state->host_rgb = malloc((size_t)out_rgb_info.size_bytes);
	if(state->host_rgb == NULL) {
		fprintf(stderr, "error: host rgb allocation failed\n");
		return false;
	}
	state->host_rgb_size_bytes = out_rgb_info.size_bytes;

	if(!gsx_check(gsx_tensor_download(state->out_rgb, state->host_rgb, out_rgb_info.size_bytes), "gsx_tensor_download(out_rgb)")) {
		return false;
	}

	if(!gsx_check(gsx_backend_major_stream_sync(state->backend), "gsx_backend_major_stream_sync")) {
		return false;
	}

	if(!gsx_check(gsx_image_write_png(
			   options->output_image_path,
			   state->host_rgb,
			   options->width,
			   options->height,
			   3,
			   GSX_DATA_TYPE_F32,
			   GSX_STORAGE_FORMAT_CHW),
		   "gsx_image_write_png")) {
		return false;
	}

	return true;
}

int main(int argc, char **argv)
{
	app_options options;
	app_state state;
	bool ok = false;

	set_default_options(&options);
	if(!parse_args(argc, argv, &options)) {
		return 1;
	}

	if(!run_render(&options, &state)) {
		cleanup_state(&state);
		return 1;
	}

	printf("rendered image written: %s\n", options.output_image_path);
	printf("input ply: %s\n", options.input_ply_path);
	printf("resolution: %d x %d\n", options.width, options.height);
	printf("intrinsics: fx=%.6f fy=%.6f cx=%.6f cy=%.6f\n", options.fx, options.fy, options.cx, options.cy);
	printf(
		"extrinsics (qvec wxyz): %.9f, %.9f, %.9f, %.9f\n",
		options.qvec_wxyz[0],
		options.qvec_wxyz[1],
		options.qvec_wxyz[2],
		options.qvec_wxyz[3]);
	printf(
		"extrinsics (tvec xyz): %.9f, %.9f, %.9f\n",
		options.tvec_xyz[0],
		options.tvec_xyz[1],
		options.tvec_xyz[2]);

	ok = true;
	cleanup_state(&state);
	return ok ? 0 : 1;
}
