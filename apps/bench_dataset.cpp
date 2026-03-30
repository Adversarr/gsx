#include <gsx/extra/gsx-io-ply.h>
#include <gsx/extra/gsx-stbi.h>
#include <gsx/gsx.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

struct app_error : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

struct app_options {
	fs::path dataset_root;
	std::string split = "train";
	std::optional<fs::path> ply_override;

	gsx_backend_type backend_type =
#if defined(__APPLE__)
		GSX_BACKEND_TYPE_METAL;
#else
		GSX_BACKEND_TYPE_CPU;
#endif
	gsx_index_t device_index = 0;
	gsx_backend_buffer_type_class buffer_type_class = GSX_BACKEND_BUFFER_TYPE_DEVICE;

	bool enable_adc = true;
	gsx_adc_algorithm adc_algorithm = GSX_ADC_ALGORITHM_MCMC;
	gsx_float_t adc_pruning_opacity_threshold = 0.01f;
	gsx_float_t adc_opacity_clamp_value = 0.1f;
	gsx_float_t adc_duplicate_grad_threshold = 0.001f;
	gsx_float_t adc_duplicate_scale_threshold = 0.05f;
	gsx_float_t adc_noise_strength = 0.1f;
	gsx_float_t adc_grow_ratio = 0.05f;
	gsx_index_t adc_max_num_gaussians = 0;

	gsx_index_t max_input_width = 1600;
	gsx_index_t warmup_epochs = 1;
	gsx_index_t iter_epochs = 5;
	bool shuffle_each_epoch = false;
	gsx_size_t seed = 42;

	bool enable_async_prefetch = true;
	gsx_size_t prefetch_count = 2;

	gsx_render_precision render_precision = GSX_RENDER_PRECISION_FLOAT32;
	gsx_index_t sh_degree = 3;
	gsx_float_t near_plane = 0.01f;
	gsx_float_t far_plane = 100.0f;
	gsx_vec3 background_color{ 0.0f, 0.0f, 0.0f };

	gsx_float_t l1_scale = 0.8f;
	gsx_float_t ssim_scale = 0.2f;

	gsx_float_t lr_mean3d = 0.00016f;
	gsx_float_t lr_logscale = 0.005f;
	gsx_float_t lr_rotation = 0.001f;
	gsx_float_t lr_opacity = 0.05f;
	gsx_float_t lr_sh0 = 0.0025f;
	gsx_float_t lr_sh1 = 0.0005f;
	gsx_float_t lr_sh2 = 0.0005f;
	gsx_float_t lr_sh3 = 0.0005f;
	gsx_float_t beta1 = 0.9f;
	gsx_float_t beta2 = 0.999f;
	gsx_float_t epsilon = 1e-8f;
	gsx_float_t weight_decay = 0.0f;
	gsx_float_t max_grad = 1.0f;

	bool override_opacity = false;
	gsx_float_t init_opacity = 0.1f;

	std::string scheduler = "constant";
	gsx_float_t scheduler_initial_lr = 0.00016f;
	gsx_float_t scheduler_final_lr = 0.00016f;
	gsx_size_t scheduler_delay_steps = 0;
	gsx_float_t scheduler_delay_multiplier = 1.0f;
	gsx_size_t scheduler_decay_begin_step = 0;
	gsx_size_t scheduler_decay_end_step = 30000;
};

struct loaded_sample {
	std::string name;
	gsx_camera_intrinsics intrinsics{};
	gsx_camera_pose pose{};
	std::vector<float> rgb_hwc;
};

struct loaded_split {
	std::string name;
	gsx_camera_intrinsics intrinsics{};
	std::vector<loaded_sample> samples;
};

struct dataset_view {
	const loaded_split *split = nullptr;
};

struct timing_series {
	std::vector<double> dataloader_us;
	std::vector<double> render_forward_us;
	std::vector<double> loss_forward_us;
	std::vector<double> loss_backward_us;
	std::vector<double> render_backward_us;
	std::vector<double> optim_step_us;
	std::vector<double> adc_step_us;
	std::vector<double> total_step_us;
	std::vector<gsx_size_t> global_step_after;
};

struct distribution_stats {
	gsx_size_t count = 0;
	double sum = 0.0;
	double mean = 0.0;
	double stddev = 0.0;
	double min = 0.0;
	double p50 = 0.0;
	double p90 = 0.0;
	double p95 = 0.0;
	double p99 = 0.0;
	double max = 0.0;
};

struct run_summary {
	gsx_size_t steps = 0;
	gsx_size_t new_epoch_markers = 0;
	gsx_size_t new_permutation_markers = 0;
	gsx_size_t adc_result_count = 0;
	gsx_size_t first_global_step = 0;
	gsx_size_t last_global_step = 0;
	bool has_lr = false;
	gsx_float_t lr_min = 0.0f;
	gsx_float_t lr_max = 0.0f;
};

static void gsx_ok(const gsx_error error, const char *context)
{
	if(gsx_error_is_success(error)) {
		return;
	}

	std::ostringstream oss;
	oss << context << " failed (" << static_cast<int>(error.code) << ")";
	if(error.message != nullptr) {
		oss << ": " << error.message;
	}
	throw app_error(oss.str());
}

static std::string path_string(const fs::path &path)
{
	return path.string();
}

static bool nearly_equal(gsx_float_t a, gsx_float_t b, gsx_float_t epsilon = 1.0e-4f)
{
	return std::fabs(a - b) <= epsilon;
}

static gsx_float_t clamped_logit(gsx_float_t p)
{
	const gsx_float_t eps = 1.0e-6f;
	const gsx_float_t clamped = std::clamp(p, eps, 1.0f - eps);
	return gsx_logf(clamped / (1.0f - clamped));
}

static std::string backend_name(gsx_backend_type backend_type)
{
	switch(backend_type) {
	case GSX_BACKEND_TYPE_CPU:
		return "cpu";
	case GSX_BACKEND_TYPE_CUDA:
		return "cuda";
	case GSX_BACKEND_TYPE_METAL:
		return "metal";
	default:
		return "unknown";
	}
}

static std::string buffer_type_name(gsx_backend_buffer_type_class type)
{
	switch(type) {
	case GSX_BACKEND_BUFFER_TYPE_HOST:
		return "host";
	case GSX_BACKEND_BUFFER_TYPE_HOST_PINNED:
		return "host_pinned";
	case GSX_BACKEND_BUFFER_TYPE_DEVICE:
		return "device";
	case GSX_BACKEND_BUFFER_TYPE_UNIFIED:
		return "unified";
	default:
		return "unknown";
	}
}

static std::string adc_algorithm_name(gsx_adc_algorithm algorithm)
{
	switch(algorithm) {
	case GSX_ADC_ALGORITHM_DEFAULT:
		return "default";
	case GSX_ADC_ALGORITHM_ABSGS:
		return "absgs";
	case GSX_ADC_ALGORITHM_MCMC:
		return "mcmc";
	case GSX_ADC_ALGORITHM_FASTGS:
		return "fastgs";
	default:
		return "unknown";
	}
}

static bool parse_i64(const char *value, long long *out_value)
{
	if(value == nullptr || out_value == nullptr) {
		return false;
	}
	char *end_ptr = nullptr;
	const long long parsed = std::strtoll(value, &end_ptr, 10);
	if(end_ptr == value || *end_ptr != '\0') {
		return false;
	}
	*out_value = parsed;
	return true;
}

static bool parse_u64(const char *value, unsigned long long *out_value)
{
	if(value == nullptr || out_value == nullptr) {
		return false;
	}
	char *end_ptr = nullptr;
	const unsigned long long parsed = std::strtoull(value, &end_ptr, 10);
	if(end_ptr == value || *end_ptr != '\0') {
		return false;
	}
	*out_value = parsed;
	return true;
}

static bool parse_f32(const char *value, float *out_value)
{
	if(value == nullptr || out_value == nullptr) {
		return false;
	}
	char *end_ptr = nullptr;
	const float parsed = std::strtof(value, &end_ptr);
	if(end_ptr == value || *end_ptr != '\0') {
		return false;
	}
	*out_value = parsed;
	return true;
}

static bool parse_bool(const char *value, bool *out)
{
	if(value == nullptr || out == nullptr) {
		return false;
	}
	if(std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 || std::strcmp(value, "on") == 0
		|| std::strcmp(value, "yes") == 0) {
		*out = true;
		return true;
	}
	if(std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 || std::strcmp(value, "off") == 0
		|| std::strcmp(value, "no") == 0) {
		*out = false;
		return true;
	}
	return false;
}

static bool parse_backend_type(const char *value, gsx_backend_type *out_backend)
{
	if(value == nullptr || out_backend == nullptr) {
		return false;
	}
	if(std::strcmp(value, "cpu") == 0) {
		*out_backend = GSX_BACKEND_TYPE_CPU;
		return true;
	}
	if(std::strcmp(value, "cuda") == 0) {
		*out_backend = GSX_BACKEND_TYPE_CUDA;
		return true;
	}
	if(std::strcmp(value, "metal") == 0) {
		*out_backend = GSX_BACKEND_TYPE_METAL;
		return true;
	}
	return false;
}

static bool parse_buffer_type_class(const char *value, gsx_backend_buffer_type_class *out_type)
{
	if(value == nullptr || out_type == nullptr) {
		return false;
	}
	if(std::strcmp(value, "host") == 0) {
		*out_type = GSX_BACKEND_BUFFER_TYPE_HOST;
		return true;
	}
	if(std::strcmp(value, "host_pinned") == 0) {
		*out_type = GSX_BACKEND_BUFFER_TYPE_HOST_PINNED;
		return true;
	}
	if(std::strcmp(value, "device") == 0) {
		*out_type = GSX_BACKEND_BUFFER_TYPE_DEVICE;
		return true;
	}
	if(std::strcmp(value, "unified") == 0) {
		*out_type = GSX_BACKEND_BUFFER_TYPE_UNIFIED;
		return true;
	}
	return false;
}

static bool parse_adc_algorithm(const char *value, gsx_adc_algorithm *out_algorithm)
{
	if(value == nullptr || out_algorithm == nullptr) {
		return false;
	}
	if(std::strcmp(value, "default") == 0) {
		*out_algorithm = GSX_ADC_ALGORITHM_DEFAULT;
		return true;
	}
	if(std::strcmp(value, "absgs") == 0) {
		*out_algorithm = GSX_ADC_ALGORITHM_ABSGS;
		return true;
	}
	if(std::strcmp(value, "mcmc") == 0) {
		*out_algorithm = GSX_ADC_ALGORITHM_MCMC;
		return true;
	}
	if(std::strcmp(value, "fastgs") == 0) {
		*out_algorithm = GSX_ADC_ALGORITHM_FASTGS;
		return true;
	}
	return false;
}

static void print_usage(const char *program)
{
	std::cout << "usage: " << program << " --dataset-root <path> [options]\n";
	std::cout << "\n";
	std::cout << "required:\n";
	std::cout << "  --dataset-root <path>               Dataset root containing <split>/cameras.json, <split>/poses.json, and <split>/images/.\n";
	std::cout << "\n";
	std::cout << "core options:\n";
	std::cout << "  --split <train|val>                Dataset split to benchmark (default: train).\n";
	std::cout << "  --ply <path>                       Override PLY path for pretrained/checkpoint GS (default: <dataset-root>/train/points3d.ply).\n";
	std::cout << "  --backend <cpu|cuda|metal>         Backend type.\n";
	std::cout << "  --device <index>                   Device index for selected backend (default: 0).\n";
	std::cout << "  --buffer-type <host|host_pinned|device|unified>  Session workspace/arena buffer type class.\n";
	std::cout << "  --enable-adc <bool>                 Include ADC in the benchmark (default: true).\n";
	std::cout << "  --adc-algorithm <default|absgs|mcmc|fastgs>  ADC policy (default: mcmc).\n";
	std::cout << "  --adc-prune-threshold <float>       ADC pruning opacity threshold (default: 0.01).\n";
	std::cout << "  --adc-grow-ratio <float>            ADC target growth ratio (default: 0.05).\n";
	std::cout << "  --adc-noise-strength <float>        ADC MCMC noise strength (default: 0.1).\n";
	std::cout << "  --adc-max-gaussians <int>           ADC max gaussian cap; 0 disables the hard cap.\n";
	std::cout << "  --max-input-width <int>            Width cap for decoded images. <=0 disables resizing.\n";
	std::cout << "  --warmup <epochs>                  Warmup epochs (default: 1).\n";
	std::cout << "  --iter <epochs>                    Measured epochs (default: 5).\n";
	std::cout << "  --shuffle-each-epoch <bool>        Shuffle sample order each epoch (default: false).\n";
	std::cout << "  --seed <u64>                       Dataloader seed (default: 42).\n";
	std::cout << "  --enable-async-prefetch <bool>     Enable dataloader async prefetch (default: true).\n";
	std::cout << "  --prefetch-count <u64>             Async prefetch queue depth (default: 2).\n";
	std::cout << "\n";
	std::cout << "render/session options:\n";
	std::cout << "  --sh-degree <0..3>                 SH degree used during session rendering (default: 3).\n";
	std::cout << "  --near <float>                     Near plane (default: 0.01).\n";
	std::cout << "  --far <float>                      Far plane (default: 100.0).\n";
	std::cout << "  --bg-r <float> --bg-g <float> --bg-b <float>  Background color (default: 0,0,0).\n";
	std::cout << "  --l1-scale <float>                 L1 loss scale (default: 0.8).\n";
	std::cout << "  --ssim-scale <float>               SSIM loss scale (default: 0.2).\n";
	std::cout << "\n";
	std::cout << "optimizer/scheduler options:\n";
	std::cout << "  --lr-mean3d <float>                Mean3D learning rate (default: 0.00016).\n";
	std::cout << "  --lr-logscale <float>              Logscale learning rate (default: 0.005).\n";
	std::cout << "  --lr-rotation <float>              Rotation learning rate (default: 0.001).\n";
	std::cout << "  --lr-opacity <float>               Opacity learning rate (default: 0.05).\n";
	std::cout << "  --lr-sh0 <float>                   SH0 learning rate (default: 0.0025).\n";
	std::cout << "  --lr-sh1 <float>                   SH1 learning rate (default: 0.0005).\n";
	std::cout << "  --lr-sh2 <float>                   SH2 learning rate (default: 0.0005).\n";
	std::cout << "  --lr-sh3 <float>                   SH3 learning rate (default: 0.0005).\n";
	std::cout << "  --beta1 <float> --beta2 <float> --epsilon <float> --weight-decay <float> --max-grad <float>\n";
	std::cout << "  --scheduler <none|constant|delayed_exponential> (default: constant).\n";
	std::cout << "  --scheduler-initial-lr <float> --scheduler-final-lr <float>\n";
	std::cout << "  --scheduler-delay-steps <u64> --scheduler-delay-multiplier <float>\n";
	std::cout << "  --scheduler-decay-begin-step <u64> --scheduler-decay-end-step <u64>\n";
	std::cout << "\n";
	std::cout << "optional GS override:\n";
	std::cout << "  --override-opacity <bool>          Force all opacities to init-opacity (default: false).\n";
	std::cout << "  --init-opacity <float>             Opacity value when override-opacity is true (default: 0.1).\n";
}

static const char *require_value(int argc, char **argv, int *io_index)
{
	const int i = *io_index;
	if(i + 1 >= argc) {
		throw app_error(std::string("missing value for argument: ") + argv[i]);
	}
	*io_index = i + 1;
	return argv[*io_index];
}

static app_options parse_options(int argc, char **argv)
{
	app_options options;

	for(int i = 1; i < argc; ++i) {
		const char *arg = argv[i];
		if(std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
			print_usage(argv[0]);
			std::exit(EXIT_SUCCESS);
		}
		if(std::strcmp(arg, "--dataset-root") == 0) {
			options.dataset_root = require_value(argc, argv, &i);
			continue;
		}
		if(std::strcmp(arg, "--split") == 0) {
			options.split = require_value(argc, argv, &i);
			continue;
		}
		if(std::strcmp(arg, "--ply") == 0) {
			options.ply_override = fs::path(require_value(argc, argv, &i));
			continue;
		}
		if(std::strcmp(arg, "--backend") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_backend_type(value, &options.backend_type)) {
				throw app_error("invalid --backend value (expected cpu|cuda|metal)");
			}
			continue;
		}
		if(std::strcmp(arg, "--device") == 0) {
			long long parsed = 0;
			const char *value = require_value(argc, argv, &i);
			if(!parse_i64(value, &parsed) || parsed < 0) {
				throw app_error("invalid --device value");
			}
			options.device_index = static_cast<gsx_index_t>(parsed);
			continue;
		}
		if(std::strcmp(arg, "--buffer-type") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_buffer_type_class(value, &options.buffer_type_class)) {
				throw app_error("invalid --buffer-type value");
			}
			continue;
		}
		if(std::strcmp(arg, "--enable-adc") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_bool(value, &options.enable_adc)) {
				throw app_error("invalid --enable-adc value");
			}
			continue;
		}
		if(std::strcmp(arg, "--adc-algorithm") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_adc_algorithm(value, &options.adc_algorithm)) {
				throw app_error("invalid --adc-algorithm value");
			}
			continue;
		}
		if(std::strcmp(arg, "--adc-prune-threshold") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.adc_pruning_opacity_threshold)) {
				throw app_error("invalid --adc-prune-threshold value");
			}
			continue;
		}
		if(std::strcmp(arg, "--adc-grow-ratio") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.adc_grow_ratio)) {
				throw app_error("invalid --adc-grow-ratio value");
			}
			continue;
		}
		if(std::strcmp(arg, "--adc-noise-strength") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.adc_noise_strength)) {
				throw app_error("invalid --adc-noise-strength value");
			}
			continue;
		}
		if(std::strcmp(arg, "--adc-max-gaussians") == 0) {
			long long parsed = 0;
			const char *value = require_value(argc, argv, &i);
			if(!parse_i64(value, &parsed) || parsed < 0) {
				throw app_error("invalid --adc-max-gaussians value");
			}
			options.adc_max_num_gaussians = static_cast<gsx_index_t>(parsed);
			continue;
		}
		if(std::strcmp(arg, "--max-input-width") == 0) {
			long long parsed = 0;
			const char *value = require_value(argc, argv, &i);
			if(!parse_i64(value, &parsed)) {
				throw app_error("invalid --max-input-width value");
			}
			options.max_input_width = static_cast<gsx_index_t>(parsed);
			continue;
		}
		if(std::strcmp(arg, "--warmup") == 0) {
			long long parsed = 0;
			const char *value = require_value(argc, argv, &i);
			if(!parse_i64(value, &parsed) || parsed < 0) {
				throw app_error("invalid --warmup value");
			}
			options.warmup_epochs = static_cast<gsx_index_t>(parsed);
			continue;
		}
		if(std::strcmp(arg, "--iter") == 0) {
			long long parsed = 0;
			const char *value = require_value(argc, argv, &i);
			if(!parse_i64(value, &parsed) || parsed <= 0) {
				throw app_error("invalid --iter value");
			}
			options.iter_epochs = static_cast<gsx_index_t>(parsed);
			continue;
		}
		if(std::strcmp(arg, "--shuffle-each-epoch") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_bool(value, &options.shuffle_each_epoch)) {
				throw app_error("invalid --shuffle-each-epoch value");
			}
			continue;
		}
		if(std::strcmp(arg, "--seed") == 0) {
			unsigned long long parsed = 0;
			const char *value = require_value(argc, argv, &i);
			if(!parse_u64(value, &parsed)) {
				throw app_error("invalid --seed value");
			}
			options.seed = static_cast<gsx_size_t>(parsed);
			continue;
		}
		if(std::strcmp(arg, "--enable-async-prefetch") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_bool(value, &options.enable_async_prefetch)) {
				throw app_error("invalid --enable-async-prefetch value");
			}
			continue;
		}
		if(std::strcmp(arg, "--prefetch-count") == 0) {
			unsigned long long parsed = 0;
			const char *value = require_value(argc, argv, &i);
			if(!parse_u64(value, &parsed)) {
				throw app_error("invalid --prefetch-count value");
			}
			options.prefetch_count = static_cast<gsx_size_t>(parsed);
			continue;
		}
		if(std::strcmp(arg, "--sh-degree") == 0) {
			long long parsed = 0;
			const char *value = require_value(argc, argv, &i);
			if(!parse_i64(value, &parsed)) {
				throw app_error("invalid --sh-degree value");
			}
			options.sh_degree = static_cast<gsx_index_t>(parsed);
			continue;
		}
		if(std::strcmp(arg, "--near") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.near_plane)) {
				throw app_error("invalid --near value");
			}
			continue;
		}
		if(std::strcmp(arg, "--far") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.far_plane)) {
				throw app_error("invalid --far value");
			}
			continue;
		}
		if(std::strcmp(arg, "--bg-r") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.background_color.x)) {
				throw app_error("invalid --bg-r value");
			}
			continue;
		}
		if(std::strcmp(arg, "--bg-g") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.background_color.y)) {
				throw app_error("invalid --bg-g value");
			}
			continue;
		}
		if(std::strcmp(arg, "--bg-b") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.background_color.z)) {
				throw app_error("invalid --bg-b value");
			}
			continue;
		}
		if(std::strcmp(arg, "--l1-scale") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.l1_scale)) {
				throw app_error("invalid --l1-scale value");
			}
			continue;
		}
		if(std::strcmp(arg, "--ssim-scale") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.ssim_scale)) {
				throw app_error("invalid --ssim-scale value");
			}
			continue;
		}
		if(std::strcmp(arg, "--lr-mean3d") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.lr_mean3d)) {
				throw app_error("invalid --lr-mean3d value");
			}
			continue;
		}
		if(std::strcmp(arg, "--lr-logscale") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.lr_logscale)) {
				throw app_error("invalid --lr-logscale value");
			}
			continue;
		}
		if(std::strcmp(arg, "--lr-rotation") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.lr_rotation)) {
				throw app_error("invalid --lr-rotation value");
			}
			continue;
		}
		if(std::strcmp(arg, "--lr-opacity") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.lr_opacity)) {
				throw app_error("invalid --lr-opacity value");
			}
			continue;
		}
		if(std::strcmp(arg, "--lr-sh0") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.lr_sh0)) {
				throw app_error("invalid --lr-sh0 value");
			}
			continue;
		}
		if(std::strcmp(arg, "--lr-sh1") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.lr_sh1)) {
				throw app_error("invalid --lr-sh1 value");
			}
			continue;
		}
		if(std::strcmp(arg, "--lr-sh2") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.lr_sh2)) {
				throw app_error("invalid --lr-sh2 value");
			}
			continue;
		}
		if(std::strcmp(arg, "--lr-sh3") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.lr_sh3)) {
				throw app_error("invalid --lr-sh3 value");
			}
			continue;
		}
		if(std::strcmp(arg, "--beta1") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.beta1)) {
				throw app_error("invalid --beta1 value");
			}
			continue;
		}
		if(std::strcmp(arg, "--beta2") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.beta2)) {
				throw app_error("invalid --beta2 value");
			}
			continue;
		}
		if(std::strcmp(arg, "--epsilon") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.epsilon)) {
				throw app_error("invalid --epsilon value");
			}
			continue;
		}
		if(std::strcmp(arg, "--weight-decay") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.weight_decay)) {
				throw app_error("invalid --weight-decay value");
			}
			continue;
		}
		if(std::strcmp(arg, "--max-grad") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.max_grad)) {
				throw app_error("invalid --max-grad value");
			}
			continue;
		}
		if(std::strcmp(arg, "--scheduler") == 0) {
			options.scheduler = require_value(argc, argv, &i);
			continue;
		}
		if(std::strcmp(arg, "--scheduler-initial-lr") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.scheduler_initial_lr)) {
				throw app_error("invalid --scheduler-initial-lr value");
			}
			continue;
		}
		if(std::strcmp(arg, "--scheduler-final-lr") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.scheduler_final_lr)) {
				throw app_error("invalid --scheduler-final-lr value");
			}
			continue;
		}
		if(std::strcmp(arg, "--scheduler-delay-steps") == 0) {
			unsigned long long parsed = 0;
			const char *value = require_value(argc, argv, &i);
			if(!parse_u64(value, &parsed)) {
				throw app_error("invalid --scheduler-delay-steps value");
			}
			options.scheduler_delay_steps = static_cast<gsx_size_t>(parsed);
			continue;
		}
		if(std::strcmp(arg, "--scheduler-delay-multiplier") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.scheduler_delay_multiplier)) {
				throw app_error("invalid --scheduler-delay-multiplier value");
			}
			continue;
		}
		if(std::strcmp(arg, "--scheduler-decay-begin-step") == 0) {
			unsigned long long parsed = 0;
			const char *value = require_value(argc, argv, &i);
			if(!parse_u64(value, &parsed)) {
				throw app_error("invalid --scheduler-decay-begin-step value");
			}
			options.scheduler_decay_begin_step = static_cast<gsx_size_t>(parsed);
			continue;
		}
		if(std::strcmp(arg, "--scheduler-decay-end-step") == 0) {
			unsigned long long parsed = 0;
			const char *value = require_value(argc, argv, &i);
			if(!parse_u64(value, &parsed)) {
				throw app_error("invalid --scheduler-decay-end-step value");
			}
			options.scheduler_decay_end_step = static_cast<gsx_size_t>(parsed);
			continue;
		}
		if(std::strcmp(arg, "--override-opacity") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_bool(value, &options.override_opacity)) {
				throw app_error("invalid --override-opacity value");
			}
			continue;
		}
		if(std::strcmp(arg, "--init-opacity") == 0) {
			const char *value = require_value(argc, argv, &i);
			if(!parse_f32(value, &options.init_opacity)) {
				throw app_error("invalid --init-opacity value");
			}
			continue;
		}

		throw app_error(std::string("unknown argument: ") + arg);
	}

	if(options.dataset_root.empty()) {
		throw app_error("--dataset-root is required");
	}
	if(options.split != "train" && options.split != "val") {
		throw app_error("--split must be train or val");
	}
	if(options.max_input_width < 0) {
		throw app_error("--max-input-width must be >= 0");
	}
	if(options.warmup_epochs < 0) {
		throw app_error("--warmup must be >= 0");
	}
	if(options.iter_epochs <= 0) {
		throw app_error("--iter must be > 0");
	}
	if(options.prefetch_count == 0 && options.enable_async_prefetch) {
		throw app_error("--prefetch-count must be > 0 when async prefetch is enabled");
	}
	if(options.sh_degree < 0 || options.sh_degree > 3) {
		throw app_error("--sh-degree must be in [0, 3]");
	}
	if(options.near_plane <= 0.0f || options.far_plane <= options.near_plane) {
		throw app_error("invalid near/far values");
	}
	if(options.l1_scale == 0.0f && options.ssim_scale == 0.0f) {
		throw app_error("at least one of --l1-scale and --ssim-scale must be non-zero");
	}
	if(options.scheduler != "none" && options.scheduler != "constant" && options.scheduler != "delayed_exponential") {
		throw app_error("--scheduler must be one of: none, constant, delayed_exponential");
	}
	return options;
}

static json read_json_file(const fs::path &path)
{
	std::ifstream input(path);
	if(!input) {
		throw app_error("failed to open json file: " + path_string(path));
	}
	json value;
	input >> value;
	return value;
}

static gsx_quat quat_wxyz_to_xyzw(const std::array<gsx_float_t, 4> &q)
{
	gsx_quat out{};
	out.x = q[1];
	out.y = q[2];
	out.z = q[3];
	out.w = q[0];
	return out;
}

static gsx_camera_intrinsics parse_camera_json(const json &camera_json)
{
	if(!camera_json.is_object()) {
		throw app_error("camera entry must be an object");
	}
	if(camera_json.value("model", "") != "PINHOLE") {
		throw app_error("only PINHOLE camera model is supported");
	}
	const auto params = camera_json.at("params");
	if(!params.is_array() || params.size() != 4) {
		throw app_error("camera params must be [fx, fy, cx, cy]");
	}

	gsx_camera_intrinsics intrinsics{};
	intrinsics.model = GSX_CAMERA_MODEL_PINHOLE;
	intrinsics.fx = params.at(0).get<gsx_float_t>();
	intrinsics.fy = params.at(1).get<gsx_float_t>();
	intrinsics.cx = params.at(2).get<gsx_float_t>();
	intrinsics.cy = params.at(3).get<gsx_float_t>();
	intrinsics.camera_id = camera_json.at("camera_id").get<gsx_index_t>();
	intrinsics.width = camera_json.at("width").get<gsx_index_t>();
	intrinsics.height = camera_json.at("height").get<gsx_index_t>();
	return intrinsics;
}

static gsx_camera_intrinsics apply_width_cap(gsx_camera_intrinsics intrinsics, gsx_index_t max_input_width)
{
	if(max_input_width <= 0 || intrinsics.width <= max_input_width) {
		return intrinsics;
	}

	const gsx_float_t scale = static_cast<gsx_float_t>(max_input_width) / static_cast<gsx_float_t>(intrinsics.width);
	const gsx_index_t scaled_height = std::max<gsx_index_t>(1, static_cast<gsx_index_t>(std::lround(static_cast<double>(intrinsics.height) * scale)));

	intrinsics.fx *= scale;
	intrinsics.fy *= scale;
	intrinsics.cx *= scale;
	intrinsics.cy *= scale;
	intrinsics.width = max_input_width;
	intrinsics.height = scaled_height;
	return intrinsics;
}

static void validate_matching_intrinsics(const gsx_camera_intrinsics &a, const gsx_camera_intrinsics &b)
{
	if(a.model != b.model || a.width != b.width || a.height != b.height || !nearly_equal(a.fx, b.fx) || !nearly_equal(a.fy, b.fy)
		|| !nearly_equal(a.cx, b.cx) || !nearly_equal(a.cy, b.cy)) {
		throw app_error("split intrinsics mismatch after width cap");
	}
}

static loaded_sample load_sample(
	const fs::path &image_path,
	const std::string &name,
	const gsx_camera_intrinsics &scaled_intrinsics,
	gsx_index_t image_id,
	const std::array<gsx_float_t, 4> &qvec_wxyz,
	const std::array<gsx_float_t, 3> &tvec,
	gsx_index_t camera_id)
{
	gsx_image image{};
	gsx_image resized{};
	loaded_sample sample;

	gsx_ok(gsx_image_load(&image, path_string(image_path).c_str(), 3, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_HWC), "gsx_image_load");

	try {
		sample.name = name;
		sample.intrinsics = scaled_intrinsics;
		if(image.width != scaled_intrinsics.width || image.height != scaled_intrinsics.height) {
			gsx_ok(gsx_image_resize(&resized, &image, scaled_intrinsics.width, scaled_intrinsics.height), "gsx_image_resize");
		}

		const gsx_image *final_image = (resized.pixels != nullptr) ? &resized : &image;
		if(final_image->width != scaled_intrinsics.width || final_image->height != scaled_intrinsics.height || final_image->channels != 3) {
			throw app_error("decoded image geometry does not match scaled intrinsics for " + image_path.filename().string());
		}

		sample.pose.rot = quat_wxyz_to_xyzw(qvec_wxyz);
		sample.pose.transl.x = tvec[0];
		sample.pose.transl.y = tvec[1];
		sample.pose.transl.z = tvec[2];
		sample.pose.camera_id = camera_id;
		sample.pose.frame_id = image_id;

		const gsx_size_t pixel_count = static_cast<gsx_size_t>(scaled_intrinsics.width) * static_cast<gsx_size_t>(scaled_intrinsics.height) * 3u;
		sample.rgb_hwc.resize(static_cast<size_t>(pixel_count));
		std::copy_n(static_cast<const float *>(final_image->pixels), static_cast<size_t>(pixel_count), sample.rgb_hwc.begin());
	} catch(...) {
		(void)gsx_image_free(&resized);
		(void)gsx_image_free(&image);
		throw;
	}

	gsx_ok(gsx_image_free(&resized), "gsx_image_free");
	gsx_ok(gsx_image_free(&image), "gsx_image_free");
	return sample;
}

static loaded_split load_split(const fs::path &dataset_root, const std::string &split_name, gsx_index_t max_input_width)
{
	const fs::path split_root = dataset_root / split_name;
	const json cameras_json = read_json_file(split_root / "cameras.json");
	const json poses_json = read_json_file(split_root / "poses.json");

	loaded_split split;
	if(!cameras_json.is_array() || cameras_json.size() != 1) {
		throw app_error(split_name + ": expected exactly one camera in cameras.json");
	}
	if(!poses_json.is_array() || poses_json.empty()) {
		throw app_error(split_name + ": poses.json must be a non-empty array");
	}

	split.name = split_name;
	split.intrinsics = apply_width_cap(parse_camera_json(cameras_json.at(0)), max_input_width);

	split.samples.reserve(poses_json.size());
	for(const json &pose_json : poses_json) {
		const std::string name = pose_json.at("name").get<std::string>();
		const fs::path image_path = split_root / "images" / name;
		const auto qvec_json = pose_json.at("qvec");
		const auto tvec_json = pose_json.at("tvec");
		std::array<gsx_float_t, 4> qvec{};
		std::array<gsx_float_t, 3> tvec{};

		if(!fs::exists(image_path)) {
			throw app_error("missing image file: " + path_string(image_path));
		}
		if(!qvec_json.is_array() || qvec_json.size() != 4 || !tvec_json.is_array() || tvec_json.size() != 3) {
			throw app_error("invalid qvec/tvec in " + path_string(split_root / "poses.json"));
		}
		for(size_t i = 0; i < 4; ++i) {
			qvec[i] = qvec_json.at(i).get<gsx_float_t>();
		}
		for(size_t i = 0; i < 3; ++i) {
			tvec[i] = tvec_json.at(i).get<gsx_float_t>();
		}

		split.samples.push_back(load_sample(
			image_path,
			name,
			split.intrinsics,
			pose_json.at("image_id").get<gsx_index_t>(),
			qvec,
			tvec,
			pose_json.at("camera_id").get<gsx_index_t>()));
	}

	return split;
}

static gsx_error dataset_get_length(void *object, gsx_size_t *out_length)
{
	const dataset_view *dataset = static_cast<const dataset_view *>(object);
	if(dataset == nullptr || dataset->split == nullptr || out_length == nullptr) {
		return gsx_error{ GSX_ERROR_INVALID_ARGUMENT, "dataset and out_length must be non-null" };
	}
	*out_length = static_cast<gsx_size_t>(dataset->split->samples.size());
	return gsx_error{ GSX_ERROR_SUCCESS, nullptr };
}

static gsx_error dataset_get_sample(void *object, gsx_size_t sample_index, gsx_dataset_cpu_sample *out_sample)
{
	const dataset_view *dataset = static_cast<const dataset_view *>(object);
	if(dataset == nullptr || dataset->split == nullptr || out_sample == nullptr) {
		return gsx_error{ GSX_ERROR_INVALID_ARGUMENT, "dataset and out_sample must be non-null" };
	}
	if(sample_index >= dataset->split->samples.size()) {
		return gsx_error{ GSX_ERROR_OUT_OF_RANGE, "sample_index out of range" };
	}

	const loaded_sample &sample = dataset->split->samples[static_cast<size_t>(sample_index)];
	*out_sample = {};
	out_sample->intrinsics = sample.intrinsics;
	out_sample->pose = sample.pose;
	out_sample->rgb_data = sample.rgb_hwc.data();
	out_sample->stable_sample_id = static_cast<gsx_id_t>(sample.pose.frame_id);
	out_sample->has_stable_sample_id = true;
	out_sample->release_token = nullptr;
	return gsx_error{ GSX_ERROR_SUCCESS, nullptr };
}

static void dataset_release_sample(void *object, gsx_dataset_cpu_sample *sample)
{
	(void)object;
	(void)sample;
}

template <typename Handle, typename FreeFn>
struct handle_owner {
	Handle handle = nullptr;
	FreeFn free_fn;

	explicit handle_owner(FreeFn fn) : free_fn(fn) {}
	~handle_owner() { reset(); }

	handle_owner(const handle_owner &) = delete;
	handle_owner &operator=(const handle_owner &) = delete;

	handle_owner(handle_owner &&other) noexcept : handle(other.handle), free_fn(other.free_fn)
	{
		other.handle = nullptr;
	}

	handle_owner &operator=(handle_owner &&other) noexcept
	{
		if(this == &other) {
			return *this;
		}
		reset();
		handle = other.handle;
		free_fn = other.free_fn;
		other.handle = nullptr;
		return *this;
	}

	void reset()
	{
		if(handle != nullptr) {
			(void)free_fn(handle);
			handle = nullptr;
		}
	}
};

using backend_owner = handle_owner<gsx_backend_t, decltype(&gsx_backend_free)>;
using dataset_owner = handle_owner<gsx_dataset_t, decltype(&gsx_dataset_free)>;
using dataloader_owner = handle_owner<gsx_dataloader_t, decltype(&gsx_dataloader_free)>;
using renderer_owner = handle_owner<gsx_renderer_t, decltype(&gsx_renderer_free)>;
using gs_owner = handle_owner<gsx_gs_t, decltype(&gsx_gs_free)>;
using loss_owner = handle_owner<gsx_loss_t, decltype(&gsx_loss_free)>;
using loss_context_owner = handle_owner<gsx_loss_context_t, decltype(&gsx_loss_context_free)>;
using optim_owner = handle_owner<gsx_optim_t, decltype(&gsx_optim_free)>;
using scheduler_owner = handle_owner<gsx_scheduler_t, decltype(&gsx_scheduler_free)>;
using adc_owner = handle_owner<gsx_adc_t, decltype(&gsx_adc_free)>;
using session_owner = handle_owner<gsx_session_t, decltype(&gsx_session_free)>;

static gsx_backend_t create_backend(const app_options &options)
{
	gsx_ok(gsx_backend_registry_init(), "gsx_backend_registry_init");

	gsx_index_t visible_device_count = 0;
	gsx_ok(gsx_count_backend_devices_by_type(options.backend_type, &visible_device_count), "gsx_count_backend_devices_by_type");
	if(visible_device_count <= 0) {
		throw app_error("no visible devices for backend type: " + backend_name(options.backend_type));
	}
	if(options.device_index < 0 || options.device_index >= visible_device_count) {
		std::ostringstream oss;
		oss << "device index out of range [0, " << visible_device_count << "): " << options.device_index;
		throw app_error(oss.str());
	}

	gsx_backend_device_t device = nullptr;
	gsx_ok(
		gsx_get_backend_device_by_type(options.backend_type, options.device_index, &device),
		"gsx_get_backend_device_by_type");

	gsx_backend_desc desc{};
	desc.device = device;

	gsx_backend_t backend = nullptr;
	gsx_ok(gsx_backend_init(&backend, &desc), "gsx_backend_init");
	return backend;
}

static gsx_backend_buffer_type_t find_buffer_type(gsx_backend_t backend, gsx_backend_buffer_type_class type_class)
{
	gsx_backend_buffer_type_t buffer_type = nullptr;
	gsx_ok(gsx_backend_find_buffer_type(backend, type_class, &buffer_type), "gsx_backend_find_buffer_type");
	return buffer_type;
}

static gsx_dataset_t create_dataset(dataset_view *view, const std::string &name)
{
	gsx_dataset_desc desc{};
	desc.name = name.c_str();
	desc.object = view;
	desc.image_data_type = GSX_DATA_TYPE_F32;
	desc.width = view->split->intrinsics.width;
	desc.height = view->split->intrinsics.height;
	desc.has_rgb = true;
	desc.has_alpha = false;
	desc.has_invdepth = false;
	desc.get_length = dataset_get_length;
	desc.get_sample = dataset_get_sample;
	desc.release_sample = dataset_release_sample;

	gsx_dataset_t dataset = nullptr;
	gsx_ok(gsx_dataset_init(&dataset, &desc), "gsx_dataset_init");
	return dataset;
}

static gsx_dataloader_t create_dataloader(
	gsx_backend_t backend,
	gsx_dataset_t dataset,
	bool shuffle_each_epoch,
	gsx_size_t seed,
	bool enable_async_prefetch,
	gsx_size_t prefetch_count)
{
	gsx_dataloader_t dataloader = nullptr;
	gsx_dataloader_desc desc{};
	desc.shuffle_each_epoch = shuffle_each_epoch;
	desc.enable_async_prefetch = enable_async_prefetch;
	desc.prefetch_count = enable_async_prefetch ? prefetch_count : 0;
	desc.seed = seed;
	desc.image_data_type = GSX_DATA_TYPE_F32;
	gsx_ok(gsx_dataloader_init(&dataloader, backend, dataset, &desc), "gsx_dataloader_init");
	return dataloader;
}

static gsx_renderer_t create_renderer(gsx_backend_t backend, const loaded_split &split)
{
	gsx_renderer_desc desc{};
	desc.width = split.intrinsics.width;
	desc.height = split.intrinsics.height;
	desc.output_data_type = GSX_DATA_TYPE_F32;

	gsx_renderer_t renderer = nullptr;
	gsx_ok(gsx_renderer_init(&renderer, backend, &desc), "gsx_renderer_init");
	return renderer;
}

static gsx_loss_t create_loss(gsx_backend_t backend, gsx_loss_algorithm algorithm)
{
	gsx_loss_t loss = nullptr;
	gsx_loss_desc desc{};
	desc.algorithm = algorithm;
	desc.grad_normalization = GSX_LOSS_GRAD_NORMALIZATION_TYPE_MEAN;
	gsx_ok(gsx_loss_init(&loss, backend, &desc), "gsx_loss_init");
	return loss;
}

static gsx_loss_context_t create_loss_context(gsx_loss_t loss)
{
	gsx_loss_context_t context = nullptr;
	gsx_ok(gsx_loss_context_init(&context, loss), "gsx_loss_context_init");
	return context;
}

static void add_param_group(
	std::vector<gsx_optim_param_group_desc> &groups,
	gsx_gs_t gs,
	gsx_gs_field parameter_field,
	gsx_gs_field gradient_field,
	gsx_optim_param_role role,
	gsx_float_t learning_rate,
	const app_options &options)
{
	gsx_tensor_t parameter = nullptr;
	gsx_tensor_t gradient = nullptr;
	gsx_ok(gsx_gs_get_field(gs, parameter_field, &parameter), "gsx_gs_get_field(parameter)");
	gsx_ok(gsx_gs_get_field(gs, gradient_field, &gradient), "gsx_gs_get_field(gradient)");

	gsx_optim_param_group_desc group{};
	group.role = role;
	group.parameter = parameter;
	group.gradient = gradient;
	group.learning_rate = learning_rate;
	group.beta1 = options.beta1;
	group.beta2 = options.beta2;
	group.weight_decay = options.weight_decay;
	group.epsilon = options.epsilon;
	group.max_grad = options.max_grad;
	groups.push_back(group);
}

static gsx_optim_t create_optimizer(gsx_backend_t backend, gsx_gs_t gs, const app_options &options)
{
	std::vector<gsx_optim_param_group_desc> groups;
	groups.reserve(8);
	add_param_group(groups, gs, GSX_GS_FIELD_MEAN3D, GSX_GS_FIELD_GRAD_MEAN3D, GSX_OPTIM_PARAM_ROLE_MEAN3D, options.lr_mean3d, options);
	add_param_group(groups, gs, GSX_GS_FIELD_LOGSCALE, GSX_GS_FIELD_GRAD_LOGSCALE, GSX_OPTIM_PARAM_ROLE_LOGSCALE, options.lr_logscale, options);
	add_param_group(groups, gs, GSX_GS_FIELD_ROTATION, GSX_GS_FIELD_GRAD_ROTATION, GSX_OPTIM_PARAM_ROLE_ROTATION, options.lr_rotation, options);
	add_param_group(groups, gs, GSX_GS_FIELD_OPACITY, GSX_GS_FIELD_GRAD_OPACITY, GSX_OPTIM_PARAM_ROLE_OPACITY, options.lr_opacity, options);
	add_param_group(groups, gs, GSX_GS_FIELD_SH0, GSX_GS_FIELD_GRAD_SH0, GSX_OPTIM_PARAM_ROLE_SH0, options.lr_sh0, options);
	if(options.sh_degree >= 1) {
		add_param_group(groups, gs, GSX_GS_FIELD_SH1, GSX_GS_FIELD_GRAD_SH1, GSX_OPTIM_PARAM_ROLE_SH1, options.lr_sh1, options);
	}
	if(options.sh_degree >= 2) {
		add_param_group(groups, gs, GSX_GS_FIELD_SH2, GSX_GS_FIELD_GRAD_SH2, GSX_OPTIM_PARAM_ROLE_SH2, options.lr_sh2, options);
	}
	if(options.sh_degree >= 3) {
		add_param_group(groups, gs, GSX_GS_FIELD_SH3, GSX_GS_FIELD_GRAD_SH3, GSX_OPTIM_PARAM_ROLE_SH3, options.lr_sh3, options);
	}

	gsx_optim_desc desc{};
	desc.algorithm = GSX_OPTIM_ALGORITHM_ADAM;
	desc.param_groups = groups.data();
	desc.param_group_count = static_cast<gsx_index_t>(groups.size());

	gsx_optim_t optim = nullptr;
	gsx_ok(gsx_optim_init(&optim, backend, &desc), "gsx_optim_init");
	return optim;
}

static std::optional<gsx_scheduler_t> create_scheduler(const app_options &options)
{
	if(options.scheduler == "none") {
		return std::nullopt;
	}

	gsx_scheduler_desc desc{};
	if(options.scheduler == "constant") {
		desc.algorithm = GSX_SCHEDULER_ALGORITHM_CONSTANT;
	} else if(options.scheduler == "delayed_exponential") {
		desc.algorithm = GSX_SCHEDULER_ALGORITHM_DELAYED_EXPONENTIAL;
	} else {
		throw app_error("invalid scheduler algorithm");
	}
	desc.initial_learning_rate = options.scheduler_initial_lr;
	desc.final_learning_rate = options.scheduler_final_lr;
	desc.delay_steps = options.scheduler_delay_steps;
	desc.delay_multiplier = options.scheduler_delay_multiplier;
	desc.decay_begin_step = options.scheduler_decay_begin_step;
	desc.decay_end_step = options.scheduler_decay_end_step;

	gsx_scheduler_t scheduler = nullptr;
	gsx_ok(gsx_scheduler_init(&scheduler, &desc), "gsx_scheduler_init");
	return scheduler;
}

static gsx_gs_aux_flags sh_degree_to_aux_flags(gsx_index_t sh_degree)
{
	gsx_gs_aux_flags flags = GSX_GS_AUX_NONE;
	if(sh_degree >= 1) {
		flags |= GSX_GS_AUX_SH1;
	}
	if(sh_degree >= 2) {
		flags |= GSX_GS_AUX_SH2;
	}
	if(sh_degree >= 3) {
		flags |= GSX_GS_AUX_SH3;
	}
	return flags;
}

static gsx_adc_desc make_adc_desc(const app_options &options, gsx_size_t steps_per_epoch, gsx_size_t total_steps)
{
	gsx_adc_desc desc{};
	const gsx_index_t uncapped_gaussian_limit = std::numeric_limits<gsx_index_t>::max();

	desc.algorithm = options.adc_algorithm;
	desc.pruning_opacity_threshold = options.adc_pruning_opacity_threshold;
	desc.opacity_clamp_value = options.adc_opacity_clamp_value;
	desc.max_world_scale = 0.0f;
	desc.max_screen_scale = 0.0f;
	desc.duplicate_grad_threshold = options.adc_duplicate_grad_threshold;
	desc.duplicate_scale_threshold = options.adc_duplicate_scale_threshold;
	desc.refine_every = static_cast<gsx_index_t>(steps_per_epoch);
	desc.start_refine = static_cast<gsx_index_t>(steps_per_epoch);
	desc.end_refine = static_cast<gsx_index_t>(total_steps);
	desc.max_num_gaussians = options.adc_max_num_gaussians > 0 ? options.adc_max_num_gaussians : uncapped_gaussian_limit;
	desc.reset_every = static_cast<gsx_index_t>(steps_per_epoch);
	desc.seed = options.seed;
	desc.prune_degenerate_rotation = true;
	desc.noise_strength = options.adc_noise_strength;
	desc.grow_ratio = options.adc_grow_ratio;
	return desc;
}

static std::optional<gsx_adc_t> create_adc(
	gsx_backend_t backend,
	const app_options &options,
	gsx_size_t steps_per_epoch,
	gsx_size_t total_steps)
{
	if(!options.enable_adc) {
		return std::nullopt;
	}

	gsx_adc_t adc = nullptr;
	const gsx_adc_desc desc = make_adc_desc(options, steps_per_epoch, total_steps);
	gsx_ok(gsx_adc_init(&adc, backend, &desc), "gsx_adc_init");
	return adc;
}

static gsx_gs_t create_initialized_gs(
	gsx_backend_buffer_type_t buffer_type,
	const app_options &options,
	const fs::path &ply_path,
	std::optional<gsx_adc_t> adc)
{
	gsx_gs_desc desc{};
	desc.buffer_type = buffer_type;
	desc.arena_desc.initial_capacity_bytes = 1u << 20;
	desc.count = 1;
	desc.aux_flags = sh_degree_to_aux_flags(options.sh_degree);
	if(adc.has_value() && *adc != nullptr) {
		gsx_gs_aux_flags adc_aux_flags = GSX_GS_AUX_NONE;
		gsx_ok(gsx_adc_get_gs_aux_fields(*adc, &adc_aux_flags), "gsx_adc_get_gs_aux_fields");
		desc.aux_flags |= adc_aux_flags;
	}

	gsx_gs_t gs = nullptr;
	gsx_ok(gsx_gs_init(&gs, &desc), "gsx_gs_init");
	gsx_ok(gsx_read_ply(&gs, path_string(ply_path).c_str()), "gsx_read_ply");

	if(options.override_opacity) {
		gsx_tensor_t opacity = nullptr;
		gsx_tensor_info info{};
		gsx_ok(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &opacity), "gsx_gs_get_field(opacity)");
		gsx_ok(gsx_tensor_get_info(opacity, &info), "gsx_tensor_get_info(opacity)");
		const size_t value_count = static_cast<size_t>(info.size_bytes / sizeof(float));
		std::vector<float> values(value_count, clamped_logit(options.init_opacity));
		gsx_ok(gsx_tensor_upload(opacity, values.data(), info.size_bytes), "gsx_tensor_upload(opacity)");
	}

	return gs;
}

static gsx_session_t create_session(
	gsx_backend_t backend,
	gsx_gs_t gs,
	gsx_optim_t optim,
	gsx_renderer_t renderer,
	gsx_dataloader_t train_dataloader,
	gsx_adc_t adc,
	const std::vector<gsx_loss_item> &loss_items,
	const app_options &options,
	std::optional<gsx_scheduler_t> scheduler)
{
	gsx_session_desc desc{};
	desc.backend = backend;
	desc.gs = gs;
	desc.optim = optim;
	desc.renderer = renderer;
	desc.train_dataloader = train_dataloader;
	desc.adc = adc;
	desc.scheduler = scheduler.value_or(nullptr);
	desc.loss_count = static_cast<gsx_size_t>(loss_items.size());
	desc.loss_items = loss_items.data();
	desc.render.near_plane = options.near_plane;
	desc.render.far_plane = options.far_plane;
	desc.render.background_color = options.background_color;
	desc.render.precision = options.render_precision;
	desc.render.sh_degree_mode = GSX_SESSION_SH_DEGREE_MODE_EXPLICIT;
	desc.render.sh_degree = options.sh_degree;
	desc.render.borrow_train_state = false;
	desc.optim_step.force_all = true;

	desc.adc_step.enabled = adc != nullptr;
	desc.adc_step.dataloader = nullptr;
	desc.adc_step.scene_scale = 1.0f;

	desc.workspace.buffer_type_class = options.buffer_type_class;
	desc.workspace.auto_plan = true;

	desc.reporting.retain_prediction = false;
	desc.reporting.retain_target = false;
	desc.reporting.retain_loss_map = false;
	desc.reporting.retain_grad_prediction = false;
	desc.reporting.collect_timings = true;

	desc.initial_global_step = 0;
	desc.initial_epoch_index = 0;

	gsx_session_t session = nullptr;
	gsx_ok(gsx_session_init(&session, &desc), "gsx_session_init");
	return session;
}

static gsx_size_t checked_mul_size(gsx_size_t a, gsx_size_t b, const char *context)
{
	if(a == 0 || b == 0) {
		return 0;
	}
	const gsx_size_t max_value = std::numeric_limits<gsx_size_t>::max();
	if(a > max_value / b) {
		throw app_error(std::string("overflow while computing ") + context);
	}
	return a * b;
}

static double percentile_sorted(const std::vector<double> &sorted, double percentile_0_to_1)
{
	if(sorted.empty()) {
		return 0.0;
	}
	if(sorted.size() == 1) {
		return sorted[0];
	}

	const double scaled = percentile_0_to_1 * static_cast<double>(sorted.size() - 1);
	const size_t lo = static_cast<size_t>(std::floor(scaled));
	const size_t hi = static_cast<size_t>(std::ceil(scaled));
	if(lo == hi) {
		return sorted[lo];
	}
	const double t = scaled - static_cast<double>(lo);
	return sorted[lo] * (1.0 - t) + sorted[hi] * t;
}

static distribution_stats compute_distribution(const std::vector<double> &values)
{
	distribution_stats stats{};
	stats.count = static_cast<gsx_size_t>(values.size());
	if(values.empty()) {
		return stats;
	}

	stats.sum = std::accumulate(values.begin(), values.end(), 0.0);
	stats.mean = stats.sum / static_cast<double>(values.size());

	double variance = 0.0;
	for(const double v : values) {
		const double d = v - stats.mean;
		variance += d * d;
	}
	variance /= static_cast<double>(values.size());
	stats.stddev = std::sqrt(variance);

	auto minmax = std::minmax_element(values.begin(), values.end());
	stats.min = *minmax.first;
	stats.max = *minmax.second;

	std::vector<double> sorted = values;
	std::sort(sorted.begin(), sorted.end());
	stats.p50 = percentile_sorted(sorted, 0.50);
	stats.p90 = percentile_sorted(sorted, 0.90);
	stats.p95 = percentile_sorted(sorted, 0.95);
	stats.p99 = percentile_sorted(sorted, 0.99);

	return stats;
}

static void reserve_timing_series(timing_series *series, gsx_size_t steps)
{
	const size_t n = static_cast<size_t>(steps);
	series->dataloader_us.reserve(n);
	series->render_forward_us.reserve(n);
	series->loss_forward_us.reserve(n);
	series->loss_backward_us.reserve(n);
	series->render_backward_us.reserve(n);
	series->optim_step_us.reserve(n);
	series->adc_step_us.reserve(n);
	series->total_step_us.reserve(n);
	series->global_step_after.reserve(n);
}

static void append_timing_series(timing_series *series, const gsx_session_step_report &report)
{
	series->dataloader_us.push_back(report.timings.dataloader_us);
	series->render_forward_us.push_back(report.timings.render_forward_us);
	series->loss_forward_us.push_back(report.timings.loss_forward_us);
	series->loss_backward_us.push_back(report.timings.loss_backward_us);
	series->render_backward_us.push_back(report.timings.render_backward_us);
	series->optim_step_us.push_back(report.timings.optim_step_us);
	series->adc_step_us.push_back(report.timings.adc_step_us);
	series->total_step_us.push_back(report.timings.total_step_us);
	series->global_step_after.push_back(report.global_step_after);
}

static run_summary run_session_steps(
	gsx_session_t session,
	gsx_size_t steps,
	bool collect_timing_series,
	timing_series *out_series,
	std::vector<std::vector<double>> *out_epoch_totals,
	gsx_size_t steps_per_epoch,
	bool print_epoch_progress,
	const std::string &progress_prefix)
{
	run_summary summary{};
	summary.steps = steps;

	gsx_session_step_report report{};
	for(gsx_size_t i = 0; i < steps; ++i) {
		gsx_ok(gsx_session_step(session), "gsx_session_step");
		gsx_ok(gsx_session_get_last_step_report(session, &report), "gsx_session_get_last_step_report");

		if(!report.has_timings) {
			throw app_error("session report did not provide timings even though collect_timings was requested");
		}

		if(i == 0) {
			summary.first_global_step = report.global_step_after;
			summary.lr_min = report.applied_learning_rate;
			summary.lr_max = report.applied_learning_rate;
			summary.has_lr = report.has_applied_learning_rate;
		}
		summary.last_global_step = report.global_step_after;

		if(report.has_applied_learning_rate) {
			if(!summary.has_lr) {
				summary.lr_min = report.applied_learning_rate;
				summary.lr_max = report.applied_learning_rate;
				summary.has_lr = true;
			} else {
				summary.lr_min = std::min(summary.lr_min, report.applied_learning_rate);
				summary.lr_max = std::max(summary.lr_max, report.applied_learning_rate);
			}
		}

		if((report.boundary_flags & GSX_DATALOADER_BOUNDARY_NEW_EPOCH) != 0u) {
			summary.new_epoch_markers += 1;
		}
		if((report.boundary_flags & GSX_DATALOADER_BOUNDARY_NEW_PERMUTATION) != 0u) {
			summary.new_permutation_markers += 1;
		}
		if(report.adc_result_available) {
			summary.adc_result_count += 1;
		}

		if(collect_timing_series) {
			append_timing_series(out_series, report);
			if(out_epoch_totals != nullptr && steps_per_epoch > 0) {
				const gsx_size_t epoch_slot = i / steps_per_epoch;
				if(epoch_slot < out_epoch_totals->size()) {
					(*out_epoch_totals)[static_cast<size_t>(epoch_slot)].push_back(report.timings.total_step_us);
				}
			}
		}

		if(print_epoch_progress && steps_per_epoch > 0 && ((i + 1) % steps_per_epoch) == 0) {
			const gsx_size_t local_epoch = (i + 1) / steps_per_epoch;
			std::cout << progress_prefix << " epoch=" << local_epoch
					  << " steps=" << (i + 1)
					  << " global_step=" << report.global_step_after
					  << " last_total_us=" << report.timings.total_step_us
					  << "\n";
		}
	}

	return summary;
}

static void print_distribution_header()
{
	std::cout << std::left << std::setw(24) << "stage"
			  << std::right << std::setw(8) << "count"
			  << std::setw(14) << "mean_us"
			  << std::setw(14) << "stddev_us"
			  << std::setw(14) << "min_us"
			  << std::setw(14) << "p50_us"
			  << std::setw(14) << "p90_us"
			  << std::setw(14) << "p95_us"
			  << std::setw(14) << "p99_us"
			  << std::setw(14) << "max_us"
			  << std::setw(12) << "mean_%"
			  << "\n";
}

static void print_distribution_row(const std::string &name, const distribution_stats &stats, double total_mean_us)
{
	const double pct = (total_mean_us > 0.0) ? (100.0 * stats.mean / total_mean_us) : 0.0;
	std::cout << std::left << std::setw(24) << name
			  << std::right << std::setw(8) << stats.count
			  << std::setw(14) << std::fixed << std::setprecision(3) << stats.mean
			  << std::setw(14) << stats.stddev
			  << std::setw(14) << stats.min
			  << std::setw(14) << stats.p50
			  << std::setw(14) << stats.p90
			  << std::setw(14) << stats.p95
			  << std::setw(14) << stats.p99
			  << std::setw(14) << stats.max
			  << std::setw(12) << pct
			  << "\n";
}

static void print_top_slowest_steps(const timing_series &series, gsx_size_t max_rows)
{
	if(series.total_step_us.empty()) {
		return;
	}

	std::vector<size_t> indices(series.total_step_us.size());
	std::iota(indices.begin(), indices.end(), 0);
	std::sort(indices.begin(), indices.end(), [&series](size_t a, size_t b) {
		return series.total_step_us[a] > series.total_step_us[b];
	});

	const size_t rows = std::min<size_t>(static_cast<size_t>(max_rows), indices.size());
	std::cout << "\nslowest measured steps (by total_step_us):\n";
	std::cout << std::left << std::setw(8) << "rank"
			  << std::right << std::setw(14) << "global_step"
			  << std::setw(14) << "total_us"
			  << std::setw(14) << "render_fwd"
			  << std::setw(14) << "render_bwd"
			  << std::setw(14) << "loss_fwd"
			  << std::setw(14) << "loss_bwd"
			  << std::setw(14) << "optim"
			  << std::setw(14) << "adc"
			  << std::setw(14) << "dataloader"
			  << "\n";

	for(size_t rank = 0; rank < rows; ++rank) {
		const size_t idx = indices[rank];
		std::cout << std::left << std::setw(8) << (rank + 1)
				  << std::right << std::setw(14) << series.global_step_after[idx]
				  << std::setw(14) << std::fixed << std::setprecision(3) << series.total_step_us[idx]
				  << std::setw(14) << series.render_forward_us[idx]
				  << std::setw(14) << series.render_backward_us[idx]
				  << std::setw(14) << series.loss_forward_us[idx]
				  << std::setw(14) << series.loss_backward_us[idx]
				  << std::setw(14) << series.optim_step_us[idx]
				  << std::setw(14) << series.adc_step_us[idx]
				  << std::setw(14) << series.dataloader_us[idx]
				  << "\n";
	}
}

static void print_epoch_stats(const std::vector<std::vector<double>> &epoch_totals)
{
	if(epoch_totals.empty()) {
		return;
	}

	std::cout << "\nper-epoch total_step_us summary:\n";
	std::cout << std::left << std::setw(8) << "epoch"
			  << std::right << std::setw(10) << "steps"
			  << std::setw(14) << "mean_us"
			  << std::setw(14) << "stddev_us"
			  << std::setw(14) << "min_us"
			  << std::setw(14) << "p95_us"
			  << std::setw(14) << "max_us"
			  << "\n";

	for(size_t i = 0; i < epoch_totals.size(); ++i) {
		const distribution_stats stats = compute_distribution(epoch_totals[i]);
		std::cout << std::left << std::setw(8) << (i + 1)
				  << std::right << std::setw(10) << stats.count
				  << std::setw(14) << std::fixed << std::setprecision(3) << stats.mean
				  << std::setw(14) << stats.stddev
				  << std::setw(14) << stats.min
				  << std::setw(14) << stats.p95
				  << std::setw(14) << stats.max
				  << "\n";
	}
}

static void print_run_summary(
	const std::string &tag,
	const run_summary &summary,
	double wall_seconds,
	const distribution_stats &total_stats)
{
	std::cout << "\n" << tag << " summary:\n";
	std::cout << "  steps=" << summary.steps
			  << " global_step_range=[" << summary.first_global_step << ", " << summary.last_global_step << "]"
			  << " wall_s=" << std::fixed << std::setprecision(6) << wall_seconds
			  << " throughput_steps_per_s=" << ((wall_seconds > 0.0) ? (static_cast<double>(summary.steps) / wall_seconds) : 0.0)
			  << "\n";
	std::cout << "  dataloader_new_epoch_markers=" << summary.new_epoch_markers
			  << " dataloader_new_permutation_markers=" << summary.new_permutation_markers
			  << " adc_result_count=" << summary.adc_result_count
			  << "\n";
	if(summary.has_lr) {
		std::cout << "  applied_learning_rate_range=[" << summary.lr_min << ", " << summary.lr_max << "]\n";
	}
	if(total_stats.count > 0) {
		std::cout << "  total_step_us: mean=" << total_stats.mean
				  << " p95=" << total_stats.p95
				  << " p99=" << total_stats.p99
				  << " max=" << total_stats.max
				  << " sum_s=" << (total_stats.sum / 1.0e6)
				  << "\n";
	}
}

} // namespace

int main(int argc, char **argv)
{
	try {
		const app_options options = parse_options(argc, argv);

		std::cout << "gsx_benchmark_dataset: loading split='" << options.split
				  << "' from " << path_string(options.dataset_root) << "\n";

		const loaded_split selected_split = load_split(options.dataset_root, options.split, options.max_input_width);
		const loaded_split train_split_for_intrinsics =
			(options.split == "train") ? selected_split : load_split(options.dataset_root, "train", options.max_input_width);
		validate_matching_intrinsics(selected_split.intrinsics, train_split_for_intrinsics.intrinsics);

		const fs::path ply_path = options.ply_override.has_value() ? *options.ply_override : (options.dataset_root / "train" / "points3d.ply");
		if(!fs::exists(ply_path)) {
			throw app_error("missing PLY file: " + path_string(ply_path));
		}

		if(selected_split.samples.empty()) {
			throw app_error("selected split has no samples");
		}

		dataset_view split_view{ &selected_split };
		backend_owner backend(&gsx_backend_free);
		dataset_owner dataset(&gsx_dataset_free);
		dataloader_owner dataloader(&gsx_dataloader_free);
		renderer_owner renderer(&gsx_renderer_free);
		gs_owner gs(&gsx_gs_free);
		optim_owner optim(&gsx_optim_free);
		scheduler_owner scheduler(&gsx_scheduler_free);
		adc_owner adc(&gsx_adc_free);
		session_owner session(&gsx_session_free);
		std::vector<loss_owner> losses;
		std::vector<loss_context_owner> loss_contexts;
		std::vector<gsx_loss_item> loss_items;

		backend.handle = create_backend(options);
		const gsx_backend_buffer_type_t selected_buffer_type = find_buffer_type(backend.handle, options.buffer_type_class);
		const gsx_size_t steps_per_epoch = static_cast<gsx_size_t>(selected_split.samples.size());
		const gsx_size_t warmup_steps = checked_mul_size(static_cast<gsx_size_t>(options.warmup_epochs), steps_per_epoch, "warmup steps");
		const gsx_size_t iter_steps = checked_mul_size(static_cast<gsx_size_t>(options.iter_epochs), steps_per_epoch, "iteration steps");
		const gsx_size_t total_steps = warmup_steps + iter_steps;

		dataset.handle = create_dataset(&split_view, options.split);
		dataloader.handle = create_dataloader(
			backend.handle,
			dataset.handle,
			options.shuffle_each_epoch,
			options.seed,
			options.enable_async_prefetch,
			options.prefetch_count);
		renderer.handle = create_renderer(backend.handle, selected_split);
		{
			if(const auto maybe_adc = create_adc(backend.handle, options, steps_per_epoch, total_steps); maybe_adc.has_value()) {
				adc.handle = *maybe_adc;
			}
		}
		gs.handle = create_initialized_gs(
			selected_buffer_type,
			options,
			ply_path,
			adc.handle != nullptr ? std::optional<gsx_adc_t>(adc.handle) : std::nullopt);
		optim.handle = create_optimizer(backend.handle, gs.handle, options);
		if(const auto maybe_scheduler = create_scheduler(options); maybe_scheduler.has_value()) {
			scheduler.handle = *maybe_scheduler;
		}

		if(options.l1_scale != 0.0f) {
			losses.emplace_back(&gsx_loss_free);
			losses.back().handle = create_loss(backend.handle, GSX_LOSS_ALGORITHM_L1);
			loss_contexts.emplace_back(&gsx_loss_context_free);
			loss_contexts.back().handle = create_loss_context(losses.back().handle);
			gsx_loss_item item{};
			item.loss = losses.back().handle;
			item.context = loss_contexts.back().handle;
			item.scale = options.l1_scale;
			loss_items.push_back(item);
		}
		if(options.ssim_scale != 0.0f) {
			losses.emplace_back(&gsx_loss_free);
			losses.back().handle = create_loss(backend.handle, GSX_LOSS_ALGORITHM_SSIM);
			loss_contexts.emplace_back(&gsx_loss_context_free);
			loss_contexts.back().handle = create_loss_context(losses.back().handle);
			gsx_loss_item item{};
			item.loss = losses.back().handle;
			item.context = loss_contexts.back().handle;
			item.scale = options.ssim_scale;
			loss_items.push_back(item);
		}

		session.handle = create_session(
			backend.handle,
			gs.handle,
			optim.handle,
			renderer.handle,
			dataloader.handle,
			adc.handle,
			loss_items,
			options,
			scheduler.handle != nullptr ? std::optional<gsx_scheduler_t>(scheduler.handle) : std::nullopt);

		gsx_backend_info backend_info{};
		gsx_backend_device_info backend_device_info{};
		gsx_dataloader_info dataloader_info{};
		gsx_gs_info gs_info_before{};
		gsx_ok(gsx_backend_get_info(backend.handle, &backend_info), "gsx_backend_get_info");
		gsx_ok(gsx_backend_device_get_info(backend_info.device, &backend_device_info), "gsx_backend_device_get_info");
		gsx_ok(gsx_dataloader_get_info(dataloader.handle, &dataloader_info), "gsx_dataloader_get_info");
		gsx_ok(gsx_gs_get_info(gs.handle, &gs_info_before), "gsx_gs_get_info(before)");

		std::cout << "\nbenchmark configuration:\n";
		std::cout << "  backend=" << backend_name(options.backend_type)
				  << " device_index=" << options.device_index
				  << " device_name='" << (backend_device_info.name != nullptr ? backend_device_info.name : "unknown") << "'"
				  << " total_memory_bytes=" << backend_device_info.total_memory_bytes
				  << "\n";
		std::cout << "  workspace_buffer_type_class=" << buffer_type_name(options.buffer_type_class) << "\n";
		std::cout << "  dataset_root=" << path_string(options.dataset_root)
				  << " split=" << options.split
				  << " samples_per_epoch=" << steps_per_epoch
				  << " resolution=" << selected_split.intrinsics.width << "x" << selected_split.intrinsics.height
				  << "\n";
		std::cout << "  ply_path=" << path_string(ply_path) << "\n";
		std::cout << "  warmup_epochs=" << options.warmup_epochs
				  << " iter_epochs=" << options.iter_epochs
				  << " warmup_steps=" << warmup_steps
				  << " iter_steps=" << iter_steps
				  << "\n";
		std::cout << "  dataloader: shuffle_each_epoch=" << std::boolalpha << options.shuffle_each_epoch
				  << " async_prefetch=" << dataloader_info.enable_async_prefetch
				  << " prefetch_count=" << dataloader_info.prefetch_count
				  << " seed=" << options.seed
				  << "\n";
		std::cout << "  session: sh_degree=" << options.sh_degree
				  << " near=" << options.near_plane
				  << " far=" << options.far_plane
				  << " losses=" << loss_items.size()
				  << " scheduler=" << options.scheduler
				  << "\n";
		std::cout << "  adc: enabled=" << std::boolalpha << (adc.handle != nullptr)
				  << " algorithm=" << adc_algorithm_name(options.adc_algorithm)
				  << " refine_every_steps=" << steps_per_epoch
				  << " grow_ratio=" << options.adc_grow_ratio
				  << " noise_strength=" << options.adc_noise_strength
				  << " prune_threshold=" << options.adc_pruning_opacity_threshold
				  << "\n";
		std::cout << "  initial_gaussian_count=" << gs_info_before.count << "\n";

		run_summary warmup_summary{};
		distribution_stats warmup_total_stats{};
		if(warmup_steps > 0) {
			std::cout << "\nstarting warmup...\n";
			timing_series warmup_series;
			reserve_timing_series(&warmup_series, warmup_steps);

			const auto warmup_begin = std::chrono::steady_clock::now();
			warmup_summary = run_session_steps(
				session.handle,
				warmup_steps,
				true,
				&warmup_series,
				nullptr,
				steps_per_epoch,
				true,
				"warmup");
			gsx_ok(gsx_backend_major_stream_sync(backend.handle), "gsx_backend_major_stream_sync(warmup)");
			const auto warmup_end = std::chrono::steady_clock::now();

			const double warmup_wall_s = std::chrono::duration<double>(warmup_end - warmup_begin).count();
			warmup_total_stats = compute_distribution(warmup_series.total_step_us);
			print_run_summary("warmup", warmup_summary, warmup_wall_s, warmup_total_stats);
		}

		std::cout << "\nstarting measured run...\n";
		timing_series measured_series;
		reserve_timing_series(&measured_series, iter_steps);
		std::vector<std::vector<double>> epoch_totals(static_cast<size_t>(options.iter_epochs));

		const auto measure_begin = std::chrono::steady_clock::now();
		const run_summary measured_summary = run_session_steps(
			session.handle,
			iter_steps,
			true,
			&measured_series,
			&epoch_totals,
			steps_per_epoch,
			true,
			"iter");
		gsx_ok(gsx_backend_major_stream_sync(backend.handle), "gsx_backend_major_stream_sync(iter)");
		const auto measure_end = std::chrono::steady_clock::now();

		const double measured_wall_s = std::chrono::duration<double>(measure_end - measure_begin).count();

		const distribution_stats dataloader_stats = compute_distribution(measured_series.dataloader_us);
		const distribution_stats render_forward_stats = compute_distribution(measured_series.render_forward_us);
		const distribution_stats loss_forward_stats = compute_distribution(measured_series.loss_forward_us);
		const distribution_stats loss_backward_stats = compute_distribution(measured_series.loss_backward_us);
		const distribution_stats render_backward_stats = compute_distribution(measured_series.render_backward_us);
		const distribution_stats optim_stats = compute_distribution(measured_series.optim_step_us);
		const distribution_stats adc_stats = compute_distribution(measured_series.adc_step_us);
		const distribution_stats total_stats = compute_distribution(measured_series.total_step_us);

		print_run_summary("measured", measured_summary, measured_wall_s, total_stats);

		std::cout << "\nper-stage timing distributions (microseconds):\n";
		print_distribution_header();
		print_distribution_row("dataloader_us", dataloader_stats, total_stats.mean);
		print_distribution_row("render_forward_us", render_forward_stats, total_stats.mean);
		print_distribution_row("loss_forward_us", loss_forward_stats, total_stats.mean);
		print_distribution_row("loss_backward_us", loss_backward_stats, total_stats.mean);
		print_distribution_row("render_backward_us", render_backward_stats, total_stats.mean);
		print_distribution_row("optim_step_us", optim_stats, total_stats.mean);
		print_distribution_row(adc.handle != nullptr ? "adc_step_us" : "adc_step_us (disabled)", adc_stats, total_stats.mean);
		print_distribution_row("total_step_us", total_stats, total_stats.mean);

		if(total_stats.sum > 0.0) {
			std::cout << "\nthroughput details:\n";
			std::cout << "  wall_time_s=" << measured_wall_s << "\n";
			std::cout << "  accumulated_total_step_s=" << (total_stats.sum / 1.0e6) << "\n";
			std::cout << "  measured_steps_per_s_wall=" << (static_cast<double>(iter_steps) / measured_wall_s) << "\n";
			std::cout << "  measured_steps_per_s_from_total_step_us=" << (static_cast<double>(iter_steps) / (total_stats.sum / 1.0e6)) << "\n";
			std::cout << "  wall_over_sum_total_step_ratio=" << (measured_wall_s / (total_stats.sum / 1.0e6)) << "\n";
		}

		print_epoch_stats(epoch_totals);
		print_top_slowest_steps(measured_series, 5);

		gsx_gs_info gs_info_after{};
		gsx_session_state state_after{};
		gsx_ok(gsx_gs_get_info(gs.handle, &gs_info_after), "gsx_gs_get_info(after)");
		gsx_ok(gsx_session_get_state(session.handle, &state_after), "gsx_session_get_state(after)");

		std::cout << "\nfinal state:\n";
		std::cout << "  session_global_step=" << state_after.global_step
				  << " session_epoch_index=" << state_after.epoch_index
				  << " successful_step_count=" << state_after.successful_step_count
				  << " failed_step_count=" << state_after.failed_step_count
				  << "\n";
		std::cout << "  gaussian_count_before=" << gs_info_before.count
				  << " gaussian_count_after=" << gs_info_after.count
				  << "\n";

		std::cout << "\nbenchmark completed successfully\n";
		return EXIT_SUCCESS;
	} catch(const std::exception &e) {
		std::cerr << "error: " << e.what() << "\n";
		return EXIT_FAILURE;
	}
}
