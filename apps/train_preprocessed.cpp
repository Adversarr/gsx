#include <gsx/extra/gsx-flann.h>
#include <gsx/extra/gsx-io-ply.h>
#include <gsx/extra/gsx-stbi.h>
#include <gsx/gsx.h>

#include <cxxopts.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
    fs::path output_dir;

    gsx_backend_type backend_type = GSX_BACKEND_TYPE_CPU;
    gsx_index_t device_index = 0;
    gsx_backend_buffer_type_class buffer_type_class = GSX_BACKEND_BUFFER_TYPE_DEVICE;

    gsx_index_t max_input_width = 1600;
    gsx_index_t steps = 30000;
    gsx_index_t log_interval = 100;
    gsx_index_t eval_interval = 1000;
    gsx_size_t seed = 42;
    bool shuffle_each_epoch = true;
    bool save_train_renders = false;
    bool save_val_renders = true;

    gsx_render_precision render_precision = GSX_RENDER_PRECISION_FLOAT32;
    gsx_index_t sh_degree = 3;
    gsx_index_t initial_sh_degree = 0;
    gsx_index_t sh_degree_interval = 1000;
    gsx_float_t near_plane = 0.01f;
    gsx_float_t far_plane = 100.0f;
    gsx_vec3 background_color{ 0.0f, 0.0f, 0.0f };

    gsx_float_t l1_scale = 0.8f;
    gsx_float_t ssim_scale = 0.2f;

    gsx_float_t lr_mean3d = 0.00016f;
    gsx_float_t lr_logscale = 0.005f;
    gsx_float_t lr_rotation = 0.001f;
    gsx_float_t lr_opacity = 0.025f;
    gsx_float_t lr_sh0 = 0.005f;
    gsx_float_t lr_sh1 = 0.0025f;
    gsx_float_t lr_sh2 = 0.0025f;
    gsx_float_t lr_sh3 = 0.0025f;
    gsx_float_t beta1 = 0.9f;
    gsx_float_t beta2 = 0.999f;
    gsx_float_t epsilon = 1e-8f;
    gsx_float_t weight_decay = 0.0f;
    gsx_float_t max_grad = 0.0f;

    bool override_opacity = false;
    gsx_float_t init_opacity = 0.1f;

    std::string scheduler = "constant";
    gsx_float_t scheduler_initial_lr = 0.00016f;
    gsx_float_t scheduler_final_lr = 0.00016f;
    gsx_size_t scheduler_delay_steps = 0;
    gsx_float_t scheduler_delay_multiplier = 1.0f;
    gsx_size_t scheduler_decay_begin_step = 0;
    gsx_size_t scheduler_decay_end_step = 30000;

    bool enable_adc = false;
    gsx_adc_algorithm adc_algorithm = GSX_ADC_ALGORITHM_DEFAULT;
    gsx_float_t adc_scene_scale = 1.0f;
    gsx_float_t adc_pruning_opacity_threshold = 0.005f;
    gsx_float_t adc_opacity_clamp_value = 0.01f;
    gsx_float_t adc_max_world_scale = 0.1f;
    gsx_float_t adc_max_screen_scale = 10.0f;
    gsx_float_t adc_duplicate_grad_threshold = 0.0002f;
    gsx_float_t adc_duplicate_scale_threshold = 0.005f;
    gsx_index_t adc_refine_every = 500;
    gsx_index_t adc_start_refine = 500;
    gsx_index_t adc_end_refine = 15000;
    gsx_index_t adc_max_num_gaussians = 1000000;
    gsx_index_t adc_reset_every = 3000;
    bool adc_prune_degenerate_rotation = true;
    gsx_float_t adc_duplicate_absgrad_threshold = 0.0012f;
    gsx_float_t adc_noise_strength = 8.0f;
    gsx_float_t adc_grow_ratio = 0.001f;
    gsx_float_t adc_loss_threshold = 0.06f;
    gsx_index_t adc_max_sampled_cameras = 16;
    gsx_float_t adc_importance_threshold = 0.0f;
    gsx_float_t adc_prune_budget_ratio = 0.0f;

    gsx_index_t flann_num_neighbors = 16;
    gsx_float_t flann_init_scaling = 1.0f;
    gsx_float_t flann_min_distance = 1.0e-7f;
    gsx_float_t flann_max_distance = 1.0e6f;
    gsx_float_t flann_default_distance = 0.01f;
    gsx_float_t flann_radius = 0.1f;
    bool flann_use_anisotropic = false;
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

struct eval_image_metric {
    gsx_id_t image_id = 0;
    std::string name;
    gsx_index_t width = 0;
    gsx_index_t height = 0;
    double psnr = 0.0;
    std::string render_path;
};

struct eval_summary {
    gsx_size_t global_step = 0;
    double psnr_mean = 0.0;
    double psnr_min = 0.0;
    double psnr_max = 0.0;
    json metrics_json;
};

static bool gsx_ok(gsx_error error, const char *context)
{
    if(gsx_error_is_success(error)) {
        return true;
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

static json read_json_file(const fs::path &path)
{
    std::ifstream input(path);
    if(!input) {
        throw app_error("failed to open JSON file: " + path_string(path));
    }
    json value;
    input >> value;
    return value;
}

static bool nearly_equal(gsx_float_t a, gsx_float_t b, gsx_float_t epsilon = 1.0e-4f)
{
    return std::fabs(a - b) <= epsilon;
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

static std::array<gsx_float_t, 9> quat_wxyz_to_rotation_matrix(const std::array<gsx_float_t, 4> &q)
{
    const gsx_float_t w = q[0];
    const gsx_float_t x = q[1];
    const gsx_float_t y = q[2];
    const gsx_float_t z = q[3];
    const gsx_float_t xx = x * x;
    const gsx_float_t yy = y * y;
    const gsx_float_t zz = z * z;
    const gsx_float_t xy = x * y;
    const gsx_float_t xz = x * z;
    const gsx_float_t yz = y * z;
    const gsx_float_t wx = w * x;
    const gsx_float_t wy = w * y;
    const gsx_float_t wz = w * z;

    return {
        1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz),         2.0f * (xz + wy),
        2.0f * (xy + wz),         1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx),
        2.0f * (xz - wy),         2.0f * (yz + wx),         1.0f - 2.0f * (xx + yy),
    };
}

static gsx_vec3 mat3_mul_vec3(const std::array<gsx_float_t, 9> &m, const std::array<gsx_float_t, 3> &v)
{
    gsx_vec3 out{};
    out.x = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
    out.y = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
    out.z = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
    return out;
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

static gsx_backend_type parse_backend_type(const std::string &value)
{
    if(value == "cpu") {
        return GSX_BACKEND_TYPE_CPU;
    }
    if(value == "cuda") {
        return GSX_BACKEND_TYPE_CUDA;
    }
    if(value == "metal") {
        return GSX_BACKEND_TYPE_METAL;
    }
    throw app_error("invalid backend: " + value);
}

static gsx_backend_buffer_type_class parse_buffer_type_class(const std::string &value)
{
    if(value == "host") {
        return GSX_BACKEND_BUFFER_TYPE_HOST;
    }
    if(value == "host_pinned") {
        return GSX_BACKEND_BUFFER_TYPE_HOST_PINNED;
    }
    if(value == "device") {
        return GSX_BACKEND_BUFFER_TYPE_DEVICE;
    }
    if(value == "unified") {
        return GSX_BACKEND_BUFFER_TYPE_UNIFIED;
    }
    throw app_error("invalid buffer type: " + value);
}

static gsx_render_precision parse_render_precision(const std::string &value)
{
    if(value == "float32") {
        return GSX_RENDER_PRECISION_FLOAT32;
    }
    if(value == "float16") {
        return GSX_RENDER_PRECISION_FLOAT16;
    }
    throw app_error("invalid render precision: " + value);
}

static gsx_adc_algorithm parse_adc_algorithm(const std::string &value)
{
    if(value == "default") {
        return GSX_ADC_ALGORITHM_DEFAULT;
    }
    if(value == "absgs") {
        return GSX_ADC_ALGORITHM_ABSGS;
    }
    if(value == "mcmc") {
        return GSX_ADC_ALGORITHM_MCMC;
    }
    if(value == "fastgs") {
        return GSX_ADC_ALGORITHM_FASTGS;
    }
    throw app_error("invalid adc algorithm: " + value);
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

static app_options parse_options(int argc, char **argv)
{
    cxxopts::Options options("train-preprocessed", "Train GSX on preprocessed garden datasets");
    options.add_options()
        ("dataset-root", "Dataset root containing train/ and val/", cxxopts::value<std::string>())
        ("output-dir", "Output directory", cxxopts::value<std::string>())
        ("backend", "Backend type", cxxopts::value<std::string>()->default_value("cpu"))
        ("device", "Backend device index", cxxopts::value<gsx_index_t>()->default_value("0"))
        ("buffer-type", "Backend buffer type class", cxxopts::value<std::string>()->default_value("device"))
        ("max-input-width", "Maximum input width, 0 disables scaling", cxxopts::value<gsx_index_t>()->default_value("1600"))
        ("steps", "Training steps", cxxopts::value<gsx_index_t>()->default_value("30000"))
        ("log-interval", "Log interval", cxxopts::value<gsx_index_t>()->default_value("100"))
        ("eval-interval", "Evaluation interval", cxxopts::value<gsx_index_t>()->default_value("1000"))
        ("seed", "Seed", cxxopts::value<gsx_size_t>()->default_value("42"))
        ("shuffle-each-epoch", "Shuffle training samples each epoch", cxxopts::value<bool>()->default_value("true"))
        ("save-train-renders", "Write train renders", cxxopts::value<bool>()->default_value("false"))
        ("save-val-renders", "Write validation renders", cxxopts::value<bool>()->default_value("true"))
        ("render-precision", "Render precision", cxxopts::value<std::string>()->default_value("float32"))
        ("sh-degree", "Maximum spherical harmonic degree", cxxopts::value<gsx_index_t>()->default_value("3"))
        ("initial-sh-degree", "Initial spherical harmonic degree", cxxopts::value<gsx_index_t>()->default_value("0"))
        ("sh-degree-interval", "Increase SH degree every N steps, <=0 disables progression", cxxopts::value<gsx_index_t>()->default_value("1000"))
        ("near", "Near plane", cxxopts::value<gsx_float_t>()->default_value("0.01"))
        ("far", "Far plane", cxxopts::value<gsx_float_t>()->default_value("100.0"))
        ("bg-r", "Background red", cxxopts::value<gsx_float_t>()->default_value("0.0"))
        ("bg-g", "Background green", cxxopts::value<gsx_float_t>()->default_value("0.0"))
        ("bg-b", "Background blue", cxxopts::value<gsx_float_t>()->default_value("0.0"))
        ("l1-scale", "L1 loss scale", cxxopts::value<gsx_float_t>()->default_value("0.8"))
        ("ssim-scale", "SSIM loss scale", cxxopts::value<gsx_float_t>()->default_value("0.2"))
        ("lr-mean3d", "Learning rate for mean3d", cxxopts::value<gsx_float_t>()->default_value("0.00016"))
        ("lr-logscale", "Learning rate for logscale", cxxopts::value<gsx_float_t>()->default_value("0.005"))
        ("lr-rotation", "Learning rate for rotation", cxxopts::value<gsx_float_t>()->default_value("0.001"))
        ("lr-opacity", "Learning rate for opacity", cxxopts::value<gsx_float_t>()->default_value("0.025"))
        ("lr-sh0", "Learning rate for sh0", cxxopts::value<gsx_float_t>()->default_value("0.005"))
        ("lr-sh1", "Learning rate for sh1", cxxopts::value<gsx_float_t>()->default_value("0.0025"))
        ("lr-sh2", "Learning rate for sh2", cxxopts::value<gsx_float_t>()->default_value("0.0025"))
        ("lr-sh3", "Learning rate for sh3", cxxopts::value<gsx_float_t>()->default_value("0.0025"))
        ("beta1", "Adam beta1", cxxopts::value<gsx_float_t>()->default_value("0.9"))
        ("beta2", "Adam beta2", cxxopts::value<gsx_float_t>()->default_value("0.999"))
        ("epsilon", "Adam epsilon", cxxopts::value<gsx_float_t>()->default_value("1e-8"))
        ("weight-decay", "Weight decay", cxxopts::value<gsx_float_t>()->default_value("0.0"))
        ("max-grad", "Max gradient clamp, <=0 disables", cxxopts::value<gsx_float_t>()->default_value("0.0"))
        ("override-opacity", "Override PLY opacity values", cxxopts::value<bool>()->default_value("false"))
        ("init-opacity", "Opacity used when override-opacity is enabled", cxxopts::value<gsx_float_t>()->default_value("0.1"))
        ("scheduler", "Scheduler: constant or delayed_exponential", cxxopts::value<std::string>()->default_value("constant"))
        ("scheduler-initial-lr", "Scheduler initial lr", cxxopts::value<gsx_float_t>()->default_value("0.00016"))
        ("scheduler-final-lr", "Scheduler final lr", cxxopts::value<gsx_float_t>()->default_value("0.00016"))
        ("scheduler-delay-steps", "Scheduler delay steps", cxxopts::value<gsx_size_t>()->default_value("0"))
        ("scheduler-delay-multiplier", "Scheduler delay multiplier", cxxopts::value<gsx_float_t>()->default_value("1.0"))
        ("scheduler-decay-begin-step", "Scheduler decay begin step", cxxopts::value<gsx_size_t>()->default_value("0"))
        ("scheduler-decay-end-step", "Scheduler decay end step", cxxopts::value<gsx_size_t>()->default_value("30000"))
        ("enable-adc", "Enable ADC", cxxopts::value<bool>()->default_value("false"))
        ("adc-algorithm", "ADC algorithm", cxxopts::value<std::string>()->default_value("default"))
        ("adc-scene-scale", "ADC scene scale", cxxopts::value<gsx_float_t>()->default_value("1.0"))
        ("adc-pruning-opacity-threshold", "ADC pruning opacity threshold", cxxopts::value<gsx_float_t>()->default_value("0.005"))
        ("adc-opacity-clamp-value", "ADC opacity clamp value", cxxopts::value<gsx_float_t>()->default_value("0.01"))
        ("adc-max-world-scale", "ADC max world scale", cxxopts::value<gsx_float_t>()->default_value("0.1"))
        ("adc-max-screen-scale", "ADC max screen scale", cxxopts::value<gsx_float_t>()->default_value("10.0"))
        ("adc-duplicate-grad-threshold", "ADC duplicate grad threshold", cxxopts::value<gsx_float_t>()->default_value("0.0002"))
        ("adc-duplicate-scale-threshold", "ADC duplicate scale threshold", cxxopts::value<gsx_float_t>()->default_value("0.005"))
        ("adc-refine-every", "ADC refine every", cxxopts::value<gsx_index_t>()->default_value("500"))
        ("adc-start-refine", "ADC start refine", cxxopts::value<gsx_index_t>()->default_value("500"))
        ("adc-end-refine", "ADC end refine", cxxopts::value<gsx_index_t>()->default_value("15000"))
        ("adc-max-num-gaussians", "ADC max num gaussians", cxxopts::value<gsx_index_t>()->default_value("1000000"))
        ("adc-reset-every", "ADC reset every", cxxopts::value<gsx_index_t>()->default_value("3000"))
        ("adc-prune-degenerate-rotation", "ADC prune degenerate rotation", cxxopts::value<bool>()->default_value("true"))
        ("adc-duplicate-absgrad-threshold", "ADC duplicate absgrad threshold", cxxopts::value<gsx_float_t>()->default_value("0.0012"))
        ("adc-noise-strength", "ADC noise strength", cxxopts::value<gsx_float_t>()->default_value("8.0"))
        ("adc-grow-ratio", "ADC grow ratio", cxxopts::value<gsx_float_t>()->default_value("0.001"))
        ("adc-loss-threshold", "ADC loss threshold", cxxopts::value<gsx_float_t>()->default_value("0.06"))
        ("adc-max-sampled-cameras", "ADC max sampled cameras", cxxopts::value<gsx_index_t>()->default_value("16"))
        ("adc-importance-threshold", "ADC importance threshold", cxxopts::value<gsx_float_t>()->default_value("0.0"))
        ("adc-prune-budget-ratio", "ADC prune budget ratio", cxxopts::value<gsx_float_t>()->default_value("0.0"))
        ("flann-num-neighbors", "FLANN neighbor count", cxxopts::value<gsx_index_t>()->default_value("16"))
        ("flann-init-scaling", "FLANN init scaling", cxxopts::value<gsx_float_t>()->default_value("1.0"))
        ("flann-min-distance", "FLANN min distance", cxxopts::value<gsx_float_t>()->default_value("1e-7"))
        ("flann-max-distance", "FLANN max distance", cxxopts::value<gsx_float_t>()->default_value("1000000.0"))
        ("flann-default-distance", "FLANN default distance", cxxopts::value<gsx_float_t>()->default_value("0.01"))
        ("flann-radius", "FLANN radius", cxxopts::value<gsx_float_t>()->default_value("0.1"))
        ("flann-use-anisotropic", "Use anisotropic FLANN initialization", cxxopts::value<bool>()->default_value("false"))
        ("help", "Print usage");

    const auto result = options.parse(argc, argv);
    if(result.count("help") > 0) {
        std::cout << options.help() << "\n";
        std::exit(0);
    }

    if(result.count("dataset-root") == 0 || result.count("output-dir") == 0) {
        throw app_error("both --dataset-root and --output-dir are required");
    }

    app_options out;
    out.dataset_root = result["dataset-root"].as<std::string>();
    out.output_dir = result["output-dir"].as<std::string>();
    out.backend_type = parse_backend_type(result["backend"].as<std::string>());
    out.device_index = result["device"].as<gsx_index_t>();
    out.buffer_type_class = parse_buffer_type_class(result["buffer-type"].as<std::string>());
    out.max_input_width = result["max-input-width"].as<gsx_index_t>();
    out.steps = result["steps"].as<gsx_index_t>();
    out.log_interval = result["log-interval"].as<gsx_index_t>();
    out.eval_interval = result["eval-interval"].as<gsx_index_t>();
    out.seed = result["seed"].as<gsx_size_t>();
    out.shuffle_each_epoch = result["shuffle-each-epoch"].as<bool>();
    out.save_train_renders = result["save-train-renders"].as<bool>();
    out.save_val_renders = result["save-val-renders"].as<bool>();
    out.render_precision = parse_render_precision(result["render-precision"].as<std::string>());
    out.sh_degree = result["sh-degree"].as<gsx_index_t>();
    out.initial_sh_degree = result["initial-sh-degree"].as<gsx_index_t>();
    out.sh_degree_interval = result["sh-degree-interval"].as<gsx_index_t>();
    out.near_plane = result["near"].as<gsx_float_t>();
    out.far_plane = result["far"].as<gsx_float_t>();
    out.background_color.x = result["bg-r"].as<gsx_float_t>();
    out.background_color.y = result["bg-g"].as<gsx_float_t>();
    out.background_color.z = result["bg-b"].as<gsx_float_t>();
    out.l1_scale = result["l1-scale"].as<gsx_float_t>();
    out.ssim_scale = result["ssim-scale"].as<gsx_float_t>();
    out.lr_mean3d = result["lr-mean3d"].as<gsx_float_t>();
    out.lr_logscale = result["lr-logscale"].as<gsx_float_t>();
    out.lr_rotation = result["lr-rotation"].as<gsx_float_t>();
    out.lr_opacity = result["lr-opacity"].as<gsx_float_t>();
    out.lr_sh0 = result["lr-sh0"].as<gsx_float_t>();
    out.lr_sh1 = result["lr-sh1"].as<gsx_float_t>();
    out.lr_sh2 = result["lr-sh2"].as<gsx_float_t>();
    out.lr_sh3 = result["lr-sh3"].as<gsx_float_t>();
    out.beta1 = result["beta1"].as<gsx_float_t>();
    out.beta2 = result["beta2"].as<gsx_float_t>();
    out.epsilon = result["epsilon"].as<gsx_float_t>();
    out.weight_decay = result["weight-decay"].as<gsx_float_t>();
    out.max_grad = result["max-grad"].as<gsx_float_t>();
    out.override_opacity = result["override-opacity"].as<bool>();
    out.init_opacity = result["init-opacity"].as<gsx_float_t>();
    out.scheduler = result["scheduler"].as<std::string>();
    out.scheduler_initial_lr = result["scheduler-initial-lr"].as<gsx_float_t>();
    out.scheduler_final_lr = result["scheduler-final-lr"].as<gsx_float_t>();
    out.scheduler_delay_steps = result["scheduler-delay-steps"].as<gsx_size_t>();
    out.scheduler_delay_multiplier = result["scheduler-delay-multiplier"].as<gsx_float_t>();
    out.scheduler_decay_begin_step = result["scheduler-decay-begin-step"].as<gsx_size_t>();
    out.scheduler_decay_end_step = result["scheduler-decay-end-step"].as<gsx_size_t>();
    out.enable_adc = result["enable-adc"].as<bool>();
    out.adc_algorithm = parse_adc_algorithm(result["adc-algorithm"].as<std::string>());
    out.adc_scene_scale = result["adc-scene-scale"].as<gsx_float_t>();
    out.adc_pruning_opacity_threshold = result["adc-pruning-opacity-threshold"].as<gsx_float_t>();
    out.adc_opacity_clamp_value = result["adc-opacity-clamp-value"].as<gsx_float_t>();
    out.adc_max_world_scale = result["adc-max-world-scale"].as<gsx_float_t>();
    out.adc_max_screen_scale = result["adc-max-screen-scale"].as<gsx_float_t>();
    out.adc_duplicate_grad_threshold = result["adc-duplicate-grad-threshold"].as<gsx_float_t>();
    out.adc_duplicate_scale_threshold = result["adc-duplicate-scale-threshold"].as<gsx_float_t>();
    out.adc_refine_every = result["adc-refine-every"].as<gsx_index_t>();
    out.adc_start_refine = result["adc-start-refine"].as<gsx_index_t>();
    out.adc_end_refine = result["adc-end-refine"].as<gsx_index_t>();
    out.adc_max_num_gaussians = result["adc-max-num-gaussians"].as<gsx_index_t>();
    out.adc_reset_every = result["adc-reset-every"].as<gsx_index_t>();
    out.adc_prune_degenerate_rotation = result["adc-prune-degenerate-rotation"].as<bool>();
    out.adc_duplicate_absgrad_threshold = result["adc-duplicate-absgrad-threshold"].as<gsx_float_t>();
    out.adc_noise_strength = result["adc-noise-strength"].as<gsx_float_t>();
    out.adc_grow_ratio = result["adc-grow-ratio"].as<gsx_float_t>();
    out.adc_loss_threshold = result["adc-loss-threshold"].as<gsx_float_t>();
    out.adc_max_sampled_cameras = result["adc-max-sampled-cameras"].as<gsx_index_t>();
    out.adc_importance_threshold = result["adc-importance-threshold"].as<gsx_float_t>();
    out.adc_prune_budget_ratio = result["adc-prune-budget-ratio"].as<gsx_float_t>();
    out.flann_num_neighbors = result["flann-num-neighbors"].as<gsx_index_t>();
    out.flann_init_scaling = result["flann-init-scaling"].as<gsx_float_t>();
    out.flann_min_distance = result["flann-min-distance"].as<gsx_float_t>();
    out.flann_max_distance = result["flann-max-distance"].as<gsx_float_t>();
    out.flann_default_distance = result["flann-default-distance"].as<gsx_float_t>();
    out.flann_radius = result["flann-radius"].as<gsx_float_t>();
    out.flann_use_anisotropic = result["flann-use-anisotropic"].as<bool>();

    if(out.max_input_width < 0) {
        throw app_error("--max-input-width must be >= 0");
    }
    if(out.steps <= 0) {
        throw app_error("--steps must be > 0");
    }
    if(out.log_interval <= 0) {
        throw app_error("--log-interval must be > 0");
    }
    if(out.eval_interval <= 0) {
        throw app_error("--eval-interval must be > 0");
    }
    if(out.sh_degree < 0 || out.sh_degree > 3) {
        throw app_error("--sh-degree must be in [0, 3]");
    }
    if(out.initial_sh_degree < 0 || out.initial_sh_degree > out.sh_degree) {
        throw app_error("--initial-sh-degree must be in [0, --sh-degree]");
    }
    if(out.near_plane <= 0.0f || out.far_plane <= out.near_plane) {
        throw app_error("invalid near/far plane values");
    }
    if(out.l1_scale == 0.0f && out.ssim_scale == 0.0f) {
        throw app_error("at least one loss weight must be non-zero");
    }
    if(out.render_precision != GSX_RENDER_PRECISION_FLOAT32) {
        throw app_error("train-preprocessed currently supports only --render-precision float32");
    }
    if(out.scheduler != "constant" && out.scheduler != "delayed_exponential") {
        throw app_error("--scheduler must be constant or delayed_exponential");
    }
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
        throw app_error("train and val intrinsics do not match after width cap");
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
    out_sample->rgb.data = sample.rgb_hwc.data();
    out_sample->rgb.data_type = GSX_DATA_TYPE_F32;
    out_sample->rgb.width = sample.intrinsics.width;
    out_sample->rgb.height = sample.intrinsics.height;
    out_sample->rgb.channel_count = 3;
    out_sample->rgb.row_stride_bytes = static_cast<gsx_size_t>(sample.intrinsics.width) * 3u * sizeof(float);
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

    explicit handle_owner(FreeFn fn)
        : free_fn(fn)
    {
    }

    ~handle_owner()
    {
        if(handle != nullptr) {
            (void)free_fn(handle);
            handle = nullptr;
        }
    }

    handle_owner(const handle_owner &) = delete;
    handle_owner &operator=(const handle_owner &) = delete;

    handle_owner(handle_owner &&other) noexcept
        : handle(other.handle)
        , free_fn(other.free_fn)
    {
        other.handle = nullptr;
    }

    handle_owner &operator=(handle_owner &&other) noexcept
    {
        if(this != &other) {
            if(handle != nullptr) {
                (void)free_fn(handle);
            }
            handle = other.handle;
            free_fn = other.free_fn;
            other.handle = nullptr;
        }
        return *this;
    }
};

using backend_owner = handle_owner<gsx_backend_t, decltype(&gsx_backend_free)>;
using dataset_owner = handle_owner<gsx_dataset_t, decltype(&gsx_dataset_free)>;
using dataloader_owner = handle_owner<gsx_dataloader_t, decltype(&gsx_dataloader_free)>;
using renderer_owner = handle_owner<gsx_renderer_t, decltype(&gsx_renderer_free)>;
using render_context_owner = handle_owner<gsx_render_context_t, decltype(&gsx_render_context_free)>;
using gs_owner = handle_owner<gsx_gs_t, decltype(&gsx_gs_free)>;
using loss_owner = handle_owner<gsx_loss_t, decltype(&gsx_loss_free)>;
using loss_context_owner = handle_owner<gsx_loss_context_t, decltype(&gsx_loss_context_free)>;
using optim_owner = handle_owner<gsx_optim_t, decltype(&gsx_optim_free)>;
using scheduler_owner = handle_owner<gsx_scheduler_t, decltype(&gsx_scheduler_free)>;
using adc_owner = handle_owner<gsx_adc_t, decltype(&gsx_adc_free)>;
using session_owner = handle_owner<gsx_session_t, decltype(&gsx_session_free)>;
using arena_owner = handle_owner<gsx_arena_t, decltype(&gsx_arena_free)>;
using tensor_owner = handle_owner<gsx_tensor_t, decltype(&gsx_tensor_free)>;

static gsx_backend_t create_backend(const app_options &options)
{
    gsx_backend_device_t device = nullptr;
    gsx_backend_desc desc{};
    gsx_ok(gsx_backend_registry_init(), "gsx_backend_registry_init");
    gsx_ok(gsx_get_backend_device_by_type(options.backend_type, options.device_index, &device), "gsx_get_backend_device_by_type");
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
    (void)name;
    desc.object = view;
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
    const loaded_split &split,
    bool shuffle_each_epoch,
    gsx_size_t seed)
{
    gsx_dataloader_t dataloader = nullptr;
    gsx_dataloader_desc desc{};
    desc.shuffle_each_epoch = shuffle_each_epoch;
    desc.enable_async_prefetch = false;
    desc.prefetch_count = 0;
    desc.seed = seed;
    desc.image_data_type = GSX_DATA_TYPE_F32;
    desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    desc.output_width = split.intrinsics.width;
    desc.output_height = split.intrinsics.height;
    desc.resize_policy = GSX_IMAGE_RESIZE_PIXEL_CENTER;
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
    gsx_ok(gsx_gs_get_field(gs, parameter_field, &parameter), "gsx_gs_get_field");
    gsx_ok(gsx_gs_get_field(gs, gradient_field, &gradient), "gsx_gs_get_field");

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
    gsx_scheduler_desc desc{};
    if(options.scheduler == "constant") {
        desc.algorithm = GSX_SCHEDULER_ALGORITHM_CONSTANT;
    } else if(options.scheduler == "delayed_exponential") {
        desc.algorithm = GSX_SCHEDULER_ALGORITHM_DELAYED_EXPONENTIAL;
    } else {
        return std::nullopt;
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

static gsx_adc_desc make_adc_desc(const app_options &options)
{
    gsx_adc_desc desc{};
    desc.algorithm = options.adc_algorithm;
    desc.pruning_opacity_threshold = options.adc_pruning_opacity_threshold;
    desc.opacity_clamp_value = options.adc_opacity_clamp_value;
    desc.max_world_scale = options.adc_max_world_scale;
    desc.max_screen_scale = options.adc_max_screen_scale;
    desc.duplicate_grad_threshold = options.adc_duplicate_grad_threshold;
    desc.duplicate_scale_threshold = options.adc_duplicate_scale_threshold;
    desc.refine_every = options.adc_refine_every;
    desc.start_refine = options.adc_start_refine;
    desc.end_refine = options.adc_end_refine;
    desc.max_num_gaussians = options.adc_max_num_gaussians;
    desc.reset_every = options.adc_reset_every;
    desc.seed = options.seed;
    desc.prune_degenerate_rotation = options.adc_prune_degenerate_rotation;
    desc.duplicate_absgrad_threshold = options.adc_duplicate_absgrad_threshold;
    desc.noise_strength = options.adc_noise_strength;
    desc.grow_ratio = options.adc_grow_ratio;
    desc.loss_threshold = options.adc_loss_threshold;
    desc.max_sampled_cameras = options.adc_max_sampled_cameras;
    desc.importance_threshold = options.adc_importance_threshold;
    desc.prune_budget_ratio = options.adc_prune_budget_ratio;
    return desc;
}

static gsx_gs_t create_initialized_gs(
    gsx_backend_buffer_type_t buffer_type,
    const app_options &options,
    const fs::path &points3d_path,
    std::optional<gsx_adc_t> adc)
{
    gsx_gs_desc desc{};
    desc.buffer_type = buffer_type;
    desc.arena_desc.initial_capacity_bytes = 1u << 20;
    desc.count = 1;
    desc.aux_flags = sh_degree_to_aux_flags(options.sh_degree);
    if(adc.has_value()) {
        gsx_gs_aux_flags adc_aux = GSX_GS_AUX_NONE;
        gsx_ok(gsx_adc_get_gs_aux_fields(*adc, &adc_aux), "gsx_adc_get_gs_aux_fields");
        desc.aux_flags |= adc_aux;
    }

    gsx_gs_t gs = nullptr;
    gsx_ok(gsx_gs_init(&gs, &desc), "gsx_gs_init");
    gsx_ok(gsx_read_ply(&gs, path_string(points3d_path).c_str()), "gsx_read_ply");
    gsx_ok(
        gsx_gs_recompute_scale_rotation_flann(
            gs,
            options.flann_num_neighbors,
            options.flann_init_scaling,
            options.flann_min_distance,
            options.flann_max_distance,
            options.flann_default_distance,
            options.flann_radius,
            options.flann_use_anisotropic),
        "gsx_gs_recompute_scale_rotation_flann");

    if(options.override_opacity) {
        gsx_tensor_t opacity = nullptr;
        gsx_tensor_info info{};
        gsx_ok(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &opacity), "gsx_gs_get_field");
        gsx_ok(gsx_tensor_get_info(opacity, &info), "gsx_tensor_get_info");
        const size_t value_count = static_cast<size_t>(info.size_bytes / sizeof(float));
        std::vector<float> values(value_count, clamped_logit(options.init_opacity));
        gsx_ok(gsx_tensor_upload(opacity, values.data(), info.size_bytes), "gsx_tensor_upload");
    }

    return gs;
}

static gsx_session_t create_session(
    gsx_backend_t backend,
    gsx_gs_t gs,
    gsx_optim_t optim,
    gsx_renderer_t renderer,
    gsx_dataloader_t train_dataloader,
    const std::vector<gsx_loss_item> &loss_items,
    const app_options &options,
    gsx_index_t current_sh_degree,
    gsx_size_t initial_global_step,
    gsx_size_t initial_epoch_index,
    std::optional<gsx_scheduler_t> scheduler,
    std::optional<gsx_adc_t> adc)
{
    gsx_session_desc desc{};
    desc.backend = backend;
    desc.gs = gs;
    desc.optim = optim;
    desc.renderer = renderer;
    desc.train_dataloader = train_dataloader;
    desc.adc = adc.value_or(nullptr);
    desc.scheduler = scheduler.value_or(nullptr);
    desc.loss_count = static_cast<gsx_size_t>(loss_items.size());
    desc.loss_items = loss_items.data();
    desc.render.near_plane = options.near_plane;
    desc.render.far_plane = options.far_plane;
    desc.render.background_color = options.background_color;
    desc.render.precision = options.render_precision;
    desc.render.sh_degree_mode = GSX_SESSION_SH_DEGREE_MODE_EXPLICIT;
    desc.render.sh_degree = current_sh_degree;
    desc.render.borrow_train_state = false;
    desc.optim_step.force_all = true;
    desc.adc_step.enabled = options.enable_adc;
    desc.adc_step.dataloader = nullptr;
    desc.adc_step.scene_scale = options.adc_scene_scale;
    desc.workspace.buffer_type_class = options.buffer_type_class;
    desc.workspace.auto_plan = true;
    desc.reporting.retain_prediction = true;
    desc.reporting.retain_target = true;
    desc.reporting.retain_loss_map = true;
    desc.reporting.retain_grad_prediction = false;
    desc.reporting.collect_timings = true;
    desc.initial_global_step = initial_global_step;
    desc.initial_epoch_index = initial_epoch_index;

    gsx_session_t session = nullptr;
    gsx_ok(gsx_session_init(&session, &desc), "gsx_session_init");
    return session;
}

static arena_owner create_eval_arena(gsx_backend_buffer_type_t buffer_type)
{
    arena_owner owner(&gsx_arena_free);
    gsx_arena_desc desc{};
    desc.initial_capacity_bytes = 1u << 20;
    gsx_ok(gsx_arena_init(&owner.handle, buffer_type, &desc), "gsx_arena_init");
    return owner;
}

static tensor_owner create_image_tensor(gsx_arena_t arena, gsx_index_t width, gsx_index_t height)
{
    tensor_owner owner(&gsx_tensor_free);
    gsx_tensor_desc desc{};
    desc.rank = 3;
    desc.shape[0] = 3;
    desc.shape[1] = height;
    desc.shape[2] = width;
    desc.data_type = GSX_DATA_TYPE_F32;
    desc.storage_format = GSX_STORAGE_FORMAT_CHW;
    desc.arena = arena;
    gsx_ok(gsx_tensor_init(&owner.handle, &desc), "gsx_tensor_init");
    return owner;
}

static gsx_index_t compute_current_sh_degree(const app_options &options, gsx_size_t global_step)
{
    if(options.sh_degree_interval <= 0) {
        return options.sh_degree;
    }
    const gsx_size_t increments = global_step / static_cast<gsx_size_t>(options.sh_degree_interval);
    const gsx_size_t candidate = static_cast<gsx_size_t>(options.initial_sh_degree) + increments;
    return static_cast<gsx_index_t>(std::min<gsx_size_t>(static_cast<gsx_size_t>(options.sh_degree), candidate));
}

static std::vector<float> download_tensor_f32(gsx_tensor_t tensor)
{
    gsx_tensor_info info{};
    gsx_ok(gsx_tensor_get_info(tensor, &info), "gsx_tensor_get_info");
    if(info.data_type != GSX_DATA_TYPE_F32) {
        throw app_error("tensor download expects float32 tensor");
    }
    std::vector<float> values(static_cast<size_t>(info.size_bytes / sizeof(float)));
    gsx_ok(gsx_tensor_download(tensor, values.data(), info.size_bytes), "gsx_tensor_download");
    return values;
}

static double compute_psnr_from_chw(const std::vector<float> &prediction, const std::vector<float> &target)
{
    if(prediction.size() != target.size() || prediction.empty()) {
        throw app_error("prediction/target size mismatch while computing PSNR");
    }

    double mse = 0.0;
    for(size_t i = 0; i < prediction.size(); ++i) {
        const double diff = static_cast<double>(prediction[i]) - static_cast<double>(target[i]);
        mse += diff * diff;
    }
    mse /= static_cast<double>(prediction.size());
    if(mse <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return -10.0 * std::log10(mse);
}

static void write_png_from_chw(const fs::path &path, const std::vector<float> &chw, gsx_index_t width, gsx_index_t height)
{
    fs::create_directories(path.parent_path());
    gsx_ok(
        gsx_image_write_png(path_string(path).c_str(), chw.data(), width, height, 3, GSX_DATA_TYPE_F32, GSX_STORAGE_FORMAT_CHW),
        "gsx_image_write_png");
}

static double compute_mean(const std::vector<float> &values)
{
    if(values.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for(float value : values) {
        sum += static_cast<double>(value);
    }
    return sum / static_cast<double>(values.size());
}

static std::string format_step_dir(gsx_size_t global_step)
{
    std::ostringstream oss;
    oss << "step_" << std::setw(6) << std::setfill('0') << global_step;
    return oss.str();
}

static eval_summary run_evaluation(
    gsx_gs_t gs,
    gsx_renderer_t renderer,
    const loaded_split &split,
    gsx_dataloader_t eval_dataloader,
    const app_options &options,
    gsx_index_t current_sh_degree,
    gsx_backend_buffer_type_t buffer_type,
    gsx_size_t global_step)
{
    render_context_owner render_context(&gsx_render_context_free);
    arena_owner eval_arena = create_eval_arena(buffer_type);
    tensor_owner prediction_tensor = create_image_tensor(eval_arena.handle, split.intrinsics.width, split.intrinsics.height);
    std::vector<eval_image_metric> per_image;
    std::vector<double> psnrs;

    gsx_tensor_t gs_mean3d = nullptr;
    gsx_tensor_t gs_rotation = nullptr;
    gsx_tensor_t gs_logscale = nullptr;
    gsx_tensor_t gs_opacity = nullptr;
    gsx_tensor_t gs_sh0 = nullptr;
    gsx_tensor_t gs_sh1 = nullptr;
    gsx_tensor_t gs_sh2 = nullptr;
    gsx_tensor_t gs_sh3 = nullptr;

    gsx_ok(gsx_render_context_init(&render_context.handle, renderer), "gsx_render_context_init");
    gsx_ok(gsx_dataloader_reset(eval_dataloader), "gsx_dataloader_reset");
    gsx_ok(gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &gs_mean3d), "gsx_gs_get_field");
    gsx_ok(gsx_gs_get_field(gs, GSX_GS_FIELD_ROTATION, &gs_rotation), "gsx_gs_get_field");
    gsx_ok(gsx_gs_get_field(gs, GSX_GS_FIELD_LOGSCALE, &gs_logscale), "gsx_gs_get_field");
    gsx_ok(gsx_gs_get_field(gs, GSX_GS_FIELD_OPACITY, &gs_opacity), "gsx_gs_get_field");
    gsx_ok(gsx_gs_get_field(gs, GSX_GS_FIELD_SH0, &gs_sh0), "gsx_gs_get_field");
    if(current_sh_degree >= 1) {
        gsx_ok(gsx_gs_get_field(gs, GSX_GS_FIELD_SH1, &gs_sh1), "gsx_gs_get_field");
    }
    if(current_sh_degree >= 2) {
        gsx_ok(gsx_gs_get_field(gs, GSX_GS_FIELD_SH2, &gs_sh2), "gsx_gs_get_field");
    }
    if(current_sh_degree >= 3) {
        gsx_ok(gsx_gs_get_field(gs, GSX_GS_FIELD_SH3, &gs_sh3), "gsx_gs_get_field");
    }

    for(size_t i = 0; i < split.samples.size(); ++i) {
        gsx_dataloader_result batch{};
        gsx_render_forward_request request{};
        const loaded_sample &sample = split.samples[i];
        std::vector<float> prediction;
        std::vector<float> target;
        fs::path render_path;

        gsx_ok(gsx_dataloader_next_ex(eval_dataloader, &batch), "gsx_dataloader_next_ex");
        request.intrinsics = &batch.intrinsics;
        request.pose = &batch.pose;
        request.near_plane = options.near_plane;
        request.far_plane = options.far_plane;
        request.background_color = options.background_color;
        request.precision = options.render_precision;
        request.sh_degree = current_sh_degree;
        request.forward_type = GSX_RENDER_FORWARD_TYPE_INFERENCE;
        request.borrow_train_state = false;
        request.gs_mean3d = gs_mean3d;
        request.gs_rotation = gs_rotation;
        request.gs_logscale = gs_logscale;
        request.gs_sh0 = gs_sh0;
        request.gs_sh1 = gs_sh1;
        request.gs_sh2 = gs_sh2;
        request.gs_sh3 = gs_sh3;
        request.gs_opacity = gs_opacity;
        request.out_rgb = prediction_tensor.handle;

        gsx_ok(gsx_renderer_render(renderer, render_context.handle, &request), "gsx_renderer_render");
        prediction = download_tensor_f32(prediction_tensor.handle);
        target = download_tensor_f32(batch.rgb_image);

        eval_image_metric metric;
        metric.image_id = static_cast<gsx_id_t>(sample.pose.frame_id);
        metric.name = sample.name;
        metric.width = batch.intrinsics.width;
        metric.height = batch.intrinsics.height;
        metric.psnr = compute_psnr_from_chw(prediction, target);

        if(options.save_val_renders) {
            render_path = options.output_dir / "eval" / format_step_dir(global_step) / "images" / sample.name;
            render_path.replace_extension(".png");
            write_png_from_chw(render_path, prediction, batch.intrinsics.width, batch.intrinsics.height);
            metric.render_path = fs::relative(render_path, options.output_dir).string();
        }

        per_image.push_back(metric);
        psnrs.push_back(metric.psnr);
    }

    const double psnr_sum = std::accumulate(psnrs.begin(), psnrs.end(), 0.0);
    const double psnr_mean = psnrs.empty() ? 0.0 : (psnr_sum / static_cast<double>(psnrs.size()));
    const double psnr_min = psnrs.empty() ? 0.0 : *std::min_element(psnrs.begin(), psnrs.end());
    const double psnr_max = psnrs.empty() ? 0.0 : *std::max_element(psnrs.begin(), psnrs.end());

    json metrics_json;
    metrics_json["split"] = split.name;
    metrics_json["global_step"] = global_step;
    metrics_json["image_count"] = per_image.size();
    metrics_json["aggregate"] = {
        { "psnr_mean", psnr_mean },
        { "psnr_min", psnr_min },
        { "psnr_max", psnr_max },
    };
    metrics_json["per_image"] = json::array();
    for(const eval_image_metric &metric : per_image) {
        metrics_json["per_image"].push_back({
            { "image_id", metric.image_id },
            { "name", metric.name },
            { "width", metric.width },
            { "height", metric.height },
            { "psnr", metric.psnr },
            { "render_path", metric.render_path },
        });
    }

    const fs::path metrics_path = options.output_dir / "eval" / format_step_dir(global_step) / "metrics.json";
    fs::create_directories(metrics_path.parent_path());
    std::ofstream output(metrics_path);
    if(!output) {
        throw app_error("failed to open metrics output file: " + path_string(metrics_path));
    }
    output << std::setw(2) << metrics_json << "\n";

    eval_summary summary;
    summary.global_step = global_step;
    summary.psnr_mean = psnr_mean;
    summary.psnr_min = psnr_min;
    summary.psnr_max = psnr_max;
    summary.metrics_json = std::move(metrics_json);
    return summary;
}

static void save_train_render(
    gsx_session_t session,
    const loaded_split &train_split,
    const app_options &options,
    const gsx_session_step_report &report)
{
    if(!options.save_train_renders) {
        return;
    }

    gsx_session_outputs outputs{};
    gsx_ok(gsx_session_get_last_outputs(session, &outputs), "gsx_session_get_last_outputs");
    if(outputs.prediction == nullptr || report.stable_sample_index >= train_split.samples.size()) {
        return;
    }

    const loaded_sample &sample = train_split.samples[static_cast<size_t>(report.stable_sample_index)];
    const std::vector<float> prediction = download_tensor_f32(outputs.prediction);
    fs::path render_path = options.output_dir / "train_renders" / format_step_dir(report.global_step_after) / sample.name;
    render_path.replace_extension(".png");
    write_png_from_chw(render_path, prediction, sample.intrinsics.width, sample.intrinsics.height);
}

static json build_summary_json(
    const app_options &options,
    const loaded_split &train_split,
    const loaded_split &val_split,
    const std::vector<eval_summary> &eval_runs)
{
    json summary;
    summary["dataset_root"] = path_string(options.dataset_root);
    summary["backend"] = backend_name(options.backend_type);
    summary["device_index"] = options.device_index;
    summary["buffer_type"] = buffer_type_name(options.buffer_type_class);
    summary["steps"] = options.steps;
    summary["max_input_width"] = options.max_input_width;
    summary["train_image_count"] = train_split.samples.size();
    summary["val_image_count"] = val_split.samples.size();
    summary["train_resolution"] = { train_split.intrinsics.width, train_split.intrinsics.height };
    summary["val_resolution"] = { val_split.intrinsics.width, val_split.intrinsics.height };
    summary["eval_runs"] = json::array();

    const eval_summary *best = nullptr;
    for(const eval_summary &run : eval_runs) {
        summary["eval_runs"].push_back({
            { "global_step", run.global_step },
            { "psnr_mean", run.psnr_mean },
            { "psnr_min", run.psnr_min },
            { "psnr_max", run.psnr_max },
        });
        if(best == nullptr || run.psnr_mean > best->psnr_mean) {
            best = &run;
        }
    }

    if(!eval_runs.empty()) {
        const eval_summary &final_run = eval_runs.back();
        summary["final_eval"] = {
            { "global_step", final_run.global_step },
            { "psnr_mean", final_run.psnr_mean },
            { "psnr_min", final_run.psnr_min },
            { "psnr_max", final_run.psnr_max },
        };
    }
    if(best != nullptr) {
        summary["best_eval"] = {
            { "global_step", best->global_step },
            { "psnr_mean", best->psnr_mean },
            { "psnr_min", best->psnr_min },
            { "psnr_max", best->psnr_max },
        };
    }
    return summary;
}

} // namespace

int main(int argc, char **argv)
{
    try {
        const app_options options = parse_options(argc, argv);
        const loaded_split train_split = load_split(options.dataset_root, "train", options.max_input_width);
        const loaded_split val_split = load_split(options.dataset_root, "val", options.max_input_width);
        const fs::path points3d_path = options.dataset_root / "train" / "points3d.ply";
        dataset_view train_view{ &train_split };
        dataset_view val_view{ &val_split };
        backend_owner backend(&gsx_backend_free);
        dataset_owner train_dataset(&gsx_dataset_free);
        dataset_owner val_dataset(&gsx_dataset_free);
        dataloader_owner train_dataloader(&gsx_dataloader_free);
        dataloader_owner val_dataloader(&gsx_dataloader_free);
        renderer_owner renderer(&gsx_renderer_free);
        gs_owner gs(&gsx_gs_free);
        optim_owner optim(&gsx_optim_free);
        scheduler_owner scheduler(&gsx_scheduler_free);
        adc_owner adc(&gsx_adc_free);
        session_owner session(&gsx_session_free);
        std::vector<loss_owner> losses;
        std::vector<loss_context_owner> loss_contexts;
        std::vector<gsx_loss_item> loss_items;
        std::vector<eval_summary> eval_runs;
        gsx_index_t current_sh_degree = compute_current_sh_degree(options, 0);
        gsx_size_t session_initial_global_step = 0;
        gsx_size_t session_initial_epoch_index = 0;

        if(!fs::exists(points3d_path)) {
            throw app_error("missing points3d file: " + path_string(points3d_path));
        }
        validate_matching_intrinsics(train_split.intrinsics, val_split.intrinsics);
        fs::create_directories(options.output_dir);

        backend.handle = create_backend(options);
        const gsx_backend_buffer_type_t selected_buffer_type = find_buffer_type(backend.handle, options.buffer_type_class);

        train_dataset.handle = create_dataset(&train_view, "train");
        val_dataset.handle = create_dataset(&val_view, "val");
        train_dataloader.handle = create_dataloader(backend.handle, train_dataset.handle, train_split, options.shuffle_each_epoch, options.seed);
        val_dataloader.handle = create_dataloader(backend.handle, val_dataset.handle, val_split, false, options.seed);
        renderer.handle = create_renderer(backend.handle, train_split);

        if(options.enable_adc) {
            const gsx_adc_desc adc_desc = make_adc_desc(options);
            gsx_ok(gsx_adc_init(&adc.handle, backend.handle, &adc_desc), "gsx_adc_init");
        }

        gs.handle = create_initialized_gs(selected_buffer_type, options, points3d_path, adc.handle != nullptr ? std::optional<gsx_adc_t>(adc.handle) : std::nullopt);
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
            train_dataloader.handle,
            loss_items,
            options,
            current_sh_degree,
            session_initial_global_step,
            session_initial_epoch_index,
            scheduler.handle != nullptr ? std::optional<gsx_scheduler_t>(scheduler.handle) : std::nullopt,
            adc.handle != nullptr ? std::optional<gsx_adc_t>(adc.handle) : std::nullopt);

        std::cout << "train-preprocessed: backend=" << backend_name(options.backend_type)
                  << " device=" << options.device_index
                          << " train_images=" << train_split.samples.size()
                          << " val_images=" << val_split.samples.size()
                          << " resolution=" << train_split.intrinsics.width << "x" << train_split.intrinsics.height
                          << " sh_degree=" << current_sh_degree << "/" << options.sh_degree
                          << " steps=" << options.steps << "\n";

        for(gsx_index_t step = 0; step < options.steps; ++step) {
            gsx_session_step_report report{};
            gsx_session_outputs outputs{};
            double train_loss = 0.0;

            gsx_ok(gsx_session_step(session.handle), "gsx_session_step");
            gsx_ok(gsx_session_get_last_step_report(session.handle, &report), "gsx_session_get_last_step_report");

            if((step + 1) % options.log_interval == 0 || step + 1 == options.steps) {
                gsx_ok(gsx_session_get_last_outputs(session.handle, &outputs), "gsx_session_get_last_outputs");
                if(outputs.loss_map != nullptr) {
                    const std::vector<float> loss_map = download_tensor_f32(outputs.loss_map);
                    train_loss = compute_mean(loss_map);
                }
                std::cout << "step=" << report.global_step_after
                          << " sample=" << report.stable_sample_index
                          << " sh_degree=" << current_sh_degree
                          << " loss=" << train_loss;
                if(report.has_applied_learning_rate) {
                    std::cout << " lr=" << report.applied_learning_rate;
                }
                if(report.has_timings) {
                    std::cout << " total_us=" << report.timings.total_step_us;
                }
                std::cout << "\n";
                save_train_render(session.handle, train_split, options, report);
            }

            {
                const gsx_index_t next_sh_degree = compute_current_sh_degree(options, report.global_step_after);
                if(next_sh_degree != current_sh_degree) {
                    gsx_session_state session_state{};
                    gsx_ok(gsx_session_get_state(session.handle, &session_state), "gsx_session_get_state");
                    current_sh_degree = next_sh_degree;
                    session_owner new_session(&gsx_session_free);
                    new_session.handle = create_session(
                        backend.handle,
                        gs.handle,
                        optim.handle,
                        renderer.handle,
                        train_dataloader.handle,
                        loss_items,
                        options,
                        current_sh_degree,
                        session_state.global_step,
                        session_state.epoch_index,
                        scheduler.handle != nullptr ? std::optional<gsx_scheduler_t>(scheduler.handle) : std::nullopt,
                        adc.handle != nullptr ? std::optional<gsx_adc_t>(adc.handle) : std::nullopt);
                    session = std::move(new_session);
                    std::cout << "sh_degree increased to " << current_sh_degree
                              << " at step=" << report.global_step_after << "\n";
                }
            }

            if((report.global_step_after % static_cast<gsx_size_t>(options.eval_interval) == 0u)
                || report.global_step_after == static_cast<gsx_size_t>(options.steps)) {
                const eval_summary summary = run_evaluation(
                    gs.handle,
                    renderer.handle,
                    val_split,
                    val_dataloader.handle,
                    options,
                    current_sh_degree,
                    selected_buffer_type,
                    report.global_step_after);
                std::cout << "eval step=" << summary.global_step
                          << " psnr_mean=" << summary.psnr_mean
                          << " psnr_min=" << summary.psnr_min
                          << " psnr_max=" << summary.psnr_max << "\n";
                eval_runs.push_back(summary);
            }
        }

        const json summary_json = build_summary_json(options, train_split, val_split, eval_runs);
        {
            std::ofstream summary_file(options.output_dir / "summary.json");
            if(!summary_file) {
                throw app_error("failed to write summary.json");
            }
            summary_file << std::setw(2) << summary_json << "\n";
        }

        gsx_ok(gsx_backend_major_stream_sync(backend.handle), "gsx_backend_major_stream_sync");
        return 0;
    } catch(const std::exception &e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
