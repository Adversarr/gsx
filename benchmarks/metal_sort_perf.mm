extern "C" {
#include "../gsx/src/gsx-impl.h"
#include "../gsx/src/gsx-metal/internal.h"
}

#include "gsx/gsx.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kRadixBits = 6u;
constexpr uint32_t kRadixSize = 1u << kRadixBits;

enum class InputDistribution {
    UniformRandom,
    AlreadySorted,
    ReverseSorted,
    SkewedDigits
};

struct SortBenchmarkOptions {
    uint32_t count = 1u << 20;
    uint32_t warmup_iterations = 5u;
    uint32_t timed_iterations = 20u;
    uint32_t significant_bits = 32u;
};

struct SortBenchmarkContext {
    gsx_backend_t backend = nullptr;
    gsx_backend_buffer_type_t device_buffer_type = nullptr;
    gsx_backend_buffer_t keys_in = nullptr;
    gsx_backend_buffer_t values_in = nullptr;
    gsx_backend_buffer_t keys_out = nullptr;
    gsx_backend_buffer_t values_out = nullptr;
    gsx_backend_buffer_t histogram = nullptr;
    gsx_backend_buffer_t global_histogram = nullptr;
    gsx_backend_buffer_t scatter_offsets = nullptr;
    gsx_backend_tensor_view keys_in_view{};
    gsx_backend_tensor_view values_in_view{};
    gsx_backend_tensor_view keys_out_view{};
    gsx_backend_tensor_view values_out_view{};
    gsx_backend_tensor_view histogram_view{};
    gsx_backend_tensor_view global_histogram_view{};
    gsx_backend_tensor_view scatter_offsets_view{};
    uint32_t count = 0u;
    uint32_t significant_bits = 0u;
    uint32_t pass_count = 0u;
    uint32_t num_threadgroups = 0u;
};

struct SortProfileAggregate {
    double histogram_ns = 0.0;
    double prefix_offsets_ns = 0.0;
    double scatter_ns = 0.0;
    double total_ns = 0.0;
};

struct SortProfileRow {
    const char *label = nullptr;
    double histogram_ms = 0.0;
    double prefix_offsets_ms = 0.0;
    double scatter_ms = 0.0;
    double total_ms = 0.0;
    double scatter_share = 0.0;
    bool scatter_dominates = false;
};

struct SortReferencePair {
    uint32_t key = 0u;
    uint32_t value = 0u;
};

static bool gsx_check(gsx_error error, const char *what)
{
    if(gsx_error_is_success(error)) {
        return true;
    }

    std::fprintf(stderr, "%s failed: %s\n", what, error.message != nullptr ? error.message : "(no message)");
    return false;
}

static void gsx_cleanup_context(SortBenchmarkContext *ctx)
{
    if(ctx == nullptr) {
        return;
    }

    if(ctx->scatter_offsets != nullptr) {
        (void)gsx_backend_buffer_free(ctx->scatter_offsets);
        ctx->scatter_offsets = nullptr;
    }
    if(ctx->global_histogram != nullptr) {
        (void)gsx_backend_buffer_free(ctx->global_histogram);
        ctx->global_histogram = nullptr;
    }
    if(ctx->histogram != nullptr) {
        (void)gsx_backend_buffer_free(ctx->histogram);
        ctx->histogram = nullptr;
    }
    if(ctx->values_out != nullptr) {
        (void)gsx_backend_buffer_free(ctx->values_out);
        ctx->values_out = nullptr;
    }
    if(ctx->keys_out != nullptr) {
        (void)gsx_backend_buffer_free(ctx->keys_out);
        ctx->keys_out = nullptr;
    }
    if(ctx->values_in != nullptr) {
        (void)gsx_backend_buffer_free(ctx->values_in);
        ctx->values_in = nullptr;
    }
    if(ctx->keys_in != nullptr) {
        (void)gsx_backend_buffer_free(ctx->keys_in);
        ctx->keys_in = nullptr;
    }
    if(ctx->backend != nullptr) {
        (void)gsx_backend_free(ctx->backend);
        ctx->backend = nullptr;
    }
}

static bool gsx_make_buffer_view(gsx_backend_buffer_t buffer, gsx_data_type data_type, gsx_backend_tensor_view *out_view)
{
    gsx_backend_buffer_info info{};

    if(buffer == nullptr || out_view == nullptr) {
        return false;
    }
    if(!gsx_check(gsx_backend_buffer_get_info(buffer, &info), "gsx_backend_buffer_get_info")) {
        return false;
    }

    out_view->buffer = buffer;
    out_view->offset_bytes = 0u;
    out_view->size_bytes = info.size_bytes;
    out_view->effective_alignment_bytes = info.alignment_bytes;
    out_view->data_type = data_type;
    return true;
}

static bool gsx_init_buffer(gsx_backend_buffer_type_t buffer_type, gsx_size_t size_bytes, gsx_backend_buffer_t *out_buffer)
{
    gsx_backend_buffer_desc desc{};

    if(buffer_type == nullptr || out_buffer == nullptr) {
        return false;
    }

    desc.buffer_type = buffer_type;
    desc.size_bytes = size_bytes;
    desc.alignment_bytes = 0u;
    return gsx_check(gsx_backend_buffer_init(out_buffer, &desc), "gsx_backend_buffer_init");
}

static bool gsx_init_context(const SortBenchmarkOptions &options, SortBenchmarkContext *out_ctx)
{
    gsx_backend_device_t backend_device = nullptr;
    gsx_backend_desc backend_desc{};
    SortBenchmarkContext ctx{};
    const gsx_size_t pair_bytes = static_cast<gsx_size_t>(options.count) * sizeof(uint32_t);
    const uint32_t num_threadgroups = (options.count + 1023u) / 1024u;
    const gsx_size_t histogram_bytes = static_cast<gsx_size_t>(kRadixSize) * static_cast<gsx_size_t>(num_threadgroups) * sizeof(uint32_t);
    const gsx_size_t global_histogram_bytes = static_cast<gsx_size_t>(kRadixSize) * sizeof(uint32_t);
    const gsx_size_t scatter_offsets_bytes = static_cast<gsx_size_t>(kRadixSize) * static_cast<gsx_size_t>(num_threadgroups) * sizeof(uint32_t);

    if(out_ctx == nullptr) {
        return false;
    }
    if(!gsx_check(gsx_backend_registry_init(), "gsx_backend_registry_init")) {
        return false;
    }
    if(!gsx_check(gsx_get_backend_device_by_type(GSX_BACKEND_TYPE_METAL, 0, &backend_device), "gsx_get_backend_device_by_type")) {
        return false;
    }

    backend_desc.device = backend_device;
    if(!gsx_check(gsx_backend_init(&ctx.backend, &backend_desc), "gsx_backend_init")) {
        return false;
    }
    if(!gsx_check(gsx_backend_find_buffer_type(ctx.backend, GSX_BACKEND_BUFFER_TYPE_DEVICE, &ctx.device_buffer_type), "gsx_backend_find_buffer_type")) {
        gsx_cleanup_context(&ctx);
        return false;
    }

    if(!gsx_init_buffer(ctx.device_buffer_type, pair_bytes, &ctx.keys_in)
        || !gsx_init_buffer(ctx.device_buffer_type, pair_bytes, &ctx.values_in)
        || !gsx_init_buffer(ctx.device_buffer_type, pair_bytes, &ctx.keys_out)
        || !gsx_init_buffer(ctx.device_buffer_type, pair_bytes, &ctx.values_out)
        || !gsx_init_buffer(ctx.device_buffer_type, histogram_bytes, &ctx.histogram)
        || !gsx_init_buffer(ctx.device_buffer_type, global_histogram_bytes, &ctx.global_histogram)
        || !gsx_init_buffer(ctx.device_buffer_type, scatter_offsets_bytes, &ctx.scatter_offsets)) {
        gsx_cleanup_context(&ctx);
        return false;
    }

    if(!gsx_make_buffer_view(ctx.keys_in, GSX_DATA_TYPE_I32, &ctx.keys_in_view)
        || !gsx_make_buffer_view(ctx.values_in, GSX_DATA_TYPE_I32, &ctx.values_in_view)
        || !gsx_make_buffer_view(ctx.keys_out, GSX_DATA_TYPE_I32, &ctx.keys_out_view)
        || !gsx_make_buffer_view(ctx.values_out, GSX_DATA_TYPE_I32, &ctx.values_out_view)
        || !gsx_make_buffer_view(ctx.histogram, GSX_DATA_TYPE_I32, &ctx.histogram_view)
        || !gsx_make_buffer_view(ctx.global_histogram, GSX_DATA_TYPE_I32, &ctx.global_histogram_view)
        || !gsx_make_buffer_view(ctx.scatter_offsets, GSX_DATA_TYPE_I32, &ctx.scatter_offsets_view)) {
        gsx_cleanup_context(&ctx);
        return false;
    }

    ctx.count = options.count;
    ctx.significant_bits = options.significant_bits > 32u ? 32u : options.significant_bits;
    ctx.pass_count = (ctx.significant_bits + kRadixBits - 1u) / kRadixBits;
    ctx.num_threadgroups = num_threadgroups;
    *out_ctx = ctx;
    return true;
}

static bool parse_u32_arg(const char *text, uint32_t *out_value)
{
    char *end = nullptr;
    unsigned long parsed = 0ul;

    if(text == nullptr || out_value == nullptr || text[0] == '\0') {
        return false;
    }

    parsed = std::strtoul(text, &end, 10);
    if(end == nullptr || *end != '\0' || parsed > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    *out_value = static_cast<uint32_t>(parsed);
    return true;
}

static bool parse_options(int argc, char *argv[], SortBenchmarkOptions *out_options)
{
    SortBenchmarkOptions options{};

    if(out_options == nullptr) {
        return false;
    }

    for(int i = 1; i < argc; ++i) {
        const char *arg = argv[i];

        if(std::strcmp(arg, "--count") == 0 && (i + 1) < argc) {
            if(!parse_u32_arg(argv[++i], &options.count) || options.count == 0u) {
                return false;
            }
        } else if(std::strcmp(arg, "--warmup") == 0 && (i + 1) < argc) {
            if(!parse_u32_arg(argv[++i], &options.warmup_iterations)) {
                return false;
            }
        } else if(std::strcmp(arg, "--iters") == 0 && (i + 1) < argc) {
            if(!parse_u32_arg(argv[++i], &options.timed_iterations) || options.timed_iterations == 0u) {
                return false;
            }
        } else if(std::strcmp(arg, "--significant-bits") == 0 && (i + 1) < argc) {
            if(!parse_u32_arg(argv[++i], &options.significant_bits) || options.significant_bits == 0u) {
                return false;
            }
        } else if(std::strcmp(arg, "--help") == 0) {
            std::printf("Usage: metal_sort_perf [--count N] [--warmup N] [--iters N] [--significant-bits N]\n");
            return false;
        } else {
            return false;
        }
    }

    *out_options = options;
    return true;
}

static const char *distribution_label(InputDistribution distribution)
{
    switch(distribution) {
    case InputDistribution::UniformRandom:
        return "uniform_random";
    case InputDistribution::AlreadySorted:
        return "already_sorted";
    case InputDistribution::ReverseSorted:
        return "reverse_sorted";
    case InputDistribution::SkewedDigits:
        return "skewed_digits";
    }

    return "unknown";
}

static std::vector<uint32_t> make_values(uint32_t count)
{
    std::vector<uint32_t> values(count);

    for(uint32_t i = 0u; i < count; ++i) {
        values[static_cast<std::size_t>(i)] = i;
    }
    return values;
}

static uint32_t spread_sorted_key(uint32_t index, uint32_t count)
{
    if(count <= 1u) {
        return 0u;
    }
    return static_cast<uint32_t>((static_cast<uint64_t>(index) * 0xFFFFFFFFull) / static_cast<uint64_t>(count - 1u));
}

static std::vector<uint32_t> make_keys(InputDistribution distribution, uint32_t count)
{
    std::vector<uint32_t> keys(count);
    std::mt19937 rng(0x5A17D3u + static_cast<uint32_t>(distribution) * 17u);
    std::uniform_int_distribution<uint32_t> uniform_dist(0u, 0xFFFFFFFFu);
    std::uniform_int_distribution<uint32_t> digit_dist(1u, kRadixSize - 1u);
    std::bernoulli_distribution heavy_zero_digit(0.95);

    switch(distribution) {
    case InputDistribution::UniformRandom:
        for(uint32_t i = 0u; i < count; ++i) {
            keys[static_cast<std::size_t>(i)] = uniform_dist(rng);
        }
        break;
    case InputDistribution::AlreadySorted:
        for(uint32_t i = 0u; i < count; ++i) {
            keys[static_cast<std::size_t>(i)] = spread_sorted_key(i, count);
        }
        break;
    case InputDistribution::ReverseSorted:
        for(uint32_t i = 0u; i < count; ++i) {
            keys[static_cast<std::size_t>(i)] = spread_sorted_key(count - 1u - i, count);
        }
        break;
    case InputDistribution::SkewedDigits:
        for(uint32_t i = 0u; i < count; ++i) {
            uint32_t key = 0u;

            for(uint32_t shift = 0u; shift < 32u; shift += kRadixBits) {
                const uint32_t remaining_bits = 32u - shift;
                const uint32_t digit_bits = remaining_bits < kRadixBits ? remaining_bits : kRadixBits;
                const uint32_t digit_mask = (1u << digit_bits) - 1u;
                uint32_t digit = 0u;

                if(!heavy_zero_digit(rng)) {
                    digit = digit_dist(rng) & digit_mask;
                }
                key |= digit << shift;
            }
            keys[static_cast<std::size_t>(i)] = key;
        }
        break;
    }

    return keys;
}

static std::vector<SortReferencePair> make_reference_pairs(const std::vector<uint32_t> &keys)
{
    std::vector<SortReferencePair> reference(keys.size());

    for(std::size_t i = 0; i < keys.size(); ++i) {
        reference[i].key = keys[i];
        reference[i].value = static_cast<uint32_t>(i);
    }

    std::stable_sort(
        reference.begin(),
        reference.end(),
        [](const SortReferencePair &lhs, const SortReferencePair &rhs) { return lhs.key < rhs.key; });
    return reference;
}

static bool upload_sort_inputs(const SortBenchmarkContext &ctx, const std::vector<uint32_t> &keys, const std::vector<uint32_t> &values)
{
    const gsx_size_t byte_count = static_cast<gsx_size_t>(keys.size()) * sizeof(uint32_t);

    if(keys.size() != values.size() || keys.size() != ctx.count) {
        return false;
    }

    return gsx_check(gsx_backend_buffer_upload(ctx.keys_in, 0u, keys.data(), byte_count), "gsx_backend_buffer_upload(keys_in)")
        && gsx_check(gsx_backend_buffer_upload(ctx.values_in, 0u, values.data(), byte_count), "gsx_backend_buffer_upload(values_in)");
}

static bool download_sort_outputs(
    const SortBenchmarkContext &ctx,
    std::vector<uint32_t> *out_keys,
    std::vector<uint32_t> *out_values)
{
    const gsx_size_t byte_count = static_cast<gsx_size_t>(ctx.count) * sizeof(uint32_t);

    if(out_keys == nullptr || out_values == nullptr) {
        return false;
    }

    out_keys->assign(ctx.count, 0u);
    out_values->assign(ctx.count, 0u);

    if(!gsx_check(gsx_backend_buffer_download(ctx.keys_out, 0u, out_keys->data(), byte_count), "gsx_backend_buffer_download(keys)")
        || !gsx_check(gsx_backend_buffer_download(ctx.values_out, 0u, out_values->data(), byte_count), "gsx_backend_buffer_download(values)")
        || !gsx_check(gsx_backend_major_stream_sync(ctx.backend), "gsx_backend_major_stream_sync")) {
        return false;
    }

    return true;
}

static bool run_sort_dispatch(const SortBenchmarkContext &ctx, gsx_metal_sort_profile *out_profile)
{
    return gsx_check(
        gsx_metal_backend_dispatch_sort_pairs_u32(
            ctx.backend,
            &ctx.keys_in_view,
            &ctx.values_in_view,
            &ctx.keys_out_view,
            &ctx.values_out_view,
            &ctx.histogram_view,
            &ctx.global_histogram_view,
            &ctx.scatter_offsets_view,
            ctx.count,
            ctx.significant_bits,
            out_profile),
        "gsx_metal_backend_dispatch_sort_pairs_u32");
}

static bool verify_sorted_output(const SortBenchmarkContext &ctx, const std::vector<uint32_t> &input_keys)
{
    std::vector<uint32_t> output_keys;
    std::vector<uint32_t> output_values;
    const std::vector<SortReferencePair> reference = make_reference_pairs(input_keys);

    if(!download_sort_outputs(ctx, &output_keys, &output_values)) {
        return false;
    }
    if(output_keys.size() != reference.size() || output_values.size() != reference.size()) {
        return false;
    }

    for(std::size_t i = 0; i < reference.size(); ++i) {
        if(output_keys[i] != reference[i].key || output_values[i] != reference[i].value) {
            std::fprintf(
                stderr,
                "sort mismatch at index %zu: got key=%" PRIu32 " value=%" PRIu32 ", expected key=%" PRIu32 " value=%" PRIu32 "\n",
                i,
                output_keys[i],
                output_values[i],
                reference[i].key,
                reference[i].value);
            return false;
        }
    }

    return true;
}

static bool measure_distribution(
    const SortBenchmarkContext &ctx,
    const SortBenchmarkOptions &options,
    InputDistribution distribution,
    SortProfileRow *out_row)
{
    const std::vector<uint32_t> keys = make_keys(distribution, ctx.count);
    const std::vector<uint32_t> values = make_values(ctx.count);
    SortProfileAggregate aggregate{};

    if(out_row == nullptr) {
        return false;
    }

    for(uint32_t i = 0u; i < options.warmup_iterations; ++i) {
        @autoreleasepool {
            gsx_metal_sort_profile warmup_profile{};

            if(!upload_sort_inputs(ctx, keys, values) || !run_sort_dispatch(ctx, &warmup_profile)) {
                return false;
            }
        }
    }

    for(uint32_t i = 0u; i < options.timed_iterations; ++i) {
        @autoreleasepool {
            gsx_metal_sort_profile profile{};

            if(!upload_sort_inputs(ctx, keys, values) || !run_sort_dispatch(ctx, &profile)) {
                return false;
            }
            if(!profile.valid) {
                std::fprintf(stderr, "profile output was not populated\n");
                return false;
            }

            aggregate.histogram_ns += profile.histogram_ns;
            aggregate.prefix_offsets_ns += profile.prefix_offsets_ns;
            aggregate.scatter_ns += profile.scatter_ns;
            aggregate.total_ns += profile.total_ns;
        }
    }

    @autoreleasepool {
        if(!upload_sort_inputs(ctx, keys, values) || !run_sort_dispatch(ctx, nullptr) || !verify_sorted_output(ctx, keys)) {
            return false;
        }
    }

    out_row->label = distribution_label(distribution);
    out_row->histogram_ms = (aggregate.histogram_ns / static_cast<double>(options.timed_iterations)) / 1000000.0;
    out_row->prefix_offsets_ms = (aggregate.prefix_offsets_ns / static_cast<double>(options.timed_iterations)) / 1000000.0;
    out_row->scatter_ms = (aggregate.scatter_ns / static_cast<double>(options.timed_iterations)) / 1000000.0;
    out_row->total_ms = (aggregate.total_ns / static_cast<double>(options.timed_iterations)) / 1000000.0;
    out_row->scatter_share = out_row->total_ms > 0.0 ? out_row->scatter_ms / out_row->total_ms : 0.0;
    out_row->scatter_dominates = out_row->scatter_ms > out_row->histogram_ms && out_row->scatter_ms > out_row->prefix_offsets_ms;
    return true;
}

static void print_report(const SortBenchmarkOptions &options, const SortBenchmarkContext &ctx, const std::array<SortProfileRow, 4> &rows)
{
    uint32_t confirmed_count = 0u;

    std::printf(
        "Metal sort kernel profile baseline\n"
        "count=%" PRIu32 ", significant_bits=%" PRIu32 ", radix_passes=%" PRIu32 ", threadgroups=%" PRIu32 ", warmup=%" PRIu32 ", timed=%" PRIu32 "\n\n",
        options.count,
        ctx.significant_bits,
        ctx.pass_count,
        ctx.num_threadgroups,
        options.warmup_iterations,
        options.timed_iterations);

    std::printf(
        "%-18s %14s %18s %18s %12s %14s %12s\n",
        "distribution",
        "histogram_ms",
        "prefix_offsets_ms",
        "scatter_ms",
        "total_ms",
        "scatter_share",
        "dominates");

    for(const SortProfileRow &row : rows) {
        if(row.scatter_dominates) {
            confirmed_count += 1u;
        }
        std::printf(
            "%-18s %14.3f %18.3f %18.3f %12.3f %13.1f%% %12s\n",
            row.label,
            row.histogram_ms,
            row.prefix_offsets_ms,
            row.scatter_ms,
            row.total_ms,
            row.scatter_share * 100.0,
            row.scatter_dominates ? "yes" : "no");
    }

    std::printf(
        "\nscatter dominance: %s (%" PRIu32 "/%zu distributions where scatter > histogram and scatter > prefix_offsets)\n",
        confirmed_count == rows.size() ? "confirmed" : "not confirmed",
        confirmed_count,
        rows.size());
}

} // namespace

int main(int argc, char *argv[])
{
    SortBenchmarkOptions options{};
    SortBenchmarkContext ctx{};
    std::array<SortProfileRow, 4> rows{};

    if(!parse_options(argc, argv, &options)) {
        std::fprintf(stderr, "invalid arguments\n");
        return 2;
    }
    if(!gsx_init_context(options, &ctx)) {
        return 1;
    }

    if(!measure_distribution(ctx, options, InputDistribution::UniformRandom, &rows[0])
        || !measure_distribution(ctx, options, InputDistribution::AlreadySorted, &rows[1])
        || !measure_distribution(ctx, options, InputDistribution::ReverseSorted, &rows[2])
        || !measure_distribution(ctx, options, InputDistribution::SkewedDigits, &rows[3])) {
        gsx_cleanup_context(&ctx);
        return 1;
    }

    print_report(options, ctx, rows);
    gsx_cleanup_context(&ctx);
    return 0;
}
