#include <gsx/gsx-backend.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct gsx_data_type_flag_desc {
    gsx_data_type_flags flag;
    const char *name;
} gsx_data_type_flag_desc;

static void fatal_on_error(gsx_error err, const char *context)
{
    if (gsx_error_is_success(err)) {
        return;
    }
    fprintf(stderr, "error: %s failed (%d)", context, err.code);
    if (err.message != NULL) {
        fprintf(stderr, ": %s", err.message);
    }
    fprintf(stderr, "\n");
    exit(EXIT_FAILURE);
}

static const char *backend_type_name(gsx_backend_type type)
{
    switch (type) {
    case GSX_BACKEND_TYPE_CPU:
        return "CPU";
    case GSX_BACKEND_TYPE_CUDA:
        return "CUDA";
    case GSX_BACKEND_TYPE_METAL:
        return "METAL";
    default:
        return "UNKNOWN";
    }
}

static const char *buffer_type_class_name(gsx_backend_buffer_type_class type)
{
    switch (type) {
    case GSX_BACKEND_BUFFER_TYPE_HOST:
        return "HOST";
    case GSX_BACKEND_BUFFER_TYPE_HOST_PINNED:
        return "HOST_PINNED";
    case GSX_BACKEND_BUFFER_TYPE_DEVICE:
        return "DEVICE";
    case GSX_BACKEND_BUFFER_TYPE_UNIFIED:
        return "UNIFIED";
    default:
        return "UNKNOWN";
    }
}

static void print_size(const char *label, gsx_size_t size_bytes)
{
    double gib = (double)size_bytes / (1024.0 * 1024.0 * 1024.0);
    printf("    %s: %" PRIu64 " bytes (%.2f GiB)\n", label, (uint64_t)size_bytes, gib);
}

static void print_supported_data_types(gsx_data_type_flags flags)
{
    static const gsx_data_type_flag_desc all_flags[] = {
        {GSX_DATA_TYPE_FLAG_F32, "F32"},
        {GSX_DATA_TYPE_FLAG_F16, "F16"},
        {GSX_DATA_TYPE_FLAG_BF16, "BF16"},
        {GSX_DATA_TYPE_FLAG_U8, "U8"},
        {GSX_DATA_TYPE_FLAG_I16, "I16"},
        {GSX_DATA_TYPE_FLAG_I32, "I32"},
        {GSX_DATA_TYPE_FLAG_U32, "U32"},
    };
    size_t supported_count = 0;
    printf("    Supported data types:");
    for (size_t i = 0; i < sizeof(all_flags) / sizeof(all_flags[0]); ++i) {
        if ((flags & all_flags[i].flag) == 0) {
            continue;
        }
        printf(" %s", all_flags[i].name);
        supported_count++;
    }
    if (supported_count == 0) {
        printf(" none");
    }
    printf("\n");
}

static void print_buffer_type_support(gsx_backend_t backend)
{
    static const gsx_backend_buffer_type_class all_classes[] = {
        GSX_BACKEND_BUFFER_TYPE_HOST,
        GSX_BACKEND_BUFFER_TYPE_HOST_PINNED,
        GSX_BACKEND_BUFFER_TYPE_DEVICE,
        GSX_BACKEND_BUFFER_TYPE_UNIFIED,
    };

    printf("    Portable buffer class support:");
    for (size_t i = 0; i < sizeof(all_classes) / sizeof(all_classes[0]); ++i) {
        gsx_backend_buffer_type_t buffer_type = NULL;
        gsx_error err = gsx_backend_find_buffer_type(backend, all_classes[i], &buffer_type);
        if (gsx_error_is_success(err)) {
            printf(" %s=yes", buffer_type_class_name(all_classes[i]));
        } else if (err.code == GSX_ERROR_NOT_SUPPORTED) {
            printf(" %s=no", buffer_type_class_name(all_classes[i]));
        } else {
            fatal_on_error(err, "gsx_backend_find_buffer_type");
        }
    }
    printf("\n");
}

static void print_backend_buffer_types(gsx_backend_t backend)
{
    gsx_index_t buffer_type_count = 0;
    fatal_on_error(gsx_backend_count_buffer_types(backend, &buffer_type_count), "gsx_backend_count_buffer_types");
    printf("    Buffer types (%d):\n", (int)buffer_type_count);

    for (gsx_index_t i = 0; i < buffer_type_count; ++i) {
        gsx_backend_buffer_type_t buffer_type = NULL;
        gsx_backend_buffer_type_info info;
        fatal_on_error(gsx_backend_get_buffer_type(backend, i, &buffer_type), "gsx_backend_get_buffer_type");
        fatal_on_error(gsx_backend_buffer_type_get_info(buffer_type, &info), "gsx_backend_buffer_type_get_info");
        printf("      [%d] %s class=%s\n", (int)i, info.name != NULL ? info.name : "(null)", buffer_type_class_name(info.type));
        printf("        alignment: %" PRIu64 " bytes\n", (uint64_t)info.alignment_bytes);
        if (info.max_allocation_size_bytes == 0) {
            printf("        max_allocation: unknown\n");
        } else {
            printf("        max_allocation: %" PRIu64 " bytes\n", (uint64_t)info.max_allocation_size_bytes);
        }
    }
}

static void print_device_details(gsx_backend_device_t device)
{
    gsx_backend_device_info device_info;
    gsx_backend_desc backend_desc;
    gsx_backend_t backend = NULL;
    gsx_backend_info backend_info;
    gsx_backend_capabilities capabilities;

    fatal_on_error(gsx_backend_device_get_info(device, &device_info), "gsx_backend_device_get_info");
    printf("  Device %d (%s)\n", (int)device_info.device_index, device_info.name != NULL ? device_info.name : "(null)");
    printf("    Backend family: %s (%s)\n", backend_type_name(device_info.backend_type), device_info.backend_name != NULL ? device_info.backend_name : "(null)");
    print_size("Total memory", device_info.total_memory_bytes);

    backend_desc.device = device;
    backend_desc.options = NULL;
    backend_desc.options_size_bytes = 0;
    fatal_on_error(gsx_backend_init(&backend, &backend_desc), "gsx_backend_init");

    fatal_on_error(gsx_backend_get_info(backend, &backend_info), "gsx_backend_get_info");
    fatal_on_error(gsx_backend_get_capabilities(backend, &capabilities), "gsx_backend_get_capabilities");
    printf("    Runtime backend type: %s\n", backend_type_name(backend_info.backend_type));
    printf("    Async prefetch: %s\n", capabilities.supports_async_prefetch ? "yes" : "no");
    print_supported_data_types(capabilities.supported_data_types);
    print_buffer_type_support(backend);
    print_backend_buffer_types(backend);

    fatal_on_error(gsx_backend_free(backend), "gsx_backend_free");
}

int main(void)
{
    static const gsx_backend_type all_backend_types[] = {
        GSX_BACKEND_TYPE_CPU,
        GSX_BACKEND_TYPE_CUDA,
        GSX_BACKEND_TYPE_METAL,
    };

    gsx_index_t total_device_count = 0;

    fatal_on_error(gsx_backend_registry_init(), "gsx_backend_registry_init");
    fatal_on_error(gsx_count_backend_devices(&total_device_count), "gsx_count_backend_devices");
    printf("GSX backend inventory\n");
    printf("Total devices: %d\n\n", (int)total_device_count);

    for (size_t i = 0; i < sizeof(all_backend_types) / sizeof(all_backend_types[0]); ++i) {
        gsx_index_t device_count = 0;
        fatal_on_error(
            gsx_count_backend_devices_by_type(all_backend_types[i], &device_count),
            "gsx_count_backend_devices_by_type"
        );
        printf("%s devices: %d\n", backend_type_name(all_backend_types[i]), (int)device_count);
        for (gsx_index_t j = 0; j < device_count; ++j) {
            gsx_backend_device_t device = NULL;
            fatal_on_error(
                gsx_get_backend_device_by_type(all_backend_types[i], j, &device),
                "gsx_get_backend_device_by_type"
            );
            print_device_details(device);
        }
        printf("\n");
    }

    return EXIT_SUCCESS;
}
