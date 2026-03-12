extern "C" {
#include "../gsx/src/gsx-impl.h"
}

#include <gtest/gtest.h>

namespace {

#define ASSERT_GSX_SUCCESS(expr)                                                                                     \
    do {                                                                                                             \
        const gsx_error gsx_assert_success_error__ = (expr);                                                         \
        ASSERT_EQ(gsx_assert_success_error__.code, GSX_ERROR_SUCCESS)                                                \
            << (gsx_assert_success_error__.message != nullptr ? gsx_assert_success_error__.message : "");           \
    } while(false)

#define EXPECT_GSX_CODE(expr, expected_code)                                                                         \
    do {                                                                                                             \
        const gsx_error gsx_expect_code_error__ = (expr);                                                            \
        EXPECT_EQ(gsx_expect_code_error__.code, (expected_code))                                                     \
            << (gsx_expect_code_error__.message != nullptr ? gsx_expect_code_error__.message : "");                \
    } while(false)

typedef struct gsx_fake_backend_device {
    struct gsx_backend_device base;
} gsx_fake_backend_device;

typedef struct gsx_fake_backend_provider {
    struct gsx_backend_provider base;
    gsx_index_t device_count;
    gsx_fake_backend_device devices[2];
} gsx_fake_backend_provider;

static gsx_error gsx_fake_backend_provider_discover_devices(gsx_backend_provider_t provider, gsx_builtin_registry_state *registry)
{
    gsx_fake_backend_provider *fake_provider = (gsx_fake_backend_provider *)provider;
    gsx_index_t device_index = 0;
    gsx_error error = { GSX_ERROR_SUCCESS, nullptr };

    for(device_index = 0; device_index < fake_provider->device_count; ++device_index) {
        error = gsx_builtin_registry_append_device(registry, &fake_provider->devices[device_index].base);
        if(!gsx_error_is_success(error)) {
            return error;
        }
    }

    return gsx_make_error(GSX_ERROR_SUCCESS, nullptr);
}

static gsx_error gsx_fake_backend_provider_create_backend(gsx_backend_device_t backend_device, const gsx_backend_desc *desc, gsx_backend_t *out_backend)
{
    (void)backend_device;
    (void)desc;
    (void)out_backend;
    return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "fake topology providers do not create runtime backends");
}

static const gsx_backend_provider_i gsx_fake_backend_provider_iface = {
    gsx_fake_backend_provider_discover_devices,
    gsx_fake_backend_provider_create_backend
};

static const gsx_backend_provider_i gsx_fake_backend_provider_missing_discover_iface = {
    nullptr,
    gsx_fake_backend_provider_create_backend
};

static const gsx_backend_provider_i gsx_fake_backend_provider_missing_create_backend_iface = {
    gsx_fake_backend_provider_discover_devices,
    nullptr
};

static void gsx_fake_backend_provider_init(
    gsx_fake_backend_provider *provider,
    gsx_backend_type backend_type,
    const char *backend_name,
    gsx_index_t device_count,
    const char *first_device_name,
    const char *second_device_name
)
{
    provider->base.iface = &gsx_fake_backend_provider_iface;
    provider->base.backend_type = backend_type;
    provider->base.backend_name = backend_name;
    provider->device_count = device_count;

    provider->devices[0].base.provider = &provider->base;
    provider->devices[0].base.info.backend_type = backend_type;
    provider->devices[0].base.info.backend_name = backend_name;
    provider->devices[0].base.info.device_index = 0;
    provider->devices[0].base.info.name = first_device_name;
    provider->devices[0].base.info.total_memory_bytes = 0;

    provider->devices[1].base.provider = &provider->base;
    provider->devices[1].base.info.backend_type = backend_type;
    provider->devices[1].base.info.backend_name = backend_name;
    provider->devices[1].base.info.device_index = 1;
    provider->devices[1].base.info.name = second_device_name;
    provider->devices[1].base.info.total_memory_bytes = 0;
}

static void gsx_fake_backend_provider_register(gsx_builtin_registry_state *registry, gsx_fake_backend_provider *provider)
{
    ASSERT_GSX_SUCCESS(provider->base.iface->discover_devices(&provider->base, registry));
    ASSERT_GSX_SUCCESS(gsx_builtin_registry_append_provider(registry, &provider->base));
}

TEST(BackendRegistryTopology, RejectsMalformedProviderRegistration)
{
    gsx_builtin_registry_state registry = {};
    gsx_fake_backend_provider provider = {};

    provider.base.iface = nullptr;
    provider.base.backend_type = GSX_BACKEND_TYPE_CPU;
    provider.base.backend_name = "cpu";
    EXPECT_GSX_CODE(gsx_builtin_registry_append_provider(&registry, &provider.base), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(registry.backend_provider_count, 0);

    provider.base.iface = &gsx_fake_backend_provider_missing_discover_iface;
    EXPECT_GSX_CODE(gsx_builtin_registry_append_provider(&registry, &provider.base), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(registry.backend_provider_count, 0);

    provider.base.iface = &gsx_fake_backend_provider_missing_create_backend_iface;
    EXPECT_GSX_CODE(gsx_builtin_registry_append_provider(&registry, &provider.base), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(registry.backend_provider_count, 0);

    provider.base.iface = &gsx_fake_backend_provider_iface;
    provider.base.backend_type = (gsx_backend_type)99;
    EXPECT_GSX_CODE(gsx_builtin_registry_append_provider(&registry, &provider.base), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(registry.backend_provider_count, 0);

    provider.base.backend_type = GSX_BACKEND_TYPE_CPU;
    provider.base.backend_name = "";
    EXPECT_GSX_CODE(gsx_builtin_registry_append_provider(&registry, &provider.base), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(registry.backend_provider_count, 0);

    gsx_builtin_registry_reset(&registry);
}

TEST(BackendRegistryTopology, RejectsMalformedDeviceRegistration)
{
    gsx_builtin_registry_state registry = {};
    gsx_fake_backend_provider provider = {};
    gsx_fake_backend_device device = {};

    gsx_fake_backend_provider_init(&provider, GSX_BACKEND_TYPE_CPU, "cpu", 1, "cpu0", nullptr);
    device.base.provider = nullptr;
    device.base.info.backend_type = GSX_BACKEND_TYPE_CPU;
    device.base.info.backend_name = "cpu";
    device.base.info.device_index = 0;
    device.base.info.name = "cpu0";
    EXPECT_GSX_CODE(gsx_builtin_registry_append_device(&registry, &device.base), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(registry.backend_device_count, 0);

    device.base.provider = &provider.base;
    device.base.info.backend_type = GSX_BACKEND_TYPE_CUDA;
    EXPECT_GSX_CODE(gsx_builtin_registry_append_device(&registry, &device.base), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(registry.backend_device_count, 0);

    device.base.info.backend_type = GSX_BACKEND_TYPE_CPU;
    device.base.info.backend_name = "cuda";
    EXPECT_GSX_CODE(gsx_builtin_registry_append_device(&registry, &device.base), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(registry.backend_device_count, 0);

    device.base.info.backend_name = "cpu";
    device.base.info.name = "";
    EXPECT_GSX_CODE(gsx_builtin_registry_append_device(&registry, &device.base), GSX_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(registry.backend_device_count, 0);

    gsx_builtin_registry_reset(&registry);
}

TEST(BackendRegistryTopology, FlattenedInventoryPreservesProviderAndDeviceOrder)
{
    gsx_builtin_registry_state registry = {};
    gsx_fake_backend_provider metal_provider = {};
    gsx_fake_backend_provider cuda_provider = {};
    gsx_fake_backend_provider cpu_provider = {};

    gsx_fake_backend_provider_init(&metal_provider, GSX_BACKEND_TYPE_METAL, "metal", 2, "igpu", "shared-nvgpu");
    gsx_fake_backend_provider_init(&cuda_provider, GSX_BACKEND_TYPE_CUDA, "cuda", 1, "shared-nvgpu", nullptr);
    gsx_fake_backend_provider_init(&cpu_provider, GSX_BACKEND_TYPE_CPU, "cpu", 1, "cpu0", nullptr);

    gsx_fake_backend_provider_register(&registry, &metal_provider);
    gsx_fake_backend_provider_register(&registry, &cuda_provider);
    gsx_fake_backend_provider_register(&registry, &cpu_provider);

    ASSERT_EQ(registry.backend_provider_count, 3);
    ASSERT_EQ(registry.backend_device_count, 4);

    EXPECT_EQ(registry.backend_devices[0]->info.backend_type, GSX_BACKEND_TYPE_METAL);
    EXPECT_EQ(registry.backend_devices[0]->info.device_index, 0);
    EXPECT_STREQ(registry.backend_devices[0]->info.name, "igpu");

    EXPECT_EQ(registry.backend_devices[1]->info.backend_type, GSX_BACKEND_TYPE_METAL);
    EXPECT_EQ(registry.backend_devices[1]->info.device_index, 1);
    EXPECT_STREQ(registry.backend_devices[1]->info.name, "shared-nvgpu");

    EXPECT_EQ(registry.backend_devices[2]->info.backend_type, GSX_BACKEND_TYPE_CUDA);
    EXPECT_EQ(registry.backend_devices[2]->info.device_index, 0);
    EXPECT_STREQ(registry.backend_devices[2]->info.name, "shared-nvgpu");

    EXPECT_EQ(registry.backend_devices[3]->info.backend_type, GSX_BACKEND_TYPE_CPU);
    EXPECT_EQ(registry.backend_devices[3]->info.device_index, 0);
    EXPECT_STREQ(registry.backend_devices[3]->info.name, "cpu0");

    EXPECT_EQ(registry.backend_devices[1]->info.backend_type, GSX_BACKEND_TYPE_METAL);
    EXPECT_EQ(registry.backend_devices[2]->info.backend_type, GSX_BACKEND_TYPE_CUDA);
    EXPECT_STREQ(registry.backend_devices[1]->info.name, registry.backend_devices[2]->info.name);

    gsx_builtin_registry_reset(&registry);
}

}  // namespace
