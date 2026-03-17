#include "gsx/extra/gsx-io-ply.h"

#include "gsx-impl.h"
#include "happly/happly.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr float GSX_PLY_RGB_NORMALIZATION_FACTOR = 255.0f;
constexpr float GSX_PLY_SH_DC_TO_RGB_FACTOR = 0.28209479177387814f;

template <typename T>
bool gsx_ply_try_get_property(happly::PLYData &ply, const std::string &name, std::vector<T> *out_values)
{
	if(out_values == nullptr) {
		return false;
	}
	out_values->clear();
	try {
		*out_values = ply.getElement("vertex").getProperty<T>(name);
		return true;
	} catch(...) {
		return false;
	}
}

bool gsx_ply_get_property_as_f32(happly::PLYData &ply, const std::string &name, std::vector<float> *out_values)
{
	std::vector<float> values_f32;
	std::vector<double> values_f64;

	if(out_values == nullptr) {
		return false;
	}
	if(gsx_ply_try_get_property<float>(ply, name, &values_f32)) {
		*out_values = std::move(values_f32);
		return true;
	}
	if(gsx_ply_try_get_property<double>(ply, name, &values_f64)) {
		out_values->assign(values_f64.begin(), values_f64.end());
		return true;
	}
	return false;
}

bool gsx_ply_get_property_as_u8(happly::PLYData &ply, const std::string &name, std::vector<uint8_t> *out_values)
{
	std::vector<int> values_i32;

	if(out_values == nullptr) {
		return false;
	}
	if(gsx_ply_try_get_property<uint8_t>(ply, name, out_values)) {
		return true;
	}
	if(gsx_ply_try_get_property<int>(ply, name, &values_i32)) {
		out_values->resize(values_i32.size());
		for(size_t i = 0; i < values_i32.size(); ++i) {
			const int value = values_i32[i];
			if(value < 0) {
				(*out_values)[i] = 0;
			} else if(value > 255) {
				(*out_values)[i] = 255;
			} else {
				(*out_values)[i] = (uint8_t)value;
			}
		}
		return true;
	}
	return false;
}

gsx_error gsx_ply_get_backend_from_gs(gsx_gs_t gs, gsx_backend_t *out_backend)
{
	gsx_gs_info gs_info = {};
	gsx_error error = { GSX_ERROR_SUCCESS, NULL };

	if(out_backend == nullptr) {
		return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_backend must be non-null");
	}
	*out_backend = NULL;

	error = gsx_gs_get_info(gs, &gs_info);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	return gsx_arena_get_backend(gs_info.arena, out_backend);
}

gsx_error gsx_ply_upload_field(gsx_gs_t gs, gsx_gs_field field, const std::vector<float> &values)
{
	gsx_tensor_t tensor = NULL;
	gsx_size_t size_bytes = 0;
	gsx_error error = gsx_gs_get_field(gs, field, &tensor);

	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_tensor_get_size_bytes(tensor, &size_bytes);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	if(values.size() * sizeof(float) != (size_t)size_bytes) {
		return gsx_make_error(GSX_ERROR_INVALID_STATE, "PLY field size does not match gs tensor layout");
	}
	return gsx_tensor_upload(tensor, values.data(), size_bytes);
}

gsx_error gsx_ply_try_upload_optional_field(gsx_gs_t gs, gsx_gs_field field, const std::vector<float> &values)
{
	gsx_tensor_t tensor = NULL;
	gsx_size_t size_bytes = 0;
	gsx_error error = gsx_gs_get_field(gs, field, &tensor);

	if(error.code == GSX_ERROR_INVALID_STATE) {
		return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
	}
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_tensor_get_size_bytes(tensor, &size_bytes);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	if(values.size() * sizeof(float) != (size_t)size_bytes) {
		return gsx_make_error(GSX_ERROR_INVALID_STATE, "PLY optional field size does not match gs tensor layout");
	}
	return gsx_tensor_upload(tensor, values.data(), size_bytes);
}

gsx_error gsx_ply_download_field(gsx_gs_t gs, gsx_gs_field field, std::vector<float> *out_values)
{
	gsx_tensor_t tensor = NULL;
	gsx_size_t size_bytes = 0;
	gsx_error error = { GSX_ERROR_SUCCESS, NULL };

	if(out_values == nullptr) {
		return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_values must be non-null");
	}
	out_values->clear();

	error = gsx_gs_get_field(gs, field, &tensor);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_tensor_get_size_bytes(tensor, &size_bytes);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	if(size_bytes % sizeof(float) != 0) {
		return gsx_make_error(GSX_ERROR_INVALID_STATE, "gs tensor byte size is not float32-aligned");
	}

	out_values->resize((size_t)(size_bytes / sizeof(float)));
	return gsx_tensor_download(tensor, out_values->data(), size_bytes);
}

gsx_error gsx_ply_try_download_optional_field(gsx_gs_t gs, gsx_gs_field field, std::vector<float> *out_values)
{
	gsx_error error = gsx_ply_download_field(gs, field, out_values);

	if(error.code == GSX_ERROR_INVALID_STATE) {
		if(out_values != nullptr) {
			out_values->clear();
		}
		return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
	}
	return error;
}

} // namespace

extern "C" {

gsx_error gsx_read_ply(gsx_gs_t *out_gs, const char *filename)
{
	std::unique_ptr<happly::PLYData> ply_data;
	gsx_size_t gaussian_count = 0;
	std::vector<double> vertex_x;
	std::vector<double> vertex_y;
	std::vector<double> vertex_z;
	std::vector<float> f_dc_r;
	std::vector<float> f_dc_g;
	std::vector<float> f_dc_b;
	std::vector<uint8_t> color_r;
	std::vector<uint8_t> color_g;
	std::vector<uint8_t> color_b;
	std::vector<float> means;
	std::vector<float> logscale;
	std::vector<float> rotation;
	std::vector<float> opacity;
	std::vector<float> sh0;
	std::vector<float> sh1;
	std::vector<float> sh2;
	std::vector<float> sh3;
	std::vector<float> scale_0;
	std::vector<float> scale_1;
	std::vector<float> scale_2;
	std::vector<float> rot_w;
	std::vector<float> rot_x;
	std::vector<float> rot_y;
	std::vector<float> rot_z;
	std::vector<float> opacity_prop;
	gsx_error error = { GSX_ERROR_SUCCESS, NULL };

	if(out_gs == NULL || filename == NULL) {
		return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_gs and filename must be non-null");
	}
	if(*out_gs == NULL) {
		return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "*out_gs must be initialized before reading PLY");
	}

	try {
		ply_data.reset(new happly::PLYData(filename));
		vertex_x = ply_data->getElement("vertex").getProperty<double>("x");
		vertex_y = ply_data->getElement("vertex").getProperty<double>("y");
		vertex_z = ply_data->getElement("vertex").getProperty<double>("z");
	} catch(const std::exception &e) {
		return gsx_make_error(GSX_ERROR_IO, e.what());
	}

	if(vertex_x.size() != vertex_y.size() || vertex_x.size() != vertex_z.size()) {
		return gsx_make_error(GSX_ERROR_CHECKPOINT_CORRUPT, "PLY vertex xyz properties have inconsistent lengths");
	}

	gaussian_count = (gsx_size_t)vertex_x.size();
	error = gsx_gs_resize(*out_gs, gaussian_count);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	if(gaussian_count == 0) {
		return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
	}

	means.resize((size_t)gaussian_count * 3u);
	logscale.assign((size_t)gaussian_count * 3u, 0.0f);
	rotation.assign((size_t)gaussian_count * 4u, 0.0f);
	opacity.assign((size_t)gaussian_count, 0.0f);
	sh0.assign((size_t)gaussian_count * 3u, 0.0f);
	sh1.assign((size_t)gaussian_count * 9u, 0.0f);
	sh2.assign((size_t)gaussian_count * 15u, 0.0f);
	sh3.assign((size_t)gaussian_count * 21u, 0.0f);

	for(size_t i = 0; i < (size_t)gaussian_count; ++i) {
		means[i * 3u + 0u] = (float)vertex_x[i];
		means[i * 3u + 1u] = (float)vertex_y[i];
		means[i * 3u + 2u] = (float)vertex_z[i];
		rotation[i * 4u + 0u] = 1.0f;
	}

	(void)gsx_ply_get_property_as_f32(*ply_data, "scale_0", &scale_0);
	(void)gsx_ply_get_property_as_f32(*ply_data, "scale_1", &scale_1);
	(void)gsx_ply_get_property_as_f32(*ply_data, "scale_2", &scale_2);
	(void)gsx_ply_get_property_as_f32(*ply_data, "opacity", &opacity_prop);
	(void)gsx_ply_get_property_as_f32(*ply_data, "rot_w", &rot_w);
	(void)gsx_ply_get_property_as_f32(*ply_data, "rot_x", &rot_x);
	(void)gsx_ply_get_property_as_f32(*ply_data, "rot_y", &rot_y);
	(void)gsx_ply_get_property_as_f32(*ply_data, "rot_z", &rot_z);
	(void)gsx_ply_get_property_as_f32(*ply_data, "f_dc_r", &f_dc_r);
	(void)gsx_ply_get_property_as_f32(*ply_data, "f_dc_g", &f_dc_g);
	(void)gsx_ply_get_property_as_f32(*ply_data, "f_dc_b", &f_dc_b);
	(void)gsx_ply_get_property_as_u8(*ply_data, "red", &color_r);
	(void)gsx_ply_get_property_as_u8(*ply_data, "green", &color_g);
	(void)gsx_ply_get_property_as_u8(*ply_data, "blue", &color_b);

	if(scale_0.size() == (size_t)gaussian_count && scale_1.size() == (size_t)gaussian_count && scale_2.size() == (size_t)gaussian_count) {
		for(size_t i = 0; i < (size_t)gaussian_count; ++i) {
			logscale[i * 3u + 0u] = scale_0[i];
			logscale[i * 3u + 1u] = scale_1[i];
			logscale[i * 3u + 2u] = scale_2[i];
		}
	}
	if(opacity_prop.size() == (size_t)gaussian_count) {
		opacity = opacity_prop;
	}
	if(rot_w.size() == (size_t)gaussian_count && rot_x.size() == (size_t)gaussian_count
		&& rot_y.size() == (size_t)gaussian_count && rot_z.size() == (size_t)gaussian_count) {
		for(size_t i = 0; i < (size_t)gaussian_count; ++i) {
			rotation[i * 4u + 0u] = rot_w[i];
			rotation[i * 4u + 1u] = rot_x[i];
			rotation[i * 4u + 2u] = rot_y[i];
			rotation[i * 4u + 3u] = rot_z[i];
		}
	}

	if(f_dc_r.size() == (size_t)gaussian_count && f_dc_g.size() == (size_t)gaussian_count && f_dc_b.size() == (size_t)gaussian_count) {
		for(size_t i = 0; i < (size_t)gaussian_count; ++i) {
			sh0[i * 3u + 0u] = f_dc_r[i];
			sh0[i * 3u + 1u] = f_dc_g[i];
			sh0[i * 3u + 2u] = f_dc_b[i];
		}
	} else if(color_r.size() == (size_t)gaussian_count && color_g.size() == (size_t)gaussian_count && color_b.size() == (size_t)gaussian_count) {
		for(size_t i = 0; i < (size_t)gaussian_count; ++i) {
			const float r = ((float)color_r[i] / GSX_PLY_RGB_NORMALIZATION_FACTOR - 0.5f) / GSX_PLY_SH_DC_TO_RGB_FACTOR;
			const float g = ((float)color_g[i] / GSX_PLY_RGB_NORMALIZATION_FACTOR - 0.5f) / GSX_PLY_SH_DC_TO_RGB_FACTOR;
			const float b = ((float)color_b[i] / GSX_PLY_RGB_NORMALIZATION_FACTOR - 0.5f) / GSX_PLY_SH_DC_TO_RGB_FACTOR;
			sh0[i * 3u + 0u] = r;
			sh0[i * 3u + 1u] = g;
			sh0[i * 3u + 2u] = b;
		}
	}

	for(size_t coeff_index = 0; coeff_index < 15; ++coeff_index) {
		std::vector<float> rest_r;
		std::vector<float> rest_g;
		std::vector<float> rest_b;
		const std::string index_str = std::to_string(coeff_index);
		const bool has_r = gsx_ply_get_property_as_f32(*ply_data, "f_rest_r_" + index_str, &rest_r);
		const bool has_g = gsx_ply_get_property_as_f32(*ply_data, "f_rest_g_" + index_str, &rest_g);
		const bool has_b = gsx_ply_get_property_as_f32(*ply_data, "f_rest_b_" + index_str, &rest_b);

		if(!(has_r && has_g && has_b)) {
			continue;
		}
		if(rest_r.size() != (size_t)gaussian_count || rest_g.size() != (size_t)gaussian_count || rest_b.size() != (size_t)gaussian_count) {
			return gsx_make_error(GSX_ERROR_CHECKPOINT_CORRUPT, "PLY SH rest property length mismatch");
		}

		for(size_t i = 0; i < (size_t)gaussian_count; ++i) {
			if(coeff_index < 3) {
				const size_t base = ((i * 3u) + coeff_index) * 3u;
				sh1[base + 0u] = rest_r[i];
				sh1[base + 1u] = rest_g[i];
				sh1[base + 2u] = rest_b[i];
			} else if(coeff_index < 8) {
				const size_t base = ((i * 5u) + (coeff_index - 3u)) * 3u;
				sh2[base + 0u] = rest_r[i];
				sh2[base + 1u] = rest_g[i];
				sh2[base + 2u] = rest_b[i];
			} else {
				const size_t base = ((i * 7u) + (coeff_index - 8u)) * 3u;
				sh3[base + 0u] = rest_r[i];
				sh3[base + 1u] = rest_g[i];
				sh3[base + 2u] = rest_b[i];
			}
		}
	}

	error = gsx_ply_upload_field(*out_gs, GSX_GS_FIELD_MEAN3D, means);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_ply_upload_field(*out_gs, GSX_GS_FIELD_LOGSCALE, logscale);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_ply_upload_field(*out_gs, GSX_GS_FIELD_ROTATION, rotation);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_ply_upload_field(*out_gs, GSX_GS_FIELD_OPACITY, opacity);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_ply_upload_field(*out_gs, GSX_GS_FIELD_SH0, sh0);
	if(!gsx_error_is_success(error)) {
		return error;
	}

	error = gsx_ply_try_upload_optional_field(*out_gs, GSX_GS_FIELD_SH1, sh1);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_ply_try_upload_optional_field(*out_gs, GSX_GS_FIELD_SH2, sh2);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_ply_try_upload_optional_field(*out_gs, GSX_GS_FIELD_SH3, sh3);
	if(!gsx_error_is_success(error)) {
		return error;
	}

	return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_write_ply(gsx_gs_t gs, const char *filename)
{
	gsx_backend_t backend = NULL;
	gsx_tensor_t mean_tensor = NULL;
	gsx_tensor_info mean_info = {};
	gsx_size_t gaussian_count = 0;
	std::vector<float> means;
	std::vector<float> logscale;
	std::vector<float> rotation;
	std::vector<float> opacity;
	std::vector<float> sh0;
	std::vector<float> sh1;
	std::vector<float> sh2;
	std::vector<float> sh3;
	std::vector<float> x;
	std::vector<float> y;
	std::vector<float> z;
	std::vector<float> scale_0;
	std::vector<float> scale_1;
	std::vector<float> scale_2;
	std::vector<float> rot_w;
	std::vector<float> rot_x;
	std::vector<float> rot_y;
	std::vector<float> rot_z;
	std::vector<uint8_t> color_r;
	std::vector<uint8_t> color_g;
	std::vector<uint8_t> color_b;
	gsx_error error = { GSX_ERROR_SUCCESS, NULL };

	if(gs == NULL || filename == NULL) {
		return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "gs and filename must be non-null");
	}

	error = gsx_ply_get_backend_from_gs(gs, &backend);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_backend_major_stream_sync(backend);
	if(!gsx_error_is_success(error)) {
		return error;
	}

	error = gsx_gs_get_field(gs, GSX_GS_FIELD_MEAN3D, &mean_tensor);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_tensor_get_info(mean_tensor, &mean_info);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	if(mean_info.rank != 2 || mean_info.shape[1] != 3) {
		return gsx_make_error(GSX_ERROR_INVALID_STATE, "gs mean3d tensor has an unexpected shape");
	}
	gaussian_count = (gsx_size_t)mean_info.shape[0];

	error = gsx_ply_download_field(gs, GSX_GS_FIELD_MEAN3D, &means);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_ply_download_field(gs, GSX_GS_FIELD_LOGSCALE, &logscale);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_ply_download_field(gs, GSX_GS_FIELD_ROTATION, &rotation);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_ply_download_field(gs, GSX_GS_FIELD_OPACITY, &opacity);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_ply_download_field(gs, GSX_GS_FIELD_SH0, &sh0);
	if(!gsx_error_is_success(error)) {
		return error;
	}

	error = gsx_ply_try_download_optional_field(gs, GSX_GS_FIELD_SH1, &sh1);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_ply_try_download_optional_field(gs, GSX_GS_FIELD_SH2, &sh2);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_ply_try_download_optional_field(gs, GSX_GS_FIELD_SH3, &sh3);
	if(!gsx_error_is_success(error)) {
		return error;
	}

	error = gsx_backend_major_stream_sync(backend);
	if(!gsx_error_is_success(error)) {
		return error;
	}

	x.resize((size_t)gaussian_count);
	y.resize((size_t)gaussian_count);
	z.resize((size_t)gaussian_count);
	scale_0.resize((size_t)gaussian_count);
	scale_1.resize((size_t)gaussian_count);
	scale_2.resize((size_t)gaussian_count);
	rot_w.resize((size_t)gaussian_count);
	rot_x.resize((size_t)gaussian_count);
	rot_y.resize((size_t)gaussian_count);
	rot_z.resize((size_t)gaussian_count);
	color_r.resize((size_t)gaussian_count);
	color_g.resize((size_t)gaussian_count);
	color_b.resize((size_t)gaussian_count);

	for(size_t i = 0; i < (size_t)gaussian_count; ++i) {
		x[i] = means[i * 3u + 0u];
		y[i] = means[i * 3u + 1u];
		z[i] = means[i * 3u + 2u];
		scale_0[i] = logscale[i * 3u + 0u];
		scale_1[i] = logscale[i * 3u + 1u];
		scale_2[i] = logscale[i * 3u + 2u];
		rot_w[i] = rotation[i * 4u + 0u];
		rot_x[i] = rotation[i * 4u + 1u];
		rot_y[i] = rotation[i * 4u + 2u];
		rot_z[i] = rotation[i * 4u + 3u];

		const float sh_r = std::clamp(sh0[i * 3u + 0u] * GSX_PLY_SH_DC_TO_RGB_FACTOR + 0.5f, 0.0f, 1.0f);
		const float sh_g = std::clamp(sh0[i * 3u + 1u] * GSX_PLY_SH_DC_TO_RGB_FACTOR + 0.5f, 0.0f, 1.0f);
		const float sh_b = std::clamp(sh0[i * 3u + 2u] * GSX_PLY_SH_DC_TO_RGB_FACTOR + 0.5f, 0.0f, 1.0f);
		color_r[i] = (uint8_t)(sh_r * GSX_PLY_RGB_NORMALIZATION_FACTOR);
		color_g[i] = (uint8_t)(sh_g * GSX_PLY_RGB_NORMALIZATION_FACTOR);
		color_b[i] = (uint8_t)(sh_b * GSX_PLY_RGB_NORMALIZATION_FACTOR);
	}

	try {
		happly::PLYData ply_data;
		ply_data.addElement("vertex", (size_t)gaussian_count);
		ply_data.getElement("vertex").addProperty<float>("x", x);
		ply_data.getElement("vertex").addProperty<float>("y", y);
		ply_data.getElement("vertex").addProperty<float>("z", z);

		ply_data.getElement("vertex").addProperty<uint8_t>("red", color_r);
		ply_data.getElement("vertex").addProperty<uint8_t>("green", color_g);
		ply_data.getElement("vertex").addProperty<uint8_t>("blue", color_b);

		std::vector<float> f_dc_r((size_t)gaussian_count);
		std::vector<float> f_dc_g((size_t)gaussian_count);
		std::vector<float> f_dc_b((size_t)gaussian_count);
		for(size_t i = 0; i < (size_t)gaussian_count; ++i) {
			f_dc_r[i] = sh0[i * 3u + 0u];
			f_dc_g[i] = sh0[i * 3u + 1u];
			f_dc_b[i] = sh0[i * 3u + 2u];
		}
		ply_data.getElement("vertex").addProperty<float>("f_dc_r", f_dc_r);
		ply_data.getElement("vertex").addProperty<float>("f_dc_g", f_dc_g);
		ply_data.getElement("vertex").addProperty<float>("f_dc_b", f_dc_b);

		if(!sh1.empty()) {
			for(size_t coeff = 0; coeff < 3; ++coeff) {
				std::vector<float> rest_r((size_t)gaussian_count);
				std::vector<float> rest_g((size_t)gaussian_count);
				std::vector<float> rest_b((size_t)gaussian_count);
				for(size_t i = 0; i < (size_t)gaussian_count; ++i) {
					const size_t base = ((i * 3u) + coeff) * 3u;
					rest_r[i] = sh1[base + 0u];
					rest_g[i] = sh1[base + 1u];
					rest_b[i] = sh1[base + 2u];
				}
				const std::string coeff_name = std::to_string(coeff);
				ply_data.getElement("vertex").addProperty<float>("f_rest_r_" + coeff_name, rest_r);
				ply_data.getElement("vertex").addProperty<float>("f_rest_g_" + coeff_name, rest_g);
				ply_data.getElement("vertex").addProperty<float>("f_rest_b_" + coeff_name, rest_b);
			}
		}
		if(!sh2.empty()) {
			for(size_t coeff = 0; coeff < 5; ++coeff) {
				std::vector<float> rest_r((size_t)gaussian_count);
				std::vector<float> rest_g((size_t)gaussian_count);
				std::vector<float> rest_b((size_t)gaussian_count);
				for(size_t i = 0; i < (size_t)gaussian_count; ++i) {
					const size_t base = ((i * 5u) + coeff) * 3u;
					rest_r[i] = sh2[base + 0u];
					rest_g[i] = sh2[base + 1u];
					rest_b[i] = sh2[base + 2u];
				}
				const std::string coeff_name = std::to_string(coeff + 3u);
				ply_data.getElement("vertex").addProperty<float>("f_rest_r_" + coeff_name, rest_r);
				ply_data.getElement("vertex").addProperty<float>("f_rest_g_" + coeff_name, rest_g);
				ply_data.getElement("vertex").addProperty<float>("f_rest_b_" + coeff_name, rest_b);
			}
		}
		if(!sh3.empty()) {
			for(size_t coeff = 0; coeff < 7; ++coeff) {
				std::vector<float> rest_r((size_t)gaussian_count);
				std::vector<float> rest_g((size_t)gaussian_count);
				std::vector<float> rest_b((size_t)gaussian_count);
				for(size_t i = 0; i < (size_t)gaussian_count; ++i) {
					const size_t base = ((i * 7u) + coeff) * 3u;
					rest_r[i] = sh3[base + 0u];
					rest_g[i] = sh3[base + 1u];
					rest_b[i] = sh3[base + 2u];
				}
				const std::string coeff_name = std::to_string(coeff + 8u);
				ply_data.getElement("vertex").addProperty<float>("f_rest_r_" + coeff_name, rest_r);
				ply_data.getElement("vertex").addProperty<float>("f_rest_g_" + coeff_name, rest_g);
				ply_data.getElement("vertex").addProperty<float>("f_rest_b_" + coeff_name, rest_b);
			}
		}

		ply_data.getElement("vertex").addProperty<float>("opacity", opacity);
		ply_data.getElement("vertex").addProperty<float>("scale_0", scale_0);
		ply_data.getElement("vertex").addProperty<float>("scale_1", scale_1);
		ply_data.getElement("vertex").addProperty<float>("scale_2", scale_2);
		ply_data.getElement("vertex").addProperty<float>("rot_w", rot_w);
		ply_data.getElement("vertex").addProperty<float>("rot_x", rot_x);
		ply_data.getElement("vertex").addProperty<float>("rot_y", rot_y);
		ply_data.getElement("vertex").addProperty<float>("rot_z", rot_z);

		ply_data.write(filename, happly::DataFormat::Binary);
	} catch(const std::exception &e) {
		return gsx_make_error(GSX_ERROR_IO, e.what());
	}

	return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}


}