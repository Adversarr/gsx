#include "gsx/extra/gsx-flann.h"

#include "gsx/gsx-backend.h"

#include "gsx-impl.h"

#include "flann/nanoflann.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#ifdef GSX_HAS_OPENMP
#include <omp.h>
#endif

namespace {

constexpr gsx_float_t GSX_FLANN_NEIGHBOR_EPS_SQR = 1e-8f;
constexpr gsx_float_t GSX_FLANN_JACOBI_EPS = 1e-12f;
constexpr int GSX_FLANN_JACOBI_MAX_ITERS = 16;

struct gsx_flann_point_cloud_adaptor {
	const std::vector<gsx_float_t> &points;

	explicit gsx_flann_point_cloud_adaptor(const std::vector<gsx_float_t> &in_points)
		: points(in_points)
	{
	}

	inline size_t kdtree_get_point_count() const
	{
		return points.size() / 3u;
	}

	inline gsx_float_t kdtree_get_pt(const size_t idx, const size_t dim) const
	{
		return points[idx * 3u + dim];
	}

	template <class BBOX>
	bool kdtree_get_bbox(BBOX &) const
	{
		return false;
	}
};

using gsx_flann_kdtree = nanoflann::KDTreeSingleIndexAdaptor<
	nanoflann::L2_Simple_Adaptor<gsx_float_t, gsx_flann_point_cloud_adaptor>,
	gsx_flann_point_cloud_adaptor,
	3>;

gsx_error gsx_flann_get_backend_from_gs(gsx_gs_t gs, gsx_backend_t *out_backend)
{
	gsx_gs_info gs_info = {};
	gsx_error error = { GSX_ERROR_SUCCESS, NULL };

	if(out_backend == NULL) {
		return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_backend must be non-null");
	}
	*out_backend = NULL;

	error = gsx_gs_get_info(gs, &gs_info);
	if(!gsx_error_is_success(error)) {
		return error;
	}

	return gsx_arena_get_backend(gs_info.arena, out_backend);
}

gsx_error gsx_flann_download_field_f32(gsx_gs_t gs, gsx_gs_field field, gsx_index_t expected_cols, std::vector<gsx_float_t> *out_values, gsx_size_t *out_rows)
{
	gsx_tensor_t tensor = NULL;
	gsx_tensor_info info = {};
	gsx_size_t size_bytes = 0;
	gsx_error error = { GSX_ERROR_SUCCESS, NULL };

	if(out_values == NULL || out_rows == NULL) {
		return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "out_values and out_rows must be non-null");
	}
	out_values->clear();
	*out_rows = 0;

	error = gsx_gs_get_field(gs, field, &tensor);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_tensor_get_info(tensor, &info);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	if(info.rank != 2 || info.shape[1] != expected_cols) {
		return gsx_make_error(GSX_ERROR_INVALID_STATE, "gs tensor shape does not match expected layout");
	}
	if(info.data_type != GSX_DATA_TYPE_F32) {
		return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "flann recompute currently requires float32 gs fields");
	}

	error = gsx_tensor_get_size_bytes(tensor, &size_bytes);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	if(size_bytes % sizeof(gsx_float_t) != 0u) {
		return gsx_make_error(GSX_ERROR_INVALID_STATE, "gs tensor byte size is not float32-aligned");
	}

	out_values->resize((size_t)(size_bytes / sizeof(gsx_float_t)));
	error = gsx_tensor_download(tensor, out_values->data(), size_bytes);
	if(!gsx_error_is_success(error)) {
		return error;
	}

	*out_rows = (gsx_size_t)info.shape[0];
	return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
}

gsx_error gsx_flann_upload_field_f32(gsx_gs_t gs, gsx_gs_field field, gsx_index_t expected_cols, const std::vector<gsx_float_t> &values, gsx_size_t expected_rows)
{
	gsx_tensor_t tensor = NULL;
	gsx_tensor_info info = {};
	gsx_size_t size_bytes = 0;
	gsx_error error = { GSX_ERROR_SUCCESS, NULL };

	error = gsx_gs_get_field(gs, field, &tensor);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_tensor_get_info(tensor, &info);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	if(info.rank != 2 || info.shape[1] != expected_cols || (gsx_size_t)info.shape[0] != expected_rows) {
		return gsx_make_error(GSX_ERROR_INVALID_STATE, "gs tensor shape does not match expected layout");
	}
	if(info.data_type != GSX_DATA_TYPE_F32) {
		return gsx_make_error(GSX_ERROR_NOT_SUPPORTED, "flann recompute currently requires float32 gs fields");
	}

	error = gsx_tensor_get_size_bytes(tensor, &size_bytes);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	if(values.size() * sizeof(gsx_float_t) != (size_t)size_bytes) {
		return gsx_make_error(GSX_ERROR_INVALID_STATE, "flann output size does not match gs tensor layout");
	}

	return gsx_tensor_upload(tensor, values.data(), size_bytes);
}

gsx_float_t gsx_flann_clamp(gsx_float_t x, gsx_float_t lo, gsx_float_t hi)
{
	if(x < lo) {
		return lo;
	}
	if(x > hi) {
		return hi;
	}
	return x;
}

void gsx_flann_set_identity_rotation(std::vector<gsx_float_t> *rotation, size_t index)
{
	if(rotation == NULL) {
		return;
	}
	(*rotation)[index * 4u + 0u] = 1.0f;
	(*rotation)[index * 4u + 1u] = 0.0f;
	(*rotation)[index * 4u + 2u] = 0.0f;
	(*rotation)[index * 4u + 3u] = 0.0f;
}

void gsx_flann_set_uniform_logscale(std::vector<gsx_float_t> *logscale, size_t index, gsx_float_t log_value)
{
	if(logscale == NULL) {
		return;
	}
	(*logscale)[index * 3u + 0u] = log_value;
	(*logscale)[index * 3u + 1u] = log_value;
	(*logscale)[index * 3u + 2u] = log_value;
}

void gsx_flann_set_uniform_scale(
	std::vector<gsx_float_t> *logscale,
	std::vector<gsx_float_t> *rotation,
	size_t index,
	gsx_float_t dist,
	gsx_float_t init_scaling,
	gsx_float_t min_distance,
	gsx_float_t max_distance)
{
	const gsx_float_t scale = gsx_flann_clamp(dist * init_scaling, min_distance, max_distance);
	gsx_flann_set_uniform_logscale(logscale, index, gsx_logf(scale));
	gsx_flann_set_identity_rotation(rotation, index);
}

bool gsx_flann_compute_neighbors(
	const std::vector<gsx_float_t> &means,
	const gsx_flann_kdtree &index,
	size_t point_index,
	gsx_index_t num_neighbors,
	gsx_float_t radius,
	std::vector<size_t> *out_neighbors,
	std::vector<gsx_float_t> *out_neighbor_dists_sqr)
{
	const size_t point_count = means.size() / 3u;
	const size_t max_results = std::min((size_t)num_neighbors + 1u, point_count);
	const gsx_float_t query_pt[3] = {
		means[point_index * 3u + 0u],
		means[point_index * 3u + 1u],
		means[point_index * 3u + 2u]
	};
	const gsx_float_t radius_sqr = radius > 0.0f ? radius * radius : std::numeric_limits<gsx_float_t>::infinity();
	std::vector<size_t> ret_indices(max_results);
	std::vector<gsx_float_t> out_dists_sqr(max_results, 0.0f);
	nanoflann::KNNResultSet<gsx_float_t> result_set(max_results);

	if(out_neighbors == NULL || out_neighbor_dists_sqr == NULL) {
		return false;
	}
	out_neighbors->clear();
	out_neighbor_dists_sqr->clear();
	result_set.init(ret_indices.data(), out_dists_sqr.data());
	index.findNeighbors(result_set, query_pt, nanoflann::SearchParameters(10));
	for(size_t j = 0u; j < max_results && out_neighbors->size() < (size_t)num_neighbors; ++j) {
		if(ret_indices[j] == point_index) {
			continue;
		}
		if(out_dists_sqr[j] <= GSX_FLANN_NEIGHBOR_EPS_SQR) {
			continue;
		}
		if(out_dists_sqr[j] > radius_sqr) {
			continue;
		}
		out_neighbors->push_back(ret_indices[j]);
		out_neighbor_dists_sqr->push_back(out_dists_sqr[j]);
	}
	return true;
}

bool gsx_flann_compute_covariance(
	const std::vector<gsx_float_t> &means,
	size_t point_index,
	const std::vector<size_t> &neighbors,
	const std::vector<gsx_float_t> &neighbor_dists_sqr,
	gsx_float_t min_distance,
	std::array<gsx_float_t, 9u> *out_covariance)
{
	const double eps = 1e-12;
	double weight_sum = 0.0;
	double mean_x = 0.0;
	double mean_y = 0.0;
	double mean_z = 0.0;
	std::vector<double> weights;

	if(out_covariance == NULL || neighbors.size() != neighbor_dists_sqr.size() || neighbors.size() < 3u) {
		return false;
	}
	weights.reserve(neighbors.size());
	for(size_t i = 0u; i < neighbors.size(); ++i) {
		const size_t neighbor_index = neighbors[i];
		const double weight = 1.0 / ((double)neighbor_dists_sqr[i] + eps);
		const double px = means[neighbor_index * 3u + 0u];
		const double py = means[neighbor_index * 3u + 1u];
		const double pz = means[neighbor_index * 3u + 2u];
		weights.push_back(weight);
		weight_sum += weight;
		mean_x += weight * px;
		mean_y += weight * py;
		mean_z += weight * pz;
	}
	if(weight_sum <= eps) {
		return false;
	}
	mean_x /= weight_sum;
	mean_y /= weight_sum;
	mean_z /= weight_sum;

	double cxx = 0.0;
	double cxy = 0.0;
	double cxz = 0.0;
	double cyy = 0.0;
	double cyz = 0.0;
	double czz = 0.0;
	for(size_t i = 0u; i < neighbors.size(); ++i) {
		const size_t neighbor_index = neighbors[i];
		const double dx = means[neighbor_index * 3u + 0u] - mean_x;
		const double dy = means[neighbor_index * 3u + 1u] - mean_y;
		const double dz = means[neighbor_index * 3u + 2u] - mean_z;
		const double weight = weights[i];
		cxx += weight * dx * dx;
		cxy += weight * dx * dy;
		cxz += weight * dx * dz;
		cyy += weight * dy * dy;
		cyz += weight * dy * dz;
		czz += weight * dz * dz;
	}

	const double inv_weight_sum = 1.0 / weight_sum;
	const gsx_float_t regularization = std::max(min_distance * min_distance, (gsx_float_t)1e-12f);
	(*out_covariance)[0] = (gsx_float_t)(cxx * inv_weight_sum + regularization);
	(*out_covariance)[1] = (gsx_float_t)(cxy * inv_weight_sum);
	(*out_covariance)[2] = (gsx_float_t)(cxz * inv_weight_sum);
	(*out_covariance)[3] = (gsx_float_t)(cxy * inv_weight_sum);
	(*out_covariance)[4] = (gsx_float_t)(cyy * inv_weight_sum + regularization);
	(*out_covariance)[5] = (gsx_float_t)(cyz * inv_weight_sum);
	(*out_covariance)[6] = (gsx_float_t)(cxz * inv_weight_sum);
	(*out_covariance)[7] = (gsx_float_t)(cyz * inv_weight_sum);
	(*out_covariance)[8] = (gsx_float_t)(czz * inv_weight_sum + regularization);
	(void)point_index;
	return true;
}

void gsx_flann_jacobi_rotate(std::array<gsx_float_t, 9u> *matrix, std::array<gsx_float_t, 9u> *eigenvectors, int p, int q)
{
	const gsx_float_t app = (*matrix)[(size_t)p * 3u + (size_t)p];
	const gsx_float_t aqq = (*matrix)[(size_t)q * 3u + (size_t)q];
	const gsx_float_t apq = (*matrix)[(size_t)p * 3u + (size_t)q];
	const gsx_float_t tau = (aqq - app) / (2.0f * apq);
	const gsx_float_t t = tau >= 0.0f
		? 1.0f / (tau + std::sqrt(1.0f + tau * tau))
		: -1.0f / (-tau + std::sqrt(1.0f + tau * tau));
	const gsx_float_t c = 1.0f / std::sqrt(1.0f + t * t);
	const gsx_float_t s = t * c;

	for(int k = 0; k < 3; ++k) {
		const gsx_float_t mkp = (*matrix)[(size_t)k * 3u + (size_t)p];
		const gsx_float_t mkq = (*matrix)[(size_t)k * 3u + (size_t)q];
		(*matrix)[(size_t)k * 3u + (size_t)p] = c * mkp - s * mkq;
		(*matrix)[(size_t)k * 3u + (size_t)q] = s * mkp + c * mkq;
	}
	for(int k = 0; k < 3; ++k) {
		const gsx_float_t mpk = (*matrix)[(size_t)p * 3u + (size_t)k];
		const gsx_float_t mqk = (*matrix)[(size_t)q * 3u + (size_t)k];
		(*matrix)[(size_t)p * 3u + (size_t)k] = c * mpk - s * mqk;
		(*matrix)[(size_t)q * 3u + (size_t)k] = s * mpk + c * mqk;
	}
	for(int k = 0; k < 3; ++k) {
		const gsx_float_t vkp = (*eigenvectors)[(size_t)k * 3u + (size_t)p];
		const gsx_float_t vkq = (*eigenvectors)[(size_t)k * 3u + (size_t)q];
		(*eigenvectors)[(size_t)k * 3u + (size_t)p] = c * vkp - s * vkq;
		(*eigenvectors)[(size_t)k * 3u + (size_t)q] = s * vkp + c * vkq;
	}
}

bool gsx_flann_symmetric_eigen_decomposition(
	const std::array<gsx_float_t, 9u> &covariance,
	std::array<gsx_float_t, 3u> *out_eigenvalues,
	std::array<gsx_float_t, 9u> *out_eigenvectors)
{
	std::array<gsx_float_t, 9u> matrix = covariance;
	std::array<gsx_float_t, 9u> eigenvectors = {
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f
	};

	if(out_eigenvalues == NULL || out_eigenvectors == NULL) {
		return false;
	}
	for(int iter = 0; iter < GSX_FLANN_JACOBI_MAX_ITERS; ++iter) {
		int p = 0;
		int q = 1;
		gsx_float_t max_offdiag = std::fabs(matrix[1]);
		if(std::fabs(matrix[2]) > max_offdiag) {
			p = 0;
			q = 2;
			max_offdiag = std::fabs(matrix[2]);
		}
		if(std::fabs(matrix[5]) > max_offdiag) {
			p = 1;
			q = 2;
			max_offdiag = std::fabs(matrix[5]);
		}
		if(max_offdiag <= GSX_FLANN_JACOBI_EPS) {
			break;
		}
		gsx_flann_jacobi_rotate(&matrix, &eigenvectors, p, q);
	}

	(*out_eigenvalues)[0] = matrix[0];
	(*out_eigenvalues)[1] = matrix[4];
	(*out_eigenvalues)[2] = matrix[8];
	*out_eigenvectors = eigenvectors;
	return true;
}

void gsx_flann_swap_columns(std::array<gsx_float_t, 9u> *matrix, size_t a, size_t b)
{
	for(size_t row = 0u; row < 3u; ++row) {
		std::swap((*matrix)[row * 3u + a], (*matrix)[row * 3u + b]);
	}
}

void gsx_flann_sort_eigensystem(std::array<gsx_float_t, 3u> *eigenvalues, std::array<gsx_float_t, 9u> *eigenvectors)
{
	if(eigenvalues == NULL || eigenvectors == NULL) {
		return;
	}
	for(size_t i = 0u; i < 3u; ++i) {
		for(size_t j = i + 1u; j < 3u; ++j) {
			if((*eigenvalues)[j] > (*eigenvalues)[i]) {
				std::swap((*eigenvalues)[i], (*eigenvalues)[j]);
				gsx_flann_swap_columns(eigenvectors, i, j);
			}
		}
	}
}

gsx_float_t gsx_flann_matrix_determinant(const std::array<gsx_float_t, 9u> &matrix)
{
	return matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
		- matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
		+ matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
}

void gsx_flann_fix_handedness(std::array<gsx_float_t, 9u> *eigenvectors)
{
	if(eigenvectors == NULL) {
		return;
	}
	if(gsx_flann_matrix_determinant(*eigenvectors) < 0.0f) {
		(*eigenvectors)[2] = -(*eigenvectors)[2];
		(*eigenvectors)[5] = -(*eigenvectors)[5];
		(*eigenvectors)[8] = -(*eigenvectors)[8];
	}
}

void gsx_flann_matrix_to_quaternion(
	const std::array<gsx_float_t, 9u> &matrix,
	gsx_float_t *out_w,
	gsx_float_t *out_x,
	gsx_float_t *out_y,
	gsx_float_t *out_z)
{
	const gsx_float_t trace = matrix[0] + matrix[4] + matrix[8];

	if(out_w == NULL || out_x == NULL || out_y == NULL || out_z == NULL) {
		return;
	}
	if(trace > 0.0f) {
		const gsx_float_t s = 2.0f * std::sqrt(trace + 1.0f);
		*out_w = 0.25f * s;
		*out_x = (matrix[7] - matrix[5]) / s;
		*out_y = (matrix[2] - matrix[6]) / s;
		*out_z = (matrix[3] - matrix[1]) / s;
	} else if(matrix[0] > matrix[4] && matrix[0] > matrix[8]) {
		const gsx_float_t s = 2.0f * std::sqrt(1.0f + matrix[0] - matrix[4] - matrix[8]);
		*out_w = (matrix[7] - matrix[5]) / s;
		*out_x = 0.25f * s;
		*out_y = (matrix[1] + matrix[3]) / s;
		*out_z = (matrix[2] + matrix[6]) / s;
	} else if(matrix[4] > matrix[8]) {
		const gsx_float_t s = 2.0f * std::sqrt(1.0f + matrix[4] - matrix[0] - matrix[8]);
		*out_w = (matrix[2] - matrix[6]) / s;
		*out_x = (matrix[1] + matrix[3]) / s;
		*out_y = 0.25f * s;
		*out_z = (matrix[5] + matrix[7]) / s;
	} else {
		const gsx_float_t s = 2.0f * std::sqrt(1.0f + matrix[8] - matrix[0] - matrix[4]);
		*out_w = (matrix[3] - matrix[1]) / s;
		*out_x = (matrix[2] + matrix[6]) / s;
		*out_y = (matrix[5] + matrix[7]) / s;
		*out_z = 0.25f * s;
	}
	const gsx_float_t norm = std::sqrt((*out_w) * (*out_w) + (*out_x) * (*out_x) + (*out_y) * (*out_y) + (*out_z) * (*out_z));
	if(norm > 1e-8f) {
		const gsx_float_t inv_norm = 1.0f / norm;
		*out_w *= inv_norm;
		*out_x *= inv_norm;
		*out_y *= inv_norm;
		*out_z *= inv_norm;
	}
	if(*out_w < 0.0f) {
		*out_w = -*out_w;
		*out_x = -*out_x;
		*out_y = -*out_y;
		*out_z = -*out_z;
	}
}

bool gsx_flann_set_anisotropic_scale_rotation(
	std::vector<gsx_float_t> *logscale,
	std::vector<gsx_float_t> *rotation,
	size_t index,
	const std::array<gsx_float_t, 9u> &covariance,
	gsx_float_t init_scaling,
	gsx_float_t min_distance,
	gsx_float_t max_distance)
{
	std::array<gsx_float_t, 3u> eigenvalues = {};
	std::array<gsx_float_t, 9u> eigenvectors = {};
	gsx_float_t qw = 1.0f;
	gsx_float_t qx = 0.0f;
	gsx_float_t qy = 0.0f;
	gsx_float_t qz = 0.0f;

	if(logscale == NULL || rotation == NULL) {
		return false;
	}
	if(!gsx_flann_symmetric_eigen_decomposition(covariance, &eigenvalues, &eigenvectors)) {
		return false;
	}
	gsx_flann_sort_eigensystem(&eigenvalues, &eigenvectors);
	gsx_flann_fix_handedness(&eigenvectors);
	gsx_flann_matrix_to_quaternion(eigenvectors, &qw, &qx, &qy, &qz);
	(*rotation)[index * 4u + 0u] = qw;
	(*rotation)[index * 4u + 1u] = qx;
	(*rotation)[index * 4u + 2u] = qy;
	(*rotation)[index * 4u + 3u] = qz;
	for(size_t axis = 0u; axis < 3u; ++axis) {
		const gsx_float_t axis_scale = gsx_flann_clamp(
			(gsx_float_t)std::sqrt(std::max(eigenvalues[axis], (gsx_float_t)1e-12f)) * init_scaling,
			min_distance,
			max_distance);
		(*logscale)[index * 3u + axis] = gsx_logf(axis_scale);
	}
	return true;
}

} // namespace

extern "C" {

gsx_error gsx_gs_recompute_scale_rotation_flann(
	gsx_gs_t in_out_gs,
	gsx_index_t num_neighbors,
	gsx_float_t init_scaling,
	gsx_float_t min_distance,
	gsx_float_t max_distance,
	gsx_float_t default_distance,
	gsx_float_t radius,
	bool use_anisotropic)
{
	gsx_backend_t backend = NULL;
	std::vector<gsx_float_t> means;
	std::vector<gsx_float_t> logscale;
	std::vector<gsx_float_t> rotation;
	std::vector<size_t> neighbors;
	std::vector<gsx_float_t> neighbor_dists_sqr;
	gsx_size_t gaussian_count = 0;
	gsx_error error = { GSX_ERROR_SUCCESS, NULL };

	if(in_out_gs == NULL) {
		return gsx_make_error(GSX_ERROR_INVALID_ARGUMENT, "in_out_gs must be non-null");
	}
	if(num_neighbors <= 0) {
		return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "num_neighbors must be > 0");
	}
	if(!(init_scaling > 0.0f) || !(min_distance > 0.0f) || !(max_distance >= min_distance) || !(default_distance > 0.0f)) {
		return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "invalid scale initialization parameters");
	}
	if(!(radius >= 0.0f)) {
		return gsx_make_error(GSX_ERROR_OUT_OF_RANGE, "radius must be >= 0");
	}

	error = gsx_flann_get_backend_from_gs(in_out_gs, &backend);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_backend_major_stream_sync(backend);
	if(!gsx_error_is_success(error)) {
		return error;
	}

	error = gsx_flann_download_field_f32(in_out_gs, GSX_GS_FIELD_MEAN3D, 3, &means, &gaussian_count);
	if(!gsx_error_is_success(error)) {
		return error;
	}

	logscale.assign((size_t)gaussian_count * 3u, 0.0f);
	rotation.assign((size_t)gaussian_count * 4u, 0.0f);

	if(gaussian_count == 0u) {
		return gsx_make_error(GSX_ERROR_SUCCESS, NULL);
	}

	if(gaussian_count == 1u) {
		gsx_flann_set_uniform_scale(&logscale, &rotation, 0u, default_distance, init_scaling, min_distance, max_distance);
		error = gsx_flann_upload_field_f32(in_out_gs, GSX_GS_FIELD_LOGSCALE, 3, logscale, gaussian_count);
		if(!gsx_error_is_success(error)) {
			return error;
		}
		error = gsx_flann_upload_field_f32(in_out_gs, GSX_GS_FIELD_ROTATION, 4, rotation, gaussian_count);
		if(!gsx_error_is_success(error)) {
			return error;
		}
		return gsx_backend_major_stream_sync(backend);
	}

	gsx_flann_point_cloud_adaptor cloud(means);
	gsx_flann_kdtree index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
	index.buildIndex();
#ifdef GSX_HAS_OPENMP
#pragma omp parallel for
#endif
	for(size_t i = 0; i < (size_t)gaussian_count; ++i) {
		gsx_float_t sum_dist = 0.0f;
		std::array<gsx_float_t, 9u> covariance = {};

		if(!gsx_flann_compute_neighbors(means, index, i, num_neighbors, radius, &neighbors, &neighbor_dists_sqr)) {
			return gsx_make_error(GSX_ERROR_INVALID_STATE, "failed to compute local neighbors");
		}
		for(size_t j = 0u; j < neighbors.size(); ++j) {
			sum_dist += (gsx_float_t)std::sqrt(neighbor_dists_sqr[j]);
		}

		gsx_float_t dist = default_distance;
		if(!neighbors.empty()) {
			dist = sum_dist / (gsx_float_t)neighbors.size();
		}
		if(use_anisotropic
			&& gsx_flann_compute_covariance(means, i, neighbors, neighbor_dists_sqr, min_distance, &covariance)
			&& gsx_flann_set_anisotropic_scale_rotation(&logscale, &rotation, i, covariance, init_scaling, min_distance, max_distance)) {
			continue;
		}
		gsx_flann_set_uniform_scale(&logscale, &rotation, i, dist, init_scaling, min_distance, max_distance);
	}

	error = gsx_flann_upload_field_f32(in_out_gs, GSX_GS_FIELD_LOGSCALE, 3, logscale, gaussian_count);
	if(!gsx_error_is_success(error)) {
		return error;
	}
	error = gsx_flann_upload_field_f32(in_out_gs, GSX_GS_FIELD_ROTATION, 4, rotation, gaussian_count);
	if(!gsx_error_is_success(error)) {
		return error;
	}

	return gsx_backend_major_stream_sync(backend);
}

}
