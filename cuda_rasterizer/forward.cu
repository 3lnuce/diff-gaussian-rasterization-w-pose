/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include "helper_math.h"
#include "math.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// #define USE_LIST

template <typename T>
__device__ void inline reduce_helper(int lane, int i, T *data) {
  if (lane < i) {
    data[lane] += data[lane + i];
  }
}

template <typename group_t, typename... Lists>
__device__ void render_cuda_reduce_sum(group_t g, Lists... lists) {
  int lane = g.thread_rank();
  g.sync();

  for (int i = g.size() / 2; i > 0; i /= 2) {
    (...,
     reduce_helper(
         lane, i, lists)); // Fold expression: apply reduce_helper for each list
    g.sync();
  }
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for
	// "Differentiable Point-Based Radiance Fields for
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002).
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters.
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles.
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depth,
	float* __restrict__ out_depth,
	float* __restrict__ out_opacity,
	int * __restrict__ n_touched)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	// uint32_t horizontal_blocks = gridDim.x; # TODO Maybe it's different?
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float D = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_depth[block.thread_rank()] = depth[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f) {
				continue;
			}
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++) {
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			}
			D += collected_depth[j] * alpha * T;
			// Keep track of how many pixels touched this Gaussian.
			if (test_T > 0.5f) {
				atomicAdd(&(n_touched[collected_id[j]]), 1);
			}
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++) {
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		}
		out_depth[pix_id] = D;
		out_opacity[pix_id] = 1 - T;
	}
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDAFast(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depth,
	float* __restrict__ out_depth,
	float* __restrict__ out_opacity,
	int * __restrict__ n_touched,
	int * __restrict__ tile_active,
	int * __restrict__ tile_active_list)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	// uint32_t horizontal_blocks = gridDim.x; # TODO Maybe it's different?

#ifdef USE_LIST
	uint32_t tile_idx = tile_active_list[block.group_index().x];
	int block_coord_y = tile_idx / horizontal_blocks;
	int block_coord_x = tile_idx % horizontal_blocks;
#else
	uint32_t tile_idx = block.group_index().y * gridDim.x + block.group_index().x;
	int block_coord_x = block.group_index().x;
	int block_coord_y = block.group_index().y;
	if (tile_active[tile_idx] == 0)
		return;
#endif

	uint2 pix_min = { block_coord_x * BLOCK_X, block_coord_y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// printf("pix_min: %d, %d, pix: %d, %d\n", pix_min.x, pix_min.y, pix.x, pix.y);

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// uint32_t tile_idx = block.group_index().y * gridDim.x + block.group_index().x;
	// printf ("debug tile_id: %d, %d, %d\n", block.group_index().x, block.group_index().y, tile_idx);

	// if (tile_active[tile_idx] == 0)
	// {
	// 	// printf("wrong tile %d - %d - %d!!!\n", block.group_index().y, block.group_index().x, tile_idx);
	// 	done = true;
	// }

	// if (pix_min.x % 32 == 0)// && pix_min.y % 32 == 0)
	// 	done = true;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block_coord_y * horizontal_blocks + block_coord_x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// printf("pix_id: %d\n", pix_id);
	// printf("todo: %d\n", toDo);
	// printf("range: %d - %d\n", range.x, range.y);

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float D = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_depth[block.thread_rank()] = depth[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f) {
				continue;
			}
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++) {
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			}
			D += collected_depth[j] * alpha * T;
			// Keep track of how many pixels touched this Gaussian.
			if (test_T > 0.5f) {
				atomicAdd(&(n_touched[collected_id[j]]), 1);
			}
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++) {
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		}
		out_depth[pix_id] = D;
		out_opacity[pix_id] = 1 - T;
	}
}

// // Main rasterization method. Collaboratively works on one tile per
// // block, each thread treats one pixel. Alternates between fetching
// // and rasterizing data.
// template <uint32_t CHANNELS>
// __global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
// renderCUDAFused(
// 	const uint2* __restrict__ ranges,
// 	const uint32_t* __restrict__ point_list,
// 	int W, int H,
// 	const float* __restrict__ ref_color,
// 	const double* __restrict__ ref_depth,
// 	const float2* __restrict__ points_xy_image,
// 	const float* __restrict__ features,
// 	const float4* __restrict__ conic_opacity,
// 	float* __restrict__ final_T,
// 	uint32_t* __restrict__ n_contrib,
// 	const float* __restrict__ bg_color,
// 	float* __restrict__ out_color,
// 	const float* __restrict__ depth,
// 	float* __restrict__ out_depth,
// 	float* __restrict__ out_opacity,
// 	int * __restrict__ n_touched,
// 	int * __restrict__ tile_active,
// 	int * __restrict__ tile_active_list,
// 	float3* __restrict__ dL_dmean2D,
// 	float4* __restrict__ dL_dconic2D,
// 	float* __restrict__ dL_dopacity,
// 	float* __restrict__ dL_dcolors,
// 	float* __restrict__ dL_ddepths)
// {
// 	// Identify current tile and associated min/max pixel range.
// 	auto block = cg::this_thread_block();
// 	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
// 	// uint32_t horizontal_blocks = gridDim.x; # TODO Maybe it's different?

// #ifdef USE_LIST
// 	uint32_t tile_idx = tile_active_list[block.group_index().x];
// 	int block_coord_y = tile_idx / horizontal_blocks;
// 	int block_coord_x = tile_idx % horizontal_blocks;
// #else
// 	uint32_t tile_idx = block.group_index().y * gridDim.x + block.group_index().x;
// 	int block_coord_x = block.group_index().x;
// 	int block_coord_y = block.group_index().y;
// 	if (tile_active[tile_idx] == 0)
// 		return;
// #endif

// 	uint2 pix_min = { block_coord_x * BLOCK_X, block_coord_y * BLOCK_Y };
// 	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
// 	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
// 	uint32_t pix_id = W * pix.y + pix.x;
// 	float2 pixf = { (float)pix.x, (float)pix.y };

// 	// if (pix_id == 1199)
// 	// 	printf ("pix id 1199, val %f, %f\n", ref_img[0 * 680 * 1200 + 1199], ref_img[1 * 680 * 1200 + 1199]);
// 	// if (pix_id == 1200)
// 	// 	printf ("pix id 1200, val %f, %f\n", ref_img[0 * 680 * 1200 + 1200], ref_img[1 * 680 * 1200 + 1200]);

// 	// printf("pix_min: %d, %d, pix: %d, %d\n", pix_min.x, pix_min.y, pix.x, pix.y);

// 	// Check if this thread is associated with a valid pixel or outside.
// 	bool inside = pix.x < W&& pix.y < H;
// 	// Done threads can help with fetching, but don't rasterize
// 	bool done = !inside;

// 	// uint32_t tile_idx = block.group_index().y * gridDim.x + block.group_index().x;
// 	// printf ("debug tile_id: %d, %d, %d\n", block.group_index().x, block.group_index().y, tile_idx);

// 	// if (tile_active[tile_idx] == 0)
// 	// {
// 	// 	// printf("wrong tile %d - %d - %d!!!\n", block.group_index().y, block.group_index().x, tile_idx);
// 	// 	done = true;
// 	// }

// 	// if (pix_min.x % 32 == 0)// && pix_min.y % 32 == 0)
// 	// 	done = true;

// 	// Load start/end range of IDs to process in bit sorted list.
// 	uint2 range = ranges[block_coord_y * horizontal_blocks + block_coord_x];
// 	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
// 	int toDo = range.y - range.x;

// 	// printf("pix_id: %d\n", pix_id);
// 	// printf("todo: %d\n", toDo);
// 	// printf("range: %d - %d\n", range.x, range.y);

// 	// Allocate storage for batches of collectively fetched data.
// 	__shared__ int collected_id[BLOCK_SIZE];
// 	__shared__ float2 collected_xy[BLOCK_SIZE];
// 	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
// 	__shared__ float collected_depth[BLOCK_SIZE];

// 	// Initialize helper variables
// 	float T = 1.0f;
// 	uint32_t contributor = 0;
// 	uint32_t last_contributor = 0;
// 	float C[CHANNELS] = { 0 };
// 	float D = 0.0f;

// 	// Iterate over batches until all done or range is complete
// 	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
// 	{
// 		// End if entire block votes that it is done rasterizing
// 		int num_done = __syncthreads_count(done);
// 		if (num_done == BLOCK_SIZE)
// 			break;

// 		// Collectively fetch per-Gaussian data from global to shared
// 		int progress = i * BLOCK_SIZE + block.thread_rank();
// 		if (range.x + progress < range.y)
// 		{
// 			int coll_id = point_list[range.x + progress];
// 			collected_id[block.thread_rank()] = coll_id;
// 			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
// 			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
// 			collected_depth[block.thread_rank()] = depth[coll_id];
// 		}
// 		block.sync();

// 		// Iterate over current batch
// 		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
// 		{
// 			// Keep track of current position in range
// 			contributor++;

// 			// Resample using conic matrix (cf. "Surface
// 			// Splatting" by Zwicker et al., 2001)
// 			float2 xy = collected_xy[j];
// 			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
// 			float4 con_o = collected_conic_opacity[j];
// 			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
// 			if (power > 0.0f)
// 				continue;

// 			// Eq. (2) from 3D Gaussian splatting paper.
// 			// Obtain alpha by multiplying with Gaussian opacity
// 			// and its exponential falloff from mean.
// 			// Avoid numerical instabilities (see paper appendix).
// 			float alpha = min(0.99f, con_o.w * exp(power));
// 			if (alpha < 1.0f / 255.0f) {
// 				continue;
// 			}
// 			float test_T = T * (1 - alpha);
// 			if (test_T < 0.0001f)
// 			{
// 				done = true;
// 				continue;
// 			}
// 			// Eq. (3) from 3D Gaussian splatting paper.
// 			for (int ch = 0; ch < CHANNELS; ch++) {
// 				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
// 			}
// 			D += collected_depth[j] * alpha * T;
// 			// Keep track of how many pixels touched this Gaussian.
// 			if (test_T > 0.5f) {
// 				atomicAdd(&(n_touched[collected_id[j]]), 1);
// 			}
// 			T = test_T;

// 			// Keep track of last range entry to update this
// 			// pixel.
// 			last_contributor = contributor;
// 		}
// 	}

// 	// float tgt_pix = 0.0;
// 	// All threads that treat valid pixel write out their final
// 	// rendering data to the frame and auxiliary buffers.
// 	if (inside)
// 	{
// 		final_T[pix_id] = T;
// 		n_contrib[pix_id] = last_contributor;
// 		for (int ch = 0; ch < CHANNELS; ch++) {
// 			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
// 			// tgt_pix += out_color[ch * H * W + pix_id];
// 		}
// 		out_depth[pix_id] = D;
// 		out_opacity[pix_id] = 1 - T;
// 	}

// 	// tgt_pix = tgt_pix / 3.f;
// 	// printf ("sanity check for ref and tgt img at %d, %f, %f\n", pix_id, ref_img[pix_id], tgt_pix);

//     // from slam_utils get_loss_mapping_rgbd
// 	// when this error is postive, the gradient is also positive
//     // l1_rgb = torch.abs(image - gt_image)
// 	// gradient for color pixel is (1 * 0.95) / (1200 * 680 * 3) ~= 3.881e-7
//     // gradient for depth pixel is (1 * 0.05) / (1200 * 680) ~= 6.127e-8
// 	// float dL_dpixels = 0.0f;
// 	// float dL_dpixels_depth = 0.0f;

// 	// if (tgt_pix > ref_img[pix_id])
// 	// {
// 	// 	dL_dpixels = 3.881e-7;
// 	// 	dL_dpixels_depth = 0.0f;//6.127e-8;
// 	// }
// 	// else if (tgt_pix == ref_img[pix_id])
// 	// {
// 	// 	dL_dpixels = 0.0f;
// 	// 	dL_dpixels_depth = 0.0f;
// 	// }
// 	// else // (tgt_pix < ref_img[pix_id])
// 	// {
// 	// 	dL_dpixels = -3.881e-7;
// 	// 	dL_dpixels_depth = 0.0f;//-6.127e-8;
// 	// }

// 	// TODO: left for depth
// 	// if (tgt_pix > ref_img[pix_id])
// 	// 	dL_dpixels = 3.881e-7;
// 	// else if (tgt_pix == ref_img[pix_id])
// 	// 	dL_dpixels = 0.0f;
// 	// else (tgt_pix < ref_img[pix_id])
// 	// 	dL_dpixels = -3.881e-7;

// 	auto tid = block.thread_rank();

// 	block.sync();

// 	done = !inside;
// 	toDo = range.y - range.x;

// 	__shared__ float2 dL_dmean2D_shared[BLOCK_SIZE];
// 	__shared__ float3 dL_dcolors_shared[BLOCK_SIZE];
// 	__shared__ float dL_ddepths_shared[BLOCK_SIZE];
// 	__shared__ float dL_dopacity_shared[BLOCK_SIZE];
// 	__shared__ float4 dL_dconic2D_shared[BLOCK_SIZE];

// 	__shared__ float collected_colors[CHANNELS * BLOCK_SIZE];
// 	__shared__ float collected_depths[BLOCK_SIZE];

// 	// In the forward, we stored the final value for T, the
// 	// product of all (1 - alpha) factors.
// 	const float T_final = inside ? final_T[pix_id] : 0;
// 	T = T_final;

// 	// We start from the back. The ID of the last contributing
// 	// Gaussian is known from each pixel from the forward.
// 	contributor = toDo;
// 	last_contributor = inside ? n_contrib[pix_id] : 0;

// 	float accum_rec[CHANNELS] = { 0.f };
// 	float dL_dpixel[CHANNELS] = { 0.f };
// 	float accum_rec_depth = 0.f;
// 	float dL_dpixel_depth = 0.f;

// 	if (inside) {
// 		#pragma unroll
// 		for (int ch = 0; ch < CHANNELS; ch++) {
// 			// dL_dpixel[i] = dL_dpixels;
// 			dL_dpixel[ch] = (out_color[ch * H * W + pix_id] > ref_color[ch * H * W + pix_id]) ? 3.881e-7 : -3.881e-7;
// 		}
// 		// dL_dpixel_depth = dL_dpixels_depth;
// 		// dL_dpixel_depth = 0.0f;
// 		dL_dpixel_depth = (out_depth[pix_id] > ref_depth[pix_id]) ? 6.127e-8 : -6.127e-8;
// 	}

// 	float last_alpha = 0.f;
// 	float last_color[CHANNELS] = { 0.f };
// 	float last_depth = 0.f;

// 	// Gradient of pixel coordinate w.r.t. normalized
// 	// screen-space viewport corrdinates (-1 to 1)
// 	const float ddelx_dx = 0.5f * W;
// 	const float ddely_dy = 0.5f * H;
// 	__shared__ int skip_counter;

// 	// Traverse all Gaussians
// 	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
// 	{
// 		// Load auxiliary data into shared memory, start in the BACK
// 		// and load them in revers order.
// 		// block.sync();
// 		const int progress = i * BLOCK_SIZE + tid;
// 		if (range.x + progress < range.y)
// 		{
// 			const int coll_id = point_list[range.y - progress - 1];
// 			collected_id[tid] = coll_id;
// 			collected_xy[tid] = points_xy_image[coll_id];
// 			collected_conic_opacity[tid] = conic_opacity[coll_id];
// 			#pragma unroll
// 			for (int i = 0; i < CHANNELS; i++) {
// 				collected_colors[i * BLOCK_SIZE + tid] = features[coll_id * CHANNELS + i];

// 			}
// 			collected_depths[tid] = depth[coll_id];
// 		}
// 		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++) {
// 			block.sync();
// 			if (tid == 0) {
// 				skip_counter = 0;
// 			}
// 			block.sync();

// 			// Keep track of current Gaussian ID. Skip, if this one
// 			// is behind the last contributor for this pixel.
// 			bool skip = done;
// 			contributor = done ? contributor : contributor - 1;
// 			skip |= contributor >= last_contributor;

// 			// Compute blending values, as before.
// 			const float2 xy = collected_xy[j];
// 			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
// 			const float4 con_o = collected_conic_opacity[j];
// 			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
// 			skip |= power > 0.0f;

// 			const float G = exp(power);
// 			const float alpha = min(0.99f, con_o.w * G);
// 			skip |= alpha < 1.0f / 255.0f;

// 			if (skip) {
// 				atomicAdd(&skip_counter, 1);
// 			}
// 			block.sync();
// 			if (skip_counter == BLOCK_SIZE) {
// 				continue;
// 			}


// 			T = skip ? T : T / (1.f - alpha);
// 			const float dchannel_dcolor = alpha * T;

// 			// Propagate gradients to per-Gaussian colors and keep
// 			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
// 			// pair).
// 			float dL_dalpha = 0.0f;
// 			const int global_id = collected_id[j];
// 			float local_dL_dcolors[3];
// 			#pragma unroll
// 			for (int ch = 0; ch < CHANNELS; ch++)
// 			{
// 				const float c = collected_colors[ch * BLOCK_SIZE + j];
// 				// Update last color (to be used in the next iteration)
// 				accum_rec[ch] = skip ? accum_rec[ch] : last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
// 				last_color[ch] = skip ? last_color[ch] : c;

// 				const float dL_dchannel = dL_dpixel[ch];
// 				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
// 				local_dL_dcolors[ch] = skip ? 0.0f : dchannel_dcolor * dL_dchannel;
// 			}
// 			dL_dcolors_shared[tid].x = local_dL_dcolors[0];
// 			dL_dcolors_shared[tid].y = local_dL_dcolors[1];
// 			dL_dcolors_shared[tid].z = local_dL_dcolors[2];

// 			const float depth = collected_depths[j];
// 			accum_rec_depth = skip ? accum_rec_depth : last_alpha * last_depth + (1.f - last_alpha) * accum_rec_depth;
// 			last_depth = skip ? last_depth : depth;
// 			dL_dalpha += (depth - accum_rec_depth) * dL_dpixel_depth;
// 			dL_ddepths_shared[tid] = skip ? 0.f : dchannel_dcolor * dL_dpixel_depth;


// 			dL_dalpha *= T;
// 			// Update last alpha (to be used in the next iteration)
// 			last_alpha = skip ? last_alpha : alpha;

// 			// Account for fact that alpha also influences how much of
// 			// the background color is added if nothing left to blend
// 			float bg_dot_dpixel = 0.f;
// 			#pragma unroll
// 			for (int i = 0; i < CHANNELS; i++) {
// 				bg_dot_dpixel +=  bg_color[i] * dL_dpixel[i];
// 			}
// 			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

// 			// Helpful reusable temporary variables
// 			const float dL_dG = con_o.w * dL_dalpha;
// 			const float gdx = G * d.x;
// 			const float gdy = G * d.y;
// 			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
// 			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

// 			dL_dmean2D_shared[tid].x = skip ? 0.f : dL_dG * dG_ddelx * ddelx_dx;
// 			dL_dmean2D_shared[tid].y = skip ? 0.f : dL_dG * dG_ddely * ddely_dy;
// 			dL_dconic2D_shared[tid].x = skip ? 0.f : -0.5f * gdx * d.x * dL_dG;
// 			dL_dconic2D_shared[tid].y = skip ? 0.f : -0.5f * gdx * d.y * dL_dG;
// 			dL_dconic2D_shared[tid].w = skip ? 0.f : -0.5f * gdy * d.y * dL_dG;
// 			dL_dopacity_shared[tid] = skip ? 0.f : G * dL_dalpha;

// 			render_cuda_reduce_sum(block,
// 				dL_dmean2D_shared,
// 				dL_dconic2D_shared,
// 				dL_dopacity_shared,
// 				dL_dcolors_shared,
// 				dL_ddepths_shared
// 			);

// 			if (tid == 0) {
// 				float2 dL_dmean2D_acc = dL_dmean2D_shared[0];
// 				float4 dL_dconic2D_acc = dL_dconic2D_shared[0];
// 				float dL_dopacity_acc = dL_dopacity_shared[0];
// 				float3 dL_dcolors_acc = dL_dcolors_shared[0];
// 				float dL_ddepths_acc = dL_ddepths_shared[0];

// 				atomicAdd(&dL_dmean2D[global_id].x, dL_dmean2D_acc.x);
// 				atomicAdd(&dL_dmean2D[global_id].y, dL_dmean2D_acc.y);
// 				atomicAdd(&dL_dconic2D[global_id].x, dL_dconic2D_acc.x);
// 				atomicAdd(&dL_dconic2D[global_id].y, dL_dconic2D_acc.y);
// 				atomicAdd(&dL_dconic2D[global_id].w, dL_dconic2D_acc.w);
// 				atomicAdd(&dL_dopacity[global_id], dL_dopacity_acc);
// 				atomicAdd(&dL_dcolors[global_id * CHANNELS + 0], dL_dcolors_acc.x);
// 				atomicAdd(&dL_dcolors[global_id * CHANNELS + 1], dL_dcolors_acc.y);
// 				atomicAdd(&dL_dcolors[global_id * CHANNELS + 2], dL_dcolors_acc.z);
// 				atomicAdd(&dL_ddepths[global_id], dL_ddepths_acc);
// 			}
// 		}
// 	}
// }


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDAFused(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ ref_color,
	const double* __restrict__ ref_depth,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depth,
	float* __restrict__ out_depth,
	float* __restrict__ out_opacity,
	int * __restrict__ n_touched,
	int * __restrict__ tile_active,
	int * __restrict__ tile_active_list,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_ddepths)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	auto tid = block.thread_rank();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	// uint32_t horizontal_blocks = gridDim.x; # TODO Maybe it's different?

#ifdef USE_LIST
	uint32_t tile_idx = tile_active_list[block.group_index().x];
	int block_coord_y = tile_idx / horizontal_blocks;
	int block_coord_x = tile_idx % horizontal_blocks;
#else
	uint32_t tile_idx = block.group_index().y * gridDim.x + block.group_index().x;
	int block_coord_x = block.group_index().x;
	int block_coord_y = block.group_index().y;
	if (tile_active[tile_idx] == 0)
		return;
#endif

	uint2 pix_min = { block_coord_x * BLOCK_X, block_coord_y * BLOCK_Y };
	// uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// if (pix_id == 1199)
	// 	printf ("pix id 1199, val %f, %f\n", ref_img[0 * 680 * 1200 + 1199], ref_img[1 * 680 * 1200 + 1199]);
	// if (pix_id == 1200)
	// 	printf ("pix id 1200, val %f, %f\n", ref_img[0 * 680 * 1200 + 1200], ref_img[1 * 680 * 1200 + 1200]);

	// printf("pix_min: %d, %d, pix: %d, %d\n", pix_min.x, pix_min.y, pix.x, pix.y);

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block_coord_y * horizontal_blocks + block_coord_x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];
	__shared__ float collected_colors[CHANNELS * BLOCK_SIZE];




	const int dummy_size = BLOCK_SIZE;
	// __shared__ bool processed[BLOCK_SIZE * dummy_size] = { false };
	// __shared__ int processed[dummy_size];
	// bool processed[dummy_size] = { false };
	// unsigned char processed[dummy_size] = { 0 };
	__shared__ unsigned char processed[dummy_size];
	// float buffer_alpha[dummy_size] = { 0.f };
	// float buffer_G[dummy_size] = { 0.f };

	// if (toDo > dummy_size)
	// 	printf ("TODO exceeding %d !!! %d\n", dummy_size, toDo);

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	float D = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + tid;
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[tid] = coll_id;
			collected_xy[tid] = points_xy_image[coll_id];
			collected_conic_opacity[tid] = conic_opacity[coll_id];
			collected_depth[tid] = depth[coll_id];
			#pragma unroll
			for (int i = 0; i < CHANNELS; i++) {
				collected_colors[i * BLOCK_SIZE + tid] = features[coll_id * CHANNELS + i];
			}
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;
			// processed[contributor] = false;
			processed[contributor-1] = 0;



			// Resample using conic matrix (cf. "Surface
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			float G = exp(power);
			float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f) {
				continue;
			}
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++) {
				// C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
				C[ch] += collected_colors[ch * BLOCK_SIZE + j] * alpha * T;
				// const float c = collected_colors[ch * BLOCK_SIZE + j];
			}
			D += collected_depth[j] * alpha * T;
			// Keep track of how many pixels touched this Gaussian.
			if (test_T > 0.5f) {
				atomicAdd(&(n_touched[collected_id[j]]), 1);
			}
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;

			// processed[tid * dummy_size + i * BLOCK_SIZE + j] = true;
			// processed[contributor] = true;
			processed[contributor-1] = 1;
			// buffer_alpha[contributor] = alpha;
			// buffer_G[contributor] = G;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++) {
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		}
		out_depth[pix_id] = D;
		out_opacity[pix_id] = 1 - T;
	}

	// block.sync();
	// if (tile_idx == 10)
	// {
	// 	for (int th_idx=0; th_idx<BLOCK_SIZE; th_idx++)
	// 	{
	// 		if (tid == th_idx)
	// 		{
	// 			printf("todo: %d, tile idx: %d, thread idx: %d: ", range.y - range.x, tile_idx, tid);
	// 			for (int i=0; i<dummy_size; i++)
	// 				printf ("%d, ", processed[i]);
	// 			printf ("\n");
	// 		}
	// 		block.sync();
	// 	}
	// }
	// block.sync();


	// block.sync();

	done = !inside;
	toDo = range.y - range.x;

	__shared__ float2 dL_dmean2D_shared[BLOCK_SIZE];
	__shared__ float3 dL_dcolors_shared[BLOCK_SIZE];
	__shared__ float dL_ddepths_shared[BLOCK_SIZE];
	__shared__ float dL_dopacity_shared[BLOCK_SIZE];
	__shared__ float4 dL_dconic2D_shared[BLOCK_SIZE];

	// __shared__ float collected_colors[CHANNELS * BLOCK_SIZE];
	// __shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors.
	// const float T_final = inside ? final_T[pix_id] : 0;
	const float T_final = inside ? T : 0;
	T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	contributor = toDo;
	// last_contributor = inside ? n_contrib[pix_id] : 0;
	last_contributor = inside ? last_contributor : 0;

	float accum_rec[CHANNELS] = { 0.f };
	float dL_dpixel[CHANNELS] = { 0.f };
	float accum_rec_depth = 0.f;
	float dL_dpixel_depth = 0.f;

	if (inside) {
		#pragma unroll
		for (int ch = 0; ch < CHANNELS; ch++) {
			// dL_dpixel[ch] = (out_color[ch * H * W + pix_id] > ref_color[ch * H * W + pix_id]) ? 3.881e-7 : -3.881e-7;
			dL_dpixel[ch] = ((C[ch] + T * bg_color[ch]) > ref_color[ch * H * W + pix_id]) ? 3.881e-7 : -3.881e-7;
		}
		// dL_dpixel_depth = (out_depth[pix_id] > ref_depth[pix_id]) ? 6.127e-8 : -6.127e-8;
		dL_dpixel_depth = (D > ref_depth[pix_id]) ? 6.127e-8 : -6.127e-8;
	}

	float last_alpha = 0.f;
	float last_color[CHANNELS] = { 0.f };
	float last_depth = 0.f;

	// Gradient of pixel coordinate w.r.t. normalized
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5f * W;
	const float ddely_dy = 0.5f * H;
	// __shared__ int skip_counter;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		// block.sync();
		// const int progress = i * BLOCK_SIZE + tid;
		// if (range.x + progress < range.y)
		// {
			// const int coll_id = point_list[range.y - progress - 1];
			// collected_id[tid] = coll_id;
			// collected_xy[tid] = points_xy_image[coll_id];
			// collected_conic_opacity[tid] = conic_opacity[coll_id];
			// #pragma unroll
			// for (int i = 0; i < CHANNELS; i++) {
			// 	collected_colors[i * BLOCK_SIZE + tid] = features[coll_id * CHANNELS + i];

			// }
			// collected_depth[tid] = depth[coll_id];
		// }
		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++) {
			block.sync();
			// if (tid == 0) {
			// 	skip_counter = 0;
			// }
			// block.sync();

			// processed[tid * dummy_size + i * BLOCK_SIZE + j] = true;

			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			bool skip = done;
			// contributor--;
			// skip = processed[contributor];
			// if (tid == 0)
				// printf ("skip: %d\n", skip);
			// float alpha = buffer_alpha[contributor];
			// float G = buffer_G[contributor];
			contributor = done ? contributor : contributor - 1;

			if (processed[contributor] == 0)
				continue;

			skip |= contributor >= last_contributor;

			// // Compute blending values, as before.
			// const float2 xy = collected_xy[j];
			const float2 xy = collected_xy[contributor];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			// const float4 con_o = collected_conic_opacity[j];
			const float4 con_o = collected_conic_opacity[contributor];


			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			skip |= power > 0.0f;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			skip |= alpha < 1.0f / 255.0f;


			// skip = !processed[contributor];
			// float power = skip ? 0.f : -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			// float G = skip ? 0.f : exp(power);
			// float alpha = skip ? 0.f : min(0.99f, con_o.w * G);


			// if (skip) {
			// 	atomicAdd(&skip_counter, 1);
			// }
			// block.sync();
			// if (skip_counter == BLOCK_SIZE) {
			// 	continue;
			// }


			T = skip ? T : T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			// const int global_id = collected_id[j];
			const int global_id = collected_id[contributor];
			float local_dL_dcolors[3];
			#pragma unroll
			for (int ch = 0; ch < CHANNELS; ch++)
			{
				// const float c = collected_colors[ch * BLOCK_SIZE + j];
				const float c = collected_colors[ch * BLOCK_SIZE + contributor];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = skip ? accum_rec[ch] : last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = skip ? last_color[ch] : c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				local_dL_dcolors[ch] = skip ? 0.0f : dchannel_dcolor * dL_dchannel;
			}
			dL_dcolors_shared[tid].x = local_dL_dcolors[0];
			dL_dcolors_shared[tid].y = local_dL_dcolors[1];
			dL_dcolors_shared[tid].z = local_dL_dcolors[2];

			// const float depth = collected_depth[j];
			const float depth = collected_depth[contributor];
			accum_rec_depth = skip ? accum_rec_depth : last_alpha * last_depth + (1.f - last_alpha) * accum_rec_depth;
			last_depth = skip ? last_depth : depth;
			dL_dalpha += (depth - accum_rec_depth) * dL_dpixel_depth;
			dL_ddepths_shared[tid] = skip ? 0.f : dchannel_dcolor * dL_dpixel_depth;


			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = skip ? last_alpha : alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0.f;
			#pragma unroll
			for (int i = 0; i < CHANNELS; i++) {
				bg_dot_dpixel +=  bg_color[i] * dL_dpixel[i];
			}
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			dL_dmean2D_shared[tid].x = skip ? 0.f : dL_dG * dG_ddelx * ddelx_dx;
			dL_dmean2D_shared[tid].y = skip ? 0.f : dL_dG * dG_ddely * ddely_dy;
			dL_dconic2D_shared[tid].x = skip ? 0.f : -0.5f * gdx * d.x * dL_dG;
			dL_dconic2D_shared[tid].y = skip ? 0.f : -0.5f * gdx * d.y * dL_dG;
			dL_dconic2D_shared[tid].w = skip ? 0.f : -0.5f * gdy * d.y * dL_dG;
			dL_dopacity_shared[tid] = skip ? 0.f : G * dL_dalpha;

			render_cuda_reduce_sum(block,
				dL_dmean2D_shared,
				dL_dconic2D_shared,
				dL_dopacity_shared,
				dL_dcolors_shared,
				dL_ddepths_shared
			);

			if (tid == 0) {
				float2 dL_dmean2D_acc = dL_dmean2D_shared[0];
				float4 dL_dconic2D_acc = dL_dconic2D_shared[0];
				float dL_dopacity_acc = dL_dopacity_shared[0];
				float3 dL_dcolors_acc = dL_dcolors_shared[0];
				float dL_ddepths_acc = dL_ddepths_shared[0];

				atomicAdd(&dL_dmean2D[global_id].x, dL_dmean2D_acc.x);
				atomicAdd(&dL_dmean2D[global_id].y, dL_dmean2D_acc.y);
				atomicAdd(&dL_dconic2D[global_id].x, dL_dconic2D_acc.x);
				atomicAdd(&dL_dconic2D[global_id].y, dL_dconic2D_acc.y);
				atomicAdd(&dL_dconic2D[global_id].w, dL_dconic2D_acc.w);
				atomicAdd(&dL_dopacity[global_id], dL_dopacity_acc);
				atomicAdd(&dL_dcolors[global_id * CHANNELS + 0], dL_dcolors_acc.x);
				atomicAdd(&dL_dcolors[global_id * CHANNELS + 1], dL_dcolors_acc.y);
				atomicAdd(&dL_dcolors[global_id * CHANNELS + 2], dL_dcolors_acc.z);
				atomicAdd(&dL_ddepths[global_id], dL_ddepths_acc);

				// dL_dmean2D[global_id].x += dL_dmean2D_acc.x;
				// dL_dmean2D[global_id].y += dL_dmean2D_acc.y;
				// dL_dconic2D[global_id].x += dL_dconic2D_acc.x;
				// dL_dconic2D[global_id].y += dL_dconic2D_acc.y;
				// dL_dconic2D[global_id].w += dL_dconic2D_acc.w;
				// dL_dopacity[global_id] += dL_dopacity_acc;
				// dL_dcolors[global_id * CHANNELS + 0] += dL_dcolors_acc.x;
				// dL_dcolors[global_id * CHANNELS + 1] += dL_dcolors_acc.y;
				// dL_dcolors[global_id * CHANNELS + 2] += dL_dcolors_acc.z;
				// dL_ddepths[global_id] += dL_ddepths_acc;
			}
		}
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	const float* depth,
	float* out_depth,
	float* out_opacity,
	int* n_touched)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depth,
		out_depth,
		out_opacity,
		n_touched);
}

void FORWARD::render_fast(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	const float* depth,
	float* out_depth,
	float* out_opacity,
	int* n_touched,
	int* tile_active,
	const int active_count,
	int* tile_active_list)
{
#ifdef USE_LIST
	dim3 grid1d(active_count, 1, 1);
	renderCUDAFast<NUM_CHANNELS> << <grid1d, block >> > (
#else
	renderCUDAFast<NUM_CHANNELS> << <grid, block >> > (
#endif
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depth,
		out_depth,
		out_opacity,
		n_touched,
		tile_active,
		tile_active_list);
}

void FORWARD::render_fused(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* ref_color,
	const double* ref_depth,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	const float* depth,
	float* out_depth,
	float* out_opacity,
	int* n_touched,
	int* tile_active,
	const int active_count,
	int* tile_active_list,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_ddepths)
{

	// size_t shared_mem_size = 32 * 1024;
	// cudaError_t err = cudaFuncSetAttribute(renderCUDAFused<NUM_CHANNELS>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
	// if (err != cudaSuccess)
	// {
	// 	printf ("Error setting shared memory size %s\n", cudaGetErrorString(err));
	// 	// return -1;
	// }

	cudaFuncSetCacheConfig(renderCUDAFused<NUM_CHANNELS>, cudaFuncCachePreferShared);

#ifdef USE_LIST
	dim3 grid1d(active_count, 1, 1);
	renderCUDAFused<NUM_CHANNELS> << <grid1d, block >> > (
#else
	renderCUDAFused<NUM_CHANNELS> << <grid, block >> > (
#endif
		ranges,
		point_list,
		W, H,
		ref_color,
		ref_depth,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depth,
		out_depth,
		out_opacity,
		n_touched,
		tile_active,
		tile_active_list,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_ddepths);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix,
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}
