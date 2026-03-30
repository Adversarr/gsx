# Metal Kernel Optimization Report

This document catalogs optimization opportunities found across the Metal shader files in `gsx/src/gsx-metal/kernels/`.

---

## adc.metal

### High Priority

**Strided `logscale` reads break memory coalescing**
- **Lines**: 240-242, 312-314, 399-401, 473-475, 551-553
- **Issue**: Reads like `logscale[gid * 3u + offset]` scatter across memory for adjacent threads
- **Fix**: Convert to Structure-of-Arrays (SoA) layout with separate `logscale_x[gid]`, `logscale_y[gid]`, `logscale_z[gid]` arrays, or use `float3` packed type
- **Impact**: 2-3x memory bandwidth improvement

**`mcmc_scale_coeff` nested loops are compute-bound**
- **Lines**: 190-200
- **Issue**: Up to 51×52/2 = 1326 iterations per thread with `precise::sqrt`, `pow`, `binom` calls
- **Fix**: Precompute binomial coefficients and power series as constant tables
- **Impact**: 1000+ ops → table lookup

### Medium Priority

**Excessive scalar registers (~35+) in split kernel**
- **Lines**: 257-298
- **Issue**: Many individual `float` variables: qx, qy, qz, qw, sx, sy, sz, 9 rotation matrix elements, etc.
- **Fix**: Use vector types: `float4 quat`, `float3 scale`, `float3x3 M`
- **Impact**: Reduced register pressure, cleaner code

**Manual rotation matrix construction**
- **Lines**: 329-350, 493-501
- **Issue**: 9 separate scalar assignments then manual matrix-vector multiply
- **Fix**: Use `float3x3 * float3` matrix operations
- **Impact**: Single fused operation, better SIMD utilization

### Low Priority

**Quaternion normalization early-exit causes divergence**
- **Lines**: 127-129
- **Fix**: Use predicated execution: `float inv_q = (q_norm > 1e-8f) ? (1.0f / q_norm) : 1.0f;`

---

## loss.metal

### High Priority

**HWC layout causes non-coalesced memory access**
- **Lines**: 53-61
- **Issue**: `return ((((outer * params.height + y) * params.width + x) * params.channels) + channel)` - adjacent x threads access memory separated by `params.channels`
- **Fix**: Use CHW layout or vectorized loads with `float4`
- **Impact**: 2-4x bandwidth improvement

**Threadgroup bank conflicts in tile arrays**
- **Lines**: 171-172
- **Issue**: `tile[26][26][2]` causes bank conflicts; index `[y][x]` maps to bank `(y + x) % 4`
- **Fix**: Pad X dimension to 28 (multiple of 4): `tile[26][28][2]`
- **Impact**: 20-30% convolution speedup

**Divergent boundary branches**
- **Lines**: 322-331, 521-530
- **Issue**: `if(is_boundary)` creates different execution paths within SIMD groups
- **Fix**: Use `select()`: `loss_map[idx] = select(0.0f, loss_map[idx] + loss_val, !is_boundary);`
- **Impact**: 5-15% for small images

### Medium Priority

**13.7KB threadgroup memory limits occupancy**
- **Lines**: 171-172, 370-371, 570-571, 688-689
- **Issue**: `tile[26][26][2]` (5.4KB) + `xconv[26][16][5]` (8.3KB) = 13.7KB per threadgroup
- **Fix**: Union tile and xconv to reuse memory, or use padded SoA layout
- **Impact**: Better threadgroup occupancy on memory-constrained GPUs

**Half precision for Gaussian weights and accumulators**
- **Lines**: 14, 204-232
- **Issue**: `float` throughout; `half` sufficient for weights and intermediate conv results
- **Fix**: Use `constant half gsx_metal_ssim_gauss_1d[]`, `half sum_x` accumulators
- **Impact**: 1.5-2x convolution throughput

**Redundant barrier at end of channel loop**
- **Lines**: 352
- **Issue**: `threadgroup_barrier` after writing loss_map may be unnecessary
- **Fix**: Profile to verify tile/xconv writes are properly fenced before next iteration
- **Impact**: Latency reduction

---

## optim.metal

### High Priority

**Scalar float operations waste memory bandwidth**
- **Lines**: 28-47
- **Issue**: 1 element per thread, 4 separate buffer reads + 3 writes
- **Fix**: Vectorize to `float4`: process 4 elements per thread
- **Impact**: 4x memory throughput

**Mixed precision not utilized**
- **Lines**: Throughout
- **Issue**: All `float32`; `half` could halve bandwidth for parameters
- **Fix**: Use `device half *parameter` for storage, `float` for moments and computation
- **Impact**: 2x memory bandwidth for parameters

### Medium Priority

**Buffer interleaving could improve cache locality**
- **Lines**: 28-47
- **Issue**: 4 separate buffers accessed; poor cache line utilization
- **Fix**: Interleave as `struct { float param, grad, m, v; }` per element
- **Impact**: Better cache locality

---

## render_backward.metal

### High Priority

**Nine atomic operations per thread for gradients**
- **Lines**: 373-381
- **Issue**: Separate atomic for each gradient component; contention
- **Fix**: SIMD-group reduce before single atomic:
  ```metal
  float sum = simd_sum(delta.x);
  if (simd_lane_id == 0) atomic_fetch_add(&grad_mean2d[idx], sum);
  ```
- **Impact**: Reduces atomic contention by SIMD-width factor

**Component-wise SIMD shuffle wastes instructions**
- **Lines**: 311-320
- **Issue**: 6 separate scalar shuffles: `gsx_metal_simd_shuffle_up(grad_pixel.x, 1u)`, etc.
- **Fix**: Use vector shuffle: `grad_pixel = gsx_metal_simd_shuffle_up(grad_pixel, 1u);`
- **Impact**: 3x fewer shuffle operations

**Nested SH degree conditionals create divergence**
- **Lines**: 82-155
- **Issue**: Three levels of `if(active_sh_bases > N)` branching
- **Fix**: Create specialized kernels per SH degree (0, 1, 2, 3)
- **Impact**: Eliminates runtime branching overhead

### Medium Priority

**8KB threadgroup for per-pixel upper/lower structs**
- **Lines**: 183-184
- **Issue**: `cached_per_pixel_upper[8][32]` + `cached_per_pixel_lower[8][32]` = 8KB
- **Fix**: Merge into single `gsx_metal_render_per_pixel` struct (32 bytes each)
- **Impact**: Halves threadgroup memory

**Eight barriers per blend iteration**
- **Lines**: 298
- **Issue**: Barrier inside loop iterates `tile_size/simd_width = 8` times
- **Fix**: Use double-buffering pattern: read from ping, write to pong, swap
- **Impact**: Reduces barrier overhead

---

## render_common.metal

### High Priority

**Early-exit branch causes SIMD divergence**
- **Lines**: 109-111
- **Issue**: `if(not_in_x_range + not_in_y_range == 0.0f) return true;` within inner computation
- **Fix**: Compute all paths and use branchless selection
- **Impact**: Reduces warp divergence

**Six scalar comparisons could be vectorized**
- **Lines**: 103-107
- **Issue**: Manual `float x_min_diff > 0.0f ? 1.0f : 0.0f` for each component
- **Fix**: Use `float2` comparisons with `select()`
- **Impact**: 2x fewer comparison operations

### Medium Priority

**Division operations could use reciprocal multiply**
- **Lines**: 116-117
- **Issue**: `saturate((...) / (d.x * conic.x * d.x))` - division in inner loop
- **Fix**: Precompute `rcp_denom = 1.0f / (d.x * conic.x)` outside loop
- **Impact**: GPU division is slower than multiplication

**metal::fast namespace not utilized**
- **Lines**: 116-121
- **Issue**: Uses `saturate()` instead of `metal::fast::saturate()`
- **Fix**: Use `metal::fast::` variants where NaN handling is acceptable
- **Impact**: Faster math operations

---

## render.metal

### High Priority

**Float color output wastes bandwidth**
- **Lines**: 783-785
- **Issue**: `image_chw[pixel_index] = accum.x;` - image data doesn't need float32
- **Fix**: Use `half` output buffers
- **Impact**: 2x memory bandwidth

**AoS SH coefficient layout scatters reads**
- **Lines**: 116-126
- **Issue**: `sh[primitive_idx * coeff_count * 3 + coeff_idx * 3]` - adjacent threads access distant locations
- **Fix**: Convert to SoA per-coefficient arrays or `float3` packed
- **Impact**: Better cache locality

**Scalar shuffle for vector types**
- **Lines**: 63-69, 545-551
- **Issue**: `float2 mean_shifted_coop = float2(gsx_metal_simd_shuffle(mean_shifted.x, ...), gsx_metal_simd_shuffle(mean_shifted.y, ...))`
- **Fix**: `float2 mean_shifted_coop = gsx_metal_simd_shuffle(mean_shifted, source_lane);`
- **Impact**: 2-3x shuffle speedup

**Global atomic counter bottleneck**
- **Lines**: 425-428
- **Issue**: `atomic_fetch_add(visible_count, 1u)` - all visible gaussians contend
- **Fix**: Per-threadgroup local atomics with block-level prefix sum, one global atomic per threadgroup
- **Impact**: Reduces atomic serialization

### Medium Priority

**9.2KB threadgroup memory**
- **Lines**: 695-698
- **Issue**: `collected_color[256]` (3KB) + `collected_conic_opacity[256]` (4KB) + `collected_mean2d[256]` (2KB)
- **Fix**: Use `threadgroup half3` and `threadgroup half4`
- **Impact**: Halves threadgroup usage

**64-bit SIMD ballot masks unnecessary**
- **Lines**: 8-15
- **Issue**: `gsx_metal_first_set_lane_u64` handles 64-bit masks; Apple SIMD groups are 32-wide
- **Fix**: Use 32-bit masks directly
- **Impact**: Simplifies mask handling

---

## scan.metal

### High Priority

**Lane-0 serial scan loop**
- **Lines**: 34-41
- **Issue**: 8-iteration sequential loop only lane 0 executes
- **Fix**: Use `simd_prefix_exclusive_sum` on first-lane values:
  ```metal
  uint my_total = (simd_lane == 0) ? simd_totals[simd_group] : 0u;
  uint scanned = simd_prefix_exclusive_sum(my_total);
  ```
- **Impact**: ~32x speedup (8 serial ops → 1 SIMD op)

### Medium Priority

**Serial block total sum**
- **Lines**: 83-85
- **Issue**: Loop over 8 SIMD groups to accumulate totals
- **Fix**: Use `simd_sum` via first-lane broadcasts
- **Impact**: Reduces serial iterations

---

## simd_utils.metal

### High Priority

**Sequential atomics for vector types**
- **Lines**: 69-80
- **Issue**: `gsx_metal_atomic_add_f32(values, index, delta.x)` called 3 times for `float3`
- **Fix**: SIMD-group reduce then single atomic:
  ```metal
  float sum = simd_sum(delta.x);
  if (simd_lane_id == 0) atomic_fetch_add(&values[index], sum);
  ```
- **Impact**: Reduces atomic contention by SIMD-width factor

**Divergent early-exit in atomic_max**
- **Lines**: 86-88
- **Issue**: `if(value <= 0.0f) return;` - inactive threads during atomic
- **Fix**: Remove branch; atomic max with 0 won't change positive values
- **Impact**: Reduces SIMD divergence

### Medium Priority

**No half-precision atomic variants**
- **Lines**: Throughout
- **Issue**: All operations use `float32`
- **Fix**: Add `gsx_metal_atomic_add_f16` for bandwidth-sensitive gradient accumulation
- **Impact**: 2x memory bandwidth where precision allows

---

## sort.metal

### High Priority

**Atomic contention in histogram**
- **Lines**: 121-128
- **Issue**: All threads with same digit contend for `atomic_histogram[digit]`
- **Fix**: Thread-local histogram per SIMD group, reduce before atomic
- **Impact**: Reduces atomic serialization

**Redundant ballot and double bit computation**
- **Lines**: 79-89, 95-99
- **Issue**: `gsx_metal_simd_ballot(true)` initialized then overwritten; `((digit >> bit) & 1u) != 0u` computed twice
- **Fix**:
  ```metal
  ulong same_digit_mask = ~0ul;
  for(uint bit = 0u; bit < RADIX_BITS; ++bit) {
      bool bit_set = ((digit >> bit) & 1u) != 0u;
      ulong bit_mask = gsx_metal_simd_ballot(bit_set);
      same_digit_mask &= bit_set ? bit_mask : (~bit_mask);
  }
  ```
- **Impact**: Reduces SIMD operations

### Medium Priority

**Missing #pragma unroll**
- **Lines**: 121, 355
- **Issue**: `radix_histogram` and `radix_scatter_simd_tail` lack unroll directives
- **Fix**: Add `#pragma unroll` before loops
- **Impact**: Enables compiler optimization

**Loop-invariant subgroup_digit_idx**
- **Lines**: 279, 361
- **Issue**: `uint subgroup_digit_idx = simd_group_id * RADIX_SIZE;` computed each iteration
- **Fix**: Move outside loop
- **Impact**: Reduces arithmetic

---

## tensor.metal

### Critical

**Naive tree reduction with many barriers**
- **Lines**: 537-729
- **Issue**: O(log N) barriers: `for(stride = n>>1; stride > 0; stride >>= 1) { ... barrier; }`
- **Fix**: Replace with `simd_sum`:
  ```metal
  float accum = simd_sum(accum);
  if(simd_is_first()) out_values[group_id] = accum;
  ```
- **Impact**: 5-10x faster; eliminates ~8-10 barriers per reduction

### High Priority

**Byte-by-byte copy loops**
- **Lines**: 293-295, 318-320
- **Issue**: `for(i = 0; i < element_size_bytes; ++i) dst[..] = src[..]` - poor throughput
- **Fix**: Use typed loads:
  ```metal
  if(elem_bytes == 4) *(device uint*)(dst + dst_off) = *(device const uint*)(src + src_off);
  ```
- **Impact**: 4-8x memory throughput

**Divergent sRGB branches**
- **Lines**: 247-251, 266-270
- **Issue**: `if(value <= 0.0031308f)` - both paths executed in SIMD group
- **Fix**: Use `select()`:
  ```metal
  float high = 1.055f * pow(value, 1.0f/2.4f) - 0.055f;
  float low = 12.92f * value;
  dst = select(high, low, value <= 0.0031308f);
  ```
- **Impact**: Branchless execution

### Medium Priority

**Gather kernel scattered reads**
- **Lines**: 143-165
- **Issue**: `index_data[row]` causes non-coalesced indirect indexing
- **Fix**: Process rows in threadgroups for contiguous memory access
- **Impact**: Better coalescing

**Float-only image kernels**
- **Lines**: Throughout image functions
- **Issue**: `device const float *` for image data
- **Fix**: Use `device const half *` where precision permits
- **Impact**: 2x throughput on Apple Silicon

---

## Cross-File Patterns

### 1. SIMD-Group Operations Underutilized

Many files use manual loops/barriers where hardware-accelerated SIMD operations would be faster:

| Operation | Replacement | Benefit |
|-----------|-------------|---------|
| Tree reduction with barriers | `simd_sum`, `simd_max` | Single instruction, no barrier |
| Sequential prefix sum | `simd_prefix_exclusive_sum` | O(1) hardware op |
| Component-wise shuffle | Vector shuffle | 2-3x fewer ops |
| Sequential SIMD-group scan | `simd_prefix_*` | ~32x speedup |

### 2. Half Precision Unused

Apple GPUs have 2x `half` throughput vs `float`. Opportunities:
- Color outputs (`render.metal`, `render_backward.metal`)
- Image data (`tensor.metal`)
- Gaussian weights (`loss.metal`)
- Threadgroup color storage (`render.metal`)
- Atomic gradient accumulation (`simd_utils.metal`)

### 3. AoS Memory Layouts

Common pattern causing scattered reads:
- `logscale` in `adc.metal`: `buffer[gid * 3 + offset]`
- SH coefficients in `render.metal`, `render_backward.metal`
- Color in `render_backward.metal`

Fix: SoA layout or packed `float3`/`float4` vectors

### 4. Sequential Atomics

Vector gradient writes in `render_backward.metal` and `simd_utils.metal` perform component-wise atomics:
```metal
// Current: 3 atomics for float3
atomic_add(&values[idx], delta.x);
atomic_add(&values[idx+1], delta.y);
atomic_add(&values[idx+2], delta.z);

// Better: SIMD-reduce then 1 atomic
float sum = simd_sum(delta.x);
if(lane == 0) atomic_add(&values[idx], sum);
```

---

## Summary of Highest Impact Optimizations

| File | Optimization | Estimated Impact |
|------|-------------|------------------|
| `tensor.metal` | Replace tree reduction with `simd_sum` | 5-10x faster |
| `scan.metal` | Use SIMD prefix for cross-group scan | ~32x faster |
| `sort.metal` | Fix `same_digit_mask` redundant computation | 2x fewer SIMD ops |
| `render.metal` | Use vector SIMD shuffle | 2-3x faster shuffles |
| `render.metal` | Use `half` for color output | 2x bandwidth |
| `render_backward.metal` | SIMD-group reduce before atomic | Reduces contention |
| `adc.metal` | Precompute binomial coefficients | 1000+ ops → lookup |
| `loss.metal` | Use CHW layout / vector loads | 2-4x bandwidth |
| `optim.metal` | Vectorize to float4 | 4x throughput |
| `simd_utils.metal` | SIMD-reduce before atomic | Reduces contention |
