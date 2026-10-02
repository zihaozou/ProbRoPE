
extern "C" __global__ void fill_gray(unsigned char* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 128;
}

extern "C" __global__ void integrate_events(
    const int4* evs, int n, float* f, float* t_map,
    float c_pos, float c_neg, int w, int h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int4 e = evs[i];
    if (e.x < 0 || e.x >= w || e.y < 0 || e.y >= h) return;
    int idx = e.y * w + e.x;
    atomicAdd(&f[idx], e.z ? c_pos : -c_neg);
    atomicMax((int*)&t_map[idx], __float_as_int((float)e.w));
}

extern "C" __global__ void decay_state(float* f, float* u, float factor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    f[i] *= factor;
    u[i] *= factor;
}

extern "C" __global__ void clamp_f(float* f, float clamp_abs, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = f[i];
    v = v >  clamp_abs ?  clamp_abs : v;
    v = v < -clamp_abs ? -clamp_abs : v;
    f[i] = v;
}

extern "C" __global__ void shift_tmap(float* t_map, float delta, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = t_map[i] - delta;
    t_map[i] = v > 0.0f ? v : 0.0f;
}


extern "C" __global__ void compute_g(const float* t_map, float* g_out, float alpha, float t_now_rel, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float dt = t_now_rel - t_map[i];
    dt = dt > 0.0f ? dt : 0.0f;
    dt *= 1e-6f;
    g_out[i] = 1.0f - expf(-alpha * dt);
}

extern "C" __global__ void pd_dual(
    float* p_x, float* p_y, const float* u_bar, const float* g,
    float sigma, int w, int h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = w * h;
    if (i >= n) return;
    int x = i % w;
    int y = i / w;

    float uc = u_bar[i];
    float gx = (x < w - 1) ? (u_bar[i + 1] - uc) : 0.0f;
    float gy = (y < h - 1) ? (u_bar[i + w] - uc) : 0.0f;

    float nx = p_x[i] + sigma * g[i] * gx;
    float ny = p_y[i] + sigma * g[i] * gy;
    float mag = sqrtf(nx * nx + ny * ny);
    float denom = mag > 1.0f ? mag : 1.0f;
    p_x[i] = nx / denom;
    p_y[i] = ny / denom;
}

extern "C" __global__ void pd_primal(
    float* u, float* u_bar_out, const float* p_x, const float* p_y, const float* g, const float* f,
    float tau, float lambda, int w, int h)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = w * h;
    if (i >= n) return;
    int x = i % w;
    int y = i / w;

    float div_x = (x == 0) ? g[i] * p_x[i]
                 : (x == w - 1) ? -(g[i - 1] * p_x[i - 1])
                 : (g[i] * p_x[i] - g[i - 1] * p_x[i - 1]);
    float div_y = (y == 0) ? g[i] * p_y[i]
                 : (y == h - 1) ? -(g[i - w] * p_y[i - w])
                 : (g[i] * p_y[i] - g[i - w] * p_y[i - w]);
    float div_gp = div_x + div_y;

    float u_old = u[i];
    float u_new = (u_old + tau * div_gp + tau * lambda * f[i]) / (1.0f + tau * lambda);
    u[i] = u_new;
    u_bar_out[i] = 2.0f * u_new - u_old;
}

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__ void pd_solve_persistent(
    float* u, float* u_bar, float* p_x, float* p_y, float* g,
    const float* f, const float* t_map,
    float alpha, float t_now_rel, float tau, float sigma, float lambda,
    int w, int h, int iters)
{
    cg::grid_group grid = cg::this_grid();
    int n = w * h;
    int idx0 = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx0; i < n; i += stride) {
        float dt = t_now_rel - t_map[i];
        dt = dt > 0.0f ? dt : 0.0f;
        dt *= 1e-6f;
        g[i] = 1.0f - expf(-alpha * dt);
    }
    grid.sync();

    for (int it = 0; it < iters; ++it) {
        for (int i = idx0; i < n; i += stride) {
            int x = i % w;
            int y = i / w;
            float uc = u_bar[i];
            float gx = (x < w - 1) ? (u_bar[i + 1] - uc) : 0.0f;
            float gy = (y < h - 1) ? (u_bar[i + w] - uc) : 0.0f;
            float nx = p_x[i] + sigma * g[i] * gx;
            float ny = p_y[i] + sigma * g[i] * gy;
            float mag = sqrtf(nx * nx + ny * ny);
            float denom = mag > 1.0f ? mag : 1.0f;
            p_x[i] = nx / denom;
            p_y[i] = ny / denom;
        }
        grid.sync();

        for (int i = idx0; i < n; i += stride) {
            int x = i % w;
            int y = i / w;
            float div_x = (x == 0) ? g[i] * p_x[i]
                         : (x == w - 1) ? -(g[i - 1] * p_x[i - 1])
                         : (g[i] * p_x[i] - g[i - 1] * p_x[i - 1]);
            float div_y = (y == 0) ? g[i] * p_y[i]
                         : (y == h - 1) ? -(g[i - w] * p_y[i - w])
                         : (g[i] * p_y[i] - g[i - w] * p_y[i - w]);
            float div_gp = div_x + div_y;

            float u_old = u[i];
            float u_new = (u_old + tau * div_gp + tau * lambda * f[i]) / (1.0f + tau * lambda);
            u[i] = u_new;
            u_bar[i] = 2.0f * u_new - u_old;
        }
        grid.sync();
    }
}

extern "C" __global__ void tonemap(const float* src, unsigned char* out, int n, float bias, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = bias + scale * src[i];
    v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
    out[i] = (unsigned char)v;
}

#define PIX_SORT(a, b) { unsigned char _pa = (a); unsigned char _pb = (b); \
    (a) = _pa < _pb ? _pa : _pb; (b) = _pa < _pb ? _pb : _pa; }

extern "C" __global__ void median_filter_3x3(const unsigned char* src, unsigned char* dst, int w, int h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = w * h;
    if (i >= n) return;
    int x = i % w;
    int y = i / w;

    unsigned char v[9];
    int k = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        int yy = y + dy;
        yy = yy < 0 ? 0 : (yy >= h ? h - 1 : yy);
        for (int dx = -1; dx <= 1; ++dx) {
            int xx = x + dx;
            xx = xx < 0 ? 0 : (xx >= w ? w - 1 : xx);
            v[k++] = src[yy * w + xx];
        }
    }

    PIX_SORT(v[1], v[2]); PIX_SORT(v[4], v[5]); PIX_SORT(v[7], v[8]);
    PIX_SORT(v[0], v[1]); PIX_SORT(v[3], v[4]); PIX_SORT(v[6], v[7]);
    PIX_SORT(v[1], v[2]); PIX_SORT(v[4], v[5]); PIX_SORT(v[7], v[8]);
    PIX_SORT(v[0], v[3]); PIX_SORT(v[5], v[8]); PIX_SORT(v[4], v[7]);
    PIX_SORT(v[3], v[6]); PIX_SORT(v[1], v[4]); PIX_SORT(v[2], v[5]);
    PIX_SORT(v[4], v[7]); PIX_SORT(v[4], v[2]); PIX_SORT(v[6], v[4]);
    PIX_SORT(v[4], v[2]);

    dst[i] = v[4];
}

#undef PIX_SORT

#define TILE_REDUCE_THREADS 256

extern "C" __global__ void compute_tile_stats(
    const float* src, float* tile_min, float* tile_max, float* tile_sum, float* tile_sumsq,
    int w, int h, int tile_w, int tile_h, int tiles_x, int tiles_y)
{
    int tile_id = blockIdx.x;
    int tx = tile_id % tiles_x;
    int ty = tile_id / tiles_x;
    int x0 = tx * tile_w;
    int y0 = ty * tile_h;
    int x1 = x0 + tile_w; x1 = x1 > w ? w : x1;
    int y1 = y0 + tile_h; y1 = y1 > h ? h : y1;
    int tw = x1 - x0;
    int th = y1 - y0;
    int npix = tw * th;

    float local_min = 3.402823e38f;
    float local_max = -3.402823e38f;
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (int idx = threadIdx.x; idx < npix; idx += blockDim.x) {
        int lx = idx % tw;
        int ly = idx / tw;
        float v = src[(y0 + ly) * w + (x0 + lx)];
        local_min = v < local_min ? v : local_min;
        local_max = v > local_max ? v : local_max;
        local_sum += v;
        local_sumsq += v * v;
    }

    __shared__ float smin[TILE_REDUCE_THREADS];
    __shared__ float smax[TILE_REDUCE_THREADS];
    __shared__ float ssum[TILE_REDUCE_THREADS];
    __shared__ float ssumsq[TILE_REDUCE_THREADS];
    smin[threadIdx.x] = local_min;
    smax[threadIdx.x] = local_max;
    ssum[threadIdx.x] = local_sum;
    ssumsq[threadIdx.x] = local_sumsq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float om = smin[threadIdx.x + s];
            float ox = smax[threadIdx.x + s];
            smin[threadIdx.x] = om < smin[threadIdx.x] ? om : smin[threadIdx.x];
            smax[threadIdx.x] = ox > smax[threadIdx.x] ? ox : smax[threadIdx.x];
            ssum[threadIdx.x] += ssum[threadIdx.x + s];
            ssumsq[threadIdx.x] += ssumsq[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        tile_min[tile_id] = smin[0];
        tile_max[tile_id] = smax[0];
        tile_sum[tile_id] = ssum[0];
        tile_sumsq[tile_id] = ssumsq[0];
    }
}

extern "C" __global__ void compute_tile_bias_scale(
    const float* tile_min, const float* tile_max, const float* tile_sum, const float* tile_sumsq,
    float* tile_bias, float* tile_scale,
    int w, int h, int tile_w, int tile_h, int tiles_x, int tiles_y,
    float robust_k, float gain_cap)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int n_tiles = tiles_x * tiles_y;
    if (t >= n_tiles) return;

    int tx = t % tiles_x;
    int ty = t / tiles_x;
    int x1 = (tx + 1) * tile_w; x1 = x1 > w ? w : x1;
    int y1 = (ty + 1) * tile_h; y1 = y1 > h ? h : y1;
    float npix = (float)((x1 - tx * tile_w) * (y1 - ty * tile_h));

    float min_val = tile_min[t];
    float max_val = tile_max[t];
    float sum = tile_sum[t];
    float sumsq = tile_sumsq[t];
    float mean = sum / npix;
    float var = sumsq / npix - mean * mean;
    var = var > 0.0f ? var : 0.0f;
    float sd = sqrtf(var);

    float lo = mean - robust_k * sd; lo = lo > min_val ? lo : min_val;
    float hi = mean + robust_k * sd; hi = hi < max_val ? hi : max_val;
    float range = hi - lo;

    float bias, scale;
    if (range > 1e-6f) {
        float mid = 0.5f * (lo + hi);
        scale = 255.0f / range;
        scale = scale < gain_cap ? scale : gain_cap;
        bias = 128.0f - scale * mid;
    } else {
        bias = 128.0f;
        scale = 0.0f;
    }
    tile_bias[t] = bias;
    tile_scale[t] = scale;
}

extern "C" __global__ void tonemap_tiled(
    const float* src, unsigned char* out, int w, int h,
    const float* tile_bias, const float* tile_scale,
    int tile_w, int tile_h, int tiles_x, int tiles_y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n = w * h;
    if (i >= n) return;
    int x = i % w;
    int y = i / w;

    float fx = (float)x / (float)tile_w - 0.5f;
    float fy = (float)y / (float)tile_h - 0.5f;
    int tx0 = (int)floorf(fx);
    int ty0 = (int)floorf(fy);
    float wx = fx - (float)tx0;
    float wy = fy - (float)ty0;

    int tx0c = tx0 < 0 ? 0 : (tx0 >= tiles_x ? tiles_x - 1 : tx0);
    int tx1r = tx0 + 1;
    int tx1c = tx1r < 0 ? 0 : (tx1r >= tiles_x ? tiles_x - 1 : tx1r);
    int ty0c = ty0 < 0 ? 0 : (ty0 >= tiles_y ? tiles_y - 1 : ty0);
    int ty1r = ty0 + 1;
    int ty1c = ty1r < 0 ? 0 : (ty1r >= tiles_y ? tiles_y - 1 : ty1r);

    float b00 = tile_bias[ty0c * tiles_x + tx0c];
    float b10 = tile_bias[ty0c * tiles_x + tx1c];
    float b01 = tile_bias[ty1c * tiles_x + tx0c];
    float b11 = tile_bias[ty1c * tiles_x + tx1c];
    float s00 = tile_scale[ty0c * tiles_x + tx0c];
    float s10 = tile_scale[ty0c * tiles_x + tx1c];
    float s01 = tile_scale[ty1c * tiles_x + tx0c];
    float s11 = tile_scale[ty1c * tiles_x + tx1c];

    float bias = (1.0f - wx) * (1.0f - wy) * b00 + wx * (1.0f - wy) * b10
               + (1.0f - wx) * wy * b01 + wx * wy * b11;
    float scale = (1.0f - wx) * (1.0f - wy) * s00 + wx * (1.0f - wy) * s10
                + (1.0f - wx) * wy * s01 + wx * wy * s11;

    float v = bias + scale * src[i];
    v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
    out[i] = (unsigned char)v;
}
