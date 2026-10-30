// gemv_sweep.cu: the q4gemv.cu v2/v3/v4 kernels with the two latency-hiding knobs exposed.
//
// Same weight format as q4gemv.cu: symmetric int4, groups of 32 along K, one fp16 scale per group,
// w = (q - 8) * scale; row n is K/2 bytes, a group is 16 bytes (one uint4 load) + a 2-byte scale.
//
// Little's Law (Volkov 2016, "Understanding Latency Hiding on GPUs", UC Berkeley PhD thesis):
//     achieved bandwidth = bytes in flight / memory latency      (until it hits the DRAM peak)
//     bytes in flight    = warps_per_sm * 32 lanes * ILP * 16 B * sm_count
// so there are two ways to hide DRAM latency, and this sweep varies both:
//
//   ILP  (compile time, template int ILP in {1,2,4}): each lane issues ILP independent 16-byte group
//        loads (groups g, g+32, ..., g+32*(ILP-1) of its row) BEFORE doing any math on them.
//        q4gemv v2/v3 are ILP=1; q4gemv v4 is ILP=2 + streaming loads.
//   TLP  (run time, blocks_per_sm): resident warps per SM. Limited by padding the dynamic shared
//        memory so that at most b blocks fit per SM; the achieved value comes from
//        cudaOccupancyMaxActiveBlocksPerMultiprocessor and is what we report (with warps_per_sm).
//
// Kernels (same structure as q4gemv.cu):
//   v2<ILP>  one warp per row, x read from global memory (L1-cached), one-warp-per-row grid
//   v3<ILP>  x staged once per block in shared memory, persistent grid (rows grid-strided)
//   v4<ILP>  v3 + streaming (__ldcs, evict-first) weight loads
//
// Shapes (Qwen3-8B): q_proj 4096x4096, gate_proj 12288x4096, down_proj 4096x12288 (N x K).
// Correctness: every config is checked against a CPU double-precision reference (as in q4gemv.cu).
// Timing: weights/scales replicated into copies totalling > 4x L2, launches rotate through them, so
// every timed launch reads its weights from DRAM.
//
// Output: one JSON line per config (gemv_sweep.jsonl schema in CONTRACTS.md; extras listed in report()).
//
// Build: nvcc -O3 -std=c++17 -arch=native -o gemv_sweep gemv_sweep.cu
// Run:   ./gemv_sweep               timed sweep -> stdout JSONL
//        ./gemv_sweep --ncu         each config launched exactly once on one weight copy (for Nsight
//                                   Compute); records are printed in launch order with "ncu_launch".
//        ./gemv_sweep --block 128   threads per block (default 256)

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    throw std::runtime_error(std::string(cudaGetErrorString(e_)) + " at " #x); } } while (0)

constexpr int GROUP = 32;           // weights per scale
constexpr int GROUP_BYTES = 16;     // packed int4 bytes per group

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// dot of one 16-byte packed group with 32 activations (as 16 half2), unscaled (identical to q4gemv.cu)
__device__ __forceinline__ float group_dot(uint4 v, const half2* x2) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    float part = 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            const uint32_t byte = (w[i] >> (8 * b)) & 0xFFu;
            const float2 xf = __half22float2(x2[i * 4 + b]);
            part += (float)((int)(byte & 0xFu) - 8) * xf.x + (float)((int)(byte >> 4) - 8) * xf.y;
        }
    }
    return part;
}

// One lane's share of one row: groups lane, lane+32, lane+64, ...
// ILP groups are loaded (weights + scales) into registers first, then consumed: the ILP loads are
// independent, so they are all in flight together. STREAMING selects __ldcs (evict-first) loads.
template <int ILP, bool STREAMING>
__device__ __forceinline__ float lane_row_dot(const uint4* __restrict__ w4, const half* __restrict__ s,
                                              const half2* x2, int G, int lane) {
    float acc = 0.f;
    int g = lane;
    // "#pragma unroll 1": without it nvcc may unroll this loop and hoist the next iteration's loads,
    // which would make the real ILP larger than the template ILP we report.
    #pragma unroll 1
    for (; g + 32 * (ILP - 1) < G; g += 32 * ILP) {
        uint4 wv[ILP];
        half sv[ILP];
        #pragma unroll
        for (int i = 0; i < ILP; ++i) {                     // 1) issue all loads
            wv[i] = STREAMING ? __ldcs(w4 + g + 32 * i) : w4[g + 32 * i];
            sv[i] = s[g + 32 * i];
        }
        #pragma unroll
        for (int i = 0; i < ILP; ++i)                       // 2) then do the math
            acc += __half2float(sv[i]) * group_dot(wv[i], x2 + (g + 32 * i) * 16);
    }
    for (; g < G; g += 32) {                                // tail: fewer than ILP groups left for this lane
        const uint4 v = STREAMING ? __ldcs(w4 + g) : w4[g];
        acc += __half2float(s[g]) * group_dot(v, x2 + g * 16);
    }
    return acc;
}

// v2: one warp per row, x from global memory. (The dynamic smem is occupancy padding only.)
template <int ILP>
__global__ void gemv_v2(const uint8_t* __restrict__ W, const half* __restrict__ S,
                        const half* __restrict__ x, float* __restrict__ y, int N, int K) {
    const int n = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (n >= N) return;
    const int G = K / GROUP;
    const uint4* w4 = reinterpret_cast<const uint4*>(W + (size_t)n * (K / 2));
    float acc = lane_row_dot<ILP, false>(w4, S + (size_t)n * G, reinterpret_cast<const half2*>(x), G, lane);
    acc = warp_sum(acc);
    if (lane == 0) y[n] = acc;
}

// v3 / v4: x staged in shared memory once per block, persistent grid, rows grid-strided.
// The first K/2 half2 of dynamic smem hold x; any extra dynamic smem is occupancy padding.
template <int ILP, bool STREAMING>
__device__ __forceinline__ void gemv_persistent(const uint8_t* __restrict__ W, const half* __restrict__ S,
                                                const half* __restrict__ x, float* __restrict__ y, int N, int K) {
    extern __shared__ half2 xs[];
    const half2* x2 = reinterpret_cast<const half2*>(x);
    for (int i = threadIdx.x; i < K / 2; i += blockDim.x) xs[i] = x2[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    const int G = K / GROUP;
    #pragma unroll 1
    for (int n = blockIdx.x * warps + warp; n < N; n += gridDim.x * warps) {
        const uint4* w4 = reinterpret_cast<const uint4*>(W + (size_t)n * (K / 2));
        float acc = lane_row_dot<ILP, STREAMING>(w4, S + (size_t)n * G, xs, G, lane);
        acc = warp_sum(acc);
        if (lane == 0) y[n] = acc;
    }
}
template <int ILP>
__global__ void gemv_v3(const uint8_t* __restrict__ W, const half* __restrict__ S,
                        const half* __restrict__ x, float* __restrict__ y, int N, int K) {
    gemv_persistent<ILP, false>(W, S, x, y, N, K);
}
template <int ILP>
__global__ void gemv_v4(const uint8_t* __restrict__ W, const half* __restrict__ S,
                        const half* __restrict__ x, float* __restrict__ y, int N, int K) {
    gemv_persistent<ILP, true>(W, S, x, y, N, K);
}

using KernelFn = void (*)(const uint8_t*, const half*, const half*, float*, int, int);

struct KernelVariant { const char* name; int ilp; bool persistent; KernelFn fn; };
static const KernelVariant VARIANTS[] = {
    {"v2", 1, false, gemv_v2<1>}, {"v2", 2, false, gemv_v2<2>}, {"v2", 4, false, gemv_v2<4>},
    {"v3", 1, true, gemv_v3<1>},  {"v3", 2, true, gemv_v3<2>},  {"v3", 4, true, gemv_v3<4>},
    {"v4", 1, true, gemv_v4<1>},  {"v4", 2, true, gemv_v4<2>},  {"v4", 4, true, gemv_v4<4>},
};

struct Shape { const char* name; int N, K; };
static const Shape SHAPES[] = {{"q_proj", 4096, 4096}, {"gate_proj", 12288, 4096}, {"down_proj", 4096, 12288}};

// Smallest dynamic smem >= base such that at most `want` blocks fit per SM (occupancy is
// non-increasing in smem per block, so binary search). *achieved = resulting blocks per SM.
static size_t smem_for_blocks(KernelFn fn, int threads, int want, size_t base, size_t smem_max, int* achieved) {
    int occ = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, threads, base));
    if (occ <= want) { *achieved = occ; return base; }
    size_t lo = base, hi = smem_max;
    while (hi - lo > 128) {
        const size_t mid = (lo + hi) / 2;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, threads, mid));
        if (occ <= want) hi = mid; else lo = mid;
    }
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, threads, hi));
    *achieved = occ;
    return hi;
}

int main(int argc, char** argv) {
    bool ncu_mode = false;
    int block = 256;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--ncu")) ncu_mode = true;
        else if (!strcmp(argv[i], "--block") && i + 1 < argc) block = atoi(argv[++i]);
    }
    if (block < 32 || block > 1024 || block % 32) { fprintf(stderr, "--block must be a multiple of 32 in [32,1024]\n"); return 2; }

    cudaDeviceProp prop;
    try { CK(cudaSetDevice(0)); CK(cudaGetDeviceProperties(&prop, 0)); }
    catch (const std::exception& e) { fprintf(stderr, "gemv_sweep: cannot open device: %s\n", e.what()); return 2; }
    const int sms = prop.multiProcessorCount;
    const size_t l2 = (size_t)prop.l2CacheSize;
    const size_t smem_max = prop.sharedMemPerBlockOptin;
    fprintf(stderr, "GPU %s, %d SMs, L2 %.1f MB, cc %d.%d, block %d, mode %s\n",
            prop.name, sms, l2 / 1e6, prop.major, prop.minor, block, ncu_mode ? "ncu" : "timed");

    // Allow every kernel to use up to the opt-in shared memory limit (needed for the padding).
    try {
        for (const auto& v : VARIANTS)
            CK(cudaFuncSetAttribute(v.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_max));
    } catch (const std::exception& e) { fprintf(stderr, "gemv_sweep: %s\n", e.what()); return 2; }

    std::mt19937 rng(1234);
    const int iters = 100, warm = 10;
    const int warps_per_block = block / 32;
    int failures = 0, launch_index = 0;

    for (const Shape& sh : SHAPES) {
        const int N = sh.N, K = sh.K, G = K / GROUP;
        const size_t wbytes = (size_t)N * K / 2, sbytes = (size_t)N * G * sizeof(half);

        // Host data and CPU double reference (same logic as q4gemv.cu).
        std::vector<uint8_t> hW(wbytes);
        std::vector<half> hS((size_t)N * G), hx(K);
        std::uniform_int_distribution<int> qd(0, 255);
        std::uniform_real_distribution<float> sd(0.002f, 0.02f), xd(-1.f, 1.f);
        for (auto& b : hW) b = (uint8_t)qd(rng);
        for (auto& s : hS) s = __float2half(sd(rng));
        for (auto& v : hx) v = __float2half(xd(rng));
        std::vector<double> ref(N), xf(K);
        for (int k = 0; k < K; ++k) xf[k] = (double)__half2float(hx[k]);
        for (int n = 0; n < N; ++n) {
            double acc = 0;
            const uint8_t* w = &hW[(size_t)n * (K / 2)];
            for (int g = 0; g < G; ++g) {
                double part = 0;
                for (int j = 0; j < GROUP_BYTES; ++j) {
                    const uint8_t b = w[g * GROUP_BYTES + j];
                    const int k = g * GROUP + 2 * j;
                    part += (double)((int)(b & 0xF) - 8) * xf[k] + (double)((int)(b >> 4) - 8) * xf[k + 1];
                }
                acc += (double)__half2float(hS[(size_t)n * G + g]) * part;
            }
            ref[n] = acc;
        }
        double refmax = 0; for (double r : ref) refmax = std::max(refmax, std::fabs(r));

        // Device copies: > 4x L2 in total so timed launches read weights from DRAM (1 copy in --ncu mode).
        const size_t per_copy = wbytes + sbytes;
        const int copies = ncu_mode ? 1 : (int)std::max<size_t>(2, (4 * l2 + per_copy - 1) / per_copy + 1);
        std::vector<uint8_t*> dW; std::vector<half*> dS;
        half* dx = nullptr; float* dy = nullptr;
        cudaEvent_t e0 = nullptr, e1 = nullptr;
        bool setup_ok = true;
        try {
            for (int c = 0; c < copies; ++c) {
                uint8_t* w = nullptr; half* s = nullptr;
                CK(cudaMalloc(&w, wbytes)); dW.push_back(w);
                CK(cudaMalloc(&s, sbytes)); dS.push_back(s);
                CK(cudaMemcpy(w, hW.data(), wbytes, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(s, hS.data(), sbytes, cudaMemcpyHostToDevice));
            }
            CK(cudaMalloc(&dx, K * sizeof(half))); CK(cudaMalloc(&dy, N * sizeof(float)));
            CK(cudaMemcpy(dx, hx.data(), K * sizeof(half), cudaMemcpyHostToDevice));
            CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        } catch (const std::exception& e) {
            fprintf(stderr, "ERROR setting up shape %s: %s\n", sh.name, e.what());
            ++failures; setup_ok = false;
        }

        // Bytes that must move per launch: weights + scales + x + y (same as q4gemv.cu).
        const size_t qbytes = wbytes + sbytes + (size_t)K * sizeof(half) + (size_t)N * sizeof(float);
        const size_t x_smem = (size_t)(K / 2) * sizeof(half2);
        std::vector<float> hy(N);

        for (const auto& v : VARIANTS) {
            if (!setup_ok) break;
            int max_bps = 0;
            const size_t base_smem = v.persistent ? x_smem : 0;   // v3/v4 need x in smem; v2 needs none
            try { CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_bps, v.fn, block, base_smem)); }
            catch (const std::exception& e) { fprintf(stderr, "ERROR occupancy %s: %s\n", v.name, e.what()); ++failures; continue; }
            int last_achieved = -1;
            for (int want = 1; want <= max_bps; ++want) {
                try {
                    int bps = 0;
                    const size_t smem = smem_for_blocks(v.fn, block, want, base_smem, smem_max, &bps);
                    if (bps < 1 || bps == last_achieved) continue;
                    last_achieved = bps;
                    // v2: one warp per row (grid may exceed the resident blocks; the rest queue up).
                    // v3/v4: persistent grid of exactly the resident blocks (capped by the number of rows).
                    const int row_blocks = (N + warps_per_block - 1) / warps_per_block;
                    const int grid = v.persistent ? std::max(1, std::min(row_blocks, bps * sms)) : row_blocks;
                    // Warps actually resident per SM on average (less than bps*warps if the grid is small).
                    const double warps_eff = (double)std::min(grid, bps * sms) * warps_per_block / sms;
                    cudaFuncAttributes fa{}; CK(cudaFuncGetAttributes(&fa, v.fn));

                    // Correctness launch (y pre-filled with NaN bytes so an unwritten row fails the check).
                    CK(cudaMemset(dy, 0xFF, N * sizeof(float)));
                    v.fn<<<grid, block, smem>>>(dW[0], dS[0], dx, dy, N, K);
                    CK(cudaGetLastError());
                    const int this_launch = launch_index++;   // in --ncu mode this is the ncu row index
                    CK(cudaDeviceSynchronize());
                    CK(cudaMemcpy(hy.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
                    double maxabs = 0; bool finite = true;
                    for (int n = 0; n < N; ++n) {
                        if (!std::isfinite(hy[n])) finite = false;
                        else maxabs = std::max(maxabs, std::fabs((double)hy[n] - ref[n]));
                    }
                    const double maxrel = finite ? maxabs / (refmax > 0 ? refmax : 1) : INFINITY;
                    const bool ok = finite && maxrel < 1e-4;

                    double us = 0.0;
                    if (!ncu_mode) {
                        for (int i = 0; i < warm; ++i) v.fn<<<grid, block, smem>>>(dW[i % copies], dS[i % copies], dx, dy, N, K);
                        CK(cudaGetLastError());
                        CK(cudaEventRecord(e0));
                        for (int i = 0; i < iters; ++i) v.fn<<<grid, block, smem>>>(dW[i % copies], dS[i % copies], dx, dy, N, K);
                        CK(cudaEventRecord(e1));
                        CK(cudaGetLastError());
                        CK(cudaEventSynchronize(e1));
                        float ms = 0.f; CK(cudaEventElapsedTime(&ms, e0, e1));
                        us = ms * 1000.0 / iters;
                    }
                    const double gbps = us > 0 ? qbytes / (us * 1e-6) / 1e9 : 0.0;
                    // Contract fields first, then extras. In --ncu mode us/GBps are 0 (not timed) and
                    // ncu_launch is this kernel's launch index (ncu rows are matched to records by it).
                    printf("{\"kernel\":\"%s\",\"ilp\":%d,\"blocks_per_sm\":%d,\"warps_per_sm\":%d,\"N\":%d,\"K\":%d,"
                           "\"us\":%.3f,\"bytes\":%zu,\"GBps\":%.1f,\"ok\":%s,"
                           "\"gpu\":\"%s\",\"shape\":\"%s\",\"requested_blocks_per_sm\":%d,\"threads_per_block\":%d,"
                           "\"grid\":%d,\"warps_per_sm_eff\":%.3f,\"smem_bytes\":%zu,\"regs\":%d,\"sm_count\":%d,"
                           "\"l2_bytes\":%zu,\"copies\":%d,\"max_rel_err\":%.3e,\"mode\":\"%s\",\"ncu_launch\":%d}\n",
                           v.name, v.ilp, bps, bps * warps_per_block, N, K, us, qbytes, gbps, ok ? "true" : "false",
                           prop.name, sh.name, want, block, grid, warps_eff, smem, fa.numRegs, sms,
                           l2, copies, maxrel, ncu_mode ? "ncu" : "timed", ncu_mode ? this_launch : -1);
                    fflush(stdout);
                } catch (const std::exception& e) {
                    ++failures;
                    fprintf(stderr, "ERROR %s ILP=%d %s bps=%d: %s\n", v.name, v.ilp, sh.name, want, e.what());
                }
                if (cudaDeviceSynchronize() != cudaSuccess) {
                    fprintf(stderr, "CUDA context is unusable; stopping\n");
                    return 3;
                }
            }
        }
        for (auto p : dW) cudaFree(p);
        for (auto p : dS) cudaFree(p);
        cudaFree(dx); cudaFree(dy);
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
    }
    return failures ? 1 : 0;
}
