// q4gemv.cu: 4-bit-weight GEMV (y = W x), the memory-bound kernel that dominates 1-user LLM decode.
//
// Weight format (simpler than llama.cpp's Q4_K, same bytes-per-weight ballpark):
//   symmetric int4, groups of 32 along K, one fp16 scale per group: w = (q - 8) * scale.
//   Row n: K/2 bytes; byte j holds k = 2j (low nibble) and k = 2j+1 (high nibble).
//   So one group = 16 bytes of weights + 2 bytes of scale = 4.5 bits per weight.
//
// Kernels (each step changes one thing, so its effect can be measured in isolation):
//   v0  one thread per row; each thread walks its own row        -> uncoalesced loads
//   v1  one warp per row; lanes read adjacent bytes              -> coalesced, 1-byte loads
//   v2  one warp per row; each lane loads 16 bytes (a group)     -> vectorized 128-bit loads
//   v3  v2 + x staged once per block in shared memory, rows grid-strided over a persistent grid
//   v4  v3 + two groups in flight per lane and streaming (evict-first) weight loads
//   cublas  dense fp16 GEMV of the same (dequantized) weights: vendor reference for achieved bandwidth
//
// Timing: weights/scales are replicated into enough copies to exceed 4x the L2 size, and launches
// rotate through the copies, so every launch reads its weights from DRAM (not from L2).
//
// Build:  nvcc -O3 -std=c++17 -arch=native -o q4gemv q4gemv.cu -lcublas
// Run:    ./q4gemv            -> one JSON line per (shape, kernel)
//         ./q4gemv --ncu      -> each kernel launched once on the gate_proj shape (for Nsight Compute)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifndef NO_CUBLAS
#include <cublas_v2.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
#ifndef NO_CUBLAS
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s_, __FILE__, __LINE__); exit(1); } } while (0)
#endif

constexpr int GROUP = 32;           // weights per scale
constexpr int GROUP_BYTES = 16;     // packed int4 bytes per group

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// dot of one 16-byte packed group with 32 activations (as 16 half2), unscaled
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

// v0: one thread per output row
__global__ void gemv_v0(const uint8_t* __restrict__ W, const half* __restrict__ S,
                        const half* __restrict__ x, float* __restrict__ y, int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const uint8_t* w = W + (size_t)n * (K / 2);
    const half* s = S + (size_t)n * (K / GROUP);
    float acc = 0.f;
    for (int g = 0; g < K / GROUP; ++g) {
        float part = 0.f;
        for (int j = 0; j < GROUP_BYTES; ++j) {
            const uint8_t b = w[g * GROUP_BYTES + j];
            const int k = g * GROUP + 2 * j;
            part += (float)((int)(b & 0xF) - 8) * __half2float(x[k]) + (float)((int)(b >> 4) - 8) * __half2float(x[k + 1]);
        }
        acc += __half2float(s[g]) * part;
    }
    y[n] = acc;
}

// v1: one warp per row, lanes read adjacent bytes (coalesced, but 1-byte loads)
__global__ void gemv_v1(const uint8_t* __restrict__ W, const half* __restrict__ S,
                        const half* __restrict__ x, float* __restrict__ y, int N, int K) {
    const int n = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (n >= N) return;
    const uint8_t* w = W + (size_t)n * (K / 2);
    const half* s = S + (size_t)n * (K / GROUP);
    const half2* x2 = reinterpret_cast<const half2*>(x);
    float acc = 0.f;
    for (int j = lane; j < K / 2; j += 32) {
        const uint8_t b = w[j];
        const float2 xf = __half22float2(x2[j]);
        acc += __half2float(s[j / GROUP_BYTES]) * ((float)((int)(b & 0xF) - 8) * xf.x + (float)((int)(b >> 4) - 8) * xf.y);
    }
    acc = warp_sum(acc);
    if (lane == 0) y[n] = acc;
}

// v2: one warp per row, each lane loads a whole 16-byte group with one 128-bit load
__global__ void gemv_v2(const uint8_t* __restrict__ W, const half* __restrict__ S,
                        const half* __restrict__ x, float* __restrict__ y, int N, int K) {
    const int n = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (n >= N) return;
    const int G = K / GROUP;
    const uint4* w4 = reinterpret_cast<const uint4*>(W + (size_t)n * (K / 2));
    const half* s = S + (size_t)n * G;
    const half2* x2 = reinterpret_cast<const half2*>(x);
    float acc = 0.f;
    for (int g = lane; g < G; g += 32) acc += __half2float(s[g]) * group_dot(w4[g], x2 + g * 16);
    acc = warp_sum(acc);
    if (lane == 0) y[n] = acc;
}

// v3: v2 + x staged in shared memory once per block; persistent grid, rows grid-strided
__global__ void gemv_v3(const uint8_t* __restrict__ W, const half* __restrict__ S,
                        const half* __restrict__ x, float* __restrict__ y, int N, int K) {
    extern __shared__ half2 xs[];               // K/2 half2
    const half2* x2 = reinterpret_cast<const half2*>(x);
    for (int i = threadIdx.x; i < K / 2; i += blockDim.x) xs[i] = x2[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    const int G = K / GROUP;
    for (int n = blockIdx.x * warps + warp; n < N; n += gridDim.x * warps) {
        const uint4* w4 = reinterpret_cast<const uint4*>(W + (size_t)n * (K / 2));
        const half* s = S + (size_t)n * G;
        float acc = 0.f;
        for (int g = lane; g < G; g += 32) acc += __half2float(s[g]) * group_dot(w4[g], xs + g * 16);
        acc = warp_sum(acc);
        if (lane == 0) y[n] = acc;
    }
}

// v4: v3 + two groups in flight per lane (both loads issued before the math) + streaming loads
__global__ void gemv_v4(const uint8_t* __restrict__ W, const half* __restrict__ S,
                        const half* __restrict__ x, float* __restrict__ y, int N, int K) {
    extern __shared__ half2 xs[];
    const half2* x2 = reinterpret_cast<const half2*>(x);
    for (int i = threadIdx.x; i < K / 2; i += blockDim.x) xs[i] = x2[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    const int G = K / GROUP;
    for (int n = blockIdx.x * warps + warp; n < N; n += gridDim.x * warps) {
        const uint4* w4 = reinterpret_cast<const uint4*>(W + (size_t)n * (K / 2));
        const half* s = S + (size_t)n * G;
        float acc = 0.f;
        int g = lane;
        for (; g + 32 < G; g += 64) {
            const uint4 a = __ldcs(w4 + g), b = __ldcs(w4 + g + 32);
            const half sa = s[g], sb = s[g + 32];
            acc += __half2float(sa) * group_dot(a, xs + g * 16) + __half2float(sb) * group_dot(b, xs + (g + 32) * 16);
        }
        if (g < G) acc += __half2float(s[g]) * group_dot(__ldcs(w4 + g), xs + g * 16);
        acc = warp_sum(acc);
        if (lane == 0) y[n] = acc;
    }
}

// fp16 dequantized copy for the cuBLAS baseline (row-major N x K)
__global__ void dequant(const uint8_t* __restrict__ W, const half* __restrict__ S, half* __restrict__ D, int N, int K) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;     // byte index
    if (idx >= (size_t)N * (K / 2)) return;
    const size_t n = idx / (K / 2), j = idx % (K / 2);
    const float sc = __half2float(S[n * (K / GROUP) + j / GROUP_BYTES]);
    const uint8_t b = W[idx];
    D[n * K + 2 * j] = __float2half((float)((int)(b & 0xF) - 8) * sc);
    D[n * K + 2 * j + 1] = __float2half((float)((int)(b >> 4) - 8) * sc);
}

struct Shape { const char* name; int N, K; };

// Qwen3-8B projections (hidden 4096, 32 q heads / 8 kv heads of dim 128, intermediate 12288, vocab 151936)
static const Shape SHAPES[] = {
    {"q_proj", 4096, 4096}, {"k_proj", 1024, 4096}, {"v_proj", 1024, 4096}, {"o_proj", 4096, 4096},
    {"gate_proj", 12288, 4096}, {"up_proj", 12288, 4096}, {"down_proj", 4096, 12288}, {"lm_head", 151936, 4096},
};

int main(int argc, char** argv) {
    const bool ncu_mode = argc > 1 && std::string(argv[1]) == "--ncu";
    int dev = 0; CK(cudaSetDevice(dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    int sms = prop.multiProcessorCount;
    const size_t l2 = (size_t)prop.l2CacheSize;
    fprintf(stderr, "GPU %s, %d SMs, L2 %.1f MB, cc %d.%d\n", prop.name, sms, l2 / 1e6, prop.major, prop.minor);

#ifndef NO_CUBLAS
    cublasHandle_t h; CB(cublasCreate(&h));
#endif
    std::mt19937 rng(1234);
    const int iters = ncu_mode ? 1 : 100, warm = ncu_mode ? 1 : 10;

    for (const Shape& sh : SHAPES) {
        if (ncu_mode && std::string(sh.name) != "gate_proj") continue;
        const int N = sh.N, K = sh.K, G = K / GROUP;
        const size_t wbytes = (size_t)N * K / 2, sbytes = (size_t)N * G * sizeof(half), dbytes = (size_t)N * K * sizeof(half);

        // host data
        std::vector<uint8_t> hW(wbytes);
        std::vector<half> hS((size_t)N * G), hx(K);
        std::uniform_int_distribution<int> qd(0, 255);
        std::uniform_real_distribution<float> sd(0.002f, 0.02f), xd(-1.f, 1.f);
        for (auto& b : hW) b = (uint8_t)qd(rng);
        for (auto& s : hS) s = __float2half(sd(rng));
        for (auto& v : hx) v = __float2half(xd(rng));

        // CPU reference (double), using the same fp16-rounded scales and activations
        std::vector<double> ref(N);
        std::vector<double> xf(K);
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

        // device copies: enough to exceed 4x L2 so timed launches read from DRAM
        const size_t per_copy = wbytes + sbytes;
        const int copies = ncu_mode ? 1 : (int)std::max<size_t>(2, (4 * l2 + per_copy - 1) / per_copy);
        std::vector<uint8_t*> dW(copies); std::vector<half*> dS(copies);
        for (int c = 0; c < copies; ++c) {
            CK(cudaMalloc(&dW[c], wbytes)); CK(cudaMalloc(&dS[c], sbytes));
            CK(cudaMemcpy(dW[c], hW.data(), wbytes, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dS[c], hS.data(), sbytes, cudaMemcpyHostToDevice));
        }
        half* dx; float* dy;
        CK(cudaMalloc(&dx, K * sizeof(half))); CK(cudaMalloc(&dy, N * sizeof(float)));
        CK(cudaMemcpy(dx, hx.data(), K * sizeof(half), cudaMemcpyHostToDevice));

        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        std::vector<float> hy(N);

        auto check = [&](double& maxrel) {
            CK(cudaMemcpy(hy.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
            double m = 0; for (int n = 0; n < N; ++n) m = std::max(m, std::fabs((double)hy[n] - ref[n]));
            maxrel = m / (refmax > 0 ? refmax : 1);
        };
        auto report = [&](const char* kname, double us, size_t bytes, double maxrel, double tol, int regs) {
            const double gbps = us > 0 ? bytes / (us * 1e-6) / 1e9 : 0.0;
            printf("{\"gpu\":\"%s\",\"shape\":\"%s\",\"N\":%d,\"K\":%d,\"kernel\":\"%s\",\"us\":%.3f,\"bytes\":%zu,"
                   "\"GBps\":%.1f,\"max_rel_err\":%.3e,\"ok\":%s,\"copies\":%d,\"regs\":%d}\n",
                   prop.name, sh.name, N, K, kname, us, bytes, gbps, maxrel, maxrel < tol ? "true" : "false", copies, regs);
            fflush(stdout);
        };

        const size_t qbytes = wbytes + sbytes + (size_t)K * sizeof(half) + (size_t)N * sizeof(float);
        const int bs = 256, warps = bs / 32;
        const size_t smem = (size_t)(K / 2) * sizeof(half2);
        int maxBlocksV3 = 0, maxBlocksV4 = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksV3, gemv_v3, bs, smem));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksV4, gemv_v4, bs, smem));
        const int gridV3 = std::max(1, std::min((N + warps - 1) / warps, sms * maxBlocksV3));
        const int gridV4 = std::max(1, std::min((N + warps - 1) / warps, sms * maxBlocksV4));

        struct K_ { const char* name; void (*fn)(const uint8_t*, const half*, const half*, float*, int, int); int grid, block; size_t smem; };
        const K_ ks[] = {
            {"v0_thread_per_row", gemv_v0, (N + bs - 1) / bs, bs, 0},
            {"v1_warp_per_row", gemv_v1, (N + warps - 1) / warps, bs, 0},
            {"v2_vector_loads", gemv_v2, (N + warps - 1) / warps, bs, 0},
            {"v3_smem_x_persistent", gemv_v3, gridV3, bs, smem},
            {"v4_two_in_flight_streaming", gemv_v4, gridV4, bs, smem},
        };
        for (const auto& k : ks) {
            cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k.fn));
            for (int i = 0; i < warm; ++i) k.fn<<<k.grid, k.block, k.smem>>>(dW[i % copies], dS[i % copies], dx, dy, N, K);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            double maxrel; check(maxrel);
            float ms = 0;
            if (!ncu_mode) {
                CK(cudaEventRecord(e0));
                for (int i = 0; i < iters; ++i) k.fn<<<k.grid, k.block, k.smem>>>(dW[i % copies], dS[i % copies], dx, dy, N, K);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1));
            }
            report(k.name, ms * 1000.0 / iters, qbytes, maxrel, 1e-4, fa.numRegs);
        }

#ifndef NO_CUBLAS
        // cuBLAS fp16 dense baseline on the dequantized weights (y = W x, W row-major N x K)
        {
            const int dcopies = ncu_mode ? 1 : (int)std::max<size_t>(2, (4 * l2 + dbytes - 1) / dbytes);
            std::vector<half*> dD(dcopies);
            for (int c = 0; c < dcopies; ++c) {
                CK(cudaMalloc(&dD[c], dbytes));
                const size_t nb = wbytes; dequant<<<(unsigned)((nb + 255) / 256), 256>>>(dW[0], dS[0], dD[c], N, K);
            }
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
            const float alpha = 1.f, beta = 0.f;
            auto run = [&](int i) {
                // column-major view: A = W^T is K x N (lda = K); y (N) = A^T (N x K) * x (K)
                CB(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, 1, K, &alpha, dD[i % dcopies], CUDA_R_16F, K,
                                dx, CUDA_R_16F, K, &beta, dy, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            };
            for (int i = 0; i < warm; ++i) run(i);
            CK(cudaDeviceSynchronize());
            double maxrel; check(maxrel);
            float ms = 0;
            if (!ncu_mode) {
                CK(cudaEventRecord(e0));
                for (int i = 0; i < iters; ++i) run(i);
                CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaEventElapsedTime(&ms, e0, e1));
            }
            // fp16 weights are rounded on dequantization, so allow a looser tolerance
            report("cublas_fp16_dense", ms * 1000.0 / iters, dbytes + (size_t)K * sizeof(half) + (size_t)N * sizeof(float), maxrel, 5e-3, 0);
            for (auto p : dD) CK(cudaFree(p));
        }
#endif

        for (int c = 0; c < copies; ++c) { CK(cudaFree(dW[c])); CK(cudaFree(dS[c])); }
        CK(cudaFree(dx)); CK(cudaFree(dy));
        CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    }
#ifndef NO_CUBLAS
    CB(cublasDestroy(h));
#endif
    return 0;
}
