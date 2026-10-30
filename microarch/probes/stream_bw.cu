// stream_bw.cu: read-only DRAM bandwidth as a function of bytes in flight.
//
// Little's Law (Little 1961; applied to GPUs by Volkov, "Understanding Latency Hiding on GPUs",
// PhD thesis, UC Berkeley 2016):   bandwidth = bytes_in_flight / latency.
// To saturate DRAM, the GPU must keep  peak_bandwidth x latency  bytes of loads outstanding.
// This probe controls bytes in flight directly with three knobs and measures the bandwidth:
//
//   blocks_per_sm      resident blocks per SM (forced with dynamic-shared-memory padding, see below)
//   threads_per_block  64 .. 1024
//   loads_per_thread   LOADS in {1,2,4,8} independent 16-byte (uint4) loads issued back to back
//                      BEFORE any of their results is used (memory-level parallelism per thread)
//
//   bytes_in_flight_total = blocks_per_sm * threads_per_block * LOADS * 16 B * sm_count
//
// Forcing blocks_per_sm: occupancy is limited by whichever resource runs out first. We request extra
// dynamic shared memory per block (the kernel never touches it) so that at most b blocks fit in an SM's
// shared memory. The smallest padding that achieves this is found with the CUDA occupancy API, and the
// ACHIEVED blocks per SM (cudaOccupancyMaxActiveBlocksPerMultiprocessor) is what we report. The grid is
// exactly achieved_blocks_per_sm * sm_count blocks, all resident at once, each grid-striding over the buffer.
//
// The buffer is >= 8x the L2 size (prop.l2CacheSize), so streaming reads come from DRAM.
//
// Output: one JSON line per config ("probe":"stream", schema in CONTRACTS.md, extras: gpu, sm_count,
// loads_per_thread, smem_pad_bytes, regs), then one "probe":"stream_peak" line with the best config.
//
// Build: nvcc -O3 -std=c++17 -arch=native -o stream_bw stream_bw.cu
// Run:   ./stream_bw

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    throw std::runtime_error(std::string(cudaGetErrorString(e_)) + " at " #x); } } while (0)

// Read n_vec uint4 (16-byte) elements. Each thread handles LOADS elements per outer iteration:
// element indices base + t, base + t + T, ..., base + t + (LOADS-1)*T, where T = total threads,
// so every individual load instruction is perfectly coalesced across the warp (32 x 16 B = 512 B).
template <int LOADS>
__global__ void stream_read(const uint4* __restrict__ buf, size_t n_vec, unsigned int* __restrict__ sink) {
    extern __shared__ unsigned char pad[];             // occupancy padding only; never accessed
    const size_t T = (size_t)gridDim.x * blockDim.x;
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int acc = 0;
    size_t base = 0;
    // "#pragma unroll 1" stops the compiler from unrolling the outer loop and hoisting the NEXT
    // iteration's loads above this iteration's math, which would silently raise loads in flight.
    #pragma unroll 1
    for (; base + (LOADS - 1) * T + t < n_vec; base += LOADS * T) {
        uint4 r[LOADS];
        #pragma unroll
        for (int i = 0; i < LOADS; ++i) r[i] = buf[base + i * T + t];     // issue all LOADS loads ...
        #pragma unroll
        for (int i = 0; i < LOADS; ++i) acc ^= r[i].x ^ r[i].y ^ r[i].z ^ r[i].w;   // ... then use them
    }
    for (size_t i = base + t; i < n_vec; i += T) {     // tail (fewer than LOADS*T elements left)
        const uint4 v = buf[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    // The buffer is zero-filled so acc == 0; the impossible-looking test keeps the loads alive
    // (the compiler cannot prove acc != magic, so it must perform every load).
    if (acc == 0x9E3779B9u) sink[0] = acc;
}

using KernelFn = void (*)(const uint4*, size_t, unsigned int*);

// Smallest dynamic smem (bytes) such that at most `want` blocks of `threads` fit per SM.
// Returns the padding and writes the achieved blocks/SM into *achieved.
static size_t pad_for_blocks(KernelFn fn, int threads, int want, int* achieved, size_t smem_max) {
    int occ = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, threads, 0));
    if (occ <= want) { *achieved = occ; return 0; }    // already limited by threads/registers/block slots
    // Occupancy is non-increasing in shared memory per block: binary search the smallest padding.
    size_t lo = 0, hi = smem_max;                      // occ(lo) > want, occ(hi) <= want (hi = full SM)
    while (hi - lo > 128) {                            // 128 B is below the smem allocation granularity
        const size_t mid = (lo + hi) / 2;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, threads, mid));
        if (occ <= want) hi = mid; else lo = mid;
    }
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, fn, threads, hi));
    *achieved = occ;
    return hi;
}

struct Result { int bps, tpb, loads; size_t bif_total; double gbps; };

int main() {
    cudaDeviceProp prop;
    try {
        CK(cudaSetDevice(0));
        CK(cudaGetDeviceProperties(&prop, 0));
    } catch (const std::exception& e) {
        fprintf(stderr, "stream_bw: cannot open device: %s\n", e.what());
        return 2;
    }
    const int sms = prop.multiProcessorCount;
    const size_t l2 = (size_t)prop.l2CacheSize;
    const size_t smem_max = prop.sharedMemPerBlockOptin;     // largest dynamic smem a block may opt into
    fprintf(stderr, "GPU %s, %d SMs, L2 %.1f MB, max threads/SM %d, smem/SM %zu, smem/block opt-in %zu\n",
            prop.name, sms, l2 / 1e6, prop.maxThreadsPerMultiProcessor, prop.sharedMemPerMultiprocessor, smem_max);

    // Buffer: 8x L2, rounded up to a multiple of 1 MB.
    const size_t buf_bytes = ((8 * l2 + (1 << 20) - 1) >> 20) << 20;
    const size_t n_vec = buf_bytes / sizeof(uint4);
    uint4* d_buf = nullptr; unsigned int* d_sink = nullptr;
    try {
        CK(cudaMalloc(&d_buf, buf_bytes));
        CK(cudaMalloc(&d_sink, sizeof(unsigned int)));
        CK(cudaMemset(d_buf, 0, buf_bytes));
        CK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        fprintf(stderr, "stream_bw: allocation of %zu B failed: %s\n", buf_bytes, e.what());
        return 2;
    }
    fprintf(stderr, "buffer %.1f MB (%.1fx L2)\n", buf_bytes / 1e6, (double)buf_bytes / l2);

    struct KV { int loads; KernelFn fn; };
    const KV kernels[] = {{1, stream_read<1>}, {2, stream_read<2>}, {4, stream_read<4>}, {8, stream_read<8>}};
    const int tpbs[] = {64, 128, 256, 512, 1024};
    const int warm = 2, iters = 5;
    std::vector<Result> results;
    int failures = 0;

    for (const KV& k : kernels) {
        cudaFuncAttributes fa{};
        try {
            // Allow padding up to the opt-in limit (default limit for dynamic smem is 48 KB).
            CK(cudaFuncSetAttribute(k.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_max));
            CK(cudaFuncGetAttributes(&fa, k.fn));
        } catch (const std::exception& e) {
            fprintf(stderr, "ERROR setting up LOADS=%d: %s\n", k.loads, e.what()); ++failures; continue;
        }
        for (int tpb : tpbs) {
            int max_bps = 0;
            try { CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_bps, k.fn, tpb, 0)); }
            catch (const std::exception& e) { fprintf(stderr, "ERROR occupancy: %s\n", e.what()); ++failures; continue; }
            int last_achieved = -1;
            for (int want = 1; want <= max_bps; ++want) {
                cudaEvent_t e0 = nullptr, e1 = nullptr;
                try {
                    int bps = 0;
                    const size_t pad = pad_for_blocks(k.fn, tpb, want, &bps, smem_max);
                    if (bps < 1 || bps == last_achieved) continue;      // unreachable or duplicate config
                    last_achieved = bps;
                    const int grid = bps * sms;
                    for (int i = 0; i < warm; ++i) k.fn<<<grid, tpb, pad>>>(d_buf, n_vec, d_sink);
                    CK(cudaGetLastError());
                    CK(cudaDeviceSynchronize());
                    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
                    CK(cudaEventRecord(e0));
                    for (int i = 0; i < iters; ++i) k.fn<<<grid, tpb, pad>>>(d_buf, n_vec, d_sink);
                    CK(cudaEventRecord(e1));
                    CK(cudaGetLastError());
                    CK(cudaEventSynchronize(e1));
                    float ms = 0.f; CK(cudaEventElapsedTime(&ms, e0, e1));
                    const double s = ms * 1e-3 / iters;
                    const double gbps = (double)buf_bytes / s / 1e9;
                    const int bpt = k.loads * (int)sizeof(uint4);                       // bytes per thread in flight
                    const size_t bif = (size_t)bps * tpb * bpt * sms;                     // whole GPU
                    printf("{\"probe\":\"stream\",\"blocks_per_sm\":%d,\"threads_per_block\":%d,"
                           "\"bytes_per_thread_in_flight\":%d,\"bytes_in_flight_total\":%zu,\"GBps\":%.2f,"
                           "\"gpu\":\"%s\",\"sm_count\":%d,\"l2_bytes\":%zu,\"loads_per_thread\":%d,"
                           "\"smem_pad_bytes\":%zu,\"regs\":%d,\"buffer_bytes\":%zu}\n",
                           bps, tpb, bpt, bif, gbps, prop.name, sms, l2, k.loads, pad, fa.numRegs, buf_bytes);
                    fflush(stdout);
                    results.push_back({bps, tpb, k.loads, bif, gbps});
                } catch (const std::exception& e) {
                    ++failures;
                    fprintf(stderr, "ERROR LOADS=%d tpb=%d bps=%d: %s\n", k.loads, tpb, want, e.what());
                }
                if (e0) cudaEventDestroy(e0);
                if (e1) cudaEventDestroy(e1);
                if (cudaDeviceSynchronize() != cudaSuccess) {
                    fprintf(stderr, "CUDA context is unusable; stopping\n");
                    return 3;
                }
            }
        }
    }

    // Peak achievable bandwidth = best config of the sweep (used as peak_bw_Bps in perfsim.microarch).
    if (!results.empty()) {
        const Result best = *std::max_element(results.begin(), results.end(),
                                              [](const Result& a, const Result& b) { return a.gbps < b.gbps; });
        printf("{\"probe\":\"stream_peak\",\"blocks_per_sm\":%d,\"threads_per_block\":%d,"
               "\"bytes_per_thread_in_flight\":%d,\"bytes_in_flight_total\":%zu,\"GBps\":%.2f,"
               "\"gpu\":\"%s\",\"sm_count\":%d,\"l2_bytes\":%zu}\n",
               best.bps, best.tpb, best.loads * 16, best.bif_total, best.gbps, prop.name, sms, l2);
        fflush(stdout);
        fprintf(stderr, "peak achievable read bandwidth: %.1f GB/s (bps=%d tpb=%d loads=%d)\n",
                best.gbps, best.bps, best.tpb, best.loads);
    }
    cudaFree(d_buf); cudaFree(d_sink);
    return failures ? 1 : 0;
}
