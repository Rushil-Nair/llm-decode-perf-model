// pointer_chase.cu: dependent-load latency probe (L1 / L2 / DRAM latency plateaus).
//
// Method (the classic "pointer chase", e.g. Mei & Chu 2016, Jia et al. 2018 "Dissecting Volta"):
//   * The buffer is split into 128-byte lines (STRIDE). Only the first 4-byte word of each line is used.
//   * Word 0 of line i holds the word index of the next line to visit. The visit order is a single
//     random cycle through all lines of the footprint (Sattolo's algorithm), so the hardware prefetchers
//     and DRAM row-buffer locality cannot help, and every line is visited before any line repeats.
//   * One thread does j = buf[j] in a loop. Each load's address depends on the previous load's value,
//     so loads cannot overlap: time / steps = latency of one load (load-to-use latency).
//   * As the footprint grows past L1, then past L2, the per-load latency steps up: the plateaus are the
//     L1, L2 and DRAM latencies. Very large footprints also add TLB-miss latency (the last plateau rises).
//
// Two clocks are reported for the same chase:
//   latency_cycles = (clock64() end - start) / steps, measured inside the kernel in SM clock cycles
//   latency_ns     = cudaEvent elapsed time / steps (includes ~a few us of launch overhead, negligible
//                    because each timed chase runs for milliseconds)
//   Their ratio gives the SM clock the chase actually ran at (clock_mhz_est). We do NOT use
//   cudaDevAttrClockRate / cudaDeviceProp::clockRate (deprecated / removed in CUDA 13).
//
// Output: one JSON line per footprint (probes.jsonl schema in CONTRACTS.md), extras: gpu, steps,
//         clock_mhz_est. Progress and errors go to stderr.
//
// Build: nvcc -O3 -std=c++17 -arch=native -o pointer_chase pointer_chase.cu
// Run:   ./pointer_chase [max_footprint_MB]          (default 1024 MB)

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Throw on any CUDA error so a failing footprint can be skipped (see main) instead of aborting silently.
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    throw std::runtime_error(std::string(cudaGetErrorString(e_)) + " at " #x); } } while (0)

constexpr int STRIDE_BYTES = 128;                       // one L2 cache line (and one L1 line) per visited node
constexpr int STRIDE_WORDS = STRIDE_BYTES / 4;          // node spacing in uint32 words

// Scatter the permutation into the line-structured buffer: buf[line * 32] = next_line * 32.
// Doing this on the GPU means the host only builds n_lines integers (32 MB for 1 GB), not the full buffer.
__global__ void scatter_next(uint32_t* buf, const uint32_t* next_line, uint32_t n_lines) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_lines) buf[i * STRIDE_WORDS] = next_line[i] * STRIDE_WORDS;
}

// The chase itself: launched with exactly one thread.
// out[0] = final index (stored so the compiler cannot delete the loop), out[1] = elapsed SM cycles.
__global__ void chase(const uint32_t* buf, uint32_t steps, unsigned long long* out) {
    uint32_t j = 0;
    const long long t0 = clock64();
    #pragma unroll 1                                   // keep the loop simple: one load per iteration
    for (uint32_t i = 0; i < steps; ++i) j = buf[j];   // next address depends on this load's result
    const long long t1 = clock64();
    out[0] = j;
    out[1] = (unsigned long long)(t1 - t0);
}

int main(int argc, char** argv) {
    const double max_mb = argc > 1 ? atof(argv[1]) : 1024.0;
    int dev = 0;
    cudaDeviceProp prop;
    try {
        CK(cudaSetDevice(dev));
        CK(cudaGetDeviceProperties(&prop, dev));
    } catch (const std::exception& e) {
        fprintf(stderr, "pointer_chase: cannot open device: %s\n", e.what());
        return 2;
    }
    fprintf(stderr, "GPU %s, cc %d.%d, L2 %.1f MB\n", prop.name, prop.major, prop.minor, prop.l2CacheSize / 1e6);
    const size_t l2 = (size_t)prop.l2CacheSize;

    // Footprints 16 KB .. max (default 1 GB) in steps of sqrt(2): 33 points for 16 KB..1 GB.
    std::vector<size_t> footprints;
    for (double f = 16.0 * 1024; f <= max_mb * 1024 * 1024 * 1.0001; f *= std::sqrt(2.0)) {
        const size_t b = ((size_t)f / STRIDE_BYTES) * STRIDE_BYTES;   // whole number of lines
        if (footprints.empty() || footprints.back() != b) footprints.push_back(b);
    }

    const uint32_t TIMED_STEPS = 1u << 18;   // 262144 dependent loads: ~0.2 s at ~800 ns DRAM latency
    std::mt19937_64 rng(42);
    int failures = 0;

    for (size_t fp : footprints) {
        uint32_t* d_buf = nullptr; uint32_t* d_next = nullptr; unsigned long long* d_out = nullptr;
        cudaEvent_t e0 = nullptr, e1 = nullptr;
        try {
            const uint32_t n_lines = (uint32_t)(fp / STRIDE_BYTES);
            size_t free_b = 0, total_b = 0;
            CK(cudaMemGetInfo(&free_b, &total_b));
            if (fp + (size_t)n_lines * 4 > free_b * 8 / 10) {           // leave headroom; skip, don't crash
                fprintf(stderr, "skip footprint %zu B: only %zu B free\n", fp, free_b);
                continue;
            }

            // Sattolo's algorithm: a uniformly random permutation that is ONE cycle of length n_lines,
            // so following next[] from line 0 visits every line exactly once before returning to 0.
            std::vector<uint32_t> order(n_lines);
            for (uint32_t i = 0; i < n_lines; ++i) order[i] = i;
            for (uint32_t i = n_lines - 1; i > 0; --i) {
                std::uniform_int_distribution<uint32_t> pick(0, i - 1);       // j < i (not <= i): one cycle
                std::swap(order[i], order[pick(rng)]);
            }
            // order[] is a cyclic permutation viewed as a successor map: next[i] = order[i]
            CK(cudaMalloc(&d_buf, fp));
            CK(cudaMalloc(&d_next, (size_t)n_lines * 4));
            CK(cudaMalloc(&d_out, 2 * sizeof(unsigned long long)));
            CK(cudaMemset(d_buf, 0, fp));
            CK(cudaMemcpy(d_next, order.data(), (size_t)n_lines * 4, cudaMemcpyHostToDevice));
            scatter_next<<<(n_lines + 255) / 256, 256>>>(d_buf, d_next, n_lines);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());

            // Warm-up chase (untimed): touches every line when the footprint could fit in cache, so the
            // timed chase sees the steady-state cache contents; for big footprints it just warms the TLB.
            const uint32_t warm = fp <= 2 * l2 ? n_lines : std::min<uint32_t>(n_lines, TIMED_STEPS);
            chase<<<1, 1>>>(d_buf, warm, d_out);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());

            // Timed chase: clock64 inside the kernel, cudaEvents around the launch.
            CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
            CK(cudaEventRecord(e0));
            chase<<<1, 1>>>(d_buf, TIMED_STEPS, d_out);
            CK(cudaEventRecord(e1));
            CK(cudaGetLastError());
            CK(cudaEventSynchronize(e1));
            float ms = 0.f; CK(cudaEventElapsedTime(&ms, e0, e1));
            unsigned long long h_out[2];
            CK(cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost));

            const double cycles = (double)h_out[1] / TIMED_STEPS;
            const double ns = (double)ms * 1e6 / TIMED_STEPS;
            const double mhz = ns > 0 ? cycles / ns * 1e3 : 0.0;     // cycles per ns * 1000 = MHz
            printf("{\"probe\":\"pointer_chase\",\"footprint_bytes\":%zu,\"stride_bytes\":%d,"
                   "\"latency_cycles\":%.2f,\"latency_ns\":%.2f,\"gpu\":\"%s\",\"steps\":%u,\"clock_mhz_est\":%.0f}\n",
                   fp, STRIDE_BYTES, cycles, ns, prop.name, TIMED_STEPS, mhz);
            fflush(stdout);
            fprintf(stderr, "footprint %10.3f MB: %8.1f cycles, %8.1f ns\n", fp / 1048576.0, cycles, ns);
        } catch (const std::exception& e) {
            ++failures;
            fprintf(stderr, "ERROR footprint %zu B: %s\n", fp, e.what());
        }
        // Free whatever was allocated (cudaFree(nullptr) is a no-op).
        cudaFree(d_buf); cudaFree(d_next); cudaFree(d_out);
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
        // A "sticky" error (e.g. illegal address) poisons the CUDA context: stop instead of looping on it.
        if (cudaDeviceSynchronize() != cudaSuccess) {
            fprintf(stderr, "CUDA context is unusable; stopping after %d failures\n", failures);
            return 3;
        }
    }
    return failures ? 1 : 0;
}
