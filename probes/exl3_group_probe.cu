// Isolated K4/N256 launch shim. Not linked into or injected into serving.
#include "quant/exl3_moe_common.cuh"
#undef MOE_SMS_PER_EXPERT
#define MOE_SMS_PER_EXPERT GLM53_GROUP_SIZE
#undef MOE_TILESIZE_K
#define MOE_TILESIZE_K GLM53_TILE_K
#undef MOE_FRAG_STAGES
#define MOE_FRAG_STAGES GLM53_FRAG_STAGES
#undef MOE_SH_STAGES
#define MOE_SH_STAGES GLM53_SH_STAGES
#define exl3_moe_kernel glm53_probe_moe_kernel
#include "quant/exl3_moe_kernel.cuh"

// Mirror exl3_gemm_inner.cuh for bits=4, M16/N256. Hadamard writeback
// reuses the same dynamic buffer, needing only 128 floats per warp.
#ifndef GLM53_TIGHT_SMEM
#define GLM53_TIGHT_SMEM 0
#endif
#ifndef GLM53_RESIDENT_BLOCKS
#define GLM53_RESIDENT_BLOCKS 1
#endif
constexpr int probe_threads = 256 * GLM53_TILE_K / 16;
constexpr int probe_gemm_smem = GLM53_SH_STAGES *
    (2 * 16 * GLM53_TILE_K + 2 * (GLM53_TILE_K / 16) * 16 * 16 * 4)
    + 4 * (4 * 256 * 4);
constexpr int probe_had_smem = probe_threads / 32 * 128 * sizeof(float);
constexpr int probe_required_smem = probe_gemm_smem > probe_had_smem
    ? probe_gemm_smem : probe_had_smem;
static_assert(probe_required_smem <= SMEM_MAX);
constexpr int probe_smem = GLM53_TIGHT_SMEM ? probe_required_smem : SMEM_MAX;
static int probe_resident_limit = 0;
static int probe_grid_groups = 0;

extern "C" int probe_init() {
    cudaError_t rc = cudaFuncSetAttribute(glm53_probe_moe_kernel<4, 256>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, probe_smem);
    if (rc != cudaSuccess) return rc;
    rc = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&probe_resident_limit,
        glm53_probe_moe_kernel<4, 256>, probe_threads, probe_smem);
    if (rc != cudaSuccess) return rc;
    if (probe_resident_limit < GLM53_RESIDENT_BLOCKS)
        return cudaErrorCooperativeLaunchTooLarge;
    int device = 0, sms = 0, cooperative = 0;
    rc = cudaGetDevice(&device);
    if (rc != cudaSuccess) return rc;
    rc = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    if (rc != cudaSuccess) return rc;
    if (GLM53_RESIDENT_BLOCKS > 1) {
        rc = cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, device);
        if (rc != cudaSuccess) return rc;
        if (!cooperative) return cudaErrorNotSupported;
    }
    if (sms % GLM53_GROUP_SIZE) return cudaErrorInvalidConfiguration;
    probe_grid_groups = sms * GLM53_RESIDENT_BLOCKS / GLM53_GROUP_SIZE;
    return cudaSuccess;
}

extern "C" int probe_concurrency() { return probe_grid_groups; }
extern "C" int probe_active_blocks_per_sm() { return probe_resident_limit; }
extern "C" int probe_smem_bytes() { return probe_smem; }

extern "C" int probe_launch(void** args, int concurrency, void* stream) {
    if (concurrency <= 0 || concurrency > probe_grid_groups)
        return cudaErrorInvalidConfiguration;
    // The persistent expert-group barriers must not be oversubscribed.
    // Cooperative launch plus the occupancy gate guarantees residency for
    // the expanded grid; there is no unchecked 2x ordinary launch path.
    if (GLM53_RESIDENT_BLOCKS > 1)
        return cudaLaunchCooperativeKernel((const void*)glm53_probe_moe_kernel<4, 256>,
            dim3(GLM53_GROUP_SIZE, 1, concurrency), dim3(probe_threads, 1, 1),
            args, probe_smem, (cudaStream_t)stream);
    return cudaLaunchKernel((const void*)glm53_probe_moe_kernel<4, 256>,
        dim3(GLM53_GROUP_SIZE, 1, concurrency), dim3(probe_threads, 1, 1),
        args, probe_smem, (cudaStream_t)stream);
}

#if GLM53_PRIVATE_OUTPUT
__global__ void glm53_reduce_private(const float* scratch, float* output,
                                    int stride, int elements, int groups) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    float value = 0.0f;
    for (int group = 0; group < groups; ++group)
        value += scratch[group * stride + i];
    output[i] += value;
}

extern "C" int probe_launch_private(void** args, int concurrency, int rows, void* stream) {
    // Matches EXL3_MOE_KERNEL_ARGS. Host launch arguments, not device reads.
    int hidden = *static_cast<int*>(args[18]);
    int cap = *static_cast<int*>(args[22]);
    if (rows <= 0 || rows > cap / 2 || cap % 2 || hidden <= 0)
        return cudaErrorInvalidValue;
    float* scratch = *static_cast<float**>(args[2]);
    float* output = *static_cast<float**>(args[5]);
    int stride = cap / 2 * hidden;
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cudaError_t rc = cudaMemset2DAsync(scratch, stride * sizeof(float), 0,
        rows * hidden * sizeof(float), concurrency, cuda_stream);
    if (rc != cudaSuccess) return rc;
    rc = static_cast<cudaError_t>(probe_launch(args, concurrency, stream));
    if (rc != cudaSuccess) return rc;
    int elements = rows * hidden;
    glm53_reduce_private<<<(elements + 255) / 256, 256, 0, cuda_stream>>>(
        scratch, output, stride, elements, concurrency);
    return cudaGetLastError();
}
#endif
