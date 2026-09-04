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

extern "C" int probe_init() {
    return cudaFuncSetAttribute(glm53_probe_moe_kernel<4, 256>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
}

extern "C" int probe_launch(void** args, int concurrency, void* stream) {
    return cudaLaunchKernel((const void*)glm53_probe_moe_kernel<4, 256>,
        dim3(GLM53_GROUP_SIZE, 1, concurrency), dim3(256 * GLM53_TILE_K / 16, 1, 1),
        args, SMEM_MAX, (cudaStream_t)stream);
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
