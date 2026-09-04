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
