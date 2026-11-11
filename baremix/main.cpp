/**
 * @file main.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_baremix_custom.h"
#else
#include "tikicpulib.h"

extern "C" void baremix_custom(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                               uint8_t *, uint8_t *);
#endif
extern void GenerateTiling(const char *socVersion, uint8_t *tilingBuf);

int32_t main(int32_t argc, char *argv[])
{
    int tensor_len = 64;
    if (const char *tl = std::getenv("BENCHMARK_TENSOR_LEN")) {
        int v = std::atoi(tl);
        if (v > 0) {
            tensor_len = v;
        }
    }
    const char *socVersion = SOC_VERSION;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    size_t aFileSize = 16 * sizeof(float);
    size_t bFileSize = 16 * sizeof(float);
    size_t cFileSize = tensor_len * sizeof(float);
    size_t biasFileSize = 4 * sizeof(float);
    size_t input1FileSize = tensor_len * sizeof(float);
    size_t input2FileSize = tensor_len * sizeof(float);
    size_t tilingFileSize = sizeof(TCubeTiling);
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
    uint8_t *tilingBuf = (uint8_t *)malloc(tilingFileSize);
    GenerateTiling(socVersion, tilingBuf);
#ifdef CUSTOM_ASCEND310P
    uint32_t blockDim = 2;
#else
    uint32_t blockDim = 1;
#endif

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *input1 = (uint8_t *)AscendC::GmAlloc(input1FileSize);
    uint8_t *input2 = (uint8_t *)AscendC::GmAlloc(input2FileSize);
    uint8_t *a = (uint8_t *)AscendC::GmAlloc(aFileSize);
    uint8_t *b = (uint8_t *)AscendC::GmAlloc(bFileSize);
    uint8_t *bias = (uint8_t *)AscendC::GmAlloc(biasFileSize);
    uint8_t *c = (uint8_t *)AscendC::GmAlloc(cFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingFileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);

    ReadFile("./input/input1_gm.bin", input1FileSize, input1, input1FileSize);
    ReadFile("./input/input2_gm.bin", input2FileSize, input2, input2FileSize);
    ReadFile("./input/x1_gm.bin", aFileSize, a, aFileSize);
    ReadFile("./input/x2_gm.bin", bFileSize, b, bFileSize);
    ReadFile("./input/bias.bin", biasFileSize, bias, biasFileSize);
    memcpy_s(tiling, tilingFileSize, tilingBuf, tilingFileSize);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(baremix_custom, blockDim, input1, input2, a, b, bias, c, workspace, tiling);

    WriteFile("./output/output.bin", c, cFileSize);
    AscendC::GmFree((void *)input1);
    AscendC::GmFree((void *)input2);
    AscendC::GmFree((void *)a);
    AscendC::GmFree((void *)b);
    AscendC::GmFree((void *)bias);
    AscendC::GmFree((void *)c);
    AscendC::GmFree((void *)tiling);
    AscendC::GmFree((void *)workspace);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    if (const char *dev = std::getenv("ASCEND_DEVICE_ID")) {
        deviceId = std::atoi(dev);
    }
    printf("[INFO] Using NPU device %d\n", deviceId);
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *inputAHost;
    uint8_t *inputADevice;
    CHECK_ACL(aclrtMallocHost((void **)(&inputAHost), aFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x1_gm.bin", aFileSize, inputAHost, aFileSize);
    CHECK_ACL(aclrtMemcpy(inputADevice, aFileSize, inputAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *inputBHost;
    uint8_t *inputBDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&inputBHost), bFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x2_gm.bin", bFileSize, inputBHost, bFileSize);
    CHECK_ACL(aclrtMemcpy(inputBDevice, bFileSize, inputBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *outputCHost;
    uint8_t *outputCDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&outputCHost), cFileSize));
    CHECK_ACL(aclrtMalloc((void **)&outputCDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *inputBiasHost;
    uint8_t *inputBiasDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&inputBiasHost), biasFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputBiasDevice, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/bias.bin", biasFileSize, inputBiasHost, biasFileSize);
    CHECK_ACL(aclrtMemcpy(inputBiasDevice, biasFileSize, inputBiasHost, biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *input1Host;
    uint8_t *input1Device;
    CHECK_ACL(aclrtMallocHost((void **)(&input1Host), input1FileSize));
    CHECK_ACL(aclrtMalloc((void **)&input1Device, input1FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/input1_gm.bin", input1FileSize, input1Host, input1FileSize);
    CHECK_ACL(aclrtMemcpy(input1Device, input1FileSize, input1Host, input1FileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *input2Host;
    uint8_t *input2Device;
    CHECK_ACL(aclrtMallocHost((void **)(&input2Host), input2FileSize));
    CHECK_ACL(aclrtMalloc((void **)&input2Device, input2FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/input2_gm.bin", input2FileSize, input2Host, input2FileSize);
    CHECK_ACL(aclrtMemcpy(input2Device, input2FileSize, input2Host, input2FileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *tilingHost;
    uint8_t *tilingDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&tilingHost), tilingFileSize));
    CHECK_ACL(aclrtMalloc((void **)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(tilingHost, tilingFileSize, tilingBuf, tilingFileSize, ACL_MEMCPY_HOST_TO_HOST));
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ACLRT_LAUNCH_KERNEL(baremix_custom)
    (blockDim, stream, input1Device, input2Device, inputADevice, inputBDevice, inputBiasDevice, outputCDevice, workspaceDevice, tilingDevice);

    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtFree(inputADevice));
    CHECK_ACL(aclrtFreeHost(inputAHost));
    CHECK_ACL(aclrtFree(inputBDevice));
    CHECK_ACL(aclrtFreeHost(inputBHost));
    CHECK_ACL(aclrtMemcpy(outputCHost, cFileSize, outputCDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output.bin", outputCHost, cFileSize);

    {
        float *outF = reinterpret_cast<float *>(outputCHost);
        int n = static_cast<int>(cFileSize / sizeof(float));
        float fmax = -1e38f, fmin = 1e38f;
        double fsum = 0.0;
        int nsentinel = 0;
        for (int i = 0; i < n; ++i) {
            float v = outF[i];
            if (v < -999.0f) { nsentinel++; continue; }
            if (v > fmax) fmax = v;
            if (v < fmin) fmin = v;
            fsum += v;
        }
        int nvalid = n - nsentinel;
        printf("\n========== OUTPUT DIAGNOSTICS ==========\n");
        printf("  total=%-6d valid=%-6d sentinel=%d\n", n, nvalid, nsentinel);
        if (nvalid > 0) {
            printf("  min=%.6e  max=%.6e  mean=%.6e\n", fmin, fmax, fsum / nvalid);
        } else {
            printf("  [WARN] ALL OUTPUT IS SENTINEL OR INVALID!\n");
        }
        printf("  first: ");
        int printed = 0;
        for (int i = 0; i < n && printed < 8; ++i) {
            if (outF[i] > -999.0f) { printf("%.4f ", outF[i]); printed++; }
        }
        printf("\n  last: %.6f\n", outF[n - 1]);
        printf("=========================================\n\n");
    }
    CHECK_ACL(aclrtFree(outputCDevice));
    CHECK_ACL(aclrtFreeHost(outputCHost));
    CHECK_ACL(aclrtFree(inputBiasDevice));
    CHECK_ACL(aclrtFreeHost(inputBiasHost));
    CHECK_ACL(aclrtFree(input1Device));
    CHECK_ACL(aclrtFreeHost(input1Host));
    CHECK_ACL(aclrtFree(input2Device));
    CHECK_ACL(aclrtFreeHost(input2Host));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFreeHost(tilingHost));
    CHECK_ACL(aclrtFree(workspaceDevice));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    free(tilingBuf);
    return 0;
}
